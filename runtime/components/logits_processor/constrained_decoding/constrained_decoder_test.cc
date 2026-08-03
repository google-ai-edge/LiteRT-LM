// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/logits_processor/constrained_decoding/constrained_decoder.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constraint_provider.h"
#include "runtime/components/logits_processor/constrained_decoding/fst_constraint_config.h"
#include "runtime/components/logits_processor/constrained_decoding/fst_constraint_provider.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::lm {
namespace {

using ::sentencepiece::ModelProto;
using ::testing::status::StatusIs;

void AddToken(ModelProto& model, std::string token,
              const ModelProto::SentencePiece::Type type) {
  ModelProto::SentencePiece& piece = *model.add_pieces();
  piece.set_piece(std::move(token));
  piece.set_type(type);
}

ModelProto MakeSpm(std::vector<std::string> tokens) {
  ModelProto model;
  model.mutable_trainer_spec()->set_pad_id(0);
  model.mutable_trainer_spec()->set_eos_id(1);
  AddToken(model, "<p>", ModelProto::SentencePiece::CONTROL);
  AddToken(model, "<e>", ModelProto::SentencePiece::CONTROL);
  for (std::string& token : tokens) {
    AddToken(model, std::move(token),
             sentencepiece::ModelProto::SentencePiece::NORMAL);
  }
  AddToken(model, "<unk>", ModelProto::SentencePiece::UNKNOWN);
  return model;
}

template <typename T>
Expected<TensorBuffer> CreateTokenIdsTensorBuffer(const Environment& env,
                                                  T data[],
                                                  std::vector<int32_t> dims) {
  LiteRtElementType element_type = LiteRtElementType::kLiteRtElementTypeNone;
  if constexpr (std::is_same_v<T, int32_t>) {
    element_type = kLiteRtElementTypeInt32;
  } else if constexpr (std::is_same_v<T, float>) {
    element_type = kLiteRtElementTypeFloat32;
  }
  RankedTensorType tokens_id_tensor_type(
      {/*.element_type=*/element_type, BuildLayout(dims)});
  size_t buffer_size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>()) *
      sizeof(data[0]);
  auto tokens_id_tensor_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  tokens_id_tensor_type, buffer_size);
  if (!tokens_id_tensor_buffer.HasValue()) {
    return tokens_id_tensor_buffer;
  }
  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tokens_id_tensor_buffer.Value(), TensorBuffer::LockMode::kWrite);
    if (lock_and_addr.HasValue()) {
      std::memcpy(lock_and_addr->second, data, buffer_size);
    }
  }
  return tokens_id_tensor_buffer;
}

class ConstrainedDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ModelProto model = MakeSpm({"a", "b", "c"});
    ASSERT_OK(spm_processor_.Load(model));
    ASSERT_OK_AND_ASSIGN(provider_,
                         FstConstraintProvider::Create(
                             model, FstConstraintProviderOptions{
                                        .check_vocabulary_type = false}));
    vocab_size_ = spm_processor_.GetPieceSize();
  }

  std::unique_ptr<ConstraintProvider> provider_;
  sentencepiece::SentencePieceProcessor spm_processor_;
  int vocab_size_;
};

TEST_F(ConstrainedDecoderTest, UpdateStateAndProcessLogitsBatchSize1) {
  ASSERT_OK_AND_ASSIGN(
      auto constraint,
      provider_->CreateConstraint(FstConstraintArg{.constraint_string = "ab"}));
  ConstrainedDecoder constrained_decoder(constraint.get(), /*batch_size=*/1);

  // Create a tensor buffer for the token ids for "a".
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  int32_t token_ids[] = {spm_processor_.PieceToId("a")};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tokens_id_tensor_buffer,
      CreateTokenIdsTensorBuffer<int32_t>(env, token_ids, {1, 1}));

  // Update state with "a".
  ASSERT_OK(constrained_decoder.UpdateState(tokens_id_tensor_buffer));

  // Create a tensor buffer for the logits with all values set to 2.0f.
  std::vector<float> logits_data(vocab_size_, 2.0f);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits_tensor_buffer,
      CreateTokenIdsTensorBuffer<float>(env, logits_data.data(),
                                        {1, 1, vocab_size_}));

  ASSERT_OK(constrained_decoder.ProcessLogits(logits_tensor_buffer));

  // Verify that only the "b" token is allowed.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto masked_logits_span,
      ReferTensorBufferAsSpan<float>(logits_tensor_buffer));
  for (int i = 0; i < vocab_size_; ++i) {
    if (i == spm_processor_.PieceToId("b")) {
      EXPECT_EQ(masked_logits_span[i], 2.0f);
    } else {
      EXPECT_EQ(masked_logits_span[i], std::numeric_limits<float>::lowest());
    }
  }

  int32_t new_token_ids[] = {spm_processor_.PieceToId("b")};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto new_token_ids_tensor_buffer,
      CreateTokenIdsTensorBuffer<int32_t>(env, new_token_ids, {1, 1}));

  // Update state with "b".
  ASSERT_OK(constrained_decoder.UpdateState(new_token_ids_tensor_buffer));

  // Create a tensor buffer for the logits with all values set to 3.0f.
  std::vector<float> new_logits_data(vocab_size_, 3.0f);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto new_logits_tensor_buffer,
      CreateTokenIdsTensorBuffer<float>(env, new_logits_data.data(),
                                        {1, 1, vocab_size_}));

  // Update state with "b".
  ASSERT_OK(constrained_decoder.ProcessLogits(new_logits_tensor_buffer));

  // Verify that only the "<e>" token is allowed.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto new_masked_logits_span,
      ReferTensorBufferAsSpan<float>(new_logits_tensor_buffer));
  for (int i = 0; i < vocab_size_; ++i) {
    if (i == spm_processor_.PieceToId("<e>")) {
      EXPECT_EQ(new_masked_logits_span[i], 3.0f);
    } else {
      EXPECT_EQ(new_masked_logits_span[i],
                std::numeric_limits<float>::lowest());
    }
  }
}

TEST_F(ConstrainedDecoderTest, UpdateStateAndProcessLogitsBatchSize2) {
  ASSERT_OK_AND_ASSIGN(auto constraint,
                       provider_->CreateConstraint(
                           FstConstraintArg{.constraint_string = "a|c"}));
  ConstrainedDecoder constrained_decoder(constraint.get(), /*batch_size=*/2);

  // Create a tensor buffer for the token ids for "a" and "c".
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  int32_t token_ids[] = {spm_processor_.PieceToId("a"),
                         spm_processor_.PieceToId("c")};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tokens_id_tensor_buffer,
      CreateTokenIdsTensorBuffer<int32_t>(env, token_ids, {2, 1}));

  // Update state with "a" and "c".
  ASSERT_OK(constrained_decoder.UpdateState(tokens_id_tensor_buffer));

  std::vector<float> logits_data(vocab_size_ * 2, 1.0f);
  RankedTensorType logits_tensor_type(
      {/*.element_type=*/kLiteRtElementTypeFloat32,
       BuildLayout({2, 1, vocab_size_})});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits_tensor_buffer,
      CreateTokenIdsTensorBuffer<float>(env, logits_data.data(),
                                        {2, 1, vocab_size_}));

  ASSERT_OK(constrained_decoder.ProcessLogits(logits_tensor_buffer));

  // Verify that only "<e>" is allowed.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto masked_logits_span,
      ReferTensorBufferAsSpan<float>(logits_tensor_buffer));
  for (int i = 0; i < masked_logits_span.size(); ++i) {
    int token_id = i % vocab_size_;
    if (token_id == spm_processor_.PieceToId("<e>")) {
      EXPECT_EQ(masked_logits_span[i], 1.0f);
    } else {
      EXPECT_EQ(masked_logits_span[i], std::numeric_limits<float>::lowest());
    }
  }
}

TEST_F(ConstrainedDecoderTest, UpdateStateFailsWithWrongBatchSize) {
  ASSERT_OK_AND_ASSIGN(
      auto constraint,
      provider_->CreateConstraint(FstConstraintArg{.constraint_string = "ab"}));
  ConstrainedDecoder constrained_decoder(constraint.get(), /*batch_size=*/2);

  // Create a tensor buffer for the token ids for "a".
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  int32_t token_ids[] = {spm_processor_.PieceToId("a")};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tokens_id_tensor_buffer,
      CreateTokenIdsTensorBuffer<int32_t>(env, token_ids, {1, 1}));

  // UpdateState should fail because the batch size does not match the expected
  // batch size.
  EXPECT_THAT(constrained_decoder.UpdateState(tokens_id_tensor_buffer),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ConstrainedDecoderTest, ProcessLogitsFailsWithWrongBatchSize) {
  ASSERT_OK_AND_ASSIGN(
      auto constraint,
      provider_->CreateConstraint(FstConstraintArg{.constraint_string = "ab"}));
  ConstrainedDecoder constrained_decoder(constraint.get(), /*batch_size=*/1);

  // Create a tensor buffer for the token ids for "a".
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  int32_t token_ids[] = {spm_processor_.PieceToId("a")};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tokens_id_tensor_buffer,
      CreateTokenIdsTensorBuffer<int32_t>(env, token_ids, {1, 1}));

  // Update state with "a".
  ASSERT_OK(constrained_decoder.UpdateState(tokens_id_tensor_buffer));

  std::vector<float> logits_data(vocab_size_ * 2, 1.0f);
  RankedTensorType logits_tensor_type(
      {/*.element_type=*/kLiteRtElementTypeFloat32,
       BuildLayout({2, 1, vocab_size_})});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits_tensor_buffer,
      CreateTokenIdsTensorBuffer<float>(env, logits_data.data(),
                                        {2, 1, vocab_size_}));
  // ProcessLogits should fail because the batch size does not match the
  // expected batch size.
  EXPECT_THAT(constrained_decoder.ProcessLogits(logits_tensor_buffer),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ConstrainedDecoderTest, ProcessLogitsFailsWithWrongVolabSize) {
  ASSERT_OK_AND_ASSIGN(
      auto constraint,
      provider_->CreateConstraint(FstConstraintArg{.constraint_string = "ab"}));
  ConstrainedDecoder constrained_decoder(constraint.get(), /*batch_size=*/1);

  // Create a tensor buffer for the token ids for "a".
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  int32_t token_ids[] = {spm_processor_.PieceToId("a")};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tokens_id_tensor_buffer,
      CreateTokenIdsTensorBuffer<int32_t>(env, token_ids, {1, 1}));

  // Update state with "a".
  ASSERT_OK(constrained_decoder.UpdateState(tokens_id_tensor_buffer));

  std::vector<float> logits_data(vocab_size_ + 1, 1.0f);
  RankedTensorType logits_tensor_type(
      {/*.element_type=*/kLiteRtElementTypeFloat32,
       BuildLayout({2, 1, vocab_size_})});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits_tensor_buffer,
      CreateTokenIdsTensorBuffer<float>(env, logits_data.data(),
                                        {1, 1, vocab_size_ + 1}));

  // ProcessLogits should fail because the vocabulary size does not match the
  // expected vocabulary size.
  EXPECT_THAT(constrained_decoder.ProcessLogits(logits_tensor_buffer),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace litert::lm
