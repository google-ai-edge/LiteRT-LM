// Copyright 2026 The ODML Authors.
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

#include "omni/asr/lm_decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/speech_recognizer.h"
#include "omni/base/mock_litert_lm_runner.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::Return;

::litert::TensorBuffer CreateTestTensorBuffer(
    const std::vector<int32_t>& dimensions,
    ::litert::ElementType element_type = ::litert::ElementType::Float32) {
  size_t num_elements = 1;
  for (int32_t d : dimensions) {
    num_elements *= d;
  }
  size_t element_size = element_type == ::litert::ElementType::Float32
                            ? sizeof(float)
                            : sizeof(int32_t);
  ::litert::RankedTensorType tensor_type(
      element_type, ::litert::Layout(::litert::Dimensions(dimensions.begin(),
                                                          dimensions.end())));
  auto buffer = ::litert::TensorBuffer::CreateManagedHostMemory(
      tensor_type, num_elements * element_size);
  return std::move(*buffer);
}

::litert::TensorBuffer CreateLogitsBuffer(int argmax_token,
                                         int vocab_size = 10) {
  auto buf =
      CreateTestTensorBuffer({1, vocab_size}, ::litert::ElementType::Float32);
  std::vector<float> logits(vocab_size, 0.0f);
  if (argmax_token >= 0 && argmax_token < vocab_size) {
    logits[argmax_token] = 10.0f;
  }
  auto status = buf.Write<float>(absl::MakeConstSpan(logits));
  EXPECT_TRUE(status);
  return buf;
}


TEST(LmDecoderTest, CreateWithInvalidMaxDecodeStepsReturnsError) {
  MockLiteRtLmRunner runner;
  EXPECT_THAT(LmDecoder::Create(&runner, /*decode_start_token_id=*/-1,
                                /*decode_stop_token_id=*/-1,
                                /*decode_skip_until_token_id=*/-1,
                                /*max_decode_steps=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LmDecoderTest, DecodeEmptyEncoderOutputsReturnsError) {
  MockLiteRtLmRunner runner;
  ASSERT_OK_AND_ASSIGN(auto decoder, LmDecoder::Create(&runner));

  std::vector<::litert::TensorBuffer> empty_inputs;
  EXPECT_THAT(decoder->Decode(empty_inputs),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LmDecoderTest, DecodePrefillFailsReturnsError) {
  MockLiteRtLmRunner runner;
  EXPECT_CALL(runner, Reset()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(runner, Prefill(_))
      .WillOnce(Return(absl::InternalError("Prefill error")));

  ASSERT_OK_AND_ASSIGN(auto decoder, LmDecoder::Create(&runner));
  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer({1, 10, 16}));

  EXPECT_THAT(decoder->Decode(encoder_outputs),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(LmDecoderTest, DecodeSuccessfulSequence) {
  MockLiteRtLmRunner runner;
  EXPECT_CALL(runner, Reset()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(runner, Prefill(_)).WillOnce(Return(absl::OkStatus()));

  // Step 1 yields token 3, Step 2 yields token 5, Step 3 yields stop token 9.
  EXPECT_CALL(runner, Decode(_))
      .WillOnce(Return(CreateLogitsBuffer(3)))
      .WillOnce(Return(CreateLogitsBuffer(5)))
      .WillOnce(Return(CreateLogitsBuffer(9)));

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      LmDecoder::Create(&runner, /*decode_start_token_id=*/1,
                        /*decode_stop_token_id=*/9));

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer({1, 10, 16}));

  auto decoded = decoder->Decode(encoder_outputs);
  ASSERT_OK(decoded);
  EXPECT_THAT(
      *decoded,
      ElementsAre(
          Field(&SpeechRecognizer::DecodedToken::token_id, 3),
          Field(&SpeechRecognizer::DecodedToken::token_id, 5)));
}

TEST(LmDecoderTest, DecodeHitsMaxSteps) {
  MockLiteRtLmRunner runner;
  EXPECT_CALL(runner, Reset()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(runner, Prefill(_)).WillOnce(Return(absl::OkStatus()));

  // Continually yields token 3, max_decode_steps = 3.
  EXPECT_CALL(runner, Decode(_))
      .WillOnce(Return(CreateLogitsBuffer(3)))
      .WillOnce(Return(CreateLogitsBuffer(3)))
      .WillOnce(Return(CreateLogitsBuffer(3)));

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      LmDecoder::Create(&runner, /*decode_start_token_id=*/0,
                        /*decode_stop_token_id=*/9,
                        /*decode_skip_until_token_id=*/-1,
                        /*max_decode_steps=*/3));

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer({1, 10, 16}));

  auto decoded = decoder->Decode(encoder_outputs);
  ASSERT_OK(decoded);
  EXPECT_THAT(
      *decoded,
      ElementsAre(
          Field(&SpeechRecognizer::DecodedToken::token_id, 3),
          Field(&SpeechRecognizer::DecodedToken::token_id, 3),
          Field(&SpeechRecognizer::DecodedToken::token_id, 3)));
}

TEST(LmDecoderTest, DecodeWithDecodeSkipUntilTokenId) {
  MockLiteRtLmRunner runner;
  EXPECT_CALL(runner, Reset()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(runner, Prefill(_)).WillOnce(Return(absl::OkStatus()));

  // Step 1: token 1 (skipped).
  // Step 2: token 2 (skip until token, skipped).
  // Step 3: token 3 (included).
  // Step 4: token 4 (included).
  // Step 5: stop token 9.
  EXPECT_CALL(runner, Decode(_))
      .WillOnce(Return(CreateLogitsBuffer(1)))
      .WillOnce(Return(CreateLogitsBuffer(2)))
      .WillOnce(Return(CreateLogitsBuffer(3)))
      .WillOnce(Return(CreateLogitsBuffer(4)))
      .WillOnce(Return(CreateLogitsBuffer(9)));

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      LmDecoder::Create(&runner, /*decode_start_token_id=*/0,
                        /*decode_stop_token_id=*/9,
                        /*decode_skip_until_token_id=*/2));

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer({1, 10, 16}));

  auto decoded = decoder->Decode(encoder_outputs);
  ASSERT_OK(decoded);
  EXPECT_THAT(
      *decoded,
      ElementsAre(
          Field(&SpeechRecognizer::DecodedToken::token_id, 3),
          Field(&SpeechRecognizer::DecodedToken::token_id, 4)));
}

}  // namespace
}  // namespace litert::omni::asr
