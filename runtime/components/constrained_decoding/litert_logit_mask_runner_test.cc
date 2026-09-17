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

#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/repetition_penalty_constraint.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

TensorBuffer CreateHostTensorBuffer(const Environment& env,
                                    RankedTensorType tensor_type,
                                    absl::Span<const float> data) {
  size_t size_in_bytes = data.size() * sizeof(float);
  auto tb = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                        tensor_type, size_in_bytes);
  EXPECT_TRUE(tb.HasValue());
  EXPECT_TRUE(tb->Write(data).HasValue());
  return std::move(*tb);
}

TensorBuffer CreateHostTensorBufferF16(const Environment& env,
                                       RankedTensorType tensor_type,
                                       absl::Span<const tflite::half> data) {
  size_t size_in_bytes = data.size() * sizeof(tflite::half);
  auto tb = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                        tensor_type, size_in_bytes);
  EXPECT_TRUE(tb.HasValue());
  EXPECT_TRUE(tb->Write(data).HasValue());
  return std::move(*tb);
}

// These tests are about the mask graphs themselves, so they opt out of the
// in-place host fast path and run the graphs on the CPU delegate against
// host-memory buffers.
absl::StatusOr<std::unique_ptr<LiteRtLogitMaskRunner>> CreateGraphRunner(
    Environment& env) {
  return LiteRtLogitMaskRunner::Create(env, HwAccelerators::kCpu,
                                       /*force_graph_on_host=*/true);
}

TEST(LiteRtLogitMaskRunnerTest, BitmapMaskFP32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 8;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));

  std::vector<float> initial_logits = {1.0f, 2.0f, 3.0f, 4.0f,
                                       5.0f, 6.0f, 7.0f, 8.0f};
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  // Allow only tokens 2 and 5.
  auto mask = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {2, 5});

  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  ASSERT_EQ(result.size(), vocab_size);

  // Allowed tokens should keep their original value.
  EXPECT_FLOAT_EQ(result[2], 3.0f);
  EXPECT_FLOAT_EQ(result[5], 6.0f);

  // Disallowed tokens should be heavily penalized (< -1e8).
  for (int i = 0; i < vocab_size; ++i) {
    if (i != 2 && i != 5) {
      EXPECT_LT(result[i], -1e8f);
    }
  }
}

TEST(LiteRtLogitMaskRunnerTest, SparseMaskFP32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));

  std::vector<float> initial_logits = {2.0f, 4.0f, 6.0f, 8.0f};
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  // Scale token 1 by 0.5 and bias token 3 by -2.0.
  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 1, .weight = 0.5f, .bias = 0.0f},
      {.token_id = 3, .weight = 1.0f, .bias = -2.0f},
  };
  auto mask = std::make_unique<SparseLogitMask>(entries);

  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  ASSERT_EQ(result.size(), vocab_size);

  EXPECT_FLOAT_EQ(result[0], 2.0f);
  EXPECT_FLOAT_EQ(result[1], 2.0f);  // 4.0 * 0.5
  EXPECT_FLOAT_EQ(result[2], 6.0f);
  EXPECT_FLOAT_EQ(result[3], 6.0f);  // 8.0 - 2.0
}

TEST(LiteRtLogitMaskRunnerTest, CompositeMaskFP32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));

  std::vector<float> initial_logits = {10.0f, 10.0f, 10.0f, 10.0f};
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  auto composite = std::make_unique<CompositeLogitMask>();
  // Disallow token 0 and 3.
  composite->AddMask(
      BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {1, 2}));
  // Penalize token 1 by weight 0.5.
  std::vector<SparseLogitMask::Entry> sparse_entries = {
      {.token_id = 1, .weight = 0.5f, .bias = 0.0f},
  };
  composite->AddMask(std::make_unique<SparseLogitMask>(sparse_entries));

  EXPECT_OK(runner->Apply(logits_tb, composite.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  EXPECT_LT(result[0], -1e8f);        // Disallowed
  EXPECT_FLOAT_EQ(result[1], 5.0f);   // 10.0 * 0.5
  EXPECT_FLOAT_EQ(result[2], 10.0f);  // Unmodified
  EXPECT_LT(result[3], -1e8f);        // Disallowed
}

TEST(LiteRtLogitMaskRunnerTest, SequenceMaskFP32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int seq_len = 2;
  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, seq_len, vocab_size}));

  std::vector<float> initial_logits = {
      1.0f, 2.0f, 3.0f, 4.0f,  // Step 0
      5.0f, 6.0f, 7.0f, 8.0f   // Step 1
  };
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  std::vector<std::unique_ptr<LogitMask>> masks;
  masks.push_back(BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {0, 1}));
  masks.push_back(BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {2, 3}));

  EXPECT_OK(runner->ApplySequence(logits_tb, masks));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  // Step 0: tokens 0, 1 allowed
  EXPECT_FLOAT_EQ(result[0], 1.0f);
  EXPECT_FLOAT_EQ(result[1], 2.0f);
  EXPECT_LT(result[2], -1e8f);
  EXPECT_LT(result[3], -1e8f);

  // Step 1: tokens 2, 3 allowed
  EXPECT_LT(result[4], -1e8f);
  EXPECT_LT(result[5], -1e8f);
  EXPECT_FLOAT_EQ(result[6], 7.0f);
  EXPECT_FLOAT_EQ(result[7], 8.0f);
}

TEST(LiteRtLogitMaskRunnerTest, BitmapMaskFP16) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float16,
                               Layout(Dimensions{1, 1, vocab_size}));

  // Include a negative logit (-20.0f) to verify it does not overflow to -inf
  // when masked, and a large positive logit (50000.0f) to verify it does not
  // cancel out with kDisallowedBiasF16 (-65504.0f).
  std::vector<tflite::half> initial_logits = {
      tflite::half(-20.0f), tflite::half(2.0f), tflite::half(50000.0f),
      tflite::half(4.0f)};
  TensorBuffer logits_tb =
      CreateHostTensorBufferF16(env, tensor_type, initial_logits);

  auto mask = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {1, 3});
  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<tflite::half>(logits_tb));
  EXPECT_TRUE(std::isfinite(static_cast<float>(result[0])));
  EXPECT_FLOAT_EQ(static_cast<float>(result[0]), -65504.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(result[1]), 2.0f);
  EXPECT_TRUE(std::isfinite(static_cast<float>(result[2])));
  EXPECT_FLOAT_EQ(static_cast<float>(result[2]), -65504.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(result[3]), 4.0f);
}

TEST(LiteRtLogitMaskRunnerTest, GraphSignDependentSparseMaskFP32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));

  std::vector<float> initial_logits = {4.0f, -4.0f, 4.0f, -4.0f};
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  // Tokens 0 and 1 are penalized sign-dependently, tokens 2 and 3 are not.
  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 0, .weight = 0.5f, .sign_dependent_weight = true},
      {.token_id = 1, .weight = 0.5f, .sign_dependent_weight = true},
      {.token_id = 2, .weight = 0.5f, .sign_dependent_weight = false},
      {.token_id = 3, .weight = 0.5f, .sign_dependent_weight = false},
  };
  auto mask = std::make_unique<SparseLogitMask>(entries);

  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  EXPECT_FLOAT_EQ(result[0], 2.0f);   // Positive: 4.0 * 0.5
  EXPECT_FLOAT_EQ(result[1], -8.0f);  // Negative: -4.0 / 0.5
  EXPECT_FLOAT_EQ(result[2], 2.0f);   // Sign agnostic: 4.0 * 0.5
  EXPECT_FLOAT_EQ(result[3], -2.0f);  // Sign agnostic: -4.0 * 0.5
}

TEST(LiteRtLogitMaskRunnerTest, GraphSignDependentSparseMaskMatchesHostFP32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 6;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));

  // Covers both signs, zero, and a token left untouched by the mask.
  std::vector<float> initial_logits = {3.0f, -3.0f, 0.0f, -0.5f, 7.0f, -7.0f};
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 0, .weight = 0.25f, .sign_dependent_weight = true},
      {.token_id = 1, .weight = 0.25f, .sign_dependent_weight = true},
      {.token_id = 2, .weight = 2.0f, .sign_dependent_weight = true},
      {.token_id = 3,
       .weight = 2.0f,
       .bias = 1.5f,
       .sign_dependent_weight = true},
      {.token_id = 4, .weight = 0.5f, .bias = -1.0f},
  };
  auto mask = std::make_unique<SparseLogitMask>(entries);

  // Reference: the same mask applied by the host implementation.
  std::vector<float> expected = initial_logits;
  EXPECT_OK(mask->Apply(absl::MakeSpan(expected)));

  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  ASSERT_EQ(result.size(), expected.size());
  for (int i = 0; i < vocab_size; ++i) {
    EXPECT_FLOAT_EQ(result[i], expected[i]) << "token " << i;
  }
}

TEST(LiteRtLogitMaskRunnerTest, GraphSignAgnosticSparseMaskFP32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));

  std::vector<float> initial_logits = {2.0f, 4.0f, 6.0f, 8.0f};
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 1, .weight = 0.5f, .bias = 0.0f},
      {.token_id = 3, .weight = 1.0f, .bias = -2.0f},
  };
  auto mask = std::make_unique<SparseLogitMask>(entries);

  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  EXPECT_FLOAT_EQ(result[0], 2.0f);  // Unmodified
  EXPECT_FLOAT_EQ(result[1], 2.0f);  // 4.0 * 0.5
  EXPECT_FLOAT_EQ(result[2], 6.0f);  // Unmodified
  EXPECT_FLOAT_EQ(result[3], 6.0f);  // 8.0 - 2.0
}

TEST(LiteRtLogitMaskRunnerTest, GraphSignDependentSparseMaskFP16) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float16,
                               Layout(Dimensions{1, 1, vocab_size}));

  std::vector<tflite::half> initial_logits = {
      tflite::half(4.0f), tflite::half(-4.0f), tflite::half(2.0f),
      tflite::half(-2.0f)};
  TensorBuffer logits_tb =
      CreateHostTensorBufferF16(env, tensor_type, initial_logits);

  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 0, .weight = 0.5f, .sign_dependent_weight = true},
      {.token_id = 1, .weight = 0.5f, .sign_dependent_weight = true},
  };
  auto mask = std::make_unique<SparseLogitMask>(entries);

  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<tflite::half>(logits_tb));
  EXPECT_FLOAT_EQ(static_cast<float>(result[0]), 2.0f);   // 4.0 * 0.5
  EXPECT_FLOAT_EQ(static_cast<float>(result[1]), -8.0f);  // -4.0 / 0.5
  EXPECT_FLOAT_EQ(static_cast<float>(result[2]), 2.0f);   // Unmodified
  EXPECT_FLOAT_EQ(static_cast<float>(result[3]), -2.0f);  // Unmodified
}

TEST(LiteRtLogitMaskRunnerTest, GraphSignDependentSequenceMaskFP32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int seq_len = 2;
  const int vocab_size = 2;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, seq_len, vocab_size}));

  std::vector<float> initial_logits = {
      4.0f, -4.0f,  // Step 0
      6.0f, -6.0f   // Step 1
  };
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  // Only step 1 is sign dependent, so the weights have to be staged per step.
  std::vector<std::unique_ptr<LogitMask>> masks;
  masks.push_back(std::make_unique<SparseLogitMask>(
      std::vector<SparseLogitMask::Entry>{{.token_id = 1, .weight = 0.5f}}));
  masks.push_back(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 0.5f, .sign_dependent_weight = true}}));

  EXPECT_OK(runner->ApplySequence(logits_tb, masks));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  EXPECT_FLOAT_EQ(result[0], 4.0f);    // Unmodified
  EXPECT_FLOAT_EQ(result[1], -2.0f);   // Sign agnostic: -4.0 * 0.5
  EXPECT_FLOAT_EQ(result[2], 6.0f);    // Unmodified
  EXPECT_FLOAT_EQ(result[3], -12.0f);  // Sign dependent: -6.0 / 0.5
}

// Differential tests against the host LogitMask::Apply reference. The runner
// stages masks into flat weight/bias tensors while the host walks the masks
// sequentially, so the two are easy to drift apart.

struct DifferentialResult {
  std::vector<float> accelerated;
  std::vector<float> host;
};

// Runs `mask` through the runner and through LogitMask::Apply on identical
// inputs and returns both results.
absl::StatusOr<DifferentialResult> RunDifferential(
    Environment& env, int vocab_size, absl::Span<const float> initial_logits,
    const LogitMask& mask) {
  LITERT_ASSIGN_OR_RETURN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);
  ABSL_RETURN_IF_ERROR(runner->Apply(logits_tb, &mask));

  LITERT_ASSIGN_OR_RETURN(auto accelerated,
                          CopyFromTensorBuffer<float>(logits_tb));

  std::vector<float> host(initial_logits.begin(), initial_logits.end());
  ABSL_RETURN_IF_ERROR(mask.Apply(absl::MakeSpan(host)));

  return DifferentialResult{.accelerated = std::move(accelerated),
                            .host = std::move(host)};
}

// The host bans tokens by writing -inf while the runner adds a large finite
// negative bias, so "banned" is compared as "overwhelmingly negative" rather
// than bit-for-bit.
void ExpectMatchesHost(const DifferentialResult& result) {
  ASSERT_EQ(result.accelerated.size(), result.host.size());
  for (size_t i = 0; i < result.host.size(); ++i) {
    SCOPED_TRACE(testing::Message() << "token " << i);
    if (result.host[i] < -1e8f) {
      EXPECT_LT(result.accelerated[i], -1e8f)
          << "token banned on the host but not by the runner";
    } else {
      EXPECT_NEAR(result.accelerated[i], result.host[i], 1e-3f);
    }
  }
}

// A bitmap nested in a composite must disallow the padding tokens past its
// vocabulary, exactly as a standalone bitmap does.
TEST(LiteRtLogitMaskRunnerTest, CompositeBitmapDisallowsPaddingBeyondVocab) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  // The logits tensor is wider than the bitmap's vocabulary, so tokens 4..7 are
  // padding and must be banned.
  const int vocab_size = 8;
  const int bitmap_vocab_size = 4;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(
      BitmapLogitMask::CreateFromAllowedTokens(bitmap_vocab_size, {1, 2}));

  const std::vector<float> initial_logits = {1.0f, 2.0f, 3.0f, 4.0f,
                                             5.0f, 6.0f, 7.0f, 8.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  EXPECT_FLOAT_EQ(result.accelerated[1], 2.0f);
  EXPECT_FLOAT_EQ(result.accelerated[2], 3.0f);
  EXPECT_LT(result.accelerated[0], -1e8f);
  EXPECT_LT(result.accelerated[3], -1e8f);
  for (int i = bitmap_vocab_size; i < vocab_size; ++i) {
    EXPECT_LT(result.accelerated[i], -1e8f)
        << "padding token " << i << " must be disallowed";
  }
  ExpectMatchesHost(result);
}

// Two sparse masks touching the same token must compose multiplicatively, i.e.
// (z * w1 + b1) * w2 + b2, rather than only accumulating the bias.
TEST(LiteRtLogitMaskRunnerTest, StackedSparseMasksComposeLikeHost) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 2.0f, .bias = 1.0f}}));
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 3.0f, .bias = 0.5f}}));

  const std::vector<float> initial_logits = {1.0f, 2.0f, 3.0f, 4.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  // (2.0 * 2.0 + 1.0) * 3.0 + 0.5 == 15.5
  EXPECT_NEAR(result.accelerated[1], 15.5f, 1e-3f);
  ExpectMatchesHost(result);
}

// A soft sparse penalty must not lift a token that a hard bitmap constraint
// already banned.
TEST(LiteRtLogitMaskRunnerTest, SparsePenaltyDoesNotResurrectBannedToken) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(
      BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {0, 2}));
  // Token 1 is banned by the bitmap; this large positive bias must not apply.
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 1.0f, .bias = 1e9f}}));

  const std::vector<float> initial_logits = {1.0f, 2.0f, 3.0f, 4.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  EXPECT_LT(result.accelerated[1], -1e8f)
      << "a hard-banned token must stay banned";
  ExpectMatchesHost(result);
}

// A staged bias that happens to equal the internal -1e9 sentinel must not be
// mistaken for a hard ban: the second mask here still has to be applied on top
// of the first, as the host does.
TEST(LiteRtLogitMaskRunnerTest, SparseBiasMatchingSentinelIsStillApplied) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 2, .weight = 1.0f, .bias = -1e9f}}));
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 2, .weight = 0.0f, .bias = 7.0f}}));

  const std::vector<float> initial_logits = {1.0f, 2.0f, 3.0f, 4.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  // Host: (3.0 * 1.0 + -1e9) * 0.0 + 7.0 == 7.0. The second mask zeroes out the
  // first one's bias entirely, so the token must come back to 7.0 rather than
  // staying at -1e9.
  EXPECT_NEAR(result.accelerated[2], 7.0f, 1e-3f);
  ExpectMatchesHost(result);
}

// The repetition/presence/frequency penalties are what ConstrainedDecoder hands
// the runner on every decode step of a plain `--repetition_penalty` run.
TEST(LiteRtLogitMaskRunnerTest, RepetitionPenaltyMaskMatchesHost) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  auto mask = std::make_unique<RepetitionPenaltyMask>(
      std::vector<RepetitionPenaltyMask::Entry>{
          {.token_id = 0, .repetition_penalty = 1.05f, .bias = 0.0f},
          {.token_id = 1, .repetition_penalty = 1.05f, .bias = -0.5f},
          // No multiplicative penalty, bias only.
          {.token_id = 3, .repetition_penalty = 1.0f, .bias = -1.0f}});

  const std::vector<float> initial_logits = {8.0f, -4.0f, 2.0f, -1.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result, RunDifferential(env, vocab_size, initial_logits, *mask));

  // Positive logits are divided by the penalty, negative ones multiplied.
  EXPECT_NEAR(result.accelerated[0], 8.0f / 1.05f, 1e-3f);
  EXPECT_NEAR(result.accelerated[1], -4.0f * 1.05f - 0.5f, 1e-3f);
  EXPECT_FLOAT_EQ(result.accelerated[2], 2.0f);  // Never generated.
  EXPECT_NEAR(result.accelerated[3], -2.0f, 1e-3f);
  ExpectMatchesHost(result);
}

// CompositeConstraint hands the runner one mask per constraint, so the penalty
// mask has to compose with the hard constraints sitting next to it.
TEST(LiteRtLogitMaskRunnerTest, RepetitionPenaltyMaskComposesWithBitmap) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(
      BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {1, 2}));
  composite->AddMask(std::make_unique<RepetitionPenaltyMask>(
      std::vector<RepetitionPenaltyMask::Entry>{
          {.token_id = 1, .repetition_penalty = 1.2f, .bias = -0.25f},
          // Already banned by the bitmap, so the penalty must not resurrect it.
          {.token_id = 3, .repetition_penalty = 1.2f, .bias = 1e9f}}));

  const std::vector<float> initial_logits = {5.0f, 6.0f, 7.0f, 8.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  EXPECT_LT(result.accelerated[0], -1e8f);
  EXPECT_NEAR(result.accelerated[1], 6.0f / 1.2f - 0.25f, 1e-3f);
  EXPECT_FLOAT_EQ(result.accelerated[2], 7.0f);
  EXPECT_LT(result.accelerated[3], -1e8f);
  ExpectMatchesHost(result);
}

// ApplyBatch is the path ConstrainedDecoder uses.
TEST(LiteRtLogitMaskRunnerTest, ApplyBatchAppliesPerSequenceMasks) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;
  const int batch_size = 3;

  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{batch_size, 1, vocab_size}));
  std::vector<float> initial_logits = {1.0f, 2.0f,  3.0f,  4.0f,  //
                                       5.0f, 6.0f,  7.0f,  8.0f,  //
                                       9.0f, 10.0f, 11.0f, 12.0f};
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  auto mask0 = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {0});
  auto mask2 = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {3});
  // The middle sequence is unconstrained.
  std::vector<const LogitMask*> masks = {mask0.get(), nullptr, mask2.get()};

  EXPECT_OK(runner->ApplyBatch(logits_tb, absl::MakeConstSpan(masks)));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  ASSERT_EQ(result.size(), batch_size * vocab_size);

  // Sequence 0: only token 0 survives.
  EXPECT_FLOAT_EQ(result[0], 1.0f);
  EXPECT_LT(result[1], -1e8f);
  EXPECT_LT(result[2], -1e8f);
  EXPECT_LT(result[3], -1e8f);

  // Sequence 1: null mask, untouched.
  for (int i = 0; i < vocab_size; ++i) {
    EXPECT_FLOAT_EQ(result[vocab_size + i], initial_logits[vocab_size + i]);
  }

  // Sequence 2: only token 3 survives.
  EXPECT_LT(result[2 * vocab_size + 0], -1e8f);
  EXPECT_FLOAT_EQ(result[2 * vocab_size + 3], 12.0f);
}

// The host staging buffers persist across calls, so a mask applied on one step
// must not bleed into the next.
TEST(LiteRtLogitMaskRunnerTest, ReusedRunnerDoesNotLeakStaleMaskState) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));

  // Step 1 bans everything except token 0.
  auto first = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {0});
  TensorBuffer first_tb = CreateHostTensorBuffer(
      env, tensor_type, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_OK(runner->Apply(first_tb, first.get()));

  // Step 2 bans everything except token 3; token 0 must now be banned and
  // token 3 must be pristine.
  auto second = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {3});
  TensorBuffer second_tb = CreateHostTensorBuffer(
      env, tensor_type, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_OK(runner->Apply(second_tb, second.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(second_tb));
  EXPECT_LT(result[0], -1e8f);
  EXPECT_FLOAT_EQ(result[3], 4.0f);
}

// The graphs bind the caller's buffer as both input and output (see
// `needs_copy_back_`), so in-place execution must match the host reference.
TEST(LiteRtLogitMaskRunnerTest, InPlaceAliasingMatchesHost) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 8;

  auto mask =
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 0, .weight = 2.0f, .bias = 0.0f},
          {.token_id = 3, .weight = 1.0f, .bias = -1.5f},
          {.token_id = 7, .weight = 0.5f, .bias = 2.0f}});

  const std::vector<float> initial_logits = {1.0f, -2.0f, 3.0f, -4.0f,
                                             5.0f, -6.0f, 7.0f, -8.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result, RunDifferential(env, vocab_size, initial_logits, *mask));
  ExpectMatchesHost(result);
}

// A mask the device graphs cannot express (such as a nested CompositeLogitMask)
// must automatically fall back to CPU masking rather than failing or silently
// ignoring constraints.
TEST(LiteRtLogitMaskRunnerTest, NestedCompositeMaskFallsBackToCpuMasking) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  auto inner = std::make_unique<CompositeLogitMask>();
  inner->AddMask(BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {0}));
  auto outer = std::make_unique<CompositeLogitMask>();
  outer->AddMask(std::move(inner));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  TensorBuffer logits_tb = CreateHostTensorBuffer(
      env, tensor_type, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

  EXPECT_OK(runner->Apply(logits_tb, outer.get()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  EXPECT_FLOAT_EQ(result[0], 1.0f);
  EXPECT_LT(result[1], -1e8f);
  EXPECT_LT(result[2], -1e8f);
  EXPECT_LT(result[3], -1e8f);
}

TEST(LiteRtLogitMaskRunnerTest, InvalidLogitsRankOrTypeIsRejected) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  auto mask = BitmapLogitMask::CreateFromAllowedTokens(4, {0});

  // 2D tensor instead of 3D [batch, seq_len, vocab_size].
  RankedTensorType rank2_type(ElementType::Float32, Layout(Dimensions{1, 4}));
  TensorBuffer rank2_tb = CreateHostTensorBuffer(
      env, rank2_type, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_FALSE(runner->Apply(rank2_tb, mask.get()).ok());

  // Sequence length > 1 passed to Apply (which expects seq_len == 1).
  RankedTensorType seq2_type(ElementType::Float32, Layout(Dimensions{1, 2, 2}));
  TensorBuffer seq2_tb = CreateHostTensorBuffer(
      env, seq2_type, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_FALSE(runner->Apply(seq2_tb, mask.get()).ok());

  // Unsupported element type (Int32).
  RankedTensorType int32_type(ElementType::Int32, Layout(Dimensions{1, 1, 4}));
  auto int32_tb = TensorBuffer::CreateManaged(
      env, TensorBufferType::kHostMemory, int32_type, 4 * sizeof(int32_t));
  ASSERT_TRUE(int32_tb.HasValue());
  EXPECT_FALSE(runner->Apply(*int32_tb, mask.get()).ok());
}

TEST(LiteRtLogitMaskRunnerTest, MismatchedMaskCountIsRejected) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{2, 1, vocab_size}));
  TensorBuffer logits_tb = CreateHostTensorBuffer(
      env, tensor_type,
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

  auto mask = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {0});
  std::vector<const LogitMask*> masks = {mask.get()};  // Only one for batch 2.
  EXPECT_FALSE(runner->ApplyBatch(logits_tb, absl::MakeConstSpan(masks)).ok());
}

// `GetHwAcceleratorForBackend(Backend::CPU)` hands the runner
// `HwAccelerators::kCpu` on every CPU inference run, which must not by itself
// push host-memory logits through the graphs: that would run a vocab-wide
// elementwise model per decode step instead of touching the handful of tokens a
// mask constrains. Both paths produce the same logits by design, so the
// compiled graph count is what tells them apart.
TEST(LiteRtLogitMaskRunnerTest, CpuAcceleratorMasksHostMemoryInPlace) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  const std::vector<float> initial_logits = {1.0f, 2.0f, 3.0f, 4.0f};
  auto mask = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {2});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto runner, LiteRtLogitMaskRunner::Create(env, HwAccelerators::kCpu));
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);
  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  EXPECT_EQ(runner->NumCompiledGraphsForTesting(), 0)
      << "host-memory logits must be masked in place on a CPU accelerator";

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  EXPECT_FLOAT_EQ(result[2], 3.0f);
  for (int i = 0; i < vocab_size; ++i) {
    if (i == 2) continue;
    EXPECT_LT(result[i], -1e8f) << "token " << i;
  }

  // Opting into the graphs, which is what the tests above do, still masks the
  // same host buffer identically.
  LITERT_ASSERT_OK_AND_ASSIGN(auto graph_runner, CreateGraphRunner(env));
  TensorBuffer graph_logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);
  EXPECT_OK(graph_runner->Apply(graph_logits_tb, mask.get()));

  EXPECT_EQ(graph_runner->NumCompiledGraphsForTesting(), 1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto graph_result,
                              CopyFromTensorBuffer<float>(graph_logits_tb));
  EXPECT_FLOAT_EQ(graph_result[2], 3.0f);
  for (int i = 0; i < vocab_size; ++i) {
    if (i == 2) continue;
    EXPECT_LT(graph_result[i], -1e8f) << "token " << i;
  }
}

TEST(LiteRtLogitMaskRunnerTest,
     HostMemoryFastPathWithKNoneSupportsNestedComposites) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto runner, LiteRtLogitMaskRunner::Create(env, HwAccelerators::kNone));

  const int vocab_size = 4;
  auto inner = std::make_unique<CompositeLogitMask>();
  inner->AddMask(BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {1, 2}));
  auto outer = std::make_unique<CompositeLogitMask>();
  outer->AddMask(std::move(inner));
  outer->AddMask(std::make_unique<SparseLogitMask>(
      std::vector<SparseLogitMask::Entry>{{.token_id = 1, .bias = -0.5f}}));

  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  TensorBuffer logits_tb = CreateHostTensorBuffer(
      env, tensor_type, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

  EXPECT_OK(runner->Apply(logits_tb, outer.get()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  EXPECT_LT(result[0], -1e8f);
  EXPECT_FLOAT_EQ(result[1], 1.5f);
  EXPECT_FLOAT_EQ(result[2], 3.0f);
  EXPECT_LT(result[3], -1e8f);
}

TEST(LiteRtLogitMaskRunnerTest, GraphRunnerHandlesVaryingShapesAndDtypes) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));

  // First call: [1, 1, 4] Float32
  RankedTensorType f32_type(ElementType::Float32, Layout(Dimensions{1, 1, 4}));
  TensorBuffer f32_tb = CreateHostTensorBuffer(
      env, f32_type, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto mask4 = BitmapLogitMask::CreateFromAllowedTokens(4, {0, 3});
  EXPECT_OK(runner->Apply(f32_tb, mask4.get()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto res4, CopyFromTensorBuffer<float>(f32_tb));
  EXPECT_FLOAT_EQ(res4[0], 1.0f);
  EXPECT_LT(res4[1], -1e8f);
  EXPECT_LT(res4[2], -1e8f);
  EXPECT_FLOAT_EQ(res4[3], 4.0f);

  // Second call on same runner: [1, 2, 3] Float16 sequence (triggers lazy
  // subgraph recompilation for the new shape and dtype).
  RankedTensorType f16_type(ElementType::Float16, Layout(Dimensions{1, 2, 3}));
  TensorBuffer f16_tb = CreateHostTensorBufferF16(
      env, f16_type,
      std::vector<tflite::half>{tflite::half(1.0f), tflite::half(2.0f),
                                tflite::half(3.0f), tflite::half(4.0f),
                                tflite::half(5.0f), tflite::half(6.0f)});
  std::vector<std::unique_ptr<LogitMask>> seq_masks;
  seq_masks.push_back(BitmapLogitMask::CreateFromAllowedTokens(3, {0}));
  seq_masks.push_back(BitmapLogitMask::CreateFromAllowedTokens(3, {2}));
  EXPECT_OK(runner->ApplySequence(f16_tb, seq_masks));
  LITERT_ASSERT_OK_AND_ASSIGN(auto res6,
                              CopyFromTensorBuffer<tflite::half>(f16_tb));
  EXPECT_FLOAT_EQ(static_cast<float>(res6[0]), 1.0f);
  EXPECT_LT(static_cast<float>(res6[1]), -1000.0f);
  EXPECT_LT(static_cast<float>(res6[2]), -1000.0f);
  EXPECT_LT(static_cast<float>(res6[3]), -1000.0f);
  EXPECT_LT(static_cast<float>(res6[4]), -1000.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(res6[5]), 6.0f);
}

// Only the staging entries the previous call touched are reset, so a token
// constrained by one call has to be back to the identity transform for the
// next. These tests reuse a single runner on purpose; a fresh one per call
// would never exercise this.
TEST(LiteRtLogitMaskRunnerTest, ConsecutiveSparseMasksRestorePreviousTokens) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  const std::vector<float> initial_logits = {2.0f, 4.0f, 6.0f, 8.0f};

  // First call constrains token 1.
  {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    std::vector<SparseLogitMask::Entry> entries = {
        {.token_id = 1, .weight = 0.5f, .bias = -1.0f}};
    auto mask = std::make_unique<SparseLogitMask>(entries);
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    EXPECT_FLOAT_EQ(result[1], 1.0f);  // 4.0 * 0.5 - 1.0
  }

  // Second call constrains a different token. Token 1 must be untouched.
  {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    std::vector<SparseLogitMask::Entry> entries = {
        {.token_id = 2, .weight = 0.25f, .bias = 0.0f}};
    auto mask = std::make_unique<SparseLogitMask>(entries);
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f);  // Restored.
    EXPECT_FLOAT_EQ(result[2], 1.5f);  // 6.0 * 0.25
    EXPECT_FLOAT_EQ(result[3], 8.0f);
  }

  // A null mask leaves the logits alone and still clears the previous call.
  {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    EXPECT_OK(runner->Apply(logits_tb, nullptr));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    EXPECT_THAT(result, ::testing::ElementsAreArray(initial_logits));
  }
}

// Past a fraction of the vocabulary the reset falls back to filling the whole
// slice instead of walking the list of touched tokens. Both sides of that
// threshold have to restore the slice identically.
TEST(LiteRtLogitMaskRunnerTest, WideMaskFollowedByNarrowMaskRestoresSlice) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 64;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  std::vector<float> initial_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) initial_logits[i] = i + 1.0f;

  // Touch far more than vocab_size / 8 tokens so the next reset takes the
  // full-fill path.
  {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    std::vector<SparseLogitMask::Entry> entries;
    for (int i = 0; i < vocab_size / 2; ++i) {
      entries.push_back({.token_id = i, .weight = 0.5f, .bias = -1.0f});
    }
    auto mask = std::make_unique<SparseLogitMask>(entries);
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    EXPECT_FLOAT_EQ(result[0], -0.5f);  // 1.0 * 0.5 - 1.0
  }

  // Now a single-token mask: every previously touched token must be back.
  {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    std::vector<SparseLogitMask::Entry> entries = {
        {.token_id = 63, .weight = 0.5f, .bias = 0.0f}};
    auto mask = std::make_unique<SparseLogitMask>(entries);
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    for (int i = 0; i < vocab_size - 1; ++i) {
      EXPECT_FLOAT_EQ(result[i], initial_logits[i]) << "token " << i;
    }
    EXPECT_FLOAT_EQ(result[63], 32.0f);  // 64.0 * 0.5
  }
}

// A hard bitmap constraint writes the disallowed sentinel into the staging
// bias (and, in fp16, zeroes the weight). A following soft mask has to see a
// clean slice.
TEST(LiteRtLogitMaskRunnerTest, BitmapMaskFollowedBySparseMaskRestoresSlice) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 8;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  const std::vector<float> initial_logits = {1.0f, 2.0f, 3.0f, 4.0f,
                                             5.0f, 6.0f, 7.0f, 8.0f};

  {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    auto mask = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {2, 5});
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    EXPECT_LT(result[0], -1e8f);
  }

  {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    std::vector<SparseLogitMask::Entry> entries = {
        {.token_id = 0, .weight = 2.0f, .bias = 0.0f}};
    auto mask = std::make_unique<SparseLogitMask>(entries);
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    // Previously banned tokens are allowed again, and token 0 only carries the
    // new weight rather than a leftover -inf bias.
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    for (int i = 1; i < vocab_size; ++i) {
      EXPECT_FLOAT_EQ(result[i], initial_logits[i]) << "token " << i;
    }
  }
}

// Biases and weights are reset against separate lists of touched tokens, so a
// bitmap banning most of the vocabulary and a sparse mask staging one weight
// land on opposite sides of the full-reset threshold within the same call.
TEST(LiteRtLogitMaskRunnerTest, WideBiasAndNarrowWeightAreBothRestored) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 64;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  std::vector<float> initial_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) initial_logits[i] = i + 1.0f;

  // Bans 62 of the 64 tokens (bias only, far past vocab_size / 8) and scales a
  // single surviving token (weight only).
  {
    auto composite = std::make_unique<CompositeLogitMask>();
    composite->AddMask(
        BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {0, 1}));
    composite->AddMask(std::make_unique<SparseLogitMask>(
        std::vector<SparseLogitMask::Entry>{{.token_id = 0, .weight = 2.0f}}));

    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    EXPECT_OK(runner->Apply(logits_tb, composite.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    EXPECT_FLOAT_EQ(result[0], 2.0f);  // 1.0 * 2.0
    EXPECT_FLOAT_EQ(result[1], 2.0f);  // Allowed, unweighted.
    EXPECT_LT(result[2], -1e8f);
  }

  // An unconstrained step: every ban and the staged weight have to be gone.
  {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    std::vector<SparseLogitMask::Entry> entries = {
        {.token_id = 63, .weight = 1.0f, .bias = -1.0f}};
    auto mask = std::make_unique<SparseLogitMask>(entries);
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<float>(logits_tb));
    for (int i = 0; i < vocab_size - 1; ++i) {
      EXPECT_FLOAT_EQ(result[i], initial_logits[i]) << "token " << i;
    }
    EXPECT_FLOAT_EQ(result[63], 63.0f);  // 64.0 - 1.0
  }
}

// In fp16 a ban zeroes the weight as well as writing the bias, so both buffers
// have to be restored on the next step.
TEST(LiteRtLogitMaskRunnerTest, WideBitmapRestoresWeightsFP16) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 64;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float16,
                               Layout(Dimensions{1, 1, vocab_size}));
  std::vector<tflite::half> initial_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    initial_logits[i] = tflite::half(i + 1.0f);
  }

  {
    TensorBuffer logits_tb =
        CreateHostTensorBufferF16(env, tensor_type, initial_logits);
    auto mask = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {0});
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<tflite::half>(logits_tb));
    EXPECT_FLOAT_EQ(static_cast<float>(result[0]), 1.0f);
    EXPECT_LT(static_cast<float>(result[1]), -1000.0f);
  }

  {
    TensorBuffer logits_tb =
        CreateHostTensorBufferF16(env, tensor_type, initial_logits);
    std::vector<SparseLogitMask::Entry> entries = {
        {.token_id = 63, .weight = 0.5f}};
    auto mask = std::make_unique<SparseLogitMask>(entries);
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<tflite::half>(logits_tb));
    // A leftover zero weight would flatten these to 0.0.
    for (int i = 0; i < vocab_size - 1; ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(result[i]),
                      static_cast<float>(initial_logits[i]))
          << "token " << i;
    }
    EXPECT_FLOAT_EQ(static_cast<float>(result[63]), 32.0f);  // 64.0 * 0.5
  }
}

// A bitmap word with no surviving token is banned in one sweep rather than bit
// by bit, which is the shape a grammar constraint mostly consists of. The
// sparse mask alongside it must still see those tokens as already banned.
TEST(LiteRtLogitMaskRunnerTest, FullyDisallowedWordsMatchHost) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  // Two words wide, with the only allowed token in the second one: the first
  // word is entirely disallowed and the second one only partly.
  const int vocab_size = 128;
  const int allowed_token = 100;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(
      BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {allowed_token}));
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          // Inside the fully disallowed word: the ban wins.
          {.token_id = 5, .weight = 2.0f, .bias = 1.0f},
          {.token_id = allowed_token, .weight = 0.5f}}));

  std::vector<float> initial_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) initial_logits[i] = i + 1.0f;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  EXPECT_LT(result.accelerated[5], -1e8f);
  EXPECT_FLOAT_EQ(result.accelerated[allowed_token], 50.5f);  // 101.0 * 0.5
  ExpectMatchesHost(result);
}

// The whole-word ban zeroes fp16 weights the same way the per-token one does,
// so the next step still has to find the identity transform staged.
TEST(LiteRtLogitMaskRunnerTest, FullyDisallowedWordRestoresWeightsFP16) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 128;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float16,
                               Layout(Dimensions{1, 1, vocab_size}));
  std::vector<tflite::half> initial_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    initial_logits[i] = tflite::half(i + 1.0f);
  }

  {
    TensorBuffer logits_tb =
        CreateHostTensorBufferF16(env, tensor_type, initial_logits);
    auto mask = BitmapLogitMask::CreateFromAllowedTokens(vocab_size, {100});
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<tflite::half>(logits_tb));
    EXPECT_FLOAT_EQ(static_cast<float>(result[100]), 101.0f);
    // Token 5 sits in the word that was banned in one sweep.
    EXPECT_FLOAT_EQ(static_cast<float>(result[5]), -65504.0f);
  }

  {
    TensorBuffer logits_tb =
        CreateHostTensorBufferF16(env, tensor_type, initial_logits);
    std::vector<SparseLogitMask::Entry> entries = {
        {.token_id = 127, .weight = 0.5f}};
    auto mask = std::make_unique<SparseLogitMask>(entries);
    EXPECT_OK(runner->Apply(logits_tb, mask.get()));
    LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                                CopyFromTensorBuffer<tflite::half>(logits_tb));
    // A leftover zero weight would flatten these to 0.0.
    for (int i = 0; i < vocab_size - 1; ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(result[i]),
                      static_cast<float>(initial_logits[i]))
          << "token " << i;
    }
    EXPECT_FLOAT_EQ(static_cast<float>(result[127]), 64.0f);  // 128.0 * 0.5
  }
}

// Nothing validates a mask's vocabulary size, and a negative one would make the
// padding loop start before the beginning of the slice. Every token is padding
// in that case, so every token is disallowed.
TEST(LiteRtLogitMaskRunnerTest, BitmapWithNegativeVocabDisallowsEverything) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 8;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  const std::vector<float> initial_logits = {1.0f, 2.0f, 3.0f, 4.0f,
                                             5.0f, 6.0f, 7.0f, 8.0f};
  TensorBuffer logits_tb =
      CreateHostTensorBuffer(env, tensor_type, initial_logits);

  auto mask = BitmapLogitMask::CreateAllAllowed(/*vocab_size=*/-1);
  EXPECT_OK(runner->Apply(logits_tb, mask.get()));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              CopyFromTensorBuffer<float>(logits_tb));
  for (int i = 0; i < vocab_size; ++i) {
    EXPECT_LT(result[i], -1e8f) << "token " << i;
  }
}

// A sign-dependent entry next to a sign-agnostic one is the case the reciprocal
// graphs cannot express, so the weight_neg staging buffer has to be there.
TEST(LiteRtLogitMaskRunnerTest, SignDependentMixedWithSignAgnosticMatchesHost) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 0, .weight = 2.0f, .sign_dependent_weight = true},
          {.token_id = 3, .weight = 2.0f, .sign_dependent_weight = true}}));
  // Sign agnostic, so weight_neg is no longer 1 / weight everywhere.
  composite->AddMask(std::make_unique<SparseLogitMask>(
      std::vector<SparseLogitMask::Entry>{{.token_id = 1, .weight = 0.5f}}));

  const std::vector<float> initial_logits = {4.0f, -4.0f, 3.0f, -3.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  EXPECT_NEAR(result.accelerated[0], 8.0f, 1e-3f);   // 4.0 * 2.0
  EXPECT_NEAR(result.accelerated[1], -2.0f, 1e-3f);  // -4.0 * 0.5
  EXPECT_FLOAT_EQ(result.accelerated[2], 3.0f);
  EXPECT_NEAR(result.accelerated[3], -1.5f, 1e-3f);  // -3.0 / 2.0
  ExpectMatchesHost(result);
}

// The reciprocal and the explicit weight_neg variants share the same staging
// buffers, so a runner must be able to switch between them across steps.
TEST(LiteRtLogitMaskRunnerTest, AlternatingSignDependentVariantsMatchHost) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));

  const int vocab_size = 4;
  LITERT_ASSERT_OK_AND_ASSIGN(auto runner, CreateGraphRunner(env));
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions{1, 1, vocab_size}));
  const std::vector<float> initial_logits = {4.0f, -4.0f, 3.0f, -3.0f};

  // Purely reciprocal: weight_neg is derived on device and never staged.
  auto reciprocal_mask =
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 2.0f, .sign_dependent_weight = true}});
  // Mixed: the graph now needs weight_neg as an explicit input.
  auto mixed_mask =
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 2.0f, .sign_dependent_weight = true},
          {.token_id = 3, .weight = 0.5f}});

  const std::vector<const LogitMask*> steps = {
      reciprocal_mask.get(), mixed_mask.get(), reciprocal_mask.get()};
  for (const LogitMask* mask : steps) {
    TensorBuffer logits_tb =
        CreateHostTensorBuffer(env, tensor_type, initial_logits);
    EXPECT_OK(runner->Apply(logits_tb, mask));
    LITERT_ASSERT_OK_AND_ASSIGN(auto accelerated,
                                CopyFromTensorBuffer<float>(logits_tb));
    std::vector<float> host(initial_logits.begin(), initial_logits.end());
    EXPECT_OK(mask->Apply(absl::MakeSpan(host)));
    ExpectMatchesHost({.accelerated = accelerated, .host = host});
  }
}

// A sign-dependent weight stacked on top of a bias staged by an earlier mask
// cannot be expressed by a single bias tensor: the host scales that bias by the
// branch multiplier it takes (here weight_neg = 2.0), while the graph can only
// scale it by `weight`. The runner has to notice and fall back to the host.
TEST(LiteRtLogitMaskRunnerTest, SignDependentWeightOnStagedBiasMatchesHost) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 1.0f, .bias = -2.0f}}));
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 0.5f, .sign_dependent_weight = true}}));

  const std::vector<float> initial_logits = {1.0f, -4.0f, 3.0f, 4.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  // (-4.0 - 2.0) / 0.5 == -12.0, not (-4.0 * 2.0) + (-2.0 * 0.5) == -9.0.
  EXPECT_NEAR(result.accelerated[1], -12.0f, 1e-3f);
  ExpectMatchesHost(result);
}

// A staged weight that flips the sign of the logit makes the host and the graph
// branch on opposite signs, since the graph only ever sees the original logit.
TEST(LiteRtLogitMaskRunnerTest, SignDependentWeightOnFlippedSignMatchesHost) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  const int vocab_size = 4;

  auto composite = std::make_unique<CompositeLogitMask>();
  composite->AddMask(std::make_unique<SparseLogitMask>(
      std::vector<SparseLogitMask::Entry>{{.token_id = 1, .weight = -1.0f}}));
  composite->AddMask(
      std::make_unique<SparseLogitMask>(std::vector<SparseLogitMask::Entry>{
          {.token_id = 1, .weight = 0.5f, .sign_dependent_weight = true}}));

  const std::vector<float> initial_logits = {1.0f, -4.0f, 3.0f, 4.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunDifferential(env, vocab_size, initial_logits, *composite));

  // -4.0 * -1.0 == 4.0 is positive, so the host multiplies: 4.0 * 0.5 == 2.0.
  EXPECT_NEAR(result.accelerated[1], 2.0f, 1e-3f);
  ExpectMatchesHost(result);
}

}  // namespace
}  // namespace litert::lm
