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

#include "omni/text2speech/qwen3_tts/qwen3_tts_acoustic_predictor_stage.h"

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
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/text2speech/qwen3_tts/common.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_model_config.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

Expected<TensorBuffer> CreateMaskBuffer(Environment& env,
                                        ElementType element_type,
                                        const Dimensions& dims,
                                        size_t buffer_size) {
  RankedTensorType tensor_type(element_type, Layout(dims));
  return TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                     std::move(tensor_type), buffer_size);
}

Expected<TensorBuffer> CreateFloatBuffer(Environment& env,
                                         const Dimensions& dims,
                                         size_t num_floats) {
  return CreateMaskBuffer(env, ElementType::Float32, dims,
                          num_floats * sizeof(float));
}

// Returns a one-element buffer list holding a `[1, 1, num_floats]` Float32
// buffer, as created for a single-input or single-output signature.
std::vector<TensorBuffer> MakeSingleBuffer(Environment& env,
                                           size_t num_floats) {
  std::vector<TensorBuffer> buffers;
  Expected<TensorBuffer> buffer = CreateFloatBuffer(
      env, Dimensions({1, 1, static_cast<int32_t>(num_floats)}), num_floats);
  EXPECT_TRUE(buffer.HasValue());
  if (buffer.HasValue()) buffers.push_back(std::move(*buffer));
  return buffers;
}

// Returns `num_heads * kMtpVocab` logits where head `h` peaks at token
// `peaks[h]`.
std::vector<float> MakeHeadLogits(absl::Span<const int> peaks) {
  std::vector<float> logits(peaks.size() * qwen3_tts::kMtpVocab, -1.0f);
  for (size_t h = 0; h < peaks.size(); ++h) {
    logits[h * qwen3_tts::kMtpVocab + peaks[h]] = 5.0f;
  }
  return logits;
}

class EmptyFrontendStage
    : public SingleThreadedStageWithDeque<Qwen3TtsFrontendOutput> {
 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TEST(Qwen3TtsAcousticPredictorStageTest, DerivesMtpCacheLenFromMaskDimensions) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  Expected<TensorBuffer> padded_mask =
      CreateMaskBuffer(*env, ElementType::Float32, Dimensions({1, 1, 1, 32}),
                       32 * sizeof(float));
  ASSERT_TRUE(padded_mask.HasValue());
  EXPECT_THAT(DeriveMtpCacheLen(*padded_mask), IsOkAndHolds(32));

  Expected<TensorBuffer> unpadded_mask =
      CreateMaskBuffer(*env, ElementType::Float32, Dimensions({1, 1, 1, 17}),
                       17 * sizeof(float));
  ASSERT_TRUE(unpadded_mask.HasValue());
  EXPECT_THAT(DeriveMtpCacheLen(*unpadded_mask), IsOkAndHolds(17));

  Expected<TensorBuffer> min_mask =
      CreateMaskBuffer(*env, ElementType::Float32, Dimensions({1, 1, 1, 16}),
                       16 * sizeof(float));
  ASSERT_TRUE(min_mask.HasValue());
  EXPECT_THAT(DeriveMtpCacheLen(*min_mask), IsOkAndHolds(16));

  Expected<TensorBuffer> rank1_mask = CreateMaskBuffer(
      *env, ElementType::Float32, Dimensions({64}), 64 * sizeof(float));
  ASSERT_TRUE(rank1_mask.HasValue());
  EXPECT_THAT(DeriveMtpCacheLen(*rank1_mask), IsOkAndHolds(64));
}

TEST(Qwen3TtsAcousticPredictorStageTest,
     DerivesMtpCacheLenWhenHardwareBufferIsPadded) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  // Hardware buffer is aligned to 128 bytes (32 floats) while the logical
  // tensor shape is [1, 1, 1, 17] (68 packed bytes).
  Expected<TensorBuffer> mask_buffer =
      CreateMaskBuffer(*env, ElementType::Float32, Dimensions({1, 1, 1, 17}),
                       /*buffer_size=*/32 * sizeof(float));
  ASSERT_TRUE(mask_buffer.HasValue());
  const Expected<size_t> buffer_size = mask_buffer->Size();
  const Expected<size_t> packed_size = mask_buffer->PackedSize();
  ASSERT_TRUE(buffer_size.HasValue());
  ASSERT_TRUE(packed_size.HasValue());
  EXPECT_EQ(*buffer_size, 32 * sizeof(float));
  EXPECT_EQ(*packed_size, 17 * sizeof(float));

  ASSERT_OK_AND_ASSIGN(const int cache_len, DeriveMtpCacheLen(*mask_buffer));
  EXPECT_EQ(cache_len, 17);

  std::vector<float> mask(cache_len, qwen3_tts::kNegInf);
  mask[0] = 0.0f;
  EXPECT_TRUE(mask_buffer->Write<float>(absl::MakeConstSpan(mask)).HasValue());
}

TEST(Qwen3TtsAcousticPredictorStageTest, DeriveMtpCacheLenRejectsNonFloat32) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  Expected<TensorBuffer> int_mask =
      CreateMaskBuffer(*env, ElementType::Int32, Dimensions({1, 1, 1, 32}),
                       32 * sizeof(int32_t));
  ASSERT_TRUE(int_mask.HasValue());
  EXPECT_THAT(DeriveMtpCacheLen(*int_mask),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Qwen3TtsAcousticPredictorStageTest,
     DeriveMtpCacheLenRejectsInvalidDimensions) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  Expected<TensorBuffer> scalar_mask = CreateMaskBuffer(
      *env, ElementType::Float32, Dimensions({}), sizeof(float));
  ASSERT_TRUE(scalar_mask.HasValue());
  EXPECT_THAT(DeriveMtpCacheLen(*scalar_mask),
              StatusIs(absl::StatusCode::kInvalidArgument));

  Expected<TensorBuffer> multi_head_mask =
      CreateMaskBuffer(*env, ElementType::Float32, Dimensions({1, 2, 1, 32}),
                       64 * sizeof(float));
  ASSERT_TRUE(multi_head_mask.HasValue());
  EXPECT_THAT(DeriveMtpCacheLen(*multi_head_mask),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Qwen3TtsAcousticPredictorStageTest,
     DeriveMtpCacheLenRejectsCacheShorterThanNumCodeGroups) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  Expected<TensorBuffer> short_mask =
      CreateMaskBuffer(*env, ElementType::Float32, Dimensions({1, 1, 1, 15}),
                       15 * sizeof(float));
  ASSERT_TRUE(short_mask.HasValue());
  EXPECT_THAT(DeriveMtpCacheLen(*short_mask),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Qwen3TtsAcousticPredictorStageTest, CreateRejectsMissingTalkerRunner) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  EmptyFrontendStage frontend;
  Qwen3TtsModelConfig config;

  EXPECT_THAT(
      Qwen3TtsAcousticPredictorStage::Create(&frontend, config, resources),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(Qwen3TtsAcousticPredictorStageTest, ResolvesMtpStepOutputLayout) {
  constexpr size_t kNumHeads = qwen3_tts::kNumCodeGroups - 1;
  EXPECT_THAT(ResolveMtpStepOutputLayout(qwen3_tts::kHiddenDim * sizeof(float)),
              IsOkAndHolds(MtpStepOutputLayout::kHiddenState));
  EXPECT_THAT(ResolveMtpStepOutputLayout(kNumHeads * qwen3_tts::kMtpVocab *
                                         sizeof(float)),
              IsOkAndHolds(MtpStepOutputLayout::kAllHeadsLogits));
  EXPECT_THAT(ResolveMtpStepOutputLayout(qwen3_tts::kMtpVocab * sizeof(float)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      ResolveMtpStepOutputLayout(qwen3_tts::kHiddenDim * sizeof(float) + 1),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ResolveMtpStepOutputLayout(0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Qwen3TtsAcousticPredictorStageTest, ValidatesMtpHeadBuffers) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  EXPECT_THAT(ValidateMtpHeadBuffers(
                  "head_0", MakeSingleBuffer(*env, qwen3_tts::kHiddenDim),
                  MakeSingleBuffer(*env, qwen3_tts::kMtpVocab)),
              IsOk());

  // Wrong number of inputs / outputs.
  EXPECT_THAT(ValidateMtpHeadBuffers(
                  "head_0", {}, MakeSingleBuffer(*env, qwen3_tts::kMtpVocab)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateMtpHeadBuffers(
                  "head_0", MakeSingleBuffer(*env, qwen3_tts::kHiddenDim), {}),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Input is not a `kHiddenDim` hidden state.
  EXPECT_THAT(ValidateMtpHeadBuffers(
                  "head_0", MakeSingleBuffer(*env, qwen3_tts::kHiddenDim / 2),
                  MakeSingleBuffer(*env, qwen3_tts::kMtpVocab)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Output is not `kMtpVocab` logits.
  EXPECT_THAT(ValidateMtpHeadBuffers(
                  "head_0", MakeSingleBuffer(*env, qwen3_tts::kHiddenDim),
                  MakeSingleBuffer(*env, qwen3_tts::kMtpVocab - 1)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Qwen3TtsAcousticPredictorStageTest, PicksTokensFromAllHeadsLogits) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  constexpr int kNumHeads = qwen3_tts::kNumCodeGroups - 1;
  std::vector<int> peaks(kNumHeads);
  for (int h = 0; h < kNumHeads; ++h) {
    peaks[h] = (h * 131 + 7) % qwen3_tts::kMtpVocab;
  }
  const std::vector<float> logits = MakeHeadLogits(peaks);
  Expected<TensorBuffer> buffer = CreateFloatBuffer(
      *env, Dimensions({1, 1, static_cast<int32_t>(logits.size())}),
      logits.size());
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<float>(absl::MakeConstSpan(logits)).HasValue());

  for (int h = 0; h < kNumHeads; ++h) {
    EXPECT_THAT(ArgmaxMtpHeadLogits(*buffer, h), IsOkAndHolds(peaks[h]))
        << "head " << h;
  }
  EXPECT_THAT(ArgmaxMtpHeadLogits(*buffer, kNumHeads),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ArgmaxMtpHeadLogits(*buffer, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Qwen3TtsAcousticPredictorStageTest, PicksTokenFromPerHeadLogits) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  constexpr int kPeak = qwen3_tts::kMtpVocab - 1;
  const std::vector<float> logits = MakeHeadLogits({kPeak});
  Expected<TensorBuffer> buffer = CreateFloatBuffer(
      *env, Dimensions({1, 1, qwen3_tts::kMtpVocab}), qwen3_tts::kMtpVocab);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<float>(absl::MakeConstSpan(logits)).HasValue());

  EXPECT_THAT(ArgmaxMtpHeadLogits(*buffer, /*head_idx=*/0),
              IsOkAndHolds(kPeak));
  EXPECT_THAT(ArgmaxMtpHeadLogits(*buffer, /*head_idx=*/1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::tts
