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

#include "runtime/core/engine_legacy_streaming_utils.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr int kDefaultMaxNumAudioTokens = 2048;

absl::StatusOr<EngineSettings> CreateEngineSettings(
    std::optional<Backend> vision_backend,
    std::optional<Backend> audio_backend) {
  ABSL_ASSIGN_OR_RETURN(ModelAssets model_assets,
                        ModelAssets::Create("test_model_path"));
  return EngineSettings::CreateDefault(model_assets, Backend::GPU,
                                       vision_backend, audio_backend);
}

TEST(ShouldCacheStreamedModelTest, CachesVisionModelsOnlyWhenVisionIsEnabled) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/Backend::GPU,
                                            /*audio_backend=*/std::nullopt));

  EXPECT_TRUE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_VISION_ENCODER, settings));
  EXPECT_TRUE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_VISION_ADAPTER, settings));
  EXPECT_TRUE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_END_OF_VISION, settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW, settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_AUDIO_ADAPTER, settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_END_OF_AUDIO, settings));
}

TEST(ShouldCacheStreamedModelTest, CachesAudioModelsOnlyWhenAudioIsEnabled) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/std::nullopt,
                                            /*audio_backend=*/Backend::GPU));

  EXPECT_TRUE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW, settings));
  EXPECT_TRUE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_AUDIO_ADAPTER, settings));
  EXPECT_TRUE(ShouldCacheStreamedModel(proto::LlmMetadata::TF_LITE_END_OF_AUDIO,
                                       settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_VISION_ENCODER, settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_VISION_ADAPTER, settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_END_OF_VISION, settings));
}

TEST(ShouldCacheStreamedModelTest, SkipsAllModelsWhenNoModalityIsEnabled) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/std::nullopt,
                                            /*audio_backend=*/std::nullopt));

  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_VISION_ENCODER, settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW, settings));
}

TEST(ShouldCacheStreamedModelTest, DoesNotCacheNonMultimodalModels) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/Backend::GPU,
                                            /*audio_backend=*/Backend::GPU));

  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_PREFILL_DECODE, settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_ARTISAN_TEXT_DECODER, settings));
  EXPECT_FALSE(ShouldCacheStreamedModel(
      proto::LlmMetadata::TF_LITE_AUDIO_FRONTEND, settings));
}

TEST(GetNumExternalAudioTokensToReserveTest,
     ReservesDefaultForLiteRtAudioEncoder) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/std::nullopt,
                                            /*audio_backend=*/Backend::GPU));
  settings.GetMutableMainExecutorSettings().SetMaxNumTokens(4096);

  EXPECT_EQ(GetNumExternalAudioTokensToReserve(
                /*has_tflite_audio_encoder=*/true,
                /*artisan_has_audio_encoder=*/false, settings,
                kDefaultMaxNumAudioTokens),
            kDefaultMaxNumAudioTokens);
}

TEST(GetNumExternalAudioTokensToReserveTest,
     ReservesDefaultWhenMaxNumTokensIsUnset) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/std::nullopt,
                                            /*audio_backend=*/Backend::GPU));
  ASSERT_EQ(settings.GetMainExecutorSettings().GetMaxNumTokens(), 0);

  EXPECT_EQ(GetNumExternalAudioTokensToReserve(
                /*has_tflite_audio_encoder=*/true,
                /*artisan_has_audio_encoder=*/false, settings,
                kDefaultMaxNumAudioTokens),
            kDefaultMaxNumAudioTokens);
}

TEST(GetNumExternalAudioTokensToReserveTest, CapsAtMaxNumTokens) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/std::nullopt,
                                            /*audio_backend=*/Backend::GPU));
  settings.GetMutableMainExecutorSettings().SetMaxNumTokens(1024);

  EXPECT_EQ(GetNumExternalAudioTokensToReserve(
                /*has_tflite_audio_encoder=*/true,
                /*artisan_has_audio_encoder=*/false, settings,
                kDefaultMaxNumAudioTokens),
            1024);
}

TEST(GetNumExternalAudioTokensToReserveTest,
     ReservesNothingWithoutLiteRtAudioEncoder) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/std::nullopt,
                                            /*audio_backend=*/Backend::GPU));
  settings.GetMutableMainExecutorSettings().SetMaxNumTokens(4096);

  EXPECT_EQ(GetNumExternalAudioTokensToReserve(
                /*has_tflite_audio_encoder=*/false,
                /*artisan_has_audio_encoder=*/false, settings,
                kDefaultMaxNumAudioTokens),
            0);
}

TEST(GetNumExternalAudioTokensToReserveTest,
     ReservesNothingWhenArtisanModelHasAudioEncoder) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/std::nullopt,
                                            /*audio_backend=*/Backend::GPU));
  settings.GetMutableMainExecutorSettings().SetMaxNumTokens(4096);

  EXPECT_EQ(GetNumExternalAudioTokensToReserve(
                /*has_tflite_audio_encoder=*/true,
                /*artisan_has_audio_encoder=*/true, settings,
                kDefaultMaxNumAudioTokens),
            0);
}

TEST(GetNumExternalAudioTokensToReserveTest,
     ReservesNothingWhenAudioIsDisabled) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateEngineSettings(/*vision_backend=*/std::nullopt,
                                            /*audio_backend=*/std::nullopt));
  settings.GetMutableMainExecutorSettings().SetMaxNumTokens(4096);

  EXPECT_EQ(GetNumExternalAudioTokensToReserve(
                /*has_tflite_audio_encoder=*/true,
                /*artisan_has_audio_encoder=*/false, settings,
                kDefaultMaxNumAudioTokens),
            0);
}

}  // namespace
}  // namespace litert::lm
