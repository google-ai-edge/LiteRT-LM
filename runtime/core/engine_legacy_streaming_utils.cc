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

#include <algorithm>

#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {

bool ShouldCacheStreamedModel(proto::LlmMetadata::TfLiteModelType model_type,
                              const EngineSettings& engine_settings) {
  switch (model_type) {
    case proto::LlmMetadata::TF_LITE_VISION_ENCODER:
    case proto::LlmMetadata::TF_LITE_VISION_ADAPTER:
    case proto::LlmMetadata::TF_LITE_END_OF_VISION:
      return engine_settings.GetVisionExecutorSettings().has_value();
    case proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW:
    case proto::LlmMetadata::TF_LITE_AUDIO_ADAPTER:
    case proto::LlmMetadata::TF_LITE_END_OF_AUDIO:
      return engine_settings.GetAudioExecutorSettings().has_value();
    default:
      return false;
  }
}

int GetNumExternalAudioTokensToReserve(bool has_tflite_audio_encoder,
                                       bool artisan_has_audio_encoder,
                                       const EngineSettings& engine_settings,
                                       int default_max_num_audio_tokens) {
  if (!has_tflite_audio_encoder || artisan_has_audio_encoder ||
      !engine_settings.GetAudioExecutorSettings().has_value()) {
    return 0;
  }
  const int max_num_tokens =
      engine_settings.GetMainExecutorSettings().GetMaxNumTokens();
  return max_num_tokens > 0
             ? std::min<int>(default_max_num_audio_tokens, max_num_tokens)
             : default_max_num_audio_tokens;
}

}  // namespace litert::lm
