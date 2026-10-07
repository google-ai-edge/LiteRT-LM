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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_FACTORY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_source.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::tts {

// Compiles and populates all Qwen3-TTS LiteRT models into shared
// ModelResources.
//
// args
// - config: Qwen3-TTS model configuration.
// - model_folder: Path to the directory containing the Qwen3-TTS models.
// - cache_dir: Path to the directory for caching model data.
// - backend: Backend to use for model execution.
// - num_threads: Number of threads to use for model execution.
// - env: LiteRT environment.
// - resources: ModelResources to add compiled models to.
//
// returns
// - absl::OkStatus on success, or error status on failure.
absl::Status InitQwen3TtsResources(const Qwen3TtsModelConfig& config,
                                   absl::string_view model_folder,
                                   absl::string_view cache_dir,
                                   lm::Backend backend, int num_threads,
                                   Environment& env, ModelResources& resources);

// Instantiates all stage components for a Qwen3-TTS inference.
//
// args
// - config: Qwen3-TTS model configuration.
// - model_folder: Path to the directory containing the Qwen3-TTS models.
// - text_source: StreamTextSource providing text chunks for the session.
// - resources: Shared ModelResources container with compiled models.
// - stages: Output vector populated with the created stages in pipeline order.
//   The first stage (`stages[0]`) must be `StreamTextSource`.
// - output_stage: Output pointer set to the final vocoder `Stage<Output>`.
//
// returns
// - absl::OkStatus on success, or error status on failure.
absl::Status CreateQwen3TtsComponents(
    const Qwen3TtsModelConfig& config, absl::string_view model_folder,
    std::unique_ptr<StreamTextSource> absl_nonnull text_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages,
    Stage<Output>* absl_nullable* absl_nonnull output_stage);

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_FACTORY_H_
