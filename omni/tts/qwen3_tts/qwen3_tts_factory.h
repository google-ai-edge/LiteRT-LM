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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_session.h"
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
                                   const std::string& model_folder,
                                   const std::string& cache_dir,
                                   lm::Backend backend, int num_threads,
                                   ::litert::Environment& env,
                                   ModelResources& resources);

// Instantiates all stage components for a Qwen3-TTS inference.
//
// args
// - config: Qwen3-TTS model configuration.
// - model_folder: Path to the directory containing the Qwen3-TTS models.
// - text_chunk_config: Configuration for text chunk processing.
// - resources: Shared ModelResources container with compiled models.
//
// returns
// - TtsSession::Components containing all stage components on success, or
// error status on failure.
absl::StatusOr<TtsSession::Components> CreateQwen3TtsComponents(
    const Qwen3TtsModelConfig& config, const std::string& model_folder,
    const TextChunkConfig& text_chunk_config,
    std::shared_ptr<ModelResources> resources);

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_FACTORY_H_
