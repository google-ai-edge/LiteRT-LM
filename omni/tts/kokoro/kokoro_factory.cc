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

#include "omni/tts/kokoro/kokoro_factory.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/tts/kokoro/kokoro_acoustic_stage.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/kokoro/kokoro_vocoder_stage.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_session.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::tts {

absl::Status InitKokoroResources(const KokoroModelConfig& config,
                                 absl::string_view model_folder,
                                 absl::string_view cache_dir,
                                 lm::Backend backend, int num_threads,
                                 ::litert::Environment& env,
                                 ModelResources& resources) {
  ModelOptions model_options;
  model_options.model_dir = model_folder;
  model_options.cache_dir = cache_dir;
  model_options.backend = backend;
  model_options.num_threads = num_threads;

  LITERT_ASSIGN_OR_RETURN(
      auto acoustic,
      CreateCompiledModel(env, model_options, config.acoustic_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "kokoro_acoustic", std::make_shared<CompiledModel>(std::move(acoustic))));

  LITERT_ASSIGN_OR_RETURN(
      auto vocoder,
      CreateCompiledModel(env, model_options, config.vocoder_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "kokoro_vocoder", std::make_shared<CompiledModel>(std::move(vocoder))));

  return absl::OkStatus();
}

absl::StatusOr<TtsSession::Components> CreateKokoroComponents(
    const KokoroModelConfig& config, absl::string_view model_folder,
    const TextChunkConfig& text_chunk_config,
    std::shared_ptr<ModelResources> resources) {
  TextChunkConfig local_chunk_config = text_chunk_config;
  if (config.target_bucket > 0 && local_chunk_config.max_buffer_size == 0) {
    local_chunk_config.max_buffer_size =
        std::min(120, static_cast<int>(config.target_bucket * 0.9));
  }

  TtsSession::Components components;
  components.text_source =
      std::make_unique<StreamTextSource>(local_chunk_config);

  // Stage 1: Text frontend, phonemization, and unified acoustic prediction.
  LITERT_ASSIGN_OR_RETURN(auto acoustic, KokoroAcousticStage::Create(
                                             components.text_source.get(),
                                             config, model_folder, resources));
  // Stage 2: Neural vocoder and iSTFT audio synthesis.
  LITERT_ASSIGN_OR_RETURN(
      auto vocoder, KokoroVocoderStage::Create(acoustic.get(), resources));

  components.intermediate_stages.push_back(std::move(acoustic));
  components.vocoder = std::move(vocoder);

  return components;
}

}  // namespace litert::omni::tts
