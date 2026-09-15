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
#include "absl/strings/str_format.h"  // from @com_google_absl
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

namespace {

ModelOptions MakeModelOptions(absl::string_view model_dir,
                              absl::string_view cache_dir, lm::Backend backend,
                              int num_threads) {
  ModelOptions options;
  options.model_dir = model_dir;
  options.cache_dir = cache_dir;
  options.backend = backend;
  options.num_threads = num_threads;
  return options;
}

}  // namespace

absl::Status InitKokoroResources(const KokoroModelConfig& config,
                                 absl::string_view model_folder,
                                 absl::string_view cache_dir,
                                 lm::Backend backend, int num_threads,
                                 ::litert::Environment& env,
                                 ModelResources& resources) {
  // The acoustic model is slower on GPU than on CPU in practice, so a GPU
  // engine backend still runs it on CPU unless explicitly configured; only the
  // vocoder follows the engine backend.
  const lm::Backend acoustic_backend = config.acoustic_backend.value_or(
      backend == lm::Backend::GPU ? lm::Backend::CPU : backend);
  const lm::Backend vocoder_backend = config.vocoder_backend.value_or(backend);

  ModelOptions acoustic_options = MakeModelOptions(
      model_folder, cache_dir, acoustic_backend, num_threads);
  LITERT_ASSIGN_OR_RETURN(
      auto acoustic,
      CreateCompiledModel(env, acoustic_options, config.acoustic_file));

  ModelOptions vocoder_options =
      MakeModelOptions(model_folder, cache_dir, vocoder_backend, num_threads);
  LITERT_ASSIGN_OR_RETURN(
      auto vocoder,
      CreateCompiledModel(env, vocoder_options, config.vocoder_file));

  // Verify that acoustic and vocoder models agree on frame capacity.
  auto acoustic_type = acoustic.GetOutputTensorType("acoustic_features");
  auto vocoder_type = vocoder.GetInputTensorType("acoustic_features");
  if (acoustic_type && vocoder_type) {
    auto acoustic_dims = acoustic_type->Layout().Dimensions();
    auto vocoder_dims = vocoder_type->Layout().Dimensions();
    if (!acoustic_dims.empty() && !vocoder_dims.empty() &&
        acoustic_dims.back() != vocoder_dims.back()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Kokoro acoustic model frame capacity (%d) does not match vocoder "
          "frame capacity (%d). Both models must be exported with the same "
          "frame capacity.",
          acoustic_dims.back(), vocoder_dims.back()));
    }
  }

  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "kokoro_acoustic", std::make_shared<CompiledModel>(std::move(acoustic))));
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

  if (acoustic->frame_capacity() != vocoder->frame_capacity()) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Kokoro acoustic stage frame capacity (%d) does not match vocoder "
        "stage frame capacity (%d). Both models must be exported with the "
        "same frame capacity.",
        acoustic->frame_capacity(), vocoder->frame_capacity()));
  }

  components.intermediate_stages.push_back(std::move(acoustic));
  components.vocoder = std::move(vocoder);

  return components;
}

}  // namespace litert::omni::tts
