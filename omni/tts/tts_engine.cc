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

#include "omni/tts/tts_engine.h"

#include <memory>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/tts/kokoro/kokoro_factory.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/qwen3_tts/qwen3_tts_factory.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/tts_session.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<TtsEngine>> TtsEngine::Create(
    const TtsEngineSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
  auto shared_env = std::make_shared<Environment>(std::move(env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  if (auto* config = std::get_if<KokoroModelConfig>(&settings.model_config)) {
    ABSL_RETURN_IF_ERROR(InitKokoroResources(
        *config, settings.model_folder, settings.cache_dir, settings.backend,
        settings.num_threads, *shared_env, *resources));
  } else if (auto* config =
                 std::get_if<Qwen3TtsModelConfig>(&settings.model_config)) {
    ABSL_RETURN_IF_ERROR(InitQwen3TtsResources(
        *config, settings.model_folder, settings.cache_dir, settings.backend,
        settings.num_threads, *shared_env, *resources));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_config in TtsEngineSettings: ",
                     static_cast<int>(settings.GetModelType())));
  }

  auto thread_pool =
      std::make_unique<lm::ThreadPool>("tts_engine_pool", settings.num_threads);

  return std::unique_ptr<TtsEngine>(
      new TtsEngine(settings, resources, std::move(thread_pool)));
}

absl::StatusOr<std::unique_ptr<TtsSession>> TtsEngine::CreateSession(
    const TtsSessionConfig& session_config) {
  TtsSession::Components components;
  if (auto* config = std::get_if<KokoroModelConfig>(&settings_.model_config)) {
    ABSL_ASSIGN_OR_RETURN(
        components, CreateKokoroComponents(*config, settings_.model_folder,
                                           session_config.text_chunk_config,
                                           model_resources_));
  } else if (auto* config =
                 std::get_if<Qwen3TtsModelConfig>(&settings_.model_config)) {
    ABSL_ASSIGN_OR_RETURN(
        components, CreateQwen3TtsComponents(*config, settings_.model_folder,
                                             session_config.text_chunk_config,
                                             model_resources_));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_config in TtsEngineSettings: ",
                     static_cast<int>(settings_.GetModelType())));
  }

  return TtsSession::Create(std::move(components), thread_pool_.get());
}

}  // namespace litert::omni::tts
