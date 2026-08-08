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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/tts/qwen3_tts/qwen3_acoustic_predictor_stage.h"
#include "omni/tts/qwen3_tts/qwen3_frontend_stage.h"
#include "omni/tts/qwen3_tts/qwen3_latent_decoder_stage.h"
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"
#include "omni/tts/qwen3_tts/qwen3_vocoder_stage.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/tts_session.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<TtsEngine>> TtsEngine::Create(
    const TtsEngineSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
  auto shared_env = std::make_shared<Environment>(std::move(env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  if (settings.model_type == ModelType::QWEN3_TTS) {
    Qwen3StageOptions options;
    options.model_dir = settings.model_folder;
    options.cache_dir = settings.cache_dir;
    options.num_threads = settings.num_threads;
    options.max_frames = settings.max_frames;

    ModelOptions model_options;
    model_options.model_dir = settings.model_folder;
    model_options.cache_dir = settings.cache_dir;
    model_options.backend = settings.backend;
    model_options.num_threads = settings.num_threads;

    // Compile and cache heavy models into ModelResources
    LITERT_ASSIGN_OR_RETURN(auto text_emb,
                            CreateCompiledModel(*shared_env, model_options,
                                                options.text_embedding_file));
    ABSL_RETURN_IF_ERROR(resources->AddCompiledModel(
        "text_embedding",
        std::make_shared<CompiledModel>(std::move(text_emb))));

    LITERT_ASSIGN_OR_RETURN(auto text_proj,
                            CreateCompiledModel(*shared_env, model_options,
                                                options.text_projection_file));
    ABSL_RETURN_IF_ERROR(resources->AddCompiledModel(
        "text_projection",
        std::make_shared<CompiledModel>(std::move(text_proj))));

    LITERT_ASSIGN_OR_RETURN(
        auto talker,
        CreateCompiledModel(*shared_env, model_options, options.talker_file));
    ABSL_RETURN_IF_ERROR(resources->AddCompiledModel(
        "talker", std::make_shared<CompiledModel>(std::move(talker))));

    LITERT_ASSIGN_OR_RETURN(
        auto mtp,
        CreateCompiledModel(*shared_env, model_options, options.mtp_file));
    ABSL_RETURN_IF_ERROR(resources->AddCompiledModel(
        "mtp", std::make_shared<CompiledModel>(std::move(mtp))));

    LITERT_ASSIGN_OR_RETURN(auto codec_emb,
                            CreateCompiledModel(*shared_env, model_options,
                                                options.codec_embedding_file));
    ABSL_RETURN_IF_ERROR(resources->AddCompiledModel(
        "codec_embedding",
        std::make_shared<CompiledModel>(std::move(codec_emb))));

    LITERT_ASSIGN_OR_RETURN(auto mtp_emb,
                            CreateCompiledModel(*shared_env, model_options,
                                                options.mtp_embedding_file));
    ABSL_RETURN_IF_ERROR(resources->AddCompiledModel(
        "mtp_embedding", std::make_shared<CompiledModel>(std::move(mtp_emb))));

    LITERT_ASSIGN_OR_RETURN(
        auto codec,
        CreateCompiledModel(*shared_env, model_options, options.codec_file));
    ABSL_RETURN_IF_ERROR(resources->AddCompiledModel(
        "codec", std::make_shared<CompiledModel>(std::move(codec))));

  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_type in TtsEngineSettings: ",
                     static_cast<int>(settings.model_type)));
  }

  auto thread_pool =
      std::make_unique<lm::ThreadPool>("tts_engine_pool", settings.num_threads);

  return std::unique_ptr<TtsEngine>(
      new TtsEngine(settings, resources, std::move(thread_pool)));
}

absl::StatusOr<std::unique_ptr<TtsSession>> TtsEngine::CreateSession() {
  TtsSession::Components components;
  components.text_source =
      std::make_unique<StreamTextSource>(settings_.text_chunk_config);

  if (settings_.model_type == ModelType::QWEN3_TTS) {
    Qwen3StageOptions options;
    options.model_dir = settings_.model_folder;
    options.cache_dir = settings_.cache_dir;
    options.num_threads = settings_.num_threads;
    options.max_frames = settings_.max_frames;

    ABSL_ASSIGN_OR_RETURN(
        auto frontend, Qwen3FrontendStage::Create(components.text_source.get(),
                                                  options, model_resources_));
    ABSL_ASSIGN_OR_RETURN(auto acoustic,
                          Qwen3AcousticPredictorStage::Create(
                              frontend.get(), options, model_resources_));
    ABSL_ASSIGN_OR_RETURN(auto latent,
                          Qwen3LatentDecoderStage::Create(acoustic.get()));
    ABSL_ASSIGN_OR_RETURN(
        auto vocoder,
        Qwen3VocoderStage::Create(latent.get(), options, model_resources_));

    components.text_frontend = std::move(frontend);
    components.acoustic_predictor = std::move(acoustic);
    components.latent_decoder = std::move(latent);
    components.vocoder = std::move(vocoder);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_type in TtsEngineSettings: ",
                     static_cast<int>(settings_.model_type)));
  }

  return TtsSession::Create(std::move(components), thread_pool_.get());
}

}  // namespace litert::omni::tts
