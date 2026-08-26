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

#include "omni/tts/qwen3_tts/qwen3_tts_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/tts/qwen3_tts/qwen3_tts_acoustic_predictor_stage.h"
#include "omni/tts/qwen3_tts/qwen3_tts_frontend_stage.h"
#include "omni/tts/qwen3_tts/qwen3_tts_latent_decoder_stage.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/qwen3_tts/qwen3_tts_vocoder_stage.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_session.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::tts {

absl::Status InitQwen3TtsResources(const Qwen3TtsModelConfig& config,
                                   const std::string& model_folder,
                                   const std::string& cache_dir,
                                   lm::Backend backend, int num_threads,
                                   ::litert::Environment& env,
                                   ModelResources& resources) {
  ModelOptions model_options;
  model_options.model_dir = model_folder;
  model_options.cache_dir = cache_dir;
  model_options.backend = backend;
  model_options.num_threads = num_threads;

  LITERT_ASSIGN_OR_RETURN(
      auto text_emb,
      CreateCompiledModel(env, model_options, config.text_embedding_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "text_embedding", std::make_shared<CompiledModel>(std::move(text_emb))));

  LITERT_ASSIGN_OR_RETURN(
      auto text_proj,
      CreateCompiledModel(env, model_options, config.text_projection_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "text_projection",
      std::make_shared<CompiledModel>(std::move(text_proj))));

  LITERT_ASSIGN_OR_RETURN(
      auto talker, CreateCompiledModel(env, model_options, config.talker_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "talker", std::make_shared<CompiledModel>(std::move(talker))));

  LITERT_ASSIGN_OR_RETURN(
      auto mtp, CreateCompiledModel(env, model_options, config.mtp_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "mtp", std::make_shared<CompiledModel>(std::move(mtp))));

  LITERT_ASSIGN_OR_RETURN(
      auto codec_emb,
      CreateCompiledModel(env, model_options, config.codec_embedding_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "codec_embedding",
      std::make_shared<CompiledModel>(std::move(codec_emb))));

  LITERT_ASSIGN_OR_RETURN(
      auto mtp_emb,
      CreateCompiledModel(env, model_options, config.mtp_embedding_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "mtp_embedding", std::make_shared<CompiledModel>(std::move(mtp_emb))));

  LITERT_ASSIGN_OR_RETURN(
      auto codec, CreateCompiledModel(env, model_options, config.codec_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "codec", std::make_shared<CompiledModel>(std::move(codec))));

  return absl::OkStatus();
}

absl::StatusOr<TtsSession::Components> CreateQwen3TtsComponents(
    const Qwen3TtsModelConfig& config, const std::string& model_folder,
    const TextChunkConfig& text_chunk_config,
    std::shared_ptr<ModelResources> resources) {
  TtsSession::Components components;
  components.text_source =
      std::make_unique<StreamTextSource>(text_chunk_config);

  LITERT_ASSIGN_OR_RETURN(
      auto frontend, Qwen3TtsFrontendStage::Create(components.text_source.get(),
                                                   model_folder, config,
                                                   resources));
  LITERT_ASSIGN_OR_RETURN(auto acoustic,
                          Qwen3TtsAcousticPredictorStage::Create(
                              frontend.get(), config, resources));
  LITERT_ASSIGN_OR_RETURN(auto latent,
                          Qwen3TtsLatentDecoderStage::Create(acoustic.get()));
  LITERT_ASSIGN_OR_RETURN(
      auto vocoder,
      Qwen3TtsVocoderStage::Create(latent.get(), config, resources));

  components.intermediate_stages.push_back(std::move(frontend));
  components.intermediate_stages.push_back(std::move(acoustic));
  components.intermediate_stages.push_back(std::move(latent));
  components.vocoder = std::move(vocoder);

  return components;
}

}  // namespace litert::omni::tts
