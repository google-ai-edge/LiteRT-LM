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

#include "omni/any2any/qwen_omni/qwen_omni_factory.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/any2any/any2any_input_fan_out_stage.h"
#include "omni/any2any/any2any_resources.h"
#include "omni/any2any/qwen_omni/qwen_omni_audio_stage.h"
#include "omni/any2any/qwen_omni/qwen_omni_frontend_stage.h"
#include "omni/any2any/qwen_omni/qwen_omni_image_stage.h"
#include "omni/any2any/qwen_omni/qwen_omni_llm_stage.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/any2any/qwen_omni/qwen_omni_text_stage.h"
#include "omni/any2any/qwen_omni/qwen_omni_tts_stage.h"
#include "omni/any2any/qwen_omni/qwen_omni_vocoder_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

void PopulateQwenOmniConfigFromProto(const lm::proto::Any2AnyMetadata& metadata,
                                     QwenOmniModelConfig& config) {
  if (metadata.has_input_sample_rate_hz() &&
      metadata.input_sample_rate_hz() > 0) {
    config.audio_input_sample_rate_hz = metadata.input_sample_rate_hz();
  }
  if (metadata.has_output_sample_rate_hz() &&
      metadata.output_sample_rate_hz() > 0) {
    config.audio_output_sample_rate_hz = metadata.output_sample_rate_hz();
  }
  if (metadata.has_system_prompt() && !metadata.system_prompt().empty()) {
    config.system_prompt = metadata.system_prompt();
  }
  if (!metadata.any2any_model_type().has_qwen_omni()) {
    return;
  }
  const lm::proto::QwenOmni& proto = metadata.any2any_model_type().qwen_omni();
  if (proto.has_llm_hidden_size() && proto.llm_hidden_size() > 0) {
    config.llm_hidden_size = proto.llm_hidden_size();
  }
  if (proto.has_llm_vocab_size() && proto.llm_vocab_size() > 0) {
    config.llm_vocab_size = proto.llm_vocab_size();
  }
  if (proto.has_tts_hidden_size() && proto.tts_hidden_size() > 0) {
    config.tts_hidden_size = proto.tts_hidden_size();
  }
  if (proto.has_tts_vocab_size() && proto.tts_vocab_size() > 0) {
    config.tts_vocab_size = proto.tts_vocab_size();
  }
  if (proto.has_tts_num_mel_bins() && proto.tts_num_mel_bins() > 0) {
    config.tts_num_mel_bins = proto.tts_num_mel_bins();
  }
  if (proto.has_tts_num_codebooks() && proto.tts_num_codebooks() > 0) {
    config.tts_num_codebooks = proto.tts_num_codebooks();
  }
  if (proto.has_vision_image_size() && proto.vision_image_size() > 0) {
    config.vision_image_size = proto.vision_image_size();
  }
  if (proto.has_vision_patch_size() && proto.vision_patch_size() > 0) {
    config.vision_patch_size = proto.vision_patch_size();
  }
  if (proto.has_vision_spatial_merge_size() &&
      proto.vision_spatial_merge_size() > 0) {
    config.vision_spatial_merge_size = proto.vision_spatial_merge_size();
  }
  if (proto.has_audio_num_mel_bins() && proto.audio_num_mel_bins() > 0) {
    config.audio_num_mel_bins = proto.audio_num_mel_bins();
  }
  if (proto.has_audio_chunk_frames() && proto.audio_chunk_frames() > 0) {
    config.audio_chunk_frames = proto.audio_chunk_frames();
  }
  if (proto.has_audio_pool_step() && proto.audio_pool_step() > 0) {
    config.audio_pool_step = proto.audio_pool_step();
  }
  if (proto.has_generate_audio()) {
    config.generate_audio = proto.generate_audio();
  }
  if (proto.has_max_new_tokens() && proto.max_new_tokens() > 0) {
    config.max_new_tokens = proto.max_new_tokens();
  }
  if (proto.has_max_tts_tokens() && proto.max_tts_tokens() > 0) {
    config.max_tts_tokens = proto.max_tts_tokens();
  }
  if (proto.has_voice() && !proto.voice().empty() &&
      !absl::EqualsIgnoreCase(proto.voice(), "default")) {
    if (absl::EqualsIgnoreCase(proto.voice(), "Ethan")) {
      config.voice_name = "Ethan";
      config.tts_text_bos_token_id = qwen_omni::kEthanTextBosTokenId;
    } else if (absl::EqualsIgnoreCase(proto.voice(), "Chelsie")) {
      config.voice_name = "Chelsie";
      config.tts_text_bos_token_id = qwen_omni::kChelsieTextBosTokenId;
    } else {
      config.voice_name = proto.voice();
    }
  }
  if (proto.has_tts_text_bos_token_id() && proto.tts_text_bos_token_id() > 0) {
    config.tts_text_bos_token_id = proto.tts_text_bos_token_id();
  }
  if (proto.has_language() && !proto.language().empty()) {
    config.language = proto.language();
  }
  if (proto.has_is_moe()) {
    config.is_moe = proto.is_moe();
  }
  if (config.llm_vocab_size > 0) {
    if (config.tts_text_pad_token_id >= config.llm_vocab_size) {
      config.tts_text_pad_token_id = 0;
    }
    if (config.tts_text_bos_token_id >= config.llm_vocab_size) {
      config.tts_text_bos_token_id = 0;
    }
    if (config.tts_text_eos_token_id >= config.llm_vocab_size) {
      config.tts_text_eos_token_id = 0;
    }
  }
  if (config.tts_vocab_size > 2) {
    if (config.tts_silence_token_id >= config.tts_vocab_size) {
      config.tts_silence_token_id = config.tts_vocab_size - 3;
    }
    if (config.tts_audio_bos_token_id >= config.tts_vocab_size) {
      config.tts_audio_bos_token_id = config.tts_vocab_size - 2;
    }
    if (config.tts_eos_token_id >= config.tts_vocab_size) {
      config.tts_eos_token_id = config.tts_vocab_size - 1;
    }
  }
}

void ApplySessionConfigToQwenOmniConfig(
    const lm::proto::Any2AnySessionConfig& session_config,
    QwenOmniModelConfig& config) {
  if (session_config.has_generate_audio()) {
    config.generate_audio = session_config.generate_audio();
  }
  if (session_config.has_max_new_tokens() &&
      session_config.max_new_tokens() > 0) {
    config.max_new_tokens = session_config.max_new_tokens();
  }
  if (session_config.has_max_tts_tokens() &&
      session_config.max_tts_tokens() > 0) {
    config.max_tts_tokens = session_config.max_tts_tokens();
  }
  if (session_config.has_temperature()) {
    config.temperature = session_config.temperature();
    config.do_sample = session_config.temperature() > 0.0f;
  }
  if (session_config.has_top_p() && session_config.top_p() > 0.0f) {
    config.top_p = session_config.top_p();
  }
  if (session_config.has_top_k() && session_config.top_k() > 0) {
    config.top_k = session_config.top_k();
  }
  if (session_config.has_tts_temperature()) {
    config.tts_temperature = session_config.tts_temperature();
    config.tts_do_sample = session_config.tts_temperature() > 0.0f;
  }
  if (session_config.has_tts_top_p() && session_config.tts_top_p() > 0.0f) {
    config.tts_top_p = session_config.tts_top_p();
  }
  if (session_config.has_tts_top_k() && session_config.tts_top_k() > 0) {
    config.tts_top_k = session_config.tts_top_k();
  }
  if (session_config.has_system_prompt()) {
    config.system_prompt = session_config.system_prompt();
  }
  if (session_config.has_voice() && !session_config.voice().empty() &&
      !absl::EqualsIgnoreCase(session_config.voice(), "default")) {
    if (absl::EqualsIgnoreCase(session_config.voice(), "Ethan")) {
      config.voice_name = "Ethan";
      if (config.llm_vocab_size > qwen_omni::kEthanTextBosTokenId) {
        config.tts_text_bos_token_id = qwen_omni::kEthanTextBosTokenId;
      }
    } else if (absl::EqualsIgnoreCase(session_config.voice(), "Chelsie")) {
      config.voice_name = "Chelsie";
      if (config.llm_vocab_size > qwen_omni::kChelsieTextBosTokenId) {
        config.tts_text_bos_token_id = qwen_omni::kChelsieTextBosTokenId;
      }
    } else {
      config.voice_name = session_config.voice();
    }
  }
  if (session_config.has_language() && !session_config.language().empty()) {
    config.language = session_config.language();
  }
  if (session_config.has_seed()) {
    config.seed = static_cast<uint64_t>(session_config.seed());
  }
}

absl::StatusOr<ModelResources> CreateQwenOmniResources(
    const QwenOmniModelConfig& config, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  const Any2AnyResourceOptions resource_options{
      .model_display_name = "Qwen-Omni",
      .generate_audio = config.generate_audio,
      .vision_backend = config.vision_backend,
      .audio_encoder_backend = config.audio_encoder_backend,
      .llm_backend = config.llm_backend,
      .tts_backend = config.tts_backend,
      .vocoder_backend = config.vocoder_backend,
      .force_fp32_on_gpu = config.force_fp32_on_gpu,
  };
  return CreateAny2AnyResources(resource_options, model_path, cache_dir,
                                backend, num_threads, std::move(env),
                                std::move(lm_resources));
}

absl::StatusOr<Stage<Output>* absl_nonnull> CreateQwenOmniComponents(
    const QwenOmniModelConfig& config,
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages) {
  if (resources == nullptr) {
    return absl::InvalidArgumentError("ModelResources must not be null.");
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> text_embedder_model,
                        resources->GetCompiledModel("text_embedder"));

  if (!resources->HasLmModelResources()) {
    return absl::FailedPreconditionError(
        "LmModelResources must be initialized for Qwen-Omni.");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<lm::Tokenizer> lm_tokenizer,
                        resources->GetLmModelResources()->GetTokenizer());
  std::shared_ptr<support::Tokenizer> tokenizer = std::move(lm_tokenizer);

  auto fan_out = std::make_unique<Any2AnyInputFanOutStage>(
      input_source.get(), resources->HasCompiledModel("audio_encoder"),
      resources->HasCompiledModel("vision_encoder"));

  ABSL_ASSIGN_OR_RETURN(
      auto text_stage,
      QwenOmniTextStage::Create(
          &fan_out->TextQueue(), &fan_out->EndOfTurnQueue(),
          &fan_out->EndOfInputQueue(), tokenizer,
          std::make_unique<LiteRtRunnerImpl>(text_embedder_model.get()),
          config));

  std::unique_ptr<QwenOmniAudioStage> audio_stage;
  if (resources->HasCompiledModel("audio_encoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> audio_encoder_model,
                          resources->GetCompiledModel("audio_encoder"));
    ABSL_ASSIGN_OR_RETURN(
        audio_stage,
        QwenOmniAudioStage::Create(
            &fan_out->AudioQueue(), &fan_out->AudioInputMetadataQueue(),
            std::make_unique<LiteRtRunnerImpl>(audio_encoder_model.get()),
            config));
  }

  std::unique_ptr<QwenOmniImageStage> image_stage;
  if (resources->HasCompiledModel("vision_encoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vision_encoder_model,
                          resources->GetCompiledModel("vision_encoder"));
    ABSL_ASSIGN_OR_RETURN(
        image_stage,
        QwenOmniImageStage::Create(
            &fan_out->ImageQueue(),
            std::make_unique<LiteRtRunnerImpl>(vision_encoder_model.get()),
            config));
  }

  auto frontend = std::make_unique<QwenOmniFrontendStage>(
      text_stage.get(), audio_stage.get(), image_stage.get(), config);

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                        resources->GetLmRunner("llm_backbone"));
  ABSL_RETURN_IF_ERROR(llm_runner->Reset());
  ABSL_ASSIGN_OR_RETURN(
      auto llm, QwenOmniLlmStage::Create(frontend.get(), tokenizer,
                                         std::move(llm_runner), config));

  std::unique_ptr<LiteRtRunner> tts_projector_runner;
  if (config.generate_audio && resources->HasCompiledModel("tts_projector")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> tts_projector_model,
                          resources->GetCompiledModel("tts_projector"));
    tts_projector_runner =
        std::make_unique<LiteRtRunnerImpl>(tts_projector_model.get());
  }
  std::unique_ptr<LiteRtRunner> tts_decoder_runner;
  if (config.generate_audio && resources->HasCompiledModel("tts_decoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> tts_decoder_model,
                          resources->GetCompiledModel("tts_decoder"));
    tts_decoder_runner =
        std::make_unique<LiteRtRunnerImpl>(tts_decoder_model.get());
  }
  ABSL_ASSIGN_OR_RETURN(
      auto tts, QwenOmniTtsStage::Create(llm.get(), config,
                                         std::move(tts_projector_runner),
                                         std::move(tts_decoder_runner)));

  std::unique_ptr<LiteRtRunner> vocoder_runner;
  if (config.generate_audio && resources->HasCompiledModel("audio_vocoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vocoder_model,
                          resources->GetCompiledModel("audio_vocoder"));
    vocoder_runner = std::make_unique<LiteRtRunnerImpl>(vocoder_model.get());
  }
  ABSL_ASSIGN_OR_RETURN(
      auto vocoder, QwenOmniVocoderStage::Create(tts.get(), config,
                                                 std::move(vocoder_runner)));

  Stage<Output>* output_stage = vocoder.get();
  // The first stage must be `InputSource`.
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(fan_out));
  stages.push_back(std::move(text_stage));
  if (audio_stage != nullptr) {
    stages.push_back(std::move(audio_stage));
  }
  if (image_stage != nullptr) {
    stages.push_back(std::move(image_stage));
  }
  stages.push_back(std::move(frontend));
  stages.push_back(std::move(llm));
  stages.push_back(std::move(tts));
  stages.push_back(std::move(vocoder));
  return output_stage;
}

}  // namespace litert::omni::any2any
