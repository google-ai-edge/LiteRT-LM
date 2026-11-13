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

#include "omni/any2any/lfm2_audio/lfm2_audio_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/any2any/any2any_session.h"
#include "omni/any2any/lfm2_audio/lfm2_audio_model_config.h"
#include "omni/any2any/lfm2_audio/lfm2_audio_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace {

ModelOptions MakeModelOptions(absl::string_view model_path,
                              absl::string_view cache_dir, lm::Backend backend,
                              int num_threads, bool force_fp32_on_gpu) {
  return ModelOptions{
      .model_dir = std::string(model_path),
      .cache_dir = std::string(cache_dir),
      .backend = backend,
      .num_threads = num_threads,
      .force_fp32_on_gpu = force_fp32_on_gpu,
  };
}

absl::Status TryCompileSectionToResources(
    Environment& env, lm::ModelResources& lm_resources,
    lm::proto::Any2AnyMetadata::TfLiteModelType section_type,
    absl::string_view key, const ModelOptions& options, bool required,
    ModelResources& resources) {
  auto buf = lm_resources.GetTFLiteModelBuffer(section_type);
  if (!buf.ok()) {
    if (required) {
      return buf.status();
    }
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(
      auto compiled, CreateCompiledModelFromBuffer(env, options, *buf, key));
  return resources.AddCompiledModel(
      key, std::make_shared<CompiledModel>(std::move(compiled)));
}

}  // namespace

absl::Status InitLfm2AudioResources(const Lfm2AudioModelConfig& config,
                                    absl::string_view model_path,
                                    absl::string_view cache_dir,
                                    lm::Backend backend, int num_threads,
                                    Environment& env,
                                    ModelResources& resources) {
  const ModelOptions cpu_options =
      MakeModelOptions(model_path, cache_dir, lm::Backend::CPU, num_threads,
                       config.force_fp32_on_gpu);
  const ModelOptions audio_enc_options = MakeModelOptions(
      model_path, cache_dir, config.audio_encoder_backend.value_or(backend),
      num_threads, config.force_fp32_on_gpu);
  const ModelOptions llm_options = MakeModelOptions(
      model_path, cache_dir, config.llm_backend.value_or(backend), num_threads,
      config.force_fp32_on_gpu);
  const ModelOptions tts_options = MakeModelOptions(
      model_path, cache_dir, config.tts_backend.value_or(lm::Backend::CPU),
      num_threads, config.force_fp32_on_gpu);
  const ModelOptions vocoder_options = MakeModelOptions(
      model_path, cache_dir, config.vocoder_backend.value_or(backend),
      num_threads, config.force_fp32_on_gpu);

  if (!resources.HasLmModelResources()) {
    if (!absl::EndsWithIgnoreCase(model_path, ".litertlm")) {
      return absl::InvalidArgumentError(absl::StrCat(
          "LFM2.5-Audio requires a .litertlm model file, got: ", model_path));
    }
    ABSL_ASSIGN_OR_RETURN(auto lm_resources,
                          CreateLmModelResources(model_path));
    resources.SetLmModelResources(std::move(lm_resources));
  }

  auto lm_resources = resources.GetLmModelResources();

  ABSL_RETURN_IF_ERROR(TryCompileSectionToResources(
      env, *lm_resources, lm::proto::Any2AnyMetadata::TF_LITE_TEXT_EMBEDDER,
      "text_embedder", cpu_options, /*required=*/true, resources));

  ABSL_RETURN_IF_ERROR(TryCompileSectionToResources(
      env, *lm_resources, lm::proto::Any2AnyMetadata::TF_LITE_AUDIO_ENCODER,
      "audio_encoder", audio_enc_options, /*required=*/false, resources));

  ABSL_ASSIGN_OR_RETURN(auto runner,
                        CreateBackboneLmRunner(env, llm_options, lm_resources));
  ABSL_RETURN_IF_ERROR(
      resources.AddLmRunner("llm_backbone", std::move(runner)));

  ABSL_RETURN_IF_ERROR(TryCompileSectionToResources(
      env, *lm_resources, lm::proto::Any2AnyMetadata::TF_LITE_TTS_PROJECTOR,
      "tts_projector", tts_options, /*required=*/config.generate_audio,
      resources));

  ABSL_RETURN_IF_ERROR(TryCompileSectionToResources(
      env, *lm_resources, lm::proto::Any2AnyMetadata::TF_LITE_TTS_DECODER,
      "tts_decoder", tts_options, /*required=*/config.generate_audio,
      resources));

  ABSL_RETURN_IF_ERROR(TryCompileSectionToResources(
      env, *lm_resources, lm::proto::Any2AnyMetadata::TF_LITE_AUDIO_VOCODER,
      "audio_vocoder", vocoder_options, /*required=*/config.generate_audio,
      resources));

  return absl::OkStatus();
}

absl::StatusOr<Stage<Output>* absl_nonnull> CreateLfm2AudioComponents(
    const Lfm2AudioModelConfig& config,
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
        "LmModelResources must be initialized for LFM2.5-Audio.");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<lm::Tokenizer> lm_tokenizer,
                        resources->GetLmModelResources()->GetTokenizer());
  std::shared_ptr<support::Tokenizer> tokenizer = std::move(lm_tokenizer);

  const bool has_audio_encoder = resources->HasCompiledModel("audio_encoder");
  ABSL_ASSIGN_OR_RETURN(
      auto fan_out,
      Lfm2AudioInputFanOutStage::Create(input_source.get(), has_audio_encoder));
  ABSL_ASSIGN_OR_RETURN(
      auto text_stage,
      Lfm2AudioTextStage::Create(
          fan_out->text_queue(), fan_out->end_of_turn_queue(), tokenizer,
          std::make_shared<LiteRtRunnerImpl>(text_embedder_model.get()),
          config));

  std::unique_ptr<Lfm2AudioStage> audio_stage;
  if (resources->HasCompiledModel("audio_encoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> audio_encoder_model,
                          resources->GetCompiledModel("audio_encoder"));
    ABSL_ASSIGN_OR_RETURN(
        audio_stage,
        Lfm2AudioStage::Create(
            fan_out->audio_queue(), fan_out->audio_input_metadata_queue(),
            std::make_unique<LiteRtRunnerImpl>(audio_encoder_model.get()),
            config));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(
          text_stage.get(), audio_stage.get(), tokenizer,
          std::make_shared<LiteRtRunnerImpl>(text_embedder_model.get()),
          config));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                        resources->GetLmRunner("llm_backbone"));
  ABSL_RETURN_IF_ERROR(llm_runner->Reset());

  std::unique_ptr<LiteRtRunner> tts_projector_runner;
  if (resources->HasCompiledModel("tts_projector")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> tts_projector_model,
                          resources->GetCompiledModel("tts_projector"));
    tts_projector_runner =
        std::make_unique<LiteRtRunnerImpl>(tts_projector_model.get());
  }
  std::unique_ptr<LiteRtRunner> tts_decoder_runner;
  if (resources->HasCompiledModel("tts_decoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> tts_decoder_model,
                          resources->GetCompiledModel("tts_decoder"));
    tts_decoder_runner =
        std::make_unique<LiteRtRunnerImpl>(tts_decoder_model.get());
  }
  std::unique_ptr<LiteRtRunner> vocoder_runner;
  if (resources->HasCompiledModel("audio_vocoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vocoder_model,
                          resources->GetCompiledModel("audio_vocoder"));
    vocoder_runner = std::make_unique<LiteRtRunnerImpl>(vocoder_model.get());
  }

  ABSL_ASSIGN_OR_RETURN(
      auto generation,
      Lfm2AudioGenerationStage::Create(
          frontend.get(), tokenizer, std::move(llm_runner),
          std::move(tts_projector_runner), std::move(tts_decoder_runner),
          std::move(vocoder_runner), config));

  Stage<Output>* output_stage = generation.get();
  // The first stage must be `InputSource`.
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(fan_out));
  stages.push_back(std::move(text_stage));
  if (audio_stage != nullptr) {
    stages.push_back(std::move(audio_stage));
  }
  stages.push_back(std::move(frontend));
  stages.push_back(std::move(generation));
  return output_stage;
}

}  // namespace litert::omni::any2any
