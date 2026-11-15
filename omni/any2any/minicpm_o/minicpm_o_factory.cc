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

#include "omni/any2any/minicpm_o/minicpm_o_factory.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_audio_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_fan_out_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_frontend_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_image_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/any2any/minicpm_o/minicpm_o_text_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_tts_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_vocoder_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace {

class BackboneLmModelResources : public lm::ModelResources {
 public:
  explicit BackboneLmModelResources(std::shared_ptr<lm::ModelResources> inner)
      : inner_(std::move(inner)) {}

  using lm::ModelResources::GetTFLiteModel;
  using lm::ModelResources::GetTFLiteModelBackendConstraint;
  using lm::ModelResources::GetTFLiteModelBuffer;
  using lm::ModelResources::GetTFLiteModelPreferActivationType;
  using lm::ModelResources::GetTFLiteModelSectionFileRegion;
  using lm::ModelResources::GetTokenizer;
  using lm::ModelResources::GetWeightInMemoryMap;
  using lm::ModelResources::GetWeightsSectionOffset;
  using lm::ModelResources::HasTFLiteModel;

  bool HasTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return HasTFLiteModel(lm::TfLiteModelTypeToWireString(model_type));
  }

  bool HasTFLiteModel(absl::string_view model_type_str) override {
    return inner_->HasTFLiteModel(MapModelTypeString(model_type_str));
  }

  absl::StatusOr<const Model*> GetTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetTFLiteModel(lm::TfLiteModelTypeToWireString(model_type));
  }

  absl::StatusOr<const Model*> GetTFLiteModel(
      absl::string_view model_type_str) override {
    return inner_->GetTFLiteModel(MapModelTypeString(model_type_str));
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetTFLiteModelBuffer(lm::TfLiteModelTypeToWireString(model_type));
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      absl::string_view model_type_str) override {
    return inner_->GetTFLiteModelBuffer(MapModelTypeString(model_type_str));
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetWeightsSectionOffset(lm::TfLiteModelTypeToWireString(model_type));
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      absl::string_view model_type_str) override {
    return inner_->GetWeightsSectionOffset(MapModelTypeString(model_type_str));
  }

  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetTFLiteModelSectionFileRegion(
        lm::TfLiteModelTypeToWireString(model_type));
  }

  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      absl::string_view model_type_str) override {
    return inner_->GetTFLiteModelSectionFileRegion(
        MapModelTypeString(model_type_str));
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetTFLiteModelBackendConstraint(
        lm::TfLiteModelTypeToWireString(model_type));
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      absl::string_view model_type_str) override {
    return inner_->GetTFLiteModelBackendConstraint(
        MapModelTypeString(model_type_str));
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetTFLiteModelPreferActivationType(
        lm::TfLiteModelTypeToWireString(model_type));
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      absl::string_view model_type_str) override {
    return inner_->GetTFLiteModelPreferActivationType(
        MapModelTypeString(model_type_str));
  }

  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(
      lm::proto::LlmMetadata::TfLiteModelType model_type) const override {
    return GetWeightInMemoryMap(lm::TfLiteModelTypeToWireString(model_type));
  }

  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(absl::string_view model_type_str) const override {
    return inner_->GetWeightInMemoryMap(MapModelTypeString(model_type_str));
  }

  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return inner_->GetScopedFile();
  }

  absl::StatusOr<std::unique_ptr<lm::Tokenizer>> GetTokenizer() override {
    return inner_->GetTokenizer();
  }

  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override {
    return inner_->GetLlmMetadata();
  }

  absl::StatusOr<const lm::proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return inner_->GetExecutorMetadata();
  }

  absl::StatusOr<const lm::proto::Any2AnyMetadata*> GetAny2AnyMetadata()
      override {
    return inner_->GetAny2AnyMetadata();
  }

 private:
  static std::string MapModelTypeString(absl::string_view model_type_str) {
    if (absl::EqualsIgnoreCase(
            model_type_str,
            lm::TfLiteModelTypeToWireString(
                lm::proto::LlmMetadata::TF_LITE_PREFILL_DECODE))) {
      return lm::TfLiteModelTypeToWireString(
          lm::proto::Any2AnyMetadata::TF_LITE_LLM_BACKBONE);
    }
    if (absl::EqualsIgnoreCase(model_type_str,
                               lm::TfLiteModelTypeToWireString(
                                   lm::proto::LlmMetadata::TF_LITE_EMBEDDER))) {
      return lm::TfLiteModelTypeToWireString(
          lm::proto::Any2AnyMetadata::TF_LITE_TEXT_EMBEDDER);
    }
    return std::string(model_type_str);
  }

  std::shared_ptr<lm::ModelResources> inner_;
};

absl::StatusOr<std::unique_ptr<LiteRtLmRunner>> CreateBackboneLmRunner(
    Environment& env, const ModelOptions& options,
    std::shared_ptr<lm::ModelResources> lm_resources) {
  std::optional<lm::ModelAssets> model_assets;
  auto scoped_file = lm_resources->GetScopedFile();
  if (scoped_file.ok()) {
    LITERT_ASSIGN_OR_RETURN(auto dup_file, scoped_file->get().Duplicate());
    auto shared_scoped_file =
        std::make_shared<lm::ScopedFile>(std::move(dup_file));
    ABSL_ASSIGN_OR_RETURN(
        model_assets,
        lm::ModelAssets::Create(shared_scoped_file, options.model_dir));
  } else {
    ABSL_ASSIGN_OR_RETURN(model_assets,
                          lm::ModelAssets::Create(options.model_dir));
  }
  ABSL_ASSIGN_OR_RETURN(auto executor_settings,
                        lm::LlmExecutorSettings::CreateDefault(
                            std::move(*model_assets), options.backend,
                            /*sampler_backend=*/lm::Backend::CPU));
  if (options.backend == lm::Backend::CPU && options.num_threads > 0) {
    lm::CpuConfig cpu_config;
    cpu_config.number_of_threads = options.num_threads;
    executor_settings.SetBackendConfig(cpu_config);
  } else if (options.backend == lm::Backend::GPU) {
    executor_settings.SetActivationDataType(
        options.force_fp32_on_gpu ? lm::ActivationDataType::FLOAT32
                                  : lm::ActivationDataType::FLOAT16);
  }
  if (!options.cache_dir.empty()) {
    executor_settings.SetCacheDir(options.cache_dir);
  }
  auto backbone_resources =
      std::make_unique<BackboneLmModelResources>(std::move(lm_resources));
  LITERT_ASSIGN_OR_RETURN(auto executor,
                          lm::CreateLlmLiteRtCompiledModelExecutor(
                              executor_settings, env, *backbone_resources));
  return std::make_unique<LiteRtLmRunnerImpl>(std::move(executor),
                                              std::move(backbone_resources));
}

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

absl::Status InitMiniCpmOResources(const MiniCpmOModelConfig& config,
                                   absl::string_view model_path,
                                   absl::string_view cache_dir,
                                   lm::Backend backend, int num_threads,
                                   Environment& env,
                                   ModelResources& resources) {
  const ModelOptions cpu_options =
      MakeModelOptions(model_path, cache_dir, lm::Backend::CPU, num_threads,
                       config.force_fp32_on_gpu);
  const ModelOptions vision_options = MakeModelOptions(
      model_path, cache_dir, config.vision_backend.value_or(backend),
      num_threads, config.force_fp32_on_gpu);
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
          "MiniCPM-o requires a .litertlm model file, got: ", model_path));
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
      env, *lm_resources, lm::proto::Any2AnyMetadata::TF_LITE_VISION_ENCODER,
      "vision_encoder", vision_options, /*required=*/false, resources));

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

absl::StatusOr<Stage<Output>* absl_nonnull> CreateMiniCpmOComponents(
    const MiniCpmOModelConfig& config,
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
        "LmModelResources must be initialized for MiniCPM-o.");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<lm::Tokenizer> lm_tokenizer,
                        resources->GetLmModelResources()->GetTokenizer());
  std::shared_ptr<support::Tokenizer> tokenizer = std::move(lm_tokenizer);

  auto fan_out = std::make_unique<MiniCpmOInputFanOutStage>(
      input_source.get(), resources->HasCompiledModel("audio_encoder"),
      resources->HasCompiledModel("vision_encoder"));
  ABSL_ASSIGN_OR_RETURN(
      auto text_stage,
      MiniCpmOTextStage::Create(
          &fan_out->TextQueue(), &fan_out->EndOfTurnQueue(),
          &fan_out->EndOfInputQueue(), tokenizer,
          std::make_unique<LiteRtRunnerImpl>(text_embedder_model.get()),
          config));

  std::unique_ptr<MiniCpmOAudioStage> audio_stage;
  if (resources->HasCompiledModel("audio_encoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> audio_encoder_model,
                          resources->GetCompiledModel("audio_encoder"));
    ABSL_ASSIGN_OR_RETURN(
        audio_stage,
        MiniCpmOAudioStage::Create(
            &fan_out->AudioQueue(), &fan_out->AudioInputMetadataQueue(),
            std::make_unique<LiteRtRunnerImpl>(audio_encoder_model.get()),
            config));
  }

  std::unique_ptr<MiniCpmOImageStage> image_stage;
  if (resources->HasCompiledModel("vision_encoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vision_encoder_model,
                          resources->GetCompiledModel("vision_encoder"));
    MiniCpmOImageStage::InputIndices input_indices{
        .images = ResolveInputIndex(*vision_encoder_model, "images")
                      .value_or(ResolveInputIndex(*vision_encoder_model,
                                                  "pixel_values")
                                    .value_or(0)),
        .positions_xy = ResolveInputIndex(*vision_encoder_model, "positions_xy")
                            .value_or(ResolveInputIndex(*vision_encoder_model,
                                                        "position_ids")
                                          .value_or(1)),
        .vit_positions =
            ResolveInputIndex(*vision_encoder_model, "vit_positions")
                .value_or(2),
    };
    ABSL_ASSIGN_OR_RETURN(
        image_stage,
        MiniCpmOImageStage::Create(
            &fan_out->ImageQueue(),
            std::make_unique<LiteRtRunnerImpl>(vision_encoder_model.get()),
            config, input_indices));
  }

  auto frontend = std::make_unique<MiniCpmOFrontendStage>(
      text_stage.get(), audio_stage.get(), image_stage.get(), config);

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                        resources->GetLmRunner("llm_backbone"));
  ABSL_RETURN_IF_ERROR(llm_runner->Reset());
  ABSL_ASSIGN_OR_RETURN(
      auto llm, MiniCpmOLlmStage::Create(frontend.get(), tokenizer,
                                         std::move(llm_runner), config));

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
  ABSL_ASSIGN_OR_RETURN(
      auto tts, MiniCpmOTtsStage::Create(llm.get(), config,
                                         std::move(tts_projector_runner),
                                         std::move(tts_decoder_runner)));

  std::unique_ptr<LiteRtRunner> vocoder_runner;
  if (resources->HasCompiledModel("audio_vocoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vocoder_model,
                          resources->GetCompiledModel("audio_vocoder"));
    vocoder_runner = std::make_unique<LiteRtRunnerImpl>(vocoder_model.get());
  }
  ABSL_ASSIGN_OR_RETURN(
      auto vocoder, MiniCpmOVocoderStage::Create(tts.get(), config,
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
