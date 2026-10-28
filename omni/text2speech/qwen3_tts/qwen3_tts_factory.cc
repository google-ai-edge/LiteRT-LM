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

#include "omni/text2speech/qwen3_tts/qwen3_tts_factory.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
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
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_acoustic_predictor_stage.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_frontend_stage.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_latent_decoder_stage.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_vocoder_stage.h"
#include "omni/text2speech/stream_text_source.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/util/scoped_file.h"

namespace litert::omni::tts {
namespace {

class TalkerLmModelResources : public lm::ModelResources {
 public:
  explicit TalkerLmModelResources(std::shared_ptr<lm::ModelResources> inner)
      : inner_(std::move(inner)) {}

  using lm::ModelResources::GetTFLiteModel;
  using lm::ModelResources::GetTFLiteModelBackendConstraint;
  using lm::ModelResources::GetTFLiteModelBuffer;
  using lm::ModelResources::GetTFLiteModelPreferActivationType;
  using lm::ModelResources::GetTFLiteModelSectionFileRegion;
  using lm::ModelResources::GetTokenizer;
  using lm::ModelResources::GetWeightsSectionOffset;
  using lm::ModelResources::HasTFLiteModel;

  bool HasTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    if (model_type == lm::proto::LlmMetadata::TF_LITE_PREFILL_DECODE) {
      return inner_->HasTFLiteModel(
          lm::proto::Text2SpeechMetadata::TF_LITE_TALKER);
    }
    return inner_->HasTFLiteModel(model_type);
  }

  bool HasTFLiteModel(absl::string_view model_type_str) override {
    if (auto llm_type = lm::ParseLlmTfLiteModelType(model_type_str);
        llm_type.ok() &&
        *llm_type == lm::proto::LlmMetadata::TF_LITE_PREFILL_DECODE) {
      return inner_->HasTFLiteModel(
          lm::proto::Text2SpeechMetadata::TF_LITE_TALKER);
    }
    return inner_->HasTFLiteModel(model_type_str);
  }

  absl::StatusOr<const Model*> GetTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    if (model_type == lm::proto::LlmMetadata::TF_LITE_PREFILL_DECODE) {
      return inner_->GetTFLiteModel(
          lm::proto::Text2SpeechMetadata::TF_LITE_TALKER);
    }
    return inner_->GetTFLiteModel(model_type);
  }

  absl::StatusOr<const Model*> GetTFLiteModel(
      absl::string_view model_type_str) override {
    return inner_->GetTFLiteModel(model_type_str);
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    if (model_type == lm::proto::LlmMetadata::TF_LITE_PREFILL_DECODE) {
      return inner_->GetTFLiteModelBuffer(
          lm::proto::Text2SpeechMetadata::TF_LITE_TALKER);
    }
    return inner_->GetTFLiteModelBuffer(model_type);
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      absl::string_view model_type_str) override {
    return inner_->GetTFLiteModelBuffer(model_type_str);
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

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return inner_->GetTFLiteModelBackendConstraint(model_type);
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return inner_->GetTFLiteModelPreferActivationType(model_type);
  }

  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return inner_->GetScopedFile();
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return inner_->GetWeightsSectionOffset(model_type);
  }

  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return inner_->GetTFLiteModelSectionFileRegion(model_type);
  }

 private:
  std::shared_ptr<lm::ModelResources> inner_;
};

absl::StatusOr<std::unique_ptr<LiteRtLmRunner>> CreateTalkerLmRunner(
    Environment& env, const ModelOptions& options,
    std::shared_ptr<lm::ModelResources> lm_resources) {
  std::optional<lm::ModelAssets> model_assets;
  const std::string litertlm_path =
      absl::EndsWithIgnoreCase(options.model_dir, ".litertlm")
          ? std::string(options.model_dir)
          : ResolveLitertLmPath(options.model_dir);
  auto scoped_file = lm_resources->GetScopedFile();
  if (scoped_file.ok()) {
    LITERT_ASSIGN_OR_RETURN(auto dup_file, scoped_file->get().Duplicate());
    auto shared_scoped_file =
        std::make_shared<lm::ScopedFile>(std::move(dup_file));
    ABSL_ASSIGN_OR_RETURN(model_assets, lm::ModelAssets::Create(
                                            shared_scoped_file, litertlm_path));
  } else {
    ABSL_ASSIGN_OR_RETURN(model_assets, lm::ModelAssets::Create(litertlm_path));
  }
  // Use `sampler_backend = lm::Backend::CPU` for Qwen3-TTS's Talker LM so
  // `Decode()` keeps its output tensor (`[cb0_logits (3072), hidden (1024)]`)
  // in FP32 for CPU-side `cb0` sampling and CPU MTP conditioning.
  ABSL_ASSIGN_OR_RETURN(auto executor_settings,
                        lm::LlmExecutorSettings::CreateDefault(
                            std::move(*model_assets), options.backend,
                            /*sampler_backend=*/lm::Backend::CPU));
  if (options.backend == lm::Backend::CPU && options.num_threads > 0) {
    lm::CpuConfig cpu_config;
    cpu_config.number_of_threads = options.num_threads;
    executor_settings.SetBackendConfig(cpu_config);
  }
  if (options.backend == lm::Backend::GPU) {
    executor_settings.SetActivationDataType(
        options.force_fp32_on_gpu ? lm::ActivationDataType::FLOAT32
                                  : lm::ActivationDataType::FLOAT16);
  }
  if (!options.cache_dir.empty()) {
    executor_settings.SetCacheDir(options.cache_dir);
  }
  auto talker_resources =
      std::make_unique<TalkerLmModelResources>(std::move(lm_resources));
  LITERT_ASSIGN_OR_RETURN(auto executor,
                          lm::CreateLlmLiteRtCompiledModelExecutor(
                              executor_settings, env, *talker_resources));
  return std::make_unique<LiteRtLmRunnerImpl>(std::move(executor),
                                              std::move(talker_resources));
}

}  // namespace

absl::Status InitQwen3TtsResources(const Qwen3TtsModelConfig& config,
                                   absl::string_view model_folder,
                                   absl::string_view cache_dir,
                                   lm::Backend backend, int num_threads,
                                   Environment& env,
                                   ModelResources& resources) {
  ModelOptions model_options;
  model_options.model_dir = model_folder;
  model_options.cache_dir = cache_dir;
  model_options.backend = backend;
  model_options.num_threads = num_threads;
  model_options.force_fp32_on_gpu = config.force_fp32_on_gpu;

  // Only the 28-layer Talker LM runs on GPU when backend == GPU.
  // Embedding lookup tables, the single-shot text projection MLP, the 16-step
  // inner MTP drafter loop (which otherwise incurs 16 synchronous CPU<->GPU
  // round-trips per 80 ms frame), and the 1D conv/transpose-conv codec vocoder
  // (whose sequence lengths up to 122,886 exceed mobile OpenCL 2D texture
  // dimension limits of 16,384) execute on CPU.
  ModelOptions cpu_model_options = model_options;
  cpu_model_options.backend = lm::Backend::CPU;

  if (!resources.HasLmModelResources() && !model_folder.empty()) {
    const std::string litertlm_path =
        absl::EndsWithIgnoreCase(model_folder, ".litertlm")
            ? std::string(model_folder)
            : ResolveLitertLmPath(model_folder);
    if (!litertlm_path.empty()) {
      LITERT_ASSIGN_OR_RETURN(auto lm_resources,
                              CreateLmModelResources(litertlm_path));
      resources.SetLmModelResources(std::move(lm_resources));
    }
  }

  if (resources.HasLmModelResources()) {
    auto lm_resources = resources.GetLmModelResources();

    LITERT_ASSIGN_OR_RETURN(
        absl::string_view text_emb_buf,
        lm_resources->GetTFLiteModelBuffer(
            lm::proto::Text2SpeechMetadata::TF_LITE_TEXT_EMBEDDING));
    LITERT_ASSIGN_OR_RETURN(auto text_emb, CreateCompiledModelFromBuffer(
                                               env, cpu_model_options,
                                               text_emb_buf, "text_embedding"));
    ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
        "text_embedding",
        std::make_shared<CompiledModel>(std::move(text_emb))));

    LITERT_ASSIGN_OR_RETURN(
        absl::string_view text_proj_buf,
        lm_resources->GetTFLiteModelBuffer(
            lm::proto::Text2SpeechMetadata::TF_LITE_TEXT_PROJECTION));
    LITERT_ASSIGN_OR_RETURN(
        auto text_proj,
        CreateCompiledModelFromBuffer(env, cpu_model_options, text_proj_buf,
                                      "text_projection"));
    ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
        "text_projection",
        std::make_shared<CompiledModel>(std::move(text_proj))));

    LITERT_ASSIGN_OR_RETURN(
        auto talker_runner,
        CreateTalkerLmRunner(env, model_options, lm_resources));
    ABSL_RETURN_IF_ERROR(
        resources.AddLmRunner("talker", std::move(talker_runner)));

    LITERT_ASSIGN_OR_RETURN(
        absl::string_view mtp_buf,
        lm_resources->GetTFLiteModelBuffer(
            lm::proto::Text2SpeechMetadata::TF_LITE_MTP_DRAFTER));
    LITERT_ASSIGN_OR_RETURN(
        auto mtp,
        CreateCompiledModelFromBuffer(env, cpu_model_options, mtp_buf, "mtp"));
    ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
        "mtp", std::make_shared<CompiledModel>(std::move(mtp))));

    LITERT_ASSIGN_OR_RETURN(
        absl::string_view codec_emb_buf,
        lm_resources->GetTFLiteModelBuffer(
            lm::proto::Text2SpeechMetadata::TF_LITE_CODEC_EMBEDDING));
    LITERT_ASSIGN_OR_RETURN(
        auto codec_emb,
        CreateCompiledModelFromBuffer(env, cpu_model_options, codec_emb_buf,
                                      "codec_embedding"));
    ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
        "codec_embedding",
        std::make_shared<CompiledModel>(std::move(codec_emb))));

    LITERT_ASSIGN_OR_RETURN(
        absl::string_view mtp_emb_buf,
        lm_resources->GetTFLiteModelBuffer(
            lm::proto::Text2SpeechMetadata::TF_LITE_MTP_EMBEDDING));
    LITERT_ASSIGN_OR_RETURN(auto mtp_emb, CreateCompiledModelFromBuffer(
                                              env, cpu_model_options,
                                              mtp_emb_buf, "mtp_embedding"));
    ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
        "mtp_embedding", std::make_shared<CompiledModel>(std::move(mtp_emb))));

    LITERT_ASSIGN_OR_RETURN(
        absl::string_view codec_buf,
        lm_resources->GetTFLiteModelBuffer(
            lm::proto::Text2SpeechMetadata::TF_LITE_CODEC_DECODER));
    LITERT_ASSIGN_OR_RETURN(
        auto codec, CreateCompiledModelFromBuffer(env, cpu_model_options,
                                                  codec_buf, "codec"));
    ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
        "codec", std::make_shared<CompiledModel>(std::move(codec))));

    return absl::OkStatus();
  }

  LITERT_ASSIGN_OR_RETURN(
      auto text_emb,
      CreateCompiledModel(env, cpu_model_options, config.text_embedding_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "text_embedding", std::make_shared<CompiledModel>(std::move(text_emb))));

  LITERT_ASSIGN_OR_RETURN(
      auto text_proj,
      CreateCompiledModel(env, cpu_model_options, config.text_projection_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "text_projection",
      std::make_shared<CompiledModel>(std::move(text_proj))));

  LITERT_ASSIGN_OR_RETURN(
      auto talker_runner,
      CreateLmRunner(env, model_options, config.talker_file));
  ABSL_RETURN_IF_ERROR(
      resources.AddLmRunner("talker", std::move(talker_runner)));

  LITERT_ASSIGN_OR_RETURN(auto mtp,
                          CreateCompiledModelForStatefulRunner(
                              env, cpu_model_options, config.mtp_file,
                              /*signature_name=*/"", /*num_non_state_inputs=*/3,
                              /*num_non_state_outputs=*/1));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "mtp", std::make_shared<CompiledModel>(std::move(mtp))));

  LITERT_ASSIGN_OR_RETURN(
      auto codec_emb,
      CreateCompiledModel(env, cpu_model_options, config.codec_embedding_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "codec_embedding",
      std::make_shared<CompiledModel>(std::move(codec_emb))));

  LITERT_ASSIGN_OR_RETURN(
      auto mtp_emb,
      CreateCompiledModel(env, cpu_model_options, config.mtp_embedding_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "mtp_embedding", std::make_shared<CompiledModel>(std::move(mtp_emb))));

  LITERT_ASSIGN_OR_RETURN(
      auto codec,
      CreateCompiledModel(env, cpu_model_options, config.codec_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "codec", std::make_shared<CompiledModel>(std::move(codec))));

  return absl::OkStatus();
}

absl::Status CreateQwen3TtsComponents(
    const Qwen3TtsModelConfig& config, absl::string_view model_folder,
    std::unique_ptr<StreamTextSource> absl_nonnull text_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages,
    Stage<Output>* absl_nullable* absl_nonnull output_stage) {
  LITERT_ASSIGN_OR_RETURN(
      auto frontend, Qwen3TtsFrontendStage::Create(
                         text_source.get(), model_folder, config, resources));
  LITERT_ASSIGN_OR_RETURN(auto acoustic,
                          Qwen3TtsAcousticPredictorStage::Create(
                              frontend.get(), config, resources));
  LITERT_ASSIGN_OR_RETURN(auto latent,
                          Qwen3TtsLatentDecoderStage::Create(acoustic.get()));
  LITERT_ASSIGN_OR_RETURN(
      auto vocoder,
      Qwen3TtsVocoderStage::Create(latent.get(), config, resources));

  *output_stage = vocoder.get();
  // The first stage must be `StreamTextSource`.
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(frontend));
  stages.push_back(std::move(acoustic));
  stages.push_back(std::move(latent));
  stages.push_back(std::move(vocoder));
  return absl::OkStatus();
}

}  // namespace litert::omni::tts
