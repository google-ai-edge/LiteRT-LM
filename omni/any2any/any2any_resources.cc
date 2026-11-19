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

#include "omni/any2any/any2any_resources.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
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
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"

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

}  // namespace

BackboneLmModelResources::BackboneLmModelResources(
    std::shared_ptr<lm::ModelResources> absl_nonnull inner)
    : inner_(std::move(inner)) {
  ABSL_CHECK(inner_ != nullptr);
}

bool BackboneLmModelResources::HasTFLiteModel(
    lm::proto::LlmMetadata::TfLiteModelType model_type) {
  return HasTFLiteModel(lm::TfLiteModelTypeToWireString(model_type));
}

bool BackboneLmModelResources::HasTFLiteModel(
    absl::string_view model_type_str) {
  return inner_->HasTFLiteModel(MapModelTypeString(model_type_str));
}

absl::StatusOr<const Model*> BackboneLmModelResources::GetTFLiteModel(
    lm::proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModel(lm::TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<const Model*> BackboneLmModelResources::GetTFLiteModel(
    absl::string_view model_type_str) {
  return inner_->GetTFLiteModel(MapModelTypeString(model_type_str));
}

absl::StatusOr<absl::string_view>
BackboneLmModelResources::GetTFLiteModelBuffer(
    lm::proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelBuffer(lm::TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<absl::string_view>
BackboneLmModelResources::GetTFLiteModelBuffer(
    absl::string_view model_type_str) {
  return inner_->GetTFLiteModelBuffer(MapModelTypeString(model_type_str));
}

absl::StatusOr<std::pair<size_t, size_t>>
BackboneLmModelResources::GetWeightsSectionOffset(
    lm::proto::LlmMetadata::TfLiteModelType model_type) {
  return GetWeightsSectionOffset(lm::TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<std::pair<size_t, size_t>>
BackboneLmModelResources::GetWeightsSectionOffset(
    absl::string_view model_type_str) {
  return inner_->GetWeightsSectionOffset(MapModelTypeString(model_type_str));
}

absl::StatusOr<lm::FileRegion>
BackboneLmModelResources::GetTFLiteModelSectionFileRegion(
    lm::proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelSectionFileRegion(
      lm::TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<lm::FileRegion>
BackboneLmModelResources::GetTFLiteModelSectionFileRegion(
    absl::string_view model_type_str) {
  return inner_->GetTFLiteModelSectionFileRegion(
      MapModelTypeString(model_type_str));
}

std::optional<std::string>
BackboneLmModelResources::GetTFLiteModelBackendConstraint(
    lm::proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelBackendConstraint(
      lm::TfLiteModelTypeToWireString(model_type));
}

std::optional<std::string>
BackboneLmModelResources::GetTFLiteModelBackendConstraint(
    absl::string_view model_type_str) {
  return inner_->GetTFLiteModelBackendConstraint(
      MapModelTypeString(model_type_str));
}

std::optional<std::string>
BackboneLmModelResources::GetTFLiteModelPreferActivationType(
    lm::proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelPreferActivationType(
      lm::TfLiteModelTypeToWireString(model_type));
}

std::optional<std::string>
BackboneLmModelResources::GetTFLiteModelPreferActivationType(
    absl::string_view model_type_str) {
  return inner_->GetTFLiteModelPreferActivationType(
      MapModelTypeString(model_type_str));
}

const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
BackboneLmModelResources::GetWeightInMemoryMap(
    lm::proto::LlmMetadata::TfLiteModelType model_type) const {
  return GetWeightInMemoryMap(lm::TfLiteModelTypeToWireString(model_type));
}

const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
BackboneLmModelResources::GetWeightInMemoryMap(
    absl::string_view model_type_str) const {
  return inner_->GetWeightInMemoryMap(MapModelTypeString(model_type_str));
}

absl::StatusOr<std::reference_wrapper<lm::ScopedFile>>
BackboneLmModelResources::GetScopedFile() {
  return inner_->GetScopedFile();
}

absl::StatusOr<std::unique_ptr<lm::Tokenizer>>
BackboneLmModelResources::GetTokenizer() {
  return inner_->GetTokenizer();
}

absl::StatusOr<const lm::proto::LlmMetadata*>
BackboneLmModelResources::GetLlmMetadata() {
  return inner_->GetLlmMetadata();
}

absl::StatusOr<const lm::proto::ExecutorMetadata*>
BackboneLmModelResources::GetExecutorMetadata() {
  return inner_->GetExecutorMetadata();
}

absl::StatusOr<const lm::proto::Any2AnyMetadata*>
BackboneLmModelResources::GetAny2AnyMetadata() {
  return inner_->GetAny2AnyMetadata();
}

std::string BackboneLmModelResources::MapModelTypeString(
    absl::string_view model_type_str) {
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

absl::StatusOr<std::unique_ptr<LiteRtLmRunner> absl_nonnull>
CreateBackboneLmRunner(
    Environment& env, const ModelOptions& options,
    std::shared_ptr<lm::ModelResources> absl_nonnull lm_resources) {
  if (lm_resources == nullptr) {
    return absl::InvalidArgumentError("lm::ModelResources must not be null.");
  }
  std::optional<lm::ModelAssets> model_assets;
  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> scoped_file =
      lm_resources->GetScopedFile();
  if (scoped_file.ok()) {
    LITERT_ASSIGN_OR_RETURN(lm::ScopedFile dup_file,
                            scoped_file->get().Duplicate());
    auto shared_scoped_file =
        std::make_shared<lm::ScopedFile>(std::move(dup_file));
    ABSL_ASSIGN_OR_RETURN(
        model_assets,
        lm::ModelAssets::Create(shared_scoped_file, options.model_dir));
  } else {
    ABSL_ASSIGN_OR_RETURN(model_assets,
                          lm::ModelAssets::Create(options.model_dir));
  }
  ABSL_ASSIGN_OR_RETURN(lm::LlmExecutorSettings executor_settings,
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
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<lm::LlmExecutor> executor,
                        lm::CreateLlmLiteRtCompiledModelExecutor(
                            executor_settings, env, *backbone_resources));
  return std::make_unique<LiteRtLmRunnerImpl>(std::move(executor),
                                              std::move(backbone_resources));
}

absl::StatusOr<std::shared_ptr<CompiledModel> absl_nullable> TryCompileSection(
    Environment& env, lm::ModelResources& lm_resources,
    lm::proto::Any2AnyMetadata::TfLiteModelType section_type,
    absl::string_view model_name, const ModelOptions& options, bool required) {
  absl::StatusOr<absl::string_view> buf =
      lm_resources.GetTFLiteModelBuffer(section_type);
  if (!buf.ok()) {
    if (!required && absl::IsNotFound(buf.status())) {
      return nullptr;
    }
    return buf.status();
  }
  ABSL_ASSIGN_OR_RETURN(
      CompiledModel compiled,
      CreateCompiledModelFromBuffer(env, options, *buf, model_name));
  return std::make_shared<CompiledModel>(std::move(compiled));
}

absl::StatusOr<ModelResources> CreateAny2AnyResources(
    const Any2AnyResourceOptions& options, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (env == nullptr) {
    return absl::InvalidArgumentError("Environment must not be null.");
  }
  if (lm_resources == nullptr) {
    if (!absl::EndsWithIgnoreCase(model_path, ".litertlm")) {
      const absl::string_view display_name =
          options.model_display_name.empty()
              ? absl::string_view("Any2Any")
              : absl::string_view(options.model_display_name);
      return absl::InvalidArgumentError(absl::StrCat(
          display_name, " requires a .litertlm model file, got: ", model_path));
    }
    ABSL_ASSIGN_OR_RETURN(lm_resources, CreateLmModelResources(model_path));
  }
  if (lm_resources == nullptr) {
    return absl::InternalError("CreateLmModelResources returned null.");
  }
  std::shared_ptr<lm::ModelResources> absl_nonnull resolved_lm_resources =
      std::move(lm_resources);

  ModelResources resources(env);
  resources.SetLmModelResources(resolved_lm_resources);

  const ModelOptions cpu_options =
      MakeModelOptions(model_path, cache_dir, lm::Backend::CPU, num_threads,
                       options.force_fp32_on_gpu);
  const ModelOptions vision_options = MakeModelOptions(
      model_path, cache_dir, options.vision_backend.value_or(backend),
      num_threads, options.force_fp32_on_gpu);
  const ModelOptions audio_enc_options = MakeModelOptions(
      model_path, cache_dir, options.audio_encoder_backend.value_or(backend),
      num_threads, options.force_fp32_on_gpu);
  const ModelOptions llm_options = MakeModelOptions(
      model_path, cache_dir, options.llm_backend.value_or(backend), num_threads,
      options.force_fp32_on_gpu);

  auto compile_and_add =
      [&](lm::proto::Any2AnyMetadata::TfLiteModelType section_type,
          absl::string_view key, const ModelOptions& section_options,
          bool required) -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<CompiledModel> model,
        TryCompileSection(*env, *resolved_lm_resources, section_type, key,
                          section_options, required));
    if (model != nullptr) {
      return resources.AddCompiledModel(key, std::move(model));
    }
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(
      compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_TEXT_EMBEDDER,
                      "text_embedder", cpu_options, /*required=*/true));
  ABSL_RETURN_IF_ERROR(
      compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_VISION_ENCODER,
                      "vision_encoder", vision_options, /*required=*/false));
  ABSL_RETURN_IF_ERROR(
      compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_AUDIO_ENCODER,
                      "audio_encoder", audio_enc_options, /*required=*/false));

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<LiteRtLmRunner> runner,
      CreateBackboneLmRunner(*env, llm_options, resolved_lm_resources));
  ABSL_RETURN_IF_ERROR(
      resources.AddLmRunner("llm_backbone", std::move(runner)));

  if (options.generate_audio) {
    const ModelOptions tts_options = MakeModelOptions(
        model_path, cache_dir, options.tts_backend.value_or(lm::Backend::CPU),
        num_threads, options.force_fp32_on_gpu);
    const ModelOptions vocoder_options = MakeModelOptions(
        model_path, cache_dir, options.vocoder_backend.value_or(backend),
        num_threads, options.force_fp32_on_gpu);
    ABSL_RETURN_IF_ERROR(
        compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_TTS_PROJECTOR,
                        "tts_projector", tts_options, /*required=*/true));
    ABSL_RETURN_IF_ERROR(
        compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_TTS_DECODER,
                        "tts_decoder", tts_options, /*required=*/true));
    ABSL_RETURN_IF_ERROR(
        compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_AUDIO_VOCODER,
                        "audio_vocoder", vocoder_options, /*required=*/true));
  }

  return resources;
}

}  // namespace litert::omni::any2any
