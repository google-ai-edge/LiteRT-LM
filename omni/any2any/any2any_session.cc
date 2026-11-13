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

#include "omni/any2any/any2any_session.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
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

lm::proto::Any2AnySessionConfig ResolveAny2AnySessionConfig(
    const MessageLite* absl_nullable session_config) {
  lm::proto::Any2AnySessionConfig resolved;
  if (session_config != nullptr) {
    resolved.CheckTypeAndMergeFrom(*session_config);
  }
  return resolved;
}

}  // namespace

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

Any2AnySessionFactory::Any2AnySessionFactory(
    const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const lm::proto::Any2AnyMetadata& metadata,
    std::vector<std::string> available_voices,
    std::shared_ptr<ModelResources> absl_nullable model_resources)
    : engine_settings_(settings),
      session_config_(ResolveAny2AnySessionConfig(session_config)),
      metadata_(metadata),
      available_voices_(std::move(available_voices)),
      model_resources_(std::move(model_resources)),
      thread_pool_(std::make_unique<lm::ThreadPool>(
          "any2any_engine_pool",
          settings.num_threads > 0 ? settings.num_threads : 4)) {}

Any2AnySessionFactory::~Any2AnySessionFactory() = default;

std::vector<std::string> Any2AnySessionFactory::GetAvailableLanguages() const {
  return std::vector<std::string>(metadata_.supported_languages().begin(),
                                  metadata_.supported_languages().end());
}

bool Any2AnySessionFactory::HasVoice(absl::string_view voice) const {
  return absl::c_linear_search(available_voices_, voice);
}

std::string Any2AnySessionFactory::GetDefaultVoice(
    absl::string_view /*language*/) const {
  if (!available_voices_.empty()) {
    return available_voices_.front();
  }
  return "";
}

lm::proto::Any2AnySessionConfig Any2AnySessionFactory::MergeSessionConfig(
    const lm::proto::Any2AnySessionConfig& override_config) const {
  lm::proto::Any2AnySessionConfig merged = session_config_;
  merged.MergeFrom(override_config);
  return merged;
}

int Any2AnySessionFactory::GetInputSampleRateHz() const {
  if (metadata_.input_sample_rate_hz() > 0) {
    return metadata_.input_sample_rate_hz();
  }
  return kDefaultInputSampleRateHz;
}

absl::StatusOr<std::unique_ptr<OmniSession>> Any2AnySessionFactory::Create(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  return CreateWithInputSource(session_config_, std::move(input_source));
}

}  // namespace litert::omni::any2any
