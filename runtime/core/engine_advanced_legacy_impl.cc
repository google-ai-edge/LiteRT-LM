// Copyright 2025 The ODML Authors.
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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/executor/litert_executor_utils.h"
#include "third_party/odml/infra/genai/inference/executor/llm_gpu_artisan_executor.h"
#include "third_party/odml/infra/genai/inference/executor/llm_litert_xnnpack_executor.h"
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_factory.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/audio_litert_compiled_model_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/metadata_util.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

absl::StatusOr<std::unique_ptr<Engine>> CreateEngineAdvancedLegacy(
    EngineSettings engine_settings);

namespace {

namespace oi = ::odml::infra;

class EngineAdvancedLegacyImpl : public Engine {
 public:
  ~EngineAdvancedLegacyImpl() override;

  static absl::StatusOr<std::unique_ptr<Engine>> Create(
      EngineSettings engine_settings, absl::string_view input_prompt_as_hint);

  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) override;

  absl::Status WaitUntilDone(absl::Duration timeout) override;

  const EngineSettings& GetEngineSettings() const override;

 private:
  explicit EngineAdvancedLegacyImpl(
      EngineSettings engine_settings,
      std::unique_ptr<odml::infra::ExecutorModelResources> model_resources,
      std::unique_ptr<ExecutionManager> execution_manager,
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<Tokenizer> task_tokenizer,
      std::optional<BenchmarkInfo> benchmark_info);

  EngineSettings engine_settings_;
  std::unique_ptr<odml::infra::ExecutorModelResources> model_resources_;
  std::shared_ptr<ExecutionManager> execution_manager_;
  Tokenizer* tokenizer_;
  std::unique_ptr<Tokenizer> task_tokenizer_;
  std::optional<BenchmarkInfo> benchmark_info_;
};

absl::StatusOr<std::unique_ptr<LlmExecutor>> BuildExecutor(
    const oi::ExecutorModelResources& model_resources,
    const EngineSettings& engine_settings) {
  bool has_model = model_resources.model;
  if ((engine_settings.GetMainExecutorSettings().GetBackend() !=
       Backend::GPU_ARTISAN) &&
      !has_model) {
    return absl::InternalError(
        "TF_LITE_PREFILL_DECODE model is expected to exist when not using "
        "GPU_ARTISAN backend. But it is null.");
  }
  // Create executor that creates and owns the interpreter and kv cache.
  std::unique_ptr<LlmExecutor> executor;
  ABSL_LOG(INFO) << "Executor settings: "
                 << engine_settings.GetMainExecutorSettings();

  if (engine_settings.GetMainExecutorSettings().GetBackend() == Backend::CPU) {
    ASSIGN_OR_RETURN(executor, oi::LlmLiteRTXnnpackExecutor::Create(
                                   engine_settings.GetMainExecutorSettings(),
                                   model_resources));
  } else if (engine_settings.GetMainExecutorSettings().GetBackend() ==
             Backend::GPU_ARTISAN) {
    if (model_resources.litert_lm_model_resources == nullptr) {
      return absl::InternalError(
          "Failed to build GPU_ARTISAN executor: "
          "model_resources.litert_lm_model_resources is null. ");
    }
    ASSIGN_OR_RETURN(executor, oi::LlmGpuArtisanExecutor::Create(
                                   engine_settings.GetMainExecutorSettings(),
                                   *model_resources.litert_lm_model_resources));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend: ",
                     engine_settings.GetMainExecutorSettings().GetBackend()));
  }

  return std::move(executor);
}

absl::StatusOr<Environment&> GetEnvironment() {
  static absl::NoDestructor<absl::StatusOr<Environment>> kEnvironment(
      [&]() -> absl::StatusOr<Environment> {
        LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
        return std::move(env);
      }());
  if (!kEnvironment->ok()) {
    return kEnvironment->status();
  }
  return **kEnvironment;
}

EngineAdvancedLegacyImpl::~EngineAdvancedLegacyImpl() {
  ABSL_QCHECK_OK(WaitUntilDone(Engine::kDefaultTimeout));
}

EngineAdvancedLegacyImpl::EngineAdvancedLegacyImpl(
    EngineSettings engine_settings,
    std::unique_ptr<oi::ExecutorModelResources> model_resources,
    std::unique_ptr<ExecutionManager> execution_manager,
    Tokenizer* absl_nonnull tokenizer,
    std::unique_ptr<Tokenizer> task_tokenizer,
    std::optional<BenchmarkInfo> benchmark_info)
    : engine_settings_(std::move(engine_settings)),
      model_resources_(std::move(model_resources)),
      execution_manager_(std::move(execution_manager)),
      tokenizer_(std::move(tokenizer)),
      task_tokenizer_(std::move(task_tokenizer)),
      benchmark_info_(std::move(benchmark_info)) {}

// Method to create the Session.
absl::StatusOr<std::unique_ptr<Engine::Session>>
EngineAdvancedLegacyImpl::CreateSession(const SessionConfig& session_config) {
  auto config = session_config;
  RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));
  if (config.AudioModalityEnabled() &&
      model_resources_->litert_lm_model_resources != nullptr) {
    ASSIGN_OR_RETURN(auto audio_executor_properties,
                     GetAudioExecutorPropertiesFromModelResources(
                         *model_resources_->litert_lm_model_resources));
    config.GetMutableAudioExecutorProperties() = audio_executor_properties;
  }
  return InitializeSessionAdvanced(execution_manager_, tokenizer_, config,
                                   benchmark_info_);
}

absl::Status EngineAdvancedLegacyImpl::WaitUntilDone(absl::Duration timeout) {
  return execution_manager_->WaitUntilAllDone(timeout);
}

const EngineSettings& EngineAdvancedLegacyImpl::GetEngineSettings() const {
  return engine_settings_;
}

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> EngineAdvancedLegacyImpl::Create(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
  ABSL_LOG(INFO) << "Constructing legacy EngineImpl...";
  std::optional<BenchmarkInfo> benchmark_info;
  if (engine_settings.IsBenchmarkEnabled()) {
    benchmark_info = std::make_optional<BenchmarkInfo>(
        engine_settings.GetBenchmarkParams().value());
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kExecutor));
  }
  ASSIGN_OR_RETURN(auto scoped_model_file,
                   engine_settings.GetMainExecutorSettings()
                       .GetModelAssets()
                       .GetOrCreateScopedFile());
  ASSIGN_OR_RETURN(
      auto model_resources,
      oi::BuildModelResources(/*model_path=*/"", scoped_model_file));

  proto::LlmMetadata llm_metadata;
  std::unique_ptr<Tokenizer> task_tokenizer;
  Tokenizer* tokenizer = nullptr;
  if (model_resources->litert_lm_model_resources == nullptr) {
    // Handle the .task file format.
    ASSIGN_OR_RETURN(auto resources, ModelAssetBundleResources::Create(
                                         /*tag=*/"", scoped_model_file));
    if (benchmark_info.has_value()) {
      RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
          BenchmarkInfo::InitPhase::kTokenizer));
    }
    ASSIGN_OR_RETURN(auto vocab_buffer, resources->GetFile("TOKENIZER_MODEL"));
    ASSIGN_OR_RETURN(task_tokenizer,
                     SentencePieceTokenizer::CreateFromBuffer(vocab_buffer));
    tokenizer = task_tokenizer.get();
    if (benchmark_info.has_value()) {
      RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
          BenchmarkInfo::InitPhase::kTokenizer));
    }
    ASSIGN_OR_RETURN(auto metadata_buffer, resources->GetFile("METADATA"));
    ASSIGN_OR_RETURN(llm_metadata,
                     ExtractOrConvertLlmMetadata(metadata_buffer));
  } else {
    // Handle the .litert_lm file format.
    ASSIGN_OR_RETURN(
        tokenizer, model_resources->litert_lm_model_resources->GetTokenizer());
    ASSIGN_OR_RETURN(
        auto metadata,
        model_resources->litert_lm_model_resources->GetLlmMetadata());
    llm_metadata = *metadata;
  }
  // Update and load the parameters from the model file and convert the tokens
  // to ids.
  RETURN_IF_ERROR(
      engine_settings.MaybeUpdateAndValidate(*tokenizer, &llm_metadata));

  ASSIGN_OR_RETURN(auto executor,
                   BuildExecutor(*model_resources, engine_settings));

  ASSIGN_OR_RETURN(auto& litert_env, GetEnvironment());

  std::unique_ptr<VisionExecutorSettings> vision_executor_settings_ptr;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    ASSIGN_OR_RETURN(
        auto vision_executor_settings,
        VisionExecutorSettings::CreateDefault(
            engine_settings.GetMainExecutorSettings().GetModelAssets(),
            /*encoder_backend=*/
            engine_settings.GetVisionExecutorSettings()->GetBackend(),
            /*adapter_backend=*/Backend::CPU));
    vision_executor_settings_ptr = std::make_unique<VisionExecutorSettings>(
        std::move(vision_executor_settings));
  }

  std::unique_ptr<AudioExecutorSettings> audio_executor_settings_ptr;
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    const auto audio_backend =
        engine_settings.GetAudioExecutorSettings()->GetBackend();
    ASSIGN_OR_RETURN(
        auto audio_executor_settings,
        AudioExecutorSettings::CreateDefault(
            engine_settings.GetMainExecutorSettings().GetModelAssets(),
            engine_settings.GetMainExecutorSettings().GetMaxNumTokens(),
            audio_backend));
    audio_executor_settings_ptr = std::make_unique<AudioExecutorSettings>(
        std::move(audio_executor_settings));
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kExecutor));
  }

  RuntimeConfig runtime_config;
  oi::proto::SamplerParameters sampler_params;
  sampler_params.set_type(oi::proto::SamplerParameters::GREEDY);
  sampler_params.set_k(1);
  sampler_params.set_temperature(0.0f);
  runtime_config.sampler_params = sampler_params;
  runtime_config.tokens_per_decode = 1;
  runtime_config.output_heads = 1;
  RETURN_IF_ERROR(executor->UpdateRuntimeConfig(runtime_config));

  ASSIGN_OR_RETURN(
      auto execution_manager,
      ExecutionManager::Create(
          tokenizer, model_resources->litert_lm_model_resources.get(),
          std::move(executor), std::move(vision_executor_settings_ptr),
          std::move(audio_executor_settings_ptr), &litert_env));

  auto llm_impl = absl::WrapUnique(new EngineAdvancedLegacyImpl(
      std::move(engine_settings), std::move(model_resources),
      std::move(execution_manager), std::move(tokenizer),
      std::move(task_tokenizer), std::move(benchmark_info)));
  return llm_impl;
};

LITERT_LM_REGISTER_ENGINE(EngineFactory::EngineType::kAdvancedLegacyTfLite,
                          [](EngineSettings settings,
                             absl::string_view input_prompt_as_hint) {
                            return EngineAdvancedLegacyImpl::Create(
                                std::move(settings), input_prompt_as_hint);
                          });
}  // namespace
}  // namespace litert::lm
