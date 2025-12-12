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

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
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
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor_settings.h"
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
namespace {

namespace oi = ::odml::infra;

absl::StatusOr<std::unique_ptr<LlmExecutor>> BuildExecutor(
    const oi::ExecutorModelResources& model_resources,
    const EngineSettings& engine_settings) {
  if ((engine_settings.GetMainExecutorSettings().GetBackend() !=
       Backend::GPU_ARTISAN) &&
      (!model_resources.model)) {
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
    ASSIGN_OR_RETURN(executor,
                     oi::LlmGpuArtisanExecutor::Create(
                         engine_settings.GetMainExecutorSettings(),
                         *(model_resources.litert_lm_model_resources.get())));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend: ",
                     engine_settings.GetMainExecutorSettings().GetBackend()));
  }

  return std::move(executor);
}

}  // namespace

class EngineImpl : public Engine {
 public:
  ~EngineImpl() override {
    ABSL_QCHECK_OK(WaitUntilDone(Engine::kDefaultTimeout));
  }

  // Constructor for EngineImpl.
  //
  // Arguments:
  //   engine_settings: The settings for the engine.
  //   model_resources: The model resources, which must outlive the executor.
  //   execution_manager: The execution manager for all sessions.
  //   tokenizer: The tokenizer from the task file.
  //   task_tokenizer: The task tokenizer, this is needed for .task and .tflite
  //     tokenizer.
  //     TODO(b/466112373): Remove this argument.
  //   benchmark_info: The benchmark info for the engine.
  explicit EngineImpl(
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
  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) const override {
    auto config = session_config;
    RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));
    return InitializeSessionAdvanced(execution_manager_, tokenizer_, config,
                                     benchmark_info_);
  }

  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return execution_manager_->WaitUntilAllDone(timeout);
  }

  const EngineSettings& GetEngineSettings() const override {
    return engine_settings_;
  }

 private:
  // Stored engine settings.
  EngineSettings engine_settings_;

  // Model resources, which must outlive `executor_`.
  std::unique_ptr<oi::ExecutorModelResources> model_resources_;

  // Execution manager for all sessions.
  std::shared_ptr<ExecutionManager> execution_manager_;

  // Tokenizer from task file, that is not owned by the model resources.
  // So we keep it here to avoid the model resources being destroyed.
  Tokenizer* tokenizer_;

  // Task tokenizer, this is needed for .task file's tokenizer.
  std::unique_ptr<Tokenizer> task_tokenizer_;

  // Benchmark info for the engine.
  std::optional<BenchmarkInfo> benchmark_info_;
};

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> Engine::CreateEngine(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
  ABSL_LOG(INFO) << "Constructing legacy EngineImpl...";
  std::optional<BenchmarkInfo> benchmark_info;
  if (engine_settings.IsBenchmarkEnabled()) {
    benchmark_info = std::make_optional<BenchmarkInfo>(
        engine_settings.GetBenchmarkParams().value());
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart("Executor initialization"));
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
    // Handle the .task or .tflite file format.
    ASSIGN_OR_RETURN(auto resources, ModelAssetBundleResources::Create(
                                         /*tag=*/"", scoped_model_file));
    if (benchmark_info.has_value()) {
      RETURN_IF_ERROR(
          benchmark_info->TimeInitPhaseStart("Tokenizer initialization"));
    }
    ASSIGN_OR_RETURN(auto vocab_buffer, resources->GetFile("TOKENIZER_MODEL"));
    ASSIGN_OR_RETURN(task_tokenizer,
                     SentencePieceTokenizer::CreateFromBuffer(vocab_buffer));
    tokenizer = task_tokenizer.get();
    if (benchmark_info.has_value()) {
      RETURN_IF_ERROR(
          benchmark_info->TimeInitPhaseEnd("Tokenizer initialization"));
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
  RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
      *tokenizer, &llm_metadata, input_prompt_as_hint));

  ASSIGN_OR_RETURN(auto executor,
                   BuildExecutor(*model_resources, engine_settings));

  LITERT_ASSIGN_OR_RETURN(auto litert_env, Environment::Create({}));

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
        benchmark_info->TimeInitPhaseEnd("Executor initialization"));
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

  ASSIGN_OR_RETURN(auto execution_manager,
                   ExecutionManager::Create(
                       tokenizer, std::move(executor),
                       std::move(vision_executor_settings_ptr),
                       std::move(audio_executor_settings_ptr),
                       std::make_unique<Environment>(std::move(litert_env))));

  auto llm_impl = std::make_unique<EngineImpl>(
      std::move(engine_settings), std::move(model_resources),
      std::move(execution_manager), std::move(tokenizer),
      std::move(task_tokenizer), std::move(benchmark_info));
  return llm_impl;
};

}  // namespace litert::lm
