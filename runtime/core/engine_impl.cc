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

// TODO(b/417209286): Remove this once the model assets are stored in the
// litertlm file format.
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/components/lora_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/core/session_factory.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/audio_litert_compiled_model_executor.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/executor/magic_number_configs_helper.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_litert_compiled_model_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {
namespace {

// Gets the singleton Environment, initializing it on the first call
// with the provided settings. This ensure we maintain the same LiteRT
// environment during the whole application lifetime. This is required for GPU
// LiteRT environment. See b/454383477 for more details.
absl::StatusOr<Environment&> GetEnvironment(EngineSettings& engine_settings,
                                            ModelResources& model_resources) {
  // Helper must be available until LlmLiteRtCompiledModelExecutor::Create() is
  // called. Since env is used multiple times, it should also be static.
  static absl::NoDestructor<MagicNumberConfigsHelper> helper;
  static absl::NoDestructor<absl::StatusOr<Environment>> kEnvironment(
      [&]() -> absl::StatusOr<Environment> {
        std::vector<Environment::Option> env_options;
        const auto& main_executor_settings =
            engine_settings.GetMainExecutorSettings();

        if ((main_executor_settings.GetBackend() == Backend::CPU) ||
            (main_executor_settings.GetBackend() == Backend::GPU)) {
          if (!main_executor_settings
                   .GetAdvancedSettings() ||  // Default is true.
              main_executor_settings.GetAdvancedSettings()
                  ->configure_magic_numbers) {
            env_options = helper->GetLiteRtEnvOptions(model_resources,
                                                      main_executor_settings);
            // Disable madvise original shared tensors for GPU if the model has
            // magic numbers as it may revert the magic number replacements.
            if (helper->magic_number_configs() &&
                helper->magic_number_configs()->num_configs > 0) {
              auto& executor_settings =
                  engine_settings.GetMutableMainExecutorSettings();
              AdvancedSettings new_settings;
              if (executor_settings.GetAdvancedSettings()) {
                new_settings = *executor_settings.GetAdvancedSettings();
              }
              new_settings.gpu_madvise_original_shared_tensors = false;
              executor_settings.SetAdvancedSettings(new_settings);
            }
          }
        } else {
#if defined(LITERT_DISABLE_NPU)
          return absl::InvalidArgumentError(
              "Only CPU and GPU backends are supported.");
#else
          if (!main_executor_settings.GetLitertDispatchLibDir().empty()) {
            // If the dispatch library directory is provided, use it.
            env_options.push_back(::litert::Environment::Option{
                ::litert::Environment::OptionTag::DispatchLibraryDir,
                main_executor_settings.GetLitertDispatchLibDir()});
            ABSL_LOG(INFO) << "Setting dispatch library path from "
                              "main_executor_settings: "
                           << main_executor_settings.GetLitertDispatchLibDir();
          } else {
            // Otherwise, use the directory of the model file.
            std::string model_path(
                main_executor_settings.GetModelAssets().GetPath().value_or(""));
            std::filesystem::path path(model_path);
            // Note: Existence check for path was here, but it's better to check
            // before calling this function if needed.
            static const absl::NoDestructor<std::string> kDispatchLibraryPath(
                path.parent_path().string());
            if (!kDispatchLibraryPath->empty()) {
              ABSL_LOG(INFO)
                  << "Setting dispatch library path: " << *kDispatchLibraryPath;
              env_options.push_back(::litert::Environment::Option{
                  ::litert::Environment::OptionTag::DispatchLibraryDir,
                  absl::string_view(*kDispatchLibraryPath)});
            } else {
              ABSL_LOG(INFO) << "No dispatch library path provided.";
            }
          }
#endif  // defined(LITERT_DISABLE_NPU)
        }
        LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create(env_options));
        return std::move(env);
      }());
  if (!kEnvironment->ok()) {
    return kEnvironment->status();
  }
  return **kEnvironment;
}

class EngineImpl : public Engine {
 public:
  ~EngineImpl() override {
    ABSL_QCHECK_OK(WaitUntilDone(Engine::kDefaultTimeout));
  }

  static absl::StatusOr<std::unique_ptr<Engine>> Create(
      EngineSettings engine_settings, absl::string_view input_prompt_as_hint);

  EngineImpl(EngineSettings engine_settings,
             std::unique_ptr<ModelResources> litert_model_resources,
             std::unique_ptr<LlmExecutor> executor,
             std::unique_ptr<VisionExecutor> vision_executor,
             std::unique_ptr<AudioExecutor> audio_executor,
             std::optional<BenchmarkInfo> benchmark_info,
             std::unique_ptr<ThreadPool> worker_thread_pool)
      : engine_settings_(std::move(engine_settings)),
        litert_model_resources_(std::move(litert_model_resources)),
        executor_(std::move(executor)),
        vision_executor_(std::move(vision_executor)),
        audio_executor_(std::move(audio_executor)),
        stop_token_ids_(),
        sampler_params_(),
        benchmark_info_(std::move(benchmark_info)),
        worker_thread_pool_(std::move(worker_thread_pool)) {}
  // Method to create the Session.
  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) override {
    std::optional<BenchmarkInfo> session_benchmark_info;
    if (benchmark_info_.has_value()) {
      // Each session will have its own benchmark info, which will be populated
      // with the session-specific information.
      session_benchmark_info = benchmark_info_;
      RETURN_IF_ERROR(session_benchmark_info->TimeInitPhaseStart(
          BenchmarkInfo::InitPhase::kSession));
    }

    SessionConfig config = session_config;
    // TODO(b/418794726): Move this logics to be part of the SessionConfig
    // class.
    RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));

    ABSL_CHECK(litert_model_resources_ != nullptr);
    ASSIGN_OR_RETURN(auto* tokenizer, litert_model_resources_->GetTokenizer());
    ASSIGN_OR_RETURN(
        auto session,
        InitializeSessionBasic(executor_.get(), tokenizer,
                               /*vision_executor=*/vision_executor_.get(),
                               /*audio_executor=*/audio_executor_.get(), config,
                               std::move(session_benchmark_info),
                               worker_thread_pool_.get()));
    if (benchmark_info_.has_value()) {
      auto session_benchmark_info_or = session->GetMutableBenchmarkInfo();
      if (session_benchmark_info_or.ok()) {
        RETURN_IF_ERROR(session_benchmark_info_or.value()->TimeInitPhaseEnd(
            BenchmarkInfo::InitPhase::kSession));
      }
    }
    return session;
  }
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return worker_thread_pool_->WaitUntilDone(timeout);
  }

  const EngineSettings& GetEngineSettings() const override {
    return engine_settings_;
  }

  absl::Status LoadLoRA(uint32_t lora_id,
                        const std::string& lora_path) override {
    auto* lora_manager = executor_->GetLoraManager();
    if (lora_manager == nullptr) {
      return absl::FailedPreconditionError(
          "No LoRA manager available. The executor does not support LoRA.");
    }
    ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(lora_path));
    RETURN_IF_ERROR(lora_manager->LoadLoRA(lora_id, model_assets));
    return lora_manager->UseLoRA(lora_id);
  }

 private:
  // Stored engine settings.
  EngineSettings engine_settings_;
  // Model resources, which must outlive `executor_`.
  std::unique_ptr<ModelResources> litert_model_resources_;
  // Shared executor for all sessions.
  std::unique_ptr<LlmExecutor> executor_;
  // Shared vision executor for all sessions.
  std::unique_ptr<VisionExecutor> vision_executor_;
  // shared audio executor for all sessions.
  std::unique_ptr<AudioExecutor> audio_executor_;
  // Default stop token ids for all sessions loaded from the model file.
  std::vector<std::vector<int>> stop_token_ids_;
  proto::SamplerParameters sampler_params_;

  // Benchmark info for the engine.
  std::optional<BenchmarkInfo> benchmark_info_;

  // Thread pool for the engine to execute the works.
  std::unique_ptr<ThreadPool> worker_thread_pool_;
};

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> EngineImpl::Create(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
  std::optional<BenchmarkInfo> benchmark_info =
      engine_settings.IsBenchmarkEnabled()
          ? std::make_optional<BenchmarkInfo>(
                engine_settings.GetBenchmarkParams().value())
          : std::nullopt;

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kModelAssets));
  }
  const auto& model_assets =
      engine_settings.GetMutableMainExecutorSettings().GetModelAssets();
  ASSIGN_OR_RETURN(auto model_resources,
                   BuildLiteRtCompiledModelResources(model_assets));
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kModelAssets));
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kTokenizer));
  }
  ASSIGN_OR_RETURN(auto* tokenizer, model_resources->GetTokenizer());
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer));
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }
  ASSIGN_OR_RETURN(auto* llm_metadata, model_resources->GetLlmMetadata());
  // Update and load the parameters from the model file and convert the
  // tokens to ids.
  RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
      *tokenizer, llm_metadata, input_prompt_as_hint,
      model_resources->GetTFLiteModelBackendConstraint(
          ModelType::kTfLitePrefillDecode),
      model_resources->GetTFLiteModelBackendConstraint(
          ModelType::kTfLiteVisionEncoder),
      model_resources->GetTFLiteModelBackendConstraint(
          ModelType::kTfLiteAudioEncoderHw)));
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kExecutor));
  }
  std::unique_ptr<LlmExecutor> executor;
  ASSIGN_OR_RETURN(auto& env,
                   GetEnvironment(engine_settings, *model_resources));
  const auto& main_executor_settings =
      engine_settings.GetMainExecutorSettings();

  switch (main_executor_settings.GetBackend()) {
    default: {
      ASSIGN_OR_RETURN(executor,
                       CreateLlmLiteRtCompiledModelExecutor(
                           main_executor_settings, env, *model_resources));
    }
  };

  // Initialize the LoRA manager so that sessions can access trainable
  // LoRA parameters for fine-tuning (e.g. via MeZO).
  RETURN_IF_ERROR(executor->InitLoraManager());

  // TODO - b/436674053: Modularize the executor creation logic into a
  // separate executor class, and have unit test for it.
  std::unique_ptr<VisionExecutor> vision_executor;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    ASSIGN_OR_RETURN(
        vision_executor,
        VisionLiteRtCompiledModelExecutor::Create(
            engine_settings.GetMutableVisionExecutorSettings().value(), env));
  }

  std::unique_ptr<AudioExecutor> audio_executor;
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    ASSIGN_OR_RETURN(
        audio_executor,
        AudioLiteRtCompiledModelExecutor::Create(
            engine_settings.GetAudioExecutorSettings().value(), env));
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kExecutor));
  }

  // Creating the thread pool of a single thread to execute the works.
  auto worker_thread_pool =
      std::make_unique<ThreadPool>(/*name_prefix=*/"engine",
                                   /*max_num_threads=*/1);
  auto llm_impl = std::make_unique<EngineImpl>(
      std::move(engine_settings), std::move(model_resources),
      std::move(executor), std::move(vision_executor),
      std::move(audio_executor), std::move(benchmark_info),
      std::move(worker_thread_pool));

  return llm_impl;
};

LITERT_LM_REGISTER_ENGINE(EngineFactory::EngineType::kLiteRTCompiledModel,
                          [](EngineSettings settings,
                             absl::string_view input_prompt_as_hint) {
                            return EngineImpl::Create(std::move(settings),
                                                      input_prompt_as_hint);
                          });
}  // namespace
}  // namespace litert::lm
