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
#include <future>      // NOLINT(build/c++11)
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
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
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
    auto status = WaitUntilDone(Engine::kDefaultTimeout);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to wait for engine to finish: " << status;
    }
  }

  static absl::StatusOr<std::unique_ptr<Engine>> Create(
      EngineSettings engine_settings, absl::string_view input_prompt_as_hint);

  EngineImpl(EngineSettings engine_settings,
             std::unique_ptr<ModelResources> litert_model_resources,
             std::unique_ptr<Tokenizer> tokenizer,
             std::unique_ptr<LlmExecutor> executor,
             std::unique_ptr<VisionExecutor> vision_executor,
             std::unique_ptr<AudioExecutor> audio_executor,
             std::optional<BenchmarkInfo> benchmark_info,
             std::unique_ptr<ThreadPool> worker_thread_pool)
      : engine_settings_(std::move(engine_settings)),
        litert_model_resources_(std::move(litert_model_resources)),
        tokenizer_(std::move(tokenizer)),
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

    if (litert_model_resources_ == nullptr) {
      return absl::FailedPreconditionError(
          "Model resources are not initialized.");
    }
    ASSIGN_OR_RETURN(
        auto session,
        InitializeSessionBasic(executor_.get(), tokenizer_.get(),
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

  const Tokenizer& GetTokenizer() const override { return *tokenizer_; }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    if (audio_executor_ == nullptr) {
      return absl::FailedPreconditionError("Audio modality is not enabled.");
    }
    return audio_executor_->GetAudioExecutorProperties();
  }

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override {
    if (vision_executor_ == nullptr) {
      return absl::FailedPreconditionError("Vision modality is not enabled.");
    }
    return vision_executor_->GetVisionExecutorProperties();
  }

 private:
  // Stored engine settings.
  EngineSettings engine_settings_;
  // Model resources, which must outlive `executor_`.
  std::unique_ptr<ModelResources> litert_model_resources_;
  // Tokenizer shared by all sessions.
  std::unique_ptr<Tokenizer> tokenizer_;
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
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
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
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }
  ASSIGN_OR_RETURN(auto* llm_metadata, model_resources->GetLlmMetadata());
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }
  bool hasLlmModelType = llm_metadata->has_llm_model_type();
  absl::Duration tokenizer_duration = absl::ZeroDuration();
  // This lambda is used to create the tokenizer asynchronously if the model
  // type is available, such that the tokenizer can be created in parallel with
  // the executor.
  auto create_tokenizer =
      [&tokenizer_duration,
       &model_resources]() -> absl::StatusOr<std::unique_ptr<Tokenizer>> {
    absl::Time start_time = absl::Now();
    ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                     model_resources->GetTokenizer());
    tokenizer_duration = absl::Now() - start_time;
    return tokenizer;
  };

  const auto& main_executor_settings =
      engine_settings.GetMainExecutorSettings();

  std::future<absl::StatusOr<std::unique_ptr<Tokenizer>>> tokenizer_future;
  std::unique_ptr<Tokenizer> tokenizer;
  if (!hasLlmModelType) {
    ABSL_LOG(INFO)
        << "Legacy model files don't have LlmModelType, loading tokenizer now";
    ASSIGN_OR_RETURN(tokenizer, create_tokenizer());
    // Update and load the parameters from the model file and convert the
    // tokens to ids.
    RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        tokenizer.get(), llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw)));
  } else {
    // If the model type is available, wait for the tokenizer to be created
    // after the model is loaded.
    ABSL_LOG(INFO) << "New model files have LlmModelType, loading tokenizer "
                      "asynchronously";

    if (engine_settings.GetParallelFileSectionLoading()) {
      tokenizer_future = std::async(std::launch::async, create_tokenizer);
    } else {
      tokenizer_future = std::async(std::launch::deferred, create_tokenizer);
    }

    RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        nullptr, llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw)));
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kExecutor));
  }
  std::unique_ptr<LlmExecutor> executor;
  ASSIGN_OR_RETURN(auto& env,
                   GetEnvironment(engine_settings, *model_resources));

  switch (main_executor_settings.GetBackend()) {
    default: {
      ASSIGN_OR_RETURN(executor,
                       CreateLlmLiteRtCompiledModelExecutor(
                           main_executor_settings, env, *model_resources));
    }
  };

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

  if (hasLlmModelType) {
    // Now load the tokenizer and update the engine settings.
    ASSIGN_OR_RETURN(tokenizer, tokenizer_future.get());
    RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        tokenizer.get(), llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw)));
    // As we load the tokenizer asynchronously, we need to update the executor
    // settings after the tokenizer is loaded.
    RETURN_IF_ERROR(executor->UpdateExecutorSettings(
        engine_settings.GetMainExecutorSettings()));
  }
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->InitPhaseRecord(
        BenchmarkInfo::InitPhase::kTokenizer, tokenizer_duration));
  }

  // Creating the thread pool of a single thread to execute the works.
  auto worker_thread_pool =
      std::make_unique<ThreadPool>(/*name_prefix=*/"engine",
                                   /*max_num_threads=*/1);

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTotal));
  }

  auto llm_impl = std::make_unique<EngineImpl>(
      std::move(engine_settings), std::move(model_resources),
      std::move(tokenizer), std::move(executor), std::move(vision_executor),
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
