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

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/core/session_factory.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/audio_litert_compiled_model_executor.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/executor/magic_number_configs_helper.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/executor/vision_litert_compiled_model_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/file_format_util.h"
#include "runtime/util/status_macros.h"  // NOLINT

#if !defined(LITERT_DISABLE_NPU)
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#endif  // !defined(LITERT_DISABLE_NPU)

namespace litert::lm {

class EngineImpl : public Engine {
 public:
  ~EngineImpl() override {
    ABSL_QCHECK_OK(WaitUntilDone(Engine::kDefaultTimeout));
  }
  explicit EngineImpl(EngineSettings engine_settings,
                      std::unique_ptr<ModelResources> litert_model_resources,
                      std::unique_ptr<Environment> lrt_env,
                      std::unique_ptr<LlmExecutor> executor,
                      std::unique_ptr<VisionExecutor> vision_executor,
                      std::unique_ptr<AudioExecutor> audio_executor,
                      std::optional<BenchmarkInfo> benchmark_info,
                      std::unique_ptr<ThreadPool> worker_thread_pool)
      : engine_settings_(std::move(engine_settings)),
        litert_model_resources_(std::move(litert_model_resources)),
        lrt_env_(std::move(lrt_env)),
        executor_(std::move(executor)),
        vision_executor_(std::move(vision_executor)),
        audio_executor_(std::move(audio_executor)),
        stop_token_ids_(),
        sampler_params_(),
        benchmark_info_(std::move(benchmark_info)),
        worker_thread_pool_(std::move(worker_thread_pool)) {}

  // Method to create the Session.
  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) const override {
    SessionConfig config = session_config;
    // TODO(b/418794726): Move this logics to be part of the SessionConfig
    // class.
    RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));

    ABSL_CHECK(litert_model_resources_ != nullptr);
    ASSIGN_OR_RETURN(auto* tokenizer, litert_model_resources_->GetTokenizer());
    return InitializeSession(executor_.get(), tokenizer,
                             /*vision_executor=*/vision_executor_.get(),
                             /*audio_executor=*/audio_executor_.get(), config,
                             benchmark_info_, worker_thread_pool_.get());
  }
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return worker_thread_pool_->WaitUntilDone(timeout);
  }

  const EngineSettings& GetEngineSettings() const override {
    return engine_settings_;
  }

 private:
  // Stored engine settings.
  EngineSettings engine_settings_;
  // Model resources, which must outlive `executor_`.
  std::unique_ptr<ModelResources> litert_model_resources_;
  // LiteRT environment.
  std::unique_ptr<Environment> lrt_env_;
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
absl::StatusOr<std::unique_ptr<Engine>> Engine::CreateEngine(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
  std::optional<BenchmarkInfo> benchmark_info;
  if (engine_settings.IsBenchmarkEnabled()) {
    benchmark_info = std::make_optional<BenchmarkInfo>(
        engine_settings.GetBenchmarkParams().value());
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart("Executor initialization"));
  }
  const auto& model_assets =
      engine_settings.GetMutableMainExecutorSettings().GetModelAssets();

  ASSIGN_OR_RETURN(auto model_resources,
                   BuildLiteRtCompiledModelResources(model_assets));
  ASSIGN_OR_RETURN(auto scoped_file, model_assets.GetOrCreateScopedFile());
  ASSIGN_OR_RETURN(auto file_format,
                   GetFileFormat(/*model_path=*/"", scoped_file));

  // TODO(b/397975034): factor out the tokenizer creation logic once the
  // model loading mechanism of the new file format is determined.
  if (file_format != FileFormat::TASK && file_format != FileFormat::LITERT_LM) {
    return absl::InvalidArgumentError(
        absl::StrCat("Not supported file format: ", file_format));
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart("Tokenizer initialization"));
  }
  ASSIGN_OR_RETURN(auto* tokenizer, model_resources->GetTokenizer());
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd("Tokenizer initialization"));
  }

  ASSIGN_OR_RETURN(auto* llm_metadata, model_resources->GetLlmMetadata());

  // Update and load the parameters from the model file and convert the
  // tokens to ids.
  RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
      *tokenizer, llm_metadata, input_prompt_as_hint));

  std::unique_ptr<LlmExecutor> executor;
  std::unique_ptr<Environment> lrt_env;
  if ((engine_settings.GetMainExecutorSettings().GetBackend() ==
       Backend::CPU) ||
      (engine_settings.GetMainExecutorSettings().GetBackend() ==
       Backend::GPU)) {
    // Create the LiteRT environment.
    std::vector<Environment::Option> env_options;
    MagicNumberConfigsHelper helper;
    const auto& main_executor_settings =
        engine_settings.GetMainExecutorSettings();
    if (!main_executor_settings.GetAdvancedSettings() ||  // Default is true.
        main_executor_settings.GetAdvancedSettings()->configure_magic_numbers) {
      env_options =
          helper.GetLiteRtEnvOptions(*model_resources, main_executor_settings);
    }
    LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create(env_options));
    lrt_env = std::make_unique<Environment>(std::move(env));
    ASSIGN_OR_RETURN(executor,
                     LlmLiteRtCompiledModelExecutor::Create(
                         main_executor_settings, *lrt_env, *model_resources));
  } else {
#if defined(LITERT_DISABLE_NPU)
    return absl::InvalidArgumentError(
        "Only CPU and GPU backends are supported.");
#else
    std::string model_path(engine_settings.GetMainExecutorSettings()
                               .GetModelAssets()
                               .GetPath()
                               .value_or(""));
    std::filesystem::path path(model_path);
    if (!std::filesystem::exists(path)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Model file not found: ", path.parent_path().string()));
    }
    std::vector<Environment::Option> environment_options = {};
    std::string dispatch_library_path = path.parent_path().string();
    if (!dispatch_library_path.empty()) {
      ABSL_LOG(INFO) << "Setting dispatch library path: "
                     << dispatch_library_path;
      environment_options.push_back(::litert::Environment::Option{
          ::litert::Environment::OptionTag::DispatchLibraryDir,
          absl::string_view(dispatch_library_path)});
    } else {
      ABSL_LOG(INFO) << "No dispatch library path provided.";
    }
    LITERT_ASSIGN_OR_RETURN(auto env,
                            ::litert::Environment::Create(environment_options));
    lrt_env = std::make_unique<Environment>(std::move(env));
    ASSIGN_OR_RETURN(executor, LlmLiteRtNpuCompiledModelExecutor::Create(
                                   engine_settings.GetMainExecutorSettings(),
                                   *model_resources, *lrt_env.get(),
                                   benchmark_info.has_value()));
#endif  // defined(LITERT_DISABLE_NPU)
  }

  // TODO - b/436674053: Modularize the executor creation logic into a
  // separate executor class, and have unit test for it.
  std::unique_ptr<VisionExecutor> vision_executor;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    ASSIGN_OR_RETURN(
        auto vision_executor_settings,
        VisionExecutorSettings::CreateDefault(
            engine_settings.GetMainExecutorSettings().GetModelAssets(),
            /*encoder_backend=*/
            engine_settings.GetVisionExecutorSettings()->GetBackend(),
            /*adapter_backend=*/Backend::CPU));
    ASSIGN_OR_RETURN(vision_executor, VisionLiteRtCompiledModelExecutor::Create(
                                          vision_executor_settings, *lrt_env));
  }

  std::unique_ptr<AudioExecutor> audio_executor;
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    ASSIGN_OR_RETURN(
        auto audio_executor_settings,
        AudioExecutorSettings::CreateDefault(
            engine_settings.GetMainExecutorSettings().GetModelAssets(),
            engine_settings.GetMainExecutorSettings().GetMaxNumTokens(),
            engine_settings.GetAudioExecutorSettings()->GetBackend()));
    ASSIGN_OR_RETURN(audio_executor, AudioLiteRtCompiledModelExecutor::Create(
                                         audio_executor_settings, *lrt_env));
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd("Executor initialization"));
  }

  // Creating the thread pool of a single thread to execute the works.
  auto worker_thread_pool =
      std::make_unique<ThreadPool>(/*name_prefix=*/"engine",
                                   /*max_num_threads=*/1);
  auto llm_impl = std::make_unique<EngineImpl>(
      std::move(engine_settings), std::move(model_resources),
      std::move(lrt_env), std::move(executor), std::move(vision_executor),
      std::move(audio_executor), std::move(benchmark_info),
      std::move(worker_thread_pool));

  return llm_impl;
};

}  // namespace litert::lm
