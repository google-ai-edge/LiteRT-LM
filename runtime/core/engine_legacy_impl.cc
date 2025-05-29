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

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/executor/google_tensor_single_graph/llm_litert_google_tensor_single_graph_executor.h"
#include "third_party/odml/infra/genai/inference/executor/litert_executor_utils.h"
#include "third_party/odml/infra/genai/inference/executor/llm_litert_opencl_executor.h"
#include "third_party/odml/infra/genai/inference/executor/llm_litert_xnnpack_executor.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_factory.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/framework/thread_options.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "util/task/status_macros.h"

namespace litert::lm {
namespace {

namespace oi = ::odml::infra;

absl::StatusOr<std::unique_ptr<LlmExecutor>> BuildExecutor(
    const std::unique_ptr<::odml::infra::ExecutorModelResources>&
        model_resources,
    const EngineSettings& engine_settings) {
  if (!(model_resources && model_resources->model)) {
    return absl::InternalError("Failed to build TF_LITE_PREFILL_DECODE model.");
  }
  // Create executor that creates and owns the interpreter and kv cache.
  std::unique_ptr<LlmExecutor> executor;
  ABSL_LOG(INFO) << "Executor settings: "
                 << engine_settings.GetMainExecutorSettings();

  if (engine_settings.GetMainExecutorSettings().GetBackend() == Backend::CPU) {
    ASSIGN_OR_RETURN(executor, oi::LlmLiteRTXnnpackExecutor::Create(
                                   engine_settings.GetMainExecutorSettings(),
                                   *model_resources));
  } else if (engine_settings.GetMainExecutorSettings().GetBackend() ==
             Backend::GPU) {
    ASSIGN_OR_RETURN(executor, oi::LlmLiteRTOpenClExecutor::Create(
                                   engine_settings.GetMainExecutorSettings(),
                                   *model_resources));
  } else if (engine_settings.GetMainExecutorSettings().GetBackend() ==
             Backend::GOOGLE_TENSOR) {
    ASSIGN_OR_RETURN(executor,
                     oi::LlmLiteRTGoogleTensorSingleGraphExecutor::Create(
                         engine_settings.GetMainExecutorSettings(),
                         *model_resources->model));
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

  explicit EngineImpl(const EngineSettings& engine_settings)
      : engine_settings_(engine_settings) {
    ABSL_LOG(INFO) << "Constructing legacy EngineImpl...";
    if (engine_settings_.IsBenchmarkEnabled()) {
      benchmark_info_ = std::make_optional<BenchmarkInfo>(
          engine_settings_.GetBenchmarkParams().value());
      ABSL_CHECK_OK(
          benchmark_info_->TimeInitPhaseStart("Executor initialization"));
    }
    auto model_path =
        engine_settings_.GetMainExecutorSettings().GetModelAssets().GetPath();
    ABSL_CHECK_OK(model_path);
    auto model_resources = oi::BuildModelResources(std::string(*model_path));
    ABSL_QCHECK_OK(model_resources);
    model_resources_ = std::move(*model_resources);

    // TODO(b/397975034): factor out the tokenizer creation logic once the model
    // loading mechanism of the new file format is determined.
    auto scoped_file = ScopedFile::Open(*model_path);
    ABSL_CHECK_OK(scoped_file);
    auto resources = ModelAssetBundleResources::Create(
        /*tag=*/"", *std::move(scoped_file));
    auto vocab_buffer = (*resources)->GetFile("TOKENIZER_MODEL");
    tokenizer_ =
        std::move(*SentencePieceTokenizer::CreateFromBuffer(*vocab_buffer));
    if (benchmark_info_.has_value()) {
      ABSL_CHECK_OK(
          benchmark_info_->TimeInitPhaseEnd("Tokenizer initialization"));
    }
    // Update and load the parameters from the model file and convert the tokens
    // to ids.
    ABSL_CHECK_OK(engine_settings_.MaybeUpdateAndValidate(tokenizer_));

    auto executor = BuildExecutor(model_resources_, engine_settings_);
    ABSL_QCHECK_OK(executor);
    executor_ = std::move(*executor);
    if (benchmark_info_.has_value()) {
      ABSL_CHECK_OK(
          benchmark_info_->TimeInitPhaseEnd("Executor initialization"));
      ABSL_CHECK_OK(
          benchmark_info_->TimeInitPhaseStart("Tokenizer initialization"));
    }

    RuntimeConfig runtime_config;
    oi::proto::SamplerParameters sampler_params;
    sampler_params.set_type(oi::proto::SamplerParameters::GREEDY);
    sampler_params.set_k(1);
    sampler_params.set_temperature(0.0f);
    runtime_config.sampler_params = sampler_params;
    runtime_config.tokens_per_decode = 1;
    runtime_config.output_heads = 1;
    ABSL_QCHECK_OK(executor_->UpdateRuntimeConfig(runtime_config));

    // Creating the thread pool of a single thread to execute the works.
    worker_thread_pool_ = std::make_shared<ThreadPool>(
        ThreadOptions(), /*name_prefix=*/"engine", /*num_threads=*/1);
    worker_thread_pool_->StartWorkers();
  }

  // Method to create the Session.
  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) const override {
    auto config = session_config;
    RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));
    // For the TfLite executors, we use the built-in sampling logic instead of
    // the sampler component. Setting the type to unspecified to disable the
    // sampler component.
    config.GetMutableSamplerParams().set_type(
        proto::SamplerParameters::TYPE_UNSPECIFIED);
    return InitializeSession(executor_, tokenizer_, config, benchmark_info_,
                             worker_thread_pool_);
  }

  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return worker_thread_pool_->WaitUntilDone(timeout);
  }

 private:
  // Stored engine settings.
  EngineSettings engine_settings_;
  // Shared executor for all sessions.
  std::shared_ptr<LlmExecutor> executor_;
  // Shared tokenizer for all sessions.
  std::shared_ptr<Tokenizer> tokenizer_;
  // Default stop token ids for all sessions loaded from the model file.
  std::vector<std::vector<int>> stop_token_ids_;

  std::unique_ptr<oi::ExecutorModelResources> model_resources_;

  // Benchmark info for the engine.
  std::optional<BenchmarkInfo> benchmark_info_;

  // Thread pool for the engine to execute the works.
  std::shared_ptr<ThreadPool> worker_thread_pool_;
};

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> Engine::CreateEngine(
    const EngineSettings& settings_struct) {
  auto llm_impl = std::make_unique<EngineImpl>(settings_struct);
  return llm_impl;
};

}  // namespace litert::lm
