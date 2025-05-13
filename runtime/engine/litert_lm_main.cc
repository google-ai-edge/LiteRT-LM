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

// ODML pipeline to execute or benchmark LLM graph on device.
//
// The pipeline does the following
// 1) Read the corresponding parameters, weight and model file paths.
// 2) Construct a graph model with the setting.
// 3) Execute model inference and generate the output.
//
// Consider run_llm_inference_engine.sh as an example to run on android device.

#include <memory>
#include <optional>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_settings.h"

ABSL_FLAG(std::optional<std::string>, backend, "gpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "", "Model path to use for LLM execution.");
ABSL_FLAG(std::string, input_prompt, "What is the highest building in Paris?",
          "Input prompt to use for testing LLM execution.");
ABSL_FLAG(bool, benchmark, false, "Benchmark the LLM execution.");
ABSL_FLAG(
    int, benchmark_prefill_tokens, 0,
    "If benchmark is true and the value is larger than 0, the benchmark will "
    "use this number to set the number of prefill tokens (regardless of the "
    "input prompt).");
ABSL_FLAG(int, benchmark_decode_tokens, 0,
          "If benchmark is true and the value is larger than 0, the benchmark "
          "will use this number to set the number of decode steps (regardless "
          "of the input prompt).");
ABSL_FLAG(bool, async, false, "Run the LLM execution asynchronously.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::CpuConfig;
using ::litert::lm::EngineSettings;
using ::litert::lm::GpuConfig;
using ::litert::lm::InferenceObservable;
using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::ModelAssets;

// Timeout duration for waiting until the engine is done with all the tasks.
const absl::Duration kWaitUntilDoneTimeout = absl::Minutes(10);

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  ABSL_LOG(INFO) << "Model path: " << model_path;
  ModelAssets model_assets;
  model_assets.model_paths.push_back(model_path);
  LlmExecutorSettings executor_settings(model_assets);

  std::string backend_str = absl::GetFlag(FLAGS_backend).value();
  ABSL_LOG(INFO) << "Choose backend: " << backend_str;
  Backend backend;
  if (backend_str == "cpu") {
    backend = Backend::CPU;
    CpuConfig config;
    config.number_of_threads = 4;
    executor_settings.SetBackendConfig(config);
  } else if (backend_str == "gpu") {
    backend = Backend::GPU;
    GpuConfig config;
    config.max_top_k = 1;
    executor_settings.SetBackendConfig(config);
  } else {
    return absl::InvalidArgumentError("Unsupported backend: " + backend_str);
  }
  executor_settings.SetBackend(backend);
  // TODO(b/397975034) Set the max num tokens based on the model.
  executor_settings.SetMaxNumTokens(160);
  ABSL_LOG(INFO) << "executor_settings: " << executor_settings;
  EngineSettings model_settings(executor_settings);

  if (absl::GetFlag(FLAGS_benchmark)) {
    litert::lm::proto::BenchmarkParams benchmark_params;
    benchmark_params.set_num_prefill_tokens(
        absl::GetFlag(FLAGS_benchmark_prefill_tokens));
    benchmark_params.set_num_decode_tokens(
        absl::GetFlag(FLAGS_benchmark_decode_tokens));
    model_settings.SetBenchmarkParams(benchmark_params);
  }
  ABSL_LOG(INFO) << "Creating engine";
  absl::StatusOr<std::unique_ptr<litert::lm::Engine>> llm =
      litert::lm::Engine::CreateEngine(model_settings);
  ABSL_CHECK_OK(llm) << "Failed to create engine";

  ABSL_LOG(INFO) << "Creating session";
  absl::StatusOr<std::unique_ptr<litert::lm::Engine::Session>> session =
      (*llm)->CreateSession(litert::lm::SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session) << "Failed to create session";

  const std::string input_prompt = absl::GetFlag(FLAGS_input_prompt);
  if (absl::GetFlag(FLAGS_async)) {
    InferenceObservable observable;
    absl::Status status =
        (*session)->RunPrefillAsync(input_prompt, &observable);
    ABSL_CHECK_OK(status);
    status = (*session)->RunDecodeAsync(&observable);
    ABSL_CHECK_OK(status);
    ABSL_CHECK_OK((*llm)->WaitUntilDone(kWaitUntilDoneTimeout));
  } else {
    absl::Status status = (*session)->RunPrefill(input_prompt);
    auto responses = (*session)->RunDecode();
    ABSL_CHECK_OK(responses);
    ABSL_LOG(INFO) << "Responses: " << *responses;
  }

  if (absl::GetFlag(FLAGS_benchmark)) {
    auto benchmark_info = (*session)->GetBenchmarkInfo();
    ABSL_LOG(INFO) << *benchmark_info;
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
