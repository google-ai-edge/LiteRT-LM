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

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/c/litert_logging.h"  // from @litert
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "tflite/profiling/memory_usage_monitor.h"  // from @litert

ABSL_FLAG(std::optional<std::string>, backend, "gpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model, "", "Model file to use for LLM execution.");
ABSL_FLAG(std::string, input_prompt,
          "What is the tallest building in the world?",
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
ABSL_FLAG(bool, async, true, "Run the LLM execution asynchronously.");
ABSL_FLAG(bool, report_peak_memory_footprint, false,
          "Report peak memory footprint.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::EngineSettings;
using ::litert::lm::InferenceObservable;
using ::litert::lm::InputText;
using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::ModelAssets;

// Memory check interval in milliseconds.
constexpr int kMemoryCheckIntervalMs = 50;
// Timeout duration for waiting until the engine is done with all the tasks.
const absl::Duration kWaitUntilDoneTimeout = absl::Minutes(10);

// Converts an absl::LogSeverityAtLeast to a LiteRtLogSeverity.
LiteRtLogSeverity AbslMinLogLevelToLiteRtLogSeverity(
    absl::LogSeverityAtLeast min_log_level) {
  int min_log_level_int = static_cast<int>(min_log_level);
  switch (min_log_level_int) {
    case -1:
      // ABSL does not support verbose logging, but passes through -1 as a log
      // level, which we can use to enable verbose logging in LiteRT.
      return LITERT_VERBOSE;
    case static_cast<int>(absl::LogSeverityAtLeast::kInfo):
      return LITERT_INFO;
    case static_cast<int>(absl::LogSeverityAtLeast::kWarning):
      return LITERT_WARNING;
    case static_cast<int>(absl::LogSeverityAtLeast::kError):
      return LITERT_ERROR;
    case static_cast<int>(absl::LogSeverityAtLeast::kFatal):
      return LITERT_SILENT;
    default:
      return LITERT_INFO;
  }
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  LiteRtSetMinLoggerSeverity(
      LiteRtGetDefaultLogger(),
      AbslMinLogLevelToLiteRtLogSeverity(absl::MinLogLevel()));

  const std::string model_path = absl::GetFlag(FLAGS_model);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  std::unique_ptr<tflite::profiling::memory::MemoryUsageMonitor> mem_monitor;
  if (absl::GetFlag(FLAGS_report_peak_memory_footprint)) {
    mem_monitor =
        std::make_unique<tflite::profiling::memory::MemoryUsageMonitor>(
            kMemoryCheckIntervalMs);
    mem_monitor->Start();
  }
  ABSL_LOG(INFO) << "Model path: " << model_path;
  ASSIGN_OR_RETURN(ModelAssets model_assets,  // NOLINT
                   ModelAssets::Create(model_path));
  std::string backend_str = absl::GetFlag(FLAGS_backend).value();
  ABSL_LOG(INFO) << "Choose backend: " << backend_str;
  ASSIGN_OR_RETURN(Backend backend,
                   litert::lm::GetBackendFromString(backend_str));
  ASSIGN_OR_RETURN(EngineSettings engine_settings,
                   EngineSettings::CreateDefault(model_assets, backend));
  auto session_config = litert::lm::SessionConfig::CreateDefault();
  ABSL_LOG(INFO) << "executor_settings: "
                 << engine_settings.GetMainExecutorSettings();

  if (absl::GetFlag(FLAGS_benchmark)) {
    litert::lm::proto::BenchmarkParams benchmark_params;
    benchmark_params.set_num_prefill_tokens(
        absl::GetFlag(FLAGS_benchmark_prefill_tokens));
    benchmark_params.set_num_decode_tokens(
        absl::GetFlag(FLAGS_benchmark_decode_tokens));
    engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  }
  ABSL_LOG(INFO) << "Creating engine";
  absl::StatusOr<std::unique_ptr<litert::lm::Engine>> llm =
      litert::lm::Engine::CreateEngine(engine_settings);
  ABSL_CHECK_OK(llm) << "Failed to create engine";

  ABSL_LOG(INFO) << "Creating session";
  absl::StatusOr<std::unique_ptr<litert::lm::Engine::Session>> session =
      (*llm)->CreateSession(session_config);
  ABSL_CHECK_OK(session) << "Failed to create session";

  // When either prefill or decode tokens is set, the input prompt will be
  // forced to be the specified value and generate a dummy input.
  const bool is_dummy_input =
      absl::GetFlag(FLAGS_benchmark_prefill_tokens) > 0 ||
      absl::GetFlag(FLAGS_benchmark_decode_tokens) > 0;
  const std::string input_prompt = absl::GetFlag(FLAGS_input_prompt);
  if (absl::GetFlag(FLAGS_async)) {
    if (is_dummy_input) {
      ABSL_LOG(FATAL)
          << "Async mode does not support benchmarking with "
             "specified number of prefill or decode tokens. If you want to "
             "benchmark the model, please try again with --async=false.";
    }
    InferenceObservable observable;
    absl::Status status = (*session)->GenerateContentStream(
        {InputText(input_prompt)}, &observable);
    ABSL_CHECK_OK(status);
    ABSL_CHECK_OK((*llm)->WaitUntilDone(kWaitUntilDoneTimeout));
  } else {
    auto responses = (*session)->GenerateContent({InputText(input_prompt)});
    ABSL_CHECK_OK(responses);
    if (!is_dummy_input) {
      ABSL_LOG(INFO) << "Responses: " << *responses;
    }
  }

  if (absl::GetFlag(FLAGS_benchmark)) {
    auto benchmark_info = (*session)->GetBenchmarkInfo();
    ABSL_LOG(INFO) << *benchmark_info;
  }

  if (absl::GetFlag(FLAGS_report_peak_memory_footprint)) {
    float peak_mem_mb = 0.0f;
    if (mem_monitor != nullptr) {
      mem_monitor->Stop();
      peak_mem_mb = mem_monitor->GetPeakMemUsageInMB();
    }
    ABSL_LOG(INFO) << "Peak system ram usage: " << peak_mem_mb << "MB.";
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
