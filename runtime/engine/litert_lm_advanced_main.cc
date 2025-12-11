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

#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"  // from @litert
#include "runtime/engine/litert_lm_lib.h"
#include "runtime/engine/shared_flags.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "gpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "", "Model path to use for LLM execution.");
ABSL_FLAG(std::string, input_prompt,
          "What is the tallest building in the world?",
          "Input prompt to use for testing LLM execution.");
ABSL_FLAG(std::string, input_prompt_file, "", "File path to the input prompt.");
ABSL_FLAG(int, prefill_chunk_size, -1,
          "Prefill chunk size for LLM execution. A positive value enables "
          "breaking the input prefill sequence into smaller chunks for "
          "incremental processing. For example, a chunk size of 128 with an "
          "input length of 300 results in 3 chunks: 128, 128, and 44 tokens. "
          "A value of -1 disables chunking. Only supported by the dynamic "
          "executor.");

namespace {

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

absl::StatusOr<std::set<int>> ParsePrefillBatchSizes(
    const std::vector<std::string>& prefill_batch_sizes) {
  std::set<int> parsed_prefill_batch_sizes;
  for (const auto& prefill_batch_size : prefill_batch_sizes) {
    int size;
    if (!absl::SimpleAtoi(prefill_batch_size, &size)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid prefill batch size: ", prefill_batch_size));
    }
    parsed_prefill_batch_sizes.insert(size);
  }
  return parsed_prefill_batch_sizes;
}

std::string GetInputPrompt() {
  const std::string input_prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string input_prompt_file = absl::GetFlag(FLAGS_input_prompt_file);
  if (!input_prompt.empty() && !input_prompt_file.empty()) {
    ABSL_LOG(FATAL) << "Only one of --input_prompt and --input_prompt_file can "
                       "be specified.";
  }
  if (!input_prompt.empty()) {
    return input_prompt;
  }
  if (!input_prompt_file.empty()) {
    std::ifstream file(input_prompt_file);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << input_prompt_file
                << std::endl;
      return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }
  // If no input prompt is provided, use the default prompt.
  return "What is the tallest building in the world?";
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  LiteRtSetMinLoggerSeverity(
      LiteRtGetDefaultLogger(),
      AbslMinLogLevelToLiteRtLogSeverity(absl::MinLogLevel()));

  if (argc <= 1) {
    ABSL_LOG(INFO)
        << "Example usage: ./litert_lm_main --model_path=<model_path> "
           "[--input_prompt=<input_prompt>] "
           "[--input_prompt_file=<input_prompt_file>] "
           "[--expected_output=<expected_output>] [--backend=<cpu|gpu|npu>] "
           "[--log_sink_file=<log_sink_file>] "
           "[--max_num_tokens=<max_num_tokens>] "
           "[--prefill_batch_sizes=<size1>[,<size2>,...]]"
           "[--prefill_chunk_size=<prefill_chunk_size>] "
           "[--vision_backend=<cpu|gpu>] [--audio_backend=<cpu|gpu>] "
           "[--sampler_backend=<cpu|gpu>] [--benchmark] "
           "[--benchmark_prefill_tokens=<num_prefill_tokens>] "
           "[--benchmark_decode_tokens=<num_decode_tokens>] "
           "[--async=<true|false>] [--force_f32=<true|false] "
           "[--report_peak_memory_footprint] [--multi_turns=<true|false>] "
           "[--num_cpu_threads=<num_cpu_threads>] "
           "[--gpu_external_tensor_mode=<true|false>] "
           "[--configure_magic_numbers=<true|false>] "
           "[--verify_magic_numbers=<true|false>] "
           "[--clear_kv_cache_before_prefill=<true|false>] "
           "[--num_logits_to_print_after_decode=<num_logits_to_print>]"
           "[--score_target_text=<target_text>]"
           "[--gpu_madvise_original_shared_tensors=<true|false>]"
           "[--preferred_device_substr=<device_substr>]";
    ABSL_LOG(INFO)
        << "To provide data for multimodality, use [image:/path/to/image.jpg] "
           "or [audio:/path/to/audio.wav] in the input prompt. e.g. \"Describe "
           "the image: [image:/path/to/image.jpg]\", or \"Transcribe the audio "
           "[audio:/path/to/audio.wav]\"";
    return absl::InvalidArgumentError("No arguments provided.");
  }

  litert::lm::LiteRtLmSettings settings;

  settings.backend = absl::GetFlag(FLAGS_backend);
  settings.vision_backend = absl::GetFlag(FLAGS_vision_backend);
  settings.audio_backend = absl::GetFlag(FLAGS_audio_backend);
  settings.sampler_backend = absl::GetFlag(FLAGS_sampler_backend);
  settings.model_path = absl::GetFlag(FLAGS_model_path);
  settings.input_prompt = GetInputPrompt();
  settings.expected_output = absl::GetFlag(FLAGS_expected_output);
  settings.log_sink_file = absl::GetFlag(FLAGS_log_sink_file);
  settings.max_num_tokens = absl::GetFlag(FLAGS_max_num_tokens);
  ASSIGN_OR_RETURN(
      settings.prefill_batch_sizes,
      ParsePrefillBatchSizes(absl::GetFlag(FLAGS_prefill_batch_sizes)));
  settings.prefill_chunk_size = absl::GetFlag(FLAGS_prefill_chunk_size);
  settings.num_output_candidates = absl::GetFlag(FLAGS_num_output_candidates);
  settings.benchmark = absl::GetFlag(FLAGS_benchmark);
  settings.benchmark_prefill_tokens =
      absl::GetFlag(FLAGS_benchmark_prefill_tokens);
  settings.benchmark_decode_tokens =
      absl::GetFlag(FLAGS_benchmark_decode_tokens);
  settings.async = absl::GetFlag(FLAGS_async);
  settings.report_peak_memory_footprint =
      absl::GetFlag(FLAGS_report_peak_memory_footprint);
  settings.force_f32 = absl::GetFlag(FLAGS_force_f32);
  settings.multi_turns = absl::GetFlag(FLAGS_multi_turns);
  settings.num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
  settings.gpu_external_tensor_mode =
      absl::GetFlag(FLAGS_gpu_external_tensor_mode);
  settings.configure_magic_numbers =
      absl::GetFlag(FLAGS_configure_magic_numbers);
  settings.verify_magic_numbers = absl::GetFlag(FLAGS_verify_magic_numbers);
  settings.clear_kv_cache_before_prefill =
      absl::GetFlag(FLAGS_clear_kv_cache_before_prefill);
  settings.num_logits_to_print_after_decode =
      absl::GetFlag(FLAGS_num_logits_to_print_after_decode);
  settings.score_target_text = absl::GetFlag(FLAGS_score_target_text);
  settings.gpu_madvise_original_shared_tensors =
      absl::GetFlag(FLAGS_gpu_madvise_original_shared_tensors);
  settings.disable_cache = absl::GetFlag(FLAGS_disable_cache);
  settings.preferred_device_substr =
      absl::GetFlag(FLAGS_preferred_device_substr);

  // Adjust max_num_tokens and prefill_batch_size if not set on benchmark mode.
  if (settings.benchmark && settings.benchmark_prefill_tokens > 0) {
    if (settings.max_num_tokens == 0 && settings.benchmark_decode_tokens > 0) {
      settings.max_num_tokens =
          settings.benchmark_prefill_tokens + settings.benchmark_decode_tokens;
    }
    if (settings.prefill_batch_sizes.empty()) {
      settings.prefill_batch_sizes.insert(settings.benchmark_prefill_tokens);
    }
  }

  return litert::lm::RunLiteRtLm(settings);
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
