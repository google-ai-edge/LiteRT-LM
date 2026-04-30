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

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/litert_lm_lib.h"
#include "runtime/engine/shared_flags.h"
#include "runtime/proto/litert_lm_metrics.pb.h"
#include "runtime/util/logging.h"
#include "runtime/util/metrics_util.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "cpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "", "Model path to use for LLM execution.");
ABSL_FLAG(
    bool, load_model_from_descriptor, false,
    "Whether to load the model from a file descriptor rather than by path.");
ABSL_FLAG(std::string, input_prompt, "",
          "Input prompt to use for testing LLM execution.");
ABSL_FLAG(std::string, input_prompt_file, "", "File path to the input prompt.");
ABSL_FLAG(std::string, metric_proto_file_path, "",
          "Path to the file where the benchmark metrics will be saved in "
          "protobuf format. Only collected when --benchmark is true.");
ABSL_FLAG(int, prefill_chunk_size, -1,
          "Prefill chunk size for LLM execution. A positive value enables "
          "breaking the input prefill sequence into smaller chunks for "
          "incremental processing. For example, a chunk size of 128 with an "
          "input length of 300 results in 3 chunks: 128, 128, and 44 tokens. "
          "A value of -1 disables chunking. Only supported by the dynamic "
          "executor.");
ABSL_FLAG(bool, use_session, false,
          "If true, use Session instead of Conversation to run inference. "
          "Note that session does not use Jinja templates. As such, if using "
          "Jinja in LLM Metadata, the user is responsible for manually "
          "applying the prompt template to the input prompt.");
ABSL_FLAG(int, min_log_severity, -1,
          "Minimum log severity (-1:Default Behavior, 0:Verbose, 1:Debug, "
          "2:Info, 3:Warning, 4:Error, 5:Fatal, 1000:Silent)");

namespace {

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
                       "be specified. Currently both are specified as "
                    << input_prompt << " and " << input_prompt_file;
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

// Writes the metrics to the given file path in protobuf format. Only used in
// benchmark mode when the metric file path is specified.
absl::Status WriteMetricsToFile(
    const std::vector<litert::lm::LitertLmMetrics>& metrics,
    const std::string& file_path) {
  if (metrics.empty()) {
    return absl::InvalidArgumentError("No metrics to write.");
  }

  ASSIGN_OR_RETURN(auto proto_list, litert::lm::ToProtoList(metrics));

  std::ofstream out(file_path, std::ios::out | std::ios::binary);
  if (!out) {
    return absl::InternalError(
        absl::StrCat("Failed to open metric file: ", file_path));
  }
  if (!proto_list.SerializeToOstream(&out)) {
    return absl::InternalError("Failed to serialize metrics to file.");
  }
  ABSL_LOG(INFO) << "Metrics written to: " << file_path;
  return absl::OkStatus();
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  int min_log_severity = absl::GetFlag(FLAGS_min_log_severity);
  if (min_log_severity >= 0) {
    litert::lm::SetMinLogSeverity(
        static_cast<litert::lm::LogSeverity>(min_log_severity));
  }

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
           "[--preferred_device_substr=<device_substr>]"
           "[--num_threads_to_upload=<num_threads_to_upload>]"
           "[--num_threads_to_compile=<num_threads_to_compile>]"
           "[--convert_weights_on_gpu=<true|false>]"
           "[--wait_for_weights_conversion_complete_in_benchmark=<true|false>]"
           "[--optimize_shader_compilation=<true|false>]"
           "[--share_constant_tensors=<true|false>]"
           "[--num_iterations=<num_iterations>]"
           "[--litert_dispatch_lib_dir=<litert_dispatch_lib_dir>]"
           "[--sampler_handles_input=<true|false>]"
           "[--disable_cache=<true|false>]"
           "[--cache_compiled_shader_only=<true|false>]"
           "[--conv_type=<auto|float|int8>]"
           "[--enable_speculative_decoding=<true|false>]";
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
  settings.load_model_from_descriptor =
      absl::GetFlag(FLAGS_load_model_from_descriptor);
  settings.input_prompt = GetInputPrompt();
  settings.expected_output = absl::GetFlag(FLAGS_expected_output);
  settings.log_sink_file = absl::GetFlag(FLAGS_log_sink_file);
  settings.max_num_tokens = absl::GetFlag(FLAGS_max_num_tokens);
  settings.max_output_tokens = absl::GetFlag(FLAGS_max_output_tokens);
  settings.max_num_images = absl::GetFlag(FLAGS_max_num_images);
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
  settings.cache_compiled_shaders_only =
      absl::GetFlag(FLAGS_cache_compiled_shaders_only);
  settings.preferred_device_substr =
      absl::GetFlag(FLAGS_preferred_device_substr);
  settings.num_threads_to_upload = absl::GetFlag(FLAGS_num_threads_to_upload);
  settings.num_threads_to_compile = absl::GetFlag(FLAGS_num_threads_to_compile);
  settings.convert_weights_on_gpu = absl::GetFlag(FLAGS_convert_weights_on_gpu);
  settings.wait_for_weights_conversion_complete_in_benchmark =
      absl::GetFlag(FLAGS_wait_for_weights_conversion_complete_in_benchmark);
  settings.optimize_shader_compilation =
      absl::GetFlag(FLAGS_optimize_shader_compilation);
  settings.share_constant_tensors = absl::GetFlag(FLAGS_share_constant_tensors);
  settings.use_session = absl::GetFlag(FLAGS_use_session);
  settings.num_iterations = absl::GetFlag(FLAGS_num_iterations);
  settings.litert_dispatch_lib_dir =
      absl::GetFlag(FLAGS_litert_dispatch_lib_dir);
  settings.sampler_handles_input = absl::GetFlag(FLAGS_sampler_handles_input);
  settings.conv_type =
      absl::GetFlag(FLAGS_conv_type) == "float"  ? litert::lm::ConvType::kFloat
      : absl::GetFlag(FLAGS_conv_type) == "int8" ? litert::lm::ConvType::kInt8
                                                 : litert::lm::ConvType::kAuto;
  settings.constraint_regex = absl::GetFlag(FLAGS_constraint_regex);
  settings.use_submodel = absl::GetFlag(FLAGS_use_submodel);
  settings.enable_speculative_decoding =
      absl::GetFlag(FLAGS_enable_speculative_decoding);
  settings.enable_neon_for_npu_greedy_sampling =
      absl::GetFlag(FLAGS_enable_neon_for_npu_greedy_sampling);
  settings.use_hw_masking_for_npu = absl::GetFlag(FLAGS_use_hw_masking_for_npu);
  settings.use_hw_cache_update_for_npu =
      absl::GetFlag(FLAGS_use_hw_cache_update_for_npu);
  settings.use_hw_ple_for_npu = absl::GetFlag(FLAGS_use_hw_ple_for_npu);
  settings.enable_npu_debug_logging =
      absl::GetFlag(FLAGS_enable_npu_debug_logging);

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

  std::vector<litert::lm::LitertLmMetrics> metrics;
  const std::string metric_proto_file_path =
      absl::GetFlag(FLAGS_metric_proto_file_path);
  const bool collect_metrics =
      (settings.benchmark && !metric_proto_file_path.empty());

  RETURN_IF_ERROR(
      litert::lm::RunLiteRtLm(settings, collect_metrics ? &metrics : nullptr));

  if (collect_metrics) {
    RETURN_IF_ERROR(WriteMetricsToFile(metrics, metric_proto_file_path));
  }

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
