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

#include "runtime/engine/shared_flags.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl

ABSL_FLAG(std::optional<std::string>, vision_backend, std::nullopt,
          "Backend to use for the vision model (cpu or gpu). If not specified, "
          "the vision backend will be chosen based on the main backend.");
ABSL_FLAG(std::optional<std::string>, audio_backend, std::nullopt,
          "Backend to use for the audio model (cpu or gpu). If not specified, "
          "the audio backend will be chosen based on the main backend.");
ABSL_FLAG(std::string, sampler_backend, "",
          "Sampler backend to use for LLM execution (cpu, gpu, etc.). If "
          "empty, the sampler backend will be chosen for the best according to "
          "the main executor, for example, gpu for gpu main executor.");
ABSL_FLAG(std::string, expected_output, "",
          "If not empty, the output will be checked against this string. If "
          "the output does not contain the string, the program will exit with "
          "an error.");
ABSL_FLAG(std::optional<std::string>, log_sink_file, std::nullopt,
          "If specified, the logs will be written to this file.");
ABSL_FLAG(int, max_num_tokens, 0,
          "Maximum number of tokens or context length to use for LLM execution "
          "of a graph with dynamic context length. If 0, the maximum context "
          "length will be determined by some heuristic. On benchmark mode, it "
          "will be set to one equal to or greater than "
          "benchmark_prefill_tokens + benchmark_decode_tokens.");
ABSL_FLAG(int, max_num_images, 1,
          "Maximum number of images to use for LLM execution.");
ABSL_FLAG(std::vector<std::string>, prefill_batch_sizes, {},
          "A list of maximum numbers of prefill tokens processed at once. If "
          "empty, it will be the list of one entry with the length of input "
          "prompt tokens or benchmark_prefill_tokens when benchmark mode is "
          "enabled.");
ABSL_FLAG(int, num_output_candidates, 1,
          "The number of candidates generated for the given prompt, or the "
          "batch size of the decode signature.");
ABSL_FLAG(bool, benchmark, false, "Benchmark the LLM execution.");
ABSL_FLAG(int, benchmark_prefill_tokens, 0,
          "If benchmark is true and the value is larger than 0, the benchmark "
          "will use this number to set the number of prefill tokens "
          "(regardless of the input prompt).");
ABSL_FLAG(int, benchmark_decode_tokens, 0,
          "If benchmark is true and the value is larger than 0, the benchmark "
          "will use this number to set the number of decode steps (regardless "
          "of the input prompt).");
ABSL_FLAG(bool, async, true, "Run the LLM execution asynchronously.");
ABSL_FLAG(bool, report_peak_memory_footprint, false,
          "Report peak memory footprint.");
ABSL_FLAG(bool, force_f32, false,
          "Force float 32 precision for the activation data type.");
ABSL_FLAG(bool, multi_turns, false,
          "If true, the command line will ask for multi-turns input.");
ABSL_FLAG(int, num_cpu_threads, 0,
          "If greater than 0, the number of CPU threads to use for the LLM "
          "execution with CPU backend.");
ABSL_FLAG(bool, gpu_external_tensor_mode, false,
          "If false (by default), the GPU backend will use no external tensor "
          "mode which runs slightly faster during decode. It should be set "
          "true when GPU backend doesn't support no external tensor mode, "
          "e.g. Vulkan or OpenGL.");
ABSL_FLAG(bool, configure_magic_numbers, true,
          "If true and the model contains magic numbers, present magic number "
          "configs when the model is initialized.");
ABSL_FLAG(bool, verify_magic_numbers, false,
          "If true and the model contains magic numbers and test signatures, "
          "verify magic number configs when the real dimensions that replaced "
          "magic numbers match with ones of test signatures.");
ABSL_FLAG(bool, clear_kv_cache_before_prefill, true,
          "If true, clear kv cache before the first prefill step. This may "
          "help to disclose any issues related to kv cache.");
ABSL_FLAG(int, num_logits_to_print_after_decode, 0,
          "The number of values at the beginning of logits, in the middle of "
          "logits, and at the end of logits to print after each decode step. "
          "If 0, disables printing logits.");
ABSL_FLAG(std::string, score_target_text, "", "Target text to score.");
ABSL_FLAG(bool, gpu_madvise_original_shared_tensors, true,
          "If true, the GPU backend will madvise the original shared tensors "
          "after use.");
ABSL_FLAG(bool, disable_cache, false, "Disable weight cache.");
ABSL_FLAG(std::string, preferred_device_substr, "",
          "Preferred WebGPU device name substring, case-insensitive. "
          "If not empty, the adapter which the device name contains the "
          "substring will be chosen. "
          "If empty, the device will be determined by other factors.");
ABSL_FLAG(int, num_threads_to_upload, -1,
          "Number of threads for WebGPU weight upload. By default (-1), it's "
          "determined by the runtime.");
ABSL_FLAG(int, num_threads_to_compile, -1,
          "Number of threads for WebGPU kernel compilation. By default (-1), "
          "it's determined by the runtime.");
ABSL_FLAG(
    bool, convert_weights_on_gpu, true,
    "If true, the executor will convert weights on GPU. It is not supported by "
    "the all backends so this flag is ignored when using non-OpenCL and "
    "non-WebGPU backends.");
ABSL_FLAG(bool, optimize_shader_compilation, true,
          "If true, optimize Vulkan shader compilation.");
ABSL_FLAG(bool, share_constant_tensors, true,
          "If true, the executor will enable constant tensor sharing.");
ABSL_FLAG(int, num_iterations, 1,
          "Number of iterations to run the model. By default, it's 1.");
ABSL_FLAG(std::string, litert_dispatch_lib_dir, "",
          "Directory of the LiteRT dispatch library. If not set, the runtime "
          "will look for the library in the path defined as the environment "
          "variables.");
ABSL_FLAG(bool, sampler_handles_input, true,
          "If true and the sampler supports, the sampler manipulates decode "
          "input tensors including tokens, positions, and mask.");
