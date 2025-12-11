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
//
// This library defines the common flags for litert_lm_main and
// litert_lm_advanced_main.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SHARED_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SHARED_FLAGS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/flags/declare.h"  // from @com_google_absl

// Common flags for litert_lm_main and litert_lm_example.
ABSL_DECLARE_FLAG(std::optional<std::string>, vision_backend);
ABSL_DECLARE_FLAG(std::optional<std::string>, audio_backend);
ABSL_DECLARE_FLAG(std::string, sampler_backend);
ABSL_DECLARE_FLAG(std::string, expected_output);
ABSL_DECLARE_FLAG(std::optional<std::string>, log_sink_file);
ABSL_DECLARE_FLAG(int, max_num_tokens);
ABSL_DECLARE_FLAG(std::vector<std::string>, prefill_batch_sizes);
ABSL_DECLARE_FLAG(int, num_output_candidates);
ABSL_DECLARE_FLAG(bool, benchmark);
ABSL_DECLARE_FLAG(int, benchmark_prefill_tokens);
ABSL_DECLARE_FLAG(int, benchmark_decode_tokens);
ABSL_DECLARE_FLAG(bool, async);
ABSL_DECLARE_FLAG(bool, report_peak_memory_footprint);
ABSL_DECLARE_FLAG(bool, force_f32);
ABSL_DECLARE_FLAG(bool, multi_turns);
ABSL_DECLARE_FLAG(int, num_cpu_threads);
ABSL_DECLARE_FLAG(bool, gpu_external_tensor_mode);
ABSL_DECLARE_FLAG(bool, configure_magic_numbers);
ABSL_DECLARE_FLAG(bool, verify_magic_numbers);
ABSL_DECLARE_FLAG(bool, clear_kv_cache_before_prefill);
ABSL_DECLARE_FLAG(int, num_logits_to_print_after_decode);
ABSL_DECLARE_FLAG(std::string, score_target_text);
ABSL_DECLARE_FLAG(bool, gpu_madvise_original_shared_tensors);
ABSL_DECLARE_FLAG(bool, disable_cache);
ABSL_DECLARE_FLAG(std::string, preferred_device_substr);

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SHARED_FLAGS_H_
