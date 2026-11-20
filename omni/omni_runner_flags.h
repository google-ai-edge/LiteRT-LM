// Copyright 2026 The ODML Authors.
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

// Command-line flags and entrypoints for standalone `omni_runner` binaries.
//
// NOTE: `omni_runner_flags` defines unprefixed global flags (e.g., `--backend`,
// `--model_path`, `--cache_dir`, `--text`, `--seed`) that can collide with
// flags defined in other libraries. This target is intended only for standalone
// `omni_runner` binaries (e.g., via `omni_runner_binary`) and should not be
// linked into general libraries; use `omni_runner_lib` (`OmniRunnerOptions` and
// `RunOmniRunner`) for programmatic invocation instead.

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_RUNNER_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_RUNNER_FLAGS_H_

#include <cstdint>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "omni/omni_runner_lib.h"

// Common engine flags.
ABSL_DECLARE_FLAG(std::string, model_name);
ABSL_DECLARE_FLAG(std::string, model_path);
ABSL_DECLARE_FLAG(std::string, backend);
ABSL_DECLARE_FLAG(bool, force_fp32_on_gpu);
ABSL_DECLARE_FLAG(int, num_threads);
ABSL_DECLARE_FLAG(std::string, cache_dir);
ABSL_DECLARE_FLAG(std::string, litert_dispatch_lib_dir);

// Input & output flags.
ABSL_DECLARE_FLAG(std::string, audio_path);
ABSL_DECLARE_FLAG(std::string, image_path);
ABSL_DECLARE_FLAG(std::string, text);
ABSL_DECLARE_FLAG(std::string, output_path);

// Any-to-Any session flags.
ABSL_DECLARE_FLAG(bool, generate_audio);
ABSL_DECLARE_FLAG(bool, generate_image);
ABSL_DECLARE_FLAG(std::optional<std::string>, system_prompt);
ABSL_DECLARE_FLAG(std::optional<int>, max_new_tokens);
ABSL_DECLARE_FLAG(std::optional<int>, max_tts_tokens);
ABSL_DECLARE_FLAG(std::optional<float>, temperature);
ABSL_DECLARE_FLAG(std::optional<float>, tts_temperature);
ABSL_DECLARE_FLAG(std::optional<int>, tts_top_k);
ABSL_DECLARE_FLAG(std::optional<float>, cfg_weight);
ABSL_DECLARE_FLAG(std::optional<float>, image_temperature);

// ASR session flags.
ABSL_DECLARE_FLAG(float, overlap_ratio);
ABSL_DECLARE_FLAG(std::string, text_merger_type);

// TTS / Any-to-Any voice & language flags.
ABSL_DECLARE_FLAG(std::string, language);
ABSL_DECLARE_FLAG(std::string, voice);

// Text-to-Image / Any-to-Any image generation flags.
ABSL_DECLARE_FLAG(int, img_size);
ABSL_DECLARE_FLAG(int, steps);
ABSL_DECLARE_FLAG(uint64_t, seed);

namespace litert::omni {

// Builds an `OmniRunnerOptions` snapshot from the current command-line flag
// values.
OmniRunnerOptions BuildOmniRunnerOptionsFromFlags();

// Runs the Omni runner using the current flag values, writing output to `out`
// and error messages to `err`. Returns 0 on success or 1 on failure.
int RunOmniRunnerFromFlags(std::ostream& out = std::cout,
                           std::ostream& err = std::cerr);

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_RUNNER_FLAGS_H_
