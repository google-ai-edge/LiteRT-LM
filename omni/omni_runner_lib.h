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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_RUNNER_LIB_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_RUNNER_LIB_H_

#include <cstdint>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl

namespace litert::omni {

// Configuration options for running the LiteRT-Omni CLI runner workflow.
struct OmniRunnerOptions {
  // Common engine options.
  std::string model_name;
  std::string model_path;
  std::string backend = "cpu";
  bool force_fp32_on_gpu = false;
  int num_threads = 4;
  std::string cache_dir = "/tmp/omni_cache";
  std::string litert_dispatch_lib_dir;

  // Input & output options.
  std::string audio_path;
  std::string image_path;
  std::string text;
  std::string output_path;

  // Any-to-Any session options.
  bool generate_audio = false;
  bool generate_image = false;
  std::optional<std::string> system_prompt;
  std::optional<int> max_new_tokens;
  std::optional<int> max_tts_tokens;
  std::optional<float> temperature;
  std::optional<float> tts_temperature;
  std::optional<int> tts_top_k;
  std::optional<float> cfg_weight;
  std::optional<float> image_temperature;

  // ASR session options.
  float overlap_ratio = 0.4f;
  std::string text_merger_type = "timestamp";

  // TTS / Any-to-Any voice & language options. When `language` is empty, TTS
  // models default to "en-US" and Any-to-Any models use the embedded metadata
  // default.
  std::string language;
  std::string voice;

  // Text-to-Image / Any-to-Any image generation options.
  int img_size = 0;
  int steps = 4;
  uint64_t seed = 42;
};

// Executes the Omni runner workflow for `options` against models registered in
// `OmniRegistry`, writing user-facing output and summaries to `out`.
absl::Status RunOmniRunner(const OmniRunnerOptions& options,
                           std::ostream& out = std::cout);

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_RUNNER_LIB_H_
