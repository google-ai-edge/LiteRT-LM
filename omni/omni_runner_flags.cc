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

#include "omni/omni_runner_flags.h"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/omni_runner_lib.h"

// Common engine flags.
ABSL_FLAG(std::string, model_name, "",
          "Model name (e.g., parakeet-tdt-0.6b-v3, kokoro, bonsai-flux2, "
          "minicpm-o-4.5, qwen2.5-omni-3b). If empty, inferred from "
          "--model_path.");
ABSL_FLAG(std::string, model_path, "",
          "Path to a .litertlm model package or model directory.");
ABSL_FLAG(std::string, backend, "cpu",
          "Hardware backend accelerator: cpu, gpu, or npu.");
ABSL_FLAG(bool, force_fp32_on_gpu, false,
          "If true, forces FP32 precision on GPU instead of FP16.");
ABSL_FLAG(int, num_threads, 4, "Number of CPU threads for LiteRT inference.");
ABSL_FLAG(std::string, cache_dir, "/tmp/omni_cache",
          "Cache directory for downloaded models and compiled backend caches.");
ABSL_FLAG(std::string, litert_dispatch_lib_dir, "",
          "Directory containing LiteRT NPU dispatch libraries. If empty, "
          "defaults to the model file's parent directory.");

// Input & output flags.
ABSL_FLAG(std::string, audio_path, "",
          "Path to input WAV audio file (for ASR or Any-to-Any).");
ABSL_FLAG(std::string, image_path, "",
          "Path to input PNG/JPEG/BMP image file (for Any-to-Any).");
ABSL_FLAG(std::string, text, "",
          "Input text prompt (for TTS, Text-to-Image, or Any-to-Any).");
ABSL_FLAG(std::string, output_path, "",
          "Output file path (.wav for TTS/Any-to-Any audio, .png for "
          "Text-to-Image/Any-to-Any image, or optional .txt file for "
          "ASR/Any-to-Any text).");

// Any-to-Any session flags.
ABSL_FLAG(bool, generate_audio, false,
          "Whether to synthesize speech audio output for Any-to-Any models.");
ABSL_FLAG(bool, generate_image, false,
          "Whether to generate an output image instead of text for Any-to-Any "
          "models.");
ABSL_FLAG(std::optional<std::string>, system_prompt, std::nullopt,
          "Optional system prompt override for Any-to-Any models.");
ABSL_FLAG(std::optional<int>, max_new_tokens, std::nullopt,
          "Optional maximum number of text tokens to generate for Any-to-Any "
          "models.");
ABSL_FLAG(std::optional<int>, max_tts_tokens, std::nullopt,
          "Optional maximum number of TTS audio frames to generate for "
          "Any-to-Any models.");
ABSL_FLAG(std::optional<float>, temperature, std::nullopt,
          "Optional sampling temperature for Any-to-Any text generation.");
ABSL_FLAG(std::optional<float>, tts_temperature, std::nullopt,
          "Optional TTS audio token sampling temperature for Any-to-Any "
          "models.");
ABSL_FLAG(std::optional<int>, tts_top_k, std::nullopt,
          "Optional TTS audio token top-k sampling limit for Any-to-Any "
          "models.");
ABSL_FLAG(std::optional<float>, cfg_weight, std::nullopt,
          "Optional classifier-free guidance weight for Any-to-Any image "
          "generation.");
ABSL_FLAG(std::optional<float>, image_temperature, std::nullopt,
          "Optional sampling temperature for Any-to-Any image token "
          "generation.");

// ASR session flags.
ABSL_FLAG(float, overlap_ratio, 0.4f,
          "ASR chunk overlap ratio between 0.0 and 1.0.");
ABSL_FLAG(std::string, text_merger_type, "timestamp",
          "ASR text merger strategy: timestamp or levenshtein.");

// TTS / Any-to-Any voice & language flags.
ABSL_FLAG(std::string, language, "",
          "Optional TTS / Any-to-Any language code (BCP-47, e.g., en-US, es, "
          "zh-CN). If empty, defaults to en-US for TTS and preserves the "
          "model's embedded default for Any-to-Any.");
ABSL_FLAG(std::string, voice, "",
          "Optional TTS / Any-to-Any voice ID (e.g., af_heart, Chelsie, "
          "Ethan). If empty, uses default.");

// Text-to-Image / Any-to-Any image generation flags.
ABSL_FLAG(int, img_size, 0,
          "Target output image width and height in pixels (e.g., 256 or 512). "
          "If 0, uses the model's default size.");
ABSL_FLAG(int, steps, 4, "Number of denoising steps for Text-to-Image.");
ABSL_FLAG(uint64_t, seed, 42,
          "Random seed for Text-to-Image or Any-to-Any image generation.");

namespace litert::omni {

OmniRunnerOptions BuildOmniRunnerOptionsFromFlags() {
  OmniRunnerOptions options;
  options.model_name = absl::GetFlag(FLAGS_model_name);
  options.model_path = absl::GetFlag(FLAGS_model_path);
  options.backend = absl::GetFlag(FLAGS_backend);
  options.force_fp32_on_gpu = absl::GetFlag(FLAGS_force_fp32_on_gpu);
  options.num_threads = absl::GetFlag(FLAGS_num_threads);
  options.cache_dir = absl::GetFlag(FLAGS_cache_dir);
  options.litert_dispatch_lib_dir =
      absl::GetFlag(FLAGS_litert_dispatch_lib_dir);

  options.audio_path = absl::GetFlag(FLAGS_audio_path);
  options.image_path = absl::GetFlag(FLAGS_image_path);
  options.text = absl::GetFlag(FLAGS_text);
  options.output_path = absl::GetFlag(FLAGS_output_path);

  options.generate_audio = absl::GetFlag(FLAGS_generate_audio);
  options.generate_image = absl::GetFlag(FLAGS_generate_image);
  options.system_prompt = absl::GetFlag(FLAGS_system_prompt);
  options.max_new_tokens = absl::GetFlag(FLAGS_max_new_tokens);
  options.max_tts_tokens = absl::GetFlag(FLAGS_max_tts_tokens);
  options.temperature = absl::GetFlag(FLAGS_temperature);
  options.tts_temperature = absl::GetFlag(FLAGS_tts_temperature);
  options.tts_top_k = absl::GetFlag(FLAGS_tts_top_k);
  options.cfg_weight = absl::GetFlag(FLAGS_cfg_weight);
  options.image_temperature = absl::GetFlag(FLAGS_image_temperature);

  options.overlap_ratio = absl::GetFlag(FLAGS_overlap_ratio);
  options.text_merger_type = absl::GetFlag(FLAGS_text_merger_type);

  options.language = absl::GetFlag(FLAGS_language);
  options.voice = absl::GetFlag(FLAGS_voice);

  options.img_size = absl::GetFlag(FLAGS_img_size);
  options.steps = absl::GetFlag(FLAGS_steps);
  options.seed = absl::GetFlag(FLAGS_seed);
  return options;
}

int RunOmniRunnerFromFlags(std::ostream& out, std::ostream& err) {
  const absl::Status status =
      RunOmniRunner(BuildOmniRunnerOptionsFromFlags(), out);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Omni runner failed: " << status;
    err << "ERROR: " << status << "\n";
    return 1;
  }
  return 0;
}

}  // namespace litert::omni
