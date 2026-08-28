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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_MODEL_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_MODEL_CONFIG_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl

namespace litert::omni::tts {

// Model-specific configuration settings for Kokoro-82M TTS.
struct KokoroModelConfig {
  // Voice profile name or filename (e.g., "af_heart", "af_bella",
  // "am_michael").
  std::string voice_name = "af_heart";
  // Language identifier or voice prefix (e.g., "en-us", "en-gb", "es", "fr-fr",
  // "hi", "it", "pt-br", "ja", "zh", "a", "b", "e", etc.). Defaults to "en-us".
  std::string language = "en-us";
  // Optional explicit path/file to voice embedding. If empty, resolves from
  // voice_name.
  std::string voice_file;
  // Target sequence length bucket (0 = auto / load all available).
  int target_bucket = 0;
  // Model filenames under model directory.
  std::string acoustic_file = "kokoro_acoustic.tflite";
  std::string vocoder_file = "kokoro_vocoder.tflite";
  // Optional custom dictionary / pronunciation overrides.
  absl::flat_hash_map<std::string, std::string> custom_lexicon;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_MODEL_CONFIG_H_
