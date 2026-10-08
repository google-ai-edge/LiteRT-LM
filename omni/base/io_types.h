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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_IO_TYPES_H_

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace litert::omni {

// Sentinel input indicating the end of the input stream.
struct EndOfInput {};

// Text input payload for TTS synthesis or image generation prompts.
struct TextInput {
  std::string text;
};

// Audio input metadata for ASR transcription applied to all following
// AudioInput until another AudioInputMetadata is pushed.
struct AudioInputMetadata {
  int sample_rate_hz = 16000;
  int num_channels = 1;
};

// Audio input payload for ASR transcription.
struct AudioInput {
  std::vector<float> pcm_samples;
};

// Image generation metadata applied to all following TextInput prompts until
// another ImageGenInputMetadata is pushed.
struct ImageGenInputMetadata {
  int width = 512;
  int height = 512;
  int num_inference_steps = 4;
  uint64_t seed = 42;
};

// Unified input variant for InputSource. Note that `EndOfInput` is the first
// alternative, so a default-constructed `Input` represents `EndOfInput`.
using Input = std::variant<EndOfInput, TextInput, AudioInputMetadata,
                           AudioInput, ImageGenInputMetadata>;

// Sentinel output indicating the end of the output stream.
// TODO(b/538727793): Define synchronous/streaming `EndOfOutput` semantics.
// Currently sessions signal end-of-stream via `absl::OutOfRangeError`. Note
// that `EndOfOutput` is the first alternative of `Output`, so a
// default-constructed `Output` represents `EndOfOutput`.
struct EndOfOutput {};

// Generic text output payload for ASR transcription or text generation.
struct TextOutput {
  // Stable text that will no longer change. Only contains newly confirmed
  // text since the last output.
  std::string confirmed_text;
  // Unstable hypothesis text from the tail of the current window.
  std::string unconfirmed_text;
};

// Generic audio synthesis output payload for vocoder and audio output.
struct AudioOutput {
  std::vector<float> pcm_samples;
  int sample_rate_hz = 24000;
};

// Generic image generation output payload in interleaved RGB888 format
// (height x width x channels).
struct ImageOutput {
  std::vector<uint8_t> rgb_data;
  int width = 0;
  int height = 0;
  int channels = 3;
};

// Unified output variant for OmniSession and output stages.
using Output = std::variant<EndOfOutput, TextOutput, AudioOutput, ImageOutput>;

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_IO_TYPES_H_
