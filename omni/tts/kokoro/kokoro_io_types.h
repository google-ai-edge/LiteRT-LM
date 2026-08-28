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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_IO_TYPES_H_

#include <vector>

namespace litert::omni::tts {

// Output payload for the Kokoro acoustic / prosody prediction stage.
struct KokoroAcousticOutput {
  // Acoustic feature representation tensor of shape [1, 512, l_speech].
  std::vector<float> asr_data;
  // Fundamental frequency (F0) pitch contour feature tensor of shape [1, 1,
  // l_speech].
  std::vector<float> f0_n_data;
  // Energy / norm auxiliary contour feature tensor of shape [1, 1, l_speech].
  std::vector<float> n_aux_data;
  // Voice style reference embedding slice forwarded to vocoder [128 floats].
  std::vector<float> ref_s_decoder;
  // Active number of speech acoustic frames.
  int l_speech = 0;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_IO_TYPES_H_
