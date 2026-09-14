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

// How one end of an acoustic slice came to be.
//
// A chunk whose phoneme sequence exceeds the model's capacity must be cut into
// several slices, and each slice is then synthesized as a self-contained
// utterance with its own BOS/EOS. The model therefore places utterance-final
// silence before every cut and utterance-initial silence after it, regardless
// of what the text says, and the downstream stage needs to know how much of
// that silence the text actually justifies.
enum class SliceJoin {
  // The real start or end of a chunk. Its silence belongs to the text.
  kChunkBoundary,
  // A capacity-forced cut that landed on punctuation, so the text does call
  // for a pause here, just a clause-length one.
  kPunctuation,
  // A capacity-forced cut that landed mid-phrase. Any pause here is a pure
  // artifact.
  kWordBoundary,
};

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
  // How each end of this slice was cut.
  SliceJoin join_before = SliceJoin::kChunkBoundary;
  SliceJoin join_after = SliceJoin::kChunkBoundary;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_IO_TYPES_H_
