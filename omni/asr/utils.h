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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_UTILS_H_

#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/speech_recognizer.h"

namespace litert::omni::asr {

// Returns true if `raw_speech` is empty or contains only digital silence /
// near-silent background noise (both RMS energy and peak amplitude fall below
// silence thresholds). Audio preprocessors use this to emit an empty feature
// vector on silent chunks so that feature normalization does not amplify
// near-zero noise and cause downstream ASR decoders to hallucinate text.
bool IsSilentAudio(absl::Span<const float> raw_speech);

// Autoregressive ASR decoders (such as LLM-based qwen3-asr-0.6b and
// tinygemma-asr in LmDecoder, and encoder-decoder whisper-tiny and
// moonshine-tiny in StatelessDecoder) use greedy argmax decoding without a
// repetition penalty. On noisy, silent, or zero-padded trailing audio
// segments, they can enter degenerate n-gram repetition loops instead of
// emitting a stop token. When any k-gram (k = 1..16) repeats 4 times
// consecutively upon appending `new_token`, this helper truncates the extra
// repeated copies from `tokens` and returns true so the decode loop can
// terminate early.
bool TruncateOnTrailingRepetition(
    std::vector<SpeechRecognizer::DecodedToken>& tokens, int new_token);

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_UTILS_H_
