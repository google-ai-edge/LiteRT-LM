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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_UTILS_H_

#include <cstddef>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "omni/speech2text/speech_recognizer.h"

namespace litert::omni::asr {

// Default minimum overlap (160 samples = 10 ms at 16 kHz) checked when
// deduplicating shared overlap prefixes across adjacent audio chunks.
inline constexpr int kDefaultMinOverlapSamples = 160;

// Returns true if `raw_speech` is empty or contains only digital silence /
// near-silent background noise (both RMS energy and peak amplitude fall below
// silence thresholds). Audio preprocessors use this to emit an empty feature
// vector on silent chunks so that feature normalization does not amplify
// near-zero noise and cause downstream ASR decoders to hallucinate text.
//
// args:
//   raw_speech: Raw PCM waveform samples to inspect.
// returns:
//   True if `raw_speech` is empty or below both RMS and peak silence
//   thresholds; false otherwise.
bool IsSilentAudio(absl::Span<const float> raw_speech);

// Shifts a short final chunk `raw_speech` back over `previous_window` to form a
// full `window_samples`-length window ending at `raw_speech.end()`,
// deduplicating any overlap prefix already shared with the tail of
// `previous_window`. Updates `raw_speech` and `previous_window` in place to the
// reconstructed window.
//
// args:
//   raw_speech: Short follow-up chunk (`raw_speech.size() < window_samples`),
//     updated in place to the reconstructed `window_samples`-length window.
//   previous_window: Full `window_samples`-length window from the immediately
//     preceding chunk, updated in place to the reconstructed window.
//   window_samples: Target window length in samples.
//   min_overlap_samples: Minimum overlap length in samples to check when
//     deduplicating shared overlap prefixes between `previous_window` and
//     `raw_speech`.
// returns:
//   The fraction in `[0, 1)` of the reconstructed window that precedes the
//   newly appended samples (`(window_samples - raw_speech.size()) /
//   window_samples` before replacement).
float ShiftShortChunkBackOverPreviousWindow(
    std::vector<float>& raw_speech, std::vector<float>& previous_window,
    size_t window_samples, int min_overlap_samples = kDefaultMinOverlapSamples);

// Autoregressive ASR decoders (such as LLM-based qwen3-asr-0.6b and
// tinygemma-asr in LmDecoder, and encoder-decoder whisper-tiny and
// moonshine-tiny in StatelessDecoder) use greedy argmax decoding without a
// repetition penalty. On noisy, silent, or zero-padded trailing audio
// segments, they can enter degenerate n-gram repetition loops instead of
// emitting a stop token. When any k-gram (k = 1..16) repeats 4 times
// consecutively upon appending `new_token`, this helper truncates the extra
// repeated copies from `tokens` and returns true so the decode loop can
// terminate early.
//
// args:
//   tokens: Decoded token sequence accumulated so far, truncated in place if a
//     trailing repetition loop is detected.
//   new_token: Candidate token ID about to be appended to `tokens`.
// returns:
//   True if appending `new_token` completes 4 consecutive repeats of any k-gram
//   (k = 1..16) and `tokens` was truncated; false otherwise.
bool TruncateOnTrailingRepetition(
    std::vector<SpeechRecognizer::DecodedToken>& tokens, int new_token);

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_UTILS_H_
