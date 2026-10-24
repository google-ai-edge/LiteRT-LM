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

#include "omni/speech2text/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "omni/speech2text/speech_recognizer.h"

namespace litert::omni::asr {
namespace {

// Maximum RMS energy (~-80 dBFS) for an audio chunk to be classified as silent.
constexpr double kSilenceMaxRms = 1e-4;

// Maximum peak amplitude (~-66 dBFS) for an audio chunk to be classified as
// silent, preventing short transient clicks, quiet consonant onsets, or
// low-gain recordings from being dropped.
constexpr float kSilenceMaxPeak = 5e-4f;

// Minimum number of consecutive repetitions of a k-gram required to classify a
// trailing sequence as a degenerate repetition loop.
constexpr size_t kMinRepeats = 4;

// Maximum k-gram length (in tokens) checked for consecutive repetitions.
// Degenerate ASR loops typically repeat a single token, word, or short phrase
// (up to ~10-12 subword tokens); capping at 16 bounds the per-step check while
// ensuring 4 consecutive repeats (64 tokens) fits within a single audio chunk's
// decode budget.
constexpr size_t kMaxRepetitionNgramLength = 16;

}  // namespace

bool IsSilentAudio(absl::Span<const float> raw_speech) {
  if (raw_speech.empty()) {
    return true;
  }
  double sum_sq = 0.0;
  float peak = 0.0f;
  for (float s : raw_speech) {
    float abs_s = std::abs(s);
    if (abs_s > peak) peak = abs_s;
    sum_sq += static_cast<double>(s) * s;
  }
  double rms = std::sqrt(sum_sq / raw_speech.size());
  return rms < kSilenceMaxRms && peak < kSilenceMaxPeak;
}

float ShiftShortChunkBackOverPreviousWindow(std::vector<float>& raw_speech,
                                            std::vector<float>& previous_window,
                                            size_t window_samples,
                                            int min_overlap_samples) {
  const size_t missing = window_samples - raw_speech.size();
  // Overlapping audio sources (`AudioInputSource`, `FileAudioSource`) prefix
  // follow-up chunks with the retained overlap samples from the end of
  // `previous_window`. Identify that exact overlap so shifting a short final
  // chunk back to a full window splices the contiguous samples preceding
  // `raw_speech.begin()` rather than duplicating the overlap.
  size_t overlap = 0;
  const size_t min_overlap =
      min_overlap_samples > 0 ? static_cast<size_t>(min_overlap_samples) : 1;
  for (size_t candidate = raw_speech.size(); candidate >= min_overlap;
       --candidate) {
    if (std::equal(previous_window.end() - candidate, previous_window.end(),
                   raw_speech.begin())) {
      overlap = candidate;
      break;
    }
  }
  std::vector<float> window;
  window.reserve(window_samples);
  window.insert(window.end(), previous_window.end() - overlap - missing,
                previous_window.end() - overlap);
  window.insert(window.end(), raw_speech.begin(), raw_speech.end());
  raw_speech = std::move(window);
  previous_window = raw_speech;
  return static_cast<float>(missing) / static_cast<float>(window_samples);
}

bool TruncateOnTrailingRepetition(
    std::vector<SpeechRecognizer::DecodedToken>& tokens, int new_token) {
  const size_t n = tokens.size();
  // Block any k-gram (k = 1..kMaxRepetitionNgramLength) repeated kMinRepeats
  // times consecutively, and remove the repeated copies from tokens.
  for (size_t k = 1; k <= kMaxRepetitionNgramLength; ++k) {
    if (n + 1 < kMinRepeats * k) continue;
    bool match = true;
    for (size_t r = 1; r < kMinRepeats && match; ++r) {
      for (size_t j = 0; j < k; ++j) {
        int tok_latest =
            (j == k - 1) ? new_token : tokens[n + 1 - k + j].token_id;
        int tok_prev = tokens[n + 1 - (r + 1) * k + j].token_id;
        if (tok_latest != tok_prev) {
          match = false;
          break;
        }
      }
    }
    if (match) {
      tokens.resize(n + 1 - (kMinRepeats - 1) * k);
      return true;
    }
  }
  return false;
}

}  // namespace litert::omni::asr
