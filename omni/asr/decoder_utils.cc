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

#include "omni/asr/decoder_utils.h"

#include <cstddef>
#include <vector>

#include "omni/asr/speech_recognizer.h"

namespace litert::omni::asr {
namespace {

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
