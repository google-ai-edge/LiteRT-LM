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

#include "omni/asr/tokenizer_detokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/asr/detokenizer.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {
namespace {

// Formatters for absl::StrJoin.
void StrAppendToken(std::string* out,
                    const SpeechRecognizer::DecodedToken& token) {
  absl::StrAppend(out, token.token_id, "(", token.timestamp_ms.value_or(-1),
                  ")");
}

void StrAppendWord(std::string* out, const Detokenizer::Word& word) {
  absl::StrAppend(out, word.text, "(", word.timestamp_ms.value_or(-1), ")");
}

}  // namespace

TokenizerDetokenizer::TokenizerDetokenizer(
    Stage<std::vector<SpeechRecognizer::DecodedToken>>* absl_nonnull
        speech_recognizer,
    ::litert::support::Tokenizer* absl_nonnull tokenizer)
    : Detokenizer(speech_recognizer), tokenizer_(tokenizer) {}

absl::Status TokenizerDetokenizer::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto decoded_tokens, speech_recognizer_.GetOutput());
  ABSL_ASSIGN_OR_RETURN(auto words, Detokenize(decoded_tokens));
  PushOutput(std::move(words));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Detokenizer::Word>> TokenizerDetokenizer::Detokenize(
    const std::vector<SpeechRecognizer::DecodedToken>& tokens) {
  if (tokens.empty()) {
    return std::vector<Detokenizer::Word>();
  }

  ABSL_VLOG(2) << "Tokens(size=" << tokens.size()
               << "): " << absl::StrJoin(tokens, ", ", StrAppendToken);

  std::vector<int> token_ids;
  token_ids.reserve(tokens.size());
  std::optional<int> end_of_chunk_timestamp_ms = std::nullopt;
  for (const auto& tok : tokens) {
    if (tok.IsEndOfChunk()) {
      end_of_chunk_timestamp_ms = tok.timestamp_ms;
      break;
    }
    token_ids.push_back(tok.token_id);
  }

  std::vector<Detokenizer::Word> words;
  if (!token_ids.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        auto text, tokenizer_->TokenIdsToText(token_ids,
                                              /*skip_special_tokens=*/true));
    ABSL_VLOG(1) << "Detokenized text: " << text;
    std::vector<absl::string_view> raw_words =
        absl::StrSplit(text, ' ', absl::SkipEmpty());

    words.reserve(raw_words.size() +
                  (end_of_chunk_timestamp_ms.has_value() ? 1 : 0));

    const size_t num_words = raw_words.size();
    const size_t num_tokens = token_ids.size();

    // Calculate word timestamps by interpolating token timestamps based on
    // relative position in the sequence. For example, if there are 3 words and
    // 5 tokens, the words correspond to token indices 0, 2, and 4.
    for (size_t i = 0; i < num_words; ++i) {
      size_t token_idx = 0;
      if (num_tokens > 0) {
        if (num_words <= 1) {
          token_idx = 0;
        } else {
          token_idx = static_cast<size_t>(std::round(
              static_cast<double>(i) * (num_tokens - 1) / (num_words - 1)));
        }
        token_idx = std::min(token_idx, num_tokens - 1);
      }

      std::optional<int> timestamp_ms = std::nullopt;
      if (token_idx < num_tokens &&
          tokens[token_idx].timestamp_ms.has_value()) {
        timestamp_ms = tokens[token_idx].timestamp_ms;
      } else if (num_tokens > 0) {
        size_t min_dist = std::numeric_limits<size_t>::max();
        for (size_t j = 0; j < num_tokens; ++j) {
          if (!tokens[j].timestamp_ms.has_value()) {
            continue;
          }
          size_t dist = (j > token_idx) ? (j - token_idx) : (token_idx - j);
          if (dist < min_dist) {
            min_dist = dist;
            timestamp_ms = tokens[j].timestamp_ms;
          }
        }
      }
      words.push_back(Detokenizer::Word{.text = std::string(raw_words[i]),
                                        .timestamp_ms = timestamp_ms});
    }
  }

  if (end_of_chunk_timestamp_ms.has_value()) {
    words.push_back(
        Detokenizer::Word{.text = "",  // end-of-chunk
                          .timestamp_ms = end_of_chunk_timestamp_ms});
  }

  ABSL_VLOG(2) << "Words(size=" << words.size()
               << "): " << absl::StrJoin(words, ", ", StrAppendWord);

  return words;
}

}  // namespace litert::omni::asr
