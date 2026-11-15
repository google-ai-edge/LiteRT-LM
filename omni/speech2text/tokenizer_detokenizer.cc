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

#include "omni/speech2text/tokenizer_detokenizer.h"

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
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/speech2text/detokenizer.h"
#include "omni/speech2text/speech_recognizer.h"
#include "omni/utils/text_utils.h"
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
    : Detokenizer(speech_recognizer), tokenizer_(*tokenizer) {}

absl::Status TokenizerDetokenizer::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto decoded_tokens, speech_recognizer_.GetOutput());
  ABSL_ASSIGN_OR_RETURN(auto words, Detokenize(decoded_tokens));
  PushOutput(std::move(words));
  return absl::OkStatus();
}

absl::StatusOr<absl::string_view> TokenizerDetokenizer::GetDecodedTokenPiece(
    int token_id, bool skip_special_tokens) {
  const std::pair<int, bool> cache_key(token_id, skip_special_tokens);
  if (auto it = token_piece_cache_.find(cache_key);
      it != token_piece_cache_.end()) {
    return it->second;
  }
  // Derive how `token_id` renders when it is not the first piece of a sequence
  // by comparing `{id, id, id}` against `{id, id}`.
  // SentencePiece (`▁`) and HuggingFace Metaspace (`Ġ`) strip the leading space
  // from index 0 of a decoded sequence:
  //   - For token "▁un": d2 = "un un", d3 = "un un un" -> caches " un"
  //   - For token "for": d2 = "forfor", d3 = "forforfor" -> caches "for"
  // Comparing `{id, id, id}` against `{id, id}` also avoids decoding a single
  // token (SentencePiece returns the raw piece string for a lone control token)
  // and identical bytes never merge into one UTF-8 code point, making it
  // byte-safe and independent of the surrounding token stream.
  const int two[2] = {token_id, token_id};
  const int three[3] = {token_id, token_id, token_id};
  ABSL_ASSIGN_OR_RETURN(std::string d2,
                        tokenizer_.TokenIdsToText(two, skip_special_tokens));
  ABSL_ASSIGN_OR_RETURN(std::string d3,
                        tokenizer_.TokenIdsToText(three, skip_special_tokens));
  if (!absl::StartsWith(d3, d2)) {
    return absl::InternalError(
        "Decoded token triple does not start with decoded token pair.");
  }
  auto [it, inserted] =
      token_piece_cache_.emplace(cache_key, d3.substr(d2.size()));
  return it->second;
}

std::optional<std::vector<size_t>>
TokenizerDetokenizer::MapWordsToOnsetTokenIndices(
    absl::Span<const int> token_ids, size_t num_words,
    bool skip_special_tokens) {
  const size_t num_tokens = token_ids.size();
  if (num_words == 0 || num_tokens <= 1) {
    return std::nullopt;
  }

  // Subword tokenizers emit variable numbers of tokens per word, so linear
  // index interpolation (`round(i * (num_tokens - 1) / (num_words - 1))`)
  // shifts word timestamps away from their acoustic onset and breaks overlap
  // alignment in `TimestampTextMerger`.
  //
  // Example: suppose a 5s chunk (frames 0..62, where the 1s overlap with the
  // next chunk starts at frame 50) contains "unfortunately it is open"
  // (num_words = 4, num_tokens = 8):
  //   k=0: " un"   (t=10) -> word 0 ("unfortunately") onset
  //   k=1: "for"   (t=14)
  //   k=2: "tu"    (t=18)
  //   k=3: "nate"  (t=22)
  //   k=4: "ly"    (t=26)
  //   k=5: " it"   (t=48) -> word 1 ("it") onset
  //   k=6: " is"   (t=52) -> word 2 ("is") onset (inside overlap >= 50)
  //   k=7: " open" (t=56) -> word 3 ("open") onset (inside overlap >= 50)
  //
  // Without this mapping, linear interpolation picks token indices
  // `round(i * 7 / 3)` = {0, 2, 5, 7}, assigning:
  //   - word 1 ("it")   -> tokens[2] ("tu", t=18 instead of 48), and
  //   - word 2 ("is")   -> tokens[5] (" it", t=48 < 50 instead of 52 >= 50),
  // pulling "is" out of the >= 50 overlap window so `TimestampTextMerger` locks
  // "is" into the pre-overlap prefix of chunk 1 and duplicates it from chunk 2.
  //
  // By scanning the cached non-initial pieces from `token_piece_cache_`
  // (" un", "for", "tu", "nate", "ly", " it", " is", " open") and starting a
  // new word whenever a non-space character follows a space (`!in_word`), we
  // record the exact starting token index for each word:
  //   - k=0 (" un"):   ' ' -> 'u' starts word 0 -> idx[0] = 0
  //   - k=1..4:        no spaces (`in_word` stays true)
  //   - k=5 (" it"):   ' ' resets `in_word`, 'i' starts word 1 -> idx[1] = 5
  //   - k=6 (" is"):   ' ' resets `in_word`, 'i' starts word 2 -> idx[2] = 6
  //   - k=7 (" open"): ' ' resets `in_word`, 'o' starts word 3 -> idx[3] = 7
  //
  // Words are counted the way `Detokenize` splits them: on spaces, and with
  // every CJK character as its own word (see `SplitCjkCharacters`), so each
  // CJK character gets the timestamp of the token that emits it. Pieces are
  // concatenated before decoding because a byte-level BPE token may hold only
  // part of a multi-byte character.
  std::string text;
  std::vector<size_t> byte_token;  // Token index of each byte of `text`.
  std::vector<size_t> word_token_idx(num_words, 0);
  size_t prev_word_count = 0;
  bool in_word = false;             // Inside a run of non-CJK characters.
  bool attach_punctuation = false;  // Punctuation joins the CJK char before.
  size_t pos = 0;
  for (size_t k = 0; k < num_tokens && prev_word_count < num_words; ++k) {
    auto piece_or = GetDecodedTokenPiece(token_ids[k], skip_special_tokens);
    if (!piece_or.ok()) {
      ABSL_LOG_FIRST_N(WARNING, 5)
          << "Failed to derive decoded token piece for token_id="
          << token_ids[k] << ": " << piece_or.status();
      continue;
    }
    text.append(*piece_or);
    byte_token.resize(text.size(), k);
    const bool last_token = k + 1 == num_tokens;
    while (pos < text.size() && prev_word_count < num_words) {
      // Wait for the next token if it may complete a multi-byte character.
      const unsigned char lead = static_cast<unsigned char>(text[pos]);
      const size_t length = lead >= 0xF0 && lead < 0xF8   ? 4
                            : lead >= 0xE0 && lead < 0xF0 ? 3
                            : lead >= 0xC0 && lead < 0xE0 ? 2
                                                          : 1;
      if (!last_token && pos + length > text.size()) break;
      const size_t start = pos;
      const char32_t cp = DecodeUtf8CodePoint(text, pos);
      bool starts_word = false;
      if (cp == ' ') {
        in_word = false;
        attach_punctuation = false;
      } else if (IsCjkCharacter(cp)) {
        starts_word = true;
        in_word = false;
        attach_punctuation = true;
      } else if (!(attach_punctuation && IsPunctuation(cp))) {
        starts_word = !in_word;
        in_word = true;
        attach_punctuation = false;
      }
      if (starts_word) {
        word_token_idx[prev_word_count++] = byte_token[start];
      }
    }
  }

  if (prev_word_count == 0) {
    ABSL_LOG_FIRST_N(WARNING, 5)
        << "Word-onset token mapping resolved 0 of " << num_words
        << " words across " << num_tokens
        << " tokens; falling back to linear index interpolation.";
    return std::nullopt;
  }

  while (prev_word_count < num_words) {
    word_token_idx[prev_word_count] = num_tokens - 1;
    ++prev_word_count;
  }
  return word_token_idx;
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
  bool is_warm_start_chunk = false;
  bool has_token_timestamps = false;
  for (const auto& tok : tokens) {
    if (tok.IsEndOfChunk()) {
      end_of_chunk_timestamp_ms = tok.timestamp_ms;
      is_warm_start_chunk = tok.is_warm_start;
      break;
    }
    if (tok.timestamp_ms.has_value()) {
      has_token_timestamps = true;
    }
    token_ids.push_back(tok.token_id);
  }

  std::vector<Detokenizer::Word> words;
  if (!token_ids.empty()) {
    bool skip_special_tokens = true;
    auto text_status =
        tokenizer_.TokenIdsToText(token_ids, skip_special_tokens);
    // SentencePieceTokenizer (used by Gemma ASR models) does not support
    // skip_special_tokens=true and returns InvalidArgumentError because special
    // tokens are handled by the SentencePiece buildenormalizer. Fall back to
    // skip_special_tokens=false.
    if (!text_status.ok() && absl::IsInvalidArgument(text_status.status())) {
      skip_special_tokens = false;
      text_status = tokenizer_.TokenIdsToText(token_ids, skip_special_tokens);
    }
    ABSL_ASSIGN_OR_RETURN(auto text, std::move(text_status));
    ABSL_VLOG(1) << "Detokenized text: " << text;
    // Every CJK character is a word of its own (see `SplitCjkCharacters`):
    // Chinese and Japanese have no spaces, and the text mergers align
    // overlapping windows word by word.
    const std::vector<std::string> space_words =
        absl::StrSplit(text, ' ', absl::SkipEmpty());
    const std::vector<std::string> raw_words = SplitCjkCharacters(space_words);

    words.reserve(
        raw_words.size() +
        ((end_of_chunk_timestamp_ms.has_value() || is_warm_start_chunk) ? 1
                                                                        : 0));

    const size_t num_words = raw_words.size();
    const size_t num_tokens = token_ids.size();

    // Find the starting token index of each word in O(N) when tokens carry
    // timestamps. Subword tokenizers (SentencePiece, BPE) often use 2-5 tokens
    // for one word and 1 token for another; linear index interpolation shifts
    // word timestamps away from their acoustic onset and breaks overlap
    // alignment.
    std::optional<std::vector<size_t>> word_token_idx;
    if (has_token_timestamps && num_words >= 1 && num_tokens > 1) {
      word_token_idx = MapWordsToOnsetTokenIndices(token_ids, num_words,
                                                   skip_special_tokens);
    }
    const bool has_word_onset_mapping = word_token_idx.has_value();

    for (size_t i = 0; i < num_words; ++i) {
      std::optional<int> timestamp_ms = std::nullopt;
      if (has_token_timestamps && num_tokens > 0) {
        size_t token_idx = 0;
        if (has_word_onset_mapping) {
          token_idx = (*word_token_idx)[i];
        } else if (num_words <= 1) {
          token_idx = 0;
        } else {
          token_idx = static_cast<size_t>(std::round(
              static_cast<double>(i) * (num_tokens - 1) / (num_words - 1)));
        }
        token_idx = std::min(token_idx, num_tokens - 1);

        if (tokens[token_idx].timestamp_ms.has_value()) {
          timestamp_ms = tokens[token_idx].timestamp_ms;
        } else {
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
      }
      words.push_back(Detokenizer::Word{.text = raw_words[i],
                                        .timestamp_ms = timestamp_ms});
    }
  }

  if (end_of_chunk_timestamp_ms.has_value() || is_warm_start_chunk) {
    words.push_back(Detokenizer::Word{.text = "",  // end-of-chunk
                                      .timestamp_ms = end_of_chunk_timestamp_ms,
                                      .is_warm_start = is_warm_start_chunk});
  }

  ABSL_VLOG(2) << "Words(size=" << words.size()
               << "): " << absl::StrJoin(words, ", ", StrAppendWord);

  return words;
}

}  // namespace litert::omni::asr
