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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_TOKENIZER_DETOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_TOKENIZER_DETOKENIZER_H_

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/speech2text/detokenizer.h"
#include "omni/speech2text/speech_recognizer.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {

// Converts SpeechRecognizer tokens into words using litert::support::Tokenizer.
class TokenizerDetokenizer : public Detokenizer {
 public:
  TokenizerDetokenizer(
      Stage<std::vector<SpeechRecognizer::DecodedToken>>* absl_nonnull
          speech_recognizer,
      ::litert::support::Tokenizer* absl_nonnull tokenizer);

  ~TokenizerDetokenizer() override = default;

 protected:
  absl::Status ScheduleInternal() override;

 private:
  absl::StatusOr<std::vector<Detokenizer::Word>> Detokenize(
      const std::vector<SpeechRecognizer::DecodedToken>& tokens);

  // Maps each word in [0, num_words) to the index of its starting token in
  // `token_ids` in O(N) time using cached non-initial token pieces, or returns
  // std::nullopt if word-onset mapping cannot be computed.
  std::optional<std::vector<size_t>> MapWordsToOnsetTokenIndices(
      absl::Span<const int> token_ids, size_t num_words,
      bool skip_special_tokens);

  // Returns the cached non-initial decoded piece for `token_id`. The returned
  // string_view points into `token_piece_cache_` and is valid only until the
  // next insertion into `token_piece_cache_`.
  absl::StatusOr<absl::string_view> GetDecodedTokenPiece(
      int token_id, bool skip_special_tokens);

  ::litert::support::Tokenizer& tokenizer_;
  // Caches each token ID's non-initial decoded rendering (`d3.substr(d2.size())`
  // from decoding `{id, id}` and `{id, id, id}`), preserving the leading space
  // (`' '`) that SentencePiece (`▁`) and HuggingFace Metaspace (`Ġ`) strip from
  // the first token of a decoded sequence.
  // For example, for the 8 tokens of "unfortunately it is open":
  //   id("▁un") -> " un", id("for") -> "for", id("tu") -> "tu",
  //   id("nate") -> "nate", id("ly") -> "ly", id("▁it") -> " it",
  //   id("▁is") -> " is", id("▁open") -> " open".
  // Keyed by `(token_id, skip_special_tokens)` and bounded by the number of
  // distinct token IDs seen across the session (realistically a few thousand
  // entries).
  absl::flat_hash_map<std::pair<int, bool>, std::string> token_piece_cache_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_TOKENIZER_DETOKENIZER_H_
