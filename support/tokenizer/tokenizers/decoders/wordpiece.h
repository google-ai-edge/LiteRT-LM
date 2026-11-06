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
//
// C++ port of `src/decoders/wordpiece.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_WORDPIECE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_WORDPIECE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace decoders {

// The WordPiece decoder takes care of decoding a list of wordpiece tokens
// back into a readable string.
//
// serde: `#[serde(tag = "type")]` ->
// `{"type":"WordPiece","prefix":"##","cleanup":true}`.
class WordPiece : public Decoder {
 public:
  // `impl Default for WordPiece`.
  WordPiece() : WordPiece("##", /*cleanup=*/true) {}
  // `WordPiece::new`.
  WordPiece(std::string prefix, bool cleanup)
      : prefix_(std::move(prefix)), cleanup_(cleanup) {}

  // The prefix to be used for continuing subwords
  const std::string& prefix() const { return prefix_; }
  void set_prefix(std::string prefix) { prefix_ = std::move(prefix); }
  // Whether to cleanup some tokenization artifacts (spaces before
  // punctuation, ...)
  bool cleanup() const { return cleanup_; }
  void set_cleanup(bool cleanup) { cleanup_ = cleanup; }

  // `impl Decoder for WordPiece`.
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

  static absl::StatusOr<WordPiece> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const WordPiece& a, const WordPiece& b) {
    return a.prefix_ == b.prefix_ && a.cleanup_ == b.cleanup_;
  }
  friend bool operator!=(const WordPiece& a, const WordPiece& b) {
    return !(a == b);
  }

 private:
  std::string prefix_;
  bool cleanup_;
};

// `pub fn cleanup(dirty_input: &str) -> String` (also used by `CTC`).
std::string Cleanup(absl::string_view dirty_input);

// Rust `str::replace(from, to)`: replaces all non-overlapping matches of
// `from` in `s` by `to`, scanning left to right. Like in Rust, an empty `from`
// matches at every character boundary (`"ab".replace("", "-") == "-a-b-"`).
std::string StrReplace(absl::string_view s, absl::string_view from,
                       absl::string_view to);

}  // namespace decoders
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_WORDPIECE_H_
