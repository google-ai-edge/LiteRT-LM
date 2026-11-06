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
// C++ port of `src/pre_tokenizers/metaspace.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_METASPACE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_METASPACE_H_

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace pre_tokenizers {

// Enum representing options for the metaspace prepending scheme.
// serde: `#[serde(rename_all = "snake_case")]` -> "first" / "never" / "always".
enum class PrependScheme {
  // Specifies that the scheme should be prepended only once, on the first
  // split.
  kFirst,
  // Specifies that the space should not be prepended.
  kNever,
  // Specifies that the scheme should always be prepended.
  kAlways,
};

// `impl std::fmt::Display for PrependScheme` (= its serde representation).
absl::string_view PrependSchemeToString(PrependScheme scheme);
absl::StatusOr<PrependScheme> PrependSchemeFromJson(const Json& json);
Json PrependSchemeToJson(PrependScheme scheme);

// Replaces all the whitespaces by the provided meta character and then
// splits on this character
//
// serde: `{"type":"Metaspace","replacement":"▁","prepend_scheme":"always",
// "split":true}`; the custom Deserialize also accepts the legacy
// `add_prefix_space` / `str_rep` fields (see `FromJson`).
class Metaspace : public PreTokenizer, public Decoder {
 public:
  // `impl Default for Metaspace`: `Self::new('▁', PrependScheme::Always,
  // true)`.
  Metaspace() : Metaspace(U'\u2581', PrependScheme::kAlways, /*split=*/true) {}
  // `Metaspace::new`.
  Metaspace(char32_t replacement, PrependScheme prepend_scheme, bool split);

  char32_t GetReplacement() const { return replacement_; }
  void SetReplacement(char32_t replacement);
  bool GetSplit() const { return split_; }
  void SetSplit(bool split) { split_ = split; }
  PrependScheme GetPrependScheme() const { return prepend_scheme_; }
  void SetPrependScheme(PrependScheme scheme) { prepend_scheme_ = scheme; }

  // `impl PreTokenizer for Metaspace`.
  absl::Status PreTokenize(PreTokenizedString& pretokenized) const override;

  // `impl Decoder for Metaspace`.
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

  static absl::StatusOr<Metaspace> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const Metaspace& a, const Metaspace& b) {
    return a.replacement_ == b.replacement_ &&
           a.prepend_scheme_ == b.prepend_scheme_ && a.split_ == b.split_ &&
           a.str_rep_ == b.str_rep_;
  }
  friend bool operator!=(const Metaspace& a, const Metaspace& b) {
    return !(a == b);
  }

 private:
  char32_t replacement_;
  PrependScheme prepend_scheme_;
  bool split_;
  // `#[serde(skip)]`: the UTF-8 encoding of `replacement_`.
  std::string str_rep_;
};

}  // namespace pre_tokenizers
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_METASPACE_H_
