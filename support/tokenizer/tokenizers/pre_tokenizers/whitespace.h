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
// C++ port of `src/pre_tokenizers/whitespace.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_WHITESPACE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_WHITESPACE_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace pre_tokenizers {

// Splits on the inverse of the regex `\w+|[^\w\s]+` (i.e. keeps the words and
// the punctuation runs, removes the whitespace).
//
// Unit struct; serde (`impl_serde_type!`): `{"type":"Whitespace"}`.
class Whitespace : public PreTokenizer {
 public:
  Whitespace() = default;

  absl::Status PreTokenize(PreTokenizedString& pretokenized) const override;

  static absl::StatusOr<Whitespace> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const Whitespace&, const Whitespace&) { return true; }
  friend bool operator!=(const Whitespace&, const Whitespace&) { return false; }
};

// Splits on `char::is_whitespace`, removing the whitespace.
//
// Unit struct; serde (`impl_serde_type!`): `{"type":"WhitespaceSplit"}`.
class WhitespaceSplit : public PreTokenizer {
 public:
  WhitespaceSplit() = default;

  absl::Status PreTokenize(PreTokenizedString& pretokenized) const override;

  static absl::StatusOr<WhitespaceSplit> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const WhitespaceSplit&, const WhitespaceSplit&) {
    return true;
  }
  friend bool operator!=(const WhitespaceSplit&, const WhitespaceSplit&) {
    return false;
  }
};

}  // namespace pre_tokenizers
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_WHITESPACE_H_
