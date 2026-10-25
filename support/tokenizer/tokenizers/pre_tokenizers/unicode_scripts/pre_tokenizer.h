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
// C++ port of `src/pre_tokenizers/unicode_scripts/pre_tokenizer.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_UNICODE_SCRIPTS_PRE_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_UNICODE_SCRIPTS_PRE_TOKENIZER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/unicode_scripts/scripts.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace pre_tokenizers {
namespace unicode_scripts {

// Splits on changes of Unicode script (Hiragana/Katakana count as Han, spaces
// belong to any script).
//
// Unit struct; serde (`impl_serde_type!`): `{"type":"UnicodeScripts"}`.
class UnicodeScripts : public PreTokenizer {
 public:
  // `UnicodeScripts::new` / `impl Default`.
  UnicodeScripts() = default;

  absl::Status PreTokenize(PreTokenizedString& pretokenized) const override;

  static absl::StatusOr<UnicodeScripts> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const UnicodeScripts&, const UnicodeScripts&) {
    return true;
  }
  friend bool operator!=(const UnicodeScripts&, const UnicodeScripts&) {
    return false;
  }
};

// This code exists in the Unigram default IsValidSentencePiece.
// It could be integrated directly within `get_script` but I
// think it's kind of tricky to see those modifications later
// I am guessing release mode will optimize this away anyway.
//
// (Private in Rust; exposed here for the tests of pre_tokenizer.rs.)
Script FixedScript(char32_t c);

}  // namespace unicode_scripts
}  // namespace pre_tokenizers
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_UNICODE_SCRIPTS_PRE_TOKENIZER_H_
