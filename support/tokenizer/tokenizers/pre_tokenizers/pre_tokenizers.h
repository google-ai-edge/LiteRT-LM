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
// C++ port of `src/pre_tokenizers/mod.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_PRE_TOKENIZERS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_PRE_TOKENIZERS_H_

#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/bert.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/delimiter.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/digits.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/fixed_length.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/metaspace.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/punctuation.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/sequence.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/split.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/unicode_scripts/unicode_scripts.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/pre_tokenizers/whitespace.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace pre_tokenizers {

// Rust: `#[serde(untagged)] pub enum PreTokenizerWrapper { ... }` with a
// custom `Deserialize` (see `FromJson`). The variants are held in a
// `std::variant` in the Rust declaration order; note that the variant named
// `Delimiter` holds a `CharDelimiterSplit`.
class PreTokenizerWrapper : public PreTokenizer {
 public:
  using Variant =
      std::variant<BertPreTokenizer, ByteLevel, CharDelimiterSplit, Metaspace,
                   Whitespace, Sequence, Split, Punctuation, WhitespaceSplit,
                   Digits, UnicodeScripts, FixedLength>;

  // `impl_enum_from!(...)`: implicit conversions from every concrete type.
  PreTokenizerWrapper(  // NOLINT(google-explicit-constructor)
      BertPreTokenizer v)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(ByteLevel v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(  // NOLINT(google-explicit-constructor)
      CharDelimiterSplit v)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(Metaspace v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(Whitespace v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(Sequence v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(Split v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(Punctuation v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(WhitespaceSplit v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(Digits v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(UnicodeScripts v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}
  PreTokenizerWrapper(FixedLength v)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(v)) {}

  // `impl PreTokenizer for PreTokenizerWrapper`.
  absl::Status PreTokenize(PreTokenizedString& normalized) const override;

  // `impl<'de> Deserialize<'de> for PreTokenizerWrapper`: first tries the
  // `type`-tagged form (`type` must be one of the 12 variant names, the rest
  // of the object is handed to that variant's `FromJson`), otherwise falls
  // back to the untagged form, trying every variant in declaration order and
  // failing with "data did not match any variant of untagged enum
  // PreTokenizerUntagged".
  static absl::StatusOr<PreTokenizerWrapper> FromJson(const Json& json);
  // `#[serde(untagged)]` Serialize: the inner value's JSON.
  Json ToJson() const;

  const Variant& variant() const { return variant_; }
  Variant& variant() { return variant_; }

  template <typename T>
  bool Is() const {
    return std::holds_alternative<T>(variant_);
  }
  template <typename T>
  const T& Get() const {
    return std::get<T>(variant_);
  }
  template <typename T>
  T& Get() {
    return std::get<T>(variant_);
  }

  friend bool operator==(const PreTokenizerWrapper& a,
                         const PreTokenizerWrapper& b) {
    return a.variant_ == b.variant_;
  }
  friend bool operator!=(const PreTokenizerWrapper& a,
                         const PreTokenizerWrapper& b) {
    return !(a == b);
  }

 private:
  Variant variant_;
};

}  // namespace pre_tokenizers
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_PRE_TOKENIZERS_H_
