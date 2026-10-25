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
// C++ port of `src/tokenizer/pre_tokenizer.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PRE_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PRE_TOKENIZER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"

namespace tokenizers {

// Various possible types of offsets
//
// `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum OffsetType`
enum class OffsetType { kByte, kChar, kNone };

// Wrapper for a subpart of a `NormalizedString`.
//
// This Split contains the underlying `NormalizedString` as well as its offsets
// in the original string. These offsets are in the `original` referential.
// It also contains any `Token` associated to the current split
//
// `#[derive(Debug, Clone, PartialEq, Eq)] pub struct Split`
struct Split {
  // The underlying `NormalizedString`. Each SubString is represented by a
  // `NormalizedString` and in the end we might be carrying a lot of SubString
  // representing various parts of the original input string.
  NormalizedString normalized;
  // Optional Tokens associated to this Split
  std::optional<std::vector<Token>> tokens;

  Split() = default;
  // `impl From<NormalizedString> for Split`
  Split(NormalizedString n)  // NOLINT(google-explicit-constructor)
      : normalized(std::move(n)) {}
  // `impl From<(NormalizedString, Option<Vec<Token>>)> for Split`
  Split(NormalizedString n, std::optional<std::vector<Token>> t)
      : normalized(std::move(n)), tokens(std::move(t)) {}

  friend bool operator==(const Split& a, const Split& b) {
    return a.normalized == b.normalized && a.tokens == b.tokens;
  }
  friend bool operator!=(const Split& a, const Split& b) { return !(a == b); }
};

// The `PreTokenizedString` is in charge of splitting an underlying string,
// making sure everything is fine while doing so, and providing ways to
// normalize and tokenize these splits.
// Once everything has been normalized and tokenized, the `PreTokenizedString`
// is able to build an `Encoding` with all the relevant offsets and word ids,
// relative to the original string.
//
// `#[derive(Debug, Clone, PartialEq, Eq)]`
class PreTokenizedString {
 public:
  // `impl From<&str> for PreTokenizedString` / `impl From<String>`
  static PreTokenizedString From(absl::string_view s);

  // `impl From<NormalizedString> for PreTokenizedString`
  explicit PreTokenizedString(NormalizedString s);

  // Split the `PreTokenizedString` by providing a `split_fn` in charge of
  // splitting each substring (`NormalizedString`) into multiple parts.
  //
  // `split_fn` takes a `NormalizedString` and is in charge of returning an
  // iterator over the produced `NormalizedString`. `split_fn` is free of
  // modifying these `NormalizedString` as relevant, as long as it respects the
  // constraint stated below.
  //
  // There are only one constraint that *MUST* be respected:
  // > The produced `NormalizedString`, if combined back together, must have
  // > the same `original` string as the original one given to `split_fn`. This
  // > concretely means that for the offset tracking to work as expected,
  // > `split_fn` must produce "splits" of the original string.
  //
  // PORT NOTE: the Rust `split_fn` may return any `IntoIterator<Item: Into<
  // Split>>`; two overloads are provided, for `Vec<Split>` and for
  // `Vec<NormalizedString>` (what `NormalizedString::split` returns).
  absl::Status Split(
      absl::FunctionRef<absl::StatusOr<std::vector<tokenizers::Split>>(
          size_t, NormalizedString)>
          split_fn);
  absl::Status Split(
      absl::FunctionRef<absl::StatusOr<std::vector<NormalizedString>>(
          size_t, NormalizedString)>
          split_fn);

  // Normalized all the splits that do not have attached `Tokens`, using the
  // provided `normalize` function.
  absl::Status Normalize(
      absl::FunctionRef<absl::Status(NormalizedString&)> normalize);

  // Tokenize all the splits that do not have attached `Tokens`, using the
  // provided `tokenize` function
  absl::Status Tokenize(absl::FunctionRef<absl::StatusOr<std::vector<Token>>(
                            const NormalizedString&)>
                            tokenize);

  // Early exits when `max_tokens` have been produced, `TruncationDirection`
  // aware.
  //
  // `pub fn tokenize_with_limit<F>(&mut self, tokenize, max_tokens, direction)`
  absl::Status TokenizeWithLimit(
      absl::FunctionRef<
          absl::StatusOr<std::vector<Token>>(const NormalizedString&)>
          tokenize,
      size_t max_tokens, TruncationDirection direction);

  // Transform the current `PreTokenizedString` into an `Encoding`.
  //
  // If a `word_idx` is provided, any word in the generated `Encoding`
  // will be set to this value. This is generally used with pre-tokenized
  // input, that do not need the `PreTokenizedString` to generate word ids.
  //
  // This method will fail if some splits do not have associated `Token`.
  absl::StatusOr<Encoding> IntoEncoding(std::optional<uint32_t> word_idx,
                                        uint32_t type_id,
                                        OffsetType offset_type) &&;

  // Returns a list of splits, each of them being a slice of the normalized
  // string, the associated offsets either in original or normalized
  // referential, as well as the potention tokens
  std::vector<std::tuple<absl::string_view, Offsets,
                         const std::optional<std::vector<Token>>*>>
  GetSplits(OffsetReferential offset_ref, OffsetType offset_type) const;

  // The `splits` field (exposed for tests and pre-tokenizers).
  const std::vector<tokenizers::Split>& Splits() const { return splits_; }
  std::vector<tokenizers::Split>& MutableSplits() { return splits_; }

  friend bool operator==(const PreTokenizedString& a,
                         const PreTokenizedString& b) {
    return a.original_ == b.original_ && a.splits_ == b.splits_;
  }
  friend bool operator!=(const PreTokenizedString& a,
                         const PreTokenizedString& b) {
    return !(a == b);
  }

 private:
  std::string original_;
  std::vector<tokenizers::Split> splits_;
};

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PRE_TOKENIZER_H_
