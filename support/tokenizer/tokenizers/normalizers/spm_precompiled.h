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
// C++ port of `src/lib.rs` from the `spm_precompiled` crate v0.1.4
// (https://github.com/huggingface/spm_precompiled, Apache License 2.0,
// authored by Nicolas Patry for the HuggingFace `tokenizers` crate, where it
// is re-exported as `tokenizers::normalizers::Precompiled`). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// This crate aims to emulate https://github.com/google/sentencepiece
// Dart::DoubleArray struct and it's Normalizer. It's main intent is to be used
// with tokenizers that is a Rust library that aims to provide facilities to
// tokenize string for use with HuggingFace's transformers library
//
// This crate is highly specialized and not intended for general use.
//
// The core of the algorithm is to read spm's binary `precompiled_charsmap`.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_SPM_PRECOMPILED_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_SPM_PRECOMPILED_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::normalizers {

// `pub type ArrayUnit = usize;`
using ArrayUnit = size_t;

// `trait ArrayUnitTrait` implemented for `ArrayUnit`.
inline bool ArrayUnitHasLeaf(ArrayUnit unit) { return ((unit >> 8) & 1) == 1; }
inline int64_t ArrayUnitValue(ArrayUnit unit) {
  return static_cast<int64_t>(unit & ((ArrayUnit{1} << 31) - 1));
}
inline size_t ArrayUnitLabel(ArrayUnit unit) {
  return unit & ((ArrayUnit{1} << 31) | 0xFF);
}
inline size_t ArrayUnitOffset(ArrayUnit unit) {
  return (unit >> 10) << ((unit & (ArrayUnit{1} << 9)) >> 6);
}

// `type Array = Vec<ArrayUnit>;`
using Array = std::vector<ArrayUnit>;

// `#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq)]`
class DoubleArray {
 public:
  DoubleArray() = default;
  // `fn from(array: Array) -> Self`
  explicit DoubleArray(Array array) : array_(std::move(array)) {}

  std::vector<int64_t> CommonPrefixSearch(absl::Span<const uint8_t> key) const;
  std::vector<int64_t> CommonPrefixSearch(absl::string_view key) const {
    return CommonPrefixSearch(absl::Span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(key.data()), key.size()));
  }

  const Array& array() const { return array_; }

  bool operator==(const DoubleArray& other) const {
    return array_ == other.array_;
  }
  bool operator!=(const DoubleArray& other) const { return !(*this == other); }

 private:
  Array array_;
};

// This struct is specifically done to be compatible with SentencePiece
// SentencePiece models embed their Normalizer within a `precompiled_charsmap`
// that both represents a Trie, and embedded rewrite rules.
// In order to be 100% compliant we need to interpret that binary format too.
// The format is [u32 (length of trie), trie: [u32], normalized: String]
// The trie has u8 as entries, and u32 as values, those u32 values
// point to offsets withing the String that correspond to the real replace value
// The normalized string contains '\0' that should indicate the end of an entry.
//
// Hence, normalized could be "abc\0", some entry in the trie could be 0 meaning
// the value is "abc" and another one be 1 meaning the actual entry was "bc".
//
// `#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
// #[serde(tag = "type", try_from = "PrecompiledDeserializer")]`
//
// The `Normalizer` implementation (`impl Normalizer for Precompiled` in the
// tokenizers crate's `src/normalizers/precompiled.rs`) lives in
// precompiled.cc, next to this file.
class Precompiled : public Normalizer {
 public:
  // `impl Default`
  Precompiled() = default;

  // `pub fn from(precompiled_charsmap: &[u8]) -> Result<Precompiled,
  // PrecompiledError>`; `precompiled_charsmap` holds raw bytes.
  // PORT NOTE: both `PrecompiledError` variants (`ParseError`,
  // `NormalizedInvalidUtf8`) display as "Cannot parse precompiled_charsmap",
  // which is the InvalidArgumentError returned here.
  static absl::StatusOr<Precompiled> From(
      absl::string_view precompiled_charsmap);

  // `pub fn transform(&self, chunk: &str) -> Option<&str>`: the returned view
  // points into this object.
  std::optional<absl::string_view> Transform(absl::string_view chunk) const;

  // `pub fn normalize_string(&self, original: &str) -> String`
  std::string NormalizeString(absl::string_view original) const;

  // Serde: `{"type":"Precompiled","precompiled_charsmap":"<base64>"}`.
  static absl::StatusOr<Precompiled> FromJson(const Json& json);
  Json ToJson() const;

  // `impl Normalizer for Precompiled` (src/normalizers/precompiled.rs).
  absl::Status Normalize(NormalizedString& normalized) const override;

  // `#[derive(PartialEq)]`
  bool operator==(const Precompiled& other) const {
    return precompiled_charsmap_ == other.precompiled_charsmap_ &&
           normalized_ == other.normalized_ && trie_ == other.trie_;
  }
  bool operator!=(const Precompiled& other) const { return !(*this == other); }

  // Field accessors (the Rust fields are private to the crate; the crate's own
  // tests read `normalized` and `trie`).
  const std::string& precompiled_charsmap() const {
    return precompiled_charsmap_;
  }
  const std::string& normalized() const { return normalized_; }
  const DoubleArray& trie() const { return trie_; }

 private:
  Precompiled(std::string precompiled_charsmap, std::string normalized,
              DoubleArray trie)
      : precompiled_charsmap_(std::move(precompiled_charsmap)),
        normalized_(std::move(normalized)),
        trie_(std::move(trie)) {}

  // `#[serde(serialize_with = "as_base64", deserialize_with = "from_base64")]`
  std::string precompiled_charsmap_;
  // `#[serde(skip)]`
  std::string normalized_;
  // `#[serde(skip)]`
  DoubleArray trie_;
};

}  // namespace tokenizers::normalizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_SPM_PRECOMPILED_H_
