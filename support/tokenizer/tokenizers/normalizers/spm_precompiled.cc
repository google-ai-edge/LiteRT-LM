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

#include "support/tokenizer/tokenizers/normalizers/spm_precompiled.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/unicode/graphemes.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {
namespace {

// `impl std::fmt::Display for PrecompiledError`: both `ParseError` and
// `NormalizedInvalidUtf8` print the same text.
absl::Status PrecompiledError() {
  return absl::InvalidArgumentError("Cannot parse precompiled_charsmap");
}

// `nom::number::complete::le_u32`
bool LeU32(absl::string_view* input, uint32_t* out) {
  if (input->size() < 4) return false;
  const auto* bytes = reinterpret_cast<const uint8_t*>(input->data());
  *out = static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
  input->remove_prefix(4);
  return true;
}

// `fn parse(precompiled_charsmap: &[u8]) -> IResult<&[u8], Array>`: returns
// (normalized_blob, trie_blob) or nullopt on a nom error.
std::optional<std::pair<absl::string_view, Array>> Parse(
    absl::string_view precompiled_charsmap) {
  absl::string_view rest = precompiled_charsmap;
  uint32_t trie_size;
  if (!LeU32(&rest, &trie_size)) return std::nullopt;
  // u8 to u32.
  uint32_t trie_char_size = trie_size / 4;
  // (Reject a truncated blob before reserving, instead of failing part-way
  // through the loop like nom does; the outcome is the same error.)
  if (rest.size() / 4 < trie_char_size) return std::nullopt;
  Array trie_blob;
  trie_blob.reserve(trie_char_size);
  for (uint32_t i = 0; i < trie_char_size; ++i) {
    uint32_t n;
    if (!LeU32(&rest, &n)) return std::nullopt;
    trie_blob.push_back(static_cast<ArrayUnit>(n));
  }
  absl::string_view normalized_blob = rest;
  return std::make_pair(normalized_blob, std::move(trie_blob));
}

}  // namespace

std::vector<int64_t> DoubleArray::CommonPrefixSearch(
    absl::Span<const uint8_t> key) const {
  size_t node_pos = 0;
  std::vector<int64_t> results;

  // PORT NOTE: the Rust code indexes `self.array[node_pos]` unchecked and
  // panics on an empty or corrupt trie (e.g. `Precompiled::default()`); the
  // bounds checks below end the search instead, returning what was found.
  if (node_pos >= array_.size()) return results;
  ArrayUnit unit = array_[node_pos];
  node_pos ^= ArrayUnitOffset(unit);
  for (uint8_t c : key) {
    if (c == 0) {
      break;
    }
    node_pos ^= static_cast<size_t>(c);
    if (node_pos >= array_.size()) return results;
    unit = array_[node_pos];
    if (ArrayUnitLabel(unit) != static_cast<size_t>(c)) {
      return results;
    }
    node_pos ^= ArrayUnitOffset(unit);
    if (ArrayUnitHasLeaf(unit)) {
      if (node_pos >= array_.size()) return results;
      results.push_back(ArrayUnitValue(array_[node_pos]));
    }
  }
  return results;
}

absl::StatusOr<Precompiled> Precompiled::From(
    absl::string_view precompiled_charsmap) {
  std::optional<std::pair<absl::string_view, Array>> parsed =
      Parse(precompiled_charsmap);
  if (!parsed.has_value()) return PrecompiledError();  // ParseError
  auto& [normalized_blob, trie_blob] = *parsed;
  if (!IsValidUtf8(normalized_blob)) {
    return PrecompiledError();  // NormalizedInvalidUtf8
  }
  std::string normalized(normalized_blob);
  DoubleArray trie(std::move(trie_blob));
  Precompiled precompiled(std::string(precompiled_charsmap),
                          std::move(normalized), std::move(trie));
  return precompiled;
}

std::optional<absl::string_view> Precompiled::Transform(
    absl::string_view chunk) const {
  std::vector<int64_t> results = trie_.CommonPrefixSearch(chunk);
  if (results.empty()) {
    return std::nullopt;
  } else {
    size_t index = static_cast<size_t>(results[0]);
    size_t index2 = index;
    while (index2 < normalized_.size()) {
      if (static_cast<uint8_t>(normalized_[index2]) == 0) {
        break;
      }
      index2 += 1;
    }
    // PORT NOTE: `&self.normalized[index..index2]` panics when `index` lies
    // beyond the end of `normalized` (a corrupt charsmap); return None instead.
    if (index > normalized_.size()) return std::nullopt;
    absl::string_view normalized =
        absl::string_view(normalized_).substr(index, index2 - index);
    return normalized;
  }
}

std::string Precompiled::NormalizeString(absl::string_view original) const {
  std::string string;
  string.reserve(original.size());
  // Future reader. From @Narsil.
  // Yes, this is weird,
  // Yes, this seems broken
  // No, I don't know why Google did this.
  // If you question this code, check this normalizer against
  // XNLI database (all languages) with Unigram model against
  // Mbart, XLMRoberta *AND* Marian. If you don't get 100% or
  // break a single test.
  // You don't pass.
  for (absl::string_view grapheme : unicode::Graphemes(original, true)) {
    if (grapheme.size() < 6) {
      if (std::optional<absl::string_view> norm = Transform(grapheme);
          norm.has_value()) {
        string.append(norm->data(), norm->size());
        continue;
      }
    }
    for (const auto& [char_index, c] : CharIndices(grapheme)) {
      absl::string_view part = grapheme.substr(char_index, Utf8Len(c));
      if (std::optional<absl::string_view> norm = Transform(part);
          norm.has_value()) {
        string.append(norm->data(), norm->size());
      } else {
        string.append(part.data(), part.size());
      }
    }
  }
  return string;
}

absl::StatusOr<Precompiled> Precompiled::FromJson(const Json& json) {
  // `struct PrecompiledDeserializer { #[serde(deserialize_with =
  // "from_base64")] precompiled_charsmap: Vec<u8> }` (`#[serde(tag = "type")]`,
  // tag not validated), then `TryFrom<PrecompiledDeserializer>` =
  // `Precompiled::from`.
  RETURN_IF_ERROR(RequireObject(json, "struct PrecompiledDeserializer"));
  const Json* charsmap = FindField(json, "precompiled_charsmap");
  if (charsmap == nullptr) return MissingFieldError("precompiled_charsmap");
  // `from_base64` deserializes a `&str`.
  if (!charsmap->is_string()) {
    return InvalidTypeError(*charsmap, "a borrowed string");
  }
  std::string precompiled_charsmap;
  if (!absl::Base64Unescape(charsmap->get_ref<const std::string&>(),
                            &precompiled_charsmap)) {
    // PORT NOTE: Rust forwards the `base64::DecodeError` text ("Invalid
    // symbol ...", "Invalid padding", ...); absl does not report details.
    return absl::InvalidArgumentError(
        "Invalid base64 in field `precompiled_charsmap`");
  }
  return From(precompiled_charsmap);
}

Json Precompiled::ToJson() const {
  Json json = Json::object();
  json["type"] = "Precompiled";
  // `as_base64`: standard alphabet with padding.
  json["precompiled_charsmap"] = absl::Base64Escape(precompiled_charsmap_);
  return json;
}

}  // namespace tokenizers::normalizers
