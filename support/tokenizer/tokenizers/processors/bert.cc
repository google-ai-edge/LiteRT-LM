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
// C++ port of `src/processors/bert.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/processors/bert.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace processors {

namespace {

// Appends `tail` to a copy of `head` (`[&head[..], &tail[..]].concat()`).
template <typename T>
std::vector<T> Concat(std::vector<T> head, const std::vector<T>& tail) {
  head.insert(head.end(), tail.begin(), tail.end());
  return head;
}

// The body of the `i == 0` branch of `process_encodings`, applied to both the
// encoding itself and each of its overflowing encodings (the Rust code spells
// it out twice).
Encoding ProcessFirst(const Encoding& encoding,
                      const std::pair<std::string, uint32_t>& cls,
                      const std::pair<std::string, uint32_t>& sep,
                      std::vector<Encoding> overflowing) {
  std::vector<uint32_t> ids = Concat(Concat({cls.second}, encoding.GetIds()),
                                     std::vector<uint32_t>{sep.second});
  std::vector<uint32_t> type_ids =
      Concat(Concat({0}, encoding.GetTypeIds()), std::vector<uint32_t>{0});
  std::vector<std::string> tokens =
      Concat(Concat({cls.first}, encoding.GetTokens()),
             std::vector<std::string>{sep.first});
  std::vector<std::optional<uint32_t>> words =
      Concat(Concat({std::nullopt}, encoding.GetWordIds()),
             std::vector<std::optional<uint32_t>>{std::nullopt});
  std::vector<Offsets> offsets =
      Concat(Concat({Offsets(0, 0)}, encoding.GetOffsets()),
             std::vector<Offsets>{Offsets(0, 0)});
  std::vector<uint32_t> special_tokens =
      Concat(Concat({1u}, std::vector<uint32_t>(encoding.GetIds().size(), 0)),
             std::vector<uint32_t>{1});
  std::vector<uint32_t> attention_mask(ids.size(), 1);

  // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't
  // contain the special tokens.
  absl::flat_hash_map<size_t, SequenceRange> sequence_ranges = {
      {0, SequenceRange(1, ids.size() - 1)}};
  return Encoding(std::move(ids), std::move(type_ids), std::move(tokens),
                  std::move(words), std::move(offsets),
                  std::move(special_tokens), std::move(attention_mask),
                  std::move(overflowing), std::move(sequence_ranges));
}

// The body of the `else` (pair) branch of `process_encodings`, applied to
// both the encoding itself and each of its overflowing encodings.
Encoding ProcessPair(const Encoding& encoding,
                     const std::pair<std::string, uint32_t>& sep,
                     std::vector<Encoding> overflowing) {
  std::vector<uint32_t> pair_ids =
      Concat(encoding.GetIds(), std::vector<uint32_t>{sep.second});
  std::vector<uint32_t> pair_type_ids =
      Concat(encoding.GetTypeIds(), std::vector<uint32_t>{1});
  std::vector<std::string> pair_tokens =
      Concat(encoding.GetTokens(), std::vector<std::string>{sep.first});
  std::vector<std::optional<uint32_t>> pair_words =
      Concat(encoding.GetWordIds(),
             std::vector<std::optional<uint32_t>>{std::nullopt});
  std::vector<Offsets> pair_offsets =
      Concat(encoding.GetOffsets(), std::vector<Offsets>{Offsets(0, 0)});
  std::vector<uint32_t> pair_special_tokens =
      Concat(std::vector<uint32_t>(encoding.GetTypeIds().size(), 0),
             std::vector<uint32_t>{1});
  std::vector<uint32_t> pair_attention_mask(pair_ids.size(), 1);

  // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't
  // contain the special tokens.
  absl::flat_hash_map<size_t, SequenceRange> pair_sequence_ranges = {
      {1, SequenceRange(0, pair_ids.size() - 1)}};
  return Encoding(std::move(pair_ids), std::move(pair_type_ids),
                  std::move(pair_tokens), std::move(pair_words),
                  std::move(pair_offsets), std::move(pair_special_tokens),
                  std::move(pair_attention_mask), std::move(overflowing),
                  std::move(pair_sequence_ranges));
}

}  // namespace

size_t BertProcessing::AddedTokens(bool is_pair) const {
  if (is_pair) {
    return 3;
  } else {
    return 2;
  }
}

absl::StatusOr<std::vector<Encoding>> BertProcessing::ProcessEncodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {
  if (!add_special_tokens) {
    return encodings;
  }

  std::vector<Encoding> result;
  result.reserve(encodings.size());
  for (size_t i = 0; i < encodings.size(); ++i) {
    Encoding& encoding = encodings[i];
    if (i == 0) {
      std::vector<Encoding> overflowing;
      for (const Encoding& overflow : encoding.TakeOverflowing()) {
        overflowing.push_back(ProcessFirst(overflow, cls_, sep_, {}));
      }
      result.push_back(
          ProcessFirst(encoding, cls_, sep_, std::move(overflowing)));
    } else {
      std::vector<Encoding> overflowing;
      for (const Encoding& overflow : encoding.TakeOverflowing()) {
        overflowing.push_back(ProcessPair(overflow, sep_, {}));
      }
      result.push_back(ProcessPair(encoding, sep_, std::move(overflowing)));
    }
  }

  return result;
}

absl::StatusOr<BertProcessing> BertProcessing::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` on a struct: serde does not validate the tag
  // when deserializing, the `type` entry is simply ignored.
  RETURN_IF_ERROR(RequireObject(json, "struct BertProcessing"));
  using SpecialToken = std::pair<std::string, uint32_t>;
  ASSIGN_OR_RETURN(SpecialToken sep, GetField<SpecialToken>(json, "sep"));
  ASSIGN_OR_RETURN(SpecialToken cls, GetField<SpecialToken>(json, "cls"));
  return BertProcessing(std::move(sep), std::move(cls));
}

Json BertProcessing::ToJson() const {
  Json json = Json::object();
  json["type"] = "BertProcessing";
  json["sep"] = Json::array({sep_.first, sep_.second});
  json["cls"] = Json::array({cls_.first, cls_.second});
  return json;
}

}  // namespace processors
}  // namespace tokenizers
