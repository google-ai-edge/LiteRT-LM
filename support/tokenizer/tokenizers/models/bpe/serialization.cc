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
// C++ port of `src/models/bpe/serialization.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: `impl Serialize for BPE` / `impl Deserialize for BPE` are the
// `BPE::ToJson` / `BPE::FromJson` members declared in model.h.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/bpe.h"
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::models::bpe {

Json BPE::ToJson() const {
  Json model = Json::object();

  // Start by small fields
  model["type"] = "BPE";
  model["dropout"] =
      dropout_.has_value() ? F32ToJson(*dropout_) : Json(nullptr);
  model["unk_token"] =
      unk_token_.has_value() ? Json(*unk_token_) : Json(nullptr);
  model["continuing_subword_prefix"] = continuing_subword_prefix_.has_value()
                                           ? Json(*continuing_subword_prefix_)
                                           : Json(nullptr);
  model["end_of_word_suffix"] = end_of_word_suffix_.has_value()
                                    ? Json(*end_of_word_suffix_)
                                    : Json(nullptr);
  model["fuse_unk"] = fuse_unk_;
  model["byte_fallback"] = byte_fallback_;
  model["ignore_merges"] = ignore_merges_;

  // Then the large ones
  std::vector<std::pair<const Pair*, uint32_t>> merges;
  merges.reserve(merges_.size());
  for (const auto& [pair, rank_and_id] : merges_) {
    merges.push_back({&pair, rank_and_id.first});
  }
  std::sort(merges.begin(), merges.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  Json merges_json = Json::array();
  for (const auto& [pair, rank] : merges) {
    // Every merge id comes from the vocab, so both lookups succeed.
    auto a = vocab_r_.find(pair->first);
    auto b = vocab_r_.find(pair->second);
    merges_json.push_back(
        Json::array({a == vocab_r_.end() ? std::string() : a->second,
                     b == vocab_r_.end() ? std::string() : b->second}));
  }
  const OrderedVocabIter ordered_vocab(vocab_r_);

  model["vocab"] = ordered_vocab.Serialize();
  model["merges"] = std::move(merges_json);

  return model;
}

namespace {

// `#[serde(untagged)] enum MergeType { Tuple(Vec<(String, String)>),
// Legacy(Vec<String>) }`
struct MergeType {
  std::variant<Merges, std::vector<std::string>> value;
};

absl::StatusOr<MergeType> MergeTypeFromJson(const Json& json) {
  if (json.is_array()) {
    // Tuple(Vec<(String, String)>)
    bool is_tuple = true;
    Merges tuple;
    for (const Json& item : json) {
      if (!item.is_array() || item.size() != 2 || !item[0].is_string() ||
          !item[1].is_string()) {
        is_tuple = false;
        break;
      }
      tuple.push_back({item[0].get<std::string>(), item[1].get<std::string>()});
    }
    if (is_tuple) {
      return MergeType{std::move(tuple)};
    }
    // Legacy(Vec<String>)
    bool is_legacy = true;
    std::vector<std::string> legacy;
    for (const Json& item : json) {
      if (!item.is_string()) {
        is_legacy = false;
        break;
      }
      legacy.push_back(item.get<std::string>());
    }
    if (is_legacy) {
      return MergeType{std::move(legacy)};
    }
  }
  return absl::InvalidArgumentError(
      "data did not match any variant of untagged enum MergeType");
}

}  // namespace

absl::StatusOr<BPE> BPE::FromJson(const Json& json) {
  if (!json.is_object()) {
    return SerdeInvalidType(json, "struct BPE");
  }
  BpeBuilder builder;
  std::optional<Vocab> vocab;

  std::optional<MergeType> merges;
  for (const auto& [key, value] : json.items()) {
    if (key == "dropout") {
      if (!value.is_null()) {
        ASSIGN_OR_RETURN(const float dropout, F32FromJson(value));
        builder.Dropout(dropout);
      }
    } else if (key == "unk_token") {
      if (!value.is_null()) {
        ASSIGN_OR_RETURN(std::string unk, StringFromJson(value));
        builder.UnkToken(std::move(unk));
      }
    } else if (key == "continuing_subword_prefix") {
      if (!value.is_null()) {
        ASSIGN_OR_RETURN(std::string prefix, StringFromJson(value));
        builder.ContinuingSubwordPrefix(std::move(prefix));
      }
    } else if (key == "end_of_word_suffix") {
      if (!value.is_null()) {
        ASSIGN_OR_RETURN(std::string suffix, StringFromJson(value));
        builder.EndOfWordSuffix(std::move(suffix));
      }
    } else if (key == "fuse_unk") {
      if (!value.is_null()) {
        ASSIGN_OR_RETURN(const bool suffix, BoolFromJson(value));
        builder.FuseUnk(suffix);
      }
    } else if (key == "byte_fallback") {
      if (!value.is_null()) {
        ASSIGN_OR_RETURN(const bool suffix, BoolFromJson(value));
        builder.ByteFallback(suffix);
      }
    } else if (key == "ignore_merges") {
      if (!value.is_null()) {
        ASSIGN_OR_RETURN(const bool suffix, BoolFromJson(value));
        builder.IgnoreMerges(suffix);
      }
    } else if (key == "vocab") {
      ASSIGN_OR_RETURN(vocab, VocabFromJson(value));
    } else if (key == "merges") {
      ASSIGN_OR_RETURN(merges, MergeTypeFromJson(value));
    } else if (key == "type") {
      if (!value.is_string()) {
        return SerdeInvalidType(value, "a borrowed string");
      }
      if (value.get_ref<const std::string&>() != "BPE") {
        return SerdeInvalidValue(value, "BPE");
      }
    }
  }
  if (vocab.has_value() && merges.has_value()) {
    Merges merge_list;
    if (auto* tuple = std::get_if<Merges>(&merges->value)) {
      merge_list = std::move(*tuple);
    } else {
      ASSIGN_OR_RETURN(
          merge_list,
          ConvertMergesToHashmap(
              std::get<std::vector<std::string>>(merges->value), *vocab));
    }
    builder.VocabAndMerges(*std::move(vocab), std::move(merge_list));
    return builder.Build();
  } else {
    return absl::InvalidArgumentError("Missing vocab/merges");
  }
}

}  // namespace tokenizers::models::bpe
