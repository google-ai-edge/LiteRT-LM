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
// C++ port of `src/models/unigram/serialization.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: `impl Serialize for Unigram` / `impl Deserialize for Unigram`
// are the `Unigram::ToJson` / `Unigram::FromJson` members declared in model.h.

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/models/unigram/model.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::models::unigram {

Json Unigram::ToJson() const {
  Json model = Json::object();

  model["type"] = "Unigram";
  model["unk_id"] = unk_id_.has_value() ? Json(*unk_id_) : Json(nullptr);
  Json vocab_json = Json::array();
  for (const auto& [token, score] : vocab_) {
    vocab_json.push_back(Json::array({token, score}));
  }
  model["vocab"] = std::move(vocab_json);
  model["byte_fallback"] = byte_fallback();

  return model;
}

absl::StatusOr<Unigram> Unigram::FromJson(const Json& json) {
  if (!json.is_object()) {
    return SerdeInvalidType(json, "struct Unigram");
  }
  std::optional<std::vector<std::pair<std::string, double>>> vocab;
  std::optional<size_t> unk_id;
  bool byte_fallback = false;
  for (const auto& [key, value] : json.items()) {
    if (key == "unk_id") {
      if (value.is_null()) {
        unk_id = std::nullopt;
      } else {
        ASSIGN_OR_RETURN(const size_t id, UsizeFromJson(value));
        unk_id = id;
      }
    } else if (key == "byte_fallback") {
      ASSIGN_OR_RETURN(byte_fallback, BoolFromJson(value));
    } else if (key == "vocab") {
      if (!value.is_array()) {
        return SerdeInvalidType(value, "a sequence");
      }
      std::vector<std::pair<std::string, double>> items;
      items.reserve(value.size());
      for (const Json& entry : value) {
        if (!entry.is_array()) {
          return SerdeInvalidType(entry, "a tuple of size 2");
        }
        if (entry.size() != 2) {
          return absl::InvalidArgumentError(absl::StrCat(
              "invalid length ", entry.size(), ", expected a tuple of size 2"));
        }
        ASSIGN_OR_RETURN(std::string token, StringFromJson(entry[0]));
        ASSIGN_OR_RETURN(const double score, F64FromJson(entry[1]));
        items.emplace_back(std::move(token), score);
      }
      vocab = std::move(items);
    } else if (key == "type") {
      if (!value.is_string()) {
        return SerdeInvalidType(value, "a borrowed string");
      }
      if (value.get_ref<const std::string&>() != "Unigram") {
        return SerdeInvalidValue(value, "Unigram");
      }
    }
  }
  if (vocab.has_value()) {
    absl::StatusOr<Unigram> model =
        Unigram::From(*std::move(vocab), unk_id, byte_fallback);
    if (!model.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unable to load vocab ", model.status().message()));
    }
    return *std::move(model);
  } else {
    return absl::InvalidArgumentError("Missing vocab");
  }
}

}  // namespace tokenizers::models::unigram
