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
// C++ port of `src/utils/padding.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/padding.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {

absl::string_view PaddingDirectionAsStr(PaddingDirection direction) {
  switch (direction) {
    case PaddingDirection::kLeft:
      return "left";
    case PaddingDirection::kRight:
      return "right";
  }
  return "right";
}

absl::StatusOr<PaddingDirection> PaddingDirectionFromJson(const Json& json) {
  if (json.is_string()) {
    const std::string s = json.get<std::string>();
    if (s == "Left") return PaddingDirection::kLeft;
    if (s == "Right") return PaddingDirection::kRight;
    return absl::InvalidArgumentError(
        absl::StrCat("unknown variant `", s, "`, expected `Left` or `Right`"));
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "invalid type: ", json.dump(), ", expected variant identifier"));
}

Json PaddingDirectionToJson(PaddingDirection direction) {
  switch (direction) {
    case PaddingDirection::kLeft:
      return "Left";
    case PaddingDirection::kRight:
      return "Right";
  }
  return "Right";
}

absl::StatusOr<PaddingStrategy> PaddingStrategy::FromJson(const Json& json) {
  // Externally tagged enum: a unit variant is a string, a newtype variant is a
  // single-entry object.
  if (json.is_string()) {
    const std::string s = json.get<std::string>();
    if (s == "BatchLongest") return PaddingStrategy::BatchLongest();
    if (s == "Fixed") {
      return absl::InvalidArgumentError(
          "invalid type: unit variant, expected newtype variant");
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "unknown variant `", s, "`, expected `BatchLongest` or `Fixed`"));
  }
  if (json.is_object() && json.size() == 1) {
    const auto item = json.begin();
    const std::string& key = item.key();
    if (key == "Fixed") {
      ASSIGN_OR_RETURN(size_t size, JsonAs<size_t>(item.value()));
      return PaddingStrategy::Fixed(size);
    }
    if (key == "BatchLongest") {
      return absl::InvalidArgumentError(
          "invalid type: newtype variant, expected unit variant");
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "unknown variant `", key, "`, expected `BatchLongest` or `Fixed`"));
  }
  return InvalidTypeError(json, "enum PaddingStrategy");
}

Json PaddingStrategy::ToJson() const {
  switch (kind) {
    case Kind::kBatchLongest:
      return "BatchLongest";
    case Kind::kFixed: {
      Json json = Json::object();
      json["Fixed"] = size;
      return json;
    }
  }
  return "BatchLongest";
}

absl::StatusOr<PaddingParams> PaddingParams::FromJson(const Json& json) {
  if (!json.is_object()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid type: ", json.dump(), ", expected struct PaddingParams"));
  }
  auto field = [&json](absl::string_view name) -> absl::StatusOr<const Json*> {
    auto it = json.find(std::string(name));
    if (it == json.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("missing field `", name, "`"));
    }
    return &*it;
  };
  auto u32 = [](const Json& j) -> absl::StatusOr<uint32_t> {
    if (!j.is_number_unsigned() || j.get<uint64_t>() > UINT32_MAX) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid type: ", j.dump(), ", expected u32"));
    }
    return j.get<uint32_t>();
  };

  PaddingParams params;
  ASSIGN_OR_RETURN(const Json* strategy, field("strategy"));
  ASSIGN_OR_RETURN(params.strategy, PaddingStrategy::FromJson(*strategy));
  ASSIGN_OR_RETURN(const Json* direction, field("direction"));
  ASSIGN_OR_RETURN(params.direction, PaddingDirectionFromJson(*direction));
  // `Option<usize>`: missing or `null` is `None`.
  auto pad_to_multiple_of = json.find("pad_to_multiple_of");
  if (pad_to_multiple_of != json.end() && !pad_to_multiple_of->is_null()) {
    if (!pad_to_multiple_of->is_number_unsigned()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "invalid type: ", pad_to_multiple_of->dump(), ", expected usize"));
    }
    params.pad_to_multiple_of =
        static_cast<size_t>(pad_to_multiple_of->get<uint64_t>());
  }
  ASSIGN_OR_RETURN(const Json* pad_id, field("pad_id"));
  ASSIGN_OR_RETURN(params.pad_id, u32(*pad_id));
  ASSIGN_OR_RETURN(const Json* pad_type_id, field("pad_type_id"));
  ASSIGN_OR_RETURN(params.pad_type_id, u32(*pad_type_id));
  ASSIGN_OR_RETURN(const Json* pad_token, field("pad_token"));
  if (!pad_token->is_string()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid type: ", pad_token->dump(), ", expected a string"));
  }
  params.pad_token = pad_token->get<std::string>();
  return params;
}

Json PaddingParams::ToJson() const {
  Json json = Json::object();
  json["strategy"] = strategy.ToJson();
  json["direction"] = PaddingDirectionToJson(direction);
  if (pad_to_multiple_of.has_value()) {
    json["pad_to_multiple_of"] = *pad_to_multiple_of;
  } else {
    json["pad_to_multiple_of"] = nullptr;
  }
  json["pad_id"] = pad_id;
  json["pad_type_id"] = pad_type_id;
  json["pad_token"] = pad_token;
  return json;
}

absl::Status PadEncodings(absl::Span<Encoding> encodings,
                          const PaddingParams& params) {
  if (encodings.empty()) {
    return absl::OkStatus();
  }

  size_t pad_length = 0;
  switch (params.strategy.kind) {
    case PaddingStrategy::Kind::kFixed:
      pad_length = params.strategy.size;
      break;
    case PaddingStrategy::Kind::kBatchLongest:
      for (const Encoding& e : encodings) {
        pad_length = std::max(pad_length, e.GetIds().size());
      }
      break;
  }

  if (params.pad_to_multiple_of.has_value()) {
    const size_t multiple = *params.pad_to_multiple_of;
    if (multiple > 0 && pad_length % multiple > 0) {
      pad_length += multiple - pad_length % multiple;
    }
  }

  for (Encoding& encoding : encodings) {
    encoding.Pad(pad_length, params.pad_id, params.pad_type_id,
                 params.pad_token, params.direction);
  }

  return absl::OkStatus();
}

}  // namespace tokenizers
