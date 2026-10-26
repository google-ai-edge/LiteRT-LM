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
// C++ port of `src/utils/truncation.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/truncation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {

absl::string_view TruncationDirectionAsStr(TruncationDirection direction) {
  switch (direction) {
    case TruncationDirection::kLeft:
      return "left";
    case TruncationDirection::kRight:
      return "right";
  }
  return "right";
}

absl::StatusOr<TruncationDirection> TruncationDirectionFromJson(
    const Json& json) {
  if (json.is_string()) {
    const std::string s = json.get<std::string>();
    if (s == "Left") return TruncationDirection::kLeft;
    if (s == "Right") return TruncationDirection::kRight;
    return absl::InvalidArgumentError(
        absl::StrCat("unknown variant `", s, "`, expected `Left` or `Right`"));
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "invalid type: ", json.dump(), ", expected variant identifier"));
}

Json TruncationDirectionToJson(TruncationDirection direction) {
  switch (direction) {
    case TruncationDirection::kLeft:
      return "Left";
    case TruncationDirection::kRight:
      return "Right";
  }
  return "Right";
}

absl::string_view TruncationStrategyAsStr(TruncationStrategy strategy) {
  switch (strategy) {
    case TruncationStrategy::kLongestFirst:
      return "longest_first";
    case TruncationStrategy::kOnlyFirst:
      return "only_first";
    case TruncationStrategy::kOnlySecond:
      return "only_second";
  }
  return "longest_first";
}

absl::StatusOr<TruncationStrategy> TruncationStrategyFromJson(
    const Json& json) {
  if (json.is_string()) {
    const std::string s = json.get<std::string>();
    if (s == "LongestFirst") return TruncationStrategy::kLongestFirst;
    if (s == "OnlyFirst") return TruncationStrategy::kOnlyFirst;
    if (s == "OnlySecond") return TruncationStrategy::kOnlySecond;
    return absl::InvalidArgumentError(absl::StrCat(
        "unknown variant `", s,
        "`, expected one of `LongestFirst`, `OnlyFirst`, `OnlySecond`"));
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "invalid type: ", json.dump(), ", expected variant identifier"));
}

Json TruncationStrategyToJson(TruncationStrategy strategy) {
  switch (strategy) {
    case TruncationStrategy::kLongestFirst:
      return "LongestFirst";
    case TruncationStrategy::kOnlyFirst:
      return "OnlyFirst";
    case TruncationStrategy::kOnlySecond:
      return "OnlySecond";
  }
  return "LongestFirst";
}

absl::StatusOr<TruncationParams> TruncationParams::FromJson(const Json& json) {
  if (!json.is_object()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid type: ", json.dump(), ", expected struct TruncationParams"));
  }
  auto field = [&json](absl::string_view name) -> absl::StatusOr<const Json*> {
    auto it = json.find(std::string(name));
    if (it == json.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("missing field `", name, "`"));
    }
    return &*it;
  };
  auto usize = [](const Json& j) -> absl::StatusOr<size_t> {
    if (!j.is_number_unsigned()) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid type: ", j.dump(), ", expected usize"));
    }
    return static_cast<size_t>(j.get<uint64_t>());
  };

  TruncationParams params;
  // `#[serde(default)] pub direction: TruncationDirection`
  auto direction = json.find("direction");
  if (direction != json.end()) {
    ASSIGN_OR_RETURN(params.direction, TruncationDirectionFromJson(*direction));
  }
  ASSIGN_OR_RETURN(const Json* max_length, field("max_length"));
  ASSIGN_OR_RETURN(params.max_length, usize(*max_length));
  ASSIGN_OR_RETURN(const Json* strategy, field("strategy"));
  ASSIGN_OR_RETURN(params.strategy, TruncationStrategyFromJson(*strategy));
  ASSIGN_OR_RETURN(const Json* stride, field("stride"));
  ASSIGN_OR_RETURN(params.stride, usize(*stride));
  return params;
}

Json TruncationParams::ToJson() const {
  Json json = Json::object();
  json["direction"] = TruncationDirectionToJson(direction);
  json["max_length"] = max_length;
  json["strategy"] = TruncationStrategyToJson(strategy);
  json["stride"] = stride;
  return json;
}

absl::StatusOr<std::pair<Encoding, std::optional<Encoding>>> TruncateEncodings(
    Encoding encoding, std::optional<Encoding> pair_encoding,
    const TruncationParams& params) {
  if (params.max_length == 0) {
    RETURN_IF_ERROR(encoding.Truncate(0, params.stride, params.direction));
    if (pair_encoding.has_value()) {
      RETURN_IF_ERROR(
          pair_encoding->Truncate(0, params.stride, params.direction));
    }
    return std::make_pair(std::move(encoding), std::move(pair_encoding));
  }

  const size_t total_length =
      encoding.GetIds().size() +
      (pair_encoding.has_value() ? pair_encoding->GetIds().size() : 0);
  size_t to_remove;
  if (total_length > params.max_length) {
    to_remove = total_length - params.max_length;
  } else {
    return std::make_pair(std::move(encoding), std::move(pair_encoding));
  }

  switch (params.strategy) {
    case TruncationStrategy::kLongestFirst: {
      if (pair_encoding.has_value()) {
        Encoding& other_encoding = *pair_encoding;
        // Assuming n1 <= n2, there are 3 cases
        // Case 1:
        //   No truncation needs to be performed.
        //   This scenario is handled before the match.
        // Case 2:
        //   Only the longer input needs to be truncated.
        //   n1 = n1
        //   n2 = max_length - n1
        // Case 3:
        //   Both inputs must be truncated.
        //   n1 = max_length / 2
        //   n2 = n1 + max_length % 2

        size_t n1 = encoding.GetIds().size();
        size_t n2 = other_encoding.GetIds().size();
        bool swap = false;

        // Ensure n1 is the length of the shortest input
        if (n1 > n2) {
          swap = true;
          std::swap(n1, n2);
        }

        if (n1 > params.max_length) {
          // This needs to be a special case
          // to avoid max_length - n1 < 0
          // since n1 and n2 are unsigned
          n2 = n1;
        } else {
          n2 = std::max(n1, params.max_length - n1);
        }

        if (n1 + n2 > params.max_length) {
          n1 = params.max_length / 2;
          n2 = n1 + params.max_length % 2;
        }

        // Swap lengths if we swapped previously
        if (swap) {
          std::swap(n1, n2);
        }
        RETURN_IF_ERROR(encoding.Truncate(n1, params.stride, params.direction));
        RETURN_IF_ERROR(
            other_encoding.Truncate(n2, params.stride, params.direction));
      } else {
        RETURN_IF_ERROR(encoding.Truncate(total_length - to_remove,
                                          params.stride, params.direction));
      }
      break;
    }
    case TruncationStrategy::kOnlyFirst:
    case TruncationStrategy::kOnlySecond: {
      Encoding* target;
      if (params.strategy == TruncationStrategy::kOnlyFirst) {
        target = &encoding;
      } else if (pair_encoding.has_value()) {
        target = &*pair_encoding;
      } else {
        return TruncationError::SecondSequenceNotProvided();
      }

      const size_t target_len = target->GetIds().size();
      if (target_len > to_remove) {
        RETURN_IF_ERROR(target->Truncate(target_len - to_remove, params.stride,
                                         params.direction));
      } else {
        return TruncationError::SequenceTooShort();
      }
      break;
    }
  }
  return std::make_pair(std::move(encoding), std::move(pair_encoding));
}

}  // namespace tokenizers
