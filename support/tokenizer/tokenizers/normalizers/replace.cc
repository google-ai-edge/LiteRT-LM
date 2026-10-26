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
// C++ port of `src/normalizers/replace.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/replace.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {

absl::StatusOr<ReplacePattern> ReplacePattern::FromJson(const Json& json) {
  // Externally tagged enum, deserialized through `serde_json::from_value`:
  // a map with a single key (the variant) or a string (a unit variant, which
  // neither `String` nor `Regex` is).
  static constexpr absl::string_view kVariants[] = {"String", "Regex"};
  if (json.is_object()) {
    if (json.size() != 1) {
      return InvalidValueError(json, "map with a single key");
    }
    const std::string& variant = json.begin().key();
    const Json& value = json.begin().value();
    if (variant != "String" && variant != "Regex") {
      return UnknownVariantError(variant, kVariants);
    }
    if (!value.is_string()) return InvalidTypeError(value, "a string");
    return variant == "String" ? String(value.get<std::string>())
                               : Regex(value.get<std::string>());
  }
  if (json.is_string()) {
    const std::string& variant = json.get_ref<const std::string&>();
    if (variant != "String" && variant != "Regex") {
      return UnknownVariantError(variant, kVariants);
    }
    return absl::InvalidArgumentError(
        "invalid type: unit variant, expected newtype variant");
  }
  return InvalidTypeError(json, "string or map");
}

Json ReplacePattern::ToJson() const {
  Json json = Json::object();
  json[kind == Kind::kString ? "String" : "Regex"] = value;
  return json;
}

absl::StatusOr<Replace> Replace::New(ReplacePattern pattern,
                                     std::string content) {
  absl::StatusOr<SysRegex> regex;
  switch (pattern.kind) {
    case ReplacePattern::Kind::kString:
      regex = SysRegex::New(RegexEscape(pattern.value));
      break;
    case ReplacePattern::Kind::kRegex:
      regex = SysRegex::New(pattern.value);
      break;
  }
  RETURN_IF_ERROR(regex.status());

  return Replace(std::move(pattern), std::move(content),
                 std::make_shared<const SysRegex>(*std::move(regex)));
}

absl::StatusOr<Replace> Replace::FromJson(const Json& json) {
  // We use this custom deserializer to provide the value for `regex` for
  // `Replace`: `struct ReplaceDeserializer { pattern: ReplacePattern,
  // content: String }` (`#[serde(tag = "type")]`, tag not validated), then
  // `TryFrom<ReplaceDeserializer> for Replace` = `Replace::new`.
  RETURN_IF_ERROR(RequireObject(json, "struct ReplaceDeserializer"));
  const Json* pattern_json = FindField(json, "pattern");
  if (pattern_json == nullptr) return MissingFieldError("pattern");
  ASSIGN_OR_RETURN(ReplacePattern pattern,
                   ReplacePattern::FromJson(*pattern_json));
  ASSIGN_OR_RETURN(std::string content, GetField<std::string>(json, "content"));
  return New(std::move(pattern), std::move(content));
}

Json Replace::ToJson() const {
  Json json = Json::object();
  json["type"] = "Replace";
  json["pattern"] = pattern_.ToJson();
  json["content"] = content_;
  return json;
}

absl::Status Replace::Normalize(NormalizedString& normalized) const {
  return normalized.Replace(Pattern(regex_.get()), content_);
}

absl::StatusOr<std::vector<std::string>> Replace::DecodeChain(
    std::vector<std::string> tokens) const {
  std::vector<std::string> result;
  result.reserve(tokens.size());
  for (const std::string& token : tokens) {
    std::string new_token;

    ASSIGN_OR_RETURN(auto matches, Pattern(regex_.get()).FindMatches(token));
    for (const auto& [offsets, is_match] : matches) {
      const auto& [start, stop] = offsets;
      if (is_match) {
        new_token.append(content_);
      } else {
        new_token.append(token, start, stop - start);
      }
    }
    result.push_back(std::move(new_token));
  }
  return result;
}

}  // namespace tokenizers::normalizers
