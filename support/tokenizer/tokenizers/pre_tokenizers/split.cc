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
// C++ port of `src/pre_tokenizers/split.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/split.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

// `#[derive(Serialize, Deserialize)] pub enum SplitPattern { String(String),
// Regex(String) }`: externally tagged, i.e. a one-entry object
// `{"String": "..."}` / `{"Regex": "..."}`.
absl::StatusOr<SplitPattern> SplitPattern::FromJson(const Json& json) {
  if (!json.is_object()) {
    return InvalidTypeError(json, "enum SplitPattern");
  }
  if (json.size() != 1) {
    if (json.empty()) {
      return absl::InvalidArgumentError(
          "invalid value: map, expected map with a single key");
    }
    return absl::InvalidArgumentError(
        "invalid length: map, expected map with a single key");
  }
  const std::string& variant = json.begin().key();
  const Json& value = json.begin().value();
  Kind kind;
  if (variant == "String") {
    kind = Kind::kString;
  } else if (variant == "Regex") {
    kind = Kind::kRegex;
  } else {
    static constexpr absl::string_view kVariants[] = {"String", "Regex"};
    return UnknownVariantError(variant, kVariants);
  }
  ASSIGN_OR_RETURN(std::string s, JsonAs<std::string>(value));
  return SplitPattern(kind, std::move(s));
}

Json SplitPattern::ToJson() const {
  Json json = Json::object();
  json[kind_ == Kind::kString ? "String" : "Regex"] = value_;
  return json;
}

absl::StatusOr<Split> Split::New(SplitPattern pattern,
                                 SplitDelimiterBehavior behavior, bool invert) {
  absl::StatusOr<SysRegex> regex;
  switch (pattern.kind()) {
    case SplitPattern::Kind::kString:
      regex = SysRegex::New(RegexEscape(pattern.value()));
      break;
    case SplitPattern::Kind::kRegex:
      regex = SysRegex::New(pattern.value());
      break;
  }
  RETURN_IF_ERROR(regex.status());

  return Split(std::move(pattern),
               std::make_shared<const SysRegex>(*std::move(regex)), behavior,
               invert);
}

absl::Status Split::PreTokenize(PreTokenizedString& pretokenized) const {
  if (invert_) {
    return pretokenized.Split(
        [this](size_t, NormalizedString normalized)
            -> absl::StatusOr<std::vector<NormalizedString>> {
          return normalized.Split(Invert(Pattern(regex_.get())), behavior_);
        });
  } else {
    return pretokenized.Split(
        [this](size_t, NormalizedString normalized)
            -> absl::StatusOr<std::vector<NormalizedString>> {
          return normalized.Split(Pattern(regex_.get()), behavior_);
        });
  }
}

// `impl<'de> Deserialize<'de> for Split`
absl::StatusOr<Split> Split::FromJson(const Json& json) {
  // struct SplitHelper {
  //     #[serde(rename = "type")] _type: Type,  // enum Type { Split }
  //     pattern: SplitPattern,
  //     behavior: SplitDelimiterBehavior,
  //     invert: bool,
  // }
  RETURN_IF_ERROR(RequireTypeTag(json, "Split"));
  const Json* pattern_json = FindField(json, "pattern");
  if (pattern_json == nullptr) return MissingFieldError("pattern");
  ASSIGN_OR_RETURN(SplitPattern pattern, SplitPattern::FromJson(*pattern_json));
  const Json* behavior_json = FindField(json, "behavior");
  if (behavior_json == nullptr) return MissingFieldError("behavior");
  ASSIGN_OR_RETURN(const SplitDelimiterBehavior behavior,
                   SplitDelimiterBehaviorFromJson(*behavior_json));
  ASSIGN_OR_RETURN(const bool invert, GetField<bool>(json, "invert"));
  return New(std::move(pattern), behavior, invert);
}

Json Split::ToJson() const {
  Json json = Json::object();
  json["type"] = "Split";
  json["pattern"] = pattern_.ToJson();
  json["behavior"] = SplitDelimiterBehaviorToJson(behavior_);
  json["invert"] = invert_;
  return json;
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
