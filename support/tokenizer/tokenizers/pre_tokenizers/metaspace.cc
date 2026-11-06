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
// C++ port of `src/pre_tokenizers/metaspace.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/metaspace.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

absl::string_view PrependSchemeToString(PrependScheme scheme) {
  switch (scheme) {
    case PrependScheme::kFirst:
      return "first";
    case PrependScheme::kNever:
      return "never";
    case PrependScheme::kAlways:
      return "always";
  }
  return "always";
}

absl::StatusOr<PrependScheme> PrependSchemeFromJson(const Json& json) {
  ASSIGN_OR_RETURN(const std::string variant, JsonAs<std::string>(json));
  if (variant == "first") return PrependScheme::kFirst;
  if (variant == "never") return PrependScheme::kNever;
  if (variant == "always") return PrependScheme::kAlways;
  static constexpr absl::string_view kVariants[] = {"first", "never", "always"};
  return UnknownVariantError(variant, kVariants);
}

Json PrependSchemeToJson(PrependScheme scheme) {
  return Json(std::string(PrependSchemeToString(scheme)));
}

Metaspace::Metaspace(char32_t replacement, PrependScheme prepend_scheme,
                     bool split)
    : replacement_(replacement),
      prepend_scheme_(prepend_scheme),
      split_(split),
      str_rep_(ToUtf8(replacement)) {}

void Metaspace::SetReplacement(char32_t replacement) {
  replacement_ = replacement;
  str_rep_ = ToUtf8(replacement);
}

absl::Status Metaspace::PreTokenize(PreTokenizedString& pretokenized) const {
  return pretokenized.Split(
      [&](size_t, NormalizedString normalized)
          -> absl::StatusOr<std::vector<NormalizedString>> {
        RETURN_IF_ERROR(normalized.Replace(Pattern(U' '), str_rep_));
        switch (prepend_scheme_) {
          case PrependScheme::kAlways:
            if (!absl::StartsWith(normalized.Get(), str_rep_)) {
              normalized.Prepend(str_rep_);
            }
            break;
          case PrependScheme::kFirst:
            if (!absl::StartsWith(normalized.Get(), str_rep_) &&
                normalized.OffsetsOriginal().first == 0) {
              normalized.Prepend(str_rep_);
            }
            break;
          case PrependScheme::kNever:
            break;
        }
        if (split_) {
          return normalized.Split(Pattern(replacement_),
                                  SplitDelimiterBehavior::kMergedWithNext);
        } else {
          std::vector<NormalizedString> result;
          result.push_back(std::move(normalized));
          return result;
        }
      });
}

absl::StatusOr<std::vector<std::string>> Metaspace::DecodeChain(
    std::vector<std::string> tokens) const {
  std::vector<std::string> result;
  result.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    const std::string& token = tokens[i];
    std::string decoded;
    for (const auto [offset, c] : CharIndices(token)) {
      if (c == replacement_) {
        if (i == 0 && prepend_scheme_ != PrependScheme::kNever) {
          // None
        } else {
          decoded.push_back(' ');
        }
      } else {
        AppendUtf8(c, &decoded);
      }
    }
    result.push_back(std::move(decoded));
  }
  return result;
}

// `impl<'de> Deserialize<'de> for Metaspace`
absl::StatusOr<Metaspace> Metaspace::FromJson(const Json& json) {
  // struct MetaspaceHelper {
  //     #[serde(rename = "type")] _type: Type,  // enum Type { Metaspace }
  //     replacement: char,
  //     pub add_prefix_space: Option<bool>,
  //     #[serde(default = "default_prepend_scheme_value")]
  //     pub prepend_scheme: PrependScheme,
  //     pub split: Option<bool>,
  //     #[serde(rename = "str_rep")] _str_rep: Option<String>,
  // }
  RETURN_IF_ERROR(RequireTypeTag(json, "Metaspace"));
  ASSIGN_OR_RETURN(const char32_t replacement,
                   GetField<char32_t>(json, "replacement"));
  ASSIGN_OR_RETURN(const std::optional<bool> add_prefix_space,
                   GetOptionalField<bool>(json, "add_prefix_space"));
  PrependScheme prepend_scheme = PrependScheme::kAlways;
  if (const Json* field = FindField(json, "prepend_scheme"); field != nullptr) {
    ASSIGN_OR_RETURN(prepend_scheme, PrependSchemeFromJson(*field));
  }
  ASSIGN_OR_RETURN(const std::optional<bool> split,
                   GetOptionalField<bool>(json, "split"));
  ASSIGN_OR_RETURN(const std::optional<std::string> str_rep,
                   GetOptionalField<std::string>(json, "str_rep"));
  (void)str_rep;

  if (add_prefix_space == false) {
    if (prepend_scheme != PrependScheme::kNever) {
      return absl::InvalidArgumentError(
          "add_prefix_space does not match declared prepend_scheme");
    }
    prepend_scheme = PrependScheme::kNever;
  }
  return Metaspace(replacement, prepend_scheme, split.value_or(true));
}

Json Metaspace::ToJson() const {
  Json json = Json::object();
  json["type"] = "Metaspace";
  json["replacement"] = str_rep_;
  json["prepend_scheme"] = PrependSchemeToJson(prepend_scheme_);
  json["split"] = split_;
  return json;
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
