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
// C++ port of `src/pre_tokenizers/whitespace.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/whitespace.h"

#include <cstddef>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/unicode/categories.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

absl::Status Whitespace::PreTokenize(PreTokenizedString& pretokenized) const {
  // `static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+|[^\w\s]+")
  // .unwrap());`
  // PORT NOTE: Rust uses the `regex` crate here, whose `\w`
  // (`regex_syntax::PERL_WORD`, UTS #18 Annex C) includes `U+200C` (ZWNJ) and
  // `U+200D` (ZWJ), i.e. `\p{Join_Control}`, whereas Oniguruma's `\w`
  // (`CR_Word`) excludes them (`\s` is identical). Including `\p{Join_Control}`
  // alongside `\w` makes Oniguruma match Rust's `regex::Regex` across all
  // Unicode scalar values. The `unwrap()` of the constant pattern becomes a
  // `Status` error should the regex engine ever reject it.
  static const absl::StatusOr<SysRegex>* const re =
      new absl::StatusOr<SysRegex>(
          SysRegex::New(R"([\w\p{Join_Control}]+|[^\w\s\p{Join_Control}]+)"));
  if (!re->ok()) return re->status();
  const SysRegex* re_ref = &**re;

  return pretokenized.Split(
      [re_ref](size_t, NormalizedString normalized)
          -> absl::StatusOr<std::vector<NormalizedString>> {
        return normalized.Split(Invert(Pattern(re_ref)),
                                SplitDelimiterBehavior::kRemoved);
      });
}

absl::StatusOr<Whitespace> Whitespace::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "Whitespace"));
  return Whitespace();
}

Json Whitespace::ToJson() const {
  Json json = Json::object();
  json["type"] = "Whitespace";
  return json;
}

absl::Status WhitespaceSplit::PreTokenize(
    PreTokenizedString& pretokenized) const {
  return pretokenized.Split(
      [](size_t, NormalizedString normalized)
          -> absl::StatusOr<std::vector<NormalizedString>> {
        return normalized.Split(Pattern(&unicode::IsWhitespace),
                                SplitDelimiterBehavior::kRemoved);
      });
}

absl::StatusOr<WhitespaceSplit> WhitespaceSplit::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "WhitespaceSplit"));
  return WhitespaceSplit();
}

Json WhitespaceSplit::ToJson() const {
  Json json = Json::object();
  json["type"] = "WhitespaceSplit";
  return json;
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
