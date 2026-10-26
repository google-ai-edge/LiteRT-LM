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
// C++ port of `src/pre_tokenizers/punctuation.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/punctuation.h"

#include <cstddef>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/unicode/categories.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

namespace {

bool IsPunc(char32_t x) {
  return unicode::IsAsciiPunctuation(x) || unicode::IsPunctuation(x);
}

// `fn default_split() -> SplitDelimiterBehavior { Isolated }`
constexpr SplitDelimiterBehavior kDefaultSplit =
    SplitDelimiterBehavior::kIsolated;

}  // namespace

absl::Status Punctuation::PreTokenize(PreTokenizedString& pretokenized) const {
  return pretokenized.Split(
      [this](
          size_t,
          NormalizedString s) -> absl::StatusOr<std::vector<NormalizedString>> {
        return s.Split(Pattern(&IsPunc), behavior_);
      });
}

absl::StatusOr<Punctuation> Punctuation::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "Punctuation"));
  // `#[serde(default = "default_split")]`
  SplitDelimiterBehavior behavior = kDefaultSplit;
  if (const Json* field = FindField(json, "behavior"); field != nullptr) {
    ASSIGN_OR_RETURN(behavior, SplitDelimiterBehaviorFromJson(*field));
  }
  return Punctuation(behavior);
}

Json Punctuation::ToJson() const {
  Json json = Json::object();
  json["type"] = "Punctuation";
  json["behavior"] = SplitDelimiterBehaviorToJson(behavior_);
  return json;
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
