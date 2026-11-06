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
// C++ port of `src/pre_tokenizers/delimiter.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/delimiter.h"

#include <cstddef>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

absl::Status CharDelimiterSplit::PreTokenize(
    PreTokenizedString& pretokenized) const {
  // TODO: Maybe add the option to specify the behavior
  return pretokenized.Split(
      [this](size_t, NormalizedString normalized)
          -> absl::StatusOr<std::vector<NormalizedString>> {
        return normalized.Split(Pattern(delimiter_),
                                SplitDelimiterBehavior::kRemoved);
      });
}

absl::StatusOr<CharDelimiterSplit> CharDelimiterSplit::FromJson(
    const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "CharDelimiterSplit"));
  ASSIGN_OR_RETURN(const char32_t delimiter,
                   GetField<char32_t>(json, "delimiter"));
  return CharDelimiterSplit(delimiter);
}

Json CharDelimiterSplit::ToJson() const {
  Json json = Json::object();
  json["type"] = "CharDelimiterSplit";
  json["delimiter"] = ToUtf8(delimiter_);
  return json;
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
