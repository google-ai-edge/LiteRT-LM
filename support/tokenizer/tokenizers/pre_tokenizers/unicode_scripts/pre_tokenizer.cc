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
// C++ port of `src/pre_tokenizers/unicode_scripts/pre_tokenizer.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/unicode_scripts/pre_tokenizer.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/unicode_scripts/scripts.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {
namespace unicode_scripts {

// This code exists in the Unigram default IsValidSentencePiece.
// It could be integrated directly within `get_script` but I
// think it's kind of tricky to see those modifications later
// I am guessing release mode will optimize this away anyway.
Script FixedScript(char32_t c) {
  const Script raw_script = GetScript(c);
  if (static_cast<uint32_t>(c) == 0x30FC) {
    return Script::kHan;
  } else if (c == U' ') {
    return Script::kAny;
  } else {
    switch (raw_script) {
      case Script::kHiragana:
        return Script::kHan;
      case Script::kKatakana:
        return Script::kHan;
      default:
        return raw_script;
    }
  }
}

absl::Status UnicodeScripts::PreTokenize(
    PreTokenizedString& pretokenized) const {
  return pretokenized.Split(
      [](size_t, NormalizedString normalized)
          -> absl::StatusOr<std::vector<NormalizedString>> {
        std::optional<Script> last_script;
        size_t offset = 0;
        std::vector<size_t> ranges;
        const absl::string_view text = normalized.Get();
        for (const auto [i, c] : CharIndices(text)) {
          const std::optional<Script> script = FixedScript(c);
          std::optional<size_t> result;
          if (script != Script::kAny && last_script != Script::kAny &&
              last_script != script) {
            result = offset;
          }
          offset += Utf8Len(c);
          if (script != Script::kAny) {
            last_script = script;
          }

          if (result.has_value()) ranges.push_back(*result);
        }
        ranges.push_back(text.size());
        std::vector<NormalizedString> splits;
        for (size_t k = 0; k + 1 < ranges.size(); ++k) {
          std::optional<NormalizedString> slice =
              normalized.Slice(Range::Normalized(ranges[k], ranges[k + 1]));
          // `.expect("NormalizedString bad split")`
          if (!slice.has_value()) {
            return absl::InternalError("NormalizedString bad split");
          }
          splits.push_back(*std::move(slice));
        }
        return splits;
      });
}

absl::StatusOr<UnicodeScripts> UnicodeScripts::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "UnicodeScripts"));
  return UnicodeScripts();
}

Json UnicodeScripts::ToJson() const {
  Json json = Json::object();
  json["type"] = "UnicodeScripts";
  return json;
}

}  // namespace unicode_scripts
}  // namespace pre_tokenizers
}  // namespace tokenizers
