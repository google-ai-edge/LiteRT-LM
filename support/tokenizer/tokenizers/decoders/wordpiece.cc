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
// C++ port of `src/decoders/wordpiece.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/wordpiece.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace decoders {

std::string StrReplace(absl::string_view s, absl::string_view from,
                       absl::string_view to) {
  std::string result;
  if (from.empty()) {
    // `str::match_indices("")` yields a match at every char boundary,
    // including both ends of the string.
    result.reserve(s.size() + (CountChars(s) + 1) * to.size());
    result.append(to.data(), to.size());
    for (auto [offset, c] : CharIndices(s)) {
      result.append(s.data() + offset, Utf8Len(c));
      result.append(to.data(), to.size());
    }
    return result;
  }
  result.reserve(s.size());
  size_t last_end = 0;
  for (size_t pos = s.find(from); pos != absl::string_view::npos;
       pos = s.find(from, pos + from.size())) {
    result.append(s.data() + last_end, pos - last_end);
    result.append(to.data(), to.size());
    last_end = pos + from.size();
  }
  result.append(s.data() + last_end, s.size() - last_end);
  return result;
}

std::string Cleanup(absl::string_view dirty_input) {
  std::string out = StrReplace(dirty_input, " .", ".");
  out = StrReplace(out, " ?", "?");
  out = StrReplace(out, " !", "!");
  out = StrReplace(out, " ,", ",");
  out = StrReplace(out, " ' ", "'");
  out = StrReplace(out, " n't", "n't");
  out = StrReplace(out, " 'm", "'m");
  out = StrReplace(out, " do not", " don't");
  out = StrReplace(out, " 's", "'s");
  out = StrReplace(out, " 've", "'ve");
  out = StrReplace(out, " 're", "'re");
  return out;
}

absl::StatusOr<std::vector<std::string>> WordPiece::DecodeChain(
    std::vector<std::string> tokens) const {
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string& token = tokens[i];
    if (i != 0) {
      absl::string_view tk = token;
      if (absl::ConsumePrefix(&tk, prefix_)) {
        token = std::string(tk);
      } else {
        token = absl::StrCat(" ", token);
      }
    }
    if (cleanup_) {
      token = Cleanup(token);
    }
  }
  return tokens;
}

absl::StatusOr<WordPiece> WordPiece::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` on a struct: serde does not validate the tag
  // when deserializing, the `type` entry is simply ignored.
  RETURN_IF_ERROR(RequireObject(json, "struct WordPiece"));
  ASSIGN_OR_RETURN(std::string prefix, GetField<std::string>(json, "prefix"));
  ASSIGN_OR_RETURN(bool cleanup, GetField<bool>(json, "cleanup"));
  return WordPiece(std::move(prefix), cleanup);
}

Json WordPiece::ToJson() const {
  Json json = Json::object();
  json["type"] = "WordPiece";
  json["prefix"] = prefix_;
  json["cleanup"] = cleanup_;
  return json;
}

}  // namespace decoders
}  // namespace tokenizers
