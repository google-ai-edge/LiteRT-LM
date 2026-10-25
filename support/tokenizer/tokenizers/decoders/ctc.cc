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
// C++ port of `src/decoders/ctc.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/ctc.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/wordpiece.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace decoders {

absl::StatusOr<std::vector<std::string>> CTC::DecodeChain(
    std::vector<std::string> tokens) const {
  std::vector<std::string> result;
  // `.dedup()` (itertools): consecutive duplicates collapse into the first.
  const std::string* previous = nullptr;
  for (std::string& token : tokens) {
    if (previous != nullptr && *previous == token) continue;
    previous = &token;
    // `.filter_map(...)`.
    std::string replaced = StrReplace(token, pad_token_, "");
    if (cleanup_) {
      replaced = StrReplace(Cleanup(replaced), word_delimiter_token_, " ");
    }
    if (replaced.empty()) {
      continue;
    }
    result.push_back(std::move(replaced));
  }
  return result;
}

absl::StatusOr<CTC> CTC::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` on a struct: serde does not validate the tag
  // when deserializing, the `type` entry is simply ignored.
  RETURN_IF_ERROR(RequireObject(json, "struct CTC"));
  ASSIGN_OR_RETURN(std::string pad_token,
                   GetField<std::string>(json, "pad_token"));
  ASSIGN_OR_RETURN(std::string word_delimiter_token,
                   GetField<std::string>(json, "word_delimiter_token"));
  ASSIGN_OR_RETURN(bool cleanup, GetField<bool>(json, "cleanup"));
  return CTC(std::move(pad_token), std::move(word_delimiter_token), cleanup);
}

Json CTC::ToJson() const {
  Json json = Json::object();
  json["type"] = "CTC";
  json["pad_token"] = pad_token_;
  json["word_delimiter_token"] = word_delimiter_token_;
  json["cleanup"] = cleanup_;
  return json;
}

}  // namespace decoders
}  // namespace tokenizers
