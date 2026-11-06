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
// C++ port of `src/decoders/bpe.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/bpe.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/wordpiece.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace decoders {

absl::StatusOr<std::vector<std::string>> BPEDecoder::DecodeChain(
    std::vector<std::string> tokens) const {
  // PORT NOTE: `tokens.len() - 1` wraps for an empty input in Rust release
  // builds (no overflow checks); `size_t` wraps the same way here and the loop
  // below does nothing, so the result is identical (an empty vector).
  const size_t n = tokens.size() - 1;
  std::vector<std::string> result;
  result.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    const absl::string_view replacement = i == n ? "" : " ";
    result.push_back(StrReplace(tokens[i], suffix_, replacement));
  }
  return result;
}

absl::StatusOr<BPEDecoder> BPEDecoder::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` on a struct: serde does not validate the tag
  // when deserializing, the `type` entry is simply ignored.
  RETURN_IF_ERROR(RequireObject(json, "struct BPEDecoder"));
  ASSIGN_OR_RETURN(std::string suffix, GetField<std::string>(json, "suffix"));
  return BPEDecoder(std::move(suffix));
}

Json BPEDecoder::ToJson() const {
  Json json = Json::object();
  json["type"] = "BPEDecoder";
  json["suffix"] = suffix_;
  return json;
}

}  // namespace decoders
}  // namespace tokenizers
