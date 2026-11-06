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
// C++ port of `src/decoders/strip.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/strip.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace decoders {

absl::StatusOr<std::vector<std::string>> Strip::DecodeChain(
    std::vector<std::string> tokens) const {
  std::vector<std::string> result;
  result.reserve(tokens.size());
  for (const std::string& token : tokens) {
    std::vector<char32_t> chars;
    for (auto [offset, c] : CharIndices(token)) chars.push_back(c);

    size_t start_cut = 0;
    for (size_t i = 0; i < chars.size() && i < start_; ++i) {
      if (chars[i] == content_) {
        start_cut = i + 1;
        continue;
      } else {
        break;
      }
    }

    size_t stop_cut = chars.size();
    for (size_t i = 0; i < stop_; ++i) {
      // PORT NOTE: `chars.len() - i - 1` underflows in Rust (a panic) when
      // `stop` exceeds the token length and every character matched so far;
      // we stop stripping instead.
      if (i >= chars.size()) break;
      const size_t index = chars.size() - i - 1;
      if (chars[index] == content_) {
        stop_cut = index;
        continue;
      } else {
        break;
      }
    }

    // PORT NOTE: `chars[start_cut..stop_cut]` panics in Rust when the two cuts
    // cross (every character of a short token matched from both ends); the
    // token becomes empty here.
    std::string new_token;
    for (size_t i = start_cut; i < stop_cut; ++i) {
      AppendUtf8(chars[i], &new_token);
    }
    result.push_back(std::move(new_token));
  }
  return result;
}

absl::StatusOr<Strip> Strip::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` on a struct: serde does not validate the tag
  // when deserializing, the `type` entry is simply ignored.
  RETURN_IF_ERROR(RequireObject(json, "struct Strip"));
  ASSIGN_OR_RETURN(char32_t content, GetField<char32_t>(json, "content"));
  ASSIGN_OR_RETURN(uint64_t start, GetField<uint64_t>(json, "start"));
  ASSIGN_OR_RETURN(uint64_t stop, GetField<uint64_t>(json, "stop"));
  return Strip(content, static_cast<size_t>(start), static_cast<size_t>(stop));
}

Json Strip::ToJson() const {
  Json json = Json::object();
  json["type"] = "Strip";
  json["content"] = ToUtf8(content_);
  json["start"] = start_;
  json["stop"] = stop_;
  return json;
}

}  // namespace decoders
}  // namespace tokenizers
