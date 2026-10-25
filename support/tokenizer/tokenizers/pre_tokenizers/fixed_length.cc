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
// C++ port of `src/pre_tokenizers/fixed_length.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/fixed_length.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

namespace {

// `fn default_length() -> usize { 5 }`
constexpr size_t kDefaultLength = 5;

}  // namespace

absl::Status FixedLength::PreTokenize(PreTokenizedString& pretokenized) const {
  return pretokenized.Split(
      [this](size_t, NormalizedString normalized)
          -> absl::StatusOr<std::vector<NormalizedString>> {
        const absl::string_view text = normalized.Get();
        if (text.empty()) {
          return std::vector<NormalizedString>();
        }
        // PORT NOTE: `char_positions.chunks(0)` panics in Rust ("chunk size
        // must be non-zero"); a zero `length` can come from the JSON, so it
        // is an error here.
        if (length_ == 0) {
          return absl::InvalidArgumentError("chunk size must be non-zero");
        }

        std::vector<NormalizedString> splits;
        std::vector<std::pair<size_t, char32_t>> char_positions;
        for (const auto [i, c] : CharIndices(text)) {
          char_positions.emplace_back(i, c);
        }
        for (size_t chunk_start = 0; chunk_start < char_positions.size();
             chunk_start += length_) {
          const size_t chunk_end =
              std::min(chunk_start + length_, char_positions.size());
          const size_t start = char_positions[chunk_start].first;
          const auto& last = char_positions[chunk_end - 1];
          const size_t end = last.first + Utf8Len(last.second);
          std::optional<NormalizedString> slice =
              normalized.Slice(Range::Normalized(start, end));
          if (!slice.has_value()) {
            return absl::InvalidArgumentError(
                "Failed to slice normalized text");
          }
          splits.push_back(*std::move(slice));
        }

        return splits;
      });
}

absl::StatusOr<FixedLength> FixedLength::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "FixedLength"));
  // `#[serde(default = "default_length")]`
  ASSIGN_OR_RETURN(const size_t length,
                   GetFieldOr<size_t>(json, "length", kDefaultLength));
  return FixedLength(length);
}

Json FixedLength::ToJson() const {
  Json json = Json::object();
  json["type"] = "FixedLength";
  json["length"] = length_;
  return json;
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
