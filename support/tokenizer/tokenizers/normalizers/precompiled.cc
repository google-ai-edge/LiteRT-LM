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
// C++ port of `src/normalizers/precompiled.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/precompiled.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/normalizers/spm_precompiled.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/unicode/graphemes.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"

namespace tokenizers::normalizers {
namespace precompiled {

void Replace(std::vector<std::pair<char32_t, int64_t>>& transformations,
             absl::string_view old_part, absl::string_view new_part) {
  int64_t old_count = static_cast<int64_t>(CountChars(old_part));
  int64_t new_count = static_cast<int64_t>(CountChars(new_part));
  int64_t diff = new_count - old_count;

  // If we are just replacing characters, all changes should be == 0
  for (const auto& [index, c] : CharIndices(new_part)) {
    transformations.push_back({c, 0});
  }

  if (diff > 0) {
    // If we are adding some characters, the last DIFF characters should be == 1
    size_t take = std::min(static_cast<size_t>(diff), transformations.size());
    for (size_t i = 0; i < take; ++i) {
      transformations[transformations.size() - 1 - i].second = 1;
    }
  } else if (diff < 0) {
    // If we are removing some characters, the last one should include the diff
    if (!transformations.empty()) {
      transformations.back().second += diff;
    }
  }
}

}  // namespace precompiled

absl::Status Precompiled::Normalize(NormalizedString& normalized) const {
  std::vector<std::pair<char32_t, int64_t>> transformations;
  transformations.reserve(normalized.Get().size());
  // Future reader. From @Narsil.
  // Yes, this is weird,
  // Yes, this seems broken
  // No, I don't know why Google did this.
  // If you question this code, check this normalizer against
  // XNLI database (all languages) with Unigram model against
  // Mbart, XLMRoberta *AND* Marian. If you don't get 100% or
  // break a single test.
  // You don't pass.
  bool modified = false;
  for (absl::string_view grapheme :
       unicode::Graphemes(normalized.Get(), true)) {
    if (grapheme.size() < 6) {
      if (std::optional<absl::string_view> norm = Transform(grapheme);
          norm.has_value()) {
        modified = true;
        precompiled::Replace(transformations, grapheme, *norm);
        continue;
      }
    }
    for (const auto& [char_index, c] : CharIndices(grapheme)) {
      absl::string_view part = grapheme.substr(char_index, Utf8Len(c));
      if (std::optional<absl::string_view> norm = Transform(part);
          norm.has_value()) {
        modified = true;
        precompiled::Replace(transformations, part, *norm);
      } else {
        transformations.push_back({c, 0});
      }
    }
  }
  if (modified) {
    normalized.Transform(transformations, 0);
  }
  return absl::OkStatus();
}

}  // namespace tokenizers::normalizers
