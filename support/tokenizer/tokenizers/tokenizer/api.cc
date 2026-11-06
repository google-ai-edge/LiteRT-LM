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
// C++ port of `src/tokenizer/mod.rs` (lines 1-220) from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/api.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/util/status_macros.h"

namespace tokenizers {

absl::Status Model::TokenizeInPretokenized(
    PreTokenizedString& pretokenized,
    std::optional<std::pair<size_t, TruncationDirection>> truncation) const {
  if (truncation.has_value()) {
    const auto [max_tokens, direction] = *truncation;
    return pretokenized.TokenizeWithLimit(
        [this](const NormalizedString& normalized) {
          return Tokenize(normalized.Get());
        },
        max_tokens, direction);
  }
  return pretokenized.Tokenize([this](const NormalizedString& normalized) {
    return Tokenize(normalized.Get());
  });
}

absl::StatusOr<Encoding> PostProcessor::Process(
    Encoding encoding, std::optional<Encoding> pair_encoding,
    bool add_special_tokens) const {
  std::vector<Encoding> encodings;
  if (pair_encoding.has_value()) {
    encodings.push_back(std::move(encoding));
    encodings.push_back(*std::move(pair_encoding));
  } else {
    encodings.push_back(std::move(encoding));
  }
  for (size_t i = 0; i < encodings.size(); ++i) {
    Encoding& enc = encodings[i];
    enc.SetSequenceId(i);
    for (Encoding& overflowing : enc.GetOverflowingMut()) {
      overflowing.SetSequenceId(i);
    }
    enc.SetTypeIds(std::vector<uint32_t>(enc.Len(), static_cast<uint32_t>(i)));
  }

  ASSIGN_OR_RETURN(std::vector<Encoding> processed,
                   ProcessEncodings(std::move(encodings), add_special_tokens));
  return Encoding::Merge(std::move(processed), false);
}

absl::StatusOr<std::vector<Encoding>> PostProcessor::DefaultProcess(
    std::vector<Encoding> encodings, bool /*add_special_tokens*/) {
  switch (encodings.size()) {
    case 1:
      return encodings;
    default: {
      Encoding final_encoding;
      for (size_t i = 0; i < encodings.size(); ++i) {
        Encoding& encoding = encodings[i];
        encoding.SetSequenceId(i);
        final_encoding.MergeWith(std::move(encoding), false);
      }
      std::vector<Encoding> result;
      result.push_back(std::move(final_encoding));
      return result;
    }
  }
}

absl::StatusOr<std::string> Decoder::Decode(
    std::vector<std::string> tokens) const {
  ASSIGN_OR_RETURN(std::vector<std::string> results,
                   DecodeChain(std::move(tokens)));
  return absl::StrJoin(results, "");
}

}  // namespace tokenizers
