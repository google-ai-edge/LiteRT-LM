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
// C++ port of `src/utils/truncation.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TRUNCATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TRUNCATION_H_

#include <cstddef>
#include <optional>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {

// `enum class TruncationDirection { kLeft, kRight }` is defined in encoding.h
// (see the PORT NOTE there). `#[default]` is `kRight`.

// `impl AsRef<str> for TruncationDirection`: "left" / "right".
absl::string_view TruncationDirectionAsStr(TruncationDirection direction);
// `#[derive(Serialize, Deserialize)]`: "Left" / "Right".
absl::StatusOr<TruncationDirection> TruncationDirectionFromJson(
    const Json& json);
Json TruncationDirectionToJson(TruncationDirection direction);

// `#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Eq,
// Default)] pub enum TruncationStrategy { #[default] LongestFirst, OnlyFirst,
// OnlySecond }`
enum class TruncationStrategy { kLongestFirst, kOnlyFirst, kOnlySecond };

// `impl AsRef<str> for TruncationStrategy`: "longest_first" / "only_first" /
// "only_second".
absl::string_view TruncationStrategyAsStr(TruncationStrategy strategy);
// `#[derive(Serialize, Deserialize)]`: "LongestFirst" / "OnlyFirst" /
// "OnlySecond".
absl::StatusOr<TruncationStrategy> TruncationStrategyFromJson(const Json& json);
Json TruncationStrategyToJson(TruncationStrategy strategy);

// `#[derive(Debug, Clone, Serialize, Deserialize)] pub struct TruncationParams`
struct TruncationParams {
  // `impl Default for TruncationParams`
  // `#[serde(default)]`
  TruncationDirection direction = TruncationDirection::kRight;
  size_t max_length = 512;
  TruncationStrategy strategy = TruncationStrategy::kLongestFirst;
  size_t stride = 0;

  static absl::StatusOr<TruncationParams> FromJson(const Json& json);
  Json ToJson() const;
};

// `#[derive(thiserror::Error, Debug)] pub enum TruncationError`
struct TruncationError {
  // We are supposed to truncate the pair sequence, but it has not been
  // provided.
  static absl::Status SecondSequenceNotProvided() {
    return absl::InvalidArgumentError(
        "Truncation error: Second sequence not provided");
  }
  // We cannot truncate the target sequence enough to respect the provided max
  // length.
  static absl::Status SequenceTooShort() {
    return absl::InvalidArgumentError(
        "Truncation error: Sequence to truncate too short to respect the "
        "provided max_length");
  }
};

// `pub fn truncate_encodings(encoding, pair_encoding, params) ->
// Result<(Encoding, Option<Encoding>)>`
absl::StatusOr<std::pair<Encoding, std::optional<Encoding>>> TruncateEncodings(
    Encoding encoding, std::optional<Encoding> pair_encoding,
    const TruncationParams& params);

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TRUNCATION_H_
