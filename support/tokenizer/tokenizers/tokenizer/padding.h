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
// C++ port of `src/utils/padding.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PADDING_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PADDING_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {

// `enum class PaddingDirection { kLeft, kRight }` is defined in encoding.h
// (see the PORT NOTE there).

// `impl AsRef<str> for PaddingDirection`: "left" / "right".
absl::string_view PaddingDirectionAsStr(PaddingDirection direction);
// `#[derive(Serialize, Deserialize)]`: "Left" / "Right".
absl::StatusOr<PaddingDirection> PaddingDirectionFromJson(const Json& json);
Json PaddingDirectionToJson(PaddingDirection direction);

// `#[derive(Debug, Clone, Serialize, Deserialize)] pub enum PaddingStrategy {
// BatchLongest, Fixed(usize) }`
struct PaddingStrategy {
  enum class Kind { kBatchLongest, kFixed };

  static PaddingStrategy BatchLongest() { return {Kind::kBatchLongest, 0}; }
  static PaddingStrategy Fixed(size_t size) { return {Kind::kFixed, size}; }

  // `"BatchLongest"` or `{"Fixed": <size>}`.
  static absl::StatusOr<PaddingStrategy> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const PaddingStrategy& a, const PaddingStrategy& b) {
    return a.kind == b.kind && (a.kind != Kind::kFixed || a.size == b.size);
  }
  friend bool operator!=(const PaddingStrategy& a, const PaddingStrategy& b) {
    return !(a == b);
  }

  Kind kind = Kind::kBatchLongest;
  // Only meaningful for `Kind::kFixed`.
  size_t size = 0;
};

// `#[derive(Debug, Clone, Serialize, Deserialize)] pub struct PaddingParams`
struct PaddingParams {
  // `impl Default for PaddingParams`
  PaddingStrategy strategy = PaddingStrategy::BatchLongest();
  PaddingDirection direction = PaddingDirection::kRight;
  std::optional<size_t> pad_to_multiple_of;
  uint32_t pad_id = 0;
  uint32_t pad_type_id = 0;
  std::string pad_token = "[PAD]";

  static absl::StatusOr<PaddingParams> FromJson(const Json& json);
  Json ToJson() const;
};

// `pub fn pad_encodings(encodings: &mut [Encoding], params: &PaddingParams)`
absl::Status PadEncodings(absl::Span<Encoding> encodings,
                          const PaddingParams& params);

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PADDING_H_
