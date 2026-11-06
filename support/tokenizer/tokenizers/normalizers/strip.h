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
// C++ port of `src/normalizers/strip.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_STRIP_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_STRIP_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::normalizers {

// `#[derive(Copy, Clone, Debug, Deserialize, Serialize)] #[serde(tag =
// "type")]`
class Strip : public Normalizer {
 public:
  Strip(bool strip_left, bool strip_right)
      : strip_left_(strip_left), strip_right_(strip_right) {}

  static absl::StatusOr<Strip> FromJson(const Json& json);
  Json ToJson() const;

  bool strip_left() const { return strip_left_; }
  void set_strip_left(bool strip_left) { strip_left_ = strip_left; }
  bool strip_right() const { return strip_right_; }
  void set_strip_right(bool strip_right) { strip_right_ = strip_right; }

  // Strip the normalized string inplace
  absl::Status Normalize(NormalizedString& normalized) const override;

 private:
  bool strip_left_;
  bool strip_right_;
};

// This normalizer removes combining marks from a normalized string
// It's different from unidecode as it does not attempt to modify
// non ascii languages.
//
// `#[derive(Copy, Clone, Debug)] #[macro_rules_attribute(impl_serde_type!)]`
class StripAccents : public Normalizer {
 public:
  StripAccents() = default;

  static absl::StatusOr<StripAccents> FromJson(const Json& json);
  Json ToJson() const;

  // Strip the normalized string inplace
  absl::Status Normalize(NormalizedString& normalized) const override;
};

}  // namespace tokenizers::normalizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_STRIP_H_
