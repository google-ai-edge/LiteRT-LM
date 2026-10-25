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
// C++ port of `src/normalizers/utils.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_UTILS_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::normalizers {

class NormalizerWrapper;

// Allows concatenating multiple other Normalizer as a Sequence.
// All the normalizers run in sequence in the given order against the same
// NormalizedString.
//
// `#[derive(Clone, Deserialize, Debug, Serialize)] #[serde(tag = "type")]`
class Sequence : public Normalizer {
 public:
  explicit Sequence(std::vector<NormalizerWrapper> normalizers);
  // PORT NOTE: the special members are defined out of line in utils.cc because
  // `NormalizerWrapper` (which itself contains `Sequence` in its variant) is
  // only forward-declared here.
  Sequence(const Sequence& other);
  Sequence(Sequence&& other) noexcept;
  Sequence& operator=(const Sequence& other);
  Sequence& operator=(Sequence&& other) noexcept;
  ~Sequence() override;

  static absl::StatusOr<Sequence> FromJson(const Json& json);
  Json ToJson() const;

  // `impl AsRef<[NormalizerWrapper]> for Sequence`
  const std::vector<NormalizerWrapper>& Normalizers() const {
    return normalizers_;
  }
  // `impl AsMut<[NormalizerWrapper]> for Sequence`
  std::vector<NormalizerWrapper>& MutableNormalizers() { return normalizers_; }
  // `impl IntoIterator for Sequence`
  std::vector<NormalizerWrapper> IntoNormalizers() &&;

  absl::Status Normalize(NormalizedString& normalized) const override;

 private:
  std::vector<NormalizerWrapper> normalizers_;
};

// Lowercases the input
//
// `#[derive(Copy, Clone, Debug)] #[macro_rules_attribute(impl_serde_type!)]`
class Lowercase : public Normalizer {
 public:
  Lowercase() = default;

  static absl::StatusOr<Lowercase> FromJson(const Json& json);
  Json ToJson() const;

  absl::Status Normalize(NormalizedString& normalized) const override;
};

}  // namespace tokenizers::normalizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_UTILS_H_
