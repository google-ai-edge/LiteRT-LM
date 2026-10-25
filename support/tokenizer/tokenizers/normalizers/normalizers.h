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
// C++ port of `src/normalizers/mod.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_NORMALIZERS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_NORMALIZERS_H_

#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/normalizers/bert.h"
#include "support/tokenizer/tokenizers/normalizers/byte_level.h"
#include "support/tokenizer/tokenizers/normalizers/precompiled.h"
#include "support/tokenizer/tokenizers/normalizers/prepend.h"
#include "support/tokenizer/tokenizers/normalizers/replace.h"
#include "support/tokenizer/tokenizers/normalizers/spm_precompiled.h"
#include "support/tokenizer/tokenizers/normalizers/strip.h"
#include "support/tokenizer/tokenizers/normalizers/unicode.h"
#include "support/tokenizer/tokenizers/normalizers/utils.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::normalizers {

// Wrapper for known Normalizers.
//
// `#[derive(Clone, Debug, Serialize)] #[serde(untagged)] pub enum
// NormalizerWrapper { ... }` with a hand-written `Deserialize`.
class NormalizerWrapper : public Normalizer {
 public:
  // Variants in the Rust declaration order.
  using Variant = std::variant<BertNormalizer, Strip, StripAccents, NFC, NFD,
                               NFKC, NFKD, Sequence, Lowercase, Nmt,
                               Precompiled, Replace, Prepend, ByteLevel>;

  // `impl_enum_from!(X, NormalizerWrapper, Variant)` for every variant.
  NormalizerWrapper(BertNormalizer n) : variant_(std::move(n)) {}  // NOLINT
  NormalizerWrapper(Strip n) : variant_(std::move(n)) {}           // NOLINT
  NormalizerWrapper(StripAccents n) : variant_(std::move(n)) {}    // NOLINT
  NormalizerWrapper(NFC n) : variant_(std::move(n)) {}             // NOLINT
  NormalizerWrapper(NFD n) : variant_(std::move(n)) {}             // NOLINT
  NormalizerWrapper(NFKC n) : variant_(std::move(n)) {}            // NOLINT
  NormalizerWrapper(NFKD n) : variant_(std::move(n)) {}            // NOLINT
  NormalizerWrapper(Sequence n) : variant_(std::move(n)) {}        // NOLINT
  NormalizerWrapper(Lowercase n) : variant_(std::move(n)) {}       // NOLINT
  NormalizerWrapper(Nmt n) : variant_(std::move(n)) {}             // NOLINT
  NormalizerWrapper(Precompiled n) : variant_(std::move(n)) {}     // NOLINT
  NormalizerWrapper(Replace n) : variant_(std::move(n)) {}         // NOLINT
  NormalizerWrapper(Prepend n) : variant_(std::move(n)) {}         // NOLINT
  NormalizerWrapper(ByteLevel n) : variant_(std::move(n)) {}       // NOLINT

  // `impl<'de> Deserialize<'de> for NormalizerWrapper`: first tries the
  // `type`-tagged form (`{"type": "Bert" | "Strip" | ... , ...}`), then falls
  // back to trying every variant in order on the untagged value.
  static absl::StatusOr<NormalizerWrapper> FromJson(const Json& json);
  // `#[serde(untagged)]` serialization: the variant's own representation.
  Json ToJson() const;

  // `impl Normalizer for NormalizerWrapper`
  absl::Status Normalize(NormalizedString& normalized) const override;

  // Variant access (`matches!(w, NormalizerWrapper::X(_))` / `if let`).
  const Variant& variant() const { return variant_; }
  Variant& variant() { return variant_; }
  template <typename T>
  bool Is() const {
    return std::holds_alternative<T>(variant_);
  }
  template <typename T>
  const T& Get() const {
    return std::get<T>(variant_);
  }
  template <typename T>
  T& Get() {
    return std::get<T>(variant_);
  }

 private:
  Variant variant_;
};

}  // namespace tokenizers::normalizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_NORMALIZERS_H_
