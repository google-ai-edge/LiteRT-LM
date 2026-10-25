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
// C++ port of `src/normalizers/unicode.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_UNICODE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_UNICODE_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::normalizers {

// `#[derive(Default, Copy, Clone, Debug)]
// #[macro_rules_attribute(impl_serde_type!)]`
class NFD : public Normalizer {
 public:
  NFD() = default;

  static absl::StatusOr<NFD> FromJson(const Json& json);
  Json ToJson() const;

  absl::Status Normalize(NormalizedString& normalized) const override;
};

class NFKD : public Normalizer {
 public:
  NFKD() = default;

  static absl::StatusOr<NFKD> FromJson(const Json& json);
  Json ToJson() const;

  absl::Status Normalize(NormalizedString& normalized) const override;
};

class NFC : public Normalizer {
 public:
  NFC() = default;

  static absl::StatusOr<NFC> FromJson(const Json& json);
  Json ToJson() const;

  absl::Status Normalize(NormalizedString& normalized) const override;
};

class NFKC : public Normalizer {
 public:
  NFKC() = default;

  static absl::StatusOr<NFKC> FromJson(const Json& json);
  Json ToJson() const;

  absl::Status Normalize(NormalizedString& normalized) const override;
};

class Nmt : public Normalizer {
 public:
  Nmt() = default;

  static absl::StatusOr<Nmt> FromJson(const Json& json);
  Json ToJson() const;

  absl::Status Normalize(NormalizedString& normalized) const override;
};

}  // namespace tokenizers::normalizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_UNICODE_H_
