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
// C++ port of `src/pre_tokenizers/sequence.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_SEQUENCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_SEQUENCE_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace pre_tokenizers {

class PreTokenizerWrapper;

// Runs a list of pre-tokenizers in sequence.
//
// serde (`impl_serde_type!`): `{"type":"Sequence","pretokenizers":[...]}`.
// `PreTokenizerWrapper` is only forward-declared here (it contains `Sequence`
// in its variant); all special members are defined in sequence.cc.
class Sequence : public PreTokenizer {
 public:
  explicit Sequence(std::vector<PreTokenizerWrapper> pretokenizers);
  ~Sequence() override;
  Sequence(const Sequence& other);
  Sequence& operator=(const Sequence& other);
  Sequence(Sequence&& other) noexcept;
  Sequence& operator=(Sequence&& other) noexcept;

  // `impl AsRef<[PreTokenizerWrapper]>` / `impl AsMut<[PreTokenizerWrapper]>`.
  const std::vector<PreTokenizerWrapper>& pretokenizers() const {
    return pretokenizers_;
  }
  std::vector<PreTokenizerWrapper>& mutable_pretokenizers() {
    return pretokenizers_;
  }

  absl::Status PreTokenize(PreTokenizedString& pretokenized) const override;

  static absl::StatusOr<Sequence> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const Sequence& a, const Sequence& b);
  friend bool operator!=(const Sequence& a, const Sequence& b) {
    return !(a == b);
  }

 private:
  std::vector<PreTokenizerWrapper> pretokenizers_;
};

}  // namespace pre_tokenizers
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_SEQUENCE_H_
