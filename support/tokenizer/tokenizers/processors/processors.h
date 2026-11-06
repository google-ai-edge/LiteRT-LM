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
// C++ port of `src/processors/mod.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_PROCESSORS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_PROCESSORS_H_

#include <cstddef>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/processors/bert.h"
#include "support/tokenizer/tokenizers/processors/roberta.h"
#include "support/tokenizer/tokenizers/processors/sequence.h"
#include "support/tokenizer/tokenizers/processors/template.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace processors {

// Re-export these as processors (`pub use super::pre_tokenizers::byte_level;`).
using ::tokenizers::pre_tokenizers::ByteLevel;

// `#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Eq)]`
// `#[serde(untagged)]`
// `pub enum PostProcessorWrapper`.
class PostProcessorWrapper : public PostProcessor {
 public:
  // Roberta must be before Bert for deserialization (serde does not validate
  // tags).
  using Variant = std::variant<RobertaProcessing, BertProcessing, ByteLevel,
                               TemplateProcessing, Sequence>;

  // `impl_enum_from!(X, PostProcessorWrapper, Variant)` for every variant.
  PostProcessorWrapper(  // NOLINT(google-explicit-constructor)
      RobertaProcessing processor)
      : variant_(std::move(processor)) {}
  PostProcessorWrapper(  // NOLINT(google-explicit-constructor)
      BertProcessing processor)
      : variant_(std::move(processor)) {}
  PostProcessorWrapper(  // NOLINT(google-explicit-constructor)
      ByteLevel processor)
      : variant_(std::move(processor)) {}
  PostProcessorWrapper(  // NOLINT(google-explicit-constructor)
      TemplateProcessing processor)
      : variant_(std::move(processor)) {}
  PostProcessorWrapper(  // NOLINT(google-explicit-constructor)
      Sequence processor)
      : variant_(std::move(processor)) {}

  // The wrapped post-processor (`match self { Self::Bert(bert) => ... }`).
  const Variant& variant() const { return variant_; }
  Variant& variant() { return variant_; }
  template <typename T>
  bool Is() const {
    return std::holds_alternative<T>(variant_);
  }
  template <typename T>
  const T* GetIf() const {
    return std::get_if<T>(&variant_);
  }
  template <typename T>
  T* GetIf() {
    return std::get_if<T>(&variant_);
  }

  // `impl PostProcessor for PostProcessorWrapper`.
  size_t AddedTokens(bool is_pair) const override;
  absl::StatusOr<std::vector<Encoding>> ProcessEncodings(
      std::vector<Encoding> encodings, bool add_special_tokens) const override;

  // `#[serde(untagged)]`: every variant is tried in declaration order and the
  // first success wins; fails with "data did not match any variant of untagged
  // enum PostProcessorWrapper".
  static absl::StatusOr<PostProcessorWrapper> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const PostProcessorWrapper& a,
                         const PostProcessorWrapper& b) {
    return a.variant_ == b.variant_;
  }
  friend bool operator!=(const PostProcessorWrapper& a,
                         const PostProcessorWrapper& b) {
    return !(a == b);
  }

 private:
  Variant variant_;
};

}  // namespace processors
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_PROCESSORS_H_
