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
// C++ port of `src/processors/bert.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_BERT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_BERT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace processors {

// serde: `#[serde(tag = "type")]` ->
// `{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}`.
class BertProcessing : public PostProcessor {
 public:
  // `impl Default for BertProcessing`.
  BertProcessing() : BertProcessing({"[SEP]", 102}, {"[CLS]", 101}) {}
  // `BertProcessing::new`.
  BertProcessing(std::pair<std::string, uint32_t> sep,
                 std::pair<std::string, uint32_t> cls)
      : sep_(std::move(sep)), cls_(std::move(cls)) {}

  // `pub sep: (String, u32)` / `pub cls: (String, u32)`.
  const std::pair<std::string, uint32_t>& sep() const { return sep_; }
  const std::pair<std::string, uint32_t>& cls() const { return cls_; }

  std::pair<std::string, uint32_t> GetSepCopy() const { return sep_; }
  std::pair<std::string, uint32_t> GetClsCopy() const { return cls_; }

  // `impl PostProcessor for BertProcessing`.
  size_t AddedTokens(bool is_pair) const override;
  absl::StatusOr<std::vector<Encoding>> ProcessEncodings(
      std::vector<Encoding> encodings, bool add_special_tokens) const override;

  static absl::StatusOr<BertProcessing> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const BertProcessing& a, const BertProcessing& b) {
    return a.sep_ == b.sep_ && a.cls_ == b.cls_;
  }
  friend bool operator!=(const BertProcessing& a, const BertProcessing& b) {
    return !(a == b);
  }

 private:
  std::pair<std::string, uint32_t> sep_;
  std::pair<std::string, uint32_t> cls_;
};

// `#[derive(thiserror::Error, Debug)] pub enum BertProcessorError`
struct BertProcessorError {
  // `#[error("encodings vector length must be either 1 or 2")]
  // InvalidEncodingsVecLength`
  static absl::Status InvalidEncodingsVecLength() {
    return absl::InvalidArgumentError(
        "encodings vector length must be either 1 or 2");
  }
};

}  // namespace processors
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_BERT_H_
