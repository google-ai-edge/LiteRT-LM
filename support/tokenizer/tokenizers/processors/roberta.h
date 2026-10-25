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
// C++ port of `src/processors/roberta.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_ROBERTA_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_ROBERTA_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace processors {

// serde: `#[serde(tag = "type")]` -> `{"type":"RobertaProcessing",
// "sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,
// "add_prefix_space":true}`.
class RobertaProcessing : public PostProcessor {
 public:
  // `impl Default for RobertaProcessing`.
  RobertaProcessing()
      : sep_("</s>", 2),
        cls_("<s>", 0),
        trim_offsets_(true),
        add_prefix_space_(true) {}
  // `RobertaProcessing::new`: `sep`, `cls` and the other fields of
  // `Default::default()`.
  RobertaProcessing(std::pair<std::string, uint32_t> sep,
                    std::pair<std::string, uint32_t> cls)
      : RobertaProcessing() {
    sep_ = std::move(sep);
    cls_ = std::move(cls);
  }

  // Builder-style setters (`#[must_use] fn trim_offsets(mut self, v) -> Self`
  // etc.): return a modified copy.
  RobertaProcessing TrimOffsets(bool v) const {
    RobertaProcessing copy = *this;
    copy.trim_offsets_ = v;
    return copy;
  }
  RobertaProcessing AddPrefixSpace(bool v) const {
    RobertaProcessing copy = *this;
    copy.add_prefix_space_ = v;
    return copy;
  }

  // `pub sep: (String, u32)` / `pub cls: (String, u32)`.
  const std::pair<std::string, uint32_t>& sep() const { return sep_; }
  const std::pair<std::string, uint32_t>& cls() const { return cls_; }
  // `pub trim_offsets: bool` / `pub add_prefix_space: bool`.
  bool trim_offsets() const { return trim_offsets_; }
  bool add_prefix_space() const { return add_prefix_space_; }

  std::pair<std::string, uint32_t> GetSepCopy() const { return sep_; }
  std::pair<std::string, uint32_t> GetClsCopy() const { return cls_; }

  // `impl PostProcessor for RobertaProcessing`.
  size_t AddedTokens(bool is_pair) const override;
  absl::StatusOr<std::vector<Encoding>> ProcessEncodings(
      std::vector<Encoding> encodings, bool add_special_tokens) const override;

  static absl::StatusOr<RobertaProcessing> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const RobertaProcessing& a,
                         const RobertaProcessing& b) {
    return a.sep_ == b.sep_ && a.cls_ == b.cls_ &&
           a.trim_offsets_ == b.trim_offsets_ &&
           a.add_prefix_space_ == b.add_prefix_space_;
  }
  friend bool operator!=(const RobertaProcessing& a,
                         const RobertaProcessing& b) {
    return !(a == b);
  }

 private:
  std::pair<std::string, uint32_t> sep_;
  std::pair<std::string, uint32_t> cls_;
  bool trim_offsets_;
  bool add_prefix_space_;
};

}  // namespace processors
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_ROBERTA_H_
