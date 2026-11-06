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
// C++ port of `src/decoders/bpe.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_BPE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_BPE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace decoders {

// Allows decoding Original BPE by joining all the tokens and then replacing
// the suffix used to identify end-of-words by whitespaces
//
// serde: `#[serde(tag = "type")]` -> `{"type":"BPEDecoder","suffix":"</w>"}`.
class BPEDecoder : public Decoder {
 public:
  // `impl Default for BPEDecoder`: `Self::new("</w>".into())`.
  BPEDecoder() : BPEDecoder("</w>") {}
  // `BPEDecoder::new`.
  explicit BPEDecoder(std::string suffix) : suffix_(std::move(suffix)) {}

  // `pub suffix: String`.
  const std::string& suffix() const { return suffix_; }
  void set_suffix(std::string suffix) { suffix_ = std::move(suffix); }

  // `impl Decoder for BPEDecoder`.
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

  static absl::StatusOr<BPEDecoder> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const BPEDecoder& a, const BPEDecoder& b) {
    return a.suffix_ == b.suffix_;
  }
  friend bool operator!=(const BPEDecoder& a, const BPEDecoder& b) {
    return !(a == b);
  }

 private:
  std::string suffix_;
};

}  // namespace decoders
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_BPE_H_
