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
// C++ port of `src/decoders/ctc.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_CTC_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_CTC_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace decoders {

// The CTC (Connectionist Temporal Classification) decoder takes care
// of sanitizing a list of inputs token.
// Due to some alignment problem the output of some models can come
// with duplicated token.
//
// serde: `#[serde(tag = "type")]` -> `{"type":"CTC","pad_token":"<pad>",
// "word_delimiter_token":"|","cleanup":true}`.
class CTC : public Decoder {
 public:
  // `impl Default for CTC`.
  CTC() : CTC("<pad>", "|", /*cleanup=*/true) {}
  // `CTC::new`.
  CTC(std::string pad_token, std::string word_delimiter_token, bool cleanup)
      : pad_token_(std::move(pad_token)),
        word_delimiter_token_(std::move(word_delimiter_token)),
        cleanup_(cleanup) {}

  // The pad token used by CTC to delimit a new token.
  const std::string& pad_token() const { return pad_token_; }
  void set_pad_token(std::string pad_token) {
    pad_token_ = std::move(pad_token);
  }
  // The word delimiter token. It will be replaced by a `<space>`.
  const std::string& word_delimiter_token() const {
    return word_delimiter_token_;
  }
  void set_word_delimiter_token(std::string word_delimiter_token) {
    word_delimiter_token_ = std::move(word_delimiter_token);
  }
  // Whether to cleanup some tokenization artifacts.
  // Mainly spaces before punctuation, and some abbreviated english forms.
  bool cleanup() const { return cleanup_; }
  void set_cleanup(bool cleanup) { cleanup_ = cleanup; }

  // `impl Decoder for CTC`.
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

  static absl::StatusOr<CTC> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const CTC& a, const CTC& b) {
    return a.pad_token_ == b.pad_token_ &&
           a.word_delimiter_token_ == b.word_delimiter_token_ &&
           a.cleanup_ == b.cleanup_;
  }
  friend bool operator!=(const CTC& a, const CTC& b) { return !(a == b); }

 private:
  std::string pad_token_;
  std::string word_delimiter_token_;
  bool cleanup_;
};

}  // namespace decoders
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_CTC_H_
