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
// C++ port of `src/decoders/byte_fallback.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_BYTE_FALLBACK_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_BYTE_FALLBACK_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace decoders {

// ByteFallback is a simple trick which converts tokens looking like `<0x61>`
// to pure bytes, and attempts to make them into a string. If the tokens
// cannot be decoded you will get U+FFFD instead for each inconvertible byte
// token
//
// serde: the only field is `#[serde(rename = "type")] type_:
// MustBe!("ByteFallback")` -> `{"type":"ByteFallback"}`.
class ByteFallback : public Decoder {
 public:
  // `ByteFallback::new` / `impl Default`.
  ByteFallback() = default;

  // `impl Decoder for ByteFallback`.
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

  static absl::StatusOr<ByteFallback> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const ByteFallback&, const ByteFallback&) {
    return true;
  }
  friend bool operator!=(const ByteFallback&, const ByteFallback&) {
    return false;
  }
};

}  // namespace decoders
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_BYTE_FALLBACK_H_
