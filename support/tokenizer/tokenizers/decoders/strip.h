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
// C++ port of `src/decoders/strip.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_STRIP_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_STRIP_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace decoders {

// Strip is a simple trick which converts tokens looking like `<0x61>`
// to pure bytes, and attempts to make them into a string. If the tokens
// cannot be decoded you will get U+FFFD instead for each inconvertible byte
// token
//
// (The doc comment above is the one of the Rust original; the decoder
// actually strips up to `start` leading and `stop` trailing `content`
// characters from every token.)
//
// serde: `#[serde(tag = "type")]` ->
// `{"type":"Strip","content":"H","start":1,"stop":0}`.
class Strip : public Decoder {
 public:
  // `#[derive(Default)]`: `content = '\0'`, `start = 0`, `stop = 0`.
  Strip() : Strip(U'\0', 0, 0) {}
  // `Strip::new`.
  Strip(char32_t content, size_t start, size_t stop)
      : content_(content), start_(start), stop_(stop) {}

  char32_t content() const { return content_; }
  void set_content(char32_t content) { content_ = content; }
  size_t start() const { return start_; }
  void set_start(size_t start) { start_ = start; }
  size_t stop() const { return stop_; }
  void set_stop(size_t stop) { stop_ = stop; }

  // `impl Decoder for Strip`.
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

  static absl::StatusOr<Strip> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const Strip& a, const Strip& b) {
    return a.content_ == b.content_ && a.start_ == b.start_ &&
           a.stop_ == b.stop_;
  }
  friend bool operator!=(const Strip& a, const Strip& b) { return !(a == b); }

 private:
  char32_t content_;
  size_t start_;
  size_t stop_;
};

}  // namespace decoders
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_STRIP_H_
