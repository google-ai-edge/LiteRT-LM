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
// C++ port of `src/tokenizer/mod.rs` (the `Offsets` and `Token` definitions)
// from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: in Rust `Offsets` and `Token` live in `tokenizer/mod.rs` together
// with the traits (see api.h). They are split into this leaf header because
// `Encoding` and `PreTokenizedString` need them while the default trait
// methods in api.cc need `Encoding` / `PreTokenizedString`; a single Rust
// module may be cyclic like that, Bazel targets may not.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TOKEN_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TOKEN_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace tokenizers {

// `pub type Offsets = (usize, usize);`
using Offsets = std::pair<size_t, size_t>;

// `#[derive(Debug, Clone, PartialEq, Eq)] pub struct Token`
struct Token {
  uint32_t id = 0;
  std::string value;
  Offsets offsets = {0, 0};

  Token() = default;
  // `Token::new(id, value, offsets)`
  Token(uint32_t id, std::string value, Offsets offsets)
      : id(id), value(std::move(value)), offsets(offsets) {}

  friend bool operator==(const Token& a, const Token& b) {
    return a.id == b.id && a.value == b.value && a.offsets == b.offsets;
  }
  friend bool operator!=(const Token& a, const Token& b) { return !(a == b); }
};

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TOKEN_H_
