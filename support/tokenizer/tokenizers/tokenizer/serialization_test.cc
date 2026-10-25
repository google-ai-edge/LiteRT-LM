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
// C++ port of `#[cfg(test)] mod tests` in `src/tokenizer/serialization.rs`
// from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: `test_from_pretrained` (`#[cfg(feature = "http")]`) is not ported
// because HTTP Hub downloads (`Tokenizer::from_pretrained`) are out of scope.

#include "support/tokenizer/tokenizers/tokenizer/serialization.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

TEST(SerializationTest, DeserializationSerializationInvariant) {
  const char* tok_json = R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 0,
      "content": "[SPECIAL_0]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 1,
      "content": "[SPECIAL_1]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 2,
      "content": "[SPECIAL_2]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "",
    "max_input_chars_per_word": 100,
    "vocab": {}
  }
})";
  ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer, Tokenizer::FromStr(tok_json));

  ASSERT_OK_AND_ASSIGN(std::string tok_str, tokenizer.ToString(true));
  // It should be exactly the same as above
  EXPECT_EQ(tok_str, tok_json);
}

}  // namespace
}  // namespace tokenizers
