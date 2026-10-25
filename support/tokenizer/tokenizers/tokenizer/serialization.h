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
// C++ port of `src/tokenizer/serialization.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// `impl Serialize for TokenizerImpl` and `impl Deserialize for TokenizerImpl`
// are the member functions `Tokenizer::ToJson()` and `Tokenizer::FromJson()`
// declared in tokenizer.h; they are implemented in serialization.cc. This
// header only exposes the constants shared with the tests.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_SERIALIZATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_SERIALIZATION_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"  // IWYU pragma: export

namespace tokenizers {

// `static SERIALIZATION_VERSION: &str = "1.0";`
inline constexpr absl::string_view kSerializationVersion = "1.0";

// The fields of `struct Tokenizer`, in serialization order.
inline constexpr absl::string_view kTokenizerFields[] = {
    "version",       "truncation",     "padding", "added_tokens", "normalizer",
    "pre_tokenizer", "post_processor", "decoder", "model",
};

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_SERIALIZATION_H_
