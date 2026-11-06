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
// C++ port of `src/models/bpe/mod.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_BPE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_BPE_H_

// [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
//
// PORT NOTE: the Rust module is split over bpe.h (this file: `Pair` and the
// `Error` enum), word.h (`Word`), model.h (`BPE`, `BpeBuilder`, the type
// aliases) and serialization.cc (serde). `trainer.rs` and `parity_trainer.rs`
// are not ported (training is out of scope), and so is the `FirstLastIterator`
// helper which is only used by the trainers.

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers::models::bpe {

using Pair = std::pair<uint32_t, uint32_t>;

// Errors that can be encountered while using or constructing a `BPE` model.
//
// PORT NOTE: the Rust `thiserror` enum becomes a set of `absl::Status`
// factories carrying the same messages.
struct Error {
  // An error encountered while reading files mainly.
  // "IoError: {0}"
  static absl::Status Io(absl::string_view message);
  // An error forwarded from Serde, while parsing JSON
  // "JsonError: {0}"
  static absl::Status JsonError(absl::string_view message);
  // When the vocab.json file is in the wrong format
  // "Bad vocabulary json file"
  static absl::Status BadVocabulary();
  // When the merges.txt file is in the wrong format. This error holds the line
  // number of the line that caused the error.
  // "Merges text file invalid at line {0}"
  static absl::Status BadMerges(size_t line);
  // If a token found in merges, is not in the vocab
  // "Token `{0}` out of vocabulary"
  static absl::Status MergeTokenOutOfVocabulary(absl::string_view token);
  // If the provided unk token is out of vocabulary
  // "Unk token `{0}` not found in the vocabulary"
  static absl::Status UnkTokenOutOfVocabulary(absl::string_view token);
  // Dropout not between 0 and 1.
  // "Dropout should be between 0 and 1, inclusive"
  static absl::Status InvalidDropout();
};

}  // namespace tokenizers::models::bpe

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_BPE_H_
