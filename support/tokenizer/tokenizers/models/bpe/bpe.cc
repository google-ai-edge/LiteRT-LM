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

#include "support/tokenizer/tokenizers/models/bpe/bpe.h"

#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers::models::bpe {

absl::Status Error::Io(absl::string_view message) {
  return absl::NotFoundError(absl::StrCat("IoError: ", message));
}

absl::Status Error::JsonError(absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat("JsonError: ", message));
}

absl::Status Error::BadVocabulary() {
  return absl::InvalidArgumentError("Bad vocabulary json file");
}

absl::Status Error::BadMerges(size_t line) {
  return absl::InvalidArgumentError(
      absl::StrCat("Merges text file invalid at line ", line));
}

absl::Status Error::MergeTokenOutOfVocabulary(absl::string_view token) {
  return absl::InvalidArgumentError(
      absl::StrCat("Token `", token, "` out of vocabulary"));
}

absl::Status Error::UnkTokenOutOfVocabulary(absl::string_view token) {
  return absl::InvalidArgumentError(
      absl::StrCat("Unk token `", token, "` not found in the vocabulary"));
}

absl::Status Error::InvalidDropout() {
  return absl::InvalidArgumentError(
      "Dropout should be between 0 and 1, inclusive");
}

}  // namespace tokenizers::models::bpe
