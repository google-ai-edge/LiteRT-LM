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
// C++ port of `src/models/bpe/word.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_WORD_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_WORD_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/bpe.h"

namespace tokenizers::models::bpe {

struct Symbol {
  uint32_t c;
  int64_t prev;
  int64_t next;
  size_t len;

  // Merges the current Symbol with the other one.
  // In order to update prev/next, we consider Self to be the Symbol on the
  // left, and other to be the next one on the right.
  void MergeWith(const Symbol& other, uint32_t new_c) {
    c = new_c;
    len += other.len;
    next = other.next;
  }
};

class Word {
 public:
  Word() = default;

  static Word New() { return Word(); }

  static Word WithCapacity(size_t capacity);

  void Add(uint32_t c, size_t byte_len);

  std::vector<std::pair<Pair, int32_t>> Merge(uint32_t c1, uint32_t c2,
                                              uint32_t replacement,
                                              size_t max_length);

  void MergeAll(
      const absl::flat_hash_map<Pair, std::pair<uint32_t, uint32_t>>& merges,
      std::optional<float> dropout);

  std::vector<uint32_t> GetChars() const;

  // PORT NOTE: `get_chars_iter` / `get_offsets_iter` are exposed as the
  // underlying symbols plus `GetOffsets()`; callers iterate `symbols()`.
  std::vector<std::pair<size_t, size_t>> GetOffsets() const;

  const std::vector<Symbol>& symbols() const { return symbols_; }

 private:
  std::vector<Symbol> symbols_;
};

}  // namespace tokenizers::models::bpe

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_WORD_H_
