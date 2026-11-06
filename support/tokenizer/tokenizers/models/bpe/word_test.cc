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
// C++ port of `src/models/bpe/word.rs` (`#[cfg(test)] mod tests`) from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/bpe/word.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/models/bpe/bpe.h"

namespace tokenizers::models::bpe {
namespace {

TEST(WordTest, TestMerge) {
  // Let's say we have the word 'hello' and a word-to-id vocab that looks
  // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
  Word word = Word::New();
  word.Add(0, 1);  // 'h'
  word.Add(1, 1);  // 'e'
  word.Add(2, 1);  // 'l'
  word.Add(2, 1);  // 'l'
  word.Add(3, 1);  // 'o'

  // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
  // say that 'll' has the ID of 4 in the updated word-to-id vocab.
  const std::vector<std::pair<Pair, int32_t>> changes =
      word.Merge(2, 2, 4, std::numeric_limits<size_t>::max());

  // So the word should now look like this:
  EXPECT_EQ(word.GetChars(), (std::vector<uint32_t>{
                                 0u,  // 'h'
                                 1u,  // 'e'
                                 4u,  // 'll'
                                 3u,  // 'o'
                             }));

  // The return value `changes` will be used to update the pair counts during
  // training. This merge affects the counts for the pairs
  // ('e', 'l') ~= (1, 2),
  // ('e', 'll') ~= (1, 4),
  // ('l', 'o') ~= (2, 3), and
  // ('ll', 'o') ~= (4, 3).
  // So the changes should reflect that:
  EXPECT_EQ(changes, (std::vector<std::pair<Pair, int32_t>>{
                         {{1u, 2u}, -1},  // count for ('e', 'l') decreased by 1
                         {{1u, 4u}, 1},  // count for ('e', 'll') increased by 1
                         {{2u, 3u}, -1},  // count for ('l', 'o') decreased by 1
                         {{4u, 3u}, 1},  // count for ('ll', 'o') increased by 1
                     }));
}

TEST(WordTest, TestMergeMaxLength) {
  // Let's say we have the word 'hello' and a word-to-id vocab that looks
  // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
  Word word = Word::New();
  word.Add(0, 1);  // 'h'
  word.Add(1, 1);  // 'e'
  word.Add(2, 1);  // 'l'
  word.Add(2, 1);  // 'l'
  word.Add(3, 1);  // 'o'

  // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
  // say that 'll' has the ID of 4 in the updated word-to-id vocab.
  const std::vector<std::pair<Pair, int32_t>> changes = word.Merge(2, 2, 4, 2);
  EXPECT_EQ(word.GetChars(), (std::vector<uint32_t>{
                                 0u,  // 'h'
                                 1u,  // 'e'
                                 4u,  // 'll'
                                 3u,  // 'o'
                             }));

  EXPECT_EQ(changes, (std::vector<std::pair<Pair, int32_t>>{
                         {{1u, 2u}, -1},  // count for ('e', 'l') decreased by 1
                         // ((1u32, 4u32), 1i32), Missing since larger than 2
                         {{2u, 3u}, -1},  // count for ('l', 'o') decreased by 1
                         // ((4u32, 3u32), 1i32), Missing since larger than 2
                     }));
}

}  // namespace
}  // namespace tokenizers::models::bpe
