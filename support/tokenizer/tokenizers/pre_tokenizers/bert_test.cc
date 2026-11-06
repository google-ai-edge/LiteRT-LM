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
// C++ port of the tests of `src/pre_tokenizers/bert.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/bert.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace pre_tokenizers {
namespace {

using SplitList = std::vector<std::pair<std::string, Offsets>>;

// `pretokenized.get_splits(ref, type).into_iter().map(|(s, o, _)| (s, o))`
SplitList Splits(const PreTokenizedString& pretokenized,
                 OffsetReferential offset_ref, OffsetType offset_type) {
  SplitList result;
  for (const auto& split : pretokenized.GetSplits(offset_ref, offset_type)) {
    result.emplace_back(std::string(std::get<0>(split)), std::get<1>(split));
  }
  return result;
}

TEST(BertPreTokenizerTest, Basic) {
  const BertPreTokenizer pretok;
  PreTokenizedString pretokenized =
      PreTokenizedString::From("Hey friend!     How are you?!?");
  ASSERT_OK(pretok.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({
          {"Hey", {0, 3}},
          {"friend", {4, 10}},
          {"!", {10, 11}},
          {"How", {16, 19}},
          {"are", {20, 23}},
          {"you", {24, 27}},
          {"?", {27, 28}},
          {"!", {28, 29}},
          {"?", {29, 30}},
      }));
}

TEST(BertPreTokenizerTest, ChineseChars) {
  NormalizedString n = NormalizedString::From("野口里佳 Noguchi Rika");
  std::vector<std::pair<char32_t, int64_t>> transformations;
  for (const auto [offset, c] : CharIndices(n.Get())) {
    if (static_cast<uint32_t>(c) > 0x4E00) {
      transformations.emplace_back(U' ', 0);
      transformations.emplace_back(c, 1);
      transformations.emplace_back(U' ', 1);
    } else {
      transformations.emplace_back(c, 0);
    }
  }
  n.Transform(transformations, 0);
  PreTokenizedString pretokenized(std::move(n));
  const BertPreTokenizer pretok;
  ASSERT_OK(pretok.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({{"野", {0, 3}},
                 {"口", {3, 6}},
                 {"里", {6, 9}},
                 {"佳", {9, 12}},
                 {"Noguchi", {13, 20}},
                 {"Rika", {21, 25}}}));
}

}  // namespace
}  // namespace pre_tokenizers
}  // namespace tokenizers
