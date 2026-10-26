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
// C++ port of the tests of `src/pre_tokenizers/unicode_scripts/
// pre_tokenizer.rs` from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/unicode_scripts/pre_tokenizer.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/pre_tokenizers/unicode_scripts/scripts.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace pre_tokenizers {
namespace unicode_scripts {
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

TEST(UnicodeScriptsTest, Basic) {
  const UnicodeScripts pretok;
  PreTokenizedString pretokenized = PreTokenizedString::From("どこで生れ。Yes");
  ASSERT_OK(pretok.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList(
          {{"どこで生れ", {0, 15}}, {"。", {15, 18}}, {"Yes", {18, 21}}}));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList(
          {{"どこで生れ", {0, 15}}, {"。", {15, 18}}, {"Yes", {18, 21}}}));
}

TEST(UnicodeScriptsTest, SpacesAreIncludedInEveryScript) {
  const UnicodeScripts pretok;
  PreTokenizedString pretokenized =
      PreTokenizedString::From("Apples are りんご 林檎");
  ASSERT_OK(pretok.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList({{"Apples are ", {0, 11}}, {"りんご 林檎", {11, 27}}}));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({{"Apples are ", {0, 11}}, {"りんご 林檎", {11, 27}}}));
}

TEST(UnicodeScriptsTest, TestUnicodeScript) {
  EXPECT_EQ(Script::kHan, FixedScript(U'京'));
  EXPECT_EQ(Script::kHan, FixedScript(U'太'));
  EXPECT_EQ(Script::kHan, FixedScript(U'い'));
  EXPECT_EQ(Script::kHan, FixedScript(U'グ'));
  EXPECT_EQ(Script::kHan, FixedScript(U'ー'));
  EXPECT_EQ(Script::kLatin, FixedScript(U'a'));
  EXPECT_EQ(Script::kLatin, FixedScript(U'A'));
  EXPECT_EQ(Script::kCommon, FixedScript(U'0'));
  EXPECT_EQ(Script::kCommon, FixedScript(U'$'));
  EXPECT_EQ(Script::kCommon, FixedScript(U'@'));
  EXPECT_EQ(Script::kCommon, FixedScript(U'-'));
  EXPECT_EQ(Script::kAny, FixedScript(U' '));
}

}  // namespace
}  // namespace unicode_scripts
}  // namespace pre_tokenizers
}  // namespace tokenizers
