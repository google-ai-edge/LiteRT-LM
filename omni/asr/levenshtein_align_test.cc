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

#include "omni/asr/levenshtein_align.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::omni::asr {
namespace {

using ::testing::ElementsAre;

TEST(LevenshteinAlignTest, EmptyInputs) {
  std::vector<std::string> ref = {};
  std::vector<std::string> hyp = {};
  EXPECT_TRUE(AlignTokens(ref, hyp).empty());
}

TEST(LevenshteinAlignTest, ExactMatch) {
  std::vector<std::string> ref = {"hello", "world"};
  std::vector<std::string> hyp = {"hello", "world"};
  EXPECT_THAT(AlignTokens(ref, hyp),
              ElementsAre(AlignCode::kCorrect, AlignCode::kCorrect));
}

TEST(LevenshteinAlignTest, OverlapWithPrefixAndSuffix) {
  std::vector<std::string> ref = {"hello", "world", "this", "is"};
  std::vector<std::string> hyp = {"this", "is", "a", "test"};
  EXPECT_THAT(AlignTokens(ref, hyp),
              ElementsAre(AlignCode::kDeletion, AlignCode::kDeletion,
                          AlignCode::kCorrect, AlignCode::kCorrect,
                          AlignCode::kInsertion, AlignCode::kInsertion));
}

TEST(LevenshteinAlignTest, ComputeLevenshteinDistance) {
  EXPECT_EQ(ComputeLevenshteinDistanceStr("kitten", "sitting"), 3);
  EXPECT_EQ(ComputeLevenshteinDistanceStr("hello", "hello"), 0);
  EXPECT_EQ(ComputeLevenshteinDistanceStr("", "abc"), 3);
  EXPECT_EQ(ComputeLevenshteinDistanceStr("abc", ""), 3);

  std::vector<std::string> v1 = {"hello", "world"};
  std::vector<std::string> v2 = {"hella", "world"};
  EXPECT_EQ(ComputeLevenshteinDistanceWords(v1, v2), 0);
}

TEST(LevenshteinAlignTest, Canonicalize) {
  EXPECT_EQ(CanonicalizeStr("Hello,"), "hello");
  std::vector<std::string> words = {"Hello,", "WORLD!"};
  EXPECT_THAT(CanonicalizeWords(words), ElementsAre("hello", "world"));
}

TEST(LevenshteinAlignTest, CloseEnoughTest) {
  EXPECT_TRUE(CloseEnoughStr("cat", "cat"));
  EXPECT_FALSE(CloseEnoughStr(
      "cat", "car"));  // min_len <= 3 requiring exact prefix match
  EXPECT_TRUE(CloseEnoughStr("hello", "hella"));  // min_len > 3 with dist <= 1
  EXPECT_FALSE(CloseEnoughStr("hello", "heyyy"));
}

TEST(LevenshteinAlignTest, CloseEnoughSpansTest) {
  std::vector<std::string> l1 = {"hello", "world"};
  std::vector<std::string> l2 = {"hella", "world"};
  EXPECT_TRUE(CloseEnoughWords(l1, l2));

  std::vector<std::string> l3 = {"hello"};
  EXPECT_FALSE(CloseEnoughWords(l1, l3));
}

TEST(LevenshteinAlignTest, DedupWordsTest) {
  std::vector<std::string> prev = {"the", "stale", "smell", "of"};
  std::vector<std::string> curr = {"smell", "of", "old", "beer"};
  EXPECT_THAT(DedupWords(prev, curr), ElementsAre("old", "beer"));
}

}  // namespace
}  // namespace litert::omni::asr
