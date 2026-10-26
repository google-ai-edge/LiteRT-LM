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
// Unit tests for `tokenizer/aho_corasick.{h,cc}` (leftmost-longest
// multi-pattern matcher replacing `daachorse::DoubleArrayAhoCorasick<u32>`).

#include "support/tokenizer/tokenizers/tokenizer/aho_corasick.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

TEST(AhoCorasickTest, LeftmostLongestMatching) {
  ASSERT_OK_AND_ASSIGN(AhoCorasick ac, AhoCorasick::BuildWithValues({
                                           {"a", 1},
                                           {"ab", 2},
                                           {"abc", 3},
                                           {"bc", 4},
                                       }));
  EXPECT_EQ(ac.NumPatterns(), 4u);

  // At offset 0, "abc" (len 3) wins over "a" and "ab".
  // At offset 3, "ab" (len 2) wins over "a", and non-overlapping resumes at 5.
  EXPECT_EQ(ac.LeftmostFindIter("abcabxbc"), (std::vector<AhoCorasickMatch>{
                                                 {0, 3, 3},
                                                 {3, 5, 2},
                                                 {6, 8, 4},
                                             }));
}

TEST(AhoCorasickTest, LeftmostWinsOverLaterLongerMatch) {
  ASSERT_OK_AND_ASSIGN(AhoCorasick ac, AhoCorasick::BuildWithValues({
                                           {"ab", 10},
                                           {"bcdef", 20},
                                       }));
  // "ab" starts at 0 and wins over "bcdef" starting at 1.
  EXPECT_EQ(ac.LeftmostFindIter("abcdef"),
            (std::vector<AhoCorasickMatch>{{0, 2, 10}}));
}

TEST(AhoCorasickTest, RejectsInvalidPatterns) {
  EXPECT_FALSE(AhoCorasick::BuildWithValues({}).ok());
  EXPECT_FALSE(AhoCorasick::BuildWithValues({{"", 0}}).ok());
  EXPECT_FALSE(AhoCorasick::BuildWithValues({{"dup", 0}, {"dup", 1}}).ok());
}

}  // namespace
}  // namespace tokenizers
