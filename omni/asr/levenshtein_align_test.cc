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

}  // namespace
}  // namespace litert::omni::asr
