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
// C++ port of the tests of `src/normalizers/byte_level.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/normalizers/byte_level.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::normalizers {
namespace {

using ::testing::ElementsAreArray;

TEST(ByteLevelTest, TestByteLevelNormalize) {
  const std::string original = "Hello 我今天能为你做什么";
  const std::string normalized = "HelloĠæĪĳä»Ĭå¤©èĥ½ä¸ºä½łåģļä»Ģä¹Ī";
  ASSERT_NE(original, normalized);
  NormalizedString n = NormalizedString::From(original);
  ByteLevel byte_level;
  ASSERT_OK(byte_level.Normalize(n));
  EXPECT_EQ(n.Get(), normalized);
  EXPECT_EQ(
      n,
      NormalizedString::New(
          original, normalized,
          {{0, 1},   {1, 2},   {2, 3},   {3, 4},   {4, 5},   {5, 6},   {5, 6},
           {6, 9},   {6, 9},   {6, 9},   {6, 9},   {6, 9},   {6, 9},   {9, 12},
           {9, 12},  {9, 12},  {9, 12},  {9, 12},  {9, 12},  {12, 15}, {12, 15},
           {12, 15}, {12, 15}, {12, 15}, {12, 15}, {15, 18}, {15, 18}, {15, 18},
           {15, 18}, {15, 18}, {15, 18}, {18, 21}, {18, 21}, {18, 21}, {18, 21},
           {18, 21}, {18, 21}, {21, 24}, {21, 24}, {21, 24}, {21, 24}, {21, 24},
           {21, 24}, {24, 27}, {24, 27}, {24, 27}, {24, 27}, {24, 27}, {24, 27},
           {27, 30}, {27, 30}, {27, 30}, {27, 30}, {27, 30}, {27, 30}, {30, 33},
           {30, 33}, {30, 33}, {30, 33}, {30, 33}, {30, 33}},
          0));
  EXPECT_THAT(
      n.AlignmentsOriginal(),
      ElementsAreArray(std::vector<std::pair<size_t, size_t>>{
          {0, 1},   {1, 2},   {2, 3},   {3, 4},   {4, 5},   {5, 7},   {7, 13},
          {7, 13},  {7, 13},  {13, 19}, {13, 19}, {13, 19}, {19, 25}, {19, 25},
          {19, 25}, {25, 31}, {25, 31}, {25, 31}, {31, 37}, {31, 37}, {31, 37},
          {37, 43}, {37, 43}, {37, 43}, {43, 49}, {43, 49}, {43, 49}, {49, 55},
          {49, 55}, {49, 55}, {55, 61}, {55, 61}, {55, 61}}));
}

}  // namespace
}  // namespace tokenizers::normalizers
