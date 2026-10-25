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
// C++ port of the tests of `src/normalizers/prepend.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/normalizers/prepend.h"

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

TEST(PrependTest, TestPrepend) {
  const std::string original = "Hello";
  const std::string normalized = "▁Hello";
  ASSERT_NE(original, normalized);
  NormalizedString n = NormalizedString::From(original);
  Prepend prepend("▁");
  ASSERT_OK(prepend.Normalize(n));
  EXPECT_EQ(n.Get(), normalized);
  EXPECT_EQ(
      n,
      NormalizedString::New(
          original, normalized,
          {{0, 1}, {0, 1}, {0, 1}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}, 0));
  EXPECT_THAT(n.AlignmentsOriginal(),
              ElementsAreArray(std::vector<std::pair<size_t, size_t>>{
                  {0, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}}));
}

}  // namespace
}  // namespace tokenizers::normalizers
