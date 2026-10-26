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
// C++ port of the `#[cfg(test)]` module of `src/utils/truncation.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/truncation.h"

#include <cstddef>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

Encoding GetEmpty() { return Encoding({}, {}, {}, {}, {}, {}, {}, {}, {}); }

Encoding GetShort() {
  return Encoding({1, 2}, {0, 0}, {"a", "b"}, {0, 1}, {{0, 1}, {1, 2}}, {0, 0},
                  {1, 1}, {}, {});
}

Encoding GetMedium() {
  return Encoding({3, 4, 5, 6}, {0, 0, 0, 0}, {"d", "e", "f", "g"},
                  {0, 1, 2, 3}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}, {0, 0, 0, 0},
                  {1, 1, 1, 1}, {}, {});
}

Encoding GetLong() {
  return Encoding(
      {7, 8, 9, 10, 11, 12, 13, 14}, {0, 0, 0, 0, 0, 0, 0, 0},
      {"h", "i", "j", "k", "l", "m", "n", "o"}, {0, 1, 2, 3, 4, 5, 6, 7},
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {6, 8}},
      {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 1}, {}, {});
}

void TruncateAndAssert(Encoding encoding1, Encoding encoding2,
                       const TruncationParams& params, size_t n1, size_t n2) {
  ASSERT_OK_AND_ASSIGN(
      auto res,
      TruncateEncodings(std::move(encoding1), std::move(encoding2), params));
  ASSERT_TRUE(res.second.has_value());
  EXPECT_EQ(res.first.GetIds().size(), n1);
  EXPECT_EQ(res.second->GetIds().size(), n2);
}

TEST(TruncationTest, TruncateEncodingsLongestFirst) {
  TruncationParams params;
  params.max_length = 7;
  params.strategy = TruncationStrategy::kLongestFirst;
  params.stride = 0;
  params.direction = TruncationDirection::kRight;

  TruncateAndAssert(GetEmpty(), GetEmpty(), params, 0, 0);
  TruncateAndAssert(GetEmpty(), GetShort(), params, 0, 2);
  TruncateAndAssert(GetEmpty(), GetMedium(), params, 0, 4);
  TruncateAndAssert(GetEmpty(), GetLong(), params, 0, 7);

  TruncateAndAssert(GetShort(), GetEmpty(), params, 2, 0);
  TruncateAndAssert(GetShort(), GetShort(), params, 2, 2);
  TruncateAndAssert(GetShort(), GetMedium(), params, 2, 4);
  TruncateAndAssert(GetShort(), GetLong(), params, 2, 5);

  TruncateAndAssert(GetMedium(), GetEmpty(), params, 4, 0);
  TruncateAndAssert(GetMedium(), GetShort(), params, 4, 2);
  TruncateAndAssert(GetMedium(), GetMedium(), params, 3, 4);
  TruncateAndAssert(GetMedium(), GetLong(), params, 3, 4);

  TruncateAndAssert(GetLong(), GetEmpty(), params, 7, 0);
  TruncateAndAssert(GetLong(), GetShort(), params, 5, 2);
  TruncateAndAssert(GetLong(), GetMedium(), params, 4, 3);
  TruncateAndAssert(GetLong(), GetLong(), params, 3, 4);
}

TEST(TruncationTest, TruncateEncodingsEmpty) {
  TruncationParams params;
  params.max_length = 0;
  params.strategy = TruncationStrategy::kLongestFirst;
  params.stride = 0;
  params.direction = TruncationDirection::kRight;

  TruncateAndAssert(GetEmpty(), GetShort(), params, 0, 0);
  TruncateAndAssert(GetMedium(), GetMedium(), params, 0, 0);
  TruncateAndAssert(GetLong(), GetLong(), params, 0, 0);
}

TEST(TruncationTest, TestDeserializeDefaults) {
  const char* old_truncation_params =
      R"({"max_length":256,"strategy":"LongestFirst","stride":0})";

  ASSERT_OK_AND_ASSIGN(Json json, ParseJson(old_truncation_params));
  ASSERT_OK_AND_ASSIGN(TruncationParams params,
                       TruncationParams::FromJson(json));

  EXPECT_EQ(params.direction, TruncationDirection::kRight);
}

}  // namespace
}  // namespace tokenizers
