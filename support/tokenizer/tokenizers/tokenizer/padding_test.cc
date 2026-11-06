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
// C++ port of the `#[cfg(test)]` module of `src/utils/padding.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/padding.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

TEST(PaddingTest, PadToMultiple) {
  auto get_encodings = []() -> std::vector<Encoding> {
    return {
        Encoding({0, 1, 2, 3, 4}, {}, {}, {}, {}, {}, {}, {}, {}),
        Encoding({0, 1, 2}, {}, {}, {}, {}, {}, {}, {}, {}),
    };
  };

  // Test fixed
  std::vector<Encoding> encodings = get_encodings();
  PaddingParams params;
  params.strategy = PaddingStrategy::Fixed(7);
  params.direction = PaddingDirection::kRight;
  params.pad_to_multiple_of = 8;
  params.pad_id = 0;
  params.pad_type_id = 0;
  params.pad_token = "[PAD]";
  ASSERT_OK(PadEncodings(absl::MakeSpan(encodings), params));
  for (const Encoding& e : encodings) {
    EXPECT_EQ(e.GetIds().size(), 8u);
  }

  // Test batch
  encodings = get_encodings();
  params.strategy = PaddingStrategy::BatchLongest();
  params.pad_to_multiple_of = 6;
  ASSERT_OK(PadEncodings(absl::MakeSpan(encodings), params));
  for (const Encoding& e : encodings) {
    EXPECT_EQ(e.GetIds().size(), 6u);
  }

  // Do not crash with 0
  params.pad_to_multiple_of = 0;
  ASSERT_OK(PadEncodings(absl::MakeSpan(encodings), params));
}

}  // namespace
}  // namespace tokenizers
