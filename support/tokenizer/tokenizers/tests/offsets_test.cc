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
// C++ port of `tests/offsets.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tests/common.h"
#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::tokenizers::test_common::GetBert;
using ::tokenizers::test_common::GetByteLevel;

void CheckOffsets(absl::string_view input, const Encoding& output,
                  size_t offset, absl::string_view result) {
  Offsets offsets = output.GetOffsets()[offset];
  EXPECT_EQ(input.substr(offsets.first, offsets.second - offsets.first),
            result);
}

TEST(OffsetsTest, ByteLevelBasic) {
  // Without trimming offsets
  Tokenizer tokenizer = GetByteLevel(true, false);

  absl::string_view input = "Hello there, how are you?";
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  CheckOffsets(input, output, 0, "Hello");
  CheckOffsets(input, output, 1, " there");
  CheckOffsets(input, output, 2, ",");
  CheckOffsets(input, output, 3, " how");
  CheckOffsets(input, output, 4, " are");
  CheckOffsets(input, output, 5, " you");
  CheckOffsets(input, output, 6, "?");

  // And when trimming offsets:
  Tokenizer tokenizer2 = GetByteLevel(true, true);

  ASSERT_OK_AND_ASSIGN(Encoding output2, tokenizer2.Encode(input, false));

  CheckOffsets(input, output2, 0, "Hello");
  CheckOffsets(input, output2, 1, "there");
  CheckOffsets(input, output2, 2, ",");
  CheckOffsets(input, output2, 3, "how");
  CheckOffsets(input, output2, 4, "are");
  CheckOffsets(input, output2, 5, "you");
  CheckOffsets(input, output2, 6, "?");
}

TEST(OffsetsTest, ByteLevelUnicode) {
  Tokenizer tokenizer = GetByteLevel(true, false);

  absl::string_view input = "i⭢j";
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  CheckOffsets(input, output, 1, "⭢");
  CheckOffsets(input, output, 2, "⭢");
  CheckOffsets(input, output, 3, "⭢");
}

TEST(OffsetsTest, ByteLevelDoubleSequence) {
  const char* input_a = "My name is Anthony";
  const char* input_b = "What is my name?";

  // Without trimming offsets
  Tokenizer tokenizer = GetByteLevel(true, false);
  ASSERT_OK_AND_ASSIGN(Encoding output,
                       tokenizer.Encode(EncodeInput(input_a, input_b), false));

  EXPECT_EQ(output.GetOffsets(), (std::vector<Offsets>{
                                     {0, 2},
                                     {2, 7},
                                     {7, 10},
                                     {10, 18},
                                     {0, 4},
                                     {4, 7},
                                     {7, 10},
                                     {10, 15},
                                     {15, 16},
                                 }));
  EXPECT_EQ(output.GetWordIds(),
            (std::vector<std::optional<uint32_t>>{0, 1, 2, 3, 0, 1, 2, 3, 4}));
  EXPECT_EQ(output.GetTypeIds(),
            (std::vector<uint32_t>{0, 0, 0, 0, 1, 1, 1, 1, 1}));

  // When trimming offsets
  Tokenizer tokenizer2 = GetByteLevel(true, true);
  ASSERT_OK_AND_ASSIGN(Encoding output2,
                       tokenizer2.Encode(EncodeInput(input_a, input_b), false));
  EXPECT_EQ(output2.GetOffsets(), (std::vector<Offsets>{
                                      {0, 2},
                                      {3, 7},
                                      {8, 10},
                                      {11, 18},
                                      {0, 4},
                                      {5, 7},
                                      {8, 10},
                                      {11, 15},
                                      {15, 16},
                                  }));
}

TEST(OffsetsTest, ByteLevelPreTokenizedSequence) {
  std::vector<absl::string_view> input = {"My", "name", "is", "Anthonino"};

  // Without trimming offsets
  Tokenizer tokenizer = GetByteLevel(true, false);
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  EXPECT_EQ(
      output.GetTokens(),
      (std::vector<std::string>{"ĠMy", "Ġname", "Ġis", "ĠAnth", "on", "ino"}));
  EXPECT_EQ(output.GetWordIds(),
            (std::vector<std::optional<uint32_t>>{0, 1, 2, 3, 3, 3}));
  EXPECT_EQ(
      output.GetOffsets(),
      (std::vector<Offsets>{{0, 2}, {0, 4}, {0, 2}, {0, 4}, {4, 6}, {6, 9}}));
}

// `#[ignore]` in the Rust original (`tests/offsets.rs:139`).
TEST(OffsetsTest, DISABLED_ByteLevelPreTokenizedSequenceWithTrimming) {
  std::vector<absl::string_view> input = {"My", "name", "is", "Anthonino"};

  // When trimming offsets (expect same result)
  Tokenizer tokenizer = GetByteLevel(true, true);
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  EXPECT_EQ(output.GetWordIds(),
            (std::vector<std::optional<uint32_t>>{0, 1, 2, 3, 3, 3}));
  EXPECT_EQ(
      output.GetOffsets(),
      (std::vector<Offsets>{{0, 2}, {0, 4}, {0, 2}, {0, 4}, {4, 6}, {6, 9}}));
}

TEST(OffsetsTest, SplitOnAddedTokensBert) {
  const char* input = "Yesterday I saw a [MASK] far away";

  Tokenizer tokenizer = GetBert();
  ASSERT_OK(
      tokenizer.AddSpecialTokens({AddedToken::From("[MASK]", true)}).status());
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  EXPECT_EQ(output.GetOffsets(), (std::vector<Offsets>{
                                     {0, 9},
                                     {10, 11},
                                     {12, 15},
                                     {16, 17},
                                     {18, 24},
                                     {25, 28},
                                     {29, 33},
                                 }));
  EXPECT_EQ(output.GetTokens(),
            (std::vector<std::string>{"yesterday", "i", "saw", "a", "[MASK]",
                                      "far", "away"}));
  EXPECT_EQ(output.GetWordIds(),
            (std::vector<std::optional<uint32_t>>{0, 1, 2, 3, 4, 5, 6}));
}

}  // namespace
}  // namespace tokenizers
