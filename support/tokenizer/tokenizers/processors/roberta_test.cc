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
// C++ port of the `#[cfg(test)]` module of `src/processors/roberta.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0, Copyright
// 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/processors/roberta.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace processors {
namespace {

TEST(RobertaTest, Serde) {
  RobertaProcessing roberta;
  const std::string roberta_r =
      R"({"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true})";
  EXPECT_EQ(DumpJson(roberta.ToJson()), roberta_r);
  ASSERT_OK_AND_ASSIGN(Json parsed, ParseJson(roberta_r));
  ASSERT_OK_AND_ASSIGN(RobertaProcessing reconstructed,
                       RobertaProcessing::FromJson(parsed));
  EXPECT_EQ(reconstructed, roberta);
}

TEST(RobertaTest, RobertaProcessing) {
  RobertaProcessing processor;
  EXPECT_EQ(processor.AddedTokens(false), 2u);
  EXPECT_EQ(processor.AddedTokens(true), 4u);

  Encoding encoding = Encoding::FromTokens(
      {Token(12, "Hello", {0, 5}), Token(14, "there", {6, 11})}, 0);
  Encoding pair = Encoding::FromTokens({Token(15, "pair", {0, 4})}, 0);
  ASSERT_OK_AND_ASSIGN(Encoding single_encoding,
                       processor.Process(encoding, std::nullopt, true));
  EXPECT_EQ(
      single_encoding,
      Encoding({0, 12, 14, 2}, {0, 0, 0, 0}, {"<s>", "Hello", "there", "</s>"},
               {std::nullopt, std::nullopt, std::nullopt, std::nullopt},
               {{0, 0}, {0, 5}, {6, 11}, {0, 0}}, {1, 0, 0, 1}, {1, 1, 1, 1},
               {}, {{0, {1, 3}}}));
  EXPECT_EQ(single_encoding.TokenToSequence(2), std::optional<size_t>(0));
  EXPECT_EQ(single_encoding.TokenToSequence(3), std::nullopt);
  ASSERT_OK_AND_ASSIGN(Encoding pair_encoding,
                       processor.Process(encoding, pair, true));
  EXPECT_EQ(pair_encoding,
            Encoding({0, 12, 14, 2, 2, 15, 2}, {0, 0, 0, 0, 0, 0, 0},
                     {"<s>", "Hello", "there", "</s>", "</s>", "pair", "</s>"},
                     {std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                      std::nullopt, std::nullopt, std::nullopt},
                     {{0, 0}, {0, 5}, {6, 11}, {0, 0}, {0, 0}, {0, 4}, {0, 0}},
                     {1, 0, 0, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 1, 1}, {},
                     {{0, {1, 3}}, {1, {5, 6}}}));
  EXPECT_EQ(pair_encoding.TokenToSequence(2), std::optional<size_t>(0));
  EXPECT_EQ(pair_encoding.TokenToSequence(3), std::nullopt);
  EXPECT_EQ(pair_encoding.TokenToSequence(4), std::nullopt);
  EXPECT_EQ(pair_encoding.TokenToSequence(5), std::optional<size_t>(1));
  EXPECT_EQ(pair_encoding.TokenToSequence(6), std::nullopt);

  // No special tokens
  ASSERT_OK_AND_ASSIGN(Encoding no_special_encoding,
                       processor.Process(encoding, pair, false));
  EXPECT_EQ(no_special_encoding,
            Encoding({12, 14, 15}, {0, 0, 0}, {"Hello", "there", "pair"},
                     {std::nullopt, std::nullopt, std::nullopt},
                     {{0, 5}, {6, 11}, {0, 4}}, {0, 0, 0}, {1, 1, 1}, {},
                     {{0, {0, 2}}, {1, {2, 3}}}));
  EXPECT_EQ(no_special_encoding.TokenToSequence(0), std::optional<size_t>(0));
  EXPECT_EQ(no_special_encoding.TokenToSequence(1), std::optional<size_t>(0));
  EXPECT_EQ(no_special_encoding.TokenToSequence(2), std::optional<size_t>(1));
}

}  // namespace
}  // namespace processors
}  // namespace tokenizers
