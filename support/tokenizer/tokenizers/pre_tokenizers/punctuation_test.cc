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
// C++ port of the tests of `src/pre_tokenizers/punctuation.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/punctuation.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace pre_tokenizers {
namespace {

using ::testing::status::StatusIs;

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

TEST(PunctuationTest, PunctuationBasic) {
  const Punctuation pretok;
  PreTokenizedString pretokenized =
      PreTokenizedString::From("Hey friend!     How are you?!?");
  ASSERT_OK(pretok.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({
          {"Hey friend", {0, 10}},
          {"!", {10, 11}},
          {"     How are you", {11, 27}},
          {"?", {27, 28}},
          {"!", {28, 29}},
          {"?", {29, 30}},
      }));
}

TEST(PunctuationTest, Deserialization) {
  ASSERT_OK_AND_ASSIGN(const Json json,
                       ParseJson(R"({"type": "Punctuation"})"));
  ASSERT_OK_AND_ASSIGN(const Punctuation punctuation,
                       Punctuation::FromJson(json));
  EXPECT_EQ(punctuation, Punctuation());
  EXPECT_EQ(punctuation, Punctuation(SplitDelimiterBehavior::kIsolated));
}

// `#[should_panic]`: the `unwrap()` of the failed deserialization.
TEST(PunctuationTest, DeserializationErroneous) {
  ASSERT_OK_AND_ASSIGN(const Json json,
                       ParseJson(R"({"type": "WhitespaceSplit"})"));
  EXPECT_THAT(Punctuation::FromJson(json),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace pre_tokenizers
}  // namespace tokenizers
