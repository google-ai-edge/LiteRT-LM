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
// C++ port of the tests of `src/pre_tokenizers/split.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/split.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace pre_tokenizers {
namespace {

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

// `serde_json::from_str::<Split>(s)`
absl::StatusOr<Split> SplitFromStr(absl::string_view s) {
  absl::StatusOr<Json> json = ParseJson(s);
  if (!json.ok()) return json.status();
  return Split::FromJson(*json);
}

TEST(SplitTest, Basic) {
  const std::vector<
      std::tuple<SplitDelimiterBehavior, absl::string_view, SplitList>>
      tests = {
          {
              SplitDelimiterBehavior::kRemoved,
              "How are you doing?",
              {
                  {"How", {0, 3}},
                  {"are", {4, 7}},
                  {"you", {8, 11}},
                  {"doing", {12, 17}},
                  {"?", {17, 18}},
              },
          },
          {
              SplitDelimiterBehavior::kIsolated,
              "How are you doing?",
              {
                  {"How", {0, 3}},
                  {" ", {3, 4}},
                  {"are", {4, 7}},
                  {" ", {7, 8}},
                  {"you", {8, 11}},
                  {" ", {11, 12}},
                  {"doing", {12, 17}},
                  {"?", {17, 18}},
              },
          },
          {
              SplitDelimiterBehavior::kMergedWithPrevious,
              "How are you doing?",
              {
                  {"How ", {0, 4}},
                  {"are ", {4, 8}},
                  {"you ", {8, 12}},
                  {"doing", {12, 17}},
                  {"?", {17, 18}},
              },
          },
          {
              SplitDelimiterBehavior::kMergedWithNext,
              "How are you doing?",
              {
                  {"How", {0, 3}},
                  {" are", {3, 7}},
                  {" you", {7, 11}},
                  {" doing", {11, 17}},
                  {"?", {17, 18}},
              },
          },
          {
              SplitDelimiterBehavior::kContiguous,
              "How are you doing?",
              {
                  {"How", {0, 3}},
                  {" ", {3, 4}},
                  {"are", {4, 7}},
                  {" ", {7, 8}},
                  {"you", {8, 11}},
                  {" ", {11, 12}},
                  {"doing?", {12, 18}},
              },
          },
      };

  // use whitespace regex
  const SplitPattern regex = SplitPattern::Regex(R"(\w+|[^\w\s]+)");

  for (const auto& [behavior, s, res] : tests) {
    PreTokenizedString pretokenized = PreTokenizedString::From(s);
    ASSERT_OK_AND_ASSIGN(const Split pretok, Split::New(regex, behavior, true));
    ASSERT_OK(pretok.PreTokenize(pretokenized));
    EXPECT_EQ(
        Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
        res);
  }
}

TEST(SplitTest, RegexString) {
  PreTokenizedString pretok_str_for_regex =
      PreTokenizedString::From("Hey, man!");
  PreTokenizedString pretok_str_for_string = pretok_str_for_regex;

  // pre-tokenizer splits on " " - one from Regex, one from string
  ASSERT_OK_AND_ASSIGN(const Split pretokenizer_regex,
                       Split::New(SplitPattern::Regex(R"(\s+)"),
                                  SplitDelimiterBehavior::kRemoved, false));
  ASSERT_OK_AND_ASSIGN(
      const Split pretokenizer_string,
      Split::New(" ", SplitDelimiterBehavior::kRemoved, false));

  ASSERT_OK(pretokenizer_regex.PreTokenize(pretok_str_for_regex));
  ASSERT_OK(pretokenizer_string.PreTokenize(pretok_str_for_string));

  EXPECT_EQ(pretok_str_for_regex, pretok_str_for_string);
}

TEST(SplitTest, Invert) {
  PreTokenizedString pretok_str = PreTokenizedString::From("Hello Hello Hello");
  PreTokenizedString pretok_str_for_invert = pretok_str;

  // one pre-tokenizer splits on " " - one splits inverted on "Hello"
  ASSERT_OK_AND_ASSIGN(
      const Split pretokenizer,
      Split::New(" ", SplitDelimiterBehavior::kRemoved, false));
  ASSERT_OK_AND_ASSIGN(
      const Split pretokenizer_invert,
      Split::New("Hello", SplitDelimiterBehavior::kRemoved, true));

  ASSERT_OK(pretokenizer.PreTokenize(pretok_str));
  ASSERT_OK(pretokenizer_invert.PreTokenize(pretok_str_for_invert));

  EXPECT_EQ(pretok_str, pretok_str_for_invert);
}

TEST(SplitTest, Serialization) {
  ASSERT_OK_AND_ASSIGN(
      const Split split,
      Split::New("Hello", SplitDelimiterBehavior::kRemoved, true));
  const std::string split_s =
      R"({"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true})";
  EXPECT_EQ(DumpJson(split.ToJson()), split_s);
  ASSERT_OK_AND_ASSIGN(Split parsed, SplitFromStr(split_s));
  EXPECT_EQ(parsed, split);

  ASSERT_OK_AND_ASSIGN(const Split split2,
                       Split::New(SplitPattern::Regex(R"(\s+)"),
                                  SplitDelimiterBehavior::kIsolated, false));
  const std::string split2_s =
      R"({"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated","invert":false})";
  EXPECT_EQ(DumpJson(split2.ToJson()), split2_s);
  ASSERT_OK_AND_ASSIGN(parsed, SplitFromStr(split2_s));
  EXPECT_EQ(parsed, split2);
}

}  // namespace
}  // namespace pre_tokenizers
}  // namespace tokenizers
