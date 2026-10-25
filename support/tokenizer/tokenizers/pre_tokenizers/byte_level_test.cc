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
// C++ port of the tests of `src/pre_tokenizers/byte_level.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace pre_tokenizers {
namespace {

using ::testing::ElementsAre;

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

// `Encoding::new(ids, type_ids, tokens, words, offsets, special_tokens_mask,
// attention_mask, overflowing, sequence_ranges)`
Encoding MakeEncoding(
    std::vector<uint32_t> ids, std::vector<uint32_t> type_ids,
    std::vector<std::string> tokens, std::vector<std::optional<uint32_t>> words,
    std::vector<Offsets> offsets, std::vector<uint32_t> special_tokens_mask,
    std::vector<uint32_t> attention_mask, std::vector<Encoding> overflowing,
    absl::flat_hash_map<size_t, std::pair<size_t, size_t>> sequence_ranges) {
  return Encoding(std::move(ids), std::move(type_ids), std::move(tokens),
                  std::move(words), std::move(offsets),
                  std::move(special_tokens_mask), std::move(attention_mask),
                  std::move(overflowing), std::move(sequence_ranges));
}

TEST(ByteLevelTest, PreTokenization) {
  const ByteLevel bytelevel = ByteLevel().AddPrefixSpace(false);
  PreTokenizedString pretokenized =
      PreTokenizedString::From("Hello my friend, how is your day going?");
  ASSERT_OK(bytelevel.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({{"Hello", {0, 5}},
                 {"Ġmy", {5, 8}},
                 {"Ġfriend", {8, 15}},
                 {",", {15, 16}},
                 {"Ġhow", {16, 20}},
                 {"Ġis", {20, 23}},
                 {"Ġyour", {23, 28}},
                 {"Ġday", {28, 32}},
                 {"Ġgoing", {32, 38}},
                 {"?", {38, 39}}}));
}

TEST(ByteLevelTest, PreTokenizationNoRegex) {
  const ByteLevel bytelevel = ByteLevel().UseRegex(false);
  PreTokenizedString pretokenized =
      PreTokenizedString::From("Hello my friend, how is your day going?");
  ASSERT_OK(bytelevel.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({{"ĠHelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?", {0, 39}}}));
}

TEST(ByteLevelTest, Decoding) {
  const ByteLevel bytelevel = ByteLevel().AddPrefixSpace(false);
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::string> decoded,
      bytelevel.DecodeChain({"Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis",
                             "Ġyour", "Ġday", "Ġgoing", "?"}));
  EXPECT_THAT(decoded, ElementsAre("Hello my friend, how is your day going?"));
}

TEST(ByteLevelTest, AddPrefixSpace) {
  const ByteLevel bytelevel = ByteLevel().AddPrefixSpace(true);
  for (const absl::string_view s :
       {" Hello my friend, how is your day going?",
        "Hello my friend, how is your day going?"}) {
    PreTokenizedString pretokenized = PreTokenizedString::From(s);
    ASSERT_OK(bytelevel.PreTokenize(pretokenized));
    EXPECT_EQ(
        Splits(pretokenized, OffsetReferential::kNormalized, OffsetType::kByte),
        SplitList({{"ĠHello", {0, 7}},
                   {"Ġmy", {7, 11}},
                   {"Ġfriend", {11, 19}},
                   {",", {19, 20}},
                   {"Ġhow", {20, 25}},
                   {"Ġis", {25, 29}},
                   {"Ġyour", {29, 35}},
                   {"Ġday", {35, 40}},
                   {"Ġgoing", {40, 47}},
                   {"?", {47, 48}}}));
  }
}

TEST(ByteLevelTest, DecodeWorksOnSeparatedTokens) {
  const std::vector<absl::string_view> samples = {
      "A Nuskhuri abbreviation of იესუ ქრისტე ( iesu kriste ) \" Jesus Christ "
      "\"",
      "An equal number have descenders , like p or q in English : გ , დ , ე , "
      "ვ , კ , ლ , ჟ , ტ , უ , ფ , ღ , ყ , ც",
  };

  const ByteLevel bytelevel = ByteLevel().AddPrefixSpace(false);
  for (const absl::string_view sample : samples) {
    PreTokenizedString pretokenized = PreTokenizedString::From(sample);
    ASSERT_OK(bytelevel.PreTokenize(pretokenized));
    // `.flat_map(|(s, _, _)| s.split("").map(|t| t.into()))`: splitting a Rust
    // string on "" yields "", every char, and "" again.
    std::vector<std::string> separated_tokens;
    for (const auto& split : pretokenized.GetSplits(
             OffsetReferential::kOriginal, OffsetType::kByte)) {
      const absl::string_view s = std::get<0>(split);
      separated_tokens.push_back("");
      for (const auto [offset, c] : CharIndices(s)) {
        separated_tokens.push_back(ToUtf8(c));
      }
      separated_tokens.push_back("");
    }
    ASSERT_OK_AND_ASSIGN(const std::vector<std::string> decoded,
                         bytelevel.DecodeChain(separated_tokens));
    EXPECT_EQ(sample, absl::StrJoin(decoded, ""));
  }
}

TEST(ByteLevelTest, HandlingOfNewlines) {
  PreTokenizedString pretokenized =
      PreTokenizedString::From("Hello there\nHello there");
  const ByteLevel bytelevel = ByteLevel().AddPrefixSpace(false);
  ASSERT_OK(bytelevel.PreTokenize(pretokenized));

  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({{"Hello", {0, 5}},
                 {"Ġthere", {5, 11}},
                 {"Ċ", {11, 12}},
                 {"Hello", {12, 17}},
                 {"Ġthere", {17, 23}}}));
}

TEST(ByteLevelTest, HandlingOfMultipleWhitespaces) {
  PreTokenizedString pretokenized =
      PreTokenizedString::From("Hello there       dear");
  const ByteLevel bytelevel = ByteLevel().AddPrefixSpace(false);
  ASSERT_OK(bytelevel.PreTokenize(pretokenized));

  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({{"Hello", {0, 5}},
                 {"Ġthere", {5, 11}},
                 {"ĠĠĠĠĠĠ", {11, 17}},
                 {"Ġdear", {17, 22}}}));
}

TEST(ByteLevelTest, OffsetsWhenCharSplitUp) {
  const absl::string_view input = "i⭢j";
  PreTokenizedString pretokenized = PreTokenizedString::From(input);
  const ByteLevel bytelevel = ByteLevel().AddPrefixSpace(false);
  ASSERT_OK(bytelevel.PreTokenize(pretokenized));

  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({{"i", {0, 1}}, {"âŃ¢", {1, 4}}, {"j", {4, 5}}}));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList({{"i", {0, 1}}, {"âŃ¢", {1, 7}}, {"j", {7, 8}}}));
  std::vector<std::string> original_slices;
  for (const auto& split : pretokenized.GetSplits(OffsetReferential::kOriginal,
                                                  OffsetType::kByte)) {
    const Offsets o = std::get<1>(split);
    original_slices.emplace_back(input.substr(o.first, o.second - o.first));
  }
  EXPECT_THAT(original_slices, ElementsAre("i", "⭢", "j"));
}

TEST(ByteLevelTest, ProcessorTrimsOffsetsPreTokenized) {
  // If user uses `is_pretokenized=True` we might have
  // offsets that might begin at the start of the string but are
  // NOT the first token.
  Encoding encoding =
      MakeEncoding(std::vector<uint32_t>(5, 0), {}, {"Ġl", "ove", "Ġl", "ove"},
                   {}, {{0, 1}, {1, 4}, {0, 1}, {1, 4}}, {}, {}, {}, {});
  ProcessOffsets(encoding, true);
  EXPECT_EQ(
      encoding,
      MakeEncoding(std::vector<uint32_t>(5, 0), {}, {"Ġl", "ove", "Ġl", "ove"},
                   {}, {{0, 1}, {1, 4}, {0, 1}, {1, 4}}, {}, {}, {}, {}));
}

TEST(ByteLevelTest, ProcessorTrimsOffsets) {
  const Encoding start = MakeEncoding(
      std::vector<uint32_t>(5, 0), {},
      {"Ġ", "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ"}, {},
      {{0, 1}, {0, 11}, {11, 18}, {18, 25}, {25, 29}}, {}, {}, {}, {});
  const Encoding expected =
      MakeEncoding(std::vector<uint32_t>(5, 0), std::vector<uint32_t>(5, 0),
                   {"Ġ", "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ"}, {},
                   {{0, 0}, {4, 9}, {13, 18}, {18, 23}, {29, 29}}, {}, {}, {},
                   {{0, {0, 5}}});

  const ByteLevel bytelevel = ByteLevel().TrimOffsets(true);
  ASSERT_OK_AND_ASSIGN(const Encoding processed,
                       bytelevel.Process(start, std::nullopt, false));
  EXPECT_EQ(expected, processed);

  const Encoding pair_expected =
      MakeEncoding(std::vector<uint32_t>(10, 0), {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                   {"Ġ", "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ", "Ġ",
                    "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ"},
                   {},
                   {{0, 0},
                    {4, 9},
                    {13, 18},
                    {18, 23},
                    {29, 29},
                    {0, 0},
                    {4, 9},
                    {13, 18},
                    {18, 23},
                    {29, 29}},
                   {}, {}, {}, {{0, {0, 5}}, {1, {5, 10}}});
  ASSERT_OK_AND_ASSIGN(const Encoding pair_processed,
                       bytelevel.Process(start, start, false));
  EXPECT_EQ(pair_expected, pair_processed);
}

TEST(ByteLevelTest, DecodeUnknownCharacters) {
  const ByteLevel byte_level;
  ASSERT_OK_AND_ASSIGN(const std::vector<std::string> decoded,
                       byte_level.DecodeChain({"Hello", "Ġthere", "Ġdear",
                                               "Ġfriend!", "Ġ", "[PA D]"}));
  EXPECT_THAT(decoded, ElementsAre("Hello there dear friend! [PA D]"));
}

TEST(ByteLevelTest, Deserialization) {
  // Before use_regex
  ASSERT_OK_AND_ASSIGN(
      const Json json1,
      ParseJson(
          R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false})"));
  ASSERT_OK_AND_ASSIGN(ByteLevel byte_level, ByteLevel::FromJson(json1));
  EXPECT_TRUE(byte_level.use_regex());

  // Loading works, new future BC test.
  ASSERT_OK_AND_ASSIGN(
      const Json json2,
      ParseJson(
          R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true})"));
  ASSERT_OK_AND_ASSIGN(byte_level, ByteLevel::FromJson(json2));
  EXPECT_TRUE(byte_level.use_regex());

  ASSERT_OK_AND_ASSIGN(
      const Json json3,
      ParseJson(
          R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false})"));
  ASSERT_OK_AND_ASSIGN(byte_level, ByteLevel::FromJson(json3));
  EXPECT_FALSE(byte_level.use_regex());
}

}  // namespace
}  // namespace pre_tokenizers
}  // namespace tokenizers
