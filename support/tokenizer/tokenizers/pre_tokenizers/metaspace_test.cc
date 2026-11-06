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
// C++ port of the tests of `src/pre_tokenizers/metaspace.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/metaspace.h"

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
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

// `serde_json::from_str::<Metaspace>(s)`
absl::StatusOr<Metaspace> MetaspaceFromStr(absl::string_view s) {
  absl::StatusOr<Json> json = ParseJson(s);
  if (!json.ok()) return json.status();
  return Metaspace::FromJson(*json);
}

TEST(MetaspaceTest, Serialization) {
  const Metaspace metaspace(U'_', PrependScheme::kAlways, true);
  const std::string metaspace_s =
      R"({"type":"Metaspace","replacement":"_","prepend_scheme":"always","split":true})";
  EXPECT_EQ(DumpJson(metaspace.ToJson()), metaspace_s);
  ASSERT_OK_AND_ASSIGN(Metaspace parsed, MetaspaceFromStr(metaspace_s));
  EXPECT_EQ(parsed, metaspace);

  // Also check it can deserialize previous versions
  EXPECT_FALSE(
      MetaspaceFromStr(
          R"({"type":"Metaspace","replacement":"_","add_prefix_space":false,"prepend_scheme":"always"})")
          .ok());

  const Metaspace metaspace2(U'_', PrependScheme::kAlways, true);
  ASSERT_OK_AND_ASSIGN(
      parsed,
      MetaspaceFromStr(
          R"({"type":"Metaspace","str_rep":"_","replacement":"_","add_prefix_space":true,"prepend_scheme":"always"})"));
  EXPECT_EQ(parsed, metaspace2);

  ASSERT_OK_AND_ASSIGN(
      const Metaspace metaspace_parsed,
      MetaspaceFromStr(
          R"({"type":"Metaspace","replacement":"_","add_prefix_space":true})"));
  EXPECT_EQ(metaspace_parsed, metaspace2);
}

TEST(MetaspaceTest, Basic) {
  const Metaspace pretok(U'▁', PrependScheme::kAlways, true);
  PreTokenizedString pretokenized = PreTokenizedString::From("Hey friend!");
  ASSERT_OK(pretok.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList({{"▁Hey", {0, 6}}, {"▁friend!", {6, 16}}}));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({{"▁Hey", {0, 3}}, {"▁friend!", {3, 11}}}));
}

TEST(MetaspaceTest, MultipleSpaces) {
  const Metaspace pretok(U'▁', PrependScheme::kAlways, true);
  PreTokenizedString pretokenized = PreTokenizedString::From("Hey   friend!");
  ASSERT_OK(pretok.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList({
          {"▁Hey", {0, 6}},
          {"▁", {6, 9}},
          {"▁", {9, 12}},
          {"▁friend!", {12, 22}},
      }));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
      SplitList({
          {"▁Hey", {0, 3}},
          {"▁", {3, 4}},
          {"▁", {4, 5}},
          {"▁friend!", {5, 13}},
      }));
}

TEST(MetaspaceTest, NonLegacyMetaSpace) {
  Metaspace pretok(U'▁', PrependScheme::kAlways, true);
  pretok.SetPrependScheme(PrependScheme::kAlways);
  EXPECT_EQ(pretok, Metaspace(U'▁', PrependScheme::kAlways, true));

  pretok.SetPrependScheme(PrependScheme::kNever);
  EXPECT_EQ(pretok, Metaspace(U'▁', PrependScheme::kNever, true));

  pretok.SetPrependScheme(PrependScheme::kFirst);
  EXPECT_EQ(pretok, Metaspace(U'▁', PrependScheme::kFirst, true));

  const Metaspace pretok_first(U'▁', PrependScheme::kFirst, false);
  PreTokenizedString pretokenized =
      PreTokenizedString::From("Hey my friend <s>how▁are you");
  ASSERT_OK_AND_ASSIGN(const SysRegex re_ref, SysRegex::New(R"((<s>))"));
  const auto split_on_re = [&re_ref](size_t, NormalizedString sequence)
      -> absl::StatusOr<std::vector<NormalizedString>> {
    return sequence.Split(Pattern(&re_ref), SplitDelimiterBehavior::kIsolated);
  };
  ASSERT_OK(pretokenized.Split(split_on_re)) << "Bad split";

  ASSERT_OK(pretok_first.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList({{"▁Hey▁my▁friend▁", {0, 23}},
                 {"<s>", {23, 26}},
                 {"how▁are▁you", {26, 41}}}));
  const Metaspace pretok_always(U'▁', PrependScheme::kAlways, true);
  ASSERT_OK(pretok_always.PreTokenize(pretokenized));
  EXPECT_EQ(
      Splits(pretokenized, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList({{"▁Hey", {0, 6}},
                 {"▁my", {6, 11}},
                 {"▁friend", {11, 20}},
                 {"▁", {20, 23}},
                 {"▁<s>", {23, 29}},
                 {"▁how", {29, 35}},
                 {"▁are", {35, 41}},
                 {"▁you", {41, 47}}}));

  const Metaspace pretok2(U'▁', PrependScheme::kFirst, false);
  // test with prefix
  PreTokenizedString pretokenized2 = PreTokenizedString::From(" Hey <s>how");
  ASSERT_OK(pretokenized2.Split(split_on_re)) << "Bad split";
  ASSERT_OK(pretok2.PreTokenize(pretokenized2));
  EXPECT_EQ(
      Splits(pretokenized2, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList({{"▁Hey▁", {0, 9}}, {"<s>", {9, 12}}, {"how", {12, 15}}}));

  // test with many splits
  PreTokenizedString pretokenized3 =
      PreTokenizedString::From(" Hey <s>how <s>are <s> you");
  ASSERT_OK(pretokenized3.Split(split_on_re)) << "Bad split";
  ASSERT_OK(pretok2.PreTokenize(pretokenized3));
  EXPECT_EQ(
      Splits(pretokenized3, OffsetReferential::kNormalized, OffsetType::kByte),
      SplitList({
          {"▁Hey▁", {0, 9}},
          {"<s>", {9, 12}},
          {"how▁", {12, 18}},
          {"<s>", {18, 21}},
          {"are▁", {21, 27}},
          {"<s>", {27, 30}},
          {"▁you", {30, 36}},
      }));
}

TEST(MetaspaceTest, Decode) {
  const Metaspace decoder(U'▁', PrependScheme::kAlways, true);
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       decoder.DecodeChain({"▁Hey", "▁friend!"}));
  EXPECT_THAT(res, ElementsAre("Hey", " friend!"));

  const Metaspace decoder_never(U'▁', PrependScheme::kNever, true);
  ASSERT_OK_AND_ASSIGN(res, decoder_never.DecodeChain({"▁Hey", "▁friend!"}));
  EXPECT_THAT(res, ElementsAre(" Hey", " friend!"));
}

}  // namespace
}  // namespace pre_tokenizers
}  // namespace tokenizers
