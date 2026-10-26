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
// C++ port of the `#[cfg(test)]` module of `src/processors/template.rs` from
// the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0, Copyright
// 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/processors/template.h"

#include <cstddef>
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

TEST(TemplateTest, PieceSerde) {
  Piece seq_0 = Piece::Sequence(SequenceId::kA, 0);
  const std::string seq_0_s = R"({"Sequence":{"id":"A","type_id":0}})";
  EXPECT_EQ(DumpJson(seq_0.ToJson()), seq_0_s);
  ASSERT_OK_AND_ASSIGN(Json seq_0_json, ParseJson(seq_0_s));
  ASSERT_OK_AND_ASSIGN(Piece seq_0_reconstructed, Piece::FromJson(seq_0_json));
  EXPECT_EQ(seq_0_reconstructed, seq_0);

  Piece seq_1 = Piece::Sequence(SequenceId::kB, 1);
  const std::string seq_1_s = R"({"Sequence":{"id":"B","type_id":1}})";
  EXPECT_EQ(DumpJson(seq_1.ToJson()), seq_1_s);
  ASSERT_OK_AND_ASSIGN(Json seq_1_json, ParseJson(seq_1_s));
  ASSERT_OK_AND_ASSIGN(Piece seq_1_reconstructed, Piece::FromJson(seq_1_json));
  EXPECT_EQ(seq_1_reconstructed, seq_1);

  Piece spe = Piece::SpecialToken("[CLS]", 0);
  const std::string spe_s = R"({"SpecialToken":{"id":"[CLS]","type_id":0}})";
  EXPECT_EQ(DumpJson(spe.ToJson()), spe_s);
  ASSERT_OK_AND_ASSIGN(Json spe_json, ParseJson(spe_s));
  ASSERT_OK_AND_ASSIGN(Piece spe_reconstructed, Piece::FromJson(spe_json));
  EXPECT_EQ(spe_reconstructed, spe);
}

TEST(TemplateTest, Piece) {
  ASSERT_OK_AND_ASSIGN(Piece p1, Piece::TryFrom("$"));
  EXPECT_EQ(p1, Piece::Sequence(SequenceId::kA, 0));
  ASSERT_OK_AND_ASSIGN(Piece p2, Piece::TryFrom("$B"));
  EXPECT_EQ(p2, Piece::Sequence(SequenceId::kB, 0));
  ASSERT_OK_AND_ASSIGN(Piece p3, Piece::TryFrom("$1"));
  EXPECT_EQ(p3, Piece::Sequence(SequenceId::kA, 1));
  ASSERT_OK_AND_ASSIGN(Piece p4, Piece::TryFrom("$B:2"));
  EXPECT_EQ(p4, Piece::Sequence(SequenceId::kB, 2));
  ASSERT_OK_AND_ASSIGN(Piece p5, Piece::TryFrom("$:1"));
  EXPECT_EQ(p5, Piece::Sequence(SequenceId::kA, 1));
  EXPECT_FALSE(Piece::TryFrom("$C:1").ok());
  EXPECT_FALSE(Piece::TryFrom("$A:").ok());
}

TEST(TemplateTest, SpecialTokenSerde) {
  SpecialToken simple("[CLS]", 0);
  const std::string simple_s = R"({"id":"[CLS]","ids":[0],"tokens":["[CLS]"]})";
  EXPECT_EQ(DumpJson(simple.ToJson()), simple_s);
  ASSERT_OK_AND_ASSIGN(Json simple_json, ParseJson(simple_s));
  ASSERT_OK_AND_ASSIGN(SpecialToken simple_reconstructed,
                       SpecialToken::FromJson(simple_json));
  EXPECT_EQ(simple_reconstructed, simple);

  ASSERT_OK_AND_ASSIGN(
      SpecialToken complete,
      SpecialToken::New("[2FR]", {1, 2, 3}, {"convert", "to", "FR"}));
  const std::string complete_s =
      R"({"id":"[2FR]","ids":[1,2,3],"tokens":["convert","to","FR"]})";
  EXPECT_EQ(DumpJson(complete.ToJson()), complete_s);
  ASSERT_OK_AND_ASSIGN(Json complete_json, ParseJson(complete_s));
  ASSERT_OK_AND_ASSIGN(SpecialToken complete_reconstructed,
                       SpecialToken::FromJson(complete_json));
  EXPECT_EQ(complete_reconstructed, complete);

  absl::StatusOr<SpecialToken> malformed1 =
      SpecialToken::New("[2FR]", {1, 2}, {"convert", "to", "FR"});
  EXPECT_FALSE(malformed1.ok());
  absl::StatusOr<SpecialToken> malformed2 =
      SpecialToken::New("[2FR]", {1, 2, 3}, {"convert", "FR"});
  EXPECT_FALSE(malformed2.ok());
}

TEST(TemplateTest, TemplateSerde) {
  Template tmpl(
      {Piece::Sequence(SequenceId::kA, 0), Piece::SpecialToken("[CLS]", 0)});
  const std::string tmpl_s =
      R"([{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[CLS]","type_id":0}}])";
  EXPECT_EQ(DumpJson(tmpl.ToJson()), tmpl_s);
  ASSERT_OK_AND_ASSIGN(Json tmpl_json, ParseJson(tmpl_s));
  ASSERT_OK_AND_ASSIGN(Template tmpl_reconstructed,
                       Template::FromJson(tmpl_json));
  EXPECT_EQ(tmpl_reconstructed, tmpl);
}

TEST(TemplateTest, TokensSerde) {
  Tokens tokens(std::vector<SpecialToken>{SpecialToken("[CLS]", 1),
                                          SpecialToken("[SEP]", 0)});
  const std::string tokens_s =
      R"({"[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[0],"tokens":["[SEP]"]}})";
  EXPECT_EQ(DumpJson(tokens.ToJson()), tokens_s);
  ASSERT_OK_AND_ASSIGN(Json tokens_json, ParseJson(tokens_s));
  ASSERT_OK_AND_ASSIGN(Tokens tokens_reconstructed,
                       Tokens::FromJson(tokens_json));
  EXPECT_EQ(tokens_reconstructed, tokens);
}

TemplateProcessing GetBertTemplate() {
  TemplateProcessingBuilder builder = TemplateProcessing::Builder();
  EXPECT_OK(
      builder.TrySingle(std::vector<std::string>{"[CLS]", "$0", "[SEP]"}));
  EXPECT_OK(builder.TryPair("[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1"));
  builder.SpecialTokens(std::vector<SpecialToken>{SpecialToken("[CLS]", 1),
                                                  SpecialToken("[SEP]", 0)});
  absl::StatusOr<TemplateProcessing> result = builder.Build();
  EXPECT_OK(result);
  return *result;
}

TEST(TemplateTest, TemplateProcessingSerde) {
  TemplateProcessing tmpl = GetBertTemplate();
  const std::string tmpl_s =
      R"({"type":"TemplateProcessing",)"
      R"("single":[)"
      R"({"SpecialToken":{"id":"[CLS]","type_id":0}},)"
      R"({"Sequence":{"id":"A","type_id":0}},)"
      R"({"SpecialToken":{"id":"[SEP]","type_id":0}})"
      R"(],)"
      R"("pair":[)"
      R"({"SpecialToken":{"id":"[CLS]","type_id":0}},)"
      R"({"Sequence":{"id":"A","type_id":0}},)"
      R"({"SpecialToken":{"id":"[SEP]","type_id":0}},)"
      R"({"Sequence":{"id":"B","type_id":1}},)"
      R"({"SpecialToken":{"id":"[SEP]","type_id":1}})"
      R"(],)"
      R"("special_tokens":{)"
      R"("[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]},)"
      R"("[SEP]":{"id":"[SEP]","ids":[0],"tokens":["[SEP]"]})"
      R"(}})";
  EXPECT_EQ(DumpJson(tmpl.ToJson()), tmpl_s);
  ASSERT_OK_AND_ASSIGN(Json tmpl_json, ParseJson(tmpl_s));
  ASSERT_OK_AND_ASSIGN(TemplateProcessing tmpl_reconstructed,
                       TemplateProcessing::FromJson(tmpl_json));
  EXPECT_EQ(tmpl_reconstructed, tmpl);
}

TEST(TemplateTest, MissingSpecialTokens) {
  TemplateProcessingBuilder builder = TemplateProcessing::Builder();
  ASSERT_OK(builder.TrySingle("[CLS] $0 [SEP]"));
  ASSERT_OK(builder.TryPair("[CLS] $A:0 [SEP] $B:1 [SEP]"));
  absl::StatusOr<TemplateProcessing> processor = builder.Build();
  ASSERT_FALSE(processor.ok());
  const absl::string_view err = processor.status().message();
  EXPECT_TRUE(err == "Missing SpecialToken(s) with id(s) `[SEP], [CLS]`" ||
              err == "Missing SpecialToken(s) with id(s) `[CLS], [SEP]`")
      << err;
}

TEST(TemplateTest, TemplateProcessing) {
  TemplateProcessing processor = GetBertTemplate();
  EXPECT_EQ(processor.AddedTokens(false), 2u);
  EXPECT_EQ(processor.AddedTokens(true), 3u);

  Encoding encoding = Encoding::FromTokens(
      {Token(12, "Hello", {0, 5}), Token(14, "there", {6, 11})}, 0);
  Encoding pair = Encoding::FromTokens({Token(15, "pair", {0, 4})}, 0);
  ASSERT_OK_AND_ASSIGN(Encoding single_encoding,
                       processor.Process(encoding, std::nullopt, true));
  EXPECT_EQ(single_encoding,
            Encoding({1, 12, 14, 0}, {0, 0, 0, 0},
                     {"[CLS]", "Hello", "there", "[SEP]"},
                     {std::nullopt, std::nullopt, std::nullopt, std::nullopt},
                     {{0, 0}, {0, 5}, {6, 11}, {0, 0}}, {1, 0, 0, 1},
                     {1, 1, 1, 1}, {}, {{0, {1, 3}}}));
  EXPECT_EQ(single_encoding.TokenToSequence(2), std::optional<size_t>(0));
  EXPECT_EQ(single_encoding.TokenToSequence(3), std::nullopt);
  ASSERT_OK_AND_ASSIGN(Encoding pair_encoding,
                       processor.Process(encoding, pair, true));
  EXPECT_EQ(pair_encoding,
            Encoding({1, 12, 14, 0, 15, 0}, {0, 0, 0, 0, 1, 1},
                     {"[CLS]", "Hello", "there", "[SEP]", "pair", "[SEP]"},
                     {std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                      std::nullopt, std::nullopt},
                     {{0, 0}, {0, 5}, {6, 11}, {0, 0}, {0, 4}, {0, 0}},
                     {1, 0, 0, 1, 0, 1}, {1, 1, 1, 1, 1, 1}, {},
                     {{0, {1, 3}}, {1, {4, 5}}}));
  EXPECT_EQ(pair_encoding.TokenToSequence(2), std::optional<size_t>(0));
  EXPECT_EQ(pair_encoding.TokenToSequence(3), std::nullopt);
  EXPECT_EQ(pair_encoding.TokenToSequence(4), std::optional<size_t>(1));
  EXPECT_EQ(pair_encoding.TokenToSequence(5), std::nullopt);
}

TEST(TemplateTest, TemplateProcessingOverflowing) {
  TemplateProcessing processor = GetBertTemplate();
  EXPECT_EQ(processor.AddedTokens(false), 2u);
  EXPECT_EQ(processor.AddedTokens(true), 3u);

  Encoding encoding = Encoding::FromTokens(
      {Token(12, "Hello", {0, 5}), Token(14, "there", {6, 11})}, 0);
  Encoding overflowing = Encoding::FromTokens({Token(13, "you", {12, 15})}, 0);
  encoding.SetOverflowing({overflowing});

  Encoding pair = Encoding::FromTokens(
      {Token(15, "pair", {0, 4}), Token(16, "with", {5, 9})}, 0);
  Encoding pair_overflowing =
      Encoding::FromTokens({Token(17, "info", {10, 14})}, 0);
  pair.SetOverflowing({pair_overflowing});

  ASSERT_OK_AND_ASSIGN(Encoding single_encoding,
                       processor.Process(encoding, std::nullopt, true));
  EXPECT_EQ(
      single_encoding,
      Encoding({1, 12, 14, 0}, {0, 0, 0, 0},
               {"[CLS]", "Hello", "there", "[SEP]"},
               {std::nullopt, std::nullopt, std::nullopt, std::nullopt},
               {{0, 0}, {0, 5}, {6, 11}, {0, 0}}, {1, 0, 0, 1}, {1, 1, 1, 1},
               {Encoding({1, 13, 0}, {0, 0, 0}, {"[CLS]", "you", "[SEP]"},
                         {std::nullopt, std::nullopt, std::nullopt},
                         {{0, 0}, {12, 15}, {0, 0}}, {1, 0, 1}, {1, 1, 1}, {},
                         {{0, {1, 2}}})},
               {{0, {1, 3}}}));
  EXPECT_EQ(single_encoding.TokenToSequence(2), std::optional<size_t>(0));
  EXPECT_EQ(single_encoding.TokenToSequence(3), std::nullopt);
  ASSERT_OK_AND_ASSIGN(Encoding pair_encoding,
                       processor.Process(encoding, pair, true));
  EXPECT_EQ(
      pair_encoding,
      Encoding({1, 12, 14, 0, 15, 16, 0}, {0, 0, 0, 0, 1, 1, 1},
               {"[CLS]", "Hello", "there", "[SEP]", "pair", "with", "[SEP]"},
               {std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt, std::nullopt},
               {{0, 0}, {0, 5}, {6, 11}, {0, 0}, {0, 4}, {5, 9}, {0, 0}},
               {1, 0, 0, 1, 0, 0, 1}, {1, 1, 1, 1, 1, 1, 1},
               {Encoding({1, 13, 0, 15, 16, 0}, {0, 0, 0, 1, 1, 1},
                         {"[CLS]", "you", "[SEP]", "pair", "with", "[SEP]"},
                         {std::nullopt, std::nullopt, std::nullopt,
                          std::nullopt, std::nullopt, std::nullopt},
                         {{0, 0}, {12, 15}, {0, 0}, {0, 4}, {5, 9}, {0, 0}},
                         {1, 0, 1, 0, 0, 1}, {1, 1, 1, 1, 1, 1},
                         {Encoding({1, 13, 0, 17, 0}, {0, 0, 0, 0, 1},
                                   {"[CLS]", "you", "[SEP]", "info", "[SEP]"},
                                   {std::nullopt, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt},
                                   {{0, 0}, {12, 15}, {0, 0}, {10, 14}, {0, 0}},
                                   {1, 0, 1, 0, 1}, {1, 1, 1, 1, 1}, {},
                                   {{0, {1, 2}}, {1, {3, 4}}})},
                         {{1, {3, 5}}, {0, {1, 2}}}),
                Encoding({1, 13, 0, 17, 0}, {0, 0, 0, 0, 1},
                         {"[CLS]", "you", "[SEP]", "info", "[SEP]"},
                         {std::nullopt, std::nullopt, std::nullopt,
                          std::nullopt, std::nullopt},
                         {{0, 0}, {12, 15}, {0, 0}, {10, 14}, {0, 0}},
                         {1, 0, 1, 0, 1}, {1, 1, 1, 1, 1}, {},
                         {{0, {1, 2}}, {1, {3, 4}}}),
                Encoding({1, 12, 14, 0, 17, 0}, {0, 0, 0, 0, 0, 1},
                         {"[CLS]", "Hello", "there", "[SEP]", "info", "[SEP]"},
                         {std::nullopt, std::nullopt, std::nullopt,
                          std::nullopt, std::nullopt, std::nullopt},
                         {{0, 0}, {0, 5}, {6, 11}, {0, 0}, {10, 14}, {0, 0}},
                         {1, 0, 0, 1, 0, 1}, {1, 1, 1, 1, 1, 1},
                         {Encoding({1, 13, 0, 17, 0}, {0, 0, 0, 0, 1},
                                   {"[CLS]", "you", "[SEP]", "info", "[SEP]"},
                                   {std::nullopt, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt},
                                   {{0, 0}, {12, 15}, {0, 0}, {10, 14}, {0, 0}},
                                   {1, 0, 1, 0, 1}, {1, 1, 1, 1, 1}, {},
                                   {{0, {1, 2}}, {1, {3, 4}}})},
                         {{0, {1, 3}}, {1, {4, 5}}})},
               {{0, {1, 3}}, {1, {4, 6}}}));
  EXPECT_EQ(pair_encoding.TokenToSequence(2), std::optional<size_t>(0));
  EXPECT_EQ(pair_encoding.TokenToSequence(3), std::nullopt);
  EXPECT_EQ(pair_encoding.TokenToSequence(4), std::optional<size_t>(1));
  EXPECT_EQ(pair_encoding.TokenToSequence(5), std::optional<size_t>(1));
  EXPECT_EQ(pair_encoding.TokenToSequence(6), std::nullopt);
}

TEST(TemplateTest, PairMustUseBothSequences) {
  TemplateProcessingBuilder builder = TemplateProcessing::Builder();
  ASSERT_OK(builder.TrySingle("$0"));
  ASSERT_OK(builder.TryPair("$0 $1"));
  absl::StatusOr<TemplateProcessing> processor = builder.Build();
  ASSERT_FALSE(processor.ok());
  EXPECT_EQ(processor.status().message(),
            "Template for `pair` must use both sequences");
}

TEST(TemplateTest, ExpectWrongErrorMessage) {
  TemplateProcessingBuilder builder = TemplateProcessing::Builder();
  ASSERT_OK(builder.TrySingle("$0"));
  ASSERT_OK(builder.TryPair("$0 $1"));
  absl::StatusOr<TemplateProcessing> processor = builder.Build();
  ASSERT_FALSE(processor.ok());
  EXPECT_NE(processor.status().message(),
            "Expect the left side error message to be different from the right "
            "side!");
}

}  // namespace
}  // namespace processors
}  // namespace tokenizers
