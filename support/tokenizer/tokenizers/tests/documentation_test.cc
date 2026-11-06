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
// C++ port of `tests/documentation.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: `train_tokenizer`, `quicktour_slow_train`, and
// `train_pipeline_bert` are not ported (`BpeTrainer` / `WordPieceTrainer`
// model training is out of scope).

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/byte_fallback.h"
#include "support/tokenizer/tokenizers/decoders/wordpiece.h"
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/normalizers/normalizers.h"
#include "support/tokenizer/tokenizers/normalizers/strip.h"
#include "support/tokenizer/tokenizers/normalizers/unicode.h"
#include "support/tokenizer/tokenizers/normalizers/utils.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/digits.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/pre_tokenizers.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/sequence.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/whitespace.h"
#include "support/tokenizer/tokenizers/processors/template.h"
#include "support/tokenizer/tokenizers/tests/common.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/padding.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::tokenizers::decoders::ByteFallback;
using ::tokenizers::models::bpe::BPE;
using ::tokenizers::normalizers::NFC;
using ::tokenizers::normalizers::NFD;
using ::tokenizers::normalizers::NormalizerWrapper;
using ::tokenizers::normalizers::Strip;
using ::tokenizers::normalizers::StripAccents;
using ::tokenizers::pre_tokenizers::ByteLevel;
using ::tokenizers::pre_tokenizers::Digits;
using ::tokenizers::pre_tokenizers::PreTokenizerWrapper;
using ::tokenizers::pre_tokenizers::Whitespace;
using ::tokenizers::processors::SpecialToken;
using ::tokenizers::processors::TemplateProcessing;
using ::tokenizers::processors::TemplateProcessingBuilder;
using ::tokenizers::test_common::TestdataPath;
using NormalizerSequence = ::tokenizers::normalizers::Sequence;
using PreTokenizerSequence = ::tokenizers::pre_tokenizers::Sequence;
using WordPieceDecoder = ::tokenizers::decoders::WordPiece;

TEST(DocumentationTest, LoadTokenizer) {
  ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer,
                       Tokenizer::FromFile(TestdataPath("roberta.json")));

  const char* example = "This is an example";
  std::vector<uint32_t> ids = {713, 16, 41, 1246};
  std::vector<std::string> tokens = {"This", "Ġis", "Ġan", "Ġexample"};

  ASSERT_OK_AND_ASSIGN(Encoding encodings, tokenizer.Encode(example, false));

  EXPECT_EQ(encodings.GetIds(), ids);
  EXPECT_EQ(encodings.GetTokens(), tokens);

  ASSERT_OK_AND_ASSIGN(std::string decoded, tokenizer.Decode(ids, false));
  EXPECT_EQ(decoded, example);
}

TEST(DocumentationTest, StreamingTokenizer) {
  ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer,
                       Tokenizer::FromFile(TestdataPath("roberta.json")));

  DecodeStream decode_stream = tokenizer.DecodeStream(false);
  EXPECT_EQ(decode_stream.Step(713).value(),
            std::optional<std::string>("This"));
  EXPECT_EQ(decode_stream.Step(16).value(), std::optional<std::string>(" is"));
  EXPECT_EQ(decode_stream.Step(41).value(), std::optional<std::string>(" an"));
  EXPECT_EQ(decode_stream.Step(1246).value(),
            std::optional<std::string>(" example"));

  ASSERT_OK_AND_ASSIGN(
      Tokenizer albert_tokenizer,
      Tokenizer::FromFile(TestdataPath("albert-base-v1-tokenizer.json")));
  ASSERT_OK_AND_ASSIGN(Encoding encoded,
                       albert_tokenizer.Encode("This is an example", false));
  EXPECT_EQ(encoded.GetIds(), (std::vector<uint32_t>{48, 25, 40, 823}));
  DecodeStream albert_stream1 = albert_tokenizer.DecodeStream(false);
  // No space anymore
  EXPECT_EQ(albert_stream1.Step(25).value(), std::optional<std::string>("is"));
  DecodeStream albert_stream2 = albert_tokenizer.DecodeStream(false);
  EXPECT_EQ(albert_stream2.Step(48).value(),
            std::optional<std::string>("this"));
  EXPECT_EQ(albert_stream2.Step(25).value(), std::optional<std::string>(" is"));
  EXPECT_EQ(albert_stream2.Step(40).value(), std::optional<std::string>(" an"));
  EXPECT_EQ(albert_stream2.Step(823).value(),
            std::optional<std::string>(" example"));

  // None example
  absl::flat_hash_map<std::string, uint32_t> vocab = {
      {"<0x20>", 0},
      {"<0xC3>", 1},
      {"<0xA9>", 2},
      {" This", 3},
  };
  ASSERT_OK_AND_ASSIGN(BPE bpe, BPE::Builder()
                                    .VocabAndMerges(std::move(vocab), {})
                                    .ByteFallback(true)
                                    .Build());
  ASSERT_OK_AND_ASSIGN(Tokenizer fallback_tokenizer,
                       TokenizerBuilder()
                           .WithModel(std::move(bpe))
                           .WithNormalizer(NormalizerSequence({
                               NormalizerWrapper(Strip(true, true)),
                               NormalizerWrapper(NFC()),
                           }))
                           .WithPreTokenizer(ByteLevel())
                           .WithPostProcessor(ByteLevel())
                           .WithDecoder(ByteFallback())
                           .Build());
  DecodeStream fallback_stream = fallback_tokenizer.DecodeStream(false);
  EXPECT_EQ(fallback_stream.Step(0).value(), std::optional<std::string>(" "));
  EXPECT_EQ(fallback_stream.Step(1).value(), std::nullopt);
  EXPECT_EQ(fallback_stream.Step(2).value(), std::optional<std::string>("é"));
  EXPECT_EQ(fallback_stream.Step(2).value(), std::nullopt);
}

TEST(DocumentationTest, Quicktour) {
  // START quicktour_reload_tokenizer
  ASSERT_OK_AND_ASSIGN(
      Tokenizer tokenizer,
      Tokenizer::FromFile(TestdataPath("tokenizer-wiki.json")));
  // END quicktour_reload_tokenizer
  // START quicktour_encode
  ASSERT_OK_AND_ASSIGN(
      Encoding output,
      tokenizer.Encode("Hello, y'all! How are you 😁 ?", true));
  // END quicktour_encode
  EXPECT_EQ(output.GetTokens(),
            (std::vector<std::string>{"Hello", ",", "y", "'", "all", "!", "How",
                                      "are", "you", "[UNK]", "?"}));
  EXPECT_EQ(output.GetIds(), (std::vector<uint32_t>{27253, 16, 93, 11, 5097, 5,
                                                    7961, 5112, 6218, 0, 35}));
  EXPECT_EQ(output.GetOffsets()[9], Offsets(26, 30));
  absl::string_view sentence = "Hello, y'all! How are you 😁 ?";
  EXPECT_EQ(sentence.substr(26, 4), "😁");
  EXPECT_EQ(tokenizer.TokenToId("[SEP]"), std::optional<uint32_t>(2));

  // START quicktour_init_template_processing
  std::vector<SpecialToken> special_tokens = {
      SpecialToken("[CLS]", tokenizer.TokenToId("[CLS]").value()),
      SpecialToken("[SEP]", tokenizer.TokenToId("[SEP]").value()),
  };
  TemplateProcessingBuilder tpl_builder = TemplateProcessing::Builder();
  ASSERT_OK(tpl_builder.TrySingle("[CLS] $A [SEP]"));
  ASSERT_OK(tpl_builder.TryPair("[CLS] $A [SEP] $B:1 [SEP]:1"));
  tpl_builder.SpecialTokens(std::move(special_tokens));
  ASSERT_OK_AND_ASSIGN(TemplateProcessing tpl, tpl_builder.Build());
  tokenizer.WithPostProcessor(std::move(tpl));
  // END quicktour_init_template_processing

  ASSERT_OK_AND_ASSIGN(
      Encoding output2,
      tokenizer.Encode("Hello, y'all! How are you 😁 ?", true));
  EXPECT_EQ(
      output2.GetTokens(),
      (std::vector<std::string>{"[CLS]", "Hello", ",", "y", "'", "all", "!",
                                "How", "are", "you", "[UNK]", "?", "[SEP]"}));

  ASSERT_OK_AND_ASSIGN(
      Encoding output_pair,
      tokenizer.Encode(EncodeInput("Hello, y'all!", "How are you 😁 ?"), true));
  EXPECT_EQ(output_pair.GetTokens(),
            (std::vector<std::string>{"[CLS]", "Hello", ",", "y", "'", "all",
                                      "!", "[SEP]", "How", "are", "you",
                                      "[UNK]", "?", "[SEP]"}));
  EXPECT_EQ(output_pair.GetTypeIds(),
            (std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}));

  ASSERT_OK_AND_ASSIGN(std::vector<Encoding> batch_out,
                       tokenizer.EncodeBatch({EncodeInput("Hello, y'all!"),
                                              EncodeInput("How are you 😁 ?")},
                                             true));
  (void)batch_out;
  ASSERT_OK_AND_ASSIGN(
      std::vector<Encoding> batch_pair_out,
      tokenizer.EncodeBatch(
          {
              EncodeInput("Hello, y'all!", "How are you 😁 ?"),
              EncodeInput("Hello to you too!", "I'm fine, thank you!"),
          },
          true));
  (void)batch_pair_out;

  PaddingParams padding_params;
  padding_params.pad_id = 3;
  padding_params.pad_token = "[PAD]";
  tokenizer.WithPadding(padding_params);

  ASSERT_OK_AND_ASSIGN(std::vector<Encoding> padded_batch,
                       tokenizer.EncodeBatch({EncodeInput("Hello, y'all!"),
                                              EncodeInput("How are you 😁 ?")},
                                             true));
  EXPECT_EQ(padded_batch[1].GetTokens(),
            (std::vector<std::string>{"[CLS]", "How", "are", "you", "[UNK]",
                                      "?", "[SEP]", "[PAD]"}));
  EXPECT_EQ(padded_batch[1].GetAttentionMask(),
            (std::vector<uint32_t>{1, 1, 1, 1, 1, 1, 1, 0}));
}

TEST(DocumentationTest, Pipeline) {
  ASSERT_OK_AND_ASSIGN(
      Tokenizer tokenizer,
      Tokenizer::FromFile(TestdataPath("tokenizer-wiki.json")));

  NormalizerSequence normalizer({
      NormalizerWrapper(NFD()),
      NormalizerWrapper(StripAccents()),
  });

  NormalizedString normalized = NormalizedString::From("Héllò hôw are ü?");
  ASSERT_OK(normalizer.Normalize(normalized));
  EXPECT_EQ(normalized.Get(), "Hello how are u?");

  ASSERT_OK(tokenizer.WithNormalizer(normalizer));

  Whitespace pre_tokenizer;
  PreTokenizedString pre_tokenized =
      PreTokenizedString::From("Hello! How are you? I'm fine, thank you.");
  ASSERT_OK(pre_tokenizer.PreTokenize(pre_tokenized));

  auto splits =
      pre_tokenized.GetSplits(OffsetReferential::kOriginal, OffsetType::kByte);
  std::vector<std::pair<std::string, Offsets>> actual_splits;
  for (const auto& s : splits) {
    actual_splits.emplace_back(std::string(std::get<0>(s)), std::get<1>(s));
    EXPECT_FALSE(std::get<2>(s)->has_value());
  }
  EXPECT_EQ(actual_splits, (std::vector<std::pair<std::string, Offsets>>{
                               {"Hello", {0, 5}},
                               {"!", {5, 6}},
                               {"How", {7, 10}},
                               {"are", {11, 14}},
                               {"you", {15, 18}},
                               {"?", {18, 19}},
                               {"I", {20, 21}},
                               {"'", {21, 22}},
                               {"m", {22, 23}},
                               {"fine", {24, 28}},
                               {",", {28, 29}},
                               {"thank", {30, 35}},
                               {"you", {36, 39}},
                               {".", {39, 40}},
                           }));

  PreTokenizerSequence combined_pre_tokenizer({
      PreTokenizerWrapper(Whitespace()),
      PreTokenizerWrapper(Digits(true)),
  });
  PreTokenizedString pre_tokenized2 = PreTokenizedString::From("Call 911!");
  ASSERT_OK(combined_pre_tokenizer.PreTokenize(pre_tokenized2));

  auto splits2 =
      pre_tokenized2.GetSplits(OffsetReferential::kOriginal, OffsetType::kByte);
  std::vector<std::pair<std::string, Offsets>> actual_splits2;
  for (const auto& s : splits2) {
    actual_splits2.emplace_back(std::string(std::get<0>(s)), std::get<1>(s));
    EXPECT_FALSE(std::get<2>(s)->has_value());
  }
  EXPECT_EQ(actual_splits2, (std::vector<std::pair<std::string, Offsets>>{
                                {"Call", {0, 4}},
                                {"9", {5, 6}},
                                {"1", {6, 7}},
                                {"1", {7, 8}},
                                {"!", {8, 9}},
                            }));

  tokenizer.WithPreTokenizer(combined_pre_tokenizer);

  TemplateProcessingBuilder tpl_builder = TemplateProcessing::Builder();
  ASSERT_OK(tpl_builder.TrySingle("[CLS] $A [SEP]"));
  ASSERT_OK(tpl_builder.TryPair("[CLS] $A [SEP] $B:1 [SEP]:1"));
  tpl_builder.SpecialTokens(std::vector<SpecialToken>{
      SpecialToken("[CLS]", 1), SpecialToken("[SEP]", 2)});
  ASSERT_OK_AND_ASSIGN(TemplateProcessing tpl, tpl_builder.Build());
  tokenizer.WithPostProcessor(std::move(tpl));

  ASSERT_OK_AND_ASSIGN(
      Encoding output,
      tokenizer.Encode("Hello, y'all! How are you 😁 ?", true));
  EXPECT_EQ(output.GetIds(),
            (std::vector<uint32_t>{1, 27253, 16, 93, 11, 5097, 5, 7961, 5112,
                                   6218, 0, 35, 2}));

  ASSERT_OK_AND_ASSIGN(
      std::string decoded,
      tokenizer.Decode(
          {1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2}, true));
  EXPECT_EQ(decoded, "Hello , y ' all ! How are you ?");
}

TEST(DocumentationTest, PipelineBert) {
  ASSERT_OK_AND_ASSIGN(Tokenizer bert_tokenizer,
                       Tokenizer::FromFile(TestdataPath("bert-wiki.json")));

  ASSERT_OK_AND_ASSIGN(
      Encoding output,
      bert_tokenizer.Encode("Welcome to the 🤗 Tokenizers library.", true));
  EXPECT_EQ(output.GetTokens(),
            (std::vector<std::string>{"[CLS]", "welcome", "to", "the", "[UNK]",
                                      "tok", "##eni", "##zer", "##s", "library",
                                      ".", "[SEP]"}));

  ASSERT_OK_AND_ASSIGN(std::string decoded,
                       bert_tokenizer.Decode(output.GetIds(), true));
  EXPECT_EQ(decoded, "welcome to the tok ##eni ##zer ##s library .");

  bert_tokenizer.WithDecoder(WordPieceDecoder());
  ASSERT_OK_AND_ASSIGN(std::string proper_decoded,
                       bert_tokenizer.Decode(output.GetIds(), true));
  EXPECT_EQ(proper_decoded, "welcome to the tokenizers library.");
}

}  // namespace
}  // namespace tokenizers
