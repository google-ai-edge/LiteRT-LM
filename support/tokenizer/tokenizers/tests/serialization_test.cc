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
// C++ port of `tests/serialization.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/decoders/decoders.h"
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/models/wordlevel/wordlevel.h"
#include "support/tokenizer/tokenizers/models/wordpiece/wordpiece.h"
#include "support/tokenizer/tokenizers/normalizers/bert.h"
#include "support/tokenizer/tokenizers/normalizers/normalizers.h"
#include "support/tokenizer/tokenizers/normalizers/unicode.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/bert.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/delimiter.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/pre_tokenizers.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/split.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/whitespace.h"
#include "support/tokenizer/tokenizers/processors/bert.h"
#include "support/tokenizer/tokenizers/processors/processors.h"
#include "support/tokenizer/tokenizers/tests/common.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::tokenizers::decoders::DecoderWrapper;
using ::tokenizers::models::ModelWrapper;
using ::tokenizers::models::bpe::BPE;
using ::tokenizers::models::wordlevel::WordLevel;
using ::tokenizers::models::wordpiece::WordPiece;
using ::tokenizers::normalizers::BertNormalizer;
using ::tokenizers::normalizers::NFC;
using ::tokenizers::normalizers::NFKC;
using ::tokenizers::normalizers::NormalizerWrapper;
using ::tokenizers::pre_tokenizers::BertPreTokenizer;
using ::tokenizers::pre_tokenizers::ByteLevel;
using ::tokenizers::pre_tokenizers::CharDelimiterSplit;
using ::tokenizers::pre_tokenizers::PreTokenizerWrapper;
using ::tokenizers::pre_tokenizers::Split;
using ::tokenizers::pre_tokenizers::SplitPattern;
using ::tokenizers::pre_tokenizers::Whitespace;
using ::tokenizers::processors::BertProcessing;
using ::tokenizers::processors::PostProcessorWrapper;
using ::tokenizers::test_common::GetBertWordpiece;
using ::tokenizers::test_common::GetByteLevelBpe;
using ::tokenizers::test_common::TestdataPath;

TEST(SerializationIntegrationTest, BpeSerde) {
  BPE bpe = GetByteLevelBpe();
  std::string ser = DumpJson(bpe.ToJson());
  ASSERT_OK_AND_ASSIGN(Json ser_json, ParseJson(ser));
  ASSERT_OK_AND_ASSIGN(BPE de, BPE::FromJson(ser_json));
  EXPECT_EQ(bpe, de);
}

TEST(SerializationIntegrationTest, WordpieceSerde) {
  WordPiece wordpiece = GetBertWordpiece();
  std::string ser = DumpJson(wordpiece.ToJson());
  ASSERT_OK_AND_ASSIGN(Json ser_json, ParseJson(ser));
  ASSERT_OK_AND_ASSIGN(WordPiece de, WordPiece::FromJson(ser_json));
  EXPECT_EQ(wordpiece, de);
}

TEST(SerializationIntegrationTest, WordlevelSerde) {
  ASSERT_OK_AND_ASSIGN(
      WordLevel wordlevel,
      WordLevel::FromFile(TestdataPath("gpt2-vocab.json"), "<unk>"));
  std::string ser = DumpJson(wordlevel.ToJson());
  ASSERT_OK_AND_ASSIGN(Json ser_json, ParseJson(ser));
  ASSERT_OK_AND_ASSIGN(WordLevel de, WordLevel::FromJson(ser_json));
  EXPECT_EQ(wordlevel, de);
}

TEST(SerializationIntegrationTest, Normalizers) {
  // Test unit struct
  NFC nfc;
  std::string nfc_ser = DumpJson(nfc.ToJson());
  EXPECT_EQ(nfc_ser, R"({"type":"NFC"})");
  ASSERT_OK_AND_ASSIGN(Json nfc_json, ParseJson(nfc_ser));
  // empty struct can deserialize from self
  ASSERT_OK(NFC::FromJson(nfc_json).status());
  EXPECT_FALSE(NFKC::FromJson(nfc_json).ok())
      << "NFKC shouldn't be deserializable from NFC";
  // wrapper can deserialize from inner
  ASSERT_OK_AND_ASSIGN(NormalizerWrapper nfc_wrapped,
                       NormalizerWrapper::FromJson(nfc_json));
  EXPECT_TRUE(nfc_wrapped.Is<NFC>()) << "NFC wrapped with incorrect variant";
  std::string ser_wrapped = DumpJson(nfc_wrapped.ToJson());
  EXPECT_EQ(ser_wrapped, nfc_ser);

  // Test non-empty roundtrip
  BertNormalizer bert;
  std::string bert_ser = DumpJson(bert.ToJson());
  EXPECT_EQ(
      bert_ser,
      R"({"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true})");
  ASSERT_OK_AND_ASSIGN(Json bert_json, ParseJson(bert_ser));
  // make sure we can deserialize to self
  ASSERT_OK(BertNormalizer::FromJson(bert_json).status());
  // wrapper can deserialize from inner serialization
  ASSERT_OK_AND_ASSIGN(NormalizerWrapper bert_wrapped,
                       NormalizerWrapper::FromJson(bert_json));
  EXPECT_TRUE(bert_wrapped.Is<BertNormalizer>())
      << "BertNormalizer wrapped with incorrect variant";
  // wrapped serializes same way as inner
  std::string bert_ser_wrapped = DumpJson(bert_wrapped.ToJson());
  EXPECT_EQ(bert_ser_wrapped, bert_ser);
}

TEST(SerializationIntegrationTest, Processors) {
  BertProcessing bert({"SEP", 0}, {"CLS", 0});
  std::string bert_ser = DumpJson(bert.ToJson());
  EXPECT_EQ(bert_ser,
            R"({"type":"BertProcessing","sep":["SEP",0],"cls":["CLS",0]})");
  ASSERT_OK_AND_ASSIGN(Json bert_json, ParseJson(bert_ser));
  ASSERT_OK(BertProcessing::FromJson(bert_json).status());
  ASSERT_OK_AND_ASSIGN(PostProcessorWrapper bert_wrapped,
                       PostProcessorWrapper::FromJson(bert_json));
  EXPECT_TRUE(bert_wrapped.Is<BertProcessing>())
      << "Bert wrapped with incorrect variant";
  std::string ser_wrapped = DumpJson(bert_wrapped.ToJson());
  EXPECT_EQ(ser_wrapped, bert_ser);
}

TEST(SerializationIntegrationTest, Pretoks) {
  // Test unit struct
  BertPreTokenizer bert;
  std::string bert_ser = DumpJson(bert.ToJson());
  EXPECT_EQ(bert_ser, R"({"type":"BertPreTokenizer"})");
  ASSERT_OK_AND_ASSIGN(Json bert_json, ParseJson(bert_ser));
  // empty struct can deserialize from self
  ASSERT_OK(BertPreTokenizer::FromJson(bert_json).status());
  EXPECT_FALSE(Whitespace::FromJson(bert_json).ok())
      << "Whitespace shouldn't be deserializable from BertPreTokenizer";
  // wrapper can deserialize from inner
  ASSERT_OK_AND_ASSIGN(PreTokenizerWrapper bert_wrapped,
                       PreTokenizerWrapper::FromJson(bert_json));
  EXPECT_TRUE(bert_wrapped.Is<BertPreTokenizer>())
      << "Bert wrapped with incorrect variant";
  std::string ser_wrapped = DumpJson(bert_wrapped.ToJson());
  EXPECT_EQ(ser_wrapped, bert_ser);

  // Test non-empty roundtrip
  CharDelimiterSplit ch(U' ');
  std::string ch_ser = DumpJson(ch.ToJson());
  EXPECT_EQ(ch_ser, R"({"type":"CharDelimiterSplit","delimiter":" "})");
  ASSERT_OK_AND_ASSIGN(Json ch_json, ParseJson(ch_ser));
  // make sure we can deserialize to self
  ASSERT_OK(CharDelimiterSplit::FromJson(ch_json).status());
  // wrapper can deserialize from inner serialization
  ASSERT_OK_AND_ASSIGN(PreTokenizerWrapper ch_wrapped,
                       PreTokenizerWrapper::FromJson(ch_json));
  EXPECT_TRUE(ch_wrapped.Is<CharDelimiterSplit>())
      << "CharDelimiterSplit wrapped with incorrect variant";
  // wrapped serializes same way as inner
  std::string ch_ser_wrapped = DumpJson(ch_wrapped.ToJson());
  EXPECT_EQ(ch_ser_wrapped, ch_ser);

  Whitespace wsp;
  std::string wsp_ser = DumpJson(wsp.ToJson());
  EXPECT_EQ(wsp_ser, R"({"type":"Whitespace"})");
  ASSERT_OK_AND_ASSIGN(Json wsp_json, ParseJson(wsp_ser));
  ASSERT_OK(Whitespace::FromJson(wsp_json).status());
  EXPECT_FALSE(BertPreTokenizer::FromJson(wsp_json).ok())
      << "BertPreTokenizer shouldn't be deserializable from Whitespace";

  SplitPattern pattern = "[SEP]";
  ASSERT_OK_AND_ASSIGN(
      Split pretok,
      Split::New(pattern, SplitDelimiterBehavior::kIsolated, false));
  std::string pretok_str = DumpJson(pretok.ToJson());
  EXPECT_EQ(
      pretok_str,
      R"({"type":"Split","pattern":{"String":"[SEP]"},"behavior":"Isolated","invert":false})");
  ASSERT_OK_AND_ASSIGN(Json pretok_json, ParseJson(pretok_str));
  ASSERT_OK_AND_ASSIGN(Split pretok_de, Split::FromJson(pretok_json));
  EXPECT_EQ(pretok_de, pretok);

  SplitPattern regex_pattern = SplitPattern::Regex("[SEP]");
  ASSERT_OK_AND_ASSIGN(
      Split regex_pretok,
      Split::New(regex_pattern, SplitDelimiterBehavior::kIsolated, false));
  std::string regex_pretok_str = DumpJson(regex_pretok.ToJson());
  EXPECT_EQ(
      regex_pretok_str,
      R"({"type":"Split","pattern":{"Regex":"[SEP]"},"behavior":"Isolated","invert":false})");
  ASSERT_OK_AND_ASSIGN(Json regex_pretok_json, ParseJson(regex_pretok_str));
  ASSERT_OK_AND_ASSIGN(Split regex_pretok_de,
                       Split::FromJson(regex_pretok_json));
  EXPECT_EQ(regex_pretok_de, regex_pretok);
}

TEST(SerializationIntegrationTest, Decoders) {
  ByteLevel byte_level;
  std::string byte_level_ser = DumpJson(byte_level.ToJson());
  EXPECT_EQ(
      byte_level_ser,
      R"({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true})");
  ASSERT_OK_AND_ASSIGN(Json byte_level_json, ParseJson(byte_level_ser));
  ASSERT_OK(ByteLevel::FromJson(byte_level_json).status());
  ASSERT_OK_AND_ASSIGN(DecoderWrapper byte_level_wrapper,
                       DecoderWrapper::FromJson(byte_level_json));
  EXPECT_TRUE(byte_level_wrapper.Is<ByteLevel>())
      << "ByteLevel wrapped with incorrect variant";
  std::string ser_wrapped = DumpJson(byte_level_wrapper.ToJson());
  EXPECT_EQ(ser_wrapped, byte_level_ser);
}

TEST(SerializationIntegrationTest, Models) {
  BPE bpe;
  std::string bpe_ser = DumpJson(bpe.ToJson());
  ASSERT_OK_AND_ASSIGN(Json bpe_json, ParseJson(bpe_ser));
  ASSERT_OK(BPE::FromJson(bpe_json).status());
  ASSERT_OK_AND_ASSIGN(ModelWrapper bpe_wrapper,
                       ModelWrapper::FromJson(bpe_json));
  EXPECT_TRUE(bpe_wrapper.Is<BPE>()) << "BPE wrapped with incorrect variant";
  std::string ser_wrapped = DumpJson(bpe_wrapper.ToJson());
  EXPECT_EQ(ser_wrapped, bpe_ser);
}

// PORT NOTE: Rust also tests deserializing `TokenizerImpl` instantiated with a
// concrete `NFC` or `NFKC` normalizer type parameter instead of
// `NormalizerWrapper`. In C++, `Tokenizer` is the concrete instantiation over
// the wrapper types (see tokenizer.h); the `NFC` vs `NFKC` rejection on the
// `normalizer` sub-object is exercised above in `Normalizers` and directly on
// the serialized JSON's `normalizer` field here.
TEST(SerializationIntegrationTest, Tokenizer) {
  WordPiece wordpiece;
  Tokenizer tokenizer(wordpiece);
  ASSERT_OK(tokenizer.WithNormalizer(NFC()));
  ASSERT_OK_AND_ASSIGN(std::string ser, tokenizer.ToString(false));
  ASSERT_OK_AND_ASSIGN(Tokenizer de, Tokenizer::FromStr(ser));
  ASSERT_NE(de.GetNormalizer(), nullptr);
  EXPECT_TRUE(de.GetNormalizer()->Is<NFC>());
  EXPECT_TRUE(de.GetModel().Is<WordPiece>());
  ASSERT_OK_AND_ASSIGN(Json ser_json, ParseJson(ser));
  EXPECT_FALSE(NFKC::FromJson(ser_json["normalizer"]).ok())
      << "NFKC shouldn't be deserializable from NFC";
  ASSERT_OK_AND_ASSIGN(std::string de_ser, de.ToString(false));
  EXPECT_EQ(de_ser, ser);
}

TEST(SerializationIntegrationTest, BpeWithDropoutSerde) {
  BPE bpe;
  bpe.set_dropout(0.1f);
  std::string ser = DumpJson(bpe.ToJson());
  ASSERT_OK_AND_ASSIGN(Json ser_json, ParseJson(ser));
  ASSERT_OK_AND_ASSIGN(BPE de, BPE::FromJson(ser_json));
  EXPECT_EQ(bpe, de);

  // set dropout to 0.0 (which is analogous to None) and reserialize
  bpe.set_dropout(0.0f);
  std::string ser2 = DumpJson(bpe.ToJson());
  ASSERT_OK_AND_ASSIGN(Json ser2_json, ParseJson(ser2));
  ASSERT_OK_AND_ASSIGN(BPE de2, BPE::FromJson(ser2_json));
  EXPECT_EQ(bpe, de2);
}

TEST(SerializationIntegrationTest, DeserializeLongFile) {
  ASSERT_OK(Tokenizer::FromFile(TestdataPath("albert-base-v1-tokenizer.json"))
                .status());
}

}  // namespace
}  // namespace tokenizers
