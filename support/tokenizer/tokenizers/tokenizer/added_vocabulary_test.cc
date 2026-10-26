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
// C++ port of the `#[cfg(test)]` module of `src/tokenizer/added_vocabulary.rs`
// from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"

#include <algorithm>
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
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

class ModelMock : public Model {
 public:
  explicit ModelMock(
      std::vector<std::pair<absl::string_view, uint32_t>> entries) {
    for (const auto& [tok, id] : entries) {
      vocab_.insert_or_assign(std::string(tok), id);
      vocab_r_.insert_or_assign(id, std::string(tok));
    }
  }

  absl::StatusOr<std::vector<Token>> Tokenize(
      absl::string_view /*sequence*/) const override {
    return absl::UnimplementedError("ModelMock::Tokenize");
  }

  std::optional<uint32_t> TokenToId(absl::string_view token) const override {
    auto it = vocab_.find(token);
    if (it == vocab_.end()) return std::nullopt;
    return it->second;
  }

  std::optional<std::string> IdToToken(uint32_t id) const override {
    auto it = vocab_r_.find(id);
    if (it == vocab_r_.end()) return std::nullopt;
    return it->second;
  }

  absl::flat_hash_map<std::string, uint32_t> GetVocab() const override {
    return vocab_;
  }

  size_t GetVocabSize() const override { return vocab_.size(); }

 private:
  absl::flat_hash_map<std::string, uint32_t> vocab_;
  absl::flat_hash_map<uint32_t, std::string> vocab_r_;
};

// Local test doubles for `Lowercase` and `ByteLevel` normalizers to avoid a
// circular Bazel dependency between `tokenizer/` and `normalizers/`.
class LowercaseNormalizer : public Normalizer {
 public:
  absl::Status Normalize(NormalizedString& normalized) const override {
    normalized.Lowercase();
    return absl::OkStatus();
  }
};

class ByteLevelTestNormalizer : public Normalizer {
 public:
  absl::Status Normalize(NormalizedString& normalized) const override {
    if (!normalized.IsEmpty()) {
      const absl::flat_hash_map<uint8_t, char32_t>& bytes_char = BytesChar();
      absl::string_view s = normalized.Get();
      std::vector<std::pair<char32_t, int64_t>> transformations;
      transformations.reserve(s.size());
      for (const auto& [i, cur_char] : CharIndices(s)) {
        const size_t size = Utf8Len(cur_char);
        for (size_t k = 0; k < size; ++k) {
          transformations.emplace_back(
              bytes_char.at(static_cast<uint8_t>(s[i + k])), k > 0 ? 1 : 0);
        }
      }
      normalized.Transform(transformations, 0);
    }
    return absl::OkStatus();
  }

 private:
  static const absl::flat_hash_map<uint8_t, char32_t>& BytesChar() {
    static const absl::flat_hash_map<uint8_t, char32_t>* const kMap = [] {
      std::vector<uint8_t> bs;
      for (int b = '!'; b <= '~'; ++b) bs.push_back(static_cast<uint8_t>(b));
      for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(static_cast<uint8_t>(b));
      for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(static_cast<uint8_t>(b));
      std::vector<uint32_t> cs(bs.begin(), bs.end());
      uint32_t n = 0;
      for (int b = 0; b <= 255; ++b) {
        if (std::find(bs.begin(), bs.end(), static_cast<uint8_t>(b)) ==
            bs.end()) {
          bs.push_back(static_cast<uint8_t>(b));
          cs.push_back(256 + n);
          ++n;
        }
      }
      auto* m = new absl::flat_hash_map<uint8_t, char32_t>();
      for (size_t i = 0; i < bs.size(); ++i) {
        m->emplace(bs[i], static_cast<char32_t>(cs[i]));
      }
      return m;
    }();
    return *kMap;
  }
};

using SimplifiedOutput = std::vector<
    std::pair<absl::string_view, std::optional<std::vector<uint32_t>>>>;

SimplifiedOutput SimplifyOutput(const PreTokenizedString& result) {
  SimplifiedOutput out;
  for (const auto& [s, offsets, tokens] :
       result.GetSplits(OffsetReferential::kOriginal, OffsetType::kByte)) {
    (void)offsets;
    if (tokens->has_value()) {
      std::vector<uint32_t> ids;
      for (const Token& t : **tokens) {
        ids.push_back(t.id);
      }
      out.emplace_back(s, std::move(ids));
    } else {
      out.emplace_back(s, std::nullopt);
    }
  }
  return out;
}

TEST(AddedVocabularyTest, CanAddTokens) {
  ModelMock model({{"test", 0}, {"tost", 1}});
  AddedVocabulary vocab;
  const Normalizer* normalizer = nullptr;

  // Add tokens normally
  ASSERT_OK_AND_ASSIGN(
      size_t added, vocab.AddTokens({AddedToken::From("added_token_1", false)},
                                    model, normalizer));
  EXPECT_EQ(added, 1u);
  EXPECT_EQ(vocab.Len(), 1u);

  // Does not add multiple time the same token
  ASSERT_OK_AND_ASSIGN(
      added, vocab.AddTokens({AddedToken::From("added_token_2", false),
                              AddedToken::From("added_token_2", false)},
                             model, normalizer));
  EXPECT_EQ(added, 1u);
  EXPECT_EQ(vocab.Len(), 2u);

  // Also adds tokens already covered by the model
  AddedToken added_token = AddedToken::From("test", false);
  ASSERT_OK_AND_ASSIGN(added,
                       vocab.AddTokens({added_token}, model, normalizer));
  EXPECT_EQ(added, 1u);
  EXPECT_EQ(vocab.Len(), 3u);

  EXPECT_EQ(vocab.GetAddedTokensDecoder().at(0), added_token);
}

TEST(AddedVocabularyTest, CanAddSpecialTokens) {
  ModelMock model({{"test", 0}, {"tost", 1}});
  AddedVocabulary vocab;
  const Normalizer* normalizer = nullptr;

  // Add tokens normally
  ASSERT_OK_AND_ASSIGN(
      size_t added,
      vocab.AddSpecialTokens({AddedToken::From("added_token_1", true)}, model,
                             normalizer));
  EXPECT_EQ(added, 1u);
  EXPECT_EQ(vocab.Len(), 1u);

  // Does not add multiple time the same token
  ASSERT_OK_AND_ASSIGN(
      added, vocab.AddSpecialTokens({AddedToken::From("added_token_2", true),
                                     AddedToken::From("added_token_2", true)},
                                    model, normalizer));
  EXPECT_EQ(added, 1u);
  EXPECT_EQ(vocab.Len(), 2u);

  // Can add tokens already covered by the model
  ASSERT_OK_AND_ASSIGN(
      added, vocab.AddSpecialTokens({AddedToken::From("test", true)}, model,
                                    normalizer));
  EXPECT_EQ(added, 1u);
  EXPECT_EQ(vocab.Len(), 3u);
  EXPECT_TRUE(vocab.IsSpecialToken("test"));
  EXPECT_EQ(vocab.GetAddedTokensDecoder(),
            (absl::flat_hash_map<uint32_t, AddedToken>{
                {0, AddedToken::From("test", true)},
                {2, AddedToken::From("added_token_1", true)},
                {3, AddedToken::From("added_token_2", true)},
            }));
  EXPECT_TRUE(vocab.GetVocab().contains("test"));
  EXPECT_TRUE(vocab.GetAddedTokensDecoder().contains(0));

  ASSERT_OK(vocab.AddTokens(
      {AddedToken::From("tost", true), AddedToken::From("another_two", false)},
      model, normalizer));
  EXPECT_EQ(vocab.Len(), 5u);
  EXPECT_EQ(vocab.GetVocab().at("another_two"), 4u);

  // Let's add an already added token again, but change normalized
  ASSERT_OK_AND_ASSIGN(
      added, vocab.AddSpecialTokens({AddedToken::From("another_two", true)},
                                    model, normalizer));
  EXPECT_EQ(added, 1u);
  EXPECT_EQ(vocab.Len(), 5u);
  EXPECT_EQ(vocab.GetVocab().at("another_two"), 4u);

  AddedToken token = AddedToken::From("Hey", false);
  token.content = "hey";
  EXPECT_EQ(token.content, "hey");

  token.special = true;
  EXPECT_TRUE(token.special);
}

TEST(AddedVocabularyTest, CanExtractAddedTokens) {
  ModelMock model({});
  AddedVocabulary vocab;
  const Normalizer* normalizer = nullptr;

  ASSERT_OK(vocab.AddTokens(
      {AddedToken::From("my", false), AddedToken::From("name", false)}, model,
      normalizer));
  ASSERT_OK(vocab.AddSpecialTokens(
      {AddedToken::From("[CLS]", true), AddedToken::From("[SEP]", true)}, model,
      normalizer));

  ASSERT_OK_AND_ASSIGN(
      PreTokenizedString result,
      vocab.ExtractAndNormalize(normalizer, "[CLS] My name is Anthony [SEP]"));
  EXPECT_EQ(SimplifyOutput(result), (SimplifiedOutput{
                                        {"[CLS]", std::vector<uint32_t>{2}},
                                        {" My ", std::nullopt},
                                        {"name", std::vector<uint32_t>{1}},
                                        {" is Anthony ", std::nullopt},
                                        {"[SEP]", std::vector<uint32_t>{3}},
                                    }));
}

TEST(AddedVocabularyTest, OptionsUseCases) {
  ModelMock model({});
  LowercaseNormalizer normalizer;
  AddedVocabulary vocab;

  ASSERT_OK(vocab.AddTokens(
      {
          AddedToken::From("my", false).Lstrip(true).Rstrip(true),
          AddedToken::From("name", false),
          AddedToken::From("ony", false).SingleWord(true),
      },
      model, &normalizer));
  ASSERT_OK(vocab.AddSpecialTokens(
      {
          AddedToken::From("[CLS]", true),
          AddedToken::From("[SEP]", true),
      },
      model, &normalizer));

  ASSERT_OK_AND_ASSIGN(
      PreTokenizedString result,
      vocab.ExtractAndNormalize(&normalizer, "[CLS] My name is Anthony [SEP]"));

  EXPECT_EQ(SimplifyOutput(result), (SimplifiedOutput{
                                        {"[CLS]", std::vector<uint32_t>{3}},
                                        {" my ", std::vector<uint32_t>{0}},
                                        {"name", std::vector<uint32_t>{1}},
                                        {" is anthony ", std::nullopt},
                                        {"[SEP]", std::vector<uint32_t>{4}},
                                    }));
}

TEST(AddedVocabularyTest, EmptyMatches) {
  AddedVocabulary vocab;
  auto matches = vocab.FindMatches("", vocab.SplitTrie());
  EXPECT_EQ(matches, (std::vector<std::pair<std::optional<uint32_t>, Offsets>>{
                         {std::nullopt, {0, 0}}}));
}

TEST(AddedVocabularyTest, TestSingleWordIsCorrect) {
  ModelMock model({});
  AddedVocabulary vocab;
  LowercaseNormalizer normalizer;

  ASSERT_OK(
      vocab.AddTokens({AddedToken::From("<mask>", false).SingleWord(true)},
                      model, &normalizer));
  ASSERT_OK_AND_ASSIGN(
      PreTokenizedString result,
      vocab.ExtractAndNormalize(
          &normalizer, "<mask> My name <mask> A<mask> <mask>ony <mask>"));
  EXPECT_EQ(SimplifyOutput(result), (SimplifiedOutput{
                                        {"<mask>", std::vector<uint32_t>{0}},
                                        {" my name ", std::nullopt},
                                        {"<mask>", std::vector<uint32_t>{0}},
                                        {" a<mask> <mask>ony ", std::nullopt},
                                        {"<mask>", std::vector<uint32_t>{0}},
                                    }));
}

TEST(AddedVocabularyTest, TestSingleWordIsUnicodeCorrect) {
  ModelMock model({});
  AddedVocabulary vocab;
  LowercaseNormalizer normalizer;

  EXPECT_EQ(vocab.Len(), 0u);

  ASSERT_OK(
      vocab.AddTokens({AddedToken::From("<mask>", false).SingleWord(true)},
                      model, &normalizer));
  ASSERT_OK_AND_ASSIGN(
      PreTokenizedString result,
      vocab.ExtractAndNormalize(
          &normalizer, "<mask>, <mask>- ◌̰<mask> \u200c<mask> <mask>\u200d"));
  EXPECT_EQ(SimplifyOutput(result),
            (SimplifiedOutput{
                {"<mask>", std::vector<uint32_t>{0}},
                {", ", std::nullopt},
                {"<mask>", std::vector<uint32_t>{0}},
                {"- ◌̰<mask> \u200c<mask> <mask>\u200d", std::nullopt},
            }));
}

TEST(AddedVocabularyTest, TestLstripUnicodeSpace) {
  ModelMock model({});
  AddedVocabulary vocab;
  LowercaseNormalizer normalizer;

  ASSERT_OK(vocab.AddTokens({AddedToken::From("<mask>", false)
                                 .Lstrip(true)
                                 .Rstrip(true)
                                 .SingleWord(true)},
                            model, &normalizer));
  ASSERT_OK_AND_ASSIGN(
      PreTokenizedString result,
      vocab.ExtractAndNormalize(&normalizer,
                                "Hi <mask> there\t<mask>\t<mask>\u2000"));
  EXPECT_EQ(SimplifyOutput(result),
            (SimplifiedOutput{
                {"hi", std::nullopt},
                {" <mask> ", std::vector<uint32_t>{0}},
                {"there", std::nullopt},
                {"\t<mask>\t", std::vector<uint32_t>{0}},
                {"<mask>\u2000", std::vector<uint32_t>{0}},
            }));
}

TEST(AddedVocabularyTest, TestEncodeSpecialTokens) {
  ModelMock model({});
  AddedVocabulary vocab;
  LowercaseNormalizer normalizer;

  ASSERT_OK(vocab.AddTokens(
      {
          AddedToken::From("<mask>", true)
              .Lstrip(true)
              .Rstrip(true)
              .SingleWord(true),
          AddedToken::From("ask>", false),
          AddedToken::From("<pad>", true),
      },
      model, &normalizer));
  vocab.SetEncodeSpecialTokens(true);

  ASSERT_OK_AND_ASSIGN(
      PreTokenizedString result,
      vocab.ExtractAndNormalize(
          &normalizer,
          "Hi <mask> there\t<mask>\t<mask>\u2000 <pad> <mask><pad><pad>"));

  EXPECT_EQ(SimplifyOutput(result), (SimplifiedOutput{
                                        {"hi <m", std::nullopt},
                                        {"ask>", std::vector<uint32_t>{1}},
                                        {" there\t<m", std::nullopt},
                                        {"ask>", std::vector<uint32_t>{1}},
                                        {"\t<m", std::nullopt},
                                        {"ask>", std::vector<uint32_t>{1}},
                                        {"\u2000 <pad> <m", std::nullopt},
                                        {"ask>", std::vector<uint32_t>{1}},
                                        {"<pad><pad>", std::nullopt},
                                    }));

  vocab.SetEncodeSpecialTokens(false);

  ASSERT_OK_AND_ASSIGN(
      result,
      vocab.ExtractAndNormalize(
          &normalizer,
          "Hi <mask> there\t<mask>\t<mask>\u2000 <pad> <mask><pad><pad>"));
  EXPECT_EQ(SimplifyOutput(result),
            (SimplifiedOutput{
                {"hi", std::nullopt},
                {" <mask> ", std::vector<uint32_t>{0}},
                {"there", std::nullopt},
                {"\t<mask>\t", std::vector<uint32_t>{0}},
                {"<mask>\u2000 ", std::vector<uint32_t>{0}},
                {"<pad>", std::vector<uint32_t>{2}},
                {" <mask>", std::vector<uint32_t>{0}},
                {"<pad>", std::vector<uint32_t>{2}},
                {"<pad>", std::vector<uint32_t>{2}},
            }));
}

TEST(AddedVocabularyTest, ContentPreservedWithNormalizer) {
  ModelMock model({});
  AddedVocabulary vocab;
  LowercaseNormalizer normalizer;

  ASSERT_OK(vocab.AddTokens(
      {
          AddedToken::From("Hello", false),
          AddedToken::From("[CLS]", true),
      },
      model, &normalizer));

  const auto& decoder = vocab.GetAddedTokensDecoder();
  bool has_hello = false;
  bool has_cls = false;
  for (const auto& [_, t] : decoder) {
    if (t.content == "Hello") has_hello = true;
    if (t.content == "[CLS]") has_cls = true;
  }
  EXPECT_TRUE(has_hello);
  EXPECT_TRUE(has_cls);

  uint32_t hello_id = vocab.GetVocab().at("Hello");
  uint32_t cls_id = vocab.GetVocab().at("[CLS]");
  EXPECT_EQ(vocab.SimpleIdToToken(hello_id), "hello");
  EXPECT_EQ(vocab.SimpleIdToToken(cls_id), "[CLS]");
}

TEST(AddedVocabularyTest, RefreshNormalizedTokensOnNormalizerChange) {
  ModelMock model({});
  AddedVocabulary vocab;
  LowercaseNormalizer normalizer;

  ASSERT_OK(
      vocab.AddTokens({AddedToken::From("Hello", false)}, model, nullptr));

  uint32_t hello_id = vocab.GetVocab().at("Hello");
  EXPECT_EQ(vocab.SimpleIdToToken(hello_id), "Hello");

  ASSERT_OK(vocab.RefreshNormalizedTokens(&normalizer));

  EXPECT_EQ(vocab.SimpleIdToToken(hello_id), "hello");

  ASSERT_OK_AND_ASSIGN(PreTokenizedString result,
                       vocab.ExtractAndNormalize(&normalizer, "Hello world"));
  SimplifiedOutput splits = SimplifyOutput(result);
  ASSERT_FALSE(splits.empty());
  EXPECT_EQ(splits[0],
            (std::pair<absl::string_view, std::optional<std::vector<uint32_t>>>(
                "hello", std::vector<uint32_t>{0})));
}

TEST(AddedVocabularyTest, ByteLevelNormalizer) {
  ModelMock model({});
  AddedVocabulary vocab;
  ByteLevelTestNormalizer normalizer;

  ASSERT_OK(vocab.AddTokens(
      {AddedToken::From("my", false), AddedToken::From("今", false)}, model,
      &normalizer));
  ASSERT_OK_AND_ASSIGN(PreTokenizedString result,
                       vocab.ExtractAndNormalize(&normalizer, "my今"));
  EXPECT_EQ(SimplifyOutput(result), (SimplifiedOutput{
                                        {"my", std::vector<uint32_t>{0}},
                                        {"ä»Ĭ", std::vector<uint32_t>{1}},
                                    }));
}

}  // namespace
}  // namespace tokenizers
