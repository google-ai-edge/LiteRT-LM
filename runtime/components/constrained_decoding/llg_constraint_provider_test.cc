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

#include "runtime/components/constrained_decoding/llg_constraint_provider.h"

#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/sentencepiece_tokenizer.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;
using TokenIds = ::litert::support::TokenIds;

namespace {

using ::testing::ElementsAre;
using ::testing::Return;
using ::testing::status::StatusIs;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(absl::StatusOr<TokenIds>, TextToTokenIds, (absl::string_view),
              (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int>, bool), (override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

class LlgConstraintProviderTest : public ::testing::Test {
 protected:
  ::testing::NiceMock<MockTokenizer> tokenizer_;
};

TEST_F(LlgConstraintProviderTest, CreateSuccess) {
  LlGuidanceConfig config;
  config.eos_ids = {1};

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto provider = LlgConstraintProvider::Create(tokenizer_, config);
  EXPECT_TRUE(provider.ok());
}

TEST_F(LlgConstraintProviderTest, CreateFailsWithoutEosId) {
  LlGuidanceConfig config;
  auto provider = LlgConstraintProvider::Create(tokenizer_, config);
  EXPECT_FALSE(provider.ok());
  EXPECT_EQ(provider.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LlgConstraintProviderTest, CreateConstraintProviderSuccess) {
  LlGuidanceConfig config;
  config.eos_ids = {1};

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto constraint_provider = LlgConstraintProvider::Create(tokenizer_, config);
  ASSERT_TRUE(constraint_provider.ok());
}

TEST_F(LlgConstraintProviderTest, CreateConstraintSuccess) {
  LlGuidanceConfig config;
  config.eos_ids = {1};

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto constraint_provider = LlgConstraintProvider::Create(tokenizer_, config);
  ASSERT_TRUE(constraint_provider.ok());
  auto provider = std::move(constraint_provider.value());

  auto constraint = provider->CreateConstraint(LlGuidanceConstraintArg{
      .constraint_type = LlgConstraintType::kRegex, .constraint_string = "a+"});
  EXPECT_TRUE(constraint.ok());
}

TEST_F(LlgConstraintProviderTest, CreateFailsWithOutOfRangeEosId) {
  LlGuidanceConfig config;
  config.eos_ids = {1, 4};

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  EXPECT_THAT(LlgConstraintProvider::Create(tokenizer_, config),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AddStopTokensAsEosIdsTest, AddsSingleTokenStopSequencesOnce) {
  LlGuidanceConfig config;
  AddStopTokensAsEosIds({{106}, {1, 2}, {50}, {106}, {-1}, {1}}, config);
  EXPECT_THAT(config.eos_ids, ElementsAre(106, 50, 1));
}

TEST(AddStopTokensAsEosIdsTest, KeepsConfiguredEosIdsFirst) {
  LlGuidanceConfig config;
  config.eos_ids = {1};
  AddStopTokensAsEosIds({{1}, {106}, {50}}, config);
  EXPECT_THAT(config.eos_ids, ElementsAre(1, 106, 50));
}

// Returns whether `mask` (an LLGuidance mask, which is always a bitmap) allows
// `token_id`.
bool IsAllowed(const LogitMask& mask, int token_id) {
  EXPECT_EQ(mask.GetType(), MaskType::kBitmap);
  return static_cast<const BitmapLogitMask&>(mask).IsAllowed(token_id);
}

// Feeds `tokens` to a new state of `constraint` one at a time. Each token must
// be allowed by the mask before it, and `eos_id` must be allowed at the end.
void ExpectConstraintAccepts(const Constraint& constraint,
                             absl::Span<const int> tokens, int eos_id) {
  std::unique_ptr<Constraint::State> state = constraint.Start();
  for (size_t i = 0; i < tokens.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask,
                         constraint.ComputeMask(*state));
    ASSERT_TRUE(IsAllowed(*mask, tokens[i]))
        << "token " << tokens[i] << " at step " << i;
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, tokens[i]));
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask,
                       constraint.ComputeMask(*state));
  EXPECT_TRUE(IsAllowed(*mask, eos_id));
}

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";
// A SentencePiece vocabulary stores a space as "▁" (U+2581) and a byte that
// has no piece of its own as a byte-fallback piece "<0xXX>".
constexpr char kSentencePieceSpace[] = "\xe2\x96\x81";
enum SentencePieceVocabId : int {
  kPad = 0,
  kEos = 1,
  kA = 2,
  kB = 3,
  kSpace = 4,         // "▁"
  kSpaceB = 5,        // "▁b"
  kNewlineByte = 6,   // "<0x0A>"
  kSpaceByte = 7,     // "<0x20>"
  kMalformedHex = 8,  // "<0xZZ>", not a byte-fallback piece.
};

std::vector<std::string> SentencePieceVocab() {
  return {"<pad>",
          "<eos>",
          "a",
          "b",
          kSentencePieceSpace,
          absl::StrCat(kSentencePieceSpace, "b"),
          "<0x0A>",
          "<0x20>",
          "<0xZZ>"};
}

// Tokenizes like SentencePiece: spaces become "▁", the longest matching text
// piece wins, and '\n' (which has no text piece) becomes its byte piece.
absl::StatusOr<TokenIds> SentencePieceEncode(absl::string_view text) {
  const std::vector<std::string> vocab = SentencePieceVocab();
  const std::string escaped =
      absl::StrReplaceAll(text, {{" ", kSentencePieceSpace}});
  absl::string_view rest = escaped;
  TokenIds ids;
  while (!rest.empty()) {
    int best_id = -1;
    for (int id : {kA, kB, kSpace, kSpaceB, kMalformedHex}) {
      if (absl::StartsWith(rest, vocab[id]) &&
          (best_id < 0 || vocab[id].size() > vocab[best_id].size())) {
        best_id = id;
      }
    }
    if (best_id >= 0) {
      ids.push_back(best_id);
      rest.remove_prefix(vocab[best_id].size());
    } else if (rest[0] == '\n') {
      ids.push_back(kNewlineByte);
      rest.remove_prefix(1);
    } else {
      return absl::InvalidArgumentError(absl::StrCat("Can't encode: ", rest));
    }
  }
  return ids;
}

class LlgConstraintProviderSentencePieceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ON_CALL(tokenizer_, GetTokens())
        .WillByDefault(Return(SentencePieceVocab()));
    ON_CALL(tokenizer_, TextToTokenIds(::testing::_))
        .WillByDefault(SentencePieceEncode);
  }

  absl::StatusOr<std::unique_ptr<Constraint>> CreateRegexConstraint(
      absl::string_view regex) {
    LlGuidanceConfig config;
    config.eos_ids = {kEos};
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ConstraintProvider> provider,
                          LlgConstraintProvider::Create(tokenizer_, config));
    provider_ = std::move(provider);
    return provider_->CreateConstraint(
        LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kRegex,
                                .constraint_string = std::string(regex)});
  }

  ::testing::NiceMock<MockTokenizer> tokenizer_;
  std::unique_ptr<ConstraintProvider> provider_;
};

// LLGuidance forces "a b\n" and tokenizes it with the tokenizer into "a", "▁b",
// "<0x0A>". If those pieces are passed as raw strings, their lengths (4 and 6
// bytes) don't match the bytes they cover (2 and 1), and LLGuidance panics.
TEST_F(LlgConstraintProviderSentencePieceTest,
       ForcedTextWithSpaceAndByteFallbackPieces) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       CreateRegexConstraint("a b\n"));
  ExpectConstraintAccepts(*constraint, {kA, kSpaceB, kNewlineByte}, kEos);
}

TEST_F(LlgConstraintProviderSentencePieceTest, SpacePiecesMatchSpaces) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       CreateRegexConstraint("a( b)?"));
  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK(constraint->ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, kA));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask,
                       constraint->ComputeMask(*state));
  EXPECT_TRUE(IsAllowed(*mask, kSpace));
  EXPECT_TRUE(IsAllowed(*mask, kSpaceB));
  EXPECT_TRUE(IsAllowed(*mask, kSpaceByte));
  EXPECT_TRUE(IsAllowed(*mask, kEos));
  EXPECT_FALSE(IsAllowed(*mask, kB));
  EXPECT_FALSE(IsAllowed(*mask, kNewlineByte));
}

// A tokenizer that adds a leading "▁" to every text (SentencePiece's
// add_dummy_prefix) returns "▁", "a", "▁b" for "a b", which covers " a b".
// LLGuidance must not force those tokens; any tokens spelling "a b" are fine.
TEST_F(LlgConstraintProviderSentencePieceTest,
       DummyPrefixTokenizerDoesNotForceTokens) {
  ON_CALL(tokenizer_, TextToTokenIds(::testing::_))
      .WillByDefault([](absl::string_view text) {
        return SentencePieceEncode(absl::StrCat(" ", text));
      });
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       CreateRegexConstraint("a b"));
  ExpectConstraintAccepts(*constraint, {kA, kSpaceB}, kEos);
}

TEST_F(LlgConstraintProviderSentencePieceTest, ByteFallbackPieceIsItsByte) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       CreateRegexConstraint("a\n?"));
  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK(constraint->ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, kA));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask,
                       constraint->ComputeMask(*state));
  EXPECT_TRUE(IsAllowed(*mask, kNewlineByte));
  EXPECT_FALSE(IsAllowed(*mask, kSpaceByte));
}

TEST_F(LlgConstraintProviderSentencePieceTest,
       MalformedByteFallbackPieceIsText) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       CreateRegexConstraint("<0xZZ>"));
  ExpectConstraintAccepts(*constraint, {kMalformedHex}, kEos);
}

// A special token keeps its name as its bytes, so it never matches as text.
TEST_F(LlgConstraintProviderSentencePieceTest, SpecialTokensAreNotDecoded) {
  LlGuidanceConfig config;
  config.eos_ids = {kEos};
  config.special_tokens = {"<0x0A>"};
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       LlgConstraintProvider::Create(tokenizer_, config));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a\n?"}));
  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK(constraint->ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, kA));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask,
                       constraint->ComputeMask(*state));
  EXPECT_FALSE(IsAllowed(*mask, kNewlineByte));
}

// The same, with the real Gemma 3 vocabulary and tokenizer: spaces, a newline,
// digits, and "￦" (U+FFE6), which has no piece and uses 3 byte-fallback
// pieces.
TEST(LlgConstraintProviderGemma3Test, FollowsRegexWithSpacesAndBytePieces) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Tokenizer> tokenizer,
      ::litert::support::SentencePieceTokenizer::CreateFromFile(
          (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
           "gemma3_sentencepiece.model")
              .string()));
  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       tokenizer->TextToTokenIds("Price: \xef\xbf\xa6"
                                                 "2000 won\n"));
  ASSERT_OK_AND_ASSIGN(int byte_ef_id, tokenizer->TokenToId("<0xEF>"));
  ASSERT_THAT(tokens, ::testing::Contains(byte_ef_id));
  LlGuidanceConfig config;
  config.eos_ids = {1};
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       LlgConstraintProvider::Create(*tokenizer, config));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Constraint> constraint,
      provider->CreateConstraint(LlGuidanceConstraintArg{
          .constraint_type = LlgConstraintType::kRegex,
          .constraint_string = "Price: \xef\xbf\xa6[0-9]+ won\n"}));
  ExpectConstraintAccepts(*constraint, tokens, /*eos_id=*/1);
}

// The same with a real SentencePiece model that has add_dummy_prefix,
// remove_extra_whitespaces and NFKC normalization.
TEST(LlgConstraintProviderDummyPrefixTest, FollowsRegex) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Tokenizer> tokenizer,
      ::litert::support::SentencePieceTokenizer::CreateFromFile(
          (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
           "sentencepiece.model")
              .string()));
  // Encode("a b") is "▁a", "▁b", so "a" and "▁b" are the tokens for "a b".
  ASSERT_OK_AND_ASSIGN(int a_id, tokenizer->TokenToId("a"));
  ASSERT_OK_AND_ASSIGN(int space_b_id, tokenizer->TokenToId(absl::StrCat(
                                           kSentencePieceSpace, "b")));
  LlGuidanceConfig config;
  ASSERT_OK_AND_ASSIGN(config.eos_ids.emplace_back(),
                       tokenizer->TokenToId("</s>"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       LlgConstraintProvider::Create(*tokenizer, config));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a b"}));
  ExpectConstraintAccepts(*constraint, {a_id, space_b_id}, config.eos_ids[0]);
}

}  // namespace
}  // namespace litert::lm
