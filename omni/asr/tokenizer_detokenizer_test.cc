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

#include "omni/asr/tokenizer_detokenizer.h"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/speech_recognizer.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

class FakeTokenizer : public ::litert::support::Tokenizer {
 public:
  ::litert::support::TokenizerType GetTokenizerType() const override {
    return ::litert::support::TokenizerType::kUnspecified;
  }

  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    return std::vector<int>{};
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override { return 0; }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool skip_special_tokens) override {
    if (token_ids_to_text_fn_) {
      return token_ids_to_text_fn_(token_ids);
    }
    return "";
  }

  std::vector<std::string> GetTokens() const override { return {}; }
  int GetVocabSize() const override { return 100; }

  std::function<absl::StatusOr<std::string>(absl::Span<const int>)>
      token_ids_to_text_fn_;
};

class DummySpeechRecognizer : public SingleThreadedStageWithDeque<
                                  std::vector<SpeechRecognizer::DecodedToken>> {
 public:
  DummySpeechRecognizer() = default;

  void PushTokens(std::vector<SpeechRecognizer::DecodedToken> tokens) {
    PushOutput(std::move(tokens));
  }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TEST(TokenizerDetokenizerTest, ProcessEmptyTokensReturnsEmptyVector) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](absl::Span<const int> ids) {
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({});

  ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ASSERT_OK(words_or);
  EXPECT_TRUE(words_or->empty());
}

TEST(TokenizerDetokenizerTest, ProcessTokensDecodesWordsWithTimestamps) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](absl::Span<const int> ids) {
    if (ids == absl::Span<const int>({101, 102})) {
      return "hello world";
    }
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 101, .timestamp_ms = 100},
      SpeechRecognizer::DecodedToken{.token_id = 102, .timestamp_ms = 200},
  });

  ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ASSERT_OK(words_or);
  auto words = std::move(*words_or);

  ASSERT_EQ(words.size(), 2);
  EXPECT_EQ(words[0].text, "hello");
  EXPECT_EQ(words[0].timestamp_ms, 100);
  EXPECT_EQ(words[1].text, "world");
  EXPECT_EQ(words[1].timestamp_ms, 200);
}

TEST(TokenizerDetokenizerTest,
     ProcessTokensInterpolatesTimestampsWhenCountsDiffer) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](absl::Span<const int> ids) {
    if (ids == absl::Span<const int>({1, 2, 3, 4, 5})) {
      return "one two three";
    }
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 100},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 200},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 300},
      SpeechRecognizer::DecodedToken{.token_id = 4, .timestamp_ms = 400},
      SpeechRecognizer::DecodedToken{.token_id = 5, .timestamp_ms = 500},
  });

  ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ASSERT_OK(words_or);
  auto words = std::move(*words_or);

  ASSERT_EQ(words.size(), 3);
  EXPECT_EQ(words[0].text, "one");
  EXPECT_EQ(words[0].timestamp_ms, 100);  // Token 0
  EXPECT_EQ(words[1].text, "two");
  EXPECT_EQ(words[1].timestamp_ms, 300);  // Token 2
  EXPECT_EQ(words[2].text, "three");
  EXPECT_EQ(words[2].timestamp_ms, 500);  // Token 4
}

TEST(TokenizerDetokenizerTest,
     ProcessTokensPicksClosestTimestampWhenTokenLacksTimestamp) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](absl::Span<const int> ids) {
    if (ids == absl::Span<const int>({1, 2, 3})) {
      return "alpha beta gamma";
    }
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  // Token 0 at index 0 has no timestamp, Token 1 at index 1 has 200ms, Token 2
  // at index 2 has no timestamp.
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 1,
                                     .timestamp_ms = std::nullopt},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 200},
      SpeechRecognizer::DecodedToken{.token_id = 3,
                                     .timestamp_ms = std::nullopt},
  });

  ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ASSERT_OK(words_or);
  auto words = std::move(*words_or);

  ASSERT_EQ(words.size(), 3);
  EXPECT_EQ(words[0].text, "alpha");
  EXPECT_EQ(words[0].timestamp_ms, 200);  // Closest is token 1 (200ms)
  EXPECT_EQ(words[1].text, "beta");
  EXPECT_EQ(words[1].timestamp_ms, 200);  // Token 1 itself
  EXPECT_EQ(words[2].text, "gamma");
  EXPECT_EQ(words[2].timestamp_ms, 200);  // Closest is token 1 (200ms)
}

TEST(TokenizerDetokenizerTest, ResetClearsOutputs) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](absl::Span<const int> ids) {
    return "test word";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 50},
  });

  ASSERT_OK(detokenizer.Schedule());
  EXPECT_TRUE(detokenizer.HasOutput());

  detokenizer.Reset();
  EXPECT_FALSE(detokenizer.HasOutput());
}

TEST(TokenizerDetokenizerTest,
     ProcessTokensWithEndOfChunkTokenAppendsEmptyWordWithTimestamp) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](absl::Span<const int> ids) {
    if (ids == absl::Span<const int>({101, 102})) {
      return "hello world";
    }
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 101, .timestamp_ms = 100},
      SpeechRecognizer::DecodedToken{.token_id = 102, .timestamp_ms = 200},
      SpeechRecognizer::DecodedToken{
          .token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
          .timestamp_ms = 5000},
  });

  ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ASSERT_OK(words_or);
  auto words = std::move(*words_or);

  ASSERT_EQ(words.size(), 3);
  EXPECT_EQ(words[0].text, "hello");
  EXPECT_EQ(words[0].timestamp_ms, 100);
  EXPECT_FALSE(words[0].IsEndOfChunk());

  EXPECT_EQ(words[1].text, "world");
  EXPECT_EQ(words[1].timestamp_ms, 200);
  EXPECT_FALSE(words[1].IsEndOfChunk());

  EXPECT_EQ(words[2].text, "");
  EXPECT_EQ(words[2].timestamp_ms, 5000);
  EXPECT_TRUE(words[2].IsEndOfChunk());
}

}  // namespace
}  // namespace litert::omni::asr
