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
#include "support/tokenizer/tokenizer.h"  // from @litert
#include "omni/asr/speech_decoder.h"
#include "omni/base/stage.h"

namespace litert_lm::omni::asr {
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
      const std::vector<int>& token_ids) override {
    if (token_ids_to_text_fn_) {
      return token_ids_to_text_fn_(token_ids);
    }
    return "";
  }

  std::vector<std::string> GetTokens() const override { return {}; }
  int GetVocabSize() const override { return 100; }

  std::function<absl::StatusOr<std::string>(const std::vector<int>&)>
      token_ids_to_text_fn_;
};

class DummySpeechDecoder : public SingleThreadedStageWithDeque<
                               std::vector<SpeechDecoder::DecodedToken>> {
 public:
  DummySpeechDecoder() = default;

  void PushTokens(std::vector<SpeechDecoder::DecodedToken> tokens) {
    PushOutput(std::move(tokens));
  }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TEST(TokenizerDetokenizerTest, ProcessEmptyTokensReturnsEmptyVector) {
  DummySpeechDecoder dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](const std::vector<int>& ids) {
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({});

  ABSL_ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ABSL_ASSERT_OK(words_or);
  EXPECT_TRUE(words_or->empty());
}

TEST(TokenizerDetokenizerTest, ProcessTokensDecodesWordsWithTimestamps) {
  DummySpeechDecoder dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](const std::vector<int>& ids) {
    if (ids == std::vector<int>{101, 102}) {
      return "hello world";
    }
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechDecoder::DecodedToken{.token_id = 101, .timestamp_ms = 100},
      SpeechDecoder::DecodedToken{.token_id = 102, .timestamp_ms = 200},
  });

  ABSL_ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ABSL_ASSERT_OK(words_or);
  auto words = std::move(*words_or);

  ASSERT_EQ(words.size(), 2);
  EXPECT_EQ(words[0].text, "hello");
  EXPECT_EQ(words[0].timestamp_ms, 100);
  EXPECT_EQ(words[1].text, "world");
  EXPECT_EQ(words[1].timestamp_ms, 200);
}

TEST(TokenizerDetokenizerTest,
     ProcessTokensInterpolatesTimestampsWhenCountsDiffer) {
  DummySpeechDecoder dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](const std::vector<int>& ids) {
    if (ids == std::vector<int>{1, 2, 3, 4, 5}) {
      return "one two three";
    }
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechDecoder::DecodedToken{.token_id = 1, .timestamp_ms = 100},
      SpeechDecoder::DecodedToken{.token_id = 2, .timestamp_ms = 200},
      SpeechDecoder::DecodedToken{.token_id = 3, .timestamp_ms = 300},
      SpeechDecoder::DecodedToken{.token_id = 4, .timestamp_ms = 400},
      SpeechDecoder::DecodedToken{.token_id = 5, .timestamp_ms = 500},
  });

  ABSL_ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ABSL_ASSERT_OK(words_or);
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
  DummySpeechDecoder dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](const std::vector<int>& ids) {
    if (ids == std::vector<int>{1, 2, 3}) {
      return "alpha beta gamma";
    }
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  // Token 0 at index 0 has no timestamp, Token 1 at index 1 has 200ms, Token 2
  // at index 2 has no timestamp.
  dummy_decoder.PushTokens({
      SpeechDecoder::DecodedToken{.token_id = 1, .timestamp_ms = std::nullopt},
      SpeechDecoder::DecodedToken{.token_id = 2, .timestamp_ms = 200},
      SpeechDecoder::DecodedToken{.token_id = 3, .timestamp_ms = std::nullopt},
  });

  ABSL_ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ABSL_ASSERT_OK(words_or);
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
  DummySpeechDecoder dummy_decoder;
  FakeTokenizer fake_tokenizer;
  fake_tokenizer.token_ids_to_text_fn_ = [](const std::vector<int>& ids) {
    return "test word";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechDecoder::DecodedToken{.token_id = 1, .timestamp_ms = 50},
  });

  ABSL_ASSERT_OK(detokenizer.Schedule());
  EXPECT_TRUE(detokenizer.HasOutput());

  detokenizer.Reset();
  EXPECT_FALSE(detokenizer.HasOutput());
}

}  // namespace
}  // namespace litert_lm::omni::asr
