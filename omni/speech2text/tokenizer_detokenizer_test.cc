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

#include "omni/speech2text/tokenizer_detokenizer.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/speech2text/speech_recognizer.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

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

  // Single-word multi-token chunk also falls back to token 0 (onset) when
  // per-piece decoding yields empty strings.
  fake_tokenizer.token_ids_to_text_fn_ = [](absl::Span<const int> ids) {
    if (ids == absl::Span<const int>({1, 2, 3, 4, 5})) {
      return "unfortunately";
    }
    return "";
  };
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 100},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 200},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 300},
      SpeechRecognizer::DecodedToken{.token_id = 4, .timestamp_ms = 400},
      SpeechRecognizer::DecodedToken{.token_id = 5, .timestamp_ms = 500},
  });
  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto single_word, detokenizer.GetOutput());
  ASSERT_EQ(single_word.size(), 1);
  EXPECT_EQ(single_word[0].text, "unfortunately");
  EXPECT_EQ(single_word[0].timestamp_ms, 100);  // Token 0 (not Token 4)
}

TEST(TokenizerDetokenizerTest,
     ProcessTokensMapsMultiTokenSubwordsToWordOnsetTimestamps) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  const std::vector<std::string> pieces = {"Th",  "ere", "'s", " his",
                                           " be", "ak",  ".",  " But"};
  fake_tokenizer.token_ids_to_text_fn_ =
      [&pieces](absl::Span<const int> ids) -> absl::StatusOr<std::string> {
    std::string out;
    for (int id : ids) {
      if (id >= 0 && id < static_cast<int>(pieces.size())) {
        out += pieces[id];
      }
    }
    return out;
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 0, .timestamp_ms = 11},
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 13},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 14},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 14},
      SpeechRecognizer::DecodedToken{.token_id = 4, .timestamp_ms = 17},
      SpeechRecognizer::DecodedToken{.token_id = 5, .timestamp_ms = 20},
      SpeechRecognizer::DecodedToken{.token_id = 6, .timestamp_ms = 38},
      SpeechRecognizer::DecodedToken{.token_id = 7, .timestamp_ms = 42},
  });

  ASSERT_OK(detokenizer.Schedule());
  auto words_or = detokenizer.GetOutput();
  ASSERT_OK(words_or);
  auto words = std::move(*words_or);

  ASSERT_EQ(words.size(), 4);
  EXPECT_EQ(words[0].text, "There's");
  EXPECT_EQ(words[0].timestamp_ms, 11);  // Token 0 ("Th")
  EXPECT_EQ(words[1].text, "his");
  EXPECT_EQ(words[1].timestamp_ms, 14);  // Token 3 (" his")
  EXPECT_EQ(words[2].text, "beak.");
  EXPECT_EQ(words[2].timestamp_ms, 17);  // Token 4 (" be")
  EXPECT_EQ(words[3].text, "But");
  EXPECT_EQ(words[3].timestamp_ms, 42);  // Token 7 (" But")
}

TEST(TokenizerDetokenizerTest,
     CachesDecodedTokenPiecesAndPreservesLeadingSpaceBoundaries) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  const std::vector<std::string> pieces = {" There", "'s", " his", " be", "ak"};
  int call_count = 0;
  int piece_decode_calls = 0;
  size_t max_piece_decode_len = 0;
  fake_tokenizer.token_ids_to_text_fn_ =
      [&pieces, &call_count, &piece_decode_calls, &max_piece_decode_len](
          absl::Span<const int> ids) -> absl::StatusOr<std::string> {
    ++call_count;
    // Full-chunk decodes pass all 5 tokens; per-piece lookups pass <= 3 tokens.
    if (ids.size() < 5) {
      ++piece_decode_calls;
      max_piece_decode_len = std::max(max_piece_decode_len, ids.size());
    }
    std::string out;
    for (int id : ids) {
      if (id >= 0 && id < static_cast<int>(pieces.size())) {
        out += pieces[id];
      }
    }
    // Simulate SentencePiece / HuggingFace Metaspace stripping the leading
    // space at the start of a decoded sequence.
    if (!out.empty() && out.front() == ' ') {
      out.erase(0, 1);
    }
    return out;
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 0, .timestamp_ms = 10},
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 12},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 15},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 20},
      SpeechRecognizer::DecodedToken{.token_id = 4, .timestamp_ms = 25},
  });

  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto words1, detokenizer.GetOutput());
  // 4 distinct token IDs (0..3) are queried by MapWordsToOnsetTokenIndices:
  // <= 2 decode calls per distinct ID, each with <= 3 tokens.
  EXPECT_LE(piece_decode_calls, 4 * 2);
  EXPECT_LE(max_piece_decode_len, 3u);
  ASSERT_EQ(words1.size(), 3);
  EXPECT_EQ(words1[0].text, "There's");
  EXPECT_EQ(words1[0].timestamp_ms, 10);
  EXPECT_EQ(words1[1].text, "his");
  EXPECT_EQ(words1[1].timestamp_ms, 15);
  EXPECT_EQ(words1[2].text, "beak");
  EXPECT_EQ(words1[2].timestamp_ms, 20);

  // Second chunk reuses already-cached token pieces: 0 piece decode calls and
  // only 1 full-sequence TokenIdsToText call.
  const int calls_after_first_chunk = call_count;
  const int piece_calls_after_first_chunk = piece_decode_calls;
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 0, .timestamp_ms = 110},
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 112},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 115},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 120},
      SpeechRecognizer::DecodedToken{.token_id = 4, .timestamp_ms = 125},
  });

  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto words2, detokenizer.GetOutput());
  EXPECT_EQ(piece_decode_calls - piece_calls_after_first_chunk, 0);
  EXPECT_EQ(call_count - calls_after_first_chunk, 1);
  ASSERT_EQ(words2.size(), 3);
  EXPECT_EQ(words2[0].text, "There's");
  EXPECT_EQ(words2[0].timestamp_ms, 110);
  EXPECT_EQ(words2[1].text, "his");
  EXPECT_EQ(words2[1].timestamp_ms, 115);
  EXPECT_EQ(words2[2].text, "beak");
  EXPECT_EQ(words2[2].timestamp_ms, 120);
}

TEST(TokenizerDetokenizerTest,
     ByteMergeTokensPreserveExactWordOnsetTimestampsAcrossChunks) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  // Simulate byte-level BPE (e.g. Whisper) where token 0 is a leading space +
  // UTF-8 lead byte (\xce) and token 1 is a continuation byte (\xa9). Any
  // unpaired byte renders as U+FFFD (\xef\xbf\xbd), whereas the consecutive
  // pair {0, 1} merges into " Ω" ("\xce\xa9").
  fake_tokenizer.token_ids_to_text_fn_ =
      [](absl::Span<const int> ids) -> absl::StatusOr<std::string> {
    std::string out;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] == 0 && i + 1 < ids.size() && ids[i + 1] == 1) {
        out += " \xce\xa9";
        ++i;
      } else if (ids[i] == 0) {
        out += " \xef\xbf\xbd";
      } else if (ids[i] == 1) {
        out += "\xef\xbf\xbd";
      } else if (ids[i] == 2) {
        out += " te";
      } else if (ids[i] == 3) {
        out += "st";
      } else if (ids[i] == 4) {
        out += " ca";
      } else if (ids[i] == 5) {
        out += "fe";
      } else if (ids[i] == 6) {
        out += " now";
      }
    }
    if (!out.empty() && out.front() == ' ') {
      out.erase(0, 1);
    }
    return out;
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  // First chunk starts with the byte-merge pair {0, 1} ("Ω") followed by
  // multi-token subword {2, 3} ("test") and single-token word {6} ("now").
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 0, .timestamp_ms = 10},
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 12},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 30},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 35},
      SpeechRecognizer::DecodedToken{.token_id = 6, .timestamp_ms = 70},
  });
  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto words1, detokenizer.GetOutput());
  ASSERT_EQ(words1.size(), 3);
  EXPECT_EQ(words1[0].text, "\xce\xa9");
  EXPECT_EQ(words1[0].timestamp_ms, 10);  // Token 0
  EXPECT_EQ(words1[1].text, "test");
  EXPECT_EQ(words1[1].timestamp_ms, 30);  // Token 2 (" te")
  EXPECT_EQ(words1[2].text, "now");
  EXPECT_EQ(words1[2].timestamp_ms, 70);  // Token 6 (" now")

  // Second chunk contains the byte-merge pair {0, 1} in the middle along with
  // multi-token words {2, 3} ("test") and {4, 5} ("cafe") and {6} ("now").
  // Linear interpolation over 7 tokens / 4 words would pick indices 0, 2, 4, 6
  // (giving timestamps 100, 120, 140, 170), whereas exact word-onset mapping
  // picks indices 0, 2, 4, 6 with token 2 = 120, token 4 = 140, token 6 = 170;
  // add a 3-token word {2, 3, 5} so interpolation (indices 0, 3, 5, 7) diverges
  // from exact onsets (indices 0, 3, 5, 7 vs 0, 1, 3, 6).
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 6, .timestamp_ms = 100},
      SpeechRecognizer::DecodedToken{.token_id = 0, .timestamp_ms = 110},
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 115},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 130},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 135},
      SpeechRecognizer::DecodedToken{.token_id = 5, .timestamp_ms = 140},
      SpeechRecognizer::DecodedToken{.token_id = 4, .timestamp_ms = 160},
      SpeechRecognizer::DecodedToken{.token_id = 5, .timestamp_ms = 165},
  });
  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto words2, detokenizer.GetOutput());
  ASSERT_EQ(words2.size(), 4);
  EXPECT_EQ(words2[0].text, "now");
  EXPECT_EQ(words2[0].timestamp_ms, 100);  // Token index 0 (id 6)
  EXPECT_EQ(words2[1].text, "\xce\xa9");
  EXPECT_EQ(words2[1].timestamp_ms, 110);  // Token index 1 (id 0, not 2!)
  EXPECT_EQ(words2[2].text, "testfe");
  EXPECT_EQ(words2[2].timestamp_ms, 130);  // Token index 3 (id 2, not 5!)
  EXPECT_EQ(words2[3].text, "cafe");
  EXPECT_EQ(words2[3].timestamp_ms, 160);  // Token index 6 (id 4, not 7!)
}

TEST(TokenizerDetokenizerTest,
     LeadingControlTokenPreservesExactWordOnsetTimestampsAcrossChunks) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  // Simulate SentencePiece's single-token control token quirk: token 0 renders
  // as "<bos>" when decoded alone (ids.size() == 1), but as "" inside any
  // multi-token sequence (ids.size() > 1).
  fake_tokenizer.token_ids_to_text_fn_ =
      [](absl::Span<const int> ids) -> absl::StatusOr<std::string> {
    if (ids.size() == 1 && ids[0] == 0) {
      return "<bos>";
    }
    const std::vector<std::string> pieces = {"",   " Th", "ere",
                                             "'s", " a",  " way"};
    std::string out;
    for (int id : ids) {
      if (id > 0 && id < static_cast<int>(pieces.size())) {
        out += pieces[id];
      }
    }
    if (!out.empty() && out.front() == ' ') {
      out.erase(0, 1);
    }
    return out;
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  // First chunk starts with control token 0 ("<bos>").
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 0, .timestamp_ms = 5},
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 10},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 15},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 20},
      SpeechRecognizer::DecodedToken{.token_id = 4, .timestamp_ms = 40},
      SpeechRecognizer::DecodedToken{.token_id = 5, .timestamp_ms = 60},
  });
  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto words1, detokenizer.GetOutput());
  ASSERT_EQ(words1.size(), 3);
  EXPECT_EQ(words1[0].text, "There's");
  EXPECT_EQ(words1[0].timestamp_ms,
            10);  // Token index 1 (" Th"), skipping <bos>
  EXPECT_EQ(words1[1].text, "a");
  EXPECT_EQ(words1[1].timestamp_ms, 40);  // Token index 4 (" a")
  EXPECT_EQ(words1[2].text, "way");
  EXPECT_EQ(words1[2].timestamp_ms, 60);  // Token index 5 (" way")

  // Following chunk keeps exact word-onset timestamps.
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 1, .timestamp_ms = 110},
      SpeechRecognizer::DecodedToken{.token_id = 2, .timestamp_ms = 115},
      SpeechRecognizer::DecodedToken{.token_id = 3, .timestamp_ms = 120},
      SpeechRecognizer::DecodedToken{.token_id = 4, .timestamp_ms = 140},
      SpeechRecognizer::DecodedToken{.token_id = 5, .timestamp_ms = 160},
  });
  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto words2, detokenizer.GetOutput());
  ASSERT_EQ(words2.size(), 3);
  EXPECT_EQ(words2[0].text, "There's");
  EXPECT_EQ(words2[0].timestamp_ms, 110);  // Token index 0 (" Th")
  EXPECT_EQ(words2[1].text, "a");
  EXPECT_EQ(words2[1].timestamp_ms, 140);  // Token index 3 (" a")
  EXPECT_EQ(words2[2].text, "way");
  EXPECT_EQ(words2[2].timestamp_ms, 160);  // Token index 4 (" way")
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

TEST(TokenizerDetokenizerTest,
     ProcessUntimestampedTokensSkipsPrefixDetokenization) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  int call_count = 0;
  fake_tokenizer.token_ids_to_text_fn_ =
      [&call_count](absl::Span<const int> ids) -> absl::StatusOr<std::string> {
    ++call_count;
    if (ids == absl::Span<const int>({1, 2, 3})) {
      return "hello world";
    }
    return "";
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      SpeechRecognizer::DecodedToken{.token_id = 1,
                                     .timestamp_ms = std::nullopt},
      SpeechRecognizer::DecodedToken{.token_id = 2,
                                     .timestamp_ms = std::nullopt},
      SpeechRecognizer::DecodedToken{.token_id = 3,
                                     .timestamp_ms = std::nullopt},
  });

  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto words, detokenizer.GetOutput());
  EXPECT_EQ(call_count, 1);
  ASSERT_EQ(words.size(), 2);
  EXPECT_EQ(words[0].text, "hello");
  EXPECT_EQ(words[0].timestamp_ms, std::nullopt);
  EXPECT_EQ(words[1].text, "world");
  EXPECT_EQ(words[1].timestamp_ms, std::nullopt);
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

TEST(TokenizerDetokenizerTest, CjkCharactersAreWordsWithTheirTokenTimestamps) {
  DummySpeechRecognizer dummy_decoder;
  FakeTokenizer fake_tokenizer;
  // Token 5 and 6 split "中" (E4 B8 AD) across two tokens, as byte-level BPE
  // tokenizers do.
  const absl::flat_hash_map<int, std::string> pieces = {
      {1, "你好"}, {2, "吗？"},     {3, " OK"},
      {4, "世界"}, {5, "\xE4\xB8"}, {6, "\xAD文"}};
  fake_tokenizer.token_ids_to_text_fn_ =
      [&pieces](absl::Span<const int> ids) -> absl::StatusOr<std::string> {
    std::string text;
    for (int id : ids) text += pieces.at(id);
    return text;
  };

  TokenizerDetokenizer detokenizer(&dummy_decoder, &fake_tokenizer);
  dummy_decoder.PushTokens({
      {.token_id = 1, .timestamp_ms = 100},
      {.token_id = 2, .timestamp_ms = 200},
      {.token_id = 3, .timestamp_ms = 300},
      {.token_id = 4, .timestamp_ms = 400},
      {.token_id = 5, .timestamp_ms = 500},
      {.token_id = 6, .timestamp_ms = 600},
  });

  ASSERT_OK(detokenizer.Schedule());
  ASSERT_OK_AND_ASSIGN(auto words, detokenizer.GetOutput());

  std::vector<std::pair<std::string, int>> actual;
  for (const auto& word : words) {
    actual.emplace_back(word.text, word.timestamp_ms.value_or(-1));
  }
  EXPECT_THAT(actual,
              ElementsAre(Pair("你", 100), Pair("好", 100), Pair("吗？", 200),
                          Pair("OK", 300), Pair("世", 400), Pair("界", 400),
                          Pair("中", 500), Pair("文", 600)));
}

}  // namespace
}  // namespace litert::omni::asr
