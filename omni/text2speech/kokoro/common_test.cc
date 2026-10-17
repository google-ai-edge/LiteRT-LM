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

#include "omni/text2speech/kokoro/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "omni/text2speech/kokoro/kokoro_io_types.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

// ModelResources stub serving a single named GenericBinaryData section.
class FakeModelResources : public lm::ModelResources {
 public:
  FakeModelResources(std::string section_name, std::string section_data)
      : section_name_(std::move(section_name)),
        section_data_(std::move(section_data)) {}

  absl::StatusOr<absl::string_view> GetGenericBinaryDataBuffer(
      absl::string_view name) override {
    if (name != section_name_) {
      return absl::NotFoundError(
          absl::StrCat("No GenericBinaryData section named ", name));
    }
    return absl::string_view(section_data_);
  }

  std::vector<std::string> GetGenericBinaryDataNames() const override {
    return {section_name_};
  }

  // Unused parts of the interface.
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }
  absl::StatusOr<std::unique_ptr<lm::Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const lm::proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("");
  }

 private:
  std::string section_name_;
  std::string section_data_;
};

// Serializes a full-size voice pack whose every element is `value`.
std::string MakeVoicePackBytes(float value) {
  const std::vector<float> floats(kokoro::kVoicePackSize, value);
  std::string bytes(floats.size() * sizeof(float), '\0');
  std::memcpy(bytes.data(), floats.data(), bytes.size());
  return bytes;
}

TEST(CommonTest, ConstantsSanity) {
  EXPECT_EQ(kokoro::kSampleRate, 24000);
  EXPECT_EQ(kokoro::kAudioHop, 600);
  EXPECT_EQ(kokoro::kVoicePackSize, 510 * 256);
  EXPECT_EQ(kokoro::kBosTokenId, 0);
  EXPECT_EQ(kokoro::kEosTokenId, 0);
  EXPECT_EQ(kokoro::kSpaceTokenId, 16);
}

TEST(CommonTest, LoadVoiceEmbeddingNotFound) {
  auto embed =
      kokoro::LoadVoiceEmbedding("/non_existent_dir", "non_existent_voice");
  EXPECT_FALSE(embed.ok());
}

TEST(CommonTest, SliceTokenIdsUnderCapacityReturnsSingleSlice) {
  std::vector<int> tokens = {kokoro::kBosTokenId, 20, 21, 22,
                             kokoro::kEosTokenId};
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_EQ(slices.size(), 1);
  EXPECT_EQ(slices[0].join_before, SliceJoin::kChunkBoundary);
  EXPECT_EQ(slices[0].join_after, SliceJoin::kChunkBoundary);
  EXPECT_EQ(slices[0].token_ids, tokens);
}

TEST(CommonTest, SliceTokenIdsPrefersPunctuationOverSpace) {
  std::vector<int> tokens = {kokoro::kBosTokenId, 20, 21, 22, 23, 24,
                             kokoro::kSpaceTokenId, 3, 25, 26, 27, 28,
                             kokoro::kEosTokenId};
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_GE(slices.size(), 2);
  EXPECT_EQ(slices[0].join_after, SliceJoin::kPunctuation);
  EXPECT_EQ(slices[1].join_before, SliceJoin::kPunctuation);
  EXPECT_EQ(slices[0].token_ids.front(), kokoro::kBosTokenId);
  EXPECT_EQ(slices[0].token_ids.back(), kokoro::kEosTokenId);
  EXPECT_EQ(slices[0].token_ids[slices[0].token_ids.size() - 2], 3);
}

TEST(CommonTest, SliceTokenIdsFallsBackToSpaceBoundary) {
  std::vector<int> tokens = {kokoro::kBosTokenId, 20, 21, 22, 23, 24,
                             kokoro::kSpaceTokenId, 25, 26, 27, 28, 29,
                             kokoro::kEosTokenId};
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_GE(slices.size(), 2);
  EXPECT_EQ(slices[0].join_after, SliceJoin::kWordBoundary);
  EXPECT_EQ(slices[1].join_before, SliceJoin::kWordBoundary);
}

TEST(CommonTest, SliceTokenIdsMidWordCutWhenNoBoundary) {
  std::vector<int> tokens(25, 20);
  tokens.front() = kokoro::kBosTokenId;
  tokens.back() = kokoro::kEosTokenId;
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_GE(slices.size(), 3);
  for (size_t i = 0; i < slices.size() - 1; ++i) {
    EXPECT_EQ(slices[i].join_after, SliceJoin::kWordBoundary);
    EXPECT_EQ(slices[i + 1].join_before, SliceJoin::kWordBoundary);
  }
}

TEST(CommonTest, SliceTokenIdsDoesNotEmitTrailingEosOnlySlice) {
  // 19 content tokens framed by BOS/EOS (21 total) with max_capacity=10: the
  // second slice ends right at the last content token (index 19), leaving only
  // the trailing EOS at index 20. Must not emit a 3rd phonemeless slice `{0}`.
  std::vector<int> mid_word(21, 20);
  mid_word.front() = kokoro::kBosTokenId;
  mid_word.back() = kokoro::kEosTokenId;
  auto mid_word_slices = kokoro::SliceTokenIds(mid_word, /*max_capacity=*/10);
  ASSERT_EQ(mid_word_slices.size(), 2);
  EXPECT_EQ(mid_word_slices[0].join_after, SliceJoin::kWordBoundary);
  EXPECT_EQ(mid_word_slices[1].join_before, SliceJoin::kWordBoundary);
  EXPECT_EQ(mid_word_slices[1].join_after, SliceJoin::kChunkBoundary);
  for (const auto& s : mid_word_slices) {
    EXPECT_GT(s.token_ids.size(), 2);
  }

  // Same boundary when the final content token before EOS (and a trailing
  // space) is sentence-final punctuation: the second slice must end with
  // kChunkBoundary rather than splitting on the final period into an empty
  // trailing slice.
  std::vector<int> punct_end = {kokoro::kBosTokenId,
                                20,
                                21,
                                22,
                                23,
                                24,
                                kokoro::kSpaceTokenId,
                                25,
                                26,
                                27,
                                28,
                                29,
                                30,
                                31,
                                32,
                                33,
                                /*.=*/4,
                                kokoro::kSpaceTokenId,
                                kokoro::kEosTokenId};
  auto punct_slices = kokoro::SliceTokenIds(punct_end, /*max_capacity=*/10);
  ASSERT_EQ(punct_slices.size(), 2);
  EXPECT_EQ(punct_slices[0].join_after, SliceJoin::kWordBoundary);
  EXPECT_EQ(punct_slices[1].join_before, SliceJoin::kWordBoundary);
  EXPECT_EQ(punct_slices[1].join_after, SliceJoin::kChunkBoundary);
  EXPECT_EQ(punct_slices[1].token_ids[punct_slices[1].token_ids.size() - 2], 4);
}

TEST(CommonTest, SliceTokenIdsKeepsClosingPunctuationWithLastWord) {
  // Text ending in `.”`: the second window ends between '.' and '”', so a split
  // on '.' would leave a phonemeless `{0, ”, 0}` slice. The trailing
  // punctuation must stay attached to the last word instead.
  constexpr int kPeriod = 4;
  constexpr int kCloseQuote = 15;
  const std::vector<int> tokens = {kokoro::kBosTokenId,
                                   20,
                                   21,
                                   22,
                                   23,
                                   24,
                                   kokoro::kSpaceTokenId,
                                   25,
                                   26,
                                   27,
                                   28,
                                   29,
                                   30,
                                   kokoro::kSpaceTokenId,
                                   31,
                                   32,
                                   kPeriod,
                                   kCloseQuote,
                                   kokoro::kEosTokenId};
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_EQ(slices.size(), 3);
  EXPECT_EQ(slices[2].token_ids,
            (std::vector<int>{kokoro::kBosTokenId, 31, 32, kPeriod, kCloseQuote,
                              kokoro::kEosTokenId}));
  EXPECT_EQ(slices[1].join_after, SliceJoin::kWordBoundary);
  EXPECT_EQ(slices[2].join_after, SliceJoin::kChunkBoundary);
}

// Builds a framed token sequence that looks like phonemizer output: words of
// voiced tokens separated by spaces, with clause/sentence punctuation, closing
// quotes and an optional trailing space before EOS.
std::vector<int> MakeSentenceLikeTokens(int num_words, uint32_t seed) {
  constexpr int kComma = 3;
  constexpr int kPeriod = 4;
  constexpr int kCloseQuote = 15;
  uint32_t state = seed;
  auto next = [&state](uint32_t bound) {
    state = state * 1664525u + 1013904223u;
    return (state >> 8) % bound;
  };
  std::vector<int> tokens = {kokoro::kBosTokenId};
  for (int w = 0; w < num_words; ++w) {
    if (w > 0) tokens.push_back(kokoro::kSpaceTokenId);
    const int word_len = 1 + static_cast<int>(next(8));
    for (int i = 0; i < word_len; ++i) {
      tokens.push_back(43 + static_cast<int>(next(100)));
    }
    switch (next(10)) {
      case 0:
        tokens.push_back(kComma);
        break;
      case 1:
        tokens.push_back(kPeriod);
        break;
      case 2:
        tokens.push_back(kPeriod);
        tokens.push_back(kCloseQuote);
        break;
      default:
        break;
    }
  }
  tokens.push_back(kPeriod);
  if (next(2) == 0) tokens.push_back(kCloseQuote);
  if (next(2) == 0) tokens.push_back(kokoro::kSpaceTokenId);
  tokens.push_back(kokoro::kEosTokenId);
  return tokens;
}

TEST(CommonTest, SliceTokenIdsPreservesEveryVoicedToken) {
  for (int max_capacity : {8, 10, 16, 31, 61}) {
    for (int num_words = 1; num_words <= 80; ++num_words) {
      for (uint32_t seed = 1; seed <= 5; ++seed) {
        const std::vector<int> tokens =
            MakeSentenceLikeTokens(num_words, seed * 7919u + num_words);
        SCOPED_TRACE(absl::StrCat("max_capacity=", max_capacity,
                                  " num_words=", num_words, " seed=", seed));
        const auto slices = kokoro::SliceTokenIds(tokens, max_capacity);
        ASSERT_FALSE(slices.empty());
        EXPECT_EQ(slices.front().join_before, SliceJoin::kChunkBoundary);
        EXPECT_EQ(slices.back().join_after, SliceJoin::kChunkBoundary);
        if (tokens.size() <= static_cast<size_t>(max_capacity)) {
          ASSERT_EQ(slices.size(), 1);
          EXPECT_EQ(slices[0].token_ids, tokens);
          continue;
        }

        // Expected content: the input without its BOS, trailing EOS and the
        // trailing spaces before it.
        size_t content_end = tokens.size();
        while (tokens[content_end - 1] == kokoro::kEosTokenId ||
               tokens[content_end - 1] == kokoro::kSpaceTokenId) {
          --content_end;
        }
        const std::vector<int> expected(tokens.begin() + 1,
                                        tokens.begin() + content_end);

        std::vector<int> concatenated;
        for (size_t i = 0; i < slices.size(); ++i) {
          const std::vector<int>& ids = slices[i].token_ids;
          ASSERT_GE(ids.size(), 3);
          EXPECT_LE(ids.size(), static_cast<size_t>(max_capacity) + 2);
          EXPECT_EQ(ids.front(), kokoro::kBosTokenId);
          EXPECT_EQ(ids.back(), kokoro::kEosTokenId);
          EXPECT_TRUE(
              std::any_of(ids.begin(), ids.end(),
                          [](int id) { return id > kokoro::kSpaceTokenId; }))
              << "slice " << i << " has no voiced token";
          if (i + 1 < slices.size()) {
            EXPECT_EQ(slices[i].join_after, slices[i + 1].join_before);
          }
          concatenated.insert(concatenated.end(), ids.begin() + 1,
                              ids.end() - 1);
        }
        EXPECT_EQ(concatenated, expected);
      }
    }
  }
}

TEST(CommonTest, SliceTokenIdsAdjacentSlicesAgreeOnJoins) {
  std::vector<int> tokens;
  for (int i = 0; i < 60; ++i) {
    if (i % 7 == 0) {
      tokens.push_back(kokoro::kMinPunctuationTokenId);
    } else if (i % 4 == 0) {
      tokens.push_back(kokoro::kSpaceTokenId);
    } else {
      tokens.push_back(30 + (i % 10));
    }
  }
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/15);
  ASSERT_GT(slices.size(), 1);
  EXPECT_EQ(slices.front().join_before, SliceJoin::kChunkBoundary);
  EXPECT_EQ(slices.back().join_after, SliceJoin::kChunkBoundary);
  for (size_t i = 0; i < slices.size() - 1; ++i) {
    EXPECT_EQ(slices[i].join_after, slices[i + 1].join_before);
    EXPECT_EQ(slices[i].token_ids.front(), kokoro::kBosTokenId);
    EXPECT_EQ(slices[i].token_ids.back(), kokoro::kEosTokenId);
  }
}

TEST(CommonTest, TrimSliceJoinSilenceChunkBoundaryBothEndsUntouched) {
  std::vector<float> pcm = {0.0f, 0.0f, 0.5f, 0.8f, 0.0f, 0.0f};
  std::vector<float> expected = pcm;
  kokoro::TrimSliceJoinSilence(pcm, SliceJoin::kChunkBoundary,
                               SliceJoin::kChunkBoundary);
  EXPECT_EQ(pcm, expected);
}

TEST(CommonTest, TrimSliceJoinSilenceDigitalSilenceTruncated) {
  std::vector<float> pcm(5000, 0.0f);
  kokoro::TrimSliceJoinSilence(pcm, SliceJoin::kWordBoundary,
                               SliceJoin::kWordBoundary);
  EXPECT_EQ(pcm.size(),
            static_cast<size_t>(kokoro::kSliceJoinMarginSamples * 2));
}

TEST(CommonTest, TrimSliceJoinSilenceTrimsLeadingAndTrailingSilenceAtWordBoundary) {
  std::vector<float> pcm(500, 0.0f);
  pcm.push_back(1.0f);
  pcm.insert(pcm.end(), 500, 0.0f);

  kokoro::TrimSliceJoinSilence(pcm, SliceJoin::kWordBoundary,
                               SliceJoin::kWordBoundary);
  EXPECT_EQ(pcm.size(),
            static_cast<size_t>(kokoro::kSliceJoinMarginSamples * 2 + 1));
  EXPECT_EQ(pcm[kokoro::kSliceJoinMarginSamples], 1.0f);
}

TEST(CommonTest, TrimSliceJoinSilencePunctuationPausePreservedOnTail) {
  std::vector<float> pcm(500, 0.0f);
  pcm.push_back(1.0f);
  pcm.insert(pcm.end(), 20000, 0.0f);

  kokoro::TrimSliceJoinSilence(pcm, SliceJoin::kChunkBoundary,
                               SliceJoin::kPunctuation);
  const size_t expected_size =
      500 + 1 + static_cast<size_t>(kokoro::kSlicePunctuationPauseSamples);
  EXPECT_EQ(pcm.size(), expected_size);
  EXPECT_EQ(pcm[500], 1.0f);
}

TEST(CommonTest, VoiceNameFromIdentifierStripsDirectoryAndExtension) {
  EXPECT_EQ(kokoro::VoiceNameFromIdentifier("af_heart"), "af_heart");
  EXPECT_EQ(kokoro::VoiceNameFromIdentifier("af_heart.bin"), "af_heart");
  EXPECT_EQ(kokoro::VoiceNameFromIdentifier("voices/af_bella.bin"),
            "af_bella");
  EXPECT_EQ(kokoro::VoiceNameFromIdentifier("/tmp/voices/am_adam"), "am_adam");
  EXPECT_EQ(kokoro::VoiceNameFromIdentifier(""), "");
}

TEST(CommonTest, LoadVoiceEmbeddingReadsRequestedContainerSection) {
  FakeModelResources resources("af_bella", MakeVoicePackBytes(0.25f));

  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> embed,
      kokoro::LoadVoiceEmbedding("/non_existent_dir", "voices/af_bella.bin",
                                 &resources));

  ASSERT_EQ(embed.size(), kokoro::kVoicePackSize);
  EXPECT_EQ(embed.front(), 0.25f);
  EXPECT_EQ(embed.back(), 0.25f);
}

TEST(CommonTest, LoadVoiceEmbeddingFallsBackToDefaultContainerVoice) {
  FakeModelResources resources(std::string(kokoro::kDefaultVoiceName),
                               MakeVoicePackBytes(0.5f));

  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> embed,
      kokoro::LoadVoiceEmbedding("/non_existent_dir", "voice_not_in_model",
                                 &resources));

  ASSERT_EQ(embed.size(), kokoro::kVoicePackSize);
  EXPECT_EQ(embed.front(), 0.5f);
}

TEST(CommonTest, LoadVoiceEmbeddingRejectsUndersizedContainerSection) {
  FakeModelResources resources(std::string(kokoro::kDefaultVoiceName),
                               "too-short");

  auto embed = kokoro::LoadVoiceEmbedding(
      "/non_existent_dir", kokoro::kDefaultVoiceName, &resources);

  EXPECT_FALSE(embed.ok());
}

}  // namespace
}  // namespace litert::omni::tts
