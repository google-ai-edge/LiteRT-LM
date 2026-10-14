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

#include "omni/asr/litert_speech_recognizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/stage.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::_;
using ::testing::Return;

::litert::TensorBuffer CreateTestTensorBuffer(size_t num_elements,
                                              size_t element_size_bytes) {
  alignas(64) static uint8_t memory_pool[8192] = {0};
  const size_t total_bytes = num_elements * element_size_bytes;
  ::litert::RankedTensorType tensor_type(
      element_size_bytes == sizeof(float) ? ::litert::ElementType::Float32
                                          : ::litert::ElementType::Int32,
      ::litert::Layout(
          ::litert::Dimensions{static_cast<int32_t>(num_elements)}));
  auto buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      tensor_type, memory_pool, total_bytes);
  return std::move(*buffer);
}

class DummyAudioPreprocessor
    : public SingleThreadedStageWithDeque<PreprocessedAudio> {
 public:
  DummyAudioPreprocessor() = default;

  void PushFeatures(
      std::vector<float> features,
      std::optional<PreprocessedAudio::DecodeRange> decode_range = std::nullopt,
      std::optional<std::vector<float>> centered_features = std::nullopt) {
    PushOutput(PreprocessedAudio{
        .features = std::move(features),
        .centered_features = std::move(centered_features),
        .decode_range = decode_range,
    });
  }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

class DummyDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  DummyDecoder() = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs,
      std::optional<DecodeRange> decode_range) override {
    const DecodeRange range = decode_range.value_or(DecodeRange{});
    ranges_seen_.emplace_back(range.start_fraction, range.valid_fraction);
    if (should_fail_) {
      return absl::InternalError("Decoder failed");
    }
    return decoded_tokens_;
  }

  void SetDecodedTokens(std::vector<SpeechRecognizer::DecodedToken> tokens) {
    decoded_tokens_ = std::move(tokens);
  }

  void SetShouldFail(bool should_fail) { should_fail_ = should_fail; }

  bool should_fail_ = false;
  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens_{
      SpeechRecognizer::DecodedToken{.token_id = 42,
                                     .timestamp_ms = std::nullopt}};
  std::vector<std::pair<float, float>> ranges_seen_;
};

TEST(LiteRtSpeechRecognizerTest, ResetClearsOutputs) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });

  DummyAudioPreprocessor preprocessor;
  auto dummy_decoder = std::make_unique<DummyDecoder>();

  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(dummy_decoder)));

  recognizer->PushOutputForTesting(
      std::vector<SpeechRecognizer::DecodedToken>{});
  EXPECT_TRUE(recognizer->HasOutput());

  recognizer->Reset();
  EXPECT_FALSE(recognizer->HasOutput());
}

TEST(LiteRtSpeechRecognizerTest, SchedulePushesDecodedTokens) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, Run("encode", _, _))
      .WillOnce(Return(absl::OkStatus()));

  DummyAudioPreprocessor preprocessor;
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});

  auto dummy_decoder = std::make_unique<DummyDecoder>();
  dummy_decoder->SetDecodedTokens({
      SpeechRecognizer::DecodedToken{.token_id = 101,
                                     .timestamp_ms = std::nullopt},
  });

  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(dummy_decoder)));

  ASSERT_OK(recognizer->Schedule());
  ASSERT_OK_AND_ASSIGN(auto tokens, recognizer->GetOutput());
  EXPECT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token_id, 101);
}

TEST(LiteRtSpeechRecognizerTest, ScheduleFailsWhenEncodeFails) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, Run("encode", _, _))
      .WillOnce(Return(absl::InternalError("Encode error")));

  DummyAudioPreprocessor preprocessor;
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});

  auto dummy_decoder = std::make_unique<DummyDecoder>();

  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(dummy_decoder)));

  EXPECT_FALSE(recognizer->Schedule().ok());
}

TEST(LiteRtSpeechRecognizerTest, ScheduleFailsWhenDecodeFails) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, Run("encode", _, _))
      .WillOnce(Return(absl::OkStatus()));

  DummyAudioPreprocessor preprocessor;
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});

  auto dummy_decoder = std::make_unique<DummyDecoder>();
  dummy_decoder->SetShouldFail(true);

  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(dummy_decoder)));

  EXPECT_FALSE(recognizer->Schedule().ok());
}

TEST(LiteRtSpeechRecognizerTest,
     ScheduleEmptyMelFeaturesReturnsEndOfChunkTokenWithoutRunningEncoder) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, Run("encode", _, _)).Times(0);

  DummyAudioPreprocessor preprocessor;
  preprocessor.PushFeatures({});

  auto dummy_decoder = std::make_unique<DummyDecoder>();
  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(dummy_decoder)));

  ASSERT_OK(recognizer->Schedule());
  ASSERT_OK_AND_ASSIGN(auto tokens, recognizer->GetOutput());
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token_id,
            SpeechRecognizer::DecodedToken::kEndOfChunkTokenId);
  EXPECT_EQ(tokens[0].timestamp_ms, std::nullopt);
}

TEST(LiteRtSpeechRecognizerTest, ScheduleForwardsChunkDecodeRangeToDecoder) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, Run("encode", _, _))
      .Times(3)
      .WillRepeatedly(Return(absl::OkStatus()));

  DummyAudioPreprocessor preprocessor;
  auto dummy_decoder = std::make_unique<DummyDecoder>();
  DummyDecoder* decoder = dummy_decoder.get();

  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(dummy_decoder)));

  // A full window, then a padded short chunk (short stream), then a chunk
  // shifted back over the previous window (final chunk of a long stream).
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});
  ASSERT_OK(recognizer->Schedule());
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f}, PreprocessedAudio::DecodeRange{
                                                    .valid_fraction = 0.3f});
  ASSERT_OK(recognizer->Schedule());
  preprocessor.PushFeatures(
      {1.0f, 2.0f, 3.0f}, PreprocessedAudio::DecodeRange{
                              .start_fraction = 0.6f, .valid_fraction = 1.0f});
  ASSERT_OK(recognizer->Schedule());

  EXPECT_THAT(decoder->ranges_seen_,
              ::testing::ElementsAre(::testing::Pair(0.0f, 1.0f),
                                     ::testing::Pair(0.0f, 0.3f),
                                     ::testing::Pair(0.6f, 1.0f)));
}

TEST(LiteRtSpeechRecognizerTest,
     ScheduleWithoutDecodeRangeTellsDecoderWholeWindowIsValid) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, Run("encode", _, _))
      .WillOnce(Return(absl::OkStatus()));

  DummyAudioPreprocessor preprocessor;
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});
  auto dummy_decoder = std::make_unique<DummyDecoder>();
  DummyDecoder* decoder = dummy_decoder.get();

  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(dummy_decoder)));

  ASSERT_OK(recognizer->Schedule());
  EXPECT_THAT(decoder->ranges_seen_,
              ::testing::ElementsAre(::testing::Pair(0.0f, 1.0f)));
}

class SequenceDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  explicit SequenceDecoder(
      std::vector<std::vector<SpeechRecognizer::DecodedToken>> sequence)
      : sequence_(std::move(sequence)) {}

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>&,
      std::optional<DecodeRange> decode_range) override {
    if (call_index_ < sequence_.size()) {
      return sequence_[call_index_++];
    }
    return sequence_.back();
  }

 private:
  std::vector<std::vector<SpeechRecognizer::DecodedToken>> sequence_;
  size_t call_index_ = 0;
};

TEST(LiteRtSpeechRecognizerTest,
     ScheduleRetriesWithCenteredFeaturesOnCollapsedFullWindow) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, Run("encode", _, _))
      .Times(2)
      .WillRepeatedly(Return(absl::OkStatus()));

  DummyAudioPreprocessor preprocessor;
  preprocessor.PushFeatures(
      {1.0f, 2.0f, 3.0f}, PreprocessedAudio::DecodeRange{},
      /*centered_features=*/std::vector<float>{4.0f, 5.0f, 6.0f});

  auto decoder = std::make_unique<SequenceDecoder>(
      std::vector<std::vector<SpeechRecognizer::DecodedToken>>{
          // Primary pass collapses in the first 35 frames, emitting a late
          // hallucination at frame 45.
          {{.token_id = 10, .timestamp_ms = 45},
           {.token_id = 11, .timestamp_ms = 50},
           {.token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId}},
          // Centered fallback recovers the leading overlap at frame 2.
          {{.token_id = 20, .timestamp_ms = 2},
           {.token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId}},
      });

  ASSERT_OK_AND_ASSIGN(auto recognizer, LiteRtSpeechRecognizer::Create(
                                            std::move(mock_runner),
                                            &preprocessor, std::move(decoder)));

  ASSERT_OK(recognizer->Schedule());
  ASSERT_OK_AND_ASSIGN(auto tokens, recognizer->GetOutput());
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 20);
  EXPECT_EQ(tokens[0].timestamp_ms, 2);
}

TEST(LiteRtSpeechRecognizerTest,
     ScheduleRetriesShiftedChunkUsingRelativeTrailingGapAndSkipsHealthyChunk) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(*mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  // First chunk is healthy (1 encode call); second chunk has a >= 15-frame
  // trailing gap relative to the shifted end-of-chunk timestamp (2 encode
  // calls); third chunk collapsed to only kEndOfChunkTokenId (2 encode calls).
  EXPECT_CALL(*mock_runner, Run("encode", _, _))
      .Times(5)
      .WillRepeatedly(Return(absl::OkStatus()));

  DummyAudioPreprocessor preprocessor;
  // Shifted chunk with start_fraction = 0.6 -> shifted end-of-chunk timestamp
  // is 26 (63 - floor(0.6 * 63) = 26). A final speech token at shifted frame
  // 22 is only 4 frames from the end of the window, so no retry should occur.
  preprocessor.PushFeatures(
      {1.0f, 2.0f, 3.0f},
      PreprocessedAudio::DecodeRange{.start_fraction = 0.6f,
                                     .valid_fraction = 1.0f},
      /*centered_features=*/std::vector<float>{4.0f, 5.0f, 6.0f});
  // Second shifted chunk has its last speech token at shifted frame 5 (gap =
  // 26 - 5 = 21 >= 15), triggering a retry.
  preprocessor.PushFeatures(
      {1.0f, 2.0f, 3.0f},
      PreprocessedAudio::DecodeRange{.start_fraction = 0.6f,
                                     .valid_fraction = 1.0f},
      /*centered_features=*/std::vector<float>{4.0f, 5.0f, 6.0f});
  // Third chunk collapses completely to only kEndOfChunkTokenId, triggering a
  // retry.
  preprocessor.PushFeatures(
      {1.0f, 2.0f, 3.0f},
      PreprocessedAudio::DecodeRange{.start_fraction = 0.6f,
                                     .valid_fraction = 1.0f},
      /*centered_features=*/std::vector<float>{4.0f, 5.0f, 6.0f});

  auto decoder = std::make_unique<SequenceDecoder>(
      std::vector<std::vector<SpeechRecognizer::DecodedToken>>{
          // Chunk 1 primary pass (healthy shifted chunk: gap = 26 - 22 = 4).
          {{.token_id = 10, .timestamp_ms = 22},
           {.token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
            .timestamp_ms = 26}},
          // Chunk 2 primary pass (collapsed tail: gap = 26 - 5 = 21 >= 15).
          {{.token_id = 11, .timestamp_ms = 5},
           {.token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
            .timestamp_ms = 26}},
          // Chunk 2 retry pass recovers the trailing token.
          {{.token_id = 11, .timestamp_ms = 5},
           {.token_id = 12, .timestamp_ms = 24},
           {.token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
            .timestamp_ms = 26}},
          // Chunk 3 primary pass (empty chunk: only kEndOfChunkTokenId).
          {{.token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
            .timestamp_ms = 26}},
          // Chunk 3 retry pass recovers speech token.
          {{.token_id = 13, .timestamp_ms = 20},
           {.token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
            .timestamp_ms = 26}},
      });

  ASSERT_OK_AND_ASSIGN(auto recognizer, LiteRtSpeechRecognizer::Create(
                                            std::move(mock_runner),
                                            &preprocessor, std::move(decoder)));

  ASSERT_OK(recognizer->Schedule());
  ASSERT_OK_AND_ASSIGN(auto chunk1_tokens, recognizer->GetOutput());
  ASSERT_EQ(chunk1_tokens.size(), 2);
  EXPECT_EQ(chunk1_tokens[0].token_id, 10);

  ASSERT_OK(recognizer->Schedule());
  ASSERT_OK_AND_ASSIGN(auto chunk2_tokens, recognizer->GetOutput());
  ASSERT_EQ(chunk2_tokens.size(), 3);
  EXPECT_EQ(chunk2_tokens[1].token_id, 12);

  ASSERT_OK(recognizer->Schedule());
  ASSERT_OK_AND_ASSIGN(auto chunk3_tokens, recognizer->GetOutput());
  ASSERT_EQ(chunk3_tokens.size(), 2);
  EXPECT_EQ(chunk3_tokens[0].token_id, 13);
}

}  // namespace
}  // namespace litert::omni::asr
