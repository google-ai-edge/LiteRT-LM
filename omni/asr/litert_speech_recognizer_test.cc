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
#include "absl/base/nullability.h"  // from @com_google_absl
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
    : public SingleThreadedStageWithDeque<std::vector<float>> {
 public:
  DummyAudioPreprocessor() = default;

  void PushFeatures(std::vector<float> features) {
    PushOutput(std::move(features));
  }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

// Minimal `AudioPreprocessor` whose only job is to report configurable
// `last_chunk_start_fraction()` / `last_chunk_valid_fraction()`.
class FakeChunkInfoPreprocessor : public AudioPreprocessor {
 public:
  explicit FakeChunkInfoPreprocessor(
      Stage<std::vector<float>>* absl_nonnull audio_source)
      : AudioPreprocessor(audio_source) {}

  void set_start_fraction(float start_fraction) {
    start_fraction_ = start_fraction;
  }
  void set_valid_fraction(float valid_fraction) {
    valid_fraction_ = valid_fraction;
  }
  float last_chunk_start_fraction() const override { return start_fraction_; }
  float last_chunk_valid_fraction() const override { return valid_fraction_; }

 protected:
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }

 private:
  float start_fraction_ = 0.0f;
  float valid_fraction_ = 1.0f;
};

class DummyDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  DummyDecoder() = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs) override {
    if (should_fail_) {
      return absl::InternalError("Decoder failed");
    }
    return decoded_tokens_;
  }

  void SetDecodeRange(const DecodeRange& range) override {
    ranges_seen_.emplace_back(range.start_fraction, range.valid_fraction);
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
  FakeChunkInfoPreprocessor chunk_info(&preprocessor);
  auto dummy_decoder = std::make_unique<DummyDecoder>();
  DummyDecoder* decoder = dummy_decoder.get();

  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(dummy_decoder), &chunk_info));

  // A full window, then a padded short chunk (short stream), then a chunk
  // shifted back over the previous window (final chunk of a long stream).
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});
  ASSERT_OK(recognizer->Schedule());
  chunk_info.set_valid_fraction(0.3f);
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});
  ASSERT_OK(recognizer->Schedule());
  chunk_info.set_valid_fraction(1.0f);
  chunk_info.set_start_fraction(0.6f);
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});
  ASSERT_OK(recognizer->Schedule());

  EXPECT_THAT(decoder->ranges_seen_,
              ::testing::ElementsAre(::testing::Pair(0.0f, 1.0f),
                                     ::testing::Pair(0.0f, 0.3f),
                                     ::testing::Pair(0.6f, 1.0f)));
}

TEST(LiteRtSpeechRecognizerTest,
     ScheduleWithoutChunkInfoTellsDecoderWholeWindowIsValid) {
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

class FakeCenteredChunkInfoPreprocessor : public FakeChunkInfoPreprocessor {
 public:
  using FakeChunkInfoPreprocessor::FakeChunkInfoPreprocessor;
  void set_centered_features(std::vector<float> features) {
    centered_features_ = std::move(features);
  }
  std::vector<float> last_chunk_centered_features() const override {
    return centered_features_;
  }

 private:
  std::vector<float> centered_features_;
};

class SequenceDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  explicit SequenceDecoder(
      std::vector<std::vector<SpeechRecognizer::DecodedToken>> sequence)
      : sequence_(std::move(sequence)) {}

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>&) override {
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
  FakeCenteredChunkInfoPreprocessor chunk_info(&preprocessor);
  chunk_info.set_centered_features({4.0f, 5.0f, 6.0f});
  preprocessor.PushFeatures({1.0f, 2.0f, 3.0f});

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

  ASSERT_OK_AND_ASSIGN(
      auto recognizer,
      LiteRtSpeechRecognizer::Create(std::move(mock_runner), &preprocessor,
                                     std::move(decoder), &chunk_info));

  ASSERT_OK(recognizer->Schedule());
  ASSERT_OK_AND_ASSIGN(auto tokens, recognizer->GetOutput());
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 20);
  EXPECT_EQ(tokens[0].timestamp_ms, 2);
}

}  // namespace
}  // namespace litert::omni::asr
