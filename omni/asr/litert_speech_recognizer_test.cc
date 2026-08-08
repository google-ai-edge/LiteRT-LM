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

  void SetDecodedTokens(std::vector<SpeechRecognizer::DecodedToken> tokens) {
    decoded_tokens_ = std::move(tokens);
  }

  void SetShouldFail(bool should_fail) { should_fail_ = should_fail; }

  bool should_fail_ = false;
  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens_{
      SpeechRecognizer::DecodedToken{.token_id = 42,
                                     .timestamp_ms = std::nullopt}};
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

}  // namespace
}  // namespace litert::omni::asr
