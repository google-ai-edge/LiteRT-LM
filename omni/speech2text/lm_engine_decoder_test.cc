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

#include "omni/speech2text/lm_engine_decoder.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_lm_engine_runner.h"
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::_;
using ::testing::Return;

class MockLiteRtLmEngineRunner : public LiteRtLmEngineRunner {
 public:
  MOCK_METHOD(absl::Status, Prefill, (std::vector<lm::InputData> inputs),
              (override));
  MOCK_METHOD(absl::StatusOr<lm::Responses>, Decode,
              (const lm::DecodeConfig& decode_config), (override));
  MOCK_METHOD(absl::Status, Reset, (), (override));
  MOCK_METHOD(const lm::ModelResources*, model_resources, (),
              (const, override));
  MOCK_METHOD(lm::ModelResources*, mutable_model_resources, (), (override));
};

TEST(LmEngineDecoderTest, DecodeAssemblesMultimodalInputsAndEmitsTokens) {
  MockLiteRtLmEngineRunner mock_runner;

  EXPECT_CALL(mock_runner, Reset()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_runner, Prefill(_))
      .WillOnce([](const std::vector<lm::InputData>& contents) {
        EXPECT_EQ(contents.size(), 3);  // prompt, audio, audio_end
        return absl::OkStatus();
      });
  lm::Responses responses(lm::TaskState::kDone,
                          /*response_texts=*/{},
                          /*scores=*/{},
                          /*token_lengths=*/{},
                          /*token_ids=*/{{100, 10, 20, 1}});
  EXPECT_CALL(mock_runner, Decode(_)).WillOnce(Return(responses));

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      LmEngineDecoder::Create(&mock_runner, /*prompt=*/"Transcribe: ",
                              /*max_output_tokens=*/64,
                              /*decode_start_token_ids=*/{},
                              /*decode_stop_token_id=*/1,
                              /*decode_skip_until_token_id=*/100));

  // Flat audio buffer: 2 frames * 128 mel bins floats.
  auto audio_tb = support::CreateTensorBuffer<float>({2 * 128});
  ASSERT_TRUE(audio_tb.HasValue());
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(std::move(audio_tb.Value()));

  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  // Token 100 was skipped, stop token 1 terminated, leaving tokens 10 and 20.
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 10);
  EXPECT_EQ(tokens[1].token_id, 20);
}

TEST(LmEngineDecoderTest, FailsOnEmptyEncoderOutputs) {
  MockLiteRtLmEngineRunner mock_runner;
  ASSERT_OK_AND_ASSIGN(auto decoder, LmEngineDecoder::Create(&mock_runner));

  std::vector<TensorBuffer> empty_outputs;
  EXPECT_FALSE(decoder->Decode(empty_outputs).ok());
}

TEST(LmEngineDecoderTest, FailsOnNonEmptyDecodeStartTokenIds) {
  MockLiteRtLmEngineRunner mock_runner;
  EXPECT_FALSE(LmEngineDecoder::Create(&mock_runner, /*prompt=*/"",
                                       /*max_output_tokens=*/64,
                                       /*decode_start_token_ids=*/{1})
                   .ok());
}

TEST(LmEngineDecoderTest, TruncatesTrailingRepetitionLoop) {
  MockLiteRtLmEngineRunner mock_runner;
  EXPECT_CALL(mock_runner, Reset()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_runner, Prefill(_)).WillOnce(Return(absl::OkStatus()));

  // Token 10 followed by a runaway repetition loop of token 42 without a stop
  // token.
  lm::Responses responses(lm::TaskState::kDone,
                          /*response_texts=*/{},
                          /*scores=*/{},
                          /*token_lengths=*/{},
                          /*token_ids=*/{{10, 42, 42, 42, 42, 42, 42, 42, 42}});
  EXPECT_CALL(mock_runner, Decode(_)).WillOnce(Return(responses));

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      LmEngineDecoder::Create(&mock_runner, /*prompt=*/"Transcribe: ",
                              /*max_output_tokens=*/64,
                              /*decode_start_token_ids=*/{},
                              /*decode_stop_token_id=*/1));
  auto audio_tb = support::CreateTensorBuffer<float>({2 * 128});
  ASSERT_TRUE(audio_tb.HasValue());
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(std::move(audio_tb.Value()));

  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  // TruncateOnTrailingRepetition keeps the first occurrence of token 42 and
  // truncates the remaining repetitions.
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 10);
  EXPECT_EQ(tokens[1].token_id, 42);
}

}  // namespace
}  // namespace litert::omni::asr
