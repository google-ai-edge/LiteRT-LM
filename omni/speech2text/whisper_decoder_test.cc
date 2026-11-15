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

#include "omni/speech2text/whisper_decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/speech2text/speech_recognizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::asr {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr size_t kVocabSize = 51865;

::litert::TensorBuffer CreateTestTensorBuffer(size_t num_elements,
                                              size_t element_size_bytes) {
  const size_t total_bytes = num_elements * element_size_bytes;
  ::litert::RankedTensorType tensor_type(
      element_size_bytes == sizeof(float) ? ::litert::ElementType::Float32
                                          : ::litert::ElementType::Int32,
      ::litert::Layout(
          ::litert::Dimensions{static_cast<int32_t>(num_elements)}));
  auto buffer = ::litert::TensorBuffer::CreateManagedHostMemory(
      tensor_type, total_bytes);
  return std::move(*buffer);
}

void SetupMockRunnerBuffers(MockLiteRtRunner& mock_runner) {
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));  // token
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));  // enc
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));  // state
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(kVocabSize, sizeof(float)));        // logits
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));  // state
        return buffers;
      });
}

TEST(WhisperDecoderTest, CreateWhenRunnerNullReturnsError) {
  LiteRtRunner* null_runner = nullptr;
  EXPECT_FALSE(WhisperDecoder::Create(null_runner).ok());
}

TEST(WhisperDecoderTest, CreateWhenMaxDecodeStepsInvalidReturnsError) {
  MockLiteRtRunner mock_runner;
  WhisperDecoder::Options options;
  options.max_decode_steps = 0;
  EXPECT_FALSE(
      WhisperDecoder::Create(&mock_runner, "decode", 2, 1, {}, options).ok());
}

TEST(WhisperDecoderTest, CreateWhenModelHasZeroStateTensorsReturnsError) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        // Non-state input buffers: token and encoder output.
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        // Non-state output buffer: logits.
        buffers.push_back(CreateTestTensorBuffer(kVocabSize, sizeof(float)));
        return buffers;
      });

  auto status =
      WhisperDecoder::Create(&mock_runner, "decode",
                             /*num_non_state_inputs=*/2,
                             /*num_non_state_outputs=*/1)
          .status();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_THAT(status.message(), HasSubstr("recurrent state tensors"));
}

TEST(WhisperDecoderTest, CreateFiltersNoTimestampsWhenTimestampsEnabled) {
  MockLiteRtRunner mock_runner;
  SetupMockRunnerBuffers(mock_runner);

  WhisperDecoder::Options options;
  options.enable_timestamps = true;
  std::vector<int> starts = {
      WhisperDecoder::kWhisperStartOfTranscriptTokenId,
      50259,  // <|en|>
      WhisperDecoder::kWhisperTranscribeTokenId,
      WhisperDecoder::kWhisperNoTimestampsTokenId,
  };

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      WhisperDecoder::Create(&mock_runner, "decode", 2, 1, starts, options));
  EXPECT_THAT(decoder->decode_start_token_ids(),
              ElementsAre(WhisperDecoder::kWhisperStartOfTranscriptTokenId,
                          50259, WhisperDecoder::kWhisperTranscribeTokenId));
}

TEST(WhisperDecoderTest, CreateRetainsNoTimestampsWhenTimestampsDisabled) {
  MockLiteRtRunner mock_runner;
  SetupMockRunnerBuffers(mock_runner);

  WhisperDecoder::Options options;
  options.enable_timestamps = false;
  std::vector<int> starts = {
      WhisperDecoder::kWhisperStartOfTranscriptTokenId,
      50259,  // <|en|>
      WhisperDecoder::kWhisperTranscribeTokenId,
      WhisperDecoder::kWhisperNoTimestampsTokenId,
  };

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      WhisperDecoder::Create(&mock_runner, "decode", 2, 1, starts, options));
  EXPECT_THAT(
      decoder->decode_start_token_ids(),
      ElementsAre(WhisperDecoder::kWhisperStartOfTranscriptTokenId, 50259,
                  WhisperDecoder::kWhisperTranscribeTokenId,
                  WhisperDecoder::kWhisperNoTimestampsTokenId));
}

TEST(WhisperDecoderTest, DecodeFailsWhenEncoderOutputsEmpty) {
  MockLiteRtRunner mock_runner;
  SetupMockRunnerBuffers(mock_runner);

  ASSERT_OK_AND_ASSIGN(auto decoder, WhisperDecoder::Create(&mock_runner));
  std::vector<::litert::TensorBuffer> empty_encoder_outputs;
  EXPECT_FALSE(decoder->Decode(empty_encoder_outputs).ok());
}

TEST(WhisperDecoderTest,
     DecodeWithTimestampsExtractsTimestampsAndOmitsTokensFromText) {
  MockLiteRtRunner mock_runner;
  SetupMockRunnerBuffers(mock_runner);

  WhisperDecoder::Options options;
  options.enable_timestamps = true;
  std::vector<int> starts = {
      WhisperDecoder::kWhisperStartOfTranscriptTokenId,
      WhisperDecoder::kWhisperTranscribeTokenId,
  };

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      WhisperDecoder::Create(&mock_runner, "decode", 2, 1, starts, options));

  int step_count = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&step_count](
                          absl::string_view,
                          absl::Span<const ::litert::TensorBuffer>,
                          absl::Span<const ::litert::TensorBuffer> outputs) {
        std::vector<float> logits(kVocabSize, 0.0f);
        if (step_count == 0) {
          // Feeding prompt token 50258
        } else if (step_count == 1) {
          // Feeding prompt token 50359 -> yields initial timestamp token 50364 (0ms)
          logits[WhisperDecoder::kWhisperTimestampBeginTokenId] = 100.0f;
        } else if (step_count == 2) {
          // Step(50364) -> yields word token 1000
          logits[1000] = 100.0f;
        } else if (step_count == 3) {
          // Step(1000) -> yields timestamp token 50374 (200ms)
          logits[WhisperDecoder::kWhisperTimestampBeginTokenId + 10] = 100.0f;
        } else if (step_count == 4) {
          // Step(50374) -> yields word token 2000
          logits[2000] = 100.0f;
        } else {
          // Step(2000) -> yields end of text token 50257
          logits[WhisperDecoder::kWhisperEndOfTextTokenId] = 100.0f;
        }
        step_count++;
        auto& mutable_out = const_cast<::litert::TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            mutable_out.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));

  ASSERT_OK_AND_ASSIGN(auto decoded_tokens, decoder->Decode(encoder_outputs));

  // Word tokens followed by end-of-chunk sentinel with chunk duration.
  ASSERT_EQ(decoded_tokens.size(), 3);
  EXPECT_EQ(decoded_tokens[0].token_id, 1000);
  ASSERT_TRUE(decoded_tokens[0].timestamp_ms.has_value());
  EXPECT_EQ(*decoded_tokens[0].timestamp_ms, 0);

  EXPECT_EQ(decoded_tokens[1].token_id, 2000);
  ASSERT_TRUE(decoded_tokens[1].timestamp_ms.has_value());
  EXPECT_EQ(*decoded_tokens[1].timestamp_ms, 200);

  EXPECT_EQ(decoded_tokens[2].token_id,
            SpeechRecognizer::DecodedToken::kEndOfChunkTokenId);
}

TEST(WhisperDecoderTest, DecodeWithoutTimestampsSuppressesTimestampTokens) {
  MockLiteRtRunner mock_runner;
  SetupMockRunnerBuffers(mock_runner);

  WhisperDecoder::Options options;
  options.enable_timestamps = false;
  std::vector<int> starts = {
      WhisperDecoder::kWhisperStartOfTranscriptTokenId,
  };

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      WhisperDecoder::Create(&mock_runner, "decode", 2, 1, starts, options));

  int step_count = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&step_count](
                          absl::string_view,
                          absl::Span<const ::litert::TensorBuffer>,
                          absl::Span<const ::litert::TensorBuffer> outputs) {
        std::vector<float> logits(kVocabSize, 0.0f);
        if (step_count == 0) {
          // Feeding prompt token 50258 -> timestamp token 50364 has highest logit,
          // but should be suppressed. Token 1000 should be selected.
          logits[WhisperDecoder::kWhisperTimestampBeginTokenId] = 100.0f;
          logits[1000] = 50.0f;
        } else {
          // Step(1000) -> yields end of text
          logits[WhisperDecoder::kWhisperEndOfTextTokenId] = 100.0f;
        }
        step_count++;
        auto& mutable_out = const_cast<::litert::TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            mutable_out.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));

  ASSERT_OK_AND_ASSIGN(auto decoded_tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(decoded_tokens.size(), 1);
  EXPECT_EQ(decoded_tokens[0].token_id, 1000);
  EXPECT_FALSE(decoded_tokens[0].timestamp_ms.has_value());
}

TEST(WhisperDecoderTest, DecodeSuppressesBeginSuppressTokensOnStep0) {
  MockLiteRtRunner mock_runner;
  SetupMockRunnerBuffers(mock_runner);

  WhisperDecoder::Options options;
  options.enable_timestamps = false;
  options.begin_suppress_tokens = {220,
                                   WhisperDecoder::kWhisperEndOfTextTokenId};
  std::vector<int> starts = {
      WhisperDecoder::kWhisperStartOfTranscriptTokenId,
  };

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      WhisperDecoder::Create(&mock_runner, "decode", 2, 1, starts, options));

  int step_count = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&step_count](
                          absl::string_view,
                          absl::Span<const ::litert::TensorBuffer>,
                          absl::Span<const ::litert::TensorBuffer> outputs) {
        std::vector<float> logits(kVocabSize, 0.0f);
        if (step_count == 0) {
          // Step 0: token 220 has highest logit, 50257 has 2nd highest,
          // token 500 has 3rd highest.
          logits[220] = 100.0f;
          logits[WhisperDecoder::kWhisperEndOfTextTokenId] = 90.0f;
          logits[500] = 80.0f;
        } else {
          // Step 1: 50257 is now allowed because begin_suppress_tokens is only on step 0.
          logits[WhisperDecoder::kWhisperEndOfTextTokenId] = 100.0f;
        }
        step_count++;
        auto& mutable_out = const_cast<::litert::TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            mutable_out.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));

  ASSERT_OK_AND_ASSIGN(auto decoded_tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(decoded_tokens.size(), 1);
  EXPECT_EQ(decoded_tokens[0].token_id, 500);
}

TEST(WhisperDecoderTest, DecodeEnforcesMonotonicTimestamps) {
  MockLiteRtRunner mock_runner;
  SetupMockRunnerBuffers(mock_runner);

  WhisperDecoder::Options options;
  options.enable_timestamps = true;
  std::vector<int> starts = {
      WhisperDecoder::kWhisperStartOfTranscriptTokenId,
  };

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      WhisperDecoder::Create(&mock_runner, "decode", 2, 1, starts, options));

  int step_count = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&step_count](
                          absl::string_view,
                          absl::Span<const ::litert::TensorBuffer>,
                          absl::Span<const ::litert::TensorBuffer> outputs) {
        std::vector<float> logits(kVocabSize, 0.0f);
        if (step_count == 0) {
          // Yields timestamp token at 50384 (400ms)
          logits[WhisperDecoder::kWhisperTimestampBeginTokenId + 20] = 100.0f;
        } else if (step_count == 1) {
          // Yields word 100
          logits[100] = 100.0f;
        } else if (step_count == 2) {
          // Model predicts an earlier timestamp 50364 (0ms) with 100.0f,
          // but monotonic constraint suppresses it in favor of 50394 (600ms) with 50.0f.
          logits[WhisperDecoder::kWhisperTimestampBeginTokenId] = 100.0f;
          logits[WhisperDecoder::kWhisperTimestampBeginTokenId + 30] = 50.0f;
        } else if (step_count == 3) {
          // Yields word 200
          logits[200] = 100.0f;
        } else {
          // End of text
          logits[WhisperDecoder::kWhisperEndOfTextTokenId] = 100.0f;
        }
        step_count++;
        auto& mutable_out = const_cast<::litert::TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            mutable_out.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));

  ASSERT_OK_AND_ASSIGN(auto decoded_tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(decoded_tokens.size(), 3);
  EXPECT_EQ(decoded_tokens[0].token_id, 100);
  EXPECT_EQ(*decoded_tokens[0].timestamp_ms, 400);

  EXPECT_EQ(decoded_tokens[1].token_id, 200);
  EXPECT_EQ(*decoded_tokens[1].timestamp_ms, 600);

  EXPECT_EQ(decoded_tokens[2].token_id,
            SpeechRecognizer::DecodedToken::kEndOfChunkTokenId);
}

}  // namespace
}  // namespace litert::omni::asr
