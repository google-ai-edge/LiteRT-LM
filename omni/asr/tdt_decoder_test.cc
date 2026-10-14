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

#include "omni/asr/tdt_decoder.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "omni/asr/audio_preprocessor.h"
#include "omni/base/mock_litert_runner.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::_;
using ::testing::Return;

TensorBuffer CreateTestTensorBuffer(size_t num_elements,
                                    size_t element_size_bytes) {
  RankedTensorType tensor_type(
      element_size_bytes == sizeof(float) ? ElementType::Float32
                                          : ElementType::Int32,
      Layout(Dimensions{static_cast<int32_t>(num_elements)}));
  auto buffer_or = TensorBuffer::CreateManagedHostMemory(
      tensor_type, num_elements * element_size_bytes);
  EXPECT_TRUE(buffer_or.HasValue());
  return std::move(*buffer_or);
}

TEST(TdtDecoderTest, CreateWhenCreateInputBuffersFailsReturnsError) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce(Return(absl::InternalError("Failed to create input buffers")));

  EXPECT_FALSE(TdtDecoder::Create(&mock_runner).ok());
}

TEST(TdtDecoderTest, CreateWithEmptyInputBuffersFails) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) { return std::vector<TensorBuffer>(); });

  EXPECT_FALSE(TdtDecoder::Create(&mock_runner).ok());
}

TEST(TdtDecoderTest, CreateWithInvalidDecodeStartTokenIdsFails) {
  MockLiteRtRunner mock_runner;
  EXPECT_FALSE(
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{}).ok());
  EXPECT_FALSE(
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{-1}).ok());
  EXPECT_FALSE(
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0, 1}).ok());
}

TEST(TdtDecoderTest, CreateSuccessfully) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 4, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  ASSERT_OK_AND_ASSIGN(auto decoder, TdtDecoder::Create(&mock_runner));
  EXPECT_NE(decoder, nullptr);
}

TEST(TdtDecoderTest, DecodeFailsWhenRunnerRunFails) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 1, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillOnce(Return(absl::InternalError("Run failed")));

  ASSERT_OK_AND_ASSIGN(auto decoder, TdtDecoder::Create(&mock_runner));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024, sizeof(float)));
  auto tokens_or = decoder->Decode(encoder_outputs);
  EXPECT_FALSE(tokens_or.ok());
}

TEST(TdtDecoderTest, DecodeIncludesEndOfChunkToken) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 1, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        auto buf0 = CreateTestTensorBuffer(100, sizeof(float));
        std::vector<float> logits(100, 0.0f);
        logits[99] =
            1.0f;  // Set duration > 0 (duration index 4 relative to start)
        EXPECT_TRUE(buf0.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        buffers.push_back(std::move(buf0));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillOnce(Return(absl::OkStatus()));

  ASSERT_OK_AND_ASSIGN(auto decoder, TdtDecoder::Create(&mock_runner));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_GE(tokens.size(), 1);
  EXPECT_TRUE(tokens.back().IsEndOfChunk());
  EXPECT_EQ(tokens.back().timestamp_ms, 1);
}

TEST(TdtDecoderTest, StatefulDecodeTransitionAndBlankHandling) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 3, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(2, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(60, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillRepeatedly([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 1, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode_1"))
      .WillRepeatedly([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(30, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });

  int run_step = 0;
  EXPECT_CALL(mock_runner, Run(_, _, _))
      .WillRepeatedly([&run_step](absl::string_view signature,
                                  absl::Span<const TensorBuffer> inputs,
                                  absl::Span<const TensorBuffer> outputs) {
        ++run_step;
        if (run_step == 1) {
          EXPECT_EQ(signature, "decode");
          // Step 1: Emit token 1 with duration 0 at time 0.
          std::vector<float> logits(60, 0.0f);
          logits[1] = 1.0f;  // token 1
          logits[5] = 1.0f;  // duration 0
          auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
          EXPECT_TRUE(
              logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        } else if (run_step == 2) {
          EXPECT_EQ(signature, "decode");
          // Step 2: In chunk 1 (offset 10), emit token 2 with duration 1 at
          // time 0.
          std::vector<float> logits(60, 0.0f);
          logits[10 + 2] = 1.0f;  // token 2
          logits[10 + 6] = 1.0f;  // duration 1
          auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
          EXPECT_TRUE(
              logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
          // Write marker to output state 0 to verify swap later.
          std::vector<float> state(10, 42.0f);
          auto& state_buf = const_cast<TensorBuffer&>(outputs[1]);
          EXPECT_TRUE(
              state_buf.Write<float>(absl::MakeConstSpan(state)).HasValue());
        } else if (run_step == 3) {
          EXPECT_EQ(signature, "decode_1");
          // Verify input state was swapped and now contains 42.0f.
          std::vector<float> state_in(10);
          auto& state_in_buf = const_cast<TensorBuffer&>(inputs[2]);
          EXPECT_TRUE(
              state_in_buf.Read<float>(absl::MakeSpan(state_in)).HasValue());
          EXPECT_FLOAT_EQ(state_in[0], 42.0f);

          // Write a different marker to output state to test blank non-swap.
          std::vector<float> state_out(10, 99.0f);
          auto& state_out_buf = const_cast<TensorBuffer&>(outputs[1]);
          EXPECT_TRUE(state_out_buf.Write<float>(absl::MakeConstSpan(state_out))
                          .HasValue());

          // Step 3: Emit blank token (0) with duration 1 at time 1 (offset 10).
          std::vector<float> logits(30, 0.0f);
          logits[10 + 0] = 1.0f;  // token 0 (blank)
          logits[10 + 6] = 1.0f;  // duration 1
          auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
          EXPECT_TRUE(
              logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        } else if (run_step == 4) {
          EXPECT_EQ(signature, "decode_1");
          // Step 4: Verify input state was NOT swapped on blank emission
          // (still 42.0f, not 99.0f).
          std::vector<float> state_in(10);
          auto& state_in_buf = const_cast<TensorBuffer&>(inputs[2]);
          EXPECT_TRUE(
              state_in_buf.Read<float>(absl::MakeSpan(state_in)).HasValue());
          EXPECT_FLOAT_EQ(state_in[0], 42.0f);

          // Emit blank token (0) with duration 1 at time 2 (offset 20).
          std::vector<float> logits(30, 0.0f);
          logits[20 + 0] = 1.0f;  // token 0 (blank)
          logits[20 + 6] = 1.0f;  // duration 1
          auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
          EXPECT_TRUE(
              logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        }
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(auto decoder,
                       TdtDecoder::Create(&mock_runner,
                                          /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024 * 3, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  EXPECT_EQ(run_step, 4);
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 0);
  EXPECT_EQ(tokens[1].token_id, 2);
  EXPECT_EQ(tokens[1].timestamp_ms, 0);
  EXPECT_TRUE(tokens[2].IsEndOfChunk());
  EXPECT_EQ(tokens[2].timestamp_ms, 3);
}

TEST(TdtDecoderTest, DecodeRangeValidFractionBoundsDecoding) {
  constexpr size_t kNumFrames = 8;
  constexpr size_t kLogitsPerFrame = 10;  // 5 token logits + 5 durations.
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        auto buf0 =
            CreateTestTensorBuffer(kLogitsPerFrame * kNumFrames, sizeof(float));
        // Every frame emits non-blank token (1 + frame % 4) with duration 1,
        // i.e. one token per decoded frame.
        std::vector<float> logits(kLogitsPerFrame * kNumFrames, 0.0f);
        for (size_t frame = 0; frame < kNumFrames; ++frame) {
          logits[frame * kLogitsPerFrame + 1 + frame % 4] = 1.0f;
          logits[frame * kLogitsPerFrame + 6] = 1.0f;
        }
        EXPECT_TRUE(buf0.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        buffers.push_back(std::move(buf0));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly(Return(absl::OkStatus()));

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(
      CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));

  // 25% of the window is real audio: round(0.25 * 8) = 2 valid frames are
  // decoded; the end-of-chunk token still marks the window end.
  ASSERT_OK_AND_ASSIGN(
      auto tokens,
      decoder->Decode(encoder_outputs,
                      PreprocessedAudio::DecodeRange{.valid_fraction = 0.25f}));
  constexpr size_t kExpectedFrames = 2;
  ASSERT_EQ(tokens.size(), kExpectedFrames + 1);
  EXPECT_EQ(tokens[kExpectedFrames - 1].timestamp_ms, kExpectedFrames - 1);
  EXPECT_TRUE(tokens.back().IsEndOfChunk());
  EXPECT_EQ(tokens.back().timestamp_ms, kNumFrames);

  // Omitting decode_range decodes the whole window.
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  EXPECT_EQ(tokens.size(), kNumFrames + 1);

  // Fractions outside (0, 1) mean "decode everything".
  ASSERT_OK_AND_ASSIGN(
      tokens, decoder->Decode(encoder_outputs, PreprocessedAudio::DecodeRange{
                                                   .valid_fraction = 1.0f}));
  EXPECT_EQ(tokens.size(), kNumFrames + 1);
  ASSERT_OK_AND_ASSIGN(
      tokens, decoder->Decode(encoder_outputs, PreprocessedAudio::DecodeRange{
                                                   .valid_fraction = 0.0f}));
  EXPECT_EQ(tokens.size(), kNumFrames + 1);

  // Frames are counted by their center: round(0.9 * 8) = round(7.2) = 7.
  ASSERT_OK_AND_ASSIGN(
      tokens, decoder->Decode(encoder_outputs, PreprocessedAudio::DecodeRange{
                                                   .valid_fraction = 0.9f}));
  EXPECT_EQ(tokens.size(), 7 + 1);

  // valid frames is capped at the window length.
  ASSERT_OK_AND_ASSIGN(
      tokens,
      decoder->Decode(encoder_outputs,
                      PreprocessedAudio::DecodeRange{
                          .valid_fraction = 0.95f}));  // round(7.6) = 8.
  EXPECT_EQ(tokens.size(), kNumFrames + 1);
}

TEST(TdtDecoderTest, DecodeRangeStartFractionDropsAndShiftsEarlyTokens) {
  constexpr size_t kNumFrames = 8;
  constexpr size_t kLogitsPerFrame = 10;  // 5 token logits + 5 durations.
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        auto buf0 =
            CreateTestTensorBuffer(kLogitsPerFrame * kNumFrames, sizeof(float));
        // Frame f emits non-blank token (1 + f % 4) with duration 1.
        std::vector<float> logits(kLogitsPerFrame * kNumFrames, 0.0f);
        for (size_t frame = 0; frame < kNumFrames; ++frame) {
          logits[frame * kLogitsPerFrame + 1 + frame % 4] = 1.0f;
          logits[frame * kLogitsPerFrame + 6] = 1.0f;
        }
        EXPECT_TRUE(buf0.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        buffers.push_back(std::move(buf0));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));
  int num_runs = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&num_runs](absl::string_view,
                                  absl::Span<const TensorBuffer>,
                                  absl::Span<const TensorBuffer>) {
        ++num_runs;
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(
      CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));

  // The first half of the window was already reported with the previous
  // chunk: floor(0.5 * 8) = 4 frames are still decoded (the prediction network
  // is autoregressive) but their tokens are dropped, and the remaining
  // timestamps, including the end-of-chunk marker, are relative to frame 4.
  ASSERT_OK_AND_ASSIGN(
      auto tokens,
      decoder->Decode(encoder_outputs,
                      PreprocessedAudio::DecodeRange{.start_fraction = 0.5f}));
  EXPECT_EQ(num_runs, kNumFrames);
  ASSERT_EQ(tokens.size(), 4 + 1);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tokens[i].token_id, 1 + (4 + i) % 4) << "i=" << i;
    EXPECT_EQ(tokens[i].timestamp_ms, i) << "i=" << i;
  }
  EXPECT_TRUE(tokens.back().IsEndOfChunk());
  EXPECT_EQ(tokens.back().timestamp_ms, kNumFrames - 4);

  // Omitting decode_range reports the whole window again.
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), kNumFrames + 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 0);
  EXPECT_EQ(tokens.back().timestamp_ms, kNumFrames);

  // Both ends at once: report frames [floor(0.25 * 8), round(0.75 * 8)) = [2,
  // 6).
  ASSERT_OK_AND_ASSIGN(
      tokens, decoder->Decode(encoder_outputs, PreprocessedAudio::DecodeRange{
                                                   .start_fraction = 0.25f,
                                                   .valid_fraction = 0.75f}));
  ASSERT_EQ(tokens.size(), 4 + 1);
  EXPECT_EQ(tokens[0].token_id, 1 + 2 % 4);
  EXPECT_EQ(tokens[0].timestamp_ms, 0);
  EXPECT_EQ(tokens[3].timestamp_ms, 3);
  EXPECT_EQ(tokens.back().timestamp_ms, kNumFrames - 2);
}

}  // namespace
}  // namespace litert::omni::asr
