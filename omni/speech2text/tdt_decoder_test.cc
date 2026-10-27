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

#include "omni/speech2text/tdt_decoder.h"

#include <algorithm>
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
#include "omni/base/mock_litert_runner.h"
#include "omni/speech2text/audio_preprocessor.h"
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
        buffers.push_back(CreateTestTensorBuffer(1024 * 4, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(2, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(80, sizeof(float)));
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
        buffers.push_back(CreateTestTensorBuffer(40, sizeof(float)));
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
          std::vector<float> logits(80, 0.0f);
          logits[1] = 1.0f;  // token 1
          logits[5] = 1.0f;  // duration 0
          auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
          EXPECT_TRUE(
              logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        } else if (run_step == 2) {
          EXPECT_EQ(signature, "decode");
          // Step 2: In chunk 1 (offset 10), emit token 2 with duration 1 at
          // time 0.
          std::vector<float> logits(80, 0.0f);
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

          // Step 3: Emit blank token (0) with duration 1 at time 1 (offset 10)
          // and non-blank token 1 with duration 1 at time 2 (offset 20). Time 1
          // must not CommitState(), and time 2 reuses these cached logits
          // without re-running decode_1 and then commits the 99.0f state once.
          std::vector<float> logits(40, 0.0f);
          logits[10 + 0] = 1.0f;  // token 0 (blank) at time 1
          logits[10 + 6] = 1.0f;  // duration 1
          logits[20 + 1] = 1.0f;  // token 1 (non-blank) at time 2
          logits[20 + 6] = 1.0f;  // duration 1
          auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
          EXPECT_TRUE(
              logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());

          std::vector<float> state(10, 99.0f);
          auto& state_buf = const_cast<TensorBuffer&>(outputs[1]);
          EXPECT_TRUE(
              state_buf.Write<float>(absl::MakeConstSpan(state)).HasValue());
        } else if (run_step == 4) {
          EXPECT_EQ(signature, "decode_1");
          // Verify input state was committed once on the non-blank at time 2
          // (not on the blank at time 1) and now contains 99.0f.
          std::vector<float> state_in(10);
          auto& state_in_buf = const_cast<TensorBuffer&>(inputs[2]);
          EXPECT_TRUE(
              state_in_buf.Read<float>(absl::MakeSpan(state_in)).HasValue());
          EXPECT_FLOAT_EQ(state_in[0], 99.0f);

          // Step 4: Emit blank token (0) with duration 1 at time 3 (offset 30).
          std::vector<float> logits(40, 0.0f);
          logits[30 + 0] = 1.0f;  // token 0 (blank) at time 3
          logits[30 + 6] = 1.0f;  // duration 1
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
  encoder_outputs.push_back(CreateTestTensorBuffer(1024 * 4, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  EXPECT_EQ(run_step, 4);
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 0);
  EXPECT_EQ(tokens[1].token_id, 2);
  EXPECT_EQ(tokens[1].timestamp_ms, 0);
  EXPECT_EQ(tokens[2].token_id, 1);
  EXPECT_EQ(tokens[2].timestamp_ms, 2);
  EXPECT_TRUE(tokens[3].IsEndOfChunk());
  EXPECT_EQ(tokens[3].timestamp_ms, 4);
}

TEST(
    TdtDecoderTest,
    ZeroDurationNonBlankLoopRollsBackFrameBurstAndStopsAfterMaxSymbolsPerStep) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 2, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(20, sizeof(float)));
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
        buffers.push_back(CreateTestTensorBuffer(20, sizeof(float)));
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
        if (run_step > 10) {
          return absl::ResourceExhaustedError(
              "Infinite decode loop on zero-duration non-blank token");
        }
        std::vector<float> logits(20, 0.0f);
        if (run_step == 1) {
          // Frame 0 emits token 1 with duration 1, advancing to frame 1.
          logits[1] = 1.0f;
          logits[6] = 1.0f;
        } else {
          // Frame 1 loops on non-blank token 3 with duration 0.
          logits[10 + 3] = 1.0f;
          logits[10 + 5] = 1.0f;
        }
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(auto decoder,
                       TdtDecoder::Create(&mock_runner,
                                          /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024 * 2, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  // 1 step at frame 0 + (max_symbols_per_step=3 + 1 overflow step) at frame 1.
  // The 3 zero-duration tokens at frame 1 are rolled back; token 1 at frame 0
  // is preserved.
  EXPECT_EQ(run_step, 5);
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 0);
  EXPECT_TRUE(tokens[1].IsEndOfChunk());
  EXPECT_EQ(tokens[1].timestamp_ms, 2);
}

TEST(TdtDecoderTest, CreateWithNonPositiveMaxSymbolsPerStepFails) {
  MockLiteRtRunner mock_runner;
  EXPECT_FALSE(TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0},
                                  /*decode_statefully_after=*/4,
                                  /*max_symbols_per_step=*/0)
                   .ok());
  EXPECT_FALSE(TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0},
                                  /*decode_statefully_after=*/4,
                                  /*max_symbols_per_step=*/-1)
                   .ok());
}

TEST(TdtDecoderTest, CreateWithInsufficientOutputBuffersFails) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 1, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });

  EXPECT_FALSE(TdtDecoder::Create(&mock_runner).ok());
}

TEST(TdtDecoderTest, DecodeAdvancesTimeIndexOnZeroDurationBlank) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 2, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        auto buf0 = CreateTestTensorBuffer(20, sizeof(float));
        // Both frames emit blank token (0) with duration 0 (index 5).
        std::vector<float> logits(20, 0.0f);
        logits[0] = 1.0f;
        logits[5] = 1.0f;
        logits[10 + 0] = 1.0f;
        logits[10 + 5] = 1.0f;
        EXPECT_TRUE(buf0.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        buffers.push_back(std::move(buf0));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024 * 2, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsEndOfChunk());
  EXPECT_EQ(tokens[0].timestamp_ms, 2);
}

TEST(TdtDecoderTest,
     StatelessDecodeStopsWhenTokenBufferIsFullWithoutStatefulRunner) {
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
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  int run_step = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&run_step](absl::string_view,
                                  absl::Span<const TensorBuffer>,
                                  absl::Span<const TensorBuffer> outputs) {
        ++run_step;
        std::vector<float> logits(60, 0.0f);
        if (run_step == 1) {
          logits[1] = 1.0f;  // token 1 at token_index 0, frame 0
          logits[5] = 1.0f;  // duration 0
        } else if (run_step == 2) {
          logits[10 + 2] = 1.0f;  // token 2 at token_index 1, frame 0
          logits[10 + 5] = 1.0f;  // duration 0
        }
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024 * 3, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  EXPECT_EQ(run_step, 2);
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[1].token_id, 2);
  EXPECT_TRUE(tokens[2].IsEndOfChunk());
}

TEST(TdtDecoderTest,
     CustomMaxSymbolsPerStepResetsWhenPositiveDurationAdvances) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 2, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(20, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  int run_step = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&run_step](absl::string_view,
                                  absl::Span<const TensorBuffer>,
                                  absl::Span<const TensorBuffer> outputs) {
        ++run_step;
        std::vector<float> logits(20, 0.0f);
        if (run_step == 1) {
          // Frame 0, step 1: emit token 1 with duration 0.
          logits[1] = 1.0f;
          logits[5] = 1.0f;
        } else if (run_step == 2) {
          // Frame 0, step 2: emit token 2 with duration 1 (advances to frame
          // 1 and resets per-frame symbol counter).
          logits[2] = 1.0f;
          logits[6] = 1.0f;
        } else if (run_step == 3) {
          // Frame 1, step 1: emit token 3 with duration 0.
          logits[10 + 3] = 1.0f;
          logits[10 + 5] = 1.0f;
        } else {
          // Frame 1, step 2: emit token 3 with duration 1 (2nd symbol at frame
          // 1 <= max_symbols_per_step = 2, then advances to frame 2).
          logits[10 + 3] = 1.0f;
          logits[10 + 6] = 1.0f;
        }
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0},
                         /*decode_statefully_after=*/4,
                         /*max_symbols_per_step=*/2));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024 * 2, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  EXPECT_EQ(run_step, 4);
  ASSERT_EQ(tokens.size(), 5);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 0);
  EXPECT_EQ(tokens[1].token_id, 2);
  EXPECT_EQ(tokens[1].timestamp_ms, 0);
  EXPECT_EQ(tokens[2].token_id, 3);
  EXPECT_EQ(tokens[2].timestamp_ms, 1);
  EXPECT_EQ(tokens[3].token_id, 3);
  EXPECT_EQ(tokens[3].timestamp_ms, 1);
  EXPECT_TRUE(tokens[4].IsEndOfChunk());
  EXPECT_EQ(tokens[4].timestamp_ms, 2);
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

TEST(TdtDecoderTest, PrematureSosZeroDurationBurstResetsToSos) {
  constexpr size_t kNumFrames = 4;
  constexpr size_t kNumTokenSlots = 4;
  constexpr size_t kLogitsPerSlot = 10;
  constexpr size_t kLogitsPerFrame = kNumTokenSlots * kLogitsPerSlot;
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
        buffers.push_back(
            CreateTestTensorBuffer(kNumTokenSlots, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(kLogitsPerFrame * kNumFrames,
                                                 sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([](absl::string_view,
                         absl::Span<const TensorBuffer> inputs,
                         absl::Span<const TensorBuffer> outputs) {
        std::vector<float> logits(kLogitsPerFrame * kNumFrames, 0.0f);
        // Frame 0 slot 0: emit premature token 1 with duration 1.
        logits[0 * kLogitsPerFrame + 0 * kLogitsPerSlot + 1] = 10.0f;
        logits[0 * kLogitsPerFrame + 0 * kLogitsPerSlot + 6] = 1.0f;
        // Frame 1 slots 1..3: zero-duration burst of token 3 (exceeding
        // max_symbols_per_step = 2), triggering reset_to_sos().
        for (size_t s = 1; s < kNumTokenSlots; ++s) {
          const size_t base = 1 * kLogitsPerFrame + s * kLogitsPerSlot;
          logits[base + 3] = 10.0f;
          logits[base + 5] = 1.0f;  // duration 0
        }
        // Frame 2 slot 0 (after reset to <sos>): emit valid token 2 with
        // duration 2.
        logits[2 * kLogitsPerFrame + 0 * kLogitsPerSlot + 2] = 10.0f;
        logits[2 * kLogitsPerFrame + 0 * kLogitsPerSlot + 7] = 1.0f;
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0},
                         /*decode_statefully_after=*/4,
                         /*max_symbols_per_step=*/2));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(
      CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 2);
  EXPECT_EQ(tokens[0].timestamp_ms, 2);
  EXPECT_TRUE(tokens[1].IsEndOfChunk());
}

TEST(TdtDecoderTest,
     NearThresholdSosTokenSelectsLocalMinimumWithoutLookaheadWinner) {
  constexpr size_t kNumFrames = 50;
  constexpr size_t kNumTokenSlots = 2;
  constexpr size_t kLogitsPerSlot = 10;  // 5 token logits + 5 durations.
  constexpr size_t kLogitsPerFrame = kNumTokenSlots * kLogitsPerSlot;
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
        buffers.push_back(
            CreateTestTensorBuffer(kNumTokenSlots, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(kLogitsPerFrame * kNumFrames,
                                                 sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  std::vector<float> active_logits(kLogitsPerFrame * kNumFrames, 0.0f);
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&active_logits](absl::string_view,
                                       absl::Span<const TensorBuffer>,
                                       absl::Span<const TensorBuffer> outputs) {
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(logits_buf.Write<float>(absl::MakeConstSpan(active_logits))
                        .HasValue());
        return absl::OkStatus();
      });

  auto init_all_blank = [&]() {
    std::fill(active_logits.begin(), active_logits.end(), 0.0f);
    for (size_t f = 0; f < kNumFrames; ++f) {
      for (size_t s = 0; s < kNumTokenSlots; ++s) {
        const size_t base = f * kLogitsPerFrame + s * kLogitsPerSlot;
        active_logits[base + 0] = 10.0f;  // blank = 10, non-blank = 0 (gap 10)
        active_logits[base + 6] = 1.0f;   // duration = 1
      }
    }
  };

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(
      CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));

  // Case 1: gap > 3.0 at t=31 (blank=10.0, tok1=6.5 -> gap=3.5) stays blank.
  init_all_blank();
  active_logits[31 * kLogitsPerFrame + 1] = 6.5f;
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsEndOfChunk());

  // Case 2: gap <= 3.0 at local minimum t=31 (blank=10.0, tok1=8.0 -> gap=2.0,
  // t=32 has gap=2.5), but t=34 inside the 12-frame lookahead has a natural
  // non-blank winner (blank=4.0, tok2=9.0 -> gap=-5.0): t=31 stays blank and
  // only the natural winner at t=34 is emitted.
  init_all_blank();
  active_logits[31 * kLogitsPerFrame + 1] = 8.0f;
  active_logits[32 * kLogitsPerFrame + 1] = 7.5f;
  active_logits[34 * kLogitsPerFrame + 0] = 4.0f;
  active_logits[34 * kLogitsPerFrame + 2] = 9.0f;
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 2);
  EXPECT_EQ(tokens[0].timestamp_ms, 34);
  EXPECT_TRUE(tokens[1].IsEndOfChunk());

  // Case 3: gap <= 3.0 at local minimum t=31 (blank=10.0, tok3=8.0 -> gap=2.0,
  // t=32 has gap=2.5) with no natural non-blank winner in the next 12 frames:
  // best non-blank token (tok3) is emitted at t=31.
  init_all_blank();
  active_logits[31 * kLogitsPerFrame + 3] = 8.0f;
  active_logits[32 * kLogitsPerFrame + 3] = 7.5f;
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 3);
  EXPECT_EQ(tokens[0].timestamp_ms, 31);
  EXPECT_TRUE(tokens[1].IsEndOfChunk());
}

TEST(TdtDecoderTest,
     MidChunkDeadZoneResetsOnceRewindsAndHandlesPrefixRetention) {
  constexpr size_t kNumFrames = 55;
  constexpr size_t kNumTokenSlots = 8;
  constexpr size_t kLogitsPerSlot = 10;
  constexpr size_t kLogitsPerFrame = kNumTokenSlots * kLogitsPerSlot;
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
        buffers.push_back(
            CreateTestTensorBuffer(kNumTokenSlots, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(kLogitsPerFrame * kNumFrames,
                                                 sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  auto set_frame_all_slots = [&](std::vector<float>& logits, size_t frame,
                                 int token_id, int duration) {
    for (size_t s = 0; s < kNumTokenSlots; ++s) {
      const size_t base = frame * kLogitsPerFrame + s * kLogitsPerSlot;
      std::fill_n(logits.begin() + base, kLogitsPerSlot, 0.0f);
      logits[base + token_id] = 10.0f;
      logits[base + 5 + duration] = 1.0f;
    }
  };

  size_t early_token_frame = 10;
  bool saw_non_blank_since_sos = false;
  int reset_count = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs) {
        std::vector<int32_t> token_ids(kNumTokenSlots, 0);
        auto& tok_buf = const_cast<TensorBuffer&>(inputs[1]);
        EXPECT_TRUE(
            tok_buf.Read<int32_t>(absl::MakeSpan(token_ids)).HasValue());
        if (token_ids[1] != 0) {
          saw_non_blank_since_sos = true;
        } else if (saw_non_blank_since_sos) {
          ++reset_count;
          saw_non_blank_since_sos = false;
        }

        std::vector<float> logits(kLogitsPerFrame * kNumFrames, 0.0f);
        for (size_t f = 0; f < kNumFrames; ++f) {
          set_frame_all_slots(logits, f, /*token_id=*/0, /*duration=*/1);
        }
        if (reset_count == 0) {
          // Before reset: emit token 1 at `early_token_frame`, then blanks.
          set_frame_all_slots(logits, early_token_frame, /*token_id=*/1,
                              /*duration=*/1);
        } else {
          // After reset (rewound to `early_token_frame + 3`): emit 5 tokens
          // spanning > 4 frames so `TrimAbortiveMidChunkSosReset` keeps them,
          // followed by >= 16 blank frames (a second dead zone).
          const size_t rewind_frame = early_token_frame + 3;
          set_frame_all_slots(logits, rewind_frame, /*token_id=*/2,
                              /*duration=*/1);
          set_frame_all_slots(logits, rewind_frame + 1, /*token_id=*/2,
                              /*duration=*/1);
          set_frame_all_slots(logits, rewind_frame + 2, /*token_id=*/2,
                              /*duration=*/1);
          set_frame_all_slots(logits, rewind_frame + 3, /*token_id=*/2,
                              /*duration=*/3);
          set_frame_all_slots(logits, rewind_frame + 6, /*token_id=*/2,
                              /*duration=*/1);
        }
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(
      CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));

  // Case 1: early emission at t=10 (> 8) -> prefix token at t=10 is kept,
  // reset fires once at t=26 (>= 25 and >= 10+16), rewinds to t=13 (10+3),
  // and the second dead zone after t=19 does not reset again.
  early_token_frame = 10;
  saw_non_blank_since_sos = false;
  reset_count = 0;
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  EXPECT_EQ(reset_count, 1);
  ASSERT_EQ(tokens.size(), 7);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 10);
  EXPECT_EQ(tokens[1].token_id, 2);
  EXPECT_EQ(tokens[1].timestamp_ms, 13);
  EXPECT_EQ(tokens[5].token_id, 2);
  EXPECT_EQ(tokens[5].timestamp_ms, 19);
  EXPECT_TRUE(tokens[6].IsEndOfChunk());

  // Case 2: early emission at t=5 (<= 8) -> prefix token at t=5 is cleared on
  // reset, time rewinds to t=8 (5+3), and only the post-reset tokens remain.
  early_token_frame = 5;
  saw_non_blank_since_sos = false;
  reset_count = 0;
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  EXPECT_EQ(reset_count, 1);
  ASSERT_EQ(tokens.size(), 6);
  EXPECT_EQ(tokens[0].token_id, 2);
  EXPECT_EQ(tokens[0].timestamp_ms, 8);
  EXPECT_EQ(tokens[4].token_id, 2);
  EXPECT_EQ(tokens[4].timestamp_ms, 14);
  EXPECT_TRUE(tokens[5].IsEndOfChunk());
}

TEST(TdtDecoderTest,
     TrimAbortiveMidChunkSosResetDropsShortOrLatePostResetTail) {
  constexpr size_t kNumFrames = 55;
  constexpr size_t kNumTokenSlots = 8;
  constexpr size_t kLogitsPerSlot = 10;
  constexpr size_t kLogitsPerFrame = kNumTokenSlots * kLogitsPerSlot;
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
        buffers.push_back(
            CreateTestTensorBuffer(kNumTokenSlots, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(kLogitsPerFrame * kNumFrames,
                                                 sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  auto set_frame_all_slots = [&](std::vector<float>& logits, size_t frame,
                                 int token_id, int duration) {
    for (size_t s = 0; s < kNumTokenSlots; ++s) {
      const size_t base = frame * kLogitsPerFrame + s * kLogitsPerSlot;
      std::fill_n(logits.begin() + base, kLogitsPerSlot, 0.0f);
      logits[base + token_id] = 10.0f;
      logits[base + 5 + duration] = 1.0f;
    }
  };

  std::vector<size_t> post_reset_token_frames;
  bool saw_prefix_token = false;
  bool after_reset = false;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs) {
        std::vector<int32_t> token_ids(kNumTokenSlots, 0);
        auto& tok_buf = const_cast<TensorBuffer&>(inputs[1]);
        EXPECT_TRUE(
            tok_buf.Read<int32_t>(absl::MakeSpan(token_ids)).HasValue());
        if (token_ids[1] != 0 && !after_reset) {
          saw_prefix_token = true;
        } else if (saw_prefix_token && token_ids[1] == 0) {
          after_reset = true;
        }

        std::vector<float> logits(kLogitsPerFrame * kNumFrames, 0.0f);
        for (size_t f = 0; f < kNumFrames; ++f) {
          set_frame_all_slots(logits, f, /*token_id=*/0, /*duration=*/1);
        }
        if (!after_reset) {
          // Emit prefix token 1 at t=9 (> 8, so prefix is kept on reset).
          set_frame_all_slots(logits, /*frame=*/9, /*token_id=*/1,
                              /*duration=*/1);
        } else {
          for (size_t f : post_reset_token_frames) {
            set_frame_all_slots(logits, f, /*token_id=*/2, /*duration=*/1);
          }
        }
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0}));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(
      CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));

  // Case 1: post-reset tail of <= 4 tokens (even spanning > 4 frames) is
  // trimmed.
  saw_prefix_token = false;
  after_reset = false;
  post_reset_token_frames = {12, 14, 16, 18};
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 9);
  EXPECT_TRUE(tokens[1].IsEndOfChunk());

  // Case 2: post-reset tail of > 4 tokens but spanning <= 4 frames (t=12..16)
  // is trimmed.
  saw_prefix_token = false;
  after_reset = false;
  post_reset_token_frames = {12, 13, 14, 15, 16};
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 9);
  EXPECT_TRUE(tokens[1].IsEndOfChunk());

  // Case 3: post-reset tail of > 4 tokens spanning > 4 frames, but starting
  // > 20 frames after reset_origin_time_index=9 (t=30 > 29), is trimmed.
  saw_prefix_token = false;
  after_reset = false;
  post_reset_token_frames = {30, 32, 34, 36, 38};
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 9);
  EXPECT_TRUE(tokens[1].IsEndOfChunk());
}

TEST(TdtDecoderTest,
     PostBurstZeroDurationTokensAreSuppressedUntilPositiveDuration) {
  constexpr size_t kNumFrames = 50;
  constexpr size_t kNumTokenSlots = 8;
  constexpr size_t kLogitsPerSlot = 10;
  constexpr size_t kLogitsPerFrame = kNumTokenSlots * kLogitsPerSlot;
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
        buffers.push_back(
            CreateTestTensorBuffer(kNumTokenSlots, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(kLogitsPerFrame * kNumFrames,
                                                 sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  auto set_frame_all_slots = [&](std::vector<float>& logits, size_t frame,
                                 int token_id, int duration) {
    for (size_t s = 0; s < kNumTokenSlots; ++s) {
      const size_t base = frame * kLogitsPerFrame + s * kLogitsPerSlot;
      std::fill_n(logits.begin() + base, kLogitsPerSlot, 0.0f);
      logits[base + token_id] = 10.0f;
      logits[base + 5 + duration] = 1.0f;
    }
  };

  int run_step = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&](absl::string_view, absl::Span<const TensorBuffer>,
                          absl::Span<const TensorBuffer> outputs) {
        ++run_step;
        std::vector<float> logits(kLogitsPerFrame * kNumFrames, 0.0f);
        for (size_t f = 0; f < kNumFrames; ++f) {
          set_frame_all_slots(logits, f, /*token_id=*/0, /*duration=*/4);
        }
        if (run_step <= 3) {
          // Steps 1..3 (t=0, 1, 2): emit 3 tokens (tokens_since_sos = 3 > 2,
          // so a subsequent burst is non-premature).
          set_frame_all_slots(logits, static_cast<size_t>(run_step - 1),
                              /*token_id=*/1, /*duration=*/1);
        } else if (run_step <= 6) {
          // Steps 4..6 (t=3): zero-duration burst of 3 tokens (>
          // max_symbols_per_step = 2), rolling back t=3 and setting
          // prev_frame_had_burst = true.
          set_frame_all_slots(logits, /*frame=*/3, /*token_id=*/4,
                              /*duration=*/0);
        } else if (run_step == 7) {
          // Step 7 (t=4, first step after burst): dur=0 token 2 is suppressed
          // from decoded_tokens because prev_frame_had_burst is true.
          set_frame_all_slots(logits, /*frame=*/4, /*token_id=*/2,
                              /*duration=*/0);
        } else if (run_step == 8) {
          // Step 8 (t=4, second step): dur=1 token 3 is kept and clears
          // prev_frame_had_burst.
          set_frame_all_slots(logits, /*frame=*/4, /*token_id=*/3,
                              /*duration=*/1);
        } else if (run_step == 9) {
          // Step 9 (t=5): dur=0 token 2 is now kept because the dur>0 token at
          // t=4 cleared prev_frame_had_burst (and fills the 8th stateless slot,
          // ending decoding).
          set_frame_all_slots(logits, /*frame=*/5, /*token_id=*/2,
                              /*duration=*/0);
        }
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(
            logits_buf.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0},
                         /*decode_statefully_after=*/4,
                         /*max_symbols_per_step=*/2));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(
      CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 6);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 0);
  EXPECT_EQ(tokens[1].token_id, 1);
  EXPECT_EQ(tokens[1].timestamp_ms, 1);
  EXPECT_EQ(tokens[2].token_id, 1);
  EXPECT_EQ(tokens[2].timestamp_ms, 2);
  EXPECT_EQ(tokens[3].token_id, 3);
  EXPECT_EQ(tokens[3].timestamp_ms, 4);
  EXPECT_EQ(tokens[4].token_id, 2);
  EXPECT_EQ(tokens[4].timestamp_ms, 5);
  EXPECT_TRUE(tokens[5].IsEndOfChunk());
  EXPECT_EQ(tokens[5].timestamp_ms, kNumFrames);
}

TEST(TdtDecoderTest,
     ForcedSosBurstRollsBackStatelessDumpAndMultiFrameHallucination) {
  constexpr size_t kNumFrames = 55;
  constexpr size_t kNumTokenSlots = 12;
  constexpr size_t kLogitsPerSlot = 10;
  constexpr size_t kLogitsPerFrame = kNumTokenSlots * kLogitsPerSlot;
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(
            CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));
        buffers.push_back(
            CreateTestTensorBuffer(kNumTokenSlots, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(kLogitsPerFrame * kNumFrames,
                                                 sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode_1"))
      .WillOnce(Return(absl::NotFoundError("No decode_1 signature")));

  auto set_slot = [&](std::vector<float>& logits, size_t frame, size_t slot,
                      float blank_logit, int non_blank_id,
                      float non_blank_logit, int duration) {
    const size_t base = frame * kLogitsPerFrame + slot * kLogitsPerSlot;
    std::fill_n(logits.begin() + base, kLogitsPerSlot, 0.0f);
    logits[base + 0] = blank_logit;
    if (non_blank_id > 0) {
      logits[base + non_blank_id] = non_blank_logit;
    }
    logits[base + 5 + duration] = 10.0f;
  };

  std::vector<float> active_logits(kLogitsPerFrame * kNumFrames, 0.0f);
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&](absl::string_view, absl::Span<const TensorBuffer>,
                          absl::Span<const TensorBuffer> outputs) {
        auto& logits_buf = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(logits_buf.Write<float>(absl::MakeConstSpan(active_logits))
                        .HasValue());
        return absl::OkStatus();
      });

  auto init_all_blank = [&]() {
    for (size_t f = 0; f < kNumFrames; ++f) {
      for (size_t s = 0; s < kNumTokenSlots; ++s) {
        set_slot(active_logits, f, s, /*blank_logit=*/10.0f,
                 /*non_blank_id=*/0, /*non_blank_logit=*/0.0f, /*duration=*/1);
      }
    }
  };

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      TdtDecoder::Create(&mock_runner, /*decode_start_token_ids=*/{0},
                         /*decode_statefully_after=*/4,
                         /*max_symbols_per_step=*/3));
  std::vector<TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(
      CreateTestTensorBuffer(1024 * kNumFrames, sizeof(float)));

  // Case 1: Forced near-threshold <sos> at t=10 (gap=2.0 <= 3.0) with d=1,
  // followed at t=11 by 3 tokens in slots 1..3 on the same frame (d=0, 0, 1).
  // Because sos_was_forced && tokens_since_sos == 1, the 3rd symbol on t=11
  // triggers RollbackFrameBurstAndAdvance and resets to <sos>, allowing the
  // natural speech token at t=25 to be decoded from slot 0.
  init_all_blank();
  set_slot(active_logits, /*frame=*/10, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/1, /*non_blank_logit=*/8.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/11, /*slot=*/1, /*blank_logit=*/0.0f,
           /*non_blank_id=*/2, /*non_blank_logit=*/10.0f, /*duration=*/0);
  set_slot(active_logits, /*frame=*/11, /*slot=*/2, /*blank_logit=*/0.0f,
           /*non_blank_id=*/3, /*non_blank_logit=*/10.0f, /*duration=*/0);
  set_slot(active_logits, /*frame=*/11, /*slot=*/3, /*blank_logit=*/0.0f,
           /*non_blank_id=*/4, /*non_blank_logit=*/10.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/25, /*slot=*/0, /*blank_logit=*/0.0f,
           /*non_blank_id=*/2, /*non_blank_logit=*/10.0f, /*duration=*/4);

  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 2);
  EXPECT_EQ(tokens[0].timestamp_ms, 25);
  EXPECT_TRUE(tokens[1].IsEndOfChunk());

  // Case 2: Forced near-threshold <sos> at t=12 (> kDeadZoneClearPrefixMaxFrame
  // = 8) emitting 2 tokens across t=12, 13 followed by a >= 16-frame dead zone,
  // and a late forced <sos> at t=48 that bursts at end-of-chunk (t=53). Both
  // forced false starts are cleared.
  init_all_blank();
  set_slot(active_logits, /*frame=*/12, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/1, /*non_blank_logit=*/8.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/13, /*slot=*/1, /*blank_logit=*/0.0f,
           /*non_blank_id=*/2, /*non_blank_logit=*/10.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/48, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/1, /*non_blank_logit=*/8.0f, /*duration=*/2);
  set_slot(active_logits, /*frame=*/50, /*slot=*/1, /*blank_logit=*/0.0f,
           /*non_blank_id=*/2, /*non_blank_logit=*/10.0f, /*duration=*/3);
  for (size_t s = 2; s < kNumTokenSlots; ++s) {
    set_slot(active_logits, /*frame=*/53, /*slot=*/s, /*blank_logit=*/0.0f,
             /*non_blank_id=*/3, /*non_blank_logit=*/10.0f, /*duration=*/0);
  }

  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsEndOfChunk());

  // Case 3: A forced <sos> at t=10 is followed by 6 more emission frames
  // (t=11..16), i.e. 7 > kMaxPrematureForcedSosFrames emission frames, before
  // a burst at t=17. The segment is no longer a premature false start, so only
  // the burst frame is removed and the 7 earlier tokens are kept.
  init_all_blank();
  set_slot(active_logits, /*frame=*/10, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/1, /*non_blank_logit=*/8.0f, /*duration=*/1);
  for (size_t f = 11; f <= 16; ++f) {
    set_slot(active_logits, f, /*slot=*/f - 10, /*blank_logit=*/0.0f,
             /*non_blank_id=*/2, /*non_blank_logit=*/10.0f, /*duration=*/1);
  }
  for (size_t s = 7; s < 11; ++s) {
    set_slot(active_logits, /*frame=*/17, s, /*blank_logit=*/0.0f,
             /*non_blank_id=*/4, /*non_blank_logit=*/10.0f, /*duration=*/0);
  }

  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 8);
  EXPECT_EQ(tokens[0].token_id, 1);
  EXPECT_EQ(tokens[0].timestamp_ms, 10);
  for (size_t i = 1; i <= 6; ++i) {
    EXPECT_EQ(tokens[i].token_id, 2);
    EXPECT_EQ(tokens[i].timestamp_ms, 10 + i);
  }
  EXPECT_TRUE(tokens[7].IsEndOfChunk());

  // Case 4: 7 unforced emission frames at t=1..7 are cleared by a mid-chunk
  // dead-zone reset, which also resets the emission-frame count. A later
  // forced <sos> at t=40 with 3 emission frames (t=40..42) is then a premature
  // false start, so the burst at t=43 removes all 3 tokens instead of only the
  // burst frame.
  init_all_blank();
  for (size_t f = 1; f <= 7; ++f) {
    set_slot(active_logits, f, /*slot=*/f - 1, /*blank_logit=*/0.0f,
             /*non_blank_id=*/2, /*non_blank_logit=*/10.0f, /*duration=*/1);
  }
  set_slot(active_logits, /*frame=*/40, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/1, /*non_blank_logit=*/8.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/41, /*slot=*/1, /*blank_logit=*/0.0f,
           /*non_blank_id=*/2, /*non_blank_logit=*/10.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/42, /*slot=*/2, /*blank_logit=*/0.0f,
           /*non_blank_id=*/3, /*non_blank_logit=*/10.0f, /*duration=*/1);
  for (size_t s = 3; s < 7; ++s) {
    set_slot(active_logits, /*frame=*/43, s, /*blank_logit=*/0.0f,
             /*non_blank_id=*/4, /*non_blank_logit=*/10.0f, /*duration=*/0);
  }

  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsEndOfChunk());

  // Case 5: An initial false-start prefix at t=2 is cleared by a mid-chunk
  // dead-zone reset (`cleared_false_start_prefix == true`), and a second forced
  // <sos> at t=32 emits 2 tokens (`<= kMaxPrematureSosTokens`) before stalling
  // in a second >= 16-frame dead zone through the end of the chunk (t=34..54).
  // `TrimAbortiveMidChunkSosReset` clears the post-reset forced false start.
  init_all_blank();
  set_slot(active_logits, /*frame=*/2, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/1, /*non_blank_logit=*/8.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/32, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/2, /*non_blank_logit=*/8.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/33, /*slot=*/1, /*blank_logit=*/0.0f,
           /*non_blank_id=*/3, /*non_blank_logit=*/10.0f, /*duration=*/1);

  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsEndOfChunk());

  // Case 6: When the post-reset forced <sos> segment ends < 16 frames before
  // `effective_max_time_index` (t=40..41, leaving 14 < 16 blank frames), it has
  // not stalled in a second dead zone and is preserved.
  init_all_blank();
  set_slot(active_logits, /*frame=*/2, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/1, /*non_blank_logit=*/8.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/40, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/2, /*non_blank_logit=*/8.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/41, /*slot=*/1, /*blank_logit=*/0.0f,
           /*non_blank_id=*/3, /*non_blank_logit=*/10.0f, /*duration=*/1);

  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0].token_id, 2);
  EXPECT_EQ(tokens[0].timestamp_ms, 40);
  EXPECT_EQ(tokens[1].token_id, 3);
  EXPECT_EQ(tokens[1].timestamp_ms, 41);
  EXPECT_TRUE(tokens[2].IsEndOfChunk());

  // Case 7: On a shifted-back final chunk (`first_reported_time_index == 20`),
  // pre-reset tokens at t=2 occurred before `first_reported_time_index` so
  // `decoded_tokens` was already empty (`cleared_false_start_prefix == false`,
  // `sos_tokens_start == 0`). A low-confidence trailing word after the reset
  // (t=32..33) is preserved rather than erased.
  init_all_blank();
  set_slot(active_logits, /*frame=*/2, /*slot=*/0, /*blank_logit=*/0.0f,
           /*non_blank_id=*/1, /*non_blank_logit=*/10.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/32, /*slot=*/0, /*blank_logit=*/10.0f,
           /*non_blank_id=*/2, /*non_blank_logit=*/8.0f, /*duration=*/1);
  set_slot(active_logits, /*frame=*/33, /*slot=*/1, /*blank_logit=*/0.0f,
           /*non_blank_id=*/3, /*non_blank_logit=*/10.0f, /*duration=*/1);

  ASSERT_OK_AND_ASSIGN(
      tokens,
      decoder->Decode(encoder_outputs,
                      PreprocessedAudio::DecodeRange{
                          .start_fraction = 20.0f / 55.0f,
                          .valid_fraction = 1.0f}));
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0].token_id, 2);
  EXPECT_EQ(tokens[0].timestamp_ms, 12);
  EXPECT_EQ(tokens[1].token_id, 3);
  EXPECT_EQ(tokens[1].timestamp_ms, 13);
  EXPECT_TRUE(tokens[2].IsEndOfChunk());
}

}  // namespace
}  // namespace litert::omni::asr
