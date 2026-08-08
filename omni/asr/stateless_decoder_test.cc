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

#include "omni/asr/stateless_decoder.h"

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
#include "omni/base/mock_litert_runner.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::_;
using ::testing::Return;

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

TEST(StatelessDecoderTest, CreateWhenCreateInputBuffersFailsReturnsError) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce(Return(absl::InternalError("Failed to create input buffers")));

  EXPECT_FALSE(StatelessDecoder::Create(&mock_runner).ok());
}

TEST(StatelessDecoderTest, CreateWithInsufficientInputBuffersFails) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(int32_t)));
        return buffers;
      });

  EXPECT_FALSE(StatelessDecoder::Create(&mock_runner).ok());
}

TEST(StatelessDecoderTest, CreateSuccessfully) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });

  ASSERT_OK_AND_ASSIGN(auto decoder, StatelessDecoder::Create(&mock_runner));
  EXPECT_NE(decoder, nullptr);
}

TEST(StatelessDecoderTest, DecodeFailsWhenRunnerRunFails) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillOnce(Return(absl::InternalError("Run failed")));

  ASSERT_OK_AND_ASSIGN(auto decoder, StatelessDecoder::Create(&mock_runner));
  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));
  auto tokens_or = decoder->Decode(encoder_outputs);
  EXPECT_FALSE(tokens_or.ok());
}

TEST(StatelessDecoderTest, DecodeWithDecodeStartTokenId) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(4, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(20, sizeof(float)));
        return buffers;
      });

  constexpr int kStartTokenId = 42;
  int32_t first_input_start_token = -1;

  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&first_input_start_token](
                          absl::string_view,
                          absl::Span<const ::litert::TensorBuffer> inputs,
                          absl::Span<const ::litert::TensorBuffer> outputs)
                          -> absl::Status {
        if (first_input_start_token == -1) {
          std::vector<int32_t> token_ids(4);
          auto res = const_cast<::litert::TensorBuffer&>(inputs[1])
                         .Read<int32_t>(absl::MakeSpan(token_ids));
          if (!res) return absl::InternalError("Read failed");
          first_input_start_token = token_ids[0];
        }
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder, StatelessDecoder::Create(&mock_runner, kStartTokenId));
  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));
  ASSERT_OK(decoder->Decode(encoder_outputs));
  EXPECT_EQ(first_input_start_token, kStartTokenId);
}

TEST(StatelessDecoderTest, DecodeWithDecodeStopTokenId) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(4, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(20, sizeof(float)));
        return buffers;
      });

  constexpr int kStopTokenId = 3;
  size_t run_count = 0;

  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&run_count](
                          absl::string_view,
                          absl::Span<const ::litert::TensorBuffer> inputs,
                          absl::Span<const ::litert::TensorBuffer> outputs)
                          -> absl::Status {
        std::vector<float> logits(20, 0.0f);
        if (run_count == 0) {
          logits[0 * 5 + 1] = 1.0f;  // Step 0 produces token 1
        } else if (run_count == 1) {
          logits[1 * 5 + 3] = 1.0f;  // Step 1 produces token 3 (kStopTokenId)
        } else {
          logits[run_count * 5 + 2] = 1.0f;  // Subsequent steps produce token 2
        }
        run_count++;
        auto res = const_cast<::litert::TensorBuffer&>(outputs[0])
                       .Write<float>(absl::MakeConstSpan(logits));
        if (!res) return absl::InternalError("Write failed");
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      StatelessDecoder::Create(&mock_runner, /*decode_start_token_id=*/-1,
                               /*decode_stop_token_id=*/kStopTokenId));
  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto decoded_tokens, decoder->Decode(encoder_outputs));

  EXPECT_EQ(run_count, 2);
  ASSERT_EQ(decoded_tokens.size(), 1);
  EXPECT_EQ(decoded_tokens[0].token_id, 1);
}

TEST(StatelessDecoderTest, DecodeWithDecodeSkipUntilTokenId) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(5, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers(_))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(25, sizeof(float)));
        return buffers;
      });

  constexpr int kSkipUntilTokenId = 4;
  size_t run_count = 0;

  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&run_count](
                          absl::string_view,
                          absl::Span<const ::litert::TensorBuffer> inputs,
                          absl::Span<const ::litert::TensorBuffer> outputs)
                          -> absl::Status {
        std::vector<float> logits(25, 0.0f);
        if (run_count == 0) {
          logits[0 * 5 + 1] = 1.0f;  // Step 0 -> token 1 (should be skipped)
        } else if (run_count == 1) {
          logits[1 * 5 + 4] =
              1.0f;  // Step 1 -> token 4 (kSkipUntilTokenId, should be skipped)
        } else if (run_count == 2) {
          logits[2 * 5 + 2] = 1.0f;  // Step 2 -> token 2 (should be included)
        } else {
          logits[3 * 5 + 3] = 1.0f;  // Step 3 -> token 3 (should be included)
        }
        run_count++;
        auto res = const_cast<::litert::TensorBuffer&>(outputs[0])
                       .Write<float>(absl::MakeConstSpan(logits));
        if (!res) return absl::InternalError("Write failed");
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      StatelessDecoder::Create(
          &mock_runner, /*decode_start_token_id=*/-1,
          /*decode_stop_token_id=*/-1,
          /*decode_skip_until_token_id=*/kSkipUntilTokenId));
  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto decoded_tokens, decoder->Decode(encoder_outputs));

  EXPECT_EQ(run_count, 4);
  ASSERT_EQ(decoded_tokens.size(), 2);
  EXPECT_EQ(decoded_tokens[0].token_id, 2);
  EXPECT_EQ(decoded_tokens[1].token_id, 3);
}

}  // namespace
}  // namespace litert::omni::asr
