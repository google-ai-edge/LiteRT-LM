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
#include "omni/base/mock_litert_runner.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::_;
using ::testing::Return;

::litert::TensorBuffer CreateTestTensorBuffer(size_t num_elements,
                                              size_t element_size_bytes) {
  ::litert::RankedTensorType tensor_type(
      element_size_bytes == sizeof(float) ? ::litert::ElementType::Float32
                                          : ::litert::ElementType::Int32,
      ::litert::Layout(
          ::litert::Dimensions{static_cast<int32_t>(num_elements)}));
  auto buffer_or = ::litert::TensorBuffer::CreateManagedHostMemory(
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
      .WillOnce([](absl::string_view) {
        return std::vector<::litert::TensorBuffer>();
      });

  EXPECT_FALSE(TdtDecoder::Create(&mock_runner).ok());
}

TEST(TdtDecoderTest, CreateSuccessfully) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 4, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
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
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 1, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
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
  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024, sizeof(float)));
  auto tokens_or = decoder->Decode(encoder_outputs);
  EXPECT_FALSE(tokens_or.ok());
}

TEST(TdtDecoderTest, DecodeIncludesEndOfChunkToken) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1024 * 1, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
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
  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(1024, sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_GE(tokens.size(), 1);
  EXPECT_TRUE(tokens.back().IsEndOfChunk());
  EXPECT_EQ(tokens.back().timestamp_ms, 1);
}

}  // namespace
}  // namespace litert::omni::asr
