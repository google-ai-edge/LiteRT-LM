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

#include "omni/speech2text/stateful_decoder.h"

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
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"

namespace litert::omni::asr {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
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

TEST(StatefulDecoderTest, CreateWhenRunnerNullReturnsError) {
  LiteRtRunner* null_runner = nullptr;
  EXPECT_FALSE(StatefulDecoder::Create(null_runner).ok());
}

TEST(StatefulDecoderTest, CreateWhenCreateInputBuffersFailsReturnsError) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers(_))
      .WillOnce(Return(absl::InternalError("Failed to create input buffers")));

  EXPECT_FALSE(StatefulDecoder::Create(&mock_runner).ok());
}

TEST(StatefulDecoderTest, CreateSuccessfully) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));   // token
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));   // enc
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));   // state
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));  // logits
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));   // state
        return buffers;
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      StatefulDecoder::Create(&mock_runner, "decode",
                              /*num_non_state_inputs=*/2,
                              /*num_non_state_outputs=*/1));
  EXPECT_NE(decoder, nullptr);
}

TEST(StatefulDecoderTest, CreateWhenModelHasZeroStateTensorsReturnsError) {
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
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        return buffers;
      });

  auto status =
      StatefulDecoder::Create(&mock_runner, "decode",
                              /*num_non_state_inputs=*/2,
                              /*num_non_state_outputs=*/1)
          .status();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_THAT(status.message(), HasSubstr("recurrent state tensors"));
}

TEST(StatefulDecoderTest, DecodeFailsWhenEncoderOutputsEmpty) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(100, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));
        return buffers;
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      StatefulDecoder::Create(&mock_runner, "decode",
                              /*num_non_state_inputs=*/2,
                              /*num_non_state_outputs=*/1));
  std::vector<::litert::TensorBuffer> empty_encoder_outputs;
  EXPECT_FALSE(decoder->Decode(empty_encoder_outputs).ok());
}

TEST(StatefulDecoderTest, DecodeGeneratesTokensAndStopsOnStopToken) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));
        return buffers;
      });

  int step_count = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&step_count](absl::string_view,
                                    absl::Span<const ::litert::TensorBuffer>,
                                    absl::Span<const ::litert::TensorBuffer> outputs) {
        std::vector<float> logits(10, 0.0f);
        if (step_count == 0) {
          logits[5] = 10.0f;  // first token
        } else if (step_count == 1) {
          logits[7] = 10.0f;  // second token
        } else {
          logits[9] = 10.0f;  // stop token
        }
        step_count++;
        auto& mutable_out = const_cast<::litert::TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(mutable_out.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      StatefulDecoder::Create(&mock_runner, "decode",
                              /*num_non_state_inputs=*/2,
                              /*num_non_state_outputs=*/1,
                              /*decode_start_token_ids=*/{1},
                              /*decode_stop_token_id=*/9,
                              /*decode_skip_until_token_id=*/-1,
                              /*max_decode_steps=*/10));

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));

  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].token_id, 5);
  EXPECT_EQ(tokens[1].token_id, 7);
}

TEST(StatefulDecoderTest, DecodeSuccessfullyWhenTokenInputIsSecondBuffer) {
  MockLiteRtRunner mock_runner;
  EXPECT_CALL(mock_runner, CreateInputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(16, sizeof(float)));   // enc (input 0)
        buffers.push_back(CreateTestTensorBuffer(1, sizeof(int32_t)));  // token (input 1)
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));   // state (input 2)
        return buffers;
      });
  EXPECT_CALL(mock_runner, CreateOutputBuffers("decode"))
      .WillOnce([](absl::string_view) {
        std::vector<::litert::TensorBuffer> buffers;
        buffers.push_back(CreateTestTensorBuffer(10, sizeof(float)));
        buffers.push_back(CreateTestTensorBuffer(64, sizeof(float)));
        return buffers;
      });

  int step_count = 0;
  EXPECT_CALL(mock_runner, Run("decode", _, _))
      .WillRepeatedly([&step_count](
                          absl::string_view,
                          absl::Span<const ::litert::TensorBuffer> inputs,
                          absl::Span<const ::litert::TensorBuffer> outputs) {
        // Verify input 1 contains the written token
        int32_t token_val = 0;
        EXPECT_TRUE(const_cast<::litert::TensorBuffer&>(inputs[1])
                        .Read<int32_t>(absl::MakeSpan(&token_val, 1))
                        .HasValue());
        if (step_count == 0) {
          EXPECT_EQ(token_val, 1);  // prompt start token
        }

        std::vector<float> logits(10, 0.0f);
        if (step_count == 0) {
          logits[4] = 10.0f;
        } else {
          logits[9] = 10.0f;  // stop token
        }
        step_count++;
        auto& mutable_out = const_cast<::litert::TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(mutable_out.Write<float>(absl::MakeConstSpan(logits)).HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      StatefulDecoder::Create(&mock_runner, "decode",
                              /*num_non_state_inputs=*/2,
                              /*num_non_state_outputs=*/1,
                              /*decode_start_token_ids=*/{1},
                              /*decode_stop_token_id=*/9,
                              /*decode_skip_until_token_id=*/-1,
                              /*max_decode_steps=*/10));

  std::vector<::litert::TensorBuffer> encoder_outputs;
  encoder_outputs.push_back(CreateTestTensorBuffer(16, sizeof(float)));

  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(encoder_outputs));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token_id, 4);
}

}  // namespace
}  // namespace litert::omni::asr
