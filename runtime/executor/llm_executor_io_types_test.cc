// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/llm_executor_io_types.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <sstream>
#include <utility>

#include <gtest/gtest.h>
#include "litert/litert/c/litert_tensor_buffer.h"  // from @litert
#include "litert/litert/cc/litert_element_type.h"  // from @litert
#include "litert/litert/cc/litert_layout.h"  // from @litert
#include "litert/litert/cc/litert_model.h"  // from @litert
#include "litert/litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {
namespace {

TEST(LlmExecutorIoTypesTest, InputsPrint) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int32_t d[6] = {1, 2, 3, 4, 5, 6};
  } data;

  auto token_ids = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(
          ::litert::ElementType::Int32,
          ::litert::Layout(::litert::Dimensions({3, 2}))),
      data.d, 6 * sizeof(int32_t));
  EXPECT_TRUE(token_ids.HasValue());

  ExecutorInputs inputs(ExecutorTextData(std::move(*token_ids)), std::nullopt,
                        std::nullopt);
  std::stringstream oss;
  oss << inputs;
  EXPECT_EQ(oss.str(),
            "ExecutorInputs: {\n  TextData: ExecutorTextData: {\n  TokenIds: "
            "TensorBuffer: [[1, 2], [3, 4], [5, 6]] shape=(3, 2)\n}\n  "
            "VisionData: nullopt\n  AudioData: nullopt\n}");
}

TEST(LlmExecutorIoTypesTest, ExecutorVisionData) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {31.0f, 32.0f};
  } emb_data;
  auto embeddings_opt = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(::litert::ElementType::Float32,
                                 ::litert::Layout(::litert::Dimensions({2}))),
      emb_data.d, sizeof(emb_data.d));
  ASSERT_TRUE(embeddings_opt.HasValue());

  ExecutorVisionData vision_data(std::move(*embeddings_opt), std::nullopt);
  std::stringstream oss;
  oss << vision_data;

  EXPECT_EQ(oss.str(),
            "ExecutorVisionData: {\n"
            "  Embeddings: TensorBuffer: [31, 32] shape=(2)\n"
            "  PerLayerEmbeddings: nullopt\n"
            "}");
}

TEST(LlmExecutorIoTypesTest, ExecutorAudioData) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {131.0f, 132.0f};
  } emb_data;
  auto embeddings_opt = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(::litert::ElementType::Float32,
                                 ::litert::Layout(::litert::Dimensions({2}))),
      emb_data.d, sizeof(emb_data.d));
  ASSERT_TRUE(embeddings_opt.HasValue());

  ExecutorAudioData audio_data(
      std::move(*embeddings_opt),
      std::nullopt);  // Uses constructor for embeddings only
  std::stringstream oss;
  oss << audio_data;

  EXPECT_EQ(oss.str(),
            "ExecutorAudioData: {\n"
            "  Embeddings: TensorBuffer: [131, 132] shape=(2)\n"
            "  PerLayerEmbeddings: nullopt\n"
            "}");
}

TEST(LlmExecutorTest, ExecutorPrefillParamsPrint) {
  std::atomic_bool cancel = true;
  ExecutorPrefillParams params(
      /*current_step=*/10, /*wait_for_completion=*/true, /*cancel=*/&cancel);
  std::stringstream oss;
  oss << params;
  EXPECT_EQ(oss.str(),
            "ExecutorPrefillParams: {\n  CurrentStep: 10\n  WaitForCompletion: "
            "true\n  CancelFlag: true (atomic)\n}");

  // Test with a null cancel flag.
  oss.str("");
  params.SetCancelFlag(nullptr);
  oss << params;
  EXPECT_EQ(oss.str(),
            "ExecutorPrefillParams: {\n  CurrentStep: 10\n  WaitForCompletion: "
            "true\n  CancelFlag: nullptr\n}");
}

}  // namespace
}  // namespace litert::lm
