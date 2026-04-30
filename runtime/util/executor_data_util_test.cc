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

#include "runtime/util/executor_data_util.h"

#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/tensor_buffer_util.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::litert::Dimensions;
using ::litert::ElementType;
using ::litert::Layout;
using ::litert::TensorBuffer;
using ::testing::ElementsAre;
using ::testing::status::StatusIs;

TEST(ExecutorDataUtilTest, CombineExecutorVisionDataTest) {
  struct alignas(::litert::kHostMemoryBufferAlignment) {
    float d[24] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                   9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                   17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
  } data1;
  auto tensor1 = TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(ElementType::Float32,
                                 Layout(Dimensions({1, 2, 4, 3}))),
      data1.d, sizeof(data1.d));
  ASSERT_TRUE(tensor1.HasValue());
  ExecutorVisionData vision_data1(std::move(*tensor1), std::nullopt);

  struct alignas(::litert::kHostMemoryBufferAlignment) {
    float d[12] = {25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                   31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f};
  } data2;
  auto tensor2 = TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(ElementType::Float32,
                                 Layout(Dimensions({1, 2, 2, 3}))),
      data2.d, sizeof(data2.d));
  ASSERT_TRUE(tensor2.HasValue());
  ExecutorVisionData vision_data2(std::move(*tensor2), std::nullopt);

  std::vector<ExecutorVisionData> vision_data_list;
  vision_data_list.push_back(std::move(vision_data1));
  vision_data_list.push_back(std::move(vision_data2));

  auto combined_vision_data = CombineExecutorVisionData(vision_data_list);
  ASSERT_OK(combined_vision_data);

  auto mutable_embeddings_ptr = combined_vision_data->GetMutableEmbeddingsPtr();
  ASSERT_OK(mutable_embeddings_ptr);

  litert::TensorBuffer* embeddings_ptr = mutable_embeddings_ptr.value();
  EXPECT_NE(embeddings_ptr, nullptr);

  auto tensor_type = embeddings_ptr->TensorType();
  ASSERT_TRUE(tensor_type.HasValue());
  EXPECT_EQ(tensor_type->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor_type->Layout(), Layout(Dimensions({1, 2, 6, 3})));

  float read_data[36];
  auto read_success = embeddings_ptr->Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  for (int i = 0; i < 36; ++i) {
    EXPECT_EQ(read_data[i], static_cast<float>(i + 1));
  }
}

TEST(ExecutorDataUtilTest, CombineExecutorAudioDataTest) {
  struct alignas(::litert::kHostMemoryBufferAlignment) {
    float d[12] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  } data1;
  auto tensor1 = TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(ElementType::Float32,
                                 Layout(Dimensions({1, 4, 3}))),
      data1.d, sizeof(data1.d));
  ASSERT_TRUE(tensor1.HasValue());
  ExecutorAudioData audio_data1(std::move(*tensor1), std::nullopt, 4);

  struct alignas(::litert::kHostMemoryBufferAlignment) {
    float d[6] = {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
  } data2;
  auto tensor2 = TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(ElementType::Float32,
                                 Layout(Dimensions({1, 2, 3}))),
      data2.d, sizeof(data2.d));
  ASSERT_TRUE(tensor2.HasValue());
  ExecutorAudioData audio_data2(std::move(*tensor2), std::nullopt, 2);

  std::vector<ExecutorAudioData> audio_data_list;
  audio_data_list.push_back(std::move(audio_data1));
  audio_data_list.push_back(std::move(audio_data2));

  auto combined_audio_data = CombineExecutorAudioData(audio_data_list);
  ASSERT_OK(combined_audio_data);

  auto mutable_embeddings_ptr = combined_audio_data->GetMutableEmbeddingsPtr();
  ASSERT_OK(mutable_embeddings_ptr);

  litert::TensorBuffer* embeddings_ptr = mutable_embeddings_ptr.value();
  EXPECT_NE(embeddings_ptr, nullptr);

  auto tensor_type = embeddings_ptr->TensorType();
  ASSERT_TRUE(tensor_type.HasValue());
  EXPECT_EQ(tensor_type->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor_type->Layout(), Layout(Dimensions({1, 6, 3})));

  EXPECT_EQ(combined_audio_data->GetValidTokens(), 6);

  float read_data[18];
  auto read_success = embeddings_ptr->Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  for (int i = 0; i < 18; ++i) {
    EXPECT_EQ(read_data[i], static_cast<float>(i + 1));
  }
}

TEST(ExecutorDataUtilTest, CombineExecutorAudioDataEmptyFails) {
  std::vector<ExecutorAudioData> executor_data;
  EXPECT_THAT(
      CombineExecutorAudioData(executor_data),
      StatusIs(absl::StatusCode::kInvalidArgument, "Executor data is empty."));
}

TEST(ExecutorDataUtilTest, CombineExecutorAudioDataSingleSuccess) {
  std::vector<ExecutorAudioData> executor_data;
  ExecutorAudioData executor_audio_data;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_buffer,
      CopyToTensorBuffer<float>({4.0, 3.0, 2.0, 1.0}, {1, 2, 2}));
  executor_audio_data.SetEmbeddings(std::move(audio_buffer));
  executor_data.push_back(std::move(executor_audio_data));
  ASSERT_OK_AND_ASSIGN(auto combined_executor_data,
                       CombineExecutorAudioData(executor_data));
  ASSERT_OK_AND_ASSIGN(auto combined_embeddings_ptr,
                       combined_executor_data.GetEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto combined_embeddings_span,
      ReferTensorBufferAsSpan<float>(*combined_embeddings_ptr));
  EXPECT_THAT(std::vector<float>(combined_embeddings_span.begin(),
                                 combined_embeddings_span.end()),
              ElementsAre(4.0, 3.0, 2.0, 1.0));
}

TEST(ExecutorDataUtilTest, CombineExecutorAudioDataMultiSuccess) {
  std::vector<ExecutorAudioData> executor_data;

  ExecutorAudioData executor_audio_data_1;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_buffer_1,
      CopyToTensorBuffer<float>({6.0, 5.0, 4.0, 3.0, 2.0, 1.0}, {1, 3, 2}));
  executor_audio_data_1.SetEmbeddings(std::move(audio_buffer_1));
  executor_audio_data_1.SetValidTokens(3);
  executor_data.push_back(std::move(executor_audio_data_1));

  ExecutorAudioData executor_audio_data_2;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_buffer_2,
      CopyToTensorBuffer<float>({5.0, 6.0, 7.0, 8.0}, {1, 2, 2}));
  executor_audio_data_2.SetEmbeddings(std::move(audio_buffer_2));
  executor_audio_data_2.SetValidTokens(2);
  executor_data.push_back(std::move(executor_audio_data_2));

  ExecutorAudioData executor_audio_data_3;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_buffer_3, CopyToTensorBuffer<float>({11.0, 12.0}, {1, 1, 2}));
  executor_audio_data_3.SetEmbeddings(std::move(audio_buffer_3));
  executor_audio_data_3.SetValidTokens(1);
  executor_data.push_back(std::move(executor_audio_data_3));

  ASSERT_OK_AND_ASSIGN(auto combined_executor_data,
                       CombineExecutorAudioData(executor_data));
  ASSERT_OK_AND_ASSIGN(auto combined_embeddings_ptr,
                       combined_executor_data.GetEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto combined_embeddings_span,
      ReferTensorBufferAsSpan<float>(*combined_embeddings_ptr));
  ASSERT_OK_AND_ASSIGN(const auto& dimensions,
                       TensorBufferDims(*combined_embeddings_ptr));
  EXPECT_THAT(dimensions, ElementsAre(1, 6, 2));
  EXPECT_THAT(std::vector<float>(combined_embeddings_span.begin(),
                                 combined_embeddings_span.end()),
              ElementsAre(6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0,
                          11.0, 12.0));
}

TEST(ExecutorDataUtilTest, CombineExecutorVisionDataEmptyFails) {
  std::vector<ExecutorVisionData> executor_data;
  EXPECT_THAT(
      CombineExecutorVisionData(executor_data),
      StatusIs(absl::StatusCode::kInvalidArgument, "Executor data is empty."));
}

TEST(ExecutorDataUtilTest, CombineExecutorVisionDataSingleSuccess) {
  std::vector<ExecutorVisionData> executor_data;
  ExecutorVisionData executor_vision_data;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_buffer,
      CopyToTensorBuffer<float>({1.0, 2.0, 3.0, 4.0}, {1, 2, 2}));
  executor_vision_data.SetEmbeddings(std::move(vision_buffer));
  executor_data.push_back(std::move(executor_vision_data));
  ASSERT_OK_AND_ASSIGN(auto combined_executor_data,
                       CombineExecutorVisionData(executor_data));
  ASSERT_OK_AND_ASSIGN(auto combined_embeddings_ptr,
                       combined_executor_data.GetEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto combined_embeddings_span,
      ReferTensorBufferAsSpan<float>(*combined_embeddings_ptr));
  EXPECT_THAT(std::vector<float>(combined_embeddings_span.begin(),
                                 combined_embeddings_span.end()),
              ElementsAre(1.0, 2.0, 3.0, 4.0));
}

TEST(ExecutorDataUtilTest, CombineExecutorVisionDataMultiSuccess) {
  std::vector<ExecutorVisionData> executor_data;

  ExecutorVisionData executor_vision_data_1;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_buffer,
      CopyToTensorBuffer<float>({1.0, 2.0, 3.0, 4.0}, {1, 2, 2}));
  executor_vision_data_1.SetEmbeddings(std::move(vision_buffer));
  executor_data.push_back(std::move(executor_vision_data_1));

  ExecutorVisionData executor_vision_data_2;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_buffer_2,
      CopyToTensorBuffer<float>({5.0, 6.0, 7.0, 8.0}, {1, 2, 2}));
  executor_vision_data_2.SetEmbeddings(std::move(vision_buffer_2));
  executor_data.push_back(std::move(executor_vision_data_2));

  ExecutorVisionData executor_vision_data_3;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_buffer_3,
      CopyToTensorBuffer<float>({9.0, 10.0, 11.0, 12.0}, {1, 2, 2}));
  executor_vision_data_3.SetEmbeddings(std::move(vision_buffer_3));
  executor_data.push_back(std::move(executor_vision_data_3));

  ASSERT_OK_AND_ASSIGN(auto combined_executor_data,
                       CombineExecutorVisionData(executor_data));
  ASSERT_OK_AND_ASSIGN(auto combined_embeddings_ptr,
                       combined_executor_data.GetEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto combined_embeddings_span,
      ReferTensorBufferAsSpan<float>(*combined_embeddings_ptr));
  ASSERT_OK_AND_ASSIGN(const auto& dimensions,
                       TensorBufferDims(*combined_embeddings_ptr));
  EXPECT_THAT(dimensions, ElementsAre(1, 1, 6, 2));
  EXPECT_THAT(std::vector<float>(combined_embeddings_span.begin(),
                                 combined_embeddings_span.end()),
              ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                          11.0, 12.0));
}

}  // namespace
}  // namespace litert::lm
