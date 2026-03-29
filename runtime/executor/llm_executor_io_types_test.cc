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
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"

namespace litert::lm {
namespace {

using ::litert::Dimensions;
using ::litert::ElementType;
using ::litert::Layout;
using ::litert::TensorBuffer;

TEST(LlmExecutorIoTypesTest, InputsPrint) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int32_t d[6] = {1, 2, 3, 4, 5, 6};
  } data;

  auto env = Environment::Create({});
  // Create a TensorBuffer for token_ids
  auto token_ids = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Int32,
                                 Layout(Dimensions({3, 2}))),
      data.d, 6 * sizeof(int32_t));
  EXPECT_TRUE(token_ids.HasValue());

  // Construct ExecutorInputs with only text_data
  ExecutorInputs inputs(ExecutorTextData(std::move(*token_ids)), std::nullopt,
                        std::nullopt);
  std::stringstream oss;
  oss << inputs;  // Invoke operator<< for ExecutorInputs

  // Define the expected output string.
  // Note the updated messages for nullopt fields due to StatusOr error
  // messages.
  const std::string expected_output =
      "ExecutorInputs: {\n"
      "  NextPosition: 0\n"
      "  TextData: ExecutorTextData: {\n"
      "  TokenIds: TensorBuffer: [[1, 2], [3, 4], [5, 6]] shape=(3, "
      "2)\n"  // Assuming TensorBuffer prints this way
      "}\n"
      "  VisionData: nullopt (ExecutorInputs::vision_data_ is not "
      "set.)\n"
      "  AudioData: nullopt (ExecutorInputs::audio_data_ is not set.)\n"
      "}";
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorIoTypesTest, ExecutorVisionDataPrint) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {31.0f, 32.0f};
  } emb_data;

  auto env = Environment::Create({});
  // Create a TensorBuffer for embeddings
  auto embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      emb_data.d, sizeof(emb_data.d));
  ASSERT_TRUE(embeddings.HasValue());

  // Construct ExecutorVisionData with embeddings and nullopt for
  // per_layer_embeddings
  ExecutorVisionData vision_data(std::move(*embeddings), std::nullopt);
  std::stringstream oss;
  oss << vision_data;  // Invoke operator<< for ExecutorVisionData

  // Define the expected output string.
  // Note the updated message for the nullopt per_layer_embeddings.
  const std::string expected_output =
      "ExecutorVisionData: {\n"
      "  Embeddings: TensorBuffer: [31, 32] shape=(2)\n"  // Assuming
                                                          // TensorBuffer prints
                                                          // this way
      "  PerLayerEmbeddings: nullopt "
      "(ExecutorVisionData::per_layer_embeddings_ is not set.)\n"
      "}";
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorIoTypesTest, ExecutorAudioDataPrint) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {131.0f, 132.0f};
  } emb_data;

  auto env = Environment::Create({});
  // Create a TensorBuffer for embeddings
  auto embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      emb_data.d, sizeof(emb_data.d));
  ASSERT_TRUE(embeddings.HasValue());
  // Construct ExecutorAudioData with embeddings and nullopt for
  // per_layer_embeddings
  ExecutorAudioData audio_data(std::move(*embeddings), std::nullopt, 1);
  std::stringstream oss;
  oss << audio_data;  // Invoke operator<< for ExecutorAudioData

  // Define the expected output string.
  // Note the updated message for the nullopt per_layer_embeddings.
  const std::string expected_output =
      "ExecutorAudioData: {\n"
      "  Embeddings: TensorBuffer: [131, 132] shape=(2)\n"
      "  PerLayerEmbeddings: nullopt (ExecutorAudioData::per_layer_embeddings_ "
      "is not set.)\n  ValidTokens: 1\n"
      "}";
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorIoTypesTest, ExecutorPrefillParamsPrint) {
  std::atomic_bool cancel = true;
  ExecutorPrefillParams params(
      /*current_step=*/10, /*wait_for_completion=*/true, /*cancel=*/&cancel,
      /*max_prefill_sequence_length=*/128);
  std::stringstream oss;
  oss << params;  // Invoke operator<< for ExecutorPrefillParams
  EXPECT_EQ(oss.str(),
            "ExecutorPrefillParams: {\n"
            "  CurrentStep: 10\n"
            "  WaitForCompletion: true\n"
            "  CancelFlag: true (atomic)\n"
            "  MaxPrefillSequenceLength: 128\n"
            "}");

  // Test with a null cancel flag.
  oss.str("");  // Clear the stringstream
  params.SetCancelFlag(nullptr);
  params.SetMaxPrefillSequenceLength(std::nullopt);
  oss << params;
  EXPECT_EQ(oss.str(),
            "ExecutorPrefillParams: {\n"
            "  CurrentStep: 10\n"
            "  WaitForCompletion: true\n"
            "  CancelFlag: nullptr\n"
            "  MaxPrefillSequenceLength: nullopt\n"
            "}");
}

TEST(LlmExecutorIoTypesTest, ExecutorTextDataGetSet) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int32_t d[2] = {7, 8};
  } data;
  auto env = Environment::Create({});
  auto token_ids = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Int32, Layout(Dimensions({2}))),
      data.d, 2 * sizeof(int32_t));
  ASSERT_TRUE(token_ids.HasValue());

  ExecutorTextData text_data(std::move(*token_ids));

  // Test GetTokenIds
  auto& get_token_ids = text_data.GetMutableTokenIds();
  auto get_token_ids_size = get_token_ids.Size();
  ASSERT_TRUE(get_token_ids_size.HasValue());
  EXPECT_EQ(get_token_ids_size.Value(), 8);
  int32_t read_data[2];
  auto read_success = get_token_ids.Read<int32_t>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(read_data[0], 7);
  EXPECT_EQ(read_data[1], 8);

  // Test GetMutableTokenIds
  auto& mutable_token_ids = text_data.GetMutableTokenIds();
  auto mutable_token_ids_size = mutable_token_ids.Size();
  ASSERT_TRUE(mutable_token_ids_size.HasValue());
  EXPECT_EQ(mutable_token_ids_size.Value(), 8);

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int32_t d[2] = {9, 10};
  } new_data;
  auto new_token_ids = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Int32, Layout(Dimensions({2}))),
      new_data.d, 2 * sizeof(int32_t));
  ASSERT_TRUE(new_token_ids.HasValue());
  // Test SetTokenIds
  text_data.SetTokenIds(std::move(*new_token_ids));
  auto& get_new_token_ids = text_data.GetMutableTokenIds();
  auto get_new_token_ids_size = get_new_token_ids.Size();
  ASSERT_TRUE(get_new_token_ids_size.HasValue());
  EXPECT_EQ(get_new_token_ids_size.Value(), 8);
  read_success = get_new_token_ids.Read<int32_t>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(read_data[0], 9);
  EXPECT_EQ(read_data[1], 10);
}

TEST(LlmExecutorIoTypesTest, ExecutorVisionDataGetSet) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {31.0f, 32.0f};
  } emb_data;

  auto env = Environment::Create({});
  // Create a TensorBuffer for embeddings
  auto embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      emb_data.d, sizeof(emb_data.d));
  ASSERT_TRUE(embeddings.HasValue());

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[4] = {33.0f, 34.0f, 35.0f, 36.0f};
  } ple_emb_data;
  // Create a TensorBuffer for per_layer_embeddings
  auto per_layer_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({4}))),
      ple_emb_data.d, sizeof(ple_emb_data.d));
  ASSERT_TRUE(per_layer_embeddings.HasValue());

  ExecutorVisionData vision_data(std::move(*embeddings),
                                 std::move(*per_layer_embeddings));

  // Test GetEmbeddingsPtr
  auto get_embeddings_ptr_status = vision_data.GetMutableEmbeddingsPtr();
  ASSERT_TRUE(get_embeddings_ptr_status.ok());
  TensorBuffer* get_embeddings_ptr = get_embeddings_ptr_status.value();
  auto get_embeddings_size = get_embeddings_ptr->Size();
  ASSERT_TRUE(get_embeddings_size.HasValue());
  EXPECT_EQ(get_embeddings_size.Value(), 8);
  float read_data[2];
  auto read_success =
      get_embeddings_ptr->Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(read_data[0], 31.0f);
  EXPECT_EQ(read_data[1], 32.0f);

  // Test GetMutableEmbeddingsPtr
  auto get_mutable_embeddings_ptr_status =
      vision_data.GetMutableEmbeddingsPtr();
  ASSERT_TRUE(get_mutable_embeddings_ptr_status.ok());
  TensorBuffer* get_mutable_embeddings_ptr =
      get_mutable_embeddings_ptr_status.value();
  auto get_mutable_embeddings_size = get_mutable_embeddings_ptr->Size();
  ASSERT_TRUE(get_mutable_embeddings_size.HasValue());
  EXPECT_EQ(get_mutable_embeddings_size.Value(), 8);

  // Test GetPerLayerEmbeddingsPtr
  auto get_per_layer_embeddings_ptr_status =
      vision_data.GetMutablePerLayerEmbeddingsPtr();
  ASSERT_TRUE(get_per_layer_embeddings_ptr_status.ok());
  TensorBuffer* get_per_layer_embeddings_ptr =
      get_per_layer_embeddings_ptr_status.value();
  auto get_per_layer_embeddings_size = get_per_layer_embeddings_ptr->Size();
  ASSERT_TRUE(get_per_layer_embeddings_size.HasValue());
  EXPECT_EQ(get_per_layer_embeddings_size.Value(), 16);
  float ple_read_data[4];
  read_success =
      get_per_layer_embeddings_ptr->Read<float>(absl::MakeSpan(ple_read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(ple_read_data[0], 33.0f);
  EXPECT_EQ(ple_read_data[1], 34.0f);
  EXPECT_EQ(ple_read_data[2], 35.0f);
  EXPECT_EQ(ple_read_data[3], 36.0f);

  // Test GetMutablePerLayerEmbeddingsPtr
  auto get_mutable_per_layer_embeddings_ptr_status =
      vision_data.GetMutablePerLayerEmbeddingsPtr();
  ASSERT_TRUE(get_mutable_per_layer_embeddings_ptr_status.ok());
  TensorBuffer* get_mutable_per_layer_embeddings_ptr =
      get_mutable_per_layer_embeddings_ptr_status.value();
  auto get_mutable_per_layer_embeddings_size =
      get_mutable_per_layer_embeddings_ptr->Size();
  ASSERT_TRUE(get_mutable_per_layer_embeddings_size.HasValue());
  EXPECT_EQ(get_mutable_per_layer_embeddings_size.Value(), 16);

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {37.0f, 38.0f};
  } new_emb_data;
  // Create a new TensorBuffer for embeddings
  auto new_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      new_emb_data.d, sizeof(new_emb_data.d));
  ASSERT_TRUE(new_embeddings.HasValue());
  // Test SetEmbeddings
  vision_data.SetEmbeddings(std::move(*new_embeddings));
  auto get_new_embeddings_ptr_status = vision_data.GetMutableEmbeddingsPtr();
  ASSERT_TRUE(get_new_embeddings_ptr_status.ok());
  TensorBuffer* get_new_embeddings_ptr = get_new_embeddings_ptr_status.value();
  auto get_new_embeddings_size = get_new_embeddings_ptr->Size();
  ASSERT_TRUE(get_new_embeddings_size.HasValue());
  EXPECT_EQ(get_new_embeddings_size.Value(), 8);
  read_success = get_new_embeddings_ptr->Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(read_data[0], 37.0f);
  EXPECT_EQ(read_data[1], 38.0f);

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[4] = {39.0f, 40.0f, 41.0f, 42.0f};
  } new_ple_emb_data;
  // Create a new TensorBuffer for per_layer_embeddings
  auto new_per_layer_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({4}))),
      new_ple_emb_data.d, sizeof(new_ple_emb_data.d));
  ASSERT_TRUE(new_per_layer_embeddings.HasValue());
  // Test SetPerLayerEmbeddings
  vision_data.SetPerLayerEmbeddings(std::move(*new_per_layer_embeddings));
  auto get_new_per_layer_embeddings_ptr_status =
      vision_data.GetMutablePerLayerEmbeddingsPtr();
  ASSERT_TRUE(get_new_per_layer_embeddings_ptr_status.ok());
  TensorBuffer* get_new_per_layer_embeddings_ptr =
      get_new_per_layer_embeddings_ptr_status.value();
  auto get_new_per_layer_embeddings_size =
      get_new_per_layer_embeddings_ptr->Size();
  ASSERT_TRUE(get_new_per_layer_embeddings_size.HasValue());
  EXPECT_EQ(get_new_per_layer_embeddings_size.Value(), 16);
  read_success = get_new_per_layer_embeddings_ptr->Read<float>(
      absl::MakeSpan(ple_read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(ple_read_data[0], 39.0f);
  EXPECT_EQ(ple_read_data[1], 40.0f);
  EXPECT_EQ(ple_read_data[2], 41.0f);
  EXPECT_EQ(ple_read_data[3], 42.0f);
}

TEST(LlmExecutorIoTypesTest, ExecutorAudioDataGetSet) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {131.0f, 132.0f};
  } emb_data;

  auto env = Environment::Create({});
  // Create a TensorBuffer for embeddings
  auto embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      emb_data.d, sizeof(emb_data.d));
  ASSERT_TRUE(embeddings.HasValue());

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[4] = {133.0f, 134.0f, 135.0f, 136.0f};
  } ple_emb_data;
  // Create a TensorBuffer for per_layer_embeddings
  auto per_layer_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({4}))),
      ple_emb_data.d, sizeof(ple_emb_data.d));
  ASSERT_TRUE(per_layer_embeddings.HasValue());

  ExecutorAudioData audio_data(std::move(*embeddings),
                               std::move(*per_layer_embeddings), 1);

  // Test GetEmbeddingsPtr
  auto get_embeddings_ptr_status = audio_data.GetMutableEmbeddingsPtr();
  ASSERT_TRUE(get_embeddings_ptr_status.ok());
  TensorBuffer* get_embeddings_ptr = get_embeddings_ptr_status.value();
  auto get_embeddings_size = get_embeddings_ptr->Size();
  ASSERT_TRUE(get_embeddings_size.HasValue());
  EXPECT_EQ(get_embeddings_size.Value(), 8);
  float read_data[2];
  auto read_success =
      get_embeddings_ptr->Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(read_data[0], 131.0f);
  EXPECT_EQ(read_data[1], 132.0f);

  // Test GetMutableEmbeddingsPtr
  auto get_mutable_embeddings_ptr_status = audio_data.GetMutableEmbeddingsPtr();
  ASSERT_TRUE(get_mutable_embeddings_ptr_status.ok());
  TensorBuffer* get_mutable_embeddings_ptr =
      get_mutable_embeddings_ptr_status.value();
  auto get_mutable_embeddings_size = get_mutable_embeddings_ptr->Size();
  ASSERT_TRUE(get_mutable_embeddings_size.HasValue());
  EXPECT_EQ(get_mutable_embeddings_size.Value(), 8);

  // Test GetPerLayerEmbeddingsPtr
  auto get_per_layer_embeddings_ptr_status =
      audio_data.GetMutablePerLayerEmbeddingsPtr();
  ASSERT_TRUE(get_per_layer_embeddings_ptr_status.ok());
  TensorBuffer* get_per_layer_embeddings_ptr =
      get_per_layer_embeddings_ptr_status.value();
  auto get_per_layer_embeddings_size = get_per_layer_embeddings_ptr->Size();
  ASSERT_TRUE(get_per_layer_embeddings_size.HasValue());
  EXPECT_EQ(get_per_layer_embeddings_size.Value(), 16);
  float ple_read_data[4];
  read_success =
      get_per_layer_embeddings_ptr->Read<float>(absl::MakeSpan(ple_read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(ple_read_data[0], 133.0f);
  EXPECT_EQ(ple_read_data[1], 134.0f);
  EXPECT_EQ(ple_read_data[2], 135.0f);
  EXPECT_EQ(ple_read_data[3], 136.0f);

  // Test GetMutablePerLayerEmbeddingsPtr
  auto get_mutable_per_layer_embeddings_ptr_status =
      audio_data.GetMutablePerLayerEmbeddingsPtr();
  ASSERT_TRUE(get_mutable_per_layer_embeddings_ptr_status.ok());
  TensorBuffer* get_mutable_per_layer_embeddings_ptr =
      get_mutable_per_layer_embeddings_ptr_status.value();
  auto get_mutable_per_layer_embeddings_size =
      get_mutable_per_layer_embeddings_ptr->Size();
  ASSERT_TRUE(get_mutable_per_layer_embeddings_size.HasValue());
  EXPECT_EQ(get_mutable_per_layer_embeddings_size.Value(), 16);

  // Test GetValidTokens
  EXPECT_EQ(audio_data.GetValidTokens(), 1);

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {137.0f, 138.0f};
  } new_emb_data;
  // Create a new TensorBuffer for embeddings
  auto new_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      new_emb_data.d, sizeof(new_emb_data.d));
  ASSERT_TRUE(new_embeddings.HasValue());
  // Test SetEmbeddings
  audio_data.SetEmbeddings(std::move(*new_embeddings));
  auto get_new_embeddings_ptr_status = audio_data.GetMutableEmbeddingsPtr();
  ASSERT_TRUE(get_new_embeddings_ptr_status.ok());
  TensorBuffer* get_new_embeddings_ptr = get_new_embeddings_ptr_status.value();
  auto get_new_embeddings_size = get_new_embeddings_ptr->Size();
  ASSERT_TRUE(get_new_embeddings_size.HasValue());
  EXPECT_EQ(get_new_embeddings_size.Value(), 8);
  read_success = get_new_embeddings_ptr->Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(read_data[0], 137.0f);
  EXPECT_EQ(read_data[1], 138.0f);

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[4] = {139.0f, 140.0f, 141.0f, 142.0f};
  } new_ple_emb_data;
  // Create a new TensorBuffer for per_layer_embeddings
  auto new_per_layer_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({4}))),
      new_ple_emb_data.d, sizeof(new_ple_emb_data.d));
  ASSERT_TRUE(new_per_layer_embeddings.HasValue());
  // Test SetPerLayerEmbeddings
  audio_data.SetPerLayerEmbeddings(std::move(*new_per_layer_embeddings));
  auto get_new_per_layer_embeddings_ptr_status =
      audio_data.GetMutablePerLayerEmbeddingsPtr();
  ASSERT_TRUE(get_new_per_layer_embeddings_ptr_status.ok());
  TensorBuffer* get_new_per_layer_embeddings_ptr =
      get_new_per_layer_embeddings_ptr_status.value();
  auto get_new_per_layer_embeddings_size =
      get_new_per_layer_embeddings_ptr->Size();
  ASSERT_TRUE(get_new_per_layer_embeddings_size.HasValue());
  EXPECT_EQ(get_new_per_layer_embeddings_size.Value(), 16);
  read_success = get_new_per_layer_embeddings_ptr->Read<float>(
      absl::MakeSpan(ple_read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(ple_read_data[0], 139.0f);
  EXPECT_EQ(ple_read_data[1], 140.0f);
  EXPECT_EQ(ple_read_data[2], 141.0f);
  EXPECT_EQ(ple_read_data[3], 142.0f);

  // Test SetValidTokens
  audio_data.SetValidTokens(2);
  EXPECT_EQ(audio_data.GetValidTokens(), 2);
}

TEST(LlmExecutorIoTypesTest, ExecutorInputsGetSet) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int32_t d[2] = {7, 8};
  } data;
  auto env = Environment::Create({});
  auto token_ids = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Int32, Layout(Dimensions({2}))),
      data.d, 2 * sizeof(int32_t));
  ASSERT_TRUE(token_ids.HasValue());
  ExecutorTextData text_data(std::move(*token_ids));

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {31.0f, 32.0f};
  } emb_data;
  // Create a TensorBuffer for embeddings
  auto embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      emb_data.d, sizeof(emb_data.d));
  ASSERT_TRUE(embeddings.HasValue());
  ExecutorVisionData vision_data(std::move(*embeddings), std::nullopt);

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {131.0f, 132.0f};
  } audio_emb_data;
  // Create a TensorBuffer for embeddings
  auto audio_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      audio_emb_data.d, sizeof(audio_emb_data.d));
  ASSERT_TRUE(audio_embeddings.HasValue());
  ExecutorAudioData audio_data(std::move(*audio_embeddings), std::nullopt, 1);

  ExecutorInputs inputs(std::move(text_data), std::move(vision_data),
                        std::move(audio_data));

  // Test GetTextDataPtr
  auto get_text_data_ptr_status = inputs.GetMutableTextDataPtr();
  ASSERT_TRUE(get_text_data_ptr_status.ok());
  ExecutorTextData* get_text_data_ptr = get_text_data_ptr_status.value();
  auto get_text_data_size = get_text_data_ptr->GetTokenIds().Size();
  ASSERT_TRUE(get_text_data_size.HasValue());
  EXPECT_EQ(get_text_data_size.Value(), 8);
  int32_t read_data[2];

  auto read_success = get_text_data_ptr->GetMutableTokenIds().Read<int32_t>(
      absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(read_data[0], 7);
  EXPECT_EQ(read_data[1], 8);

  // Test GetMutableTextDataPtr
  auto get_mutable_text_data_ptr_status = inputs.GetMutableTextDataPtr();
  ASSERT_TRUE(get_mutable_text_data_ptr_status.ok());
  ExecutorTextData* get_mutable_text_data_ptr =
      get_mutable_text_data_ptr_status.value();
  auto get_mutable_text_data_size =
      get_mutable_text_data_ptr->GetTokenIds().Size();
  ASSERT_TRUE(get_mutable_text_data_size.HasValue());
  EXPECT_EQ(get_mutable_text_data_size.Value(), 8);

  // Test GetVisionDataPtr
  auto get_vision_data_ptr_status = inputs.GetMutableVisionDataPtr();
  ASSERT_TRUE(get_vision_data_ptr_status.ok());
  ExecutorVisionData* get_vision_data_ptr = get_vision_data_ptr_status.value();
  auto get_vision_embeddings_size =
      get_vision_data_ptr->GetEmbeddingsPtr().value()->Size();
  ASSERT_TRUE(get_vision_embeddings_size.HasValue());
  EXPECT_EQ(get_vision_embeddings_size.Value(), 8);
  float float_read_data[2];
  auto read_success_float =
      get_vision_data_ptr->GetMutableEmbeddingsPtr().value()->Read<float>(
          absl::MakeSpan(float_read_data));
  ASSERT_TRUE(read_success_float);
  EXPECT_EQ(float_read_data[0], 31.0f);
  EXPECT_EQ(float_read_data[1], 32.0f);

  // Test GetMutableVisionDataPtr
  auto get_mutable_vision_data_ptr_status = inputs.GetMutableVisionDataPtr();
  ASSERT_TRUE(get_mutable_vision_data_ptr_status.ok());
  ExecutorVisionData* get_mutable_vision_data_ptr =
      get_mutable_vision_data_ptr_status.value();
  auto get_mutable_vision_embeddings_size =
      get_mutable_vision_data_ptr->GetEmbeddingsPtr().value()->Size();
  ASSERT_TRUE(get_mutable_vision_embeddings_size.HasValue());
  EXPECT_EQ(get_mutable_vision_embeddings_size.Value(), 8);

  // Test GetAudioDataPtr
  auto get_audio_data_ptr_status = inputs.GetMutableAudioDataPtr();
  ASSERT_TRUE(get_audio_data_ptr_status.ok());
  ExecutorAudioData* get_audio_data_ptr = get_audio_data_ptr_status.value();
  auto get_audio_embeddings_size =
      get_audio_data_ptr->GetEmbeddingsPtr().value()->Size();
  ASSERT_TRUE(get_audio_embeddings_size.HasValue());
  EXPECT_EQ(get_audio_embeddings_size.Value(), 8);
  read_success_float =
      get_audio_data_ptr->GetMutableEmbeddingsPtr().value()->Read<float>(
          absl::MakeSpan(float_read_data));
  ASSERT_TRUE(read_success_float);
  EXPECT_EQ(float_read_data[0], 131.0f);
  EXPECT_EQ(float_read_data[1], 132.0f);

  // Test GetMutableAudioDataPtr
  auto get_mutable_audio_data_ptr_status = inputs.GetMutableAudioDataPtr();
  ASSERT_TRUE(get_mutable_audio_data_ptr_status.ok());
  ExecutorAudioData* get_mutable_audio_data_ptr =
      get_mutable_audio_data_ptr_status.value();
  auto get_mutable_audio_embeddings_size =
      get_mutable_audio_data_ptr->GetEmbeddingsPtr().value()->Size();
  ASSERT_TRUE(get_mutable_audio_embeddings_size.HasValue());
  EXPECT_EQ(get_mutable_audio_embeddings_size.Value(), 8);

  // Test GetTextTokenIdsPtr
  auto get_text_token_ids_ptr_status = inputs.GetTextTokenIdsPtr();
  ASSERT_TRUE(get_text_token_ids_ptr_status.ok());
  const TensorBuffer* get_text_token_ids_ptr =
      get_text_token_ids_ptr_status.value();
  auto get_text_token_ids_size = get_text_token_ids_ptr->Size();
  ASSERT_TRUE(get_text_token_ids_size.HasValue());
  EXPECT_EQ(get_text_token_ids_size.Value(), 8);

  // Test GetMutableTextTokenIdsPtr
  auto get_mutable_text_token_ids_ptr_status =
      inputs.GetMutableTextTokenIdsPtr();
  ASSERT_TRUE(get_mutable_text_token_ids_ptr_status.ok());
  TensorBuffer* get_mutable_text_token_ids_ptr =
      get_mutable_text_token_ids_ptr_status.value();
  auto get_mutable_text_token_ids_size = get_mutable_text_token_ids_ptr->Size();
  ASSERT_TRUE(get_mutable_text_token_ids_size.HasValue());
  EXPECT_EQ(get_mutable_text_token_ids_size.Value(), 8);

  // Test GetVisionEmbeddingsPtr
  auto get_vision_embeddings_ptr_status = inputs.GetVisionEmbeddingsPtr();
  ASSERT_TRUE(get_vision_embeddings_ptr_status.ok());
  const TensorBuffer* get_vision_embeddings_ptr =
      get_vision_embeddings_ptr_status.value();
  get_vision_embeddings_size = get_vision_embeddings_ptr->Size();
  ASSERT_TRUE(get_vision_embeddings_size.HasValue());
  EXPECT_EQ(get_vision_embeddings_size.Value(), 8);

  // Test GetMutableVisionEmbeddingsPtr
  auto get_mutable_vision_embeddings_ptr_status =
      inputs.GetMutableVisionEmbeddingsPtr();
  ASSERT_TRUE(get_mutable_vision_embeddings_ptr_status.ok());
  TensorBuffer* get_mutable_vision_embeddings_ptr =
      get_mutable_vision_embeddings_ptr_status.value();
  get_mutable_vision_embeddings_size =
      get_mutable_vision_embeddings_ptr->Size();
  ASSERT_TRUE(get_mutable_vision_embeddings_size.HasValue());
  EXPECT_EQ(get_mutable_vision_embeddings_size.Value(), 8);

  // Test GetVisionPerLayerEmbeddingsPtr
  EXPECT_FALSE(inputs.GetVisionPerLayerEmbeddingsPtr().ok());

  // Test GetMutableVisionPerLayerEmbeddingsPtr
  EXPECT_FALSE(inputs.GetMutableVisionPerLayerEmbeddingsPtr().ok());

  // Test GetAudioEmbeddingsPtr
  auto get_audio_embeddings_ptr_status = inputs.GetAudioEmbeddingsPtr();
  ASSERT_TRUE(get_audio_embeddings_ptr_status.ok());
  const TensorBuffer* get_audio_embeddings_ptr =
      get_audio_embeddings_ptr_status.value();
  get_audio_embeddings_size = get_audio_embeddings_ptr->Size();
  ASSERT_TRUE(get_audio_embeddings_size.HasValue());
  EXPECT_EQ(get_audio_embeddings_size.Value(), 8);

  // Test GetMutableAudioEmbeddingsPtr
  auto get_mutable_audio_embeddings_ptr_status =
      inputs.GetMutableAudioEmbeddingsPtr();
  ASSERT_TRUE(get_mutable_audio_embeddings_ptr_status.ok());
  TensorBuffer* get_mutable_audio_embeddings_ptr =
      get_mutable_audio_embeddings_ptr_status.value();
  get_mutable_audio_embeddings_size = get_mutable_audio_embeddings_ptr->Size();
  ASSERT_TRUE(get_mutable_audio_embeddings_size.HasValue());
  EXPECT_EQ(get_mutable_audio_embeddings_size.Value(), 8);

  // Test GetAudioPerLayerEmbeddingsPtr
  EXPECT_FALSE(inputs.GetAudioPerLayerEmbeddingsPtr().ok());

  // Test GetMutableAudioPerLayerEmbeddingsPtr
  EXPECT_FALSE(inputs.GetMutableAudioPerLayerEmbeddingsPtr().ok());

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int32_t d[2] = {9, 10};
  } new_data;
  auto new_token_ids = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Int32, Layout(Dimensions({2}))),
      new_data.d, 2 * sizeof(int32_t));
  ASSERT_TRUE(new_token_ids.HasValue());
  ExecutorTextData new_text_data(std::move(*new_token_ids));
  // Test SetTextData
  inputs.SetTextData(std::move(new_text_data));
  auto get_new_text_data_ptr_status = inputs.GetMutableTextDataPtr();
  ASSERT_TRUE(get_new_text_data_ptr_status.ok());
  ExecutorTextData* get_new_text_data_ptr =
      get_new_text_data_ptr_status.value();
  auto get_new_text_data_size = get_new_text_data_ptr->GetTokenIds().Size();
  ASSERT_TRUE(get_new_text_data_size.HasValue());
  EXPECT_EQ(get_new_text_data_size.Value(), 8);
  read_success = get_new_text_data_ptr->GetMutableTokenIds().Read<int32_t>(
      absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  EXPECT_EQ(read_data[0], 9);
  EXPECT_EQ(read_data[1], 10);

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {37.0f, 38.0f};
  } new_emb_data;
  // Create a new TensorBuffer for embeddings
  auto new_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      new_emb_data.d, sizeof(new_emb_data.d));
  ASSERT_TRUE(new_embeddings.HasValue());
  ExecutorVisionData new_vision_data(std::move(*new_embeddings), std::nullopt);
  // Test SetVisionData
  inputs.SetVisionData(std::move(new_vision_data));
  auto get_new_vision_data_ptr_status = inputs.GetMutableVisionDataPtr();
  ASSERT_TRUE(get_new_vision_data_ptr_status.ok());
  ExecutorVisionData* get_new_vision_data_ptr =
      get_new_vision_data_ptr_status.value();
  auto get_new_vision_embeddings_size =
      get_new_vision_data_ptr->GetEmbeddingsPtr().value()->Size();
  ASSERT_TRUE(get_new_vision_embeddings_size.HasValue());
  EXPECT_EQ(get_new_vision_embeddings_size.Value(), 8);
  read_success_float =
      get_new_vision_data_ptr->GetMutableEmbeddingsPtr().value()->Read<float>(
          absl::MakeSpan(float_read_data));
  ASSERT_TRUE(read_success_float);
  EXPECT_EQ(float_read_data[0], 37.0f);
  EXPECT_EQ(float_read_data[1], 38.0f);

  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[2] = {137.0f, 138.0f};
  } new_audio_emb_data;
  // Create a new TensorBuffer for embeddings
  auto new_audio_embeddings = TensorBuffer::CreateFromHostMemory(
      *env,
      ::litert::RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
      new_audio_emb_data.d, sizeof(new_audio_emb_data.d));
  ASSERT_TRUE(new_audio_embeddings.HasValue());
  ExecutorAudioData new_audio_data(std::move(*new_audio_embeddings),
                                   std::nullopt, 2);
  // Test SetAudioData
  inputs.SetAudioData(std::move(new_audio_data));
  auto get_new_audio_data_ptr_status = inputs.GetMutableAudioDataPtr();
  ASSERT_TRUE(get_new_audio_data_ptr_status.ok());
  ExecutorAudioData* get_new_audio_data_ptr =
      get_new_audio_data_ptr_status.value();
  auto get_new_audio_embeddings_size =
      get_new_audio_data_ptr->GetEmbeddingsPtr().value()->Size();
  ASSERT_TRUE(get_new_audio_embeddings_size.HasValue());
  EXPECT_EQ(get_new_audio_embeddings_size.Value(), 8);
  read_success_float =
      get_new_audio_data_ptr->GetMutableEmbeddingsPtr().value()->Read<float>(
          absl::MakeSpan(float_read_data));
  ASSERT_TRUE(read_success_float);
  EXPECT_EQ(float_read_data[0], 137.0f);
  EXPECT_EQ(float_read_data[1], 138.0f);
}

TEST(LlmExecutorIoTypesTest, ExecutorPrefillParamsGetSet) {
  std::atomic_bool cancel = true;
  ExecutorPrefillParams params(
      /*current_step=*/10, /*wait_for_completion=*/true, /*cancel=*/&cancel);

  // Test GetCurrentStep
  EXPECT_EQ(params.GetCurrentStep(), 10);

  // Test SetCurrentStep
  params.SetCurrentStep(20);
  EXPECT_EQ(params.GetCurrentStep(), 20);

  // Test GetWaitForCompletion
  EXPECT_EQ(params.GetWaitForCompletion(), true);

  // Test SetWaitForCompletion
  params.SetWaitForCompletion(false);
  EXPECT_EQ(params.GetWaitForCompletion(), false);

  // Test GetCancelFlag
  EXPECT_EQ(params.GetCancelFlag(), &cancel);

  std::atomic_bool new_cancel = false;
  // Test SetCancelFlag
  params.SetCancelFlag(&new_cancel);
  EXPECT_EQ(params.GetCancelFlag(), &new_cancel);

  // Test GetMaxPrefillSequenceLength
  params.SetMaxPrefillSequenceLength(100);
  auto max_prefill_sequence_length_or = params.GetMaxPrefillSequenceLength();
  ASSERT_TRUE(max_prefill_sequence_length_or.ok());
  EXPECT_EQ(max_prefill_sequence_length_or.value(), 100);
}

TEST(LlmExecutorIoTypesTest, ExecutorDecodeParamsGetSet) {
  ExecutorDecodeParams params;
  EXPECT_FALSE(params.HasConstraintDecoder());
  EXPECT_EQ(params.GetConstraintDecoder(), nullptr);

  auto constraint = FakeConstraint({1, 2, 3}, /*vocabulary_size=*/10);
  ConstrainedDecoder constraint_decoder =
      ConstrainedDecoder(&constraint, /*batch_size=*/1);
  params.SetConstraintDecoder(&constraint_decoder);
  EXPECT_TRUE(params.HasConstraintDecoder());
  EXPECT_EQ(params.GetConstraintDecoder(), &constraint_decoder);
}

}  // namespace
}  // namespace litert::lm
