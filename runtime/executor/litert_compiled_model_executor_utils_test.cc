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

#include "runtime/executor/litert_compiled_model_executor_utils.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::_;  // NOLINT: Required by ASSERT_OK_AND_ASSIGN().
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::status::StatusIs;

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_Gemma2JAX) {
  std::vector<absl::string_view> input_names = {"token_ids", "positions",
                                                "attn_mask"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_EQ(signatures.input_tokens, "token_ids");
  EXPECT_EQ(signatures.input_positions, "positions");
  EXPECT_EQ(signatures.input_attn_mask, "attn_mask");
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_PyTorch) {
  std::vector<absl::string_view> input_names = {"tokens", "input_pos", "mask"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_EQ(signatures.input_tokens, "tokens");
  EXPECT_EQ(signatures.input_positions, "input_pos");
  EXPECT_EQ(signatures.input_attn_mask, "mask");
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_PyTorchCpuOnly) {
  std::vector<absl::string_view> input_names = {"tokens", "input_pos"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_EQ(signatures.input_tokens, "tokens");
  EXPECT_EQ(signatures.input_positions, "input_pos");
  EXPECT_FALSE(signatures.input_attn_mask.has_value());
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_Gemini) {
  std::vector<absl::string_view> input_names = {"token_ids", "positions",
                                                "attn_mask"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_EQ(signatures.input_tokens, "token_ids");
  EXPECT_EQ(signatures.input_positions, "positions");
  EXPECT_EQ(signatures.input_attn_mask, "attn_mask");
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_ExternalEmbeddingModel) {
  std::vector<absl::string_view> input_names = {
      "input_pos", "mask", "embeddings", "per_layer_embeddings"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_TRUE(signatures.input_tokens.empty());
  EXPECT_EQ(signatures.input_positions, "input_pos");
  EXPECT_EQ(signatures.input_attn_mask, "mask");
  EXPECT_EQ(signatures.input_embeddings, "embeddings");
  EXPECT_EQ(signatures.input_per_layer_embeddings, "per_layer_embeddings");
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_ExternalEmbeddingModelWithoutPle) {
  std::vector<absl::string_view> input_names = {"input_pos", "mask",
                                                "embeddings"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_TRUE(signatures.input_tokens.empty());
  EXPECT_EQ(signatures.input_positions, "input_pos");
  EXPECT_EQ(signatures.input_attn_mask, "mask");
  EXPECT_EQ(signatures.input_embeddings, "embeddings");
  EXPECT_FALSE(signatures.input_per_layer_embeddings.has_value());
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_Unsupported) {
  std::vector<absl::string_view> input_names = {"unknown_input"};
  std::vector<absl::string_view> output_names = {"logits"};
  EXPECT_THAT(GetModelSignaturesFromInputOutputNames(input_names, output_names),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetKVCacheRootNames_KvCache) {
  std::vector<absl::string_view> input_names = {"kv_cache_k_0", "kv_cache_v_0"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "kv_cache_k_");
  EXPECT_EQ(v_root_name, "kv_cache_v_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetKVCacheRootNames_KCache) {
  std::vector<absl::string_view> input_names = {"k_cache_0", "v_cache_0"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "k_cache_");
  EXPECT_EQ(v_root_name, "v_cache_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetKVCacheRootNames_KCacheOutput) {
  std::vector<absl::string_view> input_names = {};
  std::vector<absl::string_view> output_names = {"k_cache_0", "v_cache_0"};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "k_cache_");
  EXPECT_EQ(v_root_name, "v_cache_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetKVCacheRootNames_NotFound) {
  std::vector<absl::string_view> input_names = {"other_input"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  EXPECT_THAT(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetKVCacheRootNames_KvCacheC) {
  std::vector<absl::string_view> input_names = {"kv_cache_c_0"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "kv_cache_c_");
  EXPECT_EQ(v_root_name, "kv_cache_c_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillSingleBufferCacheParamTensor) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  auto layout = ::litert::Layout(::litert::Dimensions({12}));
  RankedTensorType ranked_tensor_type(ElementType::Int32, std::move(layout));
  auto param_tensor = TensorBuffer::CreateManaged(
      env, ::litert::TensorBufferType::kHostMemory, ranked_tensor_type,
      sizeof(int) * 12);
  ASSERT_TRUE(param_tensor);

  ASSERT_OK(FillSingleBufferCacheParamTensor(*param_tensor, /*start_index=*/5,
                                             /*update_length=*/10));
  auto lock = litert::TensorBufferScopedLock::Create(
      *param_tensor, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  int* param_ptr = static_cast<int*>(lock->second);
  EXPECT_EQ(param_ptr[0], 5);
  EXPECT_EQ(param_ptr[1], 15);
  EXPECT_EQ(param_ptr[2], 15);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_ExactMultiples) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 2048));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_1024", 1024),
                                       Pair("prefill_1024", 1024)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_MixedRunners) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 1536));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_1024", 1024),
                                       Pair("prefill_512", 512)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_RemainderWithSmallerRunner) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  // 1100 = 1024 + 76. The smallest runner >= 76 is prefill_128.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 1100));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_1024", 1024), Pair("prefill_128", 76)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_SingleRunner) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 1024));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_1024", 1024)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_LargerThanLargest) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {128, "prefill_128"}, {32, "prefill_32"}};
  // 600 = 512 + 88. The smallest runner >= 88 is prefill_128.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 600));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_512", 512), Pair("prefill_128", 88)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_SmallestRunnerOnly) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  // 100 < 128. Uses the smallest runner.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 100));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_128", 100)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetPrefillRunnerSetFromModel) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK_AND_ASSIGN(auto litert_model, model_resources->GetTFLiteModel(
                                              ModelType::kTfLitePrefillDecode));
  ASSERT_NE(litert_model, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto prefill_runner_set,
      GetPrefillRunnerSetFromModel(*litert_model, "prefill",
                                   /*input_positions_name=*/"input_pos"));
  EXPECT_THAT(prefill_runner_set, ElementsAre(Pair(160, "prefill")));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, InitializeAttentionMask_Float32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 1, 128}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 128);
  ASSERT_TRUE(mask_buffer);

  // Test with is_f16 = false
  {
    ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));
    auto lock = litert::TensorBufferScopedLock::Create(
        *mask_buffer, litert::TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock);
    float* mask_ptr = static_cast<float*>(lock->second);
    float expected_val = -0.7f * std::numeric_limits<float>::max();
    int count_float = 128;
    for (int i = 0; i < count_float; ++i) {
      EXPECT_EQ(mask_ptr[i], expected_val);
    }
  }

  // Test with is_f16 = true
  {
    ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/true));
    auto lock = litert::TensorBufferScopedLock::Create(
        *mask_buffer, litert::TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock);
    float* mask_ptr = static_cast<float*>(lock->second);
    float expected_val = -45824.0f;
    int count_float = 128;
    for (int i = 0; i < count_float; ++i) {
      EXPECT_EQ(mask_ptr[i], expected_val);
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, InitializeAttentionMask_Float16) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 1, 128}));
  RankedTensorType ranked_tensor_type(ElementType::Float16, std::move(layout));
  auto mask_buffer = TensorBuffer::CreateManaged(
      env, ::litert::TensorBufferType::kHostMemory, ranked_tensor_type,
      sizeof(tflite::half) * 128);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/true));
  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  tflite::half* mask_ptr = static_cast<tflite::half*>(lock->second);
  float expected_val = -45824.0f;
  int count_float = 128;
  for (int i = 0; i < count_float; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(mask_ptr[i]), expected_val);
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, InitializeAttentionMask_Bool) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 1, 128}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 128);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));
  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);
  int count_bool = 128;
  for (int i = 0; i < count_bool; ++i) {
    EXPECT_FALSE(mask_ptr[i]);
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, FillAttentionMask_Float32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  // Mask shape: [batch=1, seq_len=4, 1, max_kv_len=10]
  auto layout = ::litert::Layout(::litert::Dimensions({1, 4, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 40);
  ASSERT_TRUE(mask_buffer);

  // Initialize the mask with the default float value.
  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // Fill attention mask starting from timestep 5 for 2 steps.
  // channel_size = 10
  // i = 0: current_step = 5. Fills indices [0, 5] with 0.0f.
  // i = 1: current_step = 6. Fills indices [10, 16] with 0.0f.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/5, /*steps=*/2));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);
  float init_val = -0.7f * std::numeric_limits<float>::max();

  for (int i = 0; i < 40; ++i) {
    if ((i >= 0 && i <= 5) || (i >= 10 && i <= 16)) {
      EXPECT_EQ(mask_ptr[i], 0.0f) << " at index " << i;
    } else {
      EXPECT_EQ(mask_ptr[i], init_val) << " at index " << i;
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_Float32_MultipleBatches) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  // Mask shape: [batch=1, seq_len=4, 1, max_kv_len=10]
  auto layout = ::litert::Layout(::litert::Dimensions({3, 4, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 120);
  ASSERT_TRUE(mask_buffer);

  // Initialize the mask with the default float value.
  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // Fill attention mask starting from timestep 5 for 2 steps.
  // channel_size = 10
  // i = 0: current_step = 5. Fills indices [0, 5] with 0.0f.
  // i = 1: current_step = 6. Fills indices [10, 16] with 0.0f.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/5, /*steps=*/2));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);
  float init_val = -0.7f * std::numeric_limits<float>::max();

  for (int b = 0; b < 3; ++b) {
    for (int i = 0; i < 40; ++i) {
      if ((i >= 0 && i <= 5) || (i >= 10 && i <= 16)) {
        EXPECT_EQ(mask_ptr[b * 40 + i], 0.0f) << " at index " << i;
      } else {
        EXPECT_EQ(mask_ptr[b * 40 + i], init_val) << " at index " << i;
      }
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, FillAttentionMask_Bool) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  // Mask shape: [batch=1, seq_len=3, 1, max_kv_len=8]
  auto layout = ::litert::Layout(::litert::Dimensions({1, 3, 1, 8}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 24);
  ASSERT_TRUE(mask_buffer);

  // Initialize the mask with the default bool value (false).
  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // Fill attention mask starting from timestep 2 for 3 steps.
  // channel_size = 8
  // i = 0: current_step = 2. Fills indices [0, 2] with true.
  // i = 1: current_step = 3. Fills indices [8, 11] with true.
  // i = 2: current_step = 4. Fills indices [16, 20] with true.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/2, /*steps=*/3));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);

  for (int i = 0; i < 24; ++i) {
    if ((i >= 0 && i <= 2) || (i >= 8 && i <= 11) || (i >= 16 && i <= 20)) {
      EXPECT_TRUE(mask_ptr[i]) << " at index " << i;
    } else {
      EXPECT_FALSE(mask_ptr[i]) << " at index " << i;
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_Bool_MultipleBatches) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  // Mask shape: [batch=1, seq_len=3, 1, max_kv_len=8]
  auto layout = ::litert::Layout(::litert::Dimensions({4, 3, 1, 8}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 96);
  ASSERT_TRUE(mask_buffer);

  // Initialize the mask with the default bool value (false).
  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // Fill attention mask starting from timestep 2 for 3 steps.
  // channel_size = 8
  // i = 0: current_step = 2. Fills indices [0, 2] with true.
  // i = 1: current_step = 3. Fills indices [8, 11] with true.
  // i = 2: current_step = 4. Fills indices [16, 20] with true.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/2, /*steps=*/3));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);

  for (int b = 0; b < 4; ++b) {
    for (int i = 0; i < 24; ++i) {
      if ((i >= 0 && i <= 2) || (i >= 8 && i <= 11) || (i >= 16 && i <= 20)) {
        EXPECT_TRUE(mask_ptr[b * 24 + i]) << " at index " << i;
      } else {
        EXPECT_FALSE(mask_ptr[b * 24 + i]) << " at index " << i;
      }
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesTaskBundleFromPath) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";

  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesTaskBundleFromScopedFile) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));

  auto model_assets =
      ModelAssets::Create(std::make_shared<ScopedFile>(std::move(model_file)));
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesTaskBundleFromMmapFile) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto mmap_file,
                       MemoryMappedFile::Create(model_path.string()));
  auto model_assets = ModelAssets::Create(std::move(mmap_file));
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesLitertLmFromMmapFile) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";

  ASSERT_OK_AND_ASSIGN(auto mmap_file,
                       MemoryMappedFile::Create(model_path.string()));
  auto model_assets = ModelAssets::Create(std::move(mmap_file));
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     WrapOrCreateTensorBufferFromHostMemory_SizeMismatch) {
  auto tensor_type = MakeRankedTensorType<float>({1, 1536});
  std::vector<float> host_data(2560, 0.0f);
  absl::Span<float> host_span = absl::MakeSpan(host_data);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto maybe_wrapped,
      WrapOrCreateTensorBufferFromHostMemory(tensor_type, host_span));

  // The behavior depends on whether the underlying runtime allows wrapping
  // mismatched sizes. If it fails, it should return a non-wrapped managed
  // buffer.
  if (!maybe_wrapped.wrapped) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto packed_size,
                                maybe_wrapped.buffer.PackedSize());
    EXPECT_EQ(packed_size, 1536 * sizeof(float));
  }
}

}  // namespace
}  // namespace litert::lm
