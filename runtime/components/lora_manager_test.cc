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

#include "runtime/components/lora_manager.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Model;
using ::litert::Options;
using ::testing::status::StatusIs;

std::string GetLoraOnesFilePath() {
  auto path = std::filesystem::path(::testing::SrcDir()) /
              "litert_lm/runtime/testdata/test_lora_rank32_f16_all_ones.tflite";
  return path.string();
}

std::string GetLoraTwosFilePath() {
  auto path = std::filesystem::path(::testing::SrcDir()) /
              "litert_lm/runtime/testdata/test_lora_rank32_f16_all_twos.tflite";
  return path.string();
}

std::string GetModelFilePath() {
  auto path = std::filesystem::path(::testing::SrcDir()) /
              "litert_lm/runtime/testdata/litert_dummy_lora32_f16_model.tflite";
  return path.string();
}

class LoraManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Environment setup.
    LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
    env_ = std::make_unique<Environment>(std::move(env));

    LITERT_ASSERT_OK_AND_ASSIGN(Options compilation_options,
                                litert::Options::Create());

    compilation_options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);

    // Create CompiledModel.
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto compiled_model,
        CompiledModel::Create(*env_, GetModelFilePath(), compilation_options));
    compiled_model_ =
        std::make_unique<CompiledModel>(std::move(compiled_model));
    ASSERT_TRUE(*compiled_model_);

    ASSERT_OK_AND_ASSIGN(lora_manager_,
                         LoraManager::Create(*compiled_model_, "decode"));
  }

  std::unique_ptr<Environment> env_;
  std::unique_ptr<CompiledModel> compiled_model_;
  std::unique_ptr<LoraManager> lora_manager_;
};

TEST_F(LoraManagerTest, LoadLoRASuccess) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(GetLoraOnesFilePath()));
  EXPECT_OK(lora_manager_->LoadLoRA(0, model_assets));
}

TEST_F(LoraManagerTest, UseLoRASuccess) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(GetLoraOnesFilePath()));
  ASSERT_OK(lora_manager_->LoadLoRA(0, model_assets));
  EXPECT_OK(lora_manager_->UseLoRA(0));
}

TEST_F(LoraManagerTest, UseLoRAUnknownIdFails) {
  EXPECT_THAT(lora_manager_->UseLoRA(1), StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(LoraManagerTest, GetCurrentLoRAIdSuccess) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(GetLoraOnesFilePath()));
  EXPECT_EQ(lora_manager_->GetCurrentLoRAId(), std::nullopt);
  ASSERT_OK(lora_manager_->LoadLoRA(0, model_assets));
  EXPECT_EQ(lora_manager_->GetCurrentLoRAId(), std::nullopt);
  EXPECT_OK(lora_manager_->UseLoRA(0));
  EXPECT_EQ(lora_manager_->GetCurrentLoRAId(), 0);
}

TEST_F(LoraManagerTest, GetLoRABuffersFailsBeforeUse) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(GetLoraOnesFilePath()));
  ASSERT_OK(lora_manager_->LoadLoRA(0, model_assets));
  EXPECT_THAT(lora_manager_->GetLoRABuffers(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(LoraManagerTest, GetLoRABuffersSuccess) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(GetLoraOnesFilePath()));
  ASSERT_OK(lora_manager_->LoadLoRA(0, model_assets));
  ASSERT_OK(lora_manager_->UseLoRA(0));

  ASSERT_OK_AND_ASSIGN(auto buffers, lora_manager_->GetLoRABuffers());
  EXPECT_EQ(buffers.size(), 280);

  // Spot check a tensor.
  auto it = buffers.find("query_w_prime_left_10");
  ASSERT_NE(it, buffers.end());
  auto& buffer = it->second;

  LITERT_ASSERT_OK_AND_ASSIGN(size_t buffer_size, buffer.PackedSize());
  EXPECT_GT(buffer_size, 0);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_ptr, litert::TensorBufferScopedLock::Create<const uint16_t>(
                             buffer, litert::TensorBuffer::LockMode::kRead));

  auto& [lock, data_ptr] = lock_and_ptr;
  size_t num_elements = buffer_size / sizeof(uint16_t);

  const uint16_t fp16_one = 0x3C00;
  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_EQ(data_ptr[i], fp16_one);
  }
}

TEST_F(LoraManagerTest, LoadMultipleLoRAsSuccess) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets_ones,
                       ModelAssets::Create(GetLoraOnesFilePath()));
  ASSERT_OK(lora_manager_->LoadLoRA(0, model_assets_ones));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets_twos,
                       ModelAssets::Create(GetLoraTwosFilePath()));
  ASSERT_OK(lora_manager_->LoadLoRA(1, model_assets_twos));
}

TEST_F(LoraManagerTest, SwitchBetweenLoRAs) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets_ones,
                       ModelAssets::Create(GetLoraOnesFilePath()));
  ASSERT_OK(lora_manager_->LoadLoRA(0, model_assets_ones));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets_twos,
                       ModelAssets::Create(GetLoraTwosFilePath()));
  ASSERT_OK(lora_manager_->LoadLoRA(1, model_assets_twos));

  const uint16_t fp16_one = 0x3C00, fp16_two = 0x4000;
  // Use LoRA 0 (all ones).
  ASSERT_OK(lora_manager_->UseLoRA(0));
  ASSERT_OK_AND_ASSIGN(auto buffers0, lora_manager_->GetLoRABuffers());
  auto it0 = buffers0.find("query_w_prime_left_10");
  ASSERT_NE(it0, buffers0.end());
  auto& buffer0 = it0->second;
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_ptr0,
        litert::TensorBufferScopedLock::Create<const uint16_t>(
            buffer0, litert::TensorBuffer::LockMode::kRead));
    EXPECT_EQ(lock_and_ptr0.second[0], fp16_one);
  }

  // Use LoRA 1 (all twos).
  ASSERT_OK(lora_manager_->UseLoRA(1));
  ASSERT_OK_AND_ASSIGN(auto buffers1, lora_manager_->GetLoRABuffers());
  auto it1 = buffers1.find("query_w_prime_left_10");
  ASSERT_NE(it1, buffers1.end());
  auto& buffer1 = it1->second;
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_ptr1,
        litert::TensorBufferScopedLock::Create<const uint16_t>(
            buffer1, litert::TensorBuffer::LockMode::kRead));
    EXPECT_EQ(lock_and_ptr1.second[0], fp16_two);
  }

  // Switch back to LoRA 0.
  ASSERT_OK(lora_manager_->UseLoRA(0));
  ASSERT_OK_AND_ASSIGN(auto buffers2, lora_manager_->GetLoRABuffers());
  auto it2 = buffers2.find("query_w_prime_left_10");
  ASSERT_NE(it2, buffers2.end());
  auto& buffer2 = it2->second;
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_ptr2,
        litert::TensorBufferScopedLock::Create<const uint16_t>(
            buffer2, litert::TensorBuffer::LockMode::kRead));
    EXPECT_EQ(lock_and_ptr2.second[0], fp16_one);
  }
}

}  // namespace
}  // namespace litert::lm
