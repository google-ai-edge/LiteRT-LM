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

#include "runtime/components/lora.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/lora_data.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Options;
using ::testing::status::StatusIs;

std::string GetLoraFilePath() {
  auto path = std::filesystem::path(::testing::SrcDir()) /
              "litert_lm/runtime/testdata/test_lora_rank32_f16_all_ones.tflite";
  return path.string();
}

std::string GetModelFilePath() {
  auto path = std::filesystem::path(::testing::SrcDir()) /
              "litert_lm/runtime/testdata/litert_dummy_lora32_f16_model.tflite";
  return path.string();
}

class LoraTest : public ::testing::Test {
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

    ASSERT_OK_AND_ASSIGN(lora_data_,
                         LoraData::CreateFromFilePath(GetLoraFilePath()));
  }

  std::unique_ptr<Environment> env_;
  std::unique_ptr<CompiledModel> compiled_model_;
  std::unique_ptr<LoraData> lora_data_;
};

TEST_F(LoraTest, CreateLoRASuccess) {
  EXPECT_OK(LoRA::Create(std::move(lora_data_), *compiled_model_, "decode"));
}

TEST_F(LoraTest, GetLoRABufferSuccess) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  ASSERT_OK_AND_ASSIGN(auto buffer,
                       lora->GetLoRABuffer("query_w_prime_left_20"));

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

TEST_F(LoraTest, GetLoRABufferReturnsZerosForNoData) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  // Test lora doesn't have k/v for layer > 20.
  ASSERT_OK_AND_ASSIGN(auto buffer,
                       lora->GetLoRABuffer("value_w_prime_left_20"));

  LITERT_ASSERT_OK_AND_ASSIGN(size_t buffer_size, buffer.PackedSize());
  EXPECT_GT(buffer_size, 0);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_ptr, litert::TensorBufferScopedLock::Create<const uint8_t>(
                             buffer, litert::TensorBuffer::LockMode::kRead));

  auto& [lock, data_ptr] = lock_and_ptr;

  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(data_ptr[i], 0);
  }
}

TEST_F(LoraTest, GetLoRABufferReturnsErrorForUnknownTensor) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  EXPECT_THAT(lora->GetLoRABuffer("unknown_tensor"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(LoraTest, GetLoRABuffersSuccess) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  ASSERT_OK_AND_ASSIGN(auto buffers, lora->GetLoRABuffers());

  // There are 280 LoRA tensors in the model.
  EXPECT_EQ(buffers.size(), 280);

  // Spot check a few tensors.
  EXPECT_TRUE(buffers.contains("query_w_prime_left_10"));
  EXPECT_TRUE(buffers.contains("value_w_prime_right_15"));
  EXPECT_TRUE(buffers.contains("key_w_prime_left_0"));
  EXPECT_TRUE(buffers.contains("post_w_prime_right_30"));
}

}  // namespace
}  // namespace litert::lm
