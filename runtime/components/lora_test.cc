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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
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

std::string GetLoRAParamModelFilePath() {
  auto path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/litert_dummy_lora_param_tensor_model.tflite";
  return path.string();
}

// Wraps a LoraData and adds a prefix to all of its tensor names, to simulate a
// LoRA converted with tensor names that don't match the model's inputs.
class RenamedLoraData : public LoraData {
 public:
  explicit RenamedLoraData(std::unique_ptr<LoraData> lora_data)
      : lora_data_(std::move(lora_data)) {}

  absl::StatusOr<int> GetLoRARank() override {
    return lora_data_->GetLoRARank();
  }

  absl::StatusOr<std::unique_ptr<BufferRef<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    if (!absl::ConsumePrefix(&name, kPrefix)) {
      return absl::NotFoundError(absl::StrCat("No tensor: ", name));
    }
    return lora_data_->ReadTensor(name);
  }

  bool HasTensor(absl::string_view name) const override {
    return absl::ConsumePrefix(&name, kPrefix) && lora_data_->HasTensor(name);
  }

  std::vector<std::string> GetAllTensorNames() const override {
    std::vector<std::string> names = lora_data_->GetAllTensorNames();
    for (std::string& name : names) {
      name = absl::StrCat(kPrefix, name);
    }
    return names;
  }

 private:
  static constexpr absl::string_view kPrefix = "renamed_";
  std::unique_ptr<LoraData> lora_data_;
};

// Wraps a LoraData and only exposes the tensors in `names`, to simulate a LoRA
// that covers only some of the model's LoRA inputs.
class SubsetLoraData : public LoraData {
 public:
  SubsetLoraData(std::unique_ptr<LoraData> lora_data,
                 std::set<std::string> names)
      : lora_data_(std::move(lora_data)), names_(std::move(names)) {}

  absl::StatusOr<int> GetLoRARank() override {
    return lora_data_->GetLoRARank();
  }

  absl::StatusOr<std::unique_ptr<BufferRef<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    if (!HasTensor(name)) {
      return absl::NotFoundError(absl::StrCat("No tensor: ", name));
    }
    return lora_data_->ReadTensor(name);
  }

  bool HasTensor(absl::string_view name) const override {
    return names_.contains(std::string(name)) && lora_data_->HasTensor(name);
  }

  std::vector<std::string> GetAllTensorNames() const override {
    return std::vector<std::string>(names_.begin(), names_.end());
  }

 private:
  std::unique_ptr<LoraData> lora_data_;
  std::set<std::string> names_;
};

// A rank 32 LoraData whose tensors are all zeros, with the given names and
// sizes in bytes.
class ZerosLoraData : public LoraData {
 public:
  explicit ZerosLoraData(std::map<std::string, size_t> sizes)
      : sizes_(std::move(sizes)) {}

  absl::StatusOr<int> GetLoRARank() override { return 32; }

  absl::StatusOr<std::unique_ptr<BufferRef<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    auto it = sizes_.find(std::string(name));
    if (it == sizes_.end()) {
      return absl::NotFoundError(absl::StrCat("No tensor: ", name));
    }
    const std::vector<uint8_t> zeros(it->second, 0);
    // The const pointer overload copies the data.
    return std::make_unique<OwningBufferRef<uint8_t>>(
        static_cast<const uint8_t*>(zeros.data()), zeros.size());
  }

  bool HasTensor(absl::string_view name) const override {
    return sizes_.contains(std::string(name));
  }

  std::vector<std::string> GetAllTensorNames() const override {
    std::vector<std::string> names;
    for (const auto& [name, size] : sizes_) {
      names.push_back(name);
    }
    return names;
  }

 private:
  std::map<std::string, size_t> sizes_;
};

// Returns a LoraData matching the LoRA inputs of the `lora_param_tensor` test
// model, so that LoRA::Create accepts it.
absl::StatusOr<std::unique_ptr<LoraData>> CreateLoRAParamModelLoraData(
    const CompiledModel& compiled_model) {
  std::map<std::string, size_t> sizes;
  for (const char* name : {"key_w_prime_left_99", "query_w_prime_right_99"}) {
    LITERT_ASSIGN_OR_RETURN(
        auto buffer, compiled_model.CreateInputBuffer("serving_default", name));
    LITERT_ASSIGN_OR_RETURN(sizes[name], buffer.PackedSize());
  }
  return std::make_unique<ZerosLoraData>(std::move(sizes));
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

TEST_F(LoraTest, NoLoRAParamTensorForModelWithoutIt) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  EXPECT_THAT(lora->GetLoRABuffer("lora_param_tensor"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(LoraTest, FillsLoRAParamTensor) {
  // Model inputs: key_w_prime_left_99 [32, 4, 16], query_w_prime_right_99
  // [48, 32] and lora_param_tensor [1, 1, 1, 7].
  LITERT_ASSERT_OK_AND_ASSIGN(Options options, litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env_, GetLoRAParamModelFilePath(), options));

  ASSERT_OK_AND_ASSIGN(auto lora_data,
                       CreateLoRAParamModelLoraData(compiled_model));
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data),
                                               compiled_model,
                                               "serving_default"));
  ASSERT_OK_AND_ASSIGN(auto buffer, lora->GetLoRABuffer("lora_param_tensor"));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_ptr, litert::TensorBufferScopedLock::Create<const int32_t>(
                             buffer, litert::TensorBuffer::LockMode::kRead));
  auto& [lock, data_ptr] = lock_and_ptr;
  // Element 0 is the start index. The others are the largest non-rank extent
  // over LoRA inputs. The rank is 32, the only dimension both inputs have (not
  // the smallest dimension 4): max(4 * 16, 48) = 64.
  EXPECT_THAT(std::vector<int32_t>(data_ptr, data_ptr + 7),
              ::testing::ElementsAre(0, 64, 64, 64, 64, 64, 64));
}

TEST_F(LoraTest, GetLoRABuffersForOtherSignatureSuccess) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  ASSERT_OK_AND_ASSIGN(auto prefill_buffers, lora->GetLoRABuffers("prefill"));
  EXPECT_EQ(prefill_buffers.size(), 280);

  auto it = prefill_buffers.find("query_w_prime_left_20");
  ASSERT_NE(it, prefill_buffers.end());
  LITERT_ASSERT_OK_AND_ASSIGN(size_t buffer_size, it->second.PackedSize());
  EXPECT_GT(buffer_size, 0);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_ptr,
      litert::TensorBufferScopedLock::Create<const uint16_t>(
          it->second, litert::TensorBuffer::LockMode::kRead));
  auto& [lock, data_ptr] = lock_and_ptr;
  const uint16_t fp16_one = 0x3C00;
  for (size_t i = 0; i < buffer_size / sizeof(uint16_t); ++i) {
    EXPECT_EQ(data_ptr[i], fp16_one);
  }
}

TEST_F(LoraTest, GetLoRABuffersReusesBuffersWhenTypesMatch) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  ASSERT_OK_AND_ASSIGN(auto decode_buffers, lora->GetLoRABuffers("decode"));
  ASSERT_OK_AND_ASSIGN(auto prefill_buffers, lora->GetLoRABuffers("prefill"));
  ASSERT_EQ(decode_buffers.size(), prefill_buffers.size());

  // On CPU both signatures accept host memory, so no extra buffers are created
  // and both signatures share the same underlying buffers.
  for (const auto& [name, decode_buffer] : decode_buffers) {
    auto it = prefill_buffers.find(name);
    ASSERT_NE(it, prefill_buffers.end()) << name;
    EXPECT_EQ(it->second.Get(), decode_buffer.Get()) << name;
  }
}

TEST_F(LoraTest, GetLoRABuffersDefaultsToCreationSignature) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  ASSERT_OK_AND_ASSIGN(auto default_buffers, lora->GetLoRABuffers());
  ASSERT_OK_AND_ASSIGN(auto decode_buffers, lora->GetLoRABuffers("decode"));
  ASSERT_EQ(default_buffers.size(), decode_buffers.size());
  for (const auto& [name, buffer] : default_buffers) {
    auto it = decode_buffers.find(name);
    ASSERT_NE(it, decode_buffers.end()) << name;
    EXPECT_EQ(it->second.Get(), buffer.Get()) << name;
  }
}

TEST_F(LoraTest, GetLoRABuffersIncludesLoRAParamTensor) {
  // Model inputs: key_w_prime_left_99 [32, 4, 16], query_w_prime_right_99
  // [48, 32] and lora_param_tensor [1, 1, 1, 7]. The model only has the
  // "serving_default" signature.
  LITERT_ASSERT_OK_AND_ASSIGN(Options options, litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env_, GetLoRAParamModelFilePath(), options));

  ASSERT_OK_AND_ASSIGN(auto lora_data,
                       CreateLoRAParamModelLoraData(compiled_model));
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data),
                                               compiled_model,
                                               "serving_default"));
  ASSERT_OK_AND_ASSIGN(auto buffers, lora->GetLoRABuffers("serving_default"));
  EXPECT_EQ(buffers.size(), 3);
  auto it = buffers.find("lora_param_tensor");
  ASSERT_NE(it, buffers.end());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_ptr,
      litert::TensorBufferScopedLock::Create<const int32_t>(
          it->second, litert::TensorBuffer::LockMode::kRead));
  auto& [lock, data_ptr] = lock_and_ptr;
  // Same values as in FillsLoRAParamTensor: start 0, then max(4 * 16, 48).
  EXPECT_THAT(std::vector<int32_t>(data_ptr, data_ptr + 7),
              ::testing::ElementsAre(0, 64, 64, 64, 64, 64, 64));
}

TEST_F(LoraTest, GetLoRABuffersReturnsErrorForUnknownSignature) {
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(lora_data_),
                                               *compiled_model_, "decode"));
  EXPECT_THAT(lora->GetLoRABuffers("unknown_signature"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(LoraTest, CreateLoRAFailsWhenNoTensorMatchesModelInputs) {
  auto renamed_lora_data =
      std::make_unique<RenamedLoraData>(std::move(lora_data_));
  EXPECT_THAT(
      LoRA::Create(std::move(renamed_lora_data), *compiled_model_, "decode"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("None of the 280 LoRA inputs")));
}

TEST_F(LoraTest, CreateLoRAZeroFillsInputsMissingFromPartialLoRA) {
  // Only one of the 280 LoRA inputs has a tensor in the LoRA data.
  auto subset_lora_data = std::make_unique<SubsetLoraData>(
      std::move(lora_data_), std::set<std::string>{"query_w_prime_left_20"});
  ASSERT_OK_AND_ASSIGN(auto lora, LoRA::Create(std::move(subset_lora_data),
                                               *compiled_model_, "decode"));

  ASSERT_OK_AND_ASSIGN(auto matched,
                       lora->GetLoRABuffer("query_w_prime_left_20"));
  LITERT_ASSERT_OK_AND_ASSIGN(size_t matched_size, matched.PackedSize());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto matched_lock,
      litert::TensorBufferScopedLock::Create<const uint16_t>(
          matched, litert::TensorBuffer::LockMode::kRead));
  const uint16_t fp16_one = 0x3C00;
  for (size_t i = 0; i < matched_size / sizeof(uint16_t); ++i) {
    EXPECT_EQ(matched_lock.second[i], fp16_one);
  }

  ASSERT_OK_AND_ASSIGN(auto missing,
                       lora->GetLoRABuffer("query_w_prime_left_10"));
  LITERT_ASSERT_OK_AND_ASSIGN(size_t missing_size, missing.PackedSize());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto missing_lock,
      litert::TensorBufferScopedLock::Create<const uint8_t>(
          missing, litert::TensorBuffer::LockMode::kRead));
  for (size_t i = 0; i < missing_size; ++i) {
    EXPECT_EQ(missing_lock.second[i], 0);
  }
}

}  // namespace
}  // namespace litert::lm
