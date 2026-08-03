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

#include "runtime/util/lora_data.h"

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsSupersetOf;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

std::string GetLoraFilePath() {
  auto path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_gpu_lora_rank32_f16_all_ones.tflite";
  return path.string();
}

enum class LoraLoadType {
  kFilePath,
  kScopedFile,
  kBuffer,
};

class LoraDataTest : public ::testing::TestWithParam<LoraLoadType> {
 protected:
  absl::StatusOr<std::unique_ptr<LoraData>> CreateLoraData() {
    const LoraLoadType load_type = GetParam();
    switch (load_type) {
      case LoraLoadType::kFilePath: {
        return LoraData::CreateFromFilePath(GetLoraFilePath());
      }
      case LoraLoadType::kScopedFile: {
        ASSIGN_OR_RETURN(auto model_assets,
                         ::litert::lm::ModelAssets::Create(GetLoraFilePath()));
        ASSIGN_OR_RETURN(auto scoped_file,
                         model_assets.GetOrCreateScopedFile());
        return LoraData::CreateFromScopedFile(std::move(scoped_file));
      }
      case LoraLoadType::kBuffer: {
        ASSIGN_OR_RETURN(auto model_assets,
                         ::litert::lm::ModelAssets::Create(GetLoraFilePath()));
        ASSIGN_OR_RETURN(scoped_file_, model_assets.GetOrCreateScopedFile());
        ASSIGN_OR_RETURN(mapped_file_, ::litert::lm::MemoryMappedFile::Create(
                                           scoped_file_->file()));
        return LoraData::CreateFromBuffer(
            BufferRef<uint8_t>(mapped_file_->data(), mapped_file_->length()));
      }
    }
  }

 private:
  std::shared_ptr<const ScopedFile> scoped_file_;
  std::unique_ptr<MemoryMappedFile> mapped_file_;
};

TEST_P(LoraDataTest, CanCreateLoraData) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora, CreateLoraData());
  EXPECT_NE(lora, nullptr);
}

TEST_P(LoraDataTest, GetLoraRankWorksAsExpected) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora, CreateLoraData());
  EXPECT_THAT(lora->GetLoRARank(), IsOkAndHolds(32));
}

TEST_P(LoraDataTest, ReadTensorDataWorksAsExpected) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora, CreateLoraData());

  for (int i : {0, 5, 10, 15, 20}) {
    const std::string tensor_name =
        absl::StrCat("transformer.layer_", i, ".attn.q.w_prime_left");
    ASSERT_OK_AND_ASSIGN(auto tensor, lora->ReadTensor(tensor_name));
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->Size(), 32 * 2048 * 2);

    const int num_elements = 32 * 2048;
    // 1.0f in float16 is 0x3C00
    const uint16_t expected_value = 0x3C00;
    std::vector<uint16_t> expected_data(num_elements, expected_value);

    const uint16_t* actual_data =
        reinterpret_cast<const uint16_t*>(tensor->Data());
    EXPECT_THAT(std::vector<uint16_t>(actual_data, actual_data + num_elements),
                ElementsAreArray(expected_data))
        << "for tensor: " << tensor_name;
  }
}

TEST_P(LoraDataTest, ReadTensorDataFailsForUnknownTensor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora, CreateLoraData());

  const std::string tensor_name = "unknown_tensor";
  EXPECT_THAT(lora->ReadTensor(tensor_name),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(LoraDataTest, HasTensorWorksAsExpected) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora, CreateLoraData());

  for (int i : {0, 5, 10, 15, 20}) {
    const std::string tensor_name =
        absl::StrCat("transformer.layer_", i, ".attn.q.w_prime_left");
    EXPECT_TRUE(lora->HasTensor(tensor_name));
  }

  EXPECT_FALSE(lora->HasTensor("unknown_tensor"));
}

TEST_P(LoraDataTest, GetAllTensorNamesWorksAsExpected) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora, CreateLoraData());
  std::vector<std::string> tensor_names = lora->GetAllTensorNames();
  std::vector<std::string> expected_subset;
  for (int i : {0, 5, 10, 15, 20}) {
    expected_subset.push_back(
        absl::StrCat("transformer.layer_", i, ".attn.q.w_prime_left"));
  }
  EXPECT_THAT(tensor_names, IsSupersetOf(expected_subset));
}

INSTANTIATE_TEST_SUITE_P(
    LoraDataTests, LoraDataTest,
    ::testing::Values(LoraLoadType::kFilePath, LoraLoadType::kScopedFile,
                      LoraLoadType::kBuffer),
    [](const ::testing::TestParamInfo<LoraDataTest::ParamType>& info) {
      switch (info.param) {
        case LoraLoadType::kFilePath:
          return "FilePath";
        case LoraLoadType::kScopedFile:
          return "ScopedFile";
        case LoraLoadType::kBuffer:
          return "Buffer";
      }
    });

}  // namespace
}  // namespace litert::lm
