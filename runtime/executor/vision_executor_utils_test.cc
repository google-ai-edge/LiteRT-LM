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

#include "runtime/executor/vision_executor_utils.h"

#include <cstddef>
#include <filesystem>  // NOLINT
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr absl::string_view kTestVisionModelWithAdapterPath =
    "litert_lm/runtime/testdata/dummy_vision_with_adapter.litertlm";
constexpr absl::string_view kTestVisionModelWithoutAdapterPath =
    "litert_lm/runtime/testdata/dummy_vision_without_adapter.litertlm";

class FakeModelResources : public ModelResources {
 public:
  FakeModelResources(const litert::Model* encoder, const litert::Model* adapter,
                     absl::Status encoder_status = absl::OkStatus(),
                     absl::Status adapter_status = absl::OkStatus())
      : encoder_(encoder),
        adapter_(adapter),
        encoder_status_(encoder_status),
        adapter_status_(adapter_status) {}

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    if (model_type == ModelType::kTfLiteVisionEncoder) {
      if (!encoder_status_.ok()) {
        return encoder_status_;
      }
      return encoder_;
    }
    if (model_type == ModelType::kTfLiteVisionAdapter) {
      if (!adapter_status_.ok()) {
        return adapter_status_;
      }
      if (adapter_ == nullptr) {
        return absl::NotFoundError("Adapter not found");
      }
      return adapter_;
    }
    return absl::NotFoundError("Model not found");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError("");
  }

 private:
  const litert::Model* encoder_;
  const litert::Model* adapter_;
  absl::Status encoder_status_;
  absl::Status adapter_status_;
};

TEST(VisionExecutorUtilsTest, GetPropertiesWithAdapter) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(*resources));

  // For dummy_vision_with_adapter:
  // Adapter output: [1, 5, 6] -> num_tokens_per_image = 5
  EXPECT_EQ(properties.num_tokens_per_image, 5);
}

TEST(VisionExecutorUtilsTest, GetPropertiesWithoutAdapter) {
  const std::string model_path =
      (std::filesystem::path(::testing::SrcDir()) /
       std::string(kTestVisionModelWithoutAdapterPath))
          .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(*resources));

  // For dummy_vision_without_adapter:
  // Encoder output: [1, 5, 5] -> num_tokens_per_image = 5
  EXPECT_EQ(properties.num_tokens_per_image, 5);
}

TEST(VisionExecutorUtilsTest, GetPropertiesWithoutAdapterMocked) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto encoder_model,
      resources->GetTFLiteModel(ModelType::kTfLiteVisionEncoder));

  // Hide the adapter
  FakeModelResources fake_resources(encoder_model, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(fake_resources));

  // Should fallback to encoder output: [1, 5, 5] -> num_tokens_per_image = 5
  EXPECT_EQ(properties.num_tokens_per_image, 5);
}

TEST(VisionExecutorUtilsTest, EncoderError) {
  FakeModelResources fake_resources(
      nullptr, nullptr, absl::InternalError("Encoder error"), absl::OkStatus());
  auto properties_or =
      GetVisionExecutorPropertiesFromModelResources(fake_resources);
  EXPECT_EQ(properties_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(properties_or.status().message(),
              ::testing::HasSubstr("Encoder error"));
}

TEST(VisionExecutorUtilsTest, AdapterError) {
  FakeModelResources fake_resources(nullptr, nullptr, absl::OkStatus(),
                                    absl::InternalError("Adapter error"));
  auto properties_or =
      GetVisionExecutorPropertiesFromModelResources(fake_resources);
  EXPECT_EQ(properties_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(properties_or.status().message(),
              ::testing::HasSubstr("Adapter error"));
}

}  // namespace
}  // namespace litert::lm
