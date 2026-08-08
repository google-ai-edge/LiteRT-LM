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

#include "runtime/executor/audio_executor_utils.h"

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

constexpr absl::string_view kTestAudioModelPath =
    "litert_lm/runtime/testdata/dummy_audio_only.litertlm";
constexpr absl::string_view kTestAudioModelNoMaskPath =
    "litert_lm/runtime/testdata/dummy_audio_no_mask.litertlm";

class FakeModelResources : public ModelResources {
 public:
  FakeModelResources(const litert::Model* encoder, const litert::Model* adapter)
      : encoder_(encoder), adapter_(adapter) {}

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    if (model_type == ModelType::kTfLiteAudioEncoderHw) {
      return encoder_;
    }
    if (model_type == ModelType::kTfLiteAudioAdapter) {
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
};

TEST(AudioExecutorUtilsTest, GetPropertiesWithAdapter) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestAudioModelPath))
                                     .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetAudioExecutorPropertiesFromModelResources(*resources));

  // dummy_audio_only.litertlm is non-streaming
  EXPECT_FALSE(properties.is_streaming_model);
  EXPECT_EQ(properties.audio_shrink_factor, 2);
}

TEST(AudioExecutorUtilsTest, GetPropertiesWithoutAdapter) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestAudioModelPath))
                                     .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto encoder_model,
      resources->GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw));

  // Hide the adapter
  FakeModelResources fake_resources(encoder_model, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetAudioExecutorPropertiesFromModelResources(fake_resources));

  // Should still be a streaming model, and should not crash trying to load
  // feature states.
  EXPECT_TRUE(properties.is_streaming_model);
  EXPECT_EQ(properties.audio_shrink_factor, 2);
}

TEST(AudioExecutorUtilsTest, GetPropertiesWithoutMask) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestAudioModelNoMaskPath))
                                     .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetAudioExecutorPropertiesFromModelResources(*resources));

  EXPECT_TRUE(properties.is_streaming_model);
  EXPECT_EQ(properties.audio_shrink_factor, 2);
  EXPECT_EQ(properties.streaming_chunk_size, 10);
}

}  // namespace
}  // namespace litert::lm
