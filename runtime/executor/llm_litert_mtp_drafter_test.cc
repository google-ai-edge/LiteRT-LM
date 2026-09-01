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

#include "runtime/executor/llm_litert_mtp_drafter.h"

#include <cstddef>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

class TfLiteModelResources : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<TfLiteModelResources>> Create(
      const ModelAssets& model_assets, bool with_mtp_drafter = false) {
    LITERT_ASSIGN_OR_RETURN(auto path, model_assets.GetPath());
    LITERT_ASSIGN_OR_RETURN(auto model,
                            Model::CreateFromFile(std::string(path)));
    return absl::WrapUnique(
        new TfLiteModelResources(std::move(model), with_mtp_drafter));
  }

  absl::StatusOr<const Model*> GetTFLiteModel(ModelType model_type) override {
    if (model_type == ModelType::kTfLitePrefillDecode) {
      return &model_;
    }
    if (model_type == ModelType::kTfLiteMtpDrafter) {
      if (with_mtp_drafter_) {
        return &model_;
      } else {
        return absl::NotFoundError("MTP Drafter model not found");
      }
    }
    return absl::UnimplementedError("Unsupported model type");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return absl::UnimplementedError("GetTFLiteModelBuffer not implemented.");
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("GetTokenizer not implemented.");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("GetLlmMetadata not implemented.");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("GetExecutorMetadata not implemented.");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("GetScopedFile not implemented.");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

 private:
  explicit TfLiteModelResources(Model model, bool with_mtp_drafter = false)
      : model_(std::move(model)), with_mtp_drafter_(with_mtp_drafter) {}

  Model model_;
  bool with_mtp_drafter_;
};

TEST(LlmLiteRtMtpDrafterTest, CreateFromModelResources_MissingVerifySignature) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto executor_settings,
                       LlmExecutorSettings::CreateDefault(model_assets));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto base_model, CompiledModel::Create(env, model_path.string(),
                                             litert::HwAccelerators::kCpu));
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       TfLiteModelResources::Create(model_assets,
                                                    /*with_mtp_drafter=*/true));

  const std::filesystem::path embedder_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/dummy_embedder.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(auto embedder_model,
                              Model::CreateFromFile(embedder_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto embedding_lookup,
      EmbeddingLookupManager::Create(env, &embedder_model,
                                     /*fully_supports_multi_modal=*/false));

  auto drafter_or = LlmLiteRtMtpDrafter::Create(
      env, *model_resources, executor_settings, base_model, *embedding_lookup,
      /*ple_manager=*/std::nullopt);
  EXPECT_THAT(drafter_or, StatusIs(absl::StatusCode::kNotFound));
}

TEST(LlmLiteRtMtpDrafterTest,
     CreateFromPreCompiledModel_MissingVerifySignature) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto executor_settings,
                       LlmExecutorSettings::CreateDefault(model_assets));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto mtp_model,
                              Model::CreateFromFile(model_path.string()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_mtp_model,
      CompiledModel::Create(env, model_path.string(),
                            litert::HwAccelerators::kCpu));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto base_model, CompiledModel::Create(env, model_path.string(),
                                             litert::HwAccelerators::kCpu));

  const std::filesystem::path embedder_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/dummy_embedder.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(auto embedder_model,
                              Model::CreateFromFile(embedder_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto embedding_lookup,
      EmbeddingLookupManager::Create(env, &embedder_model,
                                     /*fully_supports_multi_modal=*/false));

  auto drafter_or = LlmLiteRtMtpDrafter::Create(
      env, std::move(compiled_mtp_model), executor_settings, base_model,
      mtp_model, *embedding_lookup,
      /*ple_manager=*/std::nullopt);
  EXPECT_THAT(drafter_or, StatusIs(absl::StatusCode::kNotFound));
}

TEST(LlmLiteRtMtpDrafterTest, UpdateCompilationOptions) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto settings_gpu, LlmExecutorSettings::CreateDefault(
                                              model_assets, Backend::GPU));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  EXPECT_OK(UpdateCompilationOptions(settings_gpu, options));

  ASSERT_OK_AND_ASSIGN(auto settings_cpu, LlmExecutorSettings::CreateDefault(
                                              model_assets, Backend::CPU));
  EXPECT_OK(UpdateCompilationOptions(settings_cpu, options));
}

}  // namespace
}  // namespace litert::lm
