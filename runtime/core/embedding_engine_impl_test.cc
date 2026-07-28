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

#include "runtime/core/embedding_engine_impl.h"

#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/tokenizer/tokenizer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::Environment;
using ::litert::support::Tokenizer;
using ::litert::support::TokenizerType;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::status::StatusIs;

constexpr absl::string_view kTestEmbeddingModelPath =
    "litert_lm/runtime/testdata/test_embedding.litertlm";

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (const std::vector<int>& token_ids), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

class FakeModelResources : public ModelResources {
 public:
  FakeModelResources(std::unique_ptr<ModelResources> delegate, bool has_vision,
                     bool has_audio)
      : delegate_(std::move(delegate)),
        has_vision_(has_vision),
        has_audio_(has_audio) {}
  ~FakeModelResources() override = default;

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    if (model_type == ModelType::kTfLiteVisionEncoder && has_vision_) {
      return nullptr;
    }
    if ((model_type == ModelType::kTfLiteAudioEncoderHw ||
         model_type == ModelType::kTfLiteAudioFrontend) &&
        has_audio_) {
      return nullptr;
    }
    return delegate_->GetTFLiteModel(model_type);
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return delegate_->GetTFLiteModelBuffer(model_type);
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return delegate_->GetScopedFile();
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return delegate_->GetWeightsSectionOffset(model_type);
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return delegate_->GetTFLiteModelBackendConstraint(model_type);
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return delegate_->GetTFLiteModelPreferActivationType(model_type);
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return delegate_->GetTokenizer();
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return delegate_->GetLlmMetadata();
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return delegate_->GetExecutorMetadata();
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return delegate_->GetTFLiteModelSectionFileRegion(model_type);
  }

 private:
  std::unique_ptr<ModelResources> delegate_;
  bool has_vision_;
  bool has_audio_;
};

absl::StatusOr<std::unique_ptr<ModelResources>> CreateTestModelResources(
    absl::string_view model_path) {
  LITERT_ASSIGN_OR_RETURN(auto model_file, ScopedFile::Open(model_path));
  LITERT_ASSIGN_OR_RETURN(auto loader,
                          LitertLmLoader::Create(std::move(model_file)));
  return ModelResourcesLitertLm::Create(std::move(loader));
}

absl::StatusOr<std::unique_ptr<OwnedEnvironment>> CreateTestEnvironment() {
  LITERT_ASSIGN_OR_RETURN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  return std::make_unique<OwnedEnvironment>(OwnedEnvironment{
      /*magic_number_configs_helper=*/nullptr, std::move(env)});
}

TEST(EmbeddingEngineImplTest, CreateWithNullResourcesFails) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();

  EXPECT_THAT(
      EmbeddingEngineImpl::Create(/*resources=*/nullptr, std::move(env),
                                  std::move(tokenizer), std::move(settings)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("ModelResources cannot be null.")));
}

TEST(EmbeddingEngineImplTest, CreateWithNullEnvironmentFails) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));
  auto tokenizer = std::make_unique<MockTokenizer>();

  EXPECT_THAT(
      EmbeddingEngineImpl::Create(std::move(resources), /*env=*/nullptr,
                                  std::move(tokenizer), std::move(settings)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("OwnedEnvironment cannot be null.")));
}

TEST(EmbeddingEngineImplTest, CreateWithNullTokenizerFails) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  EXPECT_THAT(
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  /*tokenizer=*/nullptr, std::move(settings)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Tokenizer cannot be null.")));
}

TEST(EmbeddingEngineImplTest, CreateSuccess) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  auto engine =
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings));
  EXPECT_OK(engine.status());
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingSuccess) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  MockTokenizer* mock_tokenizer = tokenizer.get();

  EXPECT_CALL(*mock_tokenizer, TextToTokenIds("hello"))
      .WillOnce(Return(std::vector<int>{1, 2, 3}));

  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  std::vector<InputData> contents;
  contents.push_back(InputText(
      std::variant<std::string, ::litert::TensorBuffer>(std::string("hello"))));

  EmbeddingOptions options;
  options.normalize = false;

  ASSERT_OK_AND_ASSIGN(auto response,
                       engine->ComputeEmbedding(contents, options));
  EXPECT_FALSE(response.embedding.empty());
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingWithNormalization) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  MockTokenizer* mock_tokenizer = tokenizer.get();

  EXPECT_CALL(*mock_tokenizer, TextToTokenIds("hello"))
      .WillOnce(Return(std::vector<int>{1, 2, 3}));

  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  std::vector<InputData> contents;
  contents.push_back(InputText(
      std::variant<std::string, ::litert::TensorBuffer>(std::string("hello"))));

  EmbeddingOptions options;
  options.normalize = true;

  ASSERT_OK_AND_ASSIGN(auto response,
                       engine->ComputeEmbedding(contents, options));
  EXPECT_FALSE(response.embedding.empty());

  float sum_sq = 0.0f;
  for (float val : response.embedding) {
    sum_sq += val * val;
  }
  EXPECT_NEAR(sum_sq, 1.0f, 1e-5f);
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingBatchSuccess) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  MockTokenizer* mock_tokenizer = tokenizer.get();

  EXPECT_CALL(*mock_tokenizer, TextToTokenIds("hello"))
      .WillOnce(Return(std::vector<int>{1, 2, 3}));
  EXPECT_CALL(*mock_tokenizer, TextToTokenIds("world"))
      .WillOnce(Return(std::vector<int>{4, 5}));

  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  std::vector<std::vector<InputData>> batch_contents;
  std::vector<InputData> contents1;
  contents1.push_back(InputText(
      std::variant<std::string, ::litert::TensorBuffer>(std::string("hello"))));
  batch_contents.push_back(std::move(contents1));

  std::vector<InputData> contents2;
  contents2.push_back(InputText(
      std::variant<std::string, ::litert::TensorBuffer>(std::string("world"))));
  batch_contents.push_back(std::move(contents2));

  EmbeddingOptions options;
  options.normalize = false;

  ASSERT_OK_AND_ASSIGN(auto responses,
                       engine->ComputeEmbeddingBatch(batch_contents, options));
  EXPECT_EQ(responses.size(), 2);
  EXPECT_FALSE(responses[0].embedding.empty());
  EXPECT_FALSE(responses[1].embedding.empty());
}

TEST(EmbeddingEngineImplTest,
     CreateWithVisionModelButNoVisionSettingsSucceedsButComputeFails) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto real_resources,
                       CreateTestModelResources(model_path));
  auto resources = std::make_unique<FakeModelResources>(
      std::move(real_resources), /*has_vision=*/true, /*has_audio=*/false);
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();

  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  std::vector<InputData> contents;
  contents.push_back(InputImage("dummy"));

  EmbeddingOptions options;
  options.normalize = false;

  EXPECT_THAT(engine->ComputeEmbedding(contents, options),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Vision executor is not available")));
}

TEST(EmbeddingEngineImplTest,
     CreateWithAudioModelButNoAudioSettingsSucceedsButComputeFails) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto real_resources,
                       CreateTestModelResources(model_path));
  auto resources = std::make_unique<FakeModelResources>(
      std::move(real_resources), /*has_vision=*/false, /*has_audio=*/true);
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();

  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  std::vector<InputData> contents;
  contents.push_back(InputAudio(std::string("dummy")));

  EmbeddingOptions options;
  options.normalize = false;

  EXPECT_THAT(engine->ComputeEmbedding(contents, options),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Audio executor is not available")));
}

}  // namespace
}  // namespace litert::lm
