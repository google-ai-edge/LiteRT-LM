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

#include <cmath>
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
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision/vision_executor.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/embedding_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/preprocessor/image_preprocessor.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::litert::Environment;
using ::litert::support::ImagePreprocessParameter;
using ::litert::support::ImagePreprocessor;
using ::litert::support::Tokenizer;
using ::litert::support::TokenizerType;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Key;
using ::testing::Return;
using ::testing::status::StatusIs;

constexpr absl::string_view kTestEmbeddingModelPath =
    "litert_lm/runtime/testdata/test_embedding.litertlm";

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int> token_ids, bool skip_special_tokens),
              (override));
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
  explicit FakeModelResources(
      std::unique_ptr<ModelResources> delegate, bool has_vision = false,
      bool has_audio = false,
      std::optional<const proto::EmbeddingMetadata*> metadata = std::nullopt)
      : delegate_(std::move(delegate)),
        has_vision_(has_vision),
        has_audio_(has_audio),
        metadata_(metadata) {}
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

  absl::StatusOr<const proto::EmbeddingMetadata*> GetEmbeddingMetadata()
      override {
    if (metadata_.has_value()) {
      return *metadata_;
    }
    return delegate_->GetEmbeddingMetadata();
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return delegate_->GetTFLiteModelSectionFileRegion(model_type);
  }

 private:
  std::unique_ptr<ModelResources> delegate_;
  bool has_vision_;
  bool has_audio_;
  std::optional<const proto::EmbeddingMetadata*> metadata_;
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

TEST(EmbeddingEngineImplTest, CreateWithBenchmarkEnabledSuccess) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  settings.GetMutableBenchmarkParams();

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  auto benchmark_info = engine->GetBenchmarkInfo();

  const auto& init_phases = benchmark_info->GetInitPhases();
  EXPECT_THAT(init_phases, Contains(Key("Init Total")));
  EXPECT_THAT(init_phases, Contains(Key("Init Executor")));
}

TEST(EmbeddingEngineImplTest, BenchmarkDisabledReturnsNulllopt) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  EXPECT_THAT(engine->GetBenchmarkInfo(), testing::Eq(std::nullopt));
}

TEST(EmbeddingEngineImplTest, CreateWithSettingsSuccess) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  auto engine = EmbeddingEngineImpl::Create(std::move(settings));
  EXPECT_OK(engine.status());
}

TEST(EmbeddingEngineImplTest, GetEmbeddingMetadataFromSettings) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  proto::EmbeddingMetadata expected_metadata;
  expected_metadata.mutable_embedding_model_type();
  settings.GetMutableEmbeddingMetadata() = expected_metadata;

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  const auto& metadata = engine->GetEmbeddingMetadata();
  ASSERT_TRUE(metadata.has_value());
  EXPECT_TRUE(metadata->has_embedding_model_type());
}

TEST(EmbeddingEngineImplTest, GetEmbeddingMetadataFromResources) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto real_resources,
                       CreateTestModelResources(model_path));
  proto::EmbeddingMetadata expected_metadata;
  expected_metadata.mutable_embedding_model_type();
  auto resources = std::make_unique<FakeModelResources>(
      std::move(real_resources), /*has_vision=*/false, /*has_audio=*/false,
      &expected_metadata);
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  const auto& metadata = engine->GetEmbeddingMetadata();
  ASSERT_TRUE(metadata.has_value());
  EXPECT_TRUE(metadata->has_embedding_model_type());
}

TEST(EmbeddingEngineImplTest, GetEmbeddingMetadataNullByDefault) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto real_resources,
                       CreateTestModelResources(model_path));
  auto resources = std::make_unique<FakeModelResources>(
      std::move(real_resources), /*has_vision=*/false, /*has_audio=*/false,
      /*metadata=*/nullptr);
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  const auto& metadata = engine->GetEmbeddingMetadata();
  EXPECT_FALSE(metadata.has_value());
}

TEST(EmbeddingEngineImplTest, DefaultEmbeddingOptions) {
  EmbeddingOptions options;
  EXPECT_TRUE(options.normalize);
  EXPECT_TRUE(options.insert_special_tokens);
  EXPECT_EQ(options.input_overflow_strategy, InputOverflowStrategy::kError);
  EXPECT_FALSE(options.vision_tokens_per_image.has_value());
  EXPECT_EQ(options.output_size, std::nullopt);
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
  EXPECT_EQ(response.input_length, 3);
  EXPECT_EQ(response.truncated_length, std::nullopt);
  EXPECT_EQ(response.num_chunks, 1);
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
  EXPECT_EQ(response.input_length, 3);
  EXPECT_EQ(response.truncated_length, std::nullopt);
  EXPECT_EQ(response.num_chunks, 1);

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
  EXPECT_EQ(responses[0].input_length, 3);
  EXPECT_EQ(responses[0].truncated_length, std::nullopt);
  EXPECT_EQ(responses[0].num_chunks, 1);
  EXPECT_FALSE(responses[1].embedding.empty());
  EXPECT_EQ(responses[1].input_length, 2);
  EXPECT_EQ(responses[1].truncated_length, std::nullopt);
  EXPECT_EQ(responses[1].num_chunks, 1);
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingWithBenchmarkEnabled) {
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

  settings.GetMutableBenchmarkParams();

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  std::vector<InputData> contents;
  contents.push_back(InputText(
      std::variant<std::string, ::litert::TensorBuffer>(std::string("hello"))));

  EmbeddingOptions options;
  options.normalize = false;

  ASSERT_OK(engine->ComputeEmbedding(contents, options).status());

  auto benchmark_info = engine->GetBenchmarkInfo();
  ASSERT_TRUE(benchmark_info.has_value());
  EXPECT_EQ(benchmark_info->GetTotalPrefillTurns(), 1);
  ASSERT_OK_AND_ASSIGN(auto turn, benchmark_info->GetPrefillTurn(0));
  EXPECT_EQ(turn.num_tokens, 3);
  EXPECT_GT(turn.duration, absl::ZeroDuration());
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingWithNumPrefillTokens) {
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

  settings.GetMutableBenchmarkParams().set_num_prefill_tokens(5);

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));

  std::vector<InputData> contents;
  contents.push_back(InputText(
      std::variant<std::string, ::litert::TensorBuffer>(std::string("hello"))));

  EmbeddingOptions options;
  options.normalize = false;

  ASSERT_OK(engine->ComputeEmbedding(contents, options).status());

  auto benchmark_info = engine->GetBenchmarkInfo();
  ASSERT_TRUE(benchmark_info.has_value());
  EXPECT_EQ(benchmark_info->GetTotalPrefillTurns(), 1);
  ASSERT_OK_AND_ASSIGN(auto turn, benchmark_info->GetPrefillTurn(0));
  EXPECT_EQ(turn.num_tokens, 5);
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingBatchWithBenchmarkEnabled) {
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

  settings.GetMutableBenchmarkParams();

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

  ASSERT_OK(engine->ComputeEmbeddingBatch(batch_contents, options).status());

  auto benchmark_info = engine->GetBenchmarkInfo();
  ASSERT_TRUE(benchmark_info.has_value());
  EXPECT_EQ(benchmark_info->GetTotalPrefillTurns(), 1);
  ASSERT_OK_AND_ASSIGN(auto turn, benchmark_info->GetPrefillTurn(0));
  EXPECT_EQ(turn.num_tokens, 5);
  EXPECT_GT(turn.duration, absl::ZeroDuration());
}

class FakeImagePreprocessor : public ::litert::support::ImagePreprocessor {
 public:
  explicit FakeImagePreprocessor(::litert::TensorBuffer tensor_buffer)
      : tensor_buffer_(std::move(tensor_buffer)) {}

  absl::StatusOr<InputImage> Preprocess(
      const InputImage& input_image,
      const ::litert::support::ImagePreprocessParameter& parameter) override {
    last_parameter_ = parameter;
    LITERT_ASSIGN_OR_RETURN(auto duplicate, tensor_buffer_.Duplicate());
    return InputImage(std::move(duplicate));
  }

  const std::optional<::litert::support::ImagePreprocessParameter>&
  GetLastParameter() const {
    return last_parameter_;
  }

 private:
  ::litert::TensorBuffer tensor_buffer_;
  std::optional<::litert::support::ImagePreprocessParameter> last_parameter_;
};

absl::StatusOr<::litert::TensorBuffer> CreateDummyTensorBuffer(
    const ::litert::Environment& env, const std::vector<int>& dimensions) {
  size_t num_elements = 1;
  for (int dim : dimensions) {
    num_elements *= dim;
  }
  ::litert::RankedTensorType tensor_type(
      ::litert::ElementType::Float32,
      ::litert::Layout(
          ::litert::Dimensions(dimensions.begin(), dimensions.end())));
  auto tensor = ::litert::TensorBuffer::CreateManaged(
      env, ::litert::TensorBufferType::kHostMemory, tensor_type,
      num_elements * sizeof(float));
  if (!tensor.HasValue()) {
    return absl::InternalError("Failed to create tensor buffer");
  }
  return std::move(*tensor);
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

class FakeEmbeddingExecutor : public EmbeddingExecutorBase {
 public:
  using EmbeddingExecutorBase::ComputeEmbedding;
  using EmbeddingExecutorBase::ComputeEmbeddingBatch;

  absl::StatusOr<EmbeddingOutput> ComputeEmbedding(
      const ExecutorInputs& inputs,
      const ComputeEmbeddingOptions& options) override {
    auto text_token_ids = inputs.GetTextTokenIdsPtr();
    int input_length = 0;
    if (text_token_ids.ok() && *text_token_ids != nullptr) {
      auto span = ReferTensorBufferAsSpan<int>(**text_token_ids);
      if (span.HasValue()) {
        last_token_ids_.assign(span->begin(), span->end());
        input_length = static_cast<int>(last_token_ids_.size());
      }
    }
    last_options_ = options;
    return EmbeddingOutput{
        .embedding = std::vector<float>{1.0f, 2.0f, 3.0f},
        .input_length = input_length,
        .truncated_length = std::nullopt,
        .num_chunks = 1,
    };
  }

  absl::StatusOr<std::vector<EmbeddingOutput>> ComputeEmbeddingBatch(
      const std::vector<ExecutorInputs>& batch_inputs,
      const ComputeEmbeddingOptions& options) override {
    std::vector<EmbeddingOutput> results;
    for (const auto& inputs : batch_inputs) {
      LITERT_ASSIGN_OR_RETURN(auto result, ComputeEmbedding(inputs, options));
      results.push_back(std::move(result));
    }
    return results;
  }

  absl::string_view ExecutorBackendName() const override { return "Fake"; }

  absl::StatusOr<std::vector<int>> GetExpectedInputDimension() const override {
    return std::vector<int>{1, 512};
  }

  absl::StatusOr<int> GetEmbeddingDimension() const override { return 3; }

  ::litert::Environment* GetEnvironment() const override { return nullptr; }

  const std::vector<int>& GetLastTokenIds() const { return last_token_ids_; }

  const ComputeEmbeddingOptions& GetLastOptions() const {
    return last_options_;
  }

 private:
  std::vector<int> last_token_ids_;
  ComputeEmbeddingOptions last_options_;
};

class FakeVisionExecutor : public VisionExecutor {
 public:
  absl::StatusOr<ExecutorVisionData> Encode(
      const ::litert::TensorBuffer& input_image_tensor) override {
    struct alignas(::litert::kHostMemoryBufferAlignment) {
      float d[12] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    } data;
    auto tensor = ::litert::TensorBuffer::CreateFromHostMemory(
        ::litert::RankedTensorType(
            ::litert::ElementType::Float32,
            ::litert::Layout(::litert::Dimensions({1, 4, 3}))),
        data.d, sizeof(data.d));
    if (!tensor.HasValue()) {
      return absl::InternalError("Failed to create tensor buffer");
    }
    return ExecutorVisionData(std::move(*tensor), std::nullopt);
  }

  absl::StatusOr<ExecutorVisionData> Encode(
      const absl::flat_hash_map<std::string, ::litert::TensorBuffer>&
          input_tensors) override {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::StatusOr<std::vector<int>> GetExpectedInputDimension() const override {
    return std::vector<int>{1, 224, 224, 3};
  }
};

class FakeAudioExecutor : public AudioExecutor {
 public:
  absl::StatusOr<ExecutorAudioData> Encode(
      const ::litert::TensorBuffer& spectrogram_tensor) override {
    struct alignas(::litert::kHostMemoryBufferAlignment) {
      float d[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    } data;
    auto tensor = ::litert::TensorBuffer::CreateFromHostMemory(
        ::litert::RankedTensorType(
            ::litert::ElementType::Float32,
            ::litert::Layout(::litert::Dimensions({1, 1, 3, 3}))),
        data.d, sizeof(data.d));
    if (!tensor.HasValue()) {
      return absl::InternalError("Failed to create tensor buffer");
    }
    return ExecutorAudioData(std::move(*tensor), std::nullopt,
                             /*valid_tokens=*/3);
  }
};

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensImageTrue) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.start_of_image_token_ids = {101};
  special_tokens.end_of_image_token_ids = {102};

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor),
                             std::make_unique<FakeVisionExecutor>(),
                             /*audio_executor=*/nullptr,
                             /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::move(image_tensor)));

  EmbeddingOptions options;
  options.insert_special_tokens = true;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(),
              ElementsAre(101, -1, -1, -1, -1, 102));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensImageFalse) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.start_of_image_token_ids = {101};
  special_tokens.end_of_image_token_ids = {102};

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor),
                             std::make_unique<FakeVisionExecutor>(),
                             /*audio_executor=*/nullptr,
                             /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::move(image_tensor)));

  EmbeddingOptions options;
  options.insert_special_tokens = false;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(), ElementsAre(-1, -1, -1, -1));
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingWithOutputSize) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ON_CALL(*tokenizer, TextToTokenIds(::testing::_))
      .WillByDefault(Return(std::vector<int>{1, 2, 3}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor));

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.normalize = false;
  options.output_size = 2;

  ASSERT_OK_AND_ASSIGN(auto response,
                       engine.ComputeEmbedding(contents, options));
  EXPECT_THAT(response.embedding, ElementsAre(1.0f, 2.0f));
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingWithOutputSizeAndNormalization) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ON_CALL(*tokenizer, TextToTokenIds(::testing::_))
      .WillByDefault(Return(std::vector<int>{1, 2, 3}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor));

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.normalize = true;
  options.output_size = 2;

  ASSERT_OK_AND_ASSIGN(auto response,
                       engine.ComputeEmbedding(contents, options));
  ASSERT_EQ(response.embedding.size(), 2);
  const float norm = std::sqrt(1.0f * 1.0f + 2.0f * 2.0f);
  EXPECT_NEAR(response.embedding[0], 1.0f / norm, 1e-5f);
  EXPECT_NEAR(response.embedding[1], 2.0f / norm, 1e-5f);
  const float sum_sq = response.embedding[0] * response.embedding[0] +
                       response.embedding[1] * response.embedding[1];
  EXPECT_NEAR(sum_sq, 1.0f, 1e-5f);
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingWithOutputSizeEqualToDefault) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ON_CALL(*tokenizer, TextToTokenIds(::testing::_))
      .WillByDefault(Return(std::vector<int>{1, 2, 3}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor));

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.normalize = false;
  options.output_size = 3;

  ASSERT_OK_AND_ASSIGN(auto response,
                       engine.ComputeEmbedding(contents, options));
  EXPECT_THAT(response.embedding, ElementsAre(1.0f, 2.0f, 3.0f));
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingWithOutputSizeZero) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ON_CALL(*tokenizer, TextToTokenIds(::testing::_))
      .WillByDefault(Return(std::vector<int>{1, 2, 3}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor));

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.output_size = 0;

  ASSERT_OK_AND_ASSIGN(auto response,
                       engine.ComputeEmbedding(contents, options));
  EXPECT_TRUE(response.embedding.empty());
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithOutputSizeNegativeReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ON_CALL(*tokenizer, TextToTokenIds(::testing::_))
      .WillByDefault(Return(std::vector<int>{1, 2, 3}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor));

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.output_size = -1;

  EXPECT_THAT(engine.ComputeEmbedding(contents, options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cannot be negative")));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithOutputSizeLargerThanDefaultReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ON_CALL(*tokenizer, TextToTokenIds(::testing::_))
      .WillByDefault(Return(std::vector<int>{1, 2, 3}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor));

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.output_size = 5;

  EXPECT_THAT(
      engine.ComputeEmbedding(contents, options),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("cannot be larger than default output embedding size")));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingBatchWithOutputSizeAndNormalization) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  ON_CALL(*tokenizer, TextToTokenIds(::testing::_))
      .WillByDefault(Return(std::vector<int>{1, 2, 3}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor));

  std::vector<std::vector<InputData>> contents;
  std::vector<InputData> contents1;
  contents1.push_back(InputText(std::string("hello")));
  contents.push_back(std::move(contents1));
  std::vector<InputData> contents2;
  contents2.push_back(InputText(std::string("world")));
  contents.push_back(std::move(contents2));

  EmbeddingOptions options;
  options.normalize = true;
  options.output_size = 2;

  ASSERT_OK_AND_ASSIGN(auto responses,
                       engine.ComputeEmbeddingBatch(contents, options));
  ASSERT_EQ(responses.size(), 2);
  const float norm = std::sqrt(1.0f * 1.0f + 2.0f * 2.0f);
  for (const auto& response : responses) {
    ASSERT_EQ(response.embedding.size(), 2);
    EXPECT_NEAR(response.embedding[0], 1.0f / norm, 1e-5f);
    EXPECT_NEAR(response.embedding[1], 2.0f / norm, 1e-5f);
  }
}

TEST(EmbeddingEngineImplTest, ComputeEmbeddingWithRawImageSuccess) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  auto fake_image_preprocessor =
      std::make_unique<FakeImagePreprocessor>(std::move(image_tensor));
  auto* raw_image_preprocessor = fake_image_preprocessor.get();

  ImagePreprocessParameter parameter;
  parameter.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 2520,
      .pooling_kernel_size = 3,
  });

  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      std::make_unique<FakeVisionExecutor>(),
      /*audio_executor=*/nullptr,
      /*benchmark_info=*/std::nullopt,
      /*special_tokens=*/{},
      std::move(fake_image_preprocessor), parameter);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::string("raw_image_bytes")));

  EmbeddingOptions options;
  options.insert_special_tokens = false;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(), ElementsAre(-1, -1, -1, -1));
  ASSERT_TRUE(raw_image_preprocessor->GetLastParameter().has_value());
  EXPECT_EQ(raw_image_preprocessor->GetLastParameter()
                ->GetPatchifyConfig()
                ->patch_width,
            16);
  EXPECT_EQ(raw_image_preprocessor->GetLastParameter()
                ->GetPatchifyConfig()
                ->patch_height,
            16);
  EXPECT_EQ(raw_image_preprocessor->GetLastParameter()
                ->GetPatchifyConfig()
                ->max_num_patches,
            2520);
  EXPECT_EQ(raw_image_preprocessor->GetLastParameter()
                ->GetPatchifyConfig()
                ->pooling_kernel_size,
            3);
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithRawImageOverridingVisionTokensPerImage) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  auto fake_image_preprocessor =
      std::make_unique<FakeImagePreprocessor>(std::move(image_tensor));
  auto* raw_image_preprocessor = fake_image_preprocessor.get();

  ImagePreprocessParameter parameter;
  parameter.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 2520,
      .pooling_kernel_size = 3,
  });

  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      std::make_unique<FakeVisionExecutor>(),
      /*audio_executor=*/nullptr,
      /*benchmark_info=*/std::nullopt,
      /*special_tokens=*/{}, std::move(fake_image_preprocessor), parameter);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::string("raw_image_bytes")));

  EmbeddingOptions options;
  options.insert_special_tokens = false;
  options.vision_tokens_per_image = 70;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(), ElementsAre(-1, -1, -1, -1));
  ASSERT_TRUE(raw_image_preprocessor->GetLastParameter().has_value());
  EXPECT_EQ(raw_image_preprocessor->GetLastParameter()
                ->GetPatchifyConfig()
                ->patch_width,
            16);
  EXPECT_EQ(raw_image_preprocessor->GetLastParameter()
                ->GetPatchifyConfig()
                ->patch_height,
            16);
  // max_num_patches = vision_tokens_per_image (70) * pooling_kernel_size^2 (9)
  // = 630.
  EXPECT_EQ(raw_image_preprocessor->GetLastParameter()
                ->GetPatchifyConfig()
                ->max_num_patches,
            630);
  EXPECT_EQ(raw_image_preprocessor->GetLastParameter()
                ->GetPatchifyConfig()
                ->pooling_kernel_size,
            3);
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInvalidVisionTokensPerImageFails) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto fake_image_preprocessor =
      std::make_unique<FakeImagePreprocessor>(std::move(image_tensor));

  ImagePreprocessParameter parameter;
  parameter.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 2520,
      .pooling_kernel_size = 3,
  });

  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      std::make_unique<FakeVisionExecutor>(),
      /*audio_executor=*/nullptr,
      /*benchmark_info=*/std::nullopt,
      /*special_tokens=*/{}, std::move(fake_image_preprocessor), parameter);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::string("raw_image_bytes")));

  EmbeddingOptions options_zero;
  options_zero.vision_tokens_per_image = 0;
  EXPECT_THAT(engine.ComputeEmbedding(contents, options_zero),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EmbeddingOptions options_negative;
  options_negative.vision_tokens_per_image = -5;
  EXPECT_THAT(engine.ComputeEmbedding(contents, options_negative),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithRawImageNoPreprocessorFails) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();

  ImagePreprocessParameter parameter;
  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      std::make_unique<FakeVisionExecutor>(),
      /*audio_executor=*/nullptr,
      /*benchmark_info=*/std::nullopt,
      /*special_tokens=*/{},
      /*image_preprocessor=*/nullptr, parameter);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::string("raw_image_bytes")));

  EmbeddingOptions options;
  EXPECT_THAT(engine.ComputeEmbedding(contents, options),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Image preprocessor is not available")));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithRawImageNoPreprocessParameterFails) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto fake_image_preprocessor =
      std::make_unique<FakeImagePreprocessor>(std::move(image_tensor));

  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      std::make_unique<FakeVisionExecutor>(),
      /*audio_executor=*/nullptr,
      /*benchmark_info=*/std::nullopt,
      /*special_tokens=*/{},
      std::move(fake_image_preprocessor),
      /*image_preprocess_parameter=*/std::nullopt);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::string("raw_image_bytes")));

  EmbeddingOptions options;
  EXPECT_THAT(
      engine.ComputeEmbedding(contents, options),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Image preprocess parameter is not available")));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithRawImageAndInsertSpecialTokens) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  auto fake_image_preprocessor =
      std::make_unique<FakeImagePreprocessor>(std::move(image_tensor));

  SpecialTokens special_tokens;
  special_tokens.start_of_image_token_ids = {101};
  special_tokens.end_of_image_token_ids = {102};

  ImagePreprocessParameter parameter;
  parameter.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 2520,
      .pooling_kernel_size = 3,
  });

  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      std::make_unique<FakeVisionExecutor>(),
      /*audio_executor=*/nullptr,
      /*benchmark_info=*/std::nullopt, special_tokens,
      std::move(fake_image_preprocessor), parameter);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::string("raw_image_bytes")));

  EmbeddingOptions options;
  options.insert_special_tokens = true;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(),
              ElementsAre(101, -1, -1, -1, -1, 102));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensAudioTrue) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto audio_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 1, 100, 80}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.start_of_audio_token_ids = {201};
  special_tokens.end_of_audio_token_ids = {202};

  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      /*vision_executor=*/nullptr, std::make_unique<FakeAudioExecutor>(),
      /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputAudio(std::move(audio_tensor)));

  EmbeddingOptions options;
  options.insert_special_tokens = true;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(),
              ElementsAre(201, -2, -2, -2, 202));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensAudioFalse) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto audio_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 1, 100, 80}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.start_of_audio_token_ids = {201};
  special_tokens.end_of_audio_token_ids = {202};

  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      /*vision_executor=*/nullptr, std::make_unique<FakeAudioExecutor>(),
      /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputAudio(std::move(audio_tensor)));

  EmbeddingOptions options;
  options.insert_special_tokens = false;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(), ElementsAre(-2, -2, -2));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensMultimodal) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  ASSERT_OK_AND_ASSIGN(auto audio_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 1, 100, 80}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TextToTokenIds("hello"))
      .WillRepeatedly(Return(std::vector<int>{1, 2}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.start_of_image_token_ids = {101};
  special_tokens.end_of_image_token_ids = {102};
  special_tokens.start_of_audio_token_ids = {201};
  special_tokens.end_of_audio_token_ids = {202};

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor),
                             std::make_unique<FakeVisionExecutor>(),
                             std::make_unique<FakeAudioExecutor>(),
                             /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));
  contents.push_back(InputImage(std::move(image_tensor)));
  contents.push_back(InputAudio(std::move(audio_tensor)));

  EmbeddingOptions options;
  options.insert_special_tokens = true;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(
      raw_executor->GetLastTokenIds(),
      ElementsAre(1, 2, 101, -1, -1, -1, -1, 102, 201, -2, -2, -2, 202));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensImageEndOfVisionSubmodel) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto image_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 224, 224, 3}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.start_of_image_token_ids = {101};
  special_tokens.end_of_image_token_ids = {102};
  special_tokens.has_end_of_vision_model = true;

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor),
                             std::make_unique<FakeVisionExecutor>(),
                             /*audio_executor=*/nullptr,
                             /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputImage(std::move(image_tensor)));

  EmbeddingOptions options;
  options.insert_special_tokens = true;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(),
              ElementsAre(101, -1, -1, -1, -1, -3));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensAudioEndOfAudioSubmodel) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  ASSERT_OK_AND_ASSIGN(auto audio_tensor,
                       CreateDummyTensorBuffer(env->env, {1, 1, 100, 80}));
  auto tokenizer = std::make_unique<MockTokenizer>();
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.start_of_audio_token_ids = {201};
  special_tokens.end_of_audio_token_ids = {202};
  special_tokens.has_end_of_audio_model = true;

  EmbeddingEngineImpl engine(
      std::move(env), std::move(tokenizer), std::move(fake_embedding_executor),
      /*vision_executor=*/nullptr, std::make_unique<FakeAudioExecutor>(),
      /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputAudio(std::move(audio_tensor)));

  EmbeddingOptions options;
  options.insert_special_tokens = true;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(),
              ElementsAre(201, -2, -2, -2, -4));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensBosEosTrue) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TextToTokenIds("hello"))
      .WillRepeatedly(Return(std::vector<int>{10, 11}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.bos_token_ids = {2};
  special_tokens.eos_token_ids = {1};

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor),
                             /*vision_executor=*/nullptr,
                             /*audio_executor=*/nullptr,
                             /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.insert_special_tokens = true;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(), ElementsAre(2, 10, 11, 1));
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensBosEosFalse) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TextToTokenIds("hello"))
      .WillRepeatedly(Return(std::vector<int>{10, 11}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.bos_token_ids = {2};
  special_tokens.eos_token_ids = {1};

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor),
                             /*vision_executor=*/nullptr,
                             /*audio_executor=*/nullptr,
                             /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.insert_special_tokens = false;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(), ElementsAre(10, 11));
  EXPECT_THAT(raw_executor->GetLastOptions().eos_token_ids, ElementsAre());
}

TEST(EmbeddingEngineImplTest,
     ComputeEmbeddingWithInsertSpecialTokensPassesEosTokenIdsToExecutor) {
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TextToTokenIds("hello"))
      .WillRepeatedly(Return(std::vector<int>{10, 11}));
  auto fake_embedding_executor = std::make_unique<FakeEmbeddingExecutor>();
  auto* raw_executor = fake_embedding_executor.get();

  SpecialTokens special_tokens;
  special_tokens.bos_token_ids = {2};
  special_tokens.eos_token_ids = {1};

  EmbeddingEngineImpl engine(std::move(env), std::move(tokenizer),
                             std::move(fake_embedding_executor),
                             /*vision_executor=*/nullptr,
                             /*audio_executor=*/nullptr,
                             /*benchmark_info=*/std::nullopt, special_tokens);

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string("hello")));

  EmbeddingOptions options;
  options.insert_special_tokens = true;
  options.input_overflow_strategy = InputOverflowStrategy::kTruncate;

  ASSERT_OK(engine.ComputeEmbedding(contents, options).status());
  EXPECT_THAT(raw_executor->GetLastTokenIds(), ElementsAre(2, 10, 11, 1));
  EXPECT_THAT(raw_executor->GetLastOptions().eos_token_ids, ElementsAre(1));
  EXPECT_EQ(raw_executor->GetLastOptions().input_overflow_strategy,
            InputOverflowStrategy::kTruncate);
}

TEST(EmbeddingEngineImplTest, CreateWithMaxInputLengthAutoSelectsSignatures) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();

  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));
  settings.SetMaxInputLength(128);

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));
  EXPECT_NE(engine, nullptr);
  const auto& text_sig_info = engine->GetSelectedTextSignaturesInfo();
  ASSERT_TRUE(text_sig_info.has_value());
  EXPECT_FALSE(text_sig_info->signature_names.empty());
  EXPECT_EQ(engine->GetSelectedVisionSignatureInfo(), std::nullopt);
}

TEST(EmbeddingEngineImplTest,
     CreateWithoutAutoSelectionHasNulloptSignaturesInfo) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();

  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)));
  EXPECT_NE(engine, nullptr);
  EXPECT_EQ(engine->GetSelectedTextSignaturesInfo(), std::nullopt);
  EXPECT_EQ(engine->GetSelectedVisionSignatureInfo(), std::nullopt);
}

TEST(EmbeddingEngineImplTest, CreateWithInvalidMaxInputLengthReturnsError) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();

  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));
  settings.SetMaxInputLength(0);

  EXPECT_THAT(
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EmbeddingEngineImplTest,
     CreateWithVisionTokensPerImageWithoutVisionBackendReturnsError) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();

  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));
  settings.SetVisionTokensPerImage(70);

  EXPECT_THAT(
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(EmbeddingEngineImplTest,
     CreateWithInvalidVisionTokensPerImageReturnsError) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestEmbeddingModelPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(auto resources, CreateTestModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto env, CreateTestEnvironment());
  auto tokenizer = std::make_unique<MockTokenizer>();

  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU,
                                          Backend::CPU, std::nullopt));
  settings.SetVisionTokensPerImage(0);

  EXPECT_THAT(
      EmbeddingEngineImpl::Create(std::move(resources), std::move(env),
                                  std::move(tokenizer), std::move(settings)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
