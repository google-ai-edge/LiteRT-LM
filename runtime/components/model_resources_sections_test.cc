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

#include "runtime/components/model_resources_sections.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::status::StatusIs;

std::filesystem::path TestDataDir() {
  return std::filesystem::path(::testing::SrcDir()) /
         "litert_lm/schema/testdata";
}

TEST(ModelResourcesSectionsTest, LoadTFLiteModelAndHints) {
  ModelResourcesSections::Config config;
  config.tflite_models[ModelType::kTfLitePrefillDecode] = TFLiteModelSection{
      .path = (TestDataDir() / "attention.tflite").string(),
      .backend_constraint = "cpu",
      .prefer_activation_type = "float16",
  };

  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesSections::Create(std::move(config)));
  ASSERT_OK_AND_ASSIGN(
      const litert::Model* model,
      resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  EXPECT_THAT(model, NotNull());

  ASSERT_OK_AND_ASSIGN(auto buffer, resources->GetTFLiteModelBuffer(
                                        ModelType::kTfLitePrefillDecode));
  EXPECT_GT(buffer.size(), 0);

  ASSERT_OK_AND_ASSIGN(FileRegion region,
                       resources->GetTFLiteModelSectionFileRegion(
                           ModelType::kTfLitePrefillDecode));
  EXPECT_EQ(region.offset, 0);
  EXPECT_EQ(region.size, buffer.size());

  EXPECT_EQ(resources->GetTFLiteModelBackendConstraint(
                ModelType::kTfLitePrefillDecode),
            std::optional<std::string>("cpu"));
  EXPECT_EQ(resources->GetTFLiteModelPreferActivationType(
                ModelType::kTfLitePrefillDecode),
            std::optional<std::string>("float16"));
}

TEST(ModelResourcesSectionsTest, LoadExternalWeightsPerModelType) {
  const std::string weights_path = (TestDataDir() / "data.bin").string();
  ModelResourcesSections::Config config;
  config.tflite_weights[ModelType::kTfLitePrefillDecode] =
      TFLiteWeightsSection{.path = weights_path};

  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesSections::Create(std::move(config)));

  ASSERT_OK_AND_ASSIGN(
      auto scoped_file_ref,
      resources->GetScopedFile(ModelType::kTfLitePrefillDecode));
  EXPECT_TRUE(scoped_file_ref.get().IsValid());

  ASSERT_OK_AND_ASSIGN(auto offset_pair, resources->GetWeightsSectionOffset(
                                             ModelType::kTfLitePrefillDecode));
  EXPECT_EQ(offset_pair.first, 0);
  EXPECT_GT(offset_pair.second, 0);

  EXPECT_THAT(
      resources->GetWeightsSectionOffset(ModelType::kTfLiteVisionEncoder),
      StatusIs(absl::StatusCode::kNotFound));
}

std::filesystem::path TokenizerTestDataDir() {
  return std::filesystem::path(::testing::SrcDir()) /
         "litert_lm/support/tokenizer/testdata";
}

TEST(ModelResourcesSectionsTest, LoadSentencePieceAndHuggingFaceTokenizers) {
  ModelResourcesSections::Config config;
  config.tokenizers[ModelType::kTfLitePrefillDecode] = TokenizerSection{
      .format = TokenizerFormat::kSentencePiece,
      .path = (TokenizerTestDataDir() / "sentencepiece.model").string(),
  };
  config.tokenizers[ModelType::kTfLiteTextEncoder] = TokenizerSection{
      .format = TokenizerFormat::kHuggingFaceJson,
      .path = (TokenizerTestDataDir() / "tokenizer.json").string(),
  };

  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesSections::Create(std::move(config)));

  ASSERT_OK_AND_ASSIGN(auto sp_tokenizer, resources->GetTokenizer());
  EXPECT_THAT(sp_tokenizer, NotNull());

  ASSERT_OK_AND_ASSIGN(
      const Tokenizer* cached_sp,
      resources->GetOrCreateTokenizer(ModelType::kTfLitePrefillDecode));
  EXPECT_THAT(cached_sp, NotNull());

  ASSERT_OK_AND_ASSIGN(auto hf_tokenizer,
                       resources->GetTokenizer(ModelType::kTfLiteTextEncoder));
  EXPECT_THAT(hf_tokenizer, NotNull());
}

TEST(ModelResourcesSectionsTest, LoadMetadataProtosAndMissingSections) {
  ModelResourcesSections::Config config;
  proto::LlmMetadata llm_metadata;
  llm_metadata.set_max_num_tokens(2048);
  config.llm_metadata = llm_metadata;

  proto::ExecutorMetadata executor_metadata;
  executor_metadata.mutable_llm_executor_metadata()->set_max_history_size(1024);
  config.executor_metadata = executor_metadata;

  proto::EmbeddingMetadata embedding_metadata;
  config.embedding_metadata = embedding_metadata;

  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesSections::Create(std::move(config)));

  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* got_llm,
                       resources->GetLlmMetadata());
  EXPECT_EQ(got_llm->max_num_tokens(), 2048);

  ASSERT_OK_AND_ASSIGN(const proto::ExecutorMetadata* got_exec,
                       resources->GetExecutorMetadata());
  EXPECT_EQ(got_exec->llm_executor_metadata().max_history_size(), 1024);

  ASSERT_OK_AND_ASSIGN(const proto::EmbeddingMetadata* got_emb,
                       resources->GetEmbeddingMetadata());
  EXPECT_THAT(got_emb, NotNull());

  EXPECT_THAT(resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("not found")));
}

}  // namespace
}  // namespace litert::lm
