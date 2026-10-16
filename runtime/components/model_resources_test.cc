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

#include "runtime/components/model_resources.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"

namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::litert::lm::LitertLmLoader;
using ::litert::lm::ModelAssetBundleResources;
using ::litert::lm::ModelResourcesLitertLm;
using ::litert::lm::ModelResourcesTask;
using ::litert::lm::ParseLlmTfLiteModelType;
using ::litert::lm::ScopedFile;
using ::litert::lm::TfLiteModelTypeToWireString;
using ::litert::lm::proto::EmbeddingMetadata;
using ::litert::lm::proto::LlmMetadata;

#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
TEST(ModelResourcesTest, InitializeWithFileBackedLiteRtModel) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file)));

  const auto expected_model_size =
      loader->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE).Size();
  ASSERT_GT(expected_model_size, 0);

  auto model_resources = ModelResourcesLitertLm::Create(
      std::move(loader), /*enable_file_backed_model_loading=*/true);
  ASSERT_OK(model_resources);

  auto tflite_model = model_resources.value()->GetTFLiteModel(
      LlmMetadata::TF_LITE_PREFILL_DECODE);
  ASSERT_OK(tflite_model);
  ASSERT_GT(tflite_model.value()->GetNumSignatures(), 0);
}

TEST(ModelResourcesTest, InitializeWithValidLitertLmLoader) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file.value())));
  ASSERT_GT(loader->GetSentencePieceTokenizer()->Size(), 0);
  ASSERT_GT(loader->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE).Size(),
            0);

  auto model_resources = ModelResourcesLitertLm::Create(std::move(loader));
  ASSERT_OK(model_resources);

  auto tflite_model = model_resources.value()->GetTFLiteModel(
      LlmMetadata::TF_LITE_PREFILL_DECODE);
  ASSERT_OK(tflite_model);
  ASSERT_GT(tflite_model.value()->GetNumSignatures(), 0);

  auto tokenizer = model_resources.value()->GetTokenizer();
  ASSERT_OK(tokenizer);
  ASSERT_NE(tokenizer.value(), nullptr);

  auto prefill_tokenizer = model_resources.value()->GetTokenizer(
      LlmMetadata::TF_LITE_PREFILL_DECODE);
  ASSERT_OK(prefill_tokenizer);
  ASSERT_NE(prefill_tokenizer.value(), nullptr);
}

TEST(ModelResourcesTest, InitializeWithExternalWeights) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_external_weights.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file.value())));
  ASSERT_GT(loader->GetSentencePieceTokenizer()->Size(), 0);
  ASSERT_GT(loader->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE).Size(),
            0);
  ASSERT_GT(
      loader->GetTFLiteWeights(LlmMetadata::TF_LITE_PREFILL_DECODE).Size(), 0);

  auto model_resources = ModelResourcesLitertLm::Create(std::move(loader));
  ASSERT_OK(model_resources);

  auto tflite_model = model_resources.value()->GetTFLiteModel(
      LlmMetadata::TF_LITE_PREFILL_DECODE);
  ASSERT_OK(tflite_model);
  ASSERT_GT(tflite_model.value()->GetNumSignatures(), 0);

  auto tokenizer = model_resources.value()->GetTokenizer();
  ASSERT_OK(tokenizer);
  ASSERT_NE(tokenizer.value(), nullptr);
}
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER

#ifdef ENABLE_HUGGINGFACE_TOKENIZER
TEST(ModelResourcesTest, InitializeWithHuggingFaceTokenizer) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file.value())));
  ASSERT_GT(loader->GetHuggingFaceTokenizer()->Size(), 0);

  auto model_resources = ModelResourcesLitertLm::Create(std::move(loader));
  ASSERT_OK(model_resources);

  auto tokenizer = model_resources.value()->GetTokenizer();
  ASSERT_OK(tokenizer);
  ASSERT_NE(tokenizer.value(), nullptr);
}
#endif  // ENABLE_HUGGINGFACE_TOKENIZER

#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
TEST(ModelResourcesTest, InitializeWithValidModelAssetBundleResources) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  auto model_asset_bundle_resources =
      ModelAssetBundleResources::Create("tag", std::move(model_file.value()));
  ASSERT_OK(model_asset_bundle_resources);

  auto model_resources = ModelResourcesTask::Create(
      std::move(model_asset_bundle_resources.value()));
  ASSERT_OK(model_resources);

  auto tflite_model = model_resources.value()->GetTFLiteModel(
      LlmMetadata::TF_LITE_PREFILL_DECODE);
  ASSERT_OK(tflite_model);
  ASSERT_GT(tflite_model.value()->GetNumSignatures(), 0);

  auto tokenizer = model_resources.value()->GetTokenizer();
  ASSERT_OK(tokenizer);
  ASSERT_NE(tokenizer.value(), nullptr);
}

TEST(ModelResourcesTest, GetTFLiteModelNotFound) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file.value())));

  auto model_resources = ModelResourcesLitertLm::Create(std::move(loader));
  ASSERT_OK(model_resources);

  // Attempt to get a model type that doesn't exist in the test file.
  EXPECT_THAT(model_resources.value()->GetTFLiteModelBuffer(
                  LlmMetadata::TF_LITE_EMBEDDER),
              testing::status::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      model_resources.value()->GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER),
      testing::status::StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelResourcesTest, GetTFLiteModelNotFoundTask) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  auto model_asset_bundle_resources =
      ModelAssetBundleResources::Create("tag", std::move(model_file.value()));
  ASSERT_OK(model_asset_bundle_resources);

  auto model_resources = ModelResourcesTask::Create(
      std::move(model_asset_bundle_resources.value()));
  ASSERT_OK(model_resources);

  // Attempt to get a model type that doesn't exist in the test file.
  auto tflite_model = model_resources.value()->GetTFLiteModelBuffer(
      LlmMetadata::TF_LITE_EMBEDDER);
  EXPECT_THAT(tflite_model,
              testing::status::StatusIs(absl::StatusCode::kNotFound));
}
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER

TEST(ModelResourcesTest, GetTFLiteModelSectionFileRegion) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file)));

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       ModelResourcesLitertLm::Create(std::move(loader)));

  // Success case: model is present.
  ASSERT_OK_AND_ASSIGN(auto file_region,
                       model_resources->GetTFLiteModelSectionFileRegion(
                           LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_GT(file_region.offset, 0);
  EXPECT_GT(file_region.size, 0);

  // Compare with the size from GetTFLiteModelBuffer.
  ASSERT_OK_AND_ASSIGN(auto model_buffer,
                       model_resources->GetTFLiteModelBuffer(
                           LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_EQ(file_region.size, model_buffer.size());

  // Error case: model type not found in the loader.
  EXPECT_THAT(model_resources->GetTFLiteModelSectionFileRegion(
                  LlmMetadata::TF_LITE_EMBEDDER),
              testing::status::StatusIs(absl::StatusCode::kNotFound));
}

TEST(ParseLlmTfLiteModelTypeTest, ParsesWireStringsCaseInsensitively) {
  EXPECT_THAT(ParseLlmTfLiteModelType("tf_lite_prefill_decode"),
              IsOkAndHolds(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_THAT(ParseLlmTfLiteModelType("TF_LITE_PREFILL_DECODE"),
              IsOkAndHolds(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_THAT(ParseLlmTfLiteModelType("Tf_Lite_Embedder"),
              IsOkAndHolds(LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_THAT(ParseLlmTfLiteModelType("tf_lite_artisan_text_decoder"),
              IsOkAndHolds(LlmMetadata::TF_LITE_ARTISAN_TEXT_DECODER));
  EXPECT_THAT(ParseLlmTfLiteModelType("tf_lite_mtp_drafter"),
              IsOkAndHolds(LlmMetadata::TF_LITE_MTP_DRAFTER));
}

TEST(ParseLlmTfLiteModelTypeTest, RejectsUnknownAndNonLlmTypes) {
  EXPECT_THAT(ParseLlmTfLiteModelType("unknown"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // The text encoder belongs to the embedding capability only.
  EXPECT_THAT(ParseLlmTfLiteModelType("tf_lite_text_encoder"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TfLiteModelTypeToWireStringTest, ReturnsLowercaseEnumName) {
  EXPECT_EQ(TfLiteModelTypeToWireString(LlmMetadata::TF_LITE_PREFILL_DECODE),
            "tf_lite_prefill_decode");
  EXPECT_EQ(
      TfLiteModelTypeToWireString(EmbeddingMetadata::TF_LITE_TEXT_ENCODER),
      "tf_lite_text_encoder");
}

}  // namespace
