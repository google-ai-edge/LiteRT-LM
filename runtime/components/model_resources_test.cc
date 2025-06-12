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
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"

namespace {

using ::litert::lm::LitertLmLoader;
using ::litert::lm::ModelAssetBundleResources;
using ::litert::lm::ModelResourcesLitertLm;
using ::litert::lm::ModelResourcesTask;
using ::litert::lm::ModelType;
using ::litert::lm::ModelTypeToString;
using ::litert::lm::ScopedFile;
using ::litert::lm::StringToModelType;

TEST(ModelResourcesTest, InitializeWithValidLitertLmLoader) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  ASSERT_GT(loader.GetSentencePieceTokenizer()->Size(), 0);
  ASSERT_GT(loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size(), 0);

  auto model_resources = ModelResourcesLitertLm::Create(
      std::make_unique<LitertLmLoader>(std::move(loader)));
  ASSERT_OK(model_resources);

  auto tflite_model =
      model_resources.value()->GetTFLiteModel(ModelType::kTfLitePrefillDecode);
  ASSERT_OK(tflite_model);
  ASSERT_GT(tflite_model.value()->GetNumSignatures(), 0);

  auto tokenizer = model_resources.value()->GetTokenizer();
  ASSERT_OK(tokenizer);
  ASSERT_NE(tokenizer.value(), nullptr);
}

TEST(ModelResourcesTest, InitializeWithHuggingFaceTokenizer) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  ASSERT_GT(loader.GetHuggingFaceTokenizer()->Size(), 0);

  auto model_resources = ModelResourcesLitertLm::Create(
      std::make_unique<LitertLmLoader>(std::move(loader)));
  ASSERT_OK(model_resources);

  auto tokenizer = model_resources.value()->GetTokenizer();
  ASSERT_OK(tokenizer);
  ASSERT_NE(tokenizer.value(), nullptr);
}

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

  auto tflite_model =
      model_resources.value()->GetTFLiteModel(ModelType::kTfLitePrefillDecode);
  ASSERT_OK(tflite_model);
  ASSERT_GT(tflite_model.value()->GetNumSignatures(), 0);

  auto tokenizer = model_resources.value()->GetTokenizer();
  ASSERT_OK(tokenizer);
  ASSERT_NE(tokenizer.value(), nullptr);
}

TEST(ModelTypeConversionTest, StringToModelType) {
  auto result = StringToModelType("tf_lite_prefill_decode");
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), ModelType::kTfLitePrefillDecode);

  result = StringToModelType("TF_LITE_PREFILL_DECODE");
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), ModelType::kTfLitePrefillDecode);

  result = StringToModelType("tf_lite_embedder");
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), ModelType::kTfLiteEmbedder);

  result = StringToModelType("TF_LITE_EMBEDDER");
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), ModelType::kTfLiteEmbedder);

  result = StringToModelType("tf_lite_per_layer_embedder");
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), ModelType::kTfLitePerLayerEmbedder);

  result = StringToModelType("TF_LITE_PER_LAYER_EMBEDDER");
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), ModelType::kTfLitePerLayerEmbedder);

  result = StringToModelType("unknown");
  EXPECT_FALSE(result.ok());
}

TEST(ModelTypeConversionTest, ModelTypeToString) {
  EXPECT_EQ(ModelTypeToString(ModelType::kTfLitePrefillDecode),
            "TF_LITE_PREFILL_DECODE");
  EXPECT_EQ(ModelTypeToString(ModelType::kTfLiteEmbedder), "TF_LITE_EMBEDDER");
  EXPECT_EQ(ModelTypeToString(ModelType::kTfLitePerLayerEmbedder),
            "TF_LITE_PER_LAYER_EMBEDDER");
  EXPECT_EQ(ModelTypeToString(ModelType::kUnknown), "UNKNOWN");
}

}  // namespace
