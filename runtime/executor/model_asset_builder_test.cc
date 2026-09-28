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

#include "runtime/executor/model_asset_builder.h"

#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "google/protobuf/message.h"  // from @com_google_protobuf

namespace litert::lm {
namespace {

using ::testing::NotNull;

std::filesystem::path SchemaTestDataDir() {
  return std::filesystem::path(::testing::SrcDir()) /
         "litert_lm/schema/testdata";
}

std::filesystem::path TokenizerTestDataDir() {
  return std::filesystem::path(::testing::SrcDir()) /
         "litert_lm/support/tokenizer/testdata";
}

TEST(ModelAssetBuilderTest, BuildFromSectionFilesAndCreateResources) {
  ModelAssetBuilder builder;
  builder
      .SetLlmMetadataFromFile(
          (SchemaTestDataDir() / "llm_metadata.pb").string())
      .AddTFLiteModel((SchemaTestDataDir() / "attention.tflite").string(),
                      ModelType::kTfLitePrefillDecode,
                      /*backend_constraint=*/"cpu",
                      /*prefer_activation_type=*/"float16")
      .AddTFLiteWeights((SchemaTestDataDir() / "data.bin").string(),
                        ModelType::kTfLitePrefillDecode)
      .SetSentencePieceTokenizer(
          (TokenizerTestDataDir() / "sentencepiece.model").string())
      .SetCacheKeyPath("runtime_built_model");

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, builder.Build());
  EXPECT_TRUE(model_assets.HasModelResourcesFactory());
  ASSERT_OK_AND_ASSIGN(auto path, model_assets.GetPath());
  EXPECT_EQ(path, "runtime_built_model");

  ASSERT_OK_AND_ASSIGN(
      auto resources,
      BuildLiteRtCompiledModelResources(
          model_assets, /*enable_file_backed_model_loading=*/false,
          /*enable_file_backed_for_aot_npu=*/false));
  EXPECT_THAT(resources, NotNull());

  ASSERT_OK_AND_ASSIGN(
      const litert::Model* model,
      resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  EXPECT_THAT(model, NotNull());

  ASSERT_OK_AND_ASSIGN(auto scoped_file, resources->GetScopedFile(
                                             ModelType::kTfLitePrefillDecode));
  EXPECT_TRUE(scoped_file.get().IsValid());

  ASSERT_OK_AND_ASSIGN(
      auto weights_offset,
      resources->GetWeightsSectionOffset(ModelType::kTfLitePrefillDecode));
  EXPECT_EQ(weights_offset.first, 0);
  EXPECT_GT(weights_offset.second, 0);

  ASSERT_OK_AND_ASSIGN(auto tokenizer, resources->GetTokenizer());
  EXPECT_THAT(tokenizer, NotNull());

  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* metadata,
                       resources->GetLlmMetadata());
  EXPECT_THAT(metadata, NotNull());

  ASSERT_OK_AND_ASSIGN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(model_assets, Backend::CPU));
  EXPECT_EQ(engine_settings.GetMainExecutorSettings().GetBackend(),
            Backend::CPU);
}

TEST(ModelAssetBuilderTest, BuildFromTextprotoMetadataFile) {
  if constexpr (!std::is_base_of_v<proto2::Message, proto::LlmMetadata>) {
    GTEST_SKIP() << "Textproto parsing is not supported in protobuf lite "
                    "builds";
  }
  const std::filesystem::path pbtxt_path =
      std::filesystem::path(::testing::TempDir()) / "test_llm_metadata.pbtxt";
  {
    std::ofstream out(pbtxt_path);
    out << "max_num_tokens: 4096\n";
  }

  ModelAssetBuilder builder;
  builder.SetLlmMetadataFromFile(pbtxt_path.string())
      .AddTFLiteModel((SchemaTestDataDir() / "attention.tflite").string());

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, builder.Build());
  ASSERT_OK_AND_ASSIGN(
      auto resources,
      BuildLiteRtCompiledModelResources(
          model_assets, /*enable_file_backed_model_loading=*/false,
          /*enable_file_backed_for_aot_npu=*/false));
  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* metadata,
                       resources->GetLlmMetadata());
  EXPECT_EQ(metadata->max_num_tokens(), 4096);
}

}  // namespace
}  // namespace litert::lm
