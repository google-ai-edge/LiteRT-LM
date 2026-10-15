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

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"

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
namespace schema = ::litert::lm::schema;

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
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      model_resources.value()->GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER),
      StatusIs(absl::StatusCode::kNotFound));
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
  EXPECT_THAT(tflite_model, StatusIs(absl::StatusCode::kNotFound));
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
              StatusIs(absl::StatusCode::kNotFound));
}

// Writes a .litertlm file whose header declares a per-layer embedder section
// that extends far past the end of the file. The section is present, so the
// loader must fail to map it rather than report it as missing.
std::string WriteTruncatedPerLayerEmbedderModelFile() {
  auto test_file_path = std::filesystem::path(::testing::TempDir()) /
                        "model_resources_truncated_per_layer_embedder.litertlm";

  flatbuffers::FlatBufferBuilder builder(1024);
  auto model_type = schema::CreateKeyValuePair(
      builder, "model_type",
      TfLiteModelTypeToWireString(LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER));
  auto items = builder.CreateVector(
      std::vector<flatbuffers::Offset<schema::KeyValuePair>>{model_type});
  auto section = schema::CreateSectionObject(
      builder, items, /*begin_offset=*/32, /*end_offset=*/1 << 20,
      schema::AnySectionDataType_TFLiteModel);
  auto section_metadata = schema::CreateSectionMetadata(
      builder,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<schema::SectionObject>>{section}));
  auto metadata = schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  std::ofstream file(test_file_path, std::ios::binary);
  file.write("LITERTLM", 8);
  const uint32_t major_version = 1;
  const uint32_t minor_version = 0;
  const uint32_t patch_version = 0;
  const uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&major_version), 4);
  file.write(reinterpret_cast<const char*>(&minor_version), 4);
  file.write(reinterpret_cast<const char*>(&patch_version), 4);
  file.write(reinterpret_cast<const char*>(&padding), 4);
  const uint64_t header_end_offset = 32 + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  return test_file_path.string();
}

TEST(ModelResourcesTest, GetTFLiteModelPropagatesSectionMapFailure) {
  const std::string model_path = WriteTruncatedPerLayerEmbedderModelFile();
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file)));
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       ModelResourcesLitertLm::Create(std::move(loader)));

  // Present-but-unmappable sections surface the mapping error rather than
  // kNotFound, so callers that treat kNotFound as "optional model absent"
  // fail loudly instead of running without the model.
  EXPECT_THAT(
      model_resources->GetTFLiteModel(LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER),
      StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(model_resources->GetTFLiteModelBuffer(
                  LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER),
              StatusIs(absl::StatusCode::kInternal));

  // Sections that are absent from the header are still kNotFound.
  EXPECT_THAT(model_resources->GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      model_resources->GetTFLiteModelBuffer(LlmMetadata::TF_LITE_EMBEDDER),
      StatusIs(absl::StatusCode::kNotFound));
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

class MinimalModelResources : public ::litert::lm::ModelResources {
 public:
  absl::StatusOr<std::unique_ptr<::litert::lm::Tokenizer>> GetTokenizer()
      override {
    return absl::UnimplementedError("Minimal test tokenizer");
  }
  absl::StatusOr<const LlmMetadata*> GetLlmMetadata() override {
    return &llm_metadata_;
  }

 private:
  LlmMetadata llm_metadata_;
};

TEST(ModelResourcesTest, DefaultImplementationsForMinimalSubclass) {
  MinimalModelResources resources;
  EXPECT_THAT(resources.GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(
      resources.GetTFLiteModelBuffer(LlmMetadata::TF_LITE_PREFILL_DECODE),
      StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(resources.GetScopedFile(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(
      resources.GetWeightsSectionOffset(LlmMetadata::TF_LITE_PREFILL_DECODE),
      StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(resources.GetTFLiteModelSectionFileRegion(
                  LlmMetadata::TF_LITE_PREFILL_DECODE),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_EQ(resources.GetTFLiteModelBackendConstraint(
                LlmMetadata::TF_LITE_PREFILL_DECODE),
            std::nullopt);
  EXPECT_EQ(resources.GetTFLiteModelPreferActivationType(
                LlmMetadata::TF_LITE_PREFILL_DECODE),
            std::nullopt);
  EXPECT_THAT(resources.GetExecutorMetadata(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(resources.GetLlmMetadata(), ::absl_testing::IsOk());
}

}  // namespace
