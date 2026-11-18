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

#include "google/protobuf/any.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/proto/text2speech_model_type.pb.h"
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
using ::litert::lm::proto::Any2AnyMetadata;
using ::litert::lm::proto::Any2AnySessionConfig;
using ::litert::lm::proto::EmbeddingMetadata;
using ::litert::lm::proto::Flux2Klein;
using ::litert::lm::proto::Kokoro;
using ::litert::lm::proto::LlmMetadata;
using ::litert::lm::proto::MiniCpmO;
using ::litert::lm::proto::Speech2TextMetadata;
using ::litert::lm::proto::Text2ImageMetadata;
using ::litert::lm::proto::Text2SpeechMetadata;
using ::litert::lm::proto::Whisper;
using ::testing::AnyOf;
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
absl::StatusOr<std::string> WriteTruncatedPerLayerEmbedderModelFile() {
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
  auto empty_model_type = schema::CreateKeyValuePair(
      builder, "model_type",
      TfLiteModelTypeToWireString(LlmMetadata::TF_LITE_AUX));
  auto empty_items = builder.CreateVector(
      std::vector<flatbuffers::Offset<schema::KeyValuePair>>{empty_model_type});
  auto empty_section = schema::CreateSectionObject(
      builder, empty_items, /*begin_offset=*/32, /*end_offset=*/32,
      schema::AnySectionDataType_TFLiteModel);
  auto section_metadata = schema::CreateSectionMetadata(
      builder, builder.CreateVector(
                   std::vector<flatbuffers::Offset<schema::SectionObject>>{
                       section, empty_section}));
  auto metadata = schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  std::ofstream file(test_file_path, std::ios::binary);
  if (!file.good()) {
    return absl::InternalError("Failed to open test model file for writing.");
  }
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
  if (!file.good()) {
    return absl::InternalError("Failed to write test model file.");
  }
  return test_file_path.string();
}

TEST(ModelResourcesTest, GetTFLiteModelPropagatesSectionMapFailure) {
  ASSERT_OK_AND_ASSIGN(const std::string model_path,
                       WriteTruncatedPerLayerEmbedderModelFile());
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file)));
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       ModelResourcesLitertLm::Create(std::move(loader)));

  // Present-but-unmappable sections report HasTFLiteModel() == true and
  // surface the mapping error when loaded.
  EXPECT_TRUE(
      model_resources->HasTFLiteModel(LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER));
  EXPECT_THAT(
      model_resources->GetTFLiteModel(LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER),
      StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(model_resources->GetTFLiteModelBuffer(
                  LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER),
              StatusIs(absl::StatusCode::kInternal));

  // Sections that are absent from the header (or 0-byte sections) report
  // HasTFLiteModel() == false and return kNotFound if queried directly.
  EXPECT_FALSE(model_resources->HasTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_THAT(model_resources->GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      model_resources->GetTFLiteModelBuffer(LlmMetadata::TF_LITE_EMBEDDER),
      StatusIs(absl::StatusCode::kNotFound));
  EXPECT_FALSE(model_resources->HasTFLiteModel(LlmMetadata::TF_LITE_AUX));
  EXPECT_THAT(model_resources->GetTFLiteModel(LlmMetadata::TF_LITE_AUX),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(model_resources->GetTFLiteModelBuffer(LlmMetadata::TF_LITE_AUX),
              StatusIs(absl::StatusCode::kNotFound));
}

// File-backed loading maps the section itself (or falls back to the buffer
// where that is unsupported). A section that fails to map must still not read
// as a missing model.
TEST(ModelResourcesTest,
     GetTFLiteModelPropagatesSectionMapFailureWithFileBackedLoading) {
  ASSERT_OK_AND_ASSIGN(const std::string model_path,
                       WriteTruncatedPerLayerEmbedderModelFile());
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file)));
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResourcesLitertLm::Create(
          std::move(loader), /*enable_file_backed_model_loading=*/true));

  EXPECT_THAT(
      model_resources->GetTFLiteModel(LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER),
      StatusIs(
          AnyOf(absl::StatusCode::kUnavailable, absl::StatusCode::kInternal)));
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
  EXPECT_EQ(
      TfLiteModelTypeToWireString(Any2AnyMetadata::TF_LITE_VQ_IMAGE_EMBEDDER),
      "tf_lite_vq_image_embedder");
  EXPECT_EQ(TfLiteModelTypeToWireString(Any2AnyMetadata::TF_LITE_GEN_HEAD),
            "tf_lite_gen_head");
  EXPECT_EQ(
      TfLiteModelTypeToWireString(Any2AnyMetadata::TF_LITE_VQ_IMAGE_DECODER),
      "tf_lite_vq_image_decoder");
}

absl::StatusOr<std::string> WriteMetadataLitertLmFile(
    absl::string_view file_name, schema::AnySectionDataType section_type,
    absl::string_view payload) {
  auto test_file_path =
      std::filesystem::path(::testing::TempDir()) / std::string(file_name);

  // Measure the serialized FlatBuffer header size with placeholder offsets,
  // then rebuild with exact offsets placing the payload right after the header.
  auto build_header = [&](uint64_t begin_offset, uint64_t end_offset,
                          flatbuffers::FlatBufferBuilder& builder) {
    auto section = schema::CreateSectionObject(
        builder, /*items=*/0, begin_offset, end_offset, section_type);
    auto sections = builder.CreateVector(
        std::vector<flatbuffers::Offset<schema::SectionObject>>{section});
    auto section_metadata = schema::CreateSectionMetadata(builder, sections);
    auto metadata =
        schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
    builder.Finish(metadata);
  };

  flatbuffers::FlatBufferBuilder size_probe(256);
  build_header(/*begin_offset=*/1, /*end_offset=*/1 + payload.size(),
               size_probe);
  const uint64_t header_end_offset = 32 + size_probe.GetSize();

  flatbuffers::FlatBufferBuilder builder(256);
  build_header(header_end_offset, header_end_offset + payload.size(), builder);
  const uint64_t actual_header_end_offset = 32 + builder.GetSize();
  if (actual_header_end_offset != header_end_offset) {
    return absl::InternalError("Header size changed after offset fill.");
  }

  std::ofstream file(test_file_path, std::ios::binary);
  if (!file.good()) {
    return absl::InternalError("Failed to open test model file for writing.");
  }
  file.write("LITERTLM", 8);
  const uint32_t major_version = 1;
  const uint32_t minor_version = 0;
  const uint32_t patch_version = 0;
  const uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&major_version), 4);
  file.write(reinterpret_cast<const char*>(&minor_version), 4);
  file.write(reinterpret_cast<const char*>(&patch_version), 4);
  file.write(reinterpret_cast<const char*>(&padding), 4);
  file.write(reinterpret_cast<const char*>(&actual_header_end_offset), 8);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  file.write(payload.data(), payload.size());
  if (!file.good()) {
    return absl::InternalError("Failed to write test model file.");
  }
  return test_file_path.string();
}

TEST(ModelResourcesTest, RoundTripsCustomModelAnyInAny2AnyMetadata) {
  Any2AnyMetadata input_metadata;
  input_metadata.set_input_sample_rate_hz(16000);
  MiniCpmO packed_payload;
  packed_payload.set_llm_hidden_size(4096);
  packed_payload.set_voice("custom_voice");
  ASSERT_TRUE(input_metadata.mutable_any2any_model_type()
                  ->mutable_custom_model()
                  ->PackFrom(packed_payload));

  ASSERT_OK_AND_ASSIGN(
      const std::string model_path,
      WriteMetadataLitertLmFile("any2any_custom_model.litertlm",
                                schema::AnySectionDataType_Any2AnyMetadataProto,
                                input_metadata.SerializeAsString()));
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file)));
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       ModelResourcesLitertLm::Create(std::move(loader)));

  ASSERT_OK_AND_ASSIGN(const Any2AnyMetadata* loaded,
                       model_resources->GetAny2AnyMetadata());
  EXPECT_EQ(loaded->input_sample_rate_hz(), 16000);
  ASSERT_TRUE(loaded->any2any_model_type().has_custom_model());
  const auto& custom_any = loaded->any2any_model_type().custom_model();
  EXPECT_EQ(absl::string_view(custom_any.type_url()),
            "type.googleapis.com/litert.lm.proto.MiniCpmO");
  EXPECT_TRUE(custom_any.Is<MiniCpmO>());

  MiniCpmO unpacked;
  ASSERT_TRUE(custom_any.UnpackTo(&unpacked));
  EXPECT_EQ(unpacked.llm_hidden_size(), 4096);
  EXPECT_EQ(unpacked.voice(), "custom_voice");
}

TEST(ModelResourcesTest, RoundTripsCustomModelAnyAcrossOtherOmniCapabilities) {
  // Speech2TextMetadata with packed Whisper payload.
  {
    Speech2TextMetadata input;
    Whisper payload;
    ASSERT_TRUE(
        input.mutable_asr_model_type()->mutable_custom_model()->PackFrom(
            payload));
    ASSERT_OK_AND_ASSIGN(
        const std::string path,
        WriteMetadataLitertLmFile(
            "s2t_custom_model.litertlm",
            schema::AnySectionDataType_Speech2TextMetadataProto,
            input.SerializeAsString()));
    ASSERT_OK_AND_ASSIGN(auto file, ScopedFile::Open(path));
    ASSERT_OK_AND_ASSIGN(auto loader, LitertLmLoader::Create(std::move(file)));
    ASSERT_OK_AND_ASSIGN(auto resources,
                         ModelResourcesLitertLm::Create(std::move(loader)));
    ASSERT_OK_AND_ASSIGN(const Speech2TextMetadata* loaded,
                         resources->GetSpeech2TextMetadata());
    ASSERT_TRUE(loaded->asr_model_type().has_custom_model());
    EXPECT_TRUE(loaded->asr_model_type().custom_model().Is<Whisper>());
  }

  // Text2SpeechMetadata with packed Kokoro payload.
  {
    Text2SpeechMetadata input;
    Kokoro payload;
    payload.set_voice_name("custom_kokoro");
    ASSERT_TRUE(
        input.mutable_tts_model_type()->mutable_custom_model()->PackFrom(
            payload));
    ASSERT_OK_AND_ASSIGN(
        const std::string path,
        WriteMetadataLitertLmFile(
            "t2s_custom_model.litertlm",
            schema::AnySectionDataType_Text2SpeechMetadataProto,
            input.SerializeAsString()));
    ASSERT_OK_AND_ASSIGN(auto file, ScopedFile::Open(path));
    ASSERT_OK_AND_ASSIGN(auto loader, LitertLmLoader::Create(std::move(file)));
    ASSERT_OK_AND_ASSIGN(auto resources,
                         ModelResourcesLitertLm::Create(std::move(loader)));
    ASSERT_OK_AND_ASSIGN(const Text2SpeechMetadata* loaded,
                         resources->GetText2SpeechMetadata());
    ASSERT_TRUE(loaded->tts_model_type().has_custom_model());
    Kokoro unpacked;
    ASSERT_TRUE(loaded->tts_model_type().custom_model().UnpackTo(&unpacked));
    EXPECT_EQ(unpacked.voice_name(), "custom_kokoro");
  }

  // Text2ImageMetadata with packed Flux2Klein payload.
  {
    Text2ImageMetadata input;
    Flux2Klein payload;
    payload.set_text_embed_table_key("custom_embed_key");
    ASSERT_TRUE(
        input.mutable_image_gen_model_type()->mutable_custom_model()->PackFrom(
            payload));
    ASSERT_OK_AND_ASSIGN(const std::string path,
                         WriteMetadataLitertLmFile(
                             "t2i_custom_model.litertlm",
                             schema::AnySectionDataType_Text2ImageMetadataProto,
                             input.SerializeAsString()));
    ASSERT_OK_AND_ASSIGN(auto file, ScopedFile::Open(path));
    ASSERT_OK_AND_ASSIGN(auto loader, LitertLmLoader::Create(std::move(file)));
    ASSERT_OK_AND_ASSIGN(auto resources,
                         ModelResourcesLitertLm::Create(std::move(loader)));
    ASSERT_OK_AND_ASSIGN(const Text2ImageMetadata* loaded,
                         resources->GetText2ImageMetadata());
    ASSERT_TRUE(loaded->image_gen_model_type().has_custom_model());
    Flux2Klein unpacked;
    ASSERT_TRUE(
        loaded->image_gen_model_type().custom_model().UnpackTo(&unpacked));
    EXPECT_EQ(unpacked.text_embed_table_key(), "custom_embed_key");
  }
}

TEST(ModelResourcesTest, Any2AnySessionConfigVisualGenFieldsRoundTrip) {
  Any2AnySessionConfig config;
  config.set_generate_image(true);
  config.set_cfg_weight(5.0f);
  config.set_image_temperature(0.8f);
  config.set_seed(42);

  Any2AnySessionConfig parsed;
  ASSERT_TRUE(parsed.ParseFromString(config.SerializeAsString()));
  EXPECT_TRUE(parsed.generate_image());
  EXPECT_FLOAT_EQ(parsed.cfg_weight(), 5.0f);
  EXPECT_FLOAT_EQ(parsed.image_temperature(), 0.8f);
  EXPECT_EQ(parsed.seed(), 42);
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
  EXPECT_FALSE(resources.HasTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_FALSE(resources.HasTFLiteModel("tf_lite_prefill_decode"));
  EXPECT_FALSE(resources.HasTFLiteModel("unknown"));
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
