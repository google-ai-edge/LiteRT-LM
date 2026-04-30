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

#include "runtime/util/litert_lm_loader.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "runtime/components/model_resources.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm {

namespace {

using ::testing::status::StatusIs;

void WriteDummyModelFile(
    const std::string& path,
    flatbuffers::Offset<schema::LiteRTLMMetaData> metadata_offset,
    flatbuffers::FlatBufferBuilder& builder) {
  builder.Finish(metadata_offset);
  std::ofstream file(path, std::ios::binary);
  // Write magic number
  file.write("LITERTLM", 8);
  // Write major version
  uint32_t major_version = 1;
  file.write(reinterpret_cast<const char*>(&major_version), sizeof(uint32_t));
  // Write minor version
  uint32_t minor_version = 0;
  file.write(reinterpret_cast<const char*>(&minor_version), sizeof(uint32_t));
  // Write patch version
  uint32_t patch_version = 0;
  file.write(reinterpret_cast<const char*>(&patch_version), sizeof(uint32_t));
  // Write 4 bytes of padding
  uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&padding), sizeof(uint32_t));
  // Write header end offset (32 + metadata size)
  uint64_t header_end_offset = 32 + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end_offset),
             sizeof(uint64_t));
  // Write metadata
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
}

void WriteDummyModelFile(const std::string& path,
                         const flatbuffers::FlatBufferBuilder& builder,
                         uint64_t large_decompression_size = 0) {
  std::ofstream file(path, std::ios::binary);
  file.write("LITERTLM", 8);
  uint32_t major = 1;
  uint32_t minor = 0;
  uint32_t patch = 0;
  uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&major), 4);
  file.write(reinterpret_cast<const char*>(&minor), 4);
  file.write(reinterpret_cast<const char*>(&patch), 4);
  file.write(reinterpret_cast<const char*>(&padding), 4);

  uint64_t header_size = builder.GetSize();
  uint64_t header_end_offset = 32 + header_size;
  file.write(reinterpret_cast<const char*>(&header_end_offset), 8);

  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             header_size);

  if (large_decompression_size > 0) {
    file.write(reinterpret_cast<const char*>(&large_decompression_size),
               sizeof(large_decompression_size));
  }
  file.close();
}

TEST(LitertLmLoaderTest, InitializeWithInvalidOffsets) {
  auto test_file_path =
      std::filesystem::path(::testing::TempDir()) / "invalid_offsets.litertlm";

  flatbuffers::FlatBufferBuilder builder(1024);
  auto section_object = schema::CreateSectionObject(
      builder, 0, 100, 50,
      schema::AnySectionDataType_TFLiteModel);  // begin > end
  std::vector<flatbuffers::Offset<schema::SectionObject>> sections;
  sections.push_back(section_object);
  auto sections_vector = builder.CreateVector(sections);
  auto section_metadata =
      schema::CreateSectionMetadata(builder, sections_vector);
  auto metadata = schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);

  WriteDummyModelFile(test_file_path.string(), metadata, builder);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> mapped_file,
                       MemoryMappedFile::Create(test_file_path.string()));

  auto shared_mapped_file =
      std::shared_ptr<MemoryMappedFile>(std::move(mapped_file));
  EXPECT_THAT(LitertLmLoader::Create(shared_mapped_file),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("invalid offsets")));
}

TEST(LitertLmLoaderTest, GetSectionLocationNotFound) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> mapped_file,
                       MemoryMappedFile::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader_ptr,
                       LitertLmLoader::Create(std::move(mapped_file)));
  auto& loader = *loader_ptr;

  BufferKey embedder_key(schema::AnySectionDataType_TFLiteModel,
                         ModelType::kTfLiteEmbedder);
  EXPECT_THAT(loader.GetSectionLocation(embedder_key),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(LitertLmLoaderTest, InitializeWithSentencePieceFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader_ptr,
                       LitertLmLoader::Create(std::move(model_file)));
  auto& loader = *loader_ptr;
  EXPECT_FALSE(loader.GetHuggingFaceTokenizer());
  EXPECT_GT(loader.GetSentencePieceTokenizer()->Size(), 0);
  EXPECT_GT(loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size(), 0);
  EXPECT_GT(loader.GetLlmMetadata().Size(), 0);
  // Try to get non-existent TFLite model.
  EXPECT_EQ(loader.GetTFLiteModel(ModelType::kTfLiteEmbedder).Size(), 0);
}

TEST(LitertLmLoaderTest, InitializeWithHuggingFaceFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader_ptr,
                       LitertLmLoader::Create(std::move(model_file)));
  auto& loader = *loader_ptr;
  ASSERT_GT(loader.GetHuggingFaceTokenizer()->Size(), 0);
  ASSERT_FALSE(loader.GetSentencePieceTokenizer());
}

TEST(LitertLmLoaderTest, InitializeWithMemoryMappedFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> mapped_file,
                       MemoryMappedFile::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader_ptr,
                       LitertLmLoader::Create(std::move(mapped_file)));
  auto& loader = *loader_ptr;
  EXPECT_FALSE(loader.GetHuggingFaceTokenizer());
  EXPECT_GT(loader.GetSentencePieceTokenizer()->Size(), 0);
  EXPECT_GT(loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size(), 0);
  EXPECT_GT(loader.GetLlmMetadata().Size(), 0);
  EXPECT_EQ(loader.GetTFLiteModel(ModelType::kTfLiteEmbedder).Size(), 0);
}

TEST(LitertLmLoaderTest, GetSectionLocationSizeMatch) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> mapped_file,
                       MemoryMappedFile::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader_ptr,
                       LitertLmLoader::Create(std::move(mapped_file)));
  auto& loader = *loader_ptr;

  BufferKey sp_key(schema::AnySectionDataType_SP_Tokenizer);
  ASSERT_OK_AND_ASSIGN(auto sp_location, loader.GetSectionLocation(sp_key));
  EXPECT_EQ(sp_location.second - sp_location.first,
            loader.GetSentencePieceTokenizer()->Size());

  BufferKey model_key(schema::AnySectionDataType_TFLiteModel,
                      ModelType::kTfLitePrefillDecode);
  ASSERT_OK_AND_ASSIGN(auto model_location,
                       loader.GetSectionLocation(model_key));
  EXPECT_EQ(model_location.second - model_location.first,
            loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size());

  BufferKey metadata_key(schema::AnySectionDataType_LlmMetadataProto);
  ASSERT_OK_AND_ASSIGN(auto metadata_location,
                       loader.GetSectionLocation(metadata_key));
  EXPECT_EQ(metadata_location.second - metadata_location.first,
            loader.GetLlmMetadata().Size());
}

TEST(LitertLmLoaderTest, GetHuggingFaceTokenizerLargeDecompressionSize) {
  flatbuffers::FlatBufferBuilder builder;

  auto section_object = schema::CreateSectionObject(
      builder, 0, 0, 8, schema::AnySectionDataType_HF_Tokenizer_Zlib);
  std::vector<flatbuffers::Offset<schema::SectionObject>>
      section_objects_vector = {section_object};
  auto section_metadata_offset = schema::CreateSectionMetadata(
      builder, builder.CreateVector(section_objects_vector));
  auto root_offset =
      schema::CreateLiteRTLMMetaData(builder, 0, section_metadata_offset);
  builder.Finish(root_offset);

  size_t header_size = builder.GetSize();
  size_t total_header_size = 32 + header_size;

  builder.Clear();
  section_object = schema::CreateSectionObject(
      builder, 0, total_header_size, total_header_size + 8,
      schema::AnySectionDataType_HF_Tokenizer_Zlib);
  section_objects_vector = {section_object};
  section_metadata_offset = schema::CreateSectionMetadata(
      builder, builder.CreateVector(section_objects_vector));
  root_offset =
      schema::CreateLiteRTLMMetaData(builder, 0, section_metadata_offset);
  builder.Finish(root_offset);

  auto header_path =
      std::filesystem::path(::testing::TempDir()) / "large_decompression.bin";
  uint64_t large_size = 1024ULL * 1024ULL * 1024ULL + 1ULL;  // 1GB + 1 byte
  WriteDummyModelFile(header_path.string(), builder, large_size);

  auto model_file = ScopedFile::Open(header_path.string());
  ASSERT_TRUE(model_file.ok());

  ASSERT_OK_AND_ASSIGN(auto loader, LitertLmLoader::Create(std::move(model_file.value())));

  auto tokenizer = loader->GetHuggingFaceTokenizer();
  EXPECT_FALSE(tokenizer.has_value());
}

}  // namespace
}  // namespace litert::lm
