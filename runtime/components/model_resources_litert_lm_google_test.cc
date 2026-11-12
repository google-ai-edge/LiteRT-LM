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

#include "runtime/components/model_resources_litert_lm_google.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl  // IWYU pragma: keep
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

std::string BuildArtisanTfliteBytes() {
  odml::infra::proto::LlmParameters params;
  params.set_vocab_size(32000);
  params.set_start_token_id(1);
  std::string params_bytes = params.SerializeAsString();

  constexpr uint64_t kHeaderPaddedSize = 1024;
  flatbuffers::FlatBufferBuilder b;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(b),
      tflite::CreateBuffer(b, /*data=*/0, kHeaderPaddedSize,
                           params_bytes.size()),
  };
  std::vector<flatbuffers::Offset<tflite::Metadata>> metadata = {
      tflite::CreateMetadataDirect(b, "odml.infra.proto.LlmParameters", 1),
  };
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
      tflite::CreateSubGraph(b),
  };
  auto model = tflite::CreateModel(
      b, /*version=*/3, /*operator_codes=*/0, b.CreateVector(subgraphs),
      b.CreateString("artisan_test"), b.CreateVector(buffers),
      /*metadata_buffer=*/0, b.CreateVector(metadata));
  tflite::FinishModelBuffer(b, model);
  std::string out(reinterpret_cast<const char*>(b.GetBufferPointer()),
                  b.GetSize());
  out.resize(kHeaderPaddedSize, '\0');
  out.append(params_bytes);
  return out;
}

std::filesystem::path WriteSyntheticArtisanLitertLmFile(
    absl::string_view filename, bool include_artisan_section = true) {
  std::string tflite_bytes = BuildArtisanTfliteBytes();
  constexpr uint64_t kSectionStart = 4096;

  flatbuffers::FlatBufferBuilder builder(1024);
  std::vector<flatbuffers::Offset<schema::SectionObject>> sections;
  if (include_artisan_section) {
    auto str_val = schema::CreateStringValueDirect(
        builder, "tf_lite_artisan_text_decoder");
    auto kv = schema::CreateKeyValuePairDirect(
        builder, "model_type", schema::VData_StringValue, str_val.Union());
    std::vector<flatbuffers::Offset<schema::KeyValuePair>> items = {kv};
    sections.push_back(schema::CreateSectionObjectDirect(
        builder, &items, kSectionStart, kSectionStart + tflite_bytes.size(),
        schema::AnySectionDataType_TFLiteModel));
  }
  auto section_metadata =
      schema::CreateSectionMetadataDirect(builder, &sections);
  auto metadata = schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  auto path =
      std::filesystem::path(::testing::TempDir()) / std::string(filename);
  std::ofstream file(path, std::ios::binary);
  file.write("LITERTLM", 8);
  const uint32_t header_words[4] = {1, 0, 0, 0};
  file.write(reinterpret_cast<const char*>(header_words), sizeof(header_words));
  const uint64_t header_end_offset = 32 + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  std::string pad(kSectionStart - header_end_offset, '\0');
  file.write(pad.data(), pad.size());
  if (include_artisan_section) {
    file.write(tflite_bytes.data(), tflite_bytes.size());
  }
  const size_t current_size =
      kSectionStart + (include_artisan_section ? tflite_bytes.size() : 0);
  const size_t alignment = MemoryMappedFile::GetOffsetAlignment();
  if (current_size % alignment != 0) {
    std::string tail_pad(alignment - (current_size % alignment), '\0');
    file.write(tail_pad.data(), tail_pad.size());
  }
  file.close();
  return path;
}

TEST(ModelResourcesLitertLmGoogleTest, LoadsArtisanModelDataFromSharedFile) {
  auto path = WriteSyntheticArtisanLitertLmFile("artisan_shared_file.litertlm");
  ASSERT_OK_AND_ASSIGN(ScopedFile scoped_file, ScopedFile::Open(path.string()));
  auto shared_file = std::make_shared<ScopedFile>(std::move(scoped_file));
  ASSERT_OK_AND_ASSIGN(auto loader, LitertLmLoader::Create(shared_file));
  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesLitertLmGoogle::Create(std::move(loader)));

  ASSERT_OK_AND_ASSIGN(auto model_data, resources->GetSharedArtisanModeldata());
  ASSERT_NE(model_data, nullptr);
  EXPECT_EQ(model_data->GetLlmParameters().vocab_size(), 32000);
  // Cached second call returns the same instance.
  ASSERT_OK_AND_ASSIGN(auto cached_data,
                       resources->GetSharedArtisanModeldata());
  EXPECT_EQ(cached_data.get(), model_data.get());
}

TEST(ModelResourcesLitertLmGoogleTest, LoadsArtisanModelDataFromMappedBuffer) {
  auto path = WriteSyntheticArtisanLitertLmFile("artisan_mapped.litertlm");
  ASSERT_OK_AND_ASSIGN(auto mapped_file,
                       MemoryMappedFile::Create(path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(mapped_file)));
  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesLitertLmGoogle::Create(std::move(loader)));

  ASSERT_OK_AND_ASSIGN(auto model_data, resources->GetSharedArtisanModeldata());
  ASSERT_NE(model_data, nullptr);
  EXPECT_EQ(model_data->GetLlmParameters().vocab_size(), 32000);
}

TEST(ModelResourcesLitertLmGoogleTest, FailsWhenArtisanSectionMissing) {
  auto path =
      WriteSyntheticArtisanLitertLmFile("missing_artisan.litertlm",
                                        /*include_artisan_section=*/false);
  ASSERT_OK_AND_ASSIGN(auto mapped_file,
                       MemoryMappedFile::Create(path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(mapped_file)));
  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesLitertLmGoogle::Create(std::move(loader)));

  EXPECT_THAT(resources->GetSharedArtisanModeldata(),
              StatusIs(absl::StatusCode::kDataLoss));
}

}  // namespace
}  // namespace litert::lm
