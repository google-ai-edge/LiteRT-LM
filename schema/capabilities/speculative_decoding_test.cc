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

#include "schema/capabilities/speculative_decoding.h"

#include <cstddef>
#include <cstdint>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm::schema::capabilities {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

// Helper to create a complete LiteRT-LM file in memory with a given model type.
std::string CreateLiteRTLMBinaryString(const std::string& model_type) {
  flatbuffers::FlatBufferBuilder builder;

  flatbuffers::Offset<SectionMetadata> section_metadata = 0;
  if (!model_type.empty()) {
    auto kvp = CreateKeyValuePair(builder, "model_type", model_type);
    std::vector<flatbuffers::Offset<KeyValuePair>> items_vector = {kvp};
    auto section_object =
        CreateSectionObject(builder, builder.CreateVector(items_vector), 0, 100,
                            AnySectionDataType_TFLiteModel);

    std::vector<flatbuffers::Offset<SectionObject>> section_objects_vector = {
        section_object};
    section_metadata = CreateSectionMetadata(
        builder, builder.CreateVector(section_objects_vector));
  } else {
    // Create valid section empty of model_type metadata
    auto section_object =
        CreateSectionObject(builder, 0, 0, 100, AnySectionDataType_TFLiteModel);
    std::vector<flatbuffers::Offset<SectionObject>> section_objects_vector = {
        section_object};
    section_metadata = CreateSectionMetadata(
        builder, builder.CreateVector(section_objects_vector));
  }

  auto root = CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(root);

  size_t flatbuffer_size = builder.GetSize();

  std::ostringstream output_stream(std::ios::binary);

  // 0. Write magic number and versions
  output_stream.write("LITERTLM", 8);
  output_stream.write(reinterpret_cast<const char*>(&LITERTLM_MAJOR_VERSION),
                      sizeof(uint32_t));
  output_stream.write(reinterpret_cast<const char*>(&LITERTLM_MINOR_VERSION),
                      sizeof(uint32_t));
  output_stream.write(reinterpret_cast<const char*>(&LITERTLM_PATCH_VERSION),
                      sizeof(uint32_t));

  // 1. Write padding
  uint32_t padding = 0;
  output_stream.write(reinterpret_cast<const char*>(&padding),
                      sizeof(uint32_t));

  // 2. Write header end offset
  uint64_t header_end_offset = 32 + flatbuffer_size;
  output_stream.write(reinterpret_cast<const char*>(&header_end_offset),
                      sizeof(uint64_t));

  // 3. Write flatbuffer
  output_stream.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                      flatbuffer_size);

  return output_stream.str();
}

TEST(CapabilitiesTest, HasSpeculativeDecodingSupport_WithDrafter_ReturnsTrue) {
  std::string file_data = CreateLiteRTLMBinaryString("tf_lite_mtp_drafter");
  std::istringstream stream(file_data, std::ios::binary);

  EXPECT_THAT(HasSpeculativeDecodingSupport(stream), IsOkAndHolds(true));
}

TEST(CapabilitiesTest, HasSpeculativeDecodingSupport_OtherModel_ReturnsFalse) {
  std::string file_data = CreateLiteRTLMBinaryString("base_model");
  std::istringstream stream(file_data, std::ios::binary);

  EXPECT_THAT(HasSpeculativeDecodingSupport(stream), IsOkAndHolds(false));
}

TEST(CapabilitiesTest,
     HasSpeculativeDecodingSupport_InvalidStream_ReturnsError) {
  std::istringstream stream("invalid_data");
  EXPECT_THAT(HasSpeculativeDecodingSupport(stream),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CapabilitiesTest, HasSpeculativeDecodingSupport_InvalidPath_ReturnsError) {
  EXPECT_THAT(HasSpeculativeDecodingSupport("/non/existent/path.litertlm"),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace litert::lm::schema::capabilities
