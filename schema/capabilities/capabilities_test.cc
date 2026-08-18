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

#include "schema/capabilities/capabilities.h"

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
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm::schema::capabilities {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

// Helper to create a complete LiteRT-LM file in memory with specified metadata.
std::string CreateTestLiteRTLM(
    const std::string& model_class, const std::string& tf_hub_model_id,
    const std::vector<std::string>& tflite_model_types,
    const proto::LlmMetadata* llm_metadata_proto = nullptr) {
  flatbuffers::FlatBufferBuilder builder;

  // 1. System Metadata
  std::vector<flatbuffers::Offset<KeyValuePair>> system_entries;
  if (!model_class.empty()) {
    system_entries.push_back(
        CreateKeyValuePair(builder, "model_class", model_class));
  }
  if (!tf_hub_model_id.empty()) {
    system_entries.push_back(
        CreateKeyValuePair(builder, "tf_hub_model_id", tf_hub_model_id));
  }
  flatbuffers::Offset<SystemMetadata> system_metadata = 0;
  if (!system_entries.empty()) {
    system_metadata = CreateSystemMetadata(
        builder, builder.CreateVector(system_entries));
  }

  // 2. Sections
  std::vector<flatbuffers::Offset<SectionObject>> section_objects;
  uint64_t current_offset = 16384;  // Simulated block boundary

  std::string serialized_llm_proto;
  if (llm_metadata_proto != nullptr) {
    serialized_llm_proto = llm_metadata_proto->SerializeAsString();
    uint64_t proto_begin = current_offset;
    uint64_t proto_end = proto_begin + serialized_llm_proto.size();
    current_offset = proto_end;

    section_objects.push_back(CreateSectionObject(
        builder, 0, proto_begin, proto_end,
        AnySectionDataType_LlmMetadataProto));
  }

  for (const auto& model_type : tflite_model_types) {
    auto kvp = CreateKeyValuePair(builder, "model_type", model_type);
    std::vector<flatbuffers::Offset<KeyValuePair>> items = {kvp};
    uint64_t model_begin = current_offset;
    uint64_t model_end = model_begin + 100;
    current_offset = model_end;

    section_objects.push_back(CreateSectionObject(
        builder, builder.CreateVector(items), model_begin, model_end,
        AnySectionDataType_TFLiteModel));
  }

  flatbuffers::Offset<SectionMetadata> section_metadata = 0;
  if (!section_objects.empty()) {
    section_metadata = CreateSectionMetadata(
        builder, builder.CreateVector(section_objects));
  }

  auto root =
      CreateLiteRTLMMetaData(builder, system_metadata, section_metadata);
  builder.Finish(root);

  size_t flatbuffer_size = builder.GetSize();
  std::ostringstream output_stream(std::ios::binary);

  // Magic & versions
  output_stream.write("LITERTLM", 8);
  output_stream.write(reinterpret_cast<const char*>(&LITERTLM_MAJOR_VERSION),
                      sizeof(uint32_t));
  output_stream.write(reinterpret_cast<const char*>(&LITERTLM_MINOR_VERSION),
                      sizeof(uint32_t));
  output_stream.write(reinterpret_cast<const char*>(&LITERTLM_PATCH_VERSION),
                      sizeof(uint32_t));

  // Padding
  uint32_t padding = 0;
  output_stream.write(reinterpret_cast<const char*>(&padding),
                      sizeof(uint32_t));

  // Header end offset
  uint64_t header_end_offset = 32 + flatbuffer_size;
  output_stream.write(reinterpret_cast<const char*>(&header_end_offset),
                      sizeof(uint64_t));

  // Flatbuffer
  output_stream.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                      flatbuffer_size);

  // Fill up to 16384 with padding if needed
  std::string header_data = output_stream.str();
  if (header_data.size() < 16384) {
    header_data.resize(16384, '\0');
  }

  // Append serialized proto
  header_data.append(serialized_llm_proto);

  return header_data;
}

TEST(CapabilitiesTest, InspectModel_ExtractsSystemMetadataAndLlmCapabilities) {
  proto::LlmMetadata proto_meta;
  proto_meta.set_max_num_tokens(4096);
  proto_meta.add_jinja_prompt_template_args("tools");
  proto_meta.add_jinja_prompt_template_args("thought");
  proto_meta.add_supported_backends("cpu");
  proto_meta.add_supported_backends("gpu");
  proto_meta.add_supported_vision_resolutions(270);
  proto_meta.add_supported_vision_resolutions(560);

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/gemma-3-1b-it",
      {"tf_lite_vision_adapter", "tf_lite_mtp_drafter"}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result = InspectModel(stream);
  ASSERT_OK(result);

  EXPECT_EQ(result->model_class, "IT");
  EXPECT_EQ(result->tf_hub_model_id, "google/gemma-3-1b-it");
  ASSERT_TRUE(result->llm_capability.has_value());

  const auto& llm = *result->llm_capability;
  EXPECT_EQ(llm.max_context_length, 4096);
  EXPECT_TRUE(llm.supports_function_calling);
  EXPECT_TRUE(llm.supports_thinking);
  EXPECT_TRUE(llm.supports_speculative_decoding);

  // Check modalities
  EXPECT_THAT(
      llm.input_modalities,
      testing::UnorderedElementsAre(Modality::kText, Modality::kVision));
  EXPECT_THAT(llm.output_modalities, testing::ElementsAre(Modality::kText));

  // Check backends and resolutions
  EXPECT_THAT(llm.supported_backends, testing::ElementsAre("cpu", "gpu"));
  EXPECT_THAT(llm.supported_vision_resolutions, testing::ElementsAre(270, 560));
}

TEST(CapabilitiesTest, InspectModel_InvalidStream_ReturnsError) {
  std::istringstream stream("invalid_data");
  EXPECT_THAT(InspectModel(stream),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm::schema::capabilities
