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

#include "schema/capabilities/capabilities_c.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm::schema::capabilities {
namespace {

std::string CreateTestLiteRTLMBinary(
    const std::vector<std::string>& tflite_model_types,
    const litert::lm::proto::LlmMetadata* llm_metadata_proto = nullptr) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<litert::lm::schema::SectionObject>>
      section_objects;
  uint64_t current_offset = 16384;

  std::string serialized_llm_proto;
  if (llm_metadata_proto != nullptr) {
    serialized_llm_proto = llm_metadata_proto->SerializeAsString();
    uint64_t proto_begin = current_offset;
    uint64_t proto_end = proto_begin + serialized_llm_proto.size();
    current_offset = proto_end;

    section_objects.push_back(CreateSectionObject(
        builder, 0, proto_begin, proto_end,
        litert::lm::schema::AnySectionDataType_LlmMetadataProto));
  }

  for (const auto& model_type : tflite_model_types) {
    auto kvp = CreateKeyValuePair(builder, "model_type", model_type);
    std::vector<flatbuffers::Offset<litert::lm::schema::KeyValuePair>> items = {
        kvp};
    uint64_t model_begin = current_offset;
    uint64_t model_end = model_begin + 100;
    current_offset = model_end;

    section_objects.push_back(CreateSectionObject(
        builder, builder.CreateVector(items), model_begin, model_end,
        litert::lm::schema::AnySectionDataType_TFLiteModel));
  }

  flatbuffers::Offset<litert::lm::schema::SectionMetadata> section_metadata = 0;
  if (!section_objects.empty()) {
    section_metadata = CreateSectionMetadata(
        builder, builder.CreateVector(section_objects));
  }

  auto root = CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(root);

  size_t flatbuffer_size = builder.GetSize();
  std::string header_data;
  header_data.append("LITERTLM", 8);
  uint32_t major_version = litert::lm::schema::LITERTLM_MAJOR_VERSION;
  uint32_t minor_version = litert::lm::schema::LITERTLM_MINOR_VERSION;
  uint32_t patch_version = litert::lm::schema::LITERTLM_PATCH_VERSION;
  header_data.append(reinterpret_cast<const char*>(&major_version),
                     sizeof(uint32_t));
  header_data.append(reinterpret_cast<const char*>(&minor_version),
                     sizeof(uint32_t));
  header_data.append(reinterpret_cast<const char*>(&patch_version),
                     sizeof(uint32_t));

  uint32_t padding = 0;
  header_data.append(reinterpret_cast<const char*>(&padding), sizeof(uint32_t));

  uint64_t header_end_offset = 32 + flatbuffer_size;
  header_data.append(reinterpret_cast<const char*>(&header_end_offset),
                     sizeof(uint64_t));
  header_data.append(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     flatbuffer_size);

  if (header_data.size() < 16384) {
    header_data.resize(16384, '\0');
  }
  header_data.append(serialized_llm_proto);
  return header_data;
}

TEST(CapabilitiesCTest, C_Api_ExtractsCapabilitiesCorrectly) {
  litert::lm::proto::LlmMetadata proto_meta;
  proto_meta.set_max_num_tokens(2048);
  proto_meta.add_jinja_prompt_template_args("tools");
  proto_meta.add_jinja_prompt_template_args("thought");
  proto_meta.add_supported_backends("cpu");
  proto_meta.add_supported_backends("gpu");
  proto_meta.add_supported_vision_resolutions(270);
  proto_meta.add_supported_vision_resolutions(560);
  proto_meta.mutable_sampler_params()->set_temperature(0.7f);
  proto_meta.mutable_sampler_params()->set_k(40);
  proto_meta.mutable_sampler_params()->set_p(0.95f);

  std::string file_data = CreateTestLiteRTLMBinary(
      {"tf_lite_vision_adapter", "tf_lite_audio_adapter",
       "tf_lite_mtp_drafter"},
      &proto_meta);

  // Write to temporary file
  std::string temp_file_path =
      testing::TempDir() + "/capabilities_c_test_model.litertlm";
  {
    std::ofstream out(temp_file_path, std::ios::binary);
    out.write(file_data.data(), file_data.size());
  }

  // Test C API
  LiteRtLmLoadedFile* loaded_file =
      litert_lm_loaded_file_create(temp_file_path.c_str());
  ASSERT_NE(loaded_file, nullptr);

  EXPECT_TRUE(
      litert_lm_loaded_file_has_speculative_decoding_support(loaded_file));
  EXPECT_TRUE(litert_lm_loaded_file_has_vision_support(loaded_file));
  EXPECT_TRUE(litert_lm_loaded_file_has_audio_support(loaded_file));
  EXPECT_TRUE(litert_lm_loaded_file_has_function_calling_support(loaded_file));
  EXPECT_TRUE(litert_lm_loaded_file_has_thinking_support(loaded_file));
  EXPECT_EQ(litert_lm_loaded_file_get_max_context_length(loaded_file), 2048);

  EXPECT_FLOAT_EQ(
      litert_lm_loaded_file_get_default_temperature(loaded_file), 0.7f);
  EXPECT_EQ(litert_lm_loaded_file_get_default_top_k(loaded_file), 40);
  EXPECT_FLOAT_EQ(
      litert_lm_loaded_file_get_default_top_p(loaded_file), 0.95f);

  EXPECT_EQ(
      litert_lm_loaded_file_get_supported_backend_count(loaded_file), 2);
  EXPECT_STREQ(
      litert_lm_loaded_file_get_supported_backend(loaded_file, 0), "cpu");
  EXPECT_STREQ(
      litert_lm_loaded_file_get_supported_backend(loaded_file, 1), "gpu");
  EXPECT_EQ(
      litert_lm_loaded_file_get_supported_backend(loaded_file, 2), nullptr);

  EXPECT_EQ(
      litert_lm_loaded_file_get_supported_vision_resolution_count(loaded_file),
      2);
  EXPECT_EQ(
      litert_lm_loaded_file_get_supported_vision_resolution(loaded_file, 0),
      270);
  EXPECT_EQ(
      litert_lm_loaded_file_get_supported_vision_resolution(loaded_file, 1),
      560);
  EXPECT_EQ(
      litert_lm_loaded_file_get_supported_vision_resolution(loaded_file, 2),
      -1);

  litert_lm_loaded_file_delete(loaded_file);
}

TEST(CapabilitiesCTest, C_Api_NullAndInvalidHandling) {
  EXPECT_FALSE(litert_lm_loaded_file_has_vision_support(nullptr));
  EXPECT_FALSE(litert_lm_loaded_file_has_audio_support(nullptr));
  EXPECT_FALSE(litert_lm_loaded_file_has_function_calling_support(nullptr));
  EXPECT_FALSE(litert_lm_loaded_file_has_thinking_support(nullptr));
  EXPECT_EQ(litert_lm_loaded_file_get_max_context_length(nullptr), -1);
  EXPECT_EQ(litert_lm_loaded_file_get_model_class(nullptr), nullptr);
  EXPECT_EQ(litert_lm_loaded_file_get_tf_hub_model_id(nullptr), nullptr);
  EXPECT_EQ(litert_lm_loaded_file_get_min_litertlm_version(nullptr), nullptr);
  EXPECT_FLOAT_EQ(litert_lm_loaded_file_get_default_temperature(nullptr), -1.0f);
  EXPECT_EQ(litert_lm_loaded_file_get_default_top_k(nullptr), -1);
  EXPECT_FLOAT_EQ(litert_lm_loaded_file_get_default_top_p(nullptr), -1.0f);
  EXPECT_EQ(litert_lm_loaded_file_get_supported_backend_count(nullptr), 0);
  EXPECT_EQ(litert_lm_loaded_file_get_supported_backend(nullptr, 0), nullptr);
  EXPECT_EQ(
      litert_lm_loaded_file_get_supported_vision_resolution_count(nullptr), 0);
  EXPECT_EQ(
      litert_lm_loaded_file_get_supported_vision_resolution(nullptr, 0), -1);
  EXPECT_EQ(litert_lm_loaded_file_create("/invalid/non_existent_path"),
            nullptr);
}

}  // namespace
}  // namespace litert::lm::schema::capabilities
