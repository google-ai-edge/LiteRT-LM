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
#include <optional>
#include <sstream>
#include <string>
#include <utility>
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
#include "support/util/test_utils.h"  // IWYU pragma: keep




namespace litert::lm::schema::capabilities {
namespace {

using ::absl_testing::StatusIs;

// Helper to create a complete LiteRT-LM file structure in memory with specified
// metadata and sections. This allows testing the capabilities parser without
// having to read large actual model files from disk.
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
    system_metadata =
        CreateSystemMetadata(builder, builder.CreateVector(system_entries));
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

    section_objects.push_back(
        CreateSectionObject(builder, 0, proto_begin, proto_end,
                            AnySectionDataType_LlmMetadataProto));
  }

  for (const auto& model_type : tflite_model_types) {
    auto kvp = CreateKeyValuePair(builder, "model_type", model_type);
    std::vector<flatbuffers::Offset<KeyValuePair>> items = {kvp};
    uint64_t model_begin = current_offset;
    uint64_t model_end = model_begin + 100;
    current_offset = model_end;

    section_objects.push_back(
        CreateSectionObject(builder, builder.CreateVector(items), model_begin,
                            model_end, AnySectionDataType_TFLiteModel));
  }

  flatbuffers::Offset<SectionMetadata> section_metadata = 0;
  if (!section_objects.empty()) {
    section_metadata =
        CreateSectionMetadata(builder, builder.CreateVector(section_objects));
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

// Tests that the parser successfully extracts explicit capabilities and
// metadata when the LlmMetadata proto is fully populated.
TEST(CapabilitiesTest, InspectModel_ExtractsSystemMetadataAndLlmCapabilities) {
  proto::LlmMetadata proto_meta;
  proto_meta.set_supports_thinking(true);
  proto_meta.set_supports_function_calling(true);
  auto* sp = proto_meta.mutable_sampler_params();
  sp->set_type(proto::SamplerParameters::TOP_P);
  sp->set_k(10);
  sp->set_p(0.95f);
  sp->set_temperature(0.7f);

  std::string file_data =
      CreateTestLiteRTLM("IT", "google/gemma-3-1b-it",
                         {"tf_lite_vision_adapter", "tf_lite_mtp_drafter",
                          "tf_lite_video_encoder"},
                         &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.supports_function_calling, true);
  EXPECT_EQ(llm.supports_thinking, true);
  EXPECT_TRUE(llm.supports_speculative_decoding);

  // Check default sampler parameters
  EXPECT_EQ(llm.default_sampler_params.type, SamplerType::kTopP);
  EXPECT_EQ(llm.default_sampler_params.k, 10);
  EXPECT_FLOAT_EQ(llm.default_sampler_params.p, 0.95f);
  EXPECT_FLOAT_EQ(llm.default_sampler_params.temperature, 0.7f);

  // Check modalities (Text, Vision, Video)
  EXPECT_TRUE(llm.input_modalities.text);
  EXPECT_TRUE(llm.input_modalities.vision);
  EXPECT_FALSE(llm.input_modalities.audio);
  EXPECT_TRUE(llm.input_modalities.video);

  EXPECT_TRUE(llm.output_modalities.text);
  EXPECT_FALSE(llm.output_modalities.vision);
  EXPECT_FALSE(llm.output_modalities.audio);
  EXPECT_FALSE(llm.output_modalities.video);
}

TEST(CapabilitiesTest, InspectModel_ExtractsMaxVisionTokenBudget) {
  proto::LlmMetadata proto_meta;
  auto* model_type = proto_meta.mutable_llm_model_type();
  auto* gemma4 = model_type->mutable_gemma4();
  gemma4->set_max_num_patches(2520);
  gemma4->set_pooling_kernel_size(3);

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/gemma-4-2b-it", {"tf_lite_vision_adapter"}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.max_vision_token_budget, 280);
}

TEST(CapabilitiesTest, InspectModel_ExtractsMaxVisionTokenBudget_GenericModel) {
  proto::LlmMetadata proto_meta;
  auto* model_type = proto_meta.mutable_llm_model_type();
  auto* generic = model_type->mutable_generic_model();
  generic->set_max_num_patches(100);
  generic->set_pooling_kernel_size(2);

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/generic-it", {"tf_lite_vision_adapter"}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.max_vision_token_budget, 25);
}

TEST(CapabilitiesTest,
     InspectModel_ExtractsMaxVisionTokenBudget_GenericModel_DefaultPooling) {
  proto::LlmMetadata proto_meta;
  auto* model_type = proto_meta.mutable_llm_model_type();
  auto* generic = model_type->mutable_generic_model();
  generic->set_max_num_patches(100);

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/generic-it", {"tf_lite_vision_adapter"}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.max_vision_token_budget, 100);  // Defaults to 1
}


TEST(CapabilitiesTest, InspectModel_ExtractsMaxVisionTokenBudget_Lfm2) {
  proto::LlmMetadata proto_meta;
  auto* model_type = proto_meta.mutable_llm_model_type();
  auto* lfm2 = model_type->mutable_lfm2();
  lfm2->set_max_num_patches(180);
  lfm2->set_pooling_kernel_size(3);

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/lfm2-it", {"tf_lite_vision_adapter"}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.max_vision_token_budget, 20);
}

TEST(CapabilitiesTest,
     InspectModel_ExtractsMaxVisionTokenBudget_Lfm2_DefaultPooling) {
  proto::LlmMetadata proto_meta;
  auto* model_type = proto_meta.mutable_llm_model_type();
  auto* lfm2 = model_type->mutable_lfm2();
  lfm2->set_max_num_patches(180);

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/lfm2-it", {"tf_lite_vision_adapter"}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.max_vision_token_budget, 45);  // Defaults to 2
}

TEST(CapabilitiesTest,
     InspectModel_ExtractsMaxVisionTokenBudget_Gemma4_DefaultPooling) {
  proto::LlmMetadata proto_meta;
  auto* model_type = proto_meta.mutable_llm_model_type();
  auto* gemma4 = model_type->mutable_gemma4();
  gemma4->set_max_num_patches(2520);

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/gemma-4-2b-it", {"tf_lite_vision_adapter"}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.max_vision_token_budget, 280);  // Defaults to 3
}

// Tests that unconfigured capability flags return std::nullopt for
// older models.
TEST(CapabilitiesTest, InspectModel_NoExplicitCapabilities_ReturnsFalse) {
  proto::LlmMetadata proto_meta;
  // Do not set explicit supports_thinking/supports_function_calling

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/gemma-3-1b-it", {"tf_lite_audio_adapter"}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.supports_thinking, false);
  EXPECT_EQ(llm.supports_function_calling, false);

  EXPECT_TRUE(llm.input_modalities.text);
  EXPECT_TRUE(llm.input_modalities.audio);
  EXPECT_FALSE(llm.input_modalities.vision);
  EXPECT_FALSE(llm.input_modalities.video);

  // Check default fallback sampler parameters (proto defaults)
  EXPECT_EQ(llm.default_sampler_params.type, SamplerType::kUnspecified);
  EXPECT_EQ(llm.default_sampler_params.k, 0);
  EXPECT_FLOAT_EQ(llm.default_sampler_params.p, 0.0f);
  EXPECT_FLOAT_EQ(llm.default_sampler_params.temperature, 0.0f);
}

// Tests that standard defaults and FlatBuffer scanner fallbacks are filled
// correctly when no LlmMetadata proto is packed in the container file.
TEST(CapabilitiesTest, InspectModel_NoLlmProto_FillsDefaultsAndScanning) {
  std::string file_data =
      CreateTestLiteRTLM("IT", "google/gemma-3-1b-it",
                         {"tf_lite_vision_adapter", "tf_lite_mtp_drafter"},
                         /*llm_metadata_proto=*/nullptr);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_TRUE(llm.input_modalities.text);
  EXPECT_TRUE(llm.input_modalities.vision);
  EXPECT_FALSE(llm.input_modalities.audio);
  EXPECT_FALSE(llm.input_modalities.video);
  EXPECT_TRUE(llm.supports_speculative_decoding);

  EXPECT_EQ(llm.supports_thinking, false);
  EXPECT_EQ(llm.supports_function_calling, false);

  // Check default fallback sampler parameters (proto defaults)
  EXPECT_EQ(llm.default_sampler_params.type, SamplerType::kUnspecified);
  EXPECT_EQ(llm.default_sampler_params.k, 0);
  EXPECT_FLOAT_EQ(llm.default_sampler_params.p, 0.0f);
  EXPECT_FLOAT_EQ(llm.default_sampler_params.temperature, 0.0f);
}

// Tests that explicit false capability values in the proto are parsed as false.
TEST(CapabilitiesTest, InspectModel_ExplicitFalseCapabilities_ReturnsFalse) {
  proto::LlmMetadata proto_meta;
  proto_meta.set_supports_thinking(false);
  proto_meta.set_supports_function_calling(false);

  std::string file_data = CreateTestLiteRTLM(
      "IT", "google/gemma-3-1b-it", {}, &proto_meta);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;
  EXPECT_EQ(llm.supports_thinking, false);
  EXPECT_EQ(llm.supports_function_calling, false);
}

// Tests that the parser returns an invalid argument status if the input
// stream is corrupted.
TEST(CapabilitiesTest, InspectModel_InvalidStream_ReturnsError) {
  std::istringstream stream("invalid_data");
  EXPECT_THAT(InspectModel(stream),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm::schema::capabilities
