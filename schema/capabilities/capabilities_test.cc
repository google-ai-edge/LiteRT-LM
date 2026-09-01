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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert
#include "tflite/version.h"  // from @litert

namespace litert::lm::schema::capabilities {
namespace {

using ::absl_testing::StatusIs;

// Helper to create a minimal TFLite model in memory with signatures of
// specified token lengths.
std::string CreateMinimalTFLiteModel(const std::vector<int>& token_lengths) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<tflite::SignatureDef>> signature_defs;
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;

  // We need to create a subgraph with tensors for the signatures to point to.
  // We will create one output tensor for each signature.
  for (size_t i = 0; i < token_lengths.size(); ++i) {
    int length = token_lengths[i];
    std::vector<int32_t> shape_dims = {1, length, 2048};
    auto shape_offset = builder.CreateVector(shape_dims);

    auto tensor_offset = tflite::CreateTensor(
        builder, shape_offset, tflite::TensorType_FLOAT32, /*buffer=*/0,
        builder.CreateString(absl::StrCat("features_", i)));
    tensors.push_back(tensor_offset);
  }

  // Create subgraph with the tensors.
  auto subgraph_offset = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors), /*inputs=*/0, /*outputs=*/0);
  subgraphs.push_back(subgraph_offset);

  // Create signatures pointing to the tensors in the subgraph.
  for (size_t i = 0; i < token_lengths.size(); ++i) {
    std::vector<flatbuffers::Offset<tflite::TensorMap>> outputs;
    outputs.push_back(tflite::CreateTensorMap(
        builder, builder.CreateString("features"), /*tensor_index=*/i));

    auto signature_offset = tflite::CreateSignatureDef(
        builder, /*inputs=*/0, builder.CreateVector(outputs),
        builder.CreateString(absl::StrCat("vision_sig_", i)),
        /*subgraph_index=*/0);
    signature_defs.push_back(signature_offset);
  }

  // Create the model.
  auto model_offset = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, /*operator_codes=*/0,
      builder.CreateVector(subgraphs), /*description=*/0, /*buffers=*/0,
      /*metadata_buffer=*/0, /*metadata=*/0,
      builder.CreateVector(signature_defs));

  tflite::FinishModelBuffer(builder, model_offset);
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

struct TFLiteSectionConfig {
  std::string model_type;
  std::string backend_constraint = "";
  std::string payload = "";
};

std::string CreateTestLiteRTLMWithConfigs(
    const std::string& model_class, const std::string& tf_hub_model_id,
    const std::vector<TFLiteSectionConfig>& section_configs,
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

  std::vector<std::string> payloads;
  for (const auto& config : section_configs) {
    std::vector<flatbuffers::Offset<KeyValuePair>> items;
    items.push_back(
        CreateKeyValuePair(builder, "model_type", config.model_type));
    if (!config.backend_constraint.empty()) {
      items.push_back(CreateKeyValuePair(builder, "backend_constraint",
                                         config.backend_constraint));
    }

    std::string payload = config.payload;
    if (payload.empty()) {
      if (config.model_type == "tf_lite_vision_adapter" ||
          config.model_type == "tf_lite_vision_encoder") {
        payload = CreateMinimalTFLiteModel({280});
      } else {
        payload = std::string(100, '\0');
      }
    }
    payloads.push_back(payload);

    uint64_t model_begin = current_offset;
    uint64_t model_end = model_begin + payload.size();
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

  // Append payloads
  for (const auto& p : payloads) {
    header_data.append(p);
  }

  return header_data;
}

// Helper to create a LiteRT-LM file structure in memory with optional model
// payloads.
std::string CreateTestLiteRTLM(
    const std::string& model_class, const std::string& tf_hub_model_id,
    const std::vector<std::string>& tflite_model_types,
    const proto::LlmMetadata* llm_metadata_proto = nullptr,
    const std::vector<std::string>& model_payloads = {}) {
  std::vector<TFLiteSectionConfig> configs;
  for (size_t i = 0; i < tflite_model_types.size(); ++i) {
    TFLiteSectionConfig cfg;
    cfg.model_type = tflite_model_types[i];
    if (i < model_payloads.size()) {
      cfg.payload = model_payloads[i];
    }
    configs.push_back(cfg);
  }
  return CreateTestLiteRTLMWithConfigs(model_class, tf_hub_model_id, configs,
                                       llm_metadata_proto);
}

// Tests that the parser successfully extracts explicit capabilities and
// metadata when the LlmMetadata proto is fully populated.
TEST(CapabilitiesTest, InspectModel_ExtractsSystemMetadataAndLlmCapabilities) {
  proto::LlmMetadata proto_meta;
  proto_meta.set_supports_thinking(true);
  proto_meta.set_supports_function_calling(true);
  proto_meta.set_max_num_tokens(10007);  // Prime number
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
  EXPECT_EQ(llm.max_context_tokens, 10007);
  EXPECT_TRUE(llm.is_dynamic_context);

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
  proto_meta.set_max_num_tokens(2048);  // Non-prime number
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
  EXPECT_EQ(llm.max_context_tokens, 2048);
  EXPECT_FALSE(llm.is_dynamic_context);

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

TEST(CapabilitiesTest, InspectModel_ExtractsSupportedVisionTokenLengths) {
  // Generate a mock TFLite model with 3 vision signatures of sizes 1024, 64,
  // 256.
  std::string tflite_model = CreateMinimalTFLiteModel({1024, 64, 256});

  std::string file_data = CreateTestLiteRTLM(
      /*model_class=*/"", /*tf_hub_model_id=*/"", {"tf_lite_vision_encoder"},
      /*llm_metadata_proto=*/nullptr, {tflite_model});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  // Modality should be detected
  EXPECT_TRUE(llm.input_modalities.vision);

  // Lengths should be extracted, sorted, and unique
  ASSERT_TRUE(llm.vision_signature_selection.has_value());
  EXPECT_THAT(*llm.vision_signature_selection,
              ::testing::ElementsAre(64, 256, 1024));
}

TEST(CapabilitiesTest, InspectModel_NoVision_ReturnsNullopt) {
  // Text-only model (no vision section)
  std::string file_data = CreateTestLiteRTLM("IT", "google/gemma-3-1b-it",
                                             {"tf_lite_audio_adapter"});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_FALSE(llm.input_modalities.vision);
  // Should return null opt (not supported)
  EXPECT_FALSE(llm.vision_signature_selection.has_value());
}

TEST(CapabilitiesTest, InspectModel_VisionModelNoSignatures_ReturnsNullopt) {
  // Vision model without signature defs (e.g. legacy model or adapter).
  std::string tflite_model = CreateMinimalTFLiteModel({});

  std::string file_data = CreateTestLiteRTLM(
      /*model_class=*/"", /*tf_hub_model_id=*/"", {"tf_lite_vision_adapter"},
      /*llm_metadata_proto=*/nullptr, {tflite_model});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_TRUE(llm.input_modalities.vision);
  EXPECT_FALSE(llm.vision_signature_selection.has_value());
}

TEST(CapabilitiesTest, InspectModel_CorruptVisionModel_ReturnsError) {
  // We specify we have a vision adapter, but we pass invalid dummy payload
  // (corrupt).
  std::string file_data = CreateTestLiteRTLM(
      /*model_class=*/"", /*tf_hub_model_id=*/"", {"tf_lite_vision_adapter"},
      /*llm_metadata_proto=*/nullptr,
      {"corrupt_dummy_data_not_a_flatbuffer"});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  // Should fail-fast and return error status (Strict check)
  EXPECT_THAT(result_or, StatusIs(absl::StatusCode::kInternal));
}

TEST(CapabilitiesTest, InspectModel_NoBackendConstraints_DefaultsToCpuAndGpu) {
  std::string file_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {{"tf_lite_prefill_decode", "", "main_payload"}});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_TRUE(llm.text_supported_backends.cpu);
  EXPECT_TRUE(llm.text_supported_backends.gpu);
  EXPECT_FALSE(llm.text_supported_backends.npu);
}

TEST(CapabilitiesTest, InspectModel_CpuBackendConstraint_CpuOnly) {
  std::string file_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {{"tf_lite_prefill_decode", "cpu", "main_payload"}});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_TRUE(llm.text_supported_backends.cpu);
  EXPECT_FALSE(llm.text_supported_backends.gpu);
  EXPECT_FALSE(llm.text_supported_backends.npu);
}

TEST(CapabilitiesTest, InspectModel_GpuBackendConstraint_GpuOnly) {
  std::string file_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {{"tf_lite_prefill_decode", "gpu_artisan", "main_payload"}});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_FALSE(llm.text_supported_backends.cpu);
  EXPECT_TRUE(llm.text_supported_backends.gpu);
  EXPECT_FALSE(llm.text_supported_backends.npu);
}

TEST(CapabilitiesTest, InspectModel_GpuAndNpuBackendConstraints_GpuAndNpu) {
  std::string file_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {{"tf_lite_prefill_decode", "gpu, npu", "main_payload"}});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_FALSE(llm.text_supported_backends.cpu);
  EXPECT_TRUE(llm.text_supported_backends.gpu);
  EXPECT_TRUE(llm.text_supported_backends.npu);
}

TEST(CapabilitiesTest, InspectModel_AuxModelPresent_ForcesNpu) {
  std::string file_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {{"tf_lite_prefill_decode", "cpu", "main_payload"},
       {"tf_lite_aux", "", "aux_payload"}});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_TRUE(llm.text_supported_backends.cpu);
  EXPECT_FALSE(llm.text_supported_backends.gpu);
  EXPECT_TRUE(llm.text_supported_backends.npu);
}

TEST(CapabilitiesTest, InspectModel_MinRuntimeVersionExposed) {
  proto::LlmMetadata llm_metadata;
  llm_metadata.set_min_runtime_version("0.12.3");

  std::string file_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {{"tf_lite_prefill_decode", "", "main_payload"}},
      &llm_metadata);

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_EQ(llm.min_runtime_version, "0.12.3");
}

TEST(CapabilitiesTest, InspectModel_ArtisanModelType_GpuOnly) {
  std::string file_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {{"tf_lite_artisan_text_decoder", "gpu_artisan", "main_payload"}});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_FALSE(llm.text_supported_backends.cpu);
  EXPECT_TRUE(llm.text_supported_backends.gpu);
  EXPECT_FALSE(llm.text_supported_backends.npu);
}

TEST(CapabilitiesTest, InspectModel_ArtisanModelType_NpuOnly) {
  std::string file_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {{"tf_lite_artisan_text_decoder", "google_tensor_artisan",
        "main_payload"}});

  std::istringstream stream(file_data, std::ios::binary);
  auto result_or = InspectModel(stream);
  ASSERT_OK(result_or);
  ModelCapabilities result = std::move(*result_or);
  ASSERT_TRUE(result.llm_capability.has_value());
  const auto& llm = *result.llm_capability;

  EXPECT_FALSE(llm.text_supported_backends.cpu);
  EXPECT_FALSE(llm.text_supported_backends.gpu);
  EXPECT_TRUE(llm.text_supported_backends.npu);
  EXPECT_EQ(llm.text_supported_backends.npu_brand, NpuBrand::kGoogleTensor);
}

std::string CreateMockNpuTfliteModel(const std::string& dispatch_name) {
  flatbuffers::FlatBufferBuilder fbb;

  flexbuffers::Builder flex_builder;
  flex_builder.Map([&]() {
    flex_builder.String("name", dispatch_name);
    flex_builder.Int("bytecode_offset", 0);
    flex_builder.Int("bytecode_size", 0);
  });
  flex_builder.Finish();
  auto opt_vec = fbb.CreateVector(flex_builder.GetBuffer());

  auto custom_code = fbb.CreateString("DISPATCH_OP");
  auto op_code = tflite::CreateOperatorCode(
      fbb, tflite::BuiltinOperator_CUSTOM, custom_code);
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> op_codes = {op_code};
  auto op_codes_vec = fbb.CreateVector(op_codes);

  std::vector<int32_t> inputs = {};
  std::vector<int32_t> outputs = {};
  auto op = tflite::CreateOperator(
      fbb, 0, fbb.CreateVector(inputs), fbb.CreateVector(outputs),
      tflite::BuiltinOptions_NONE, 0, opt_vec,
      tflite::CustomOptionsFormat_FLEXBUFFERS);
  std::vector<flatbuffers::Offset<tflite::Operator>> ops = {op};

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {};
  auto subgraph = tflite::CreateSubGraph(
      fbb, fbb.CreateVector(tensors), fbb.CreateVector(inputs),
      fbb.CreateVector(outputs), fbb.CreateVector(ops));
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {subgraph};
  auto subgraphs_vec = fbb.CreateVector(subgraphs);

  auto model = tflite::CreateModel(fbb, 3, op_codes_vec, subgraphs_vec);
  tflite::FinishModelBuffer(fbb, model);

  const uint8_t* buf = fbb.GetBufferPointer();
  size_t size = fbb.GetSize();
  return std::string(reinterpret_cast<const char*>(buf), size);
}

TEST(CapabilitiesTest, DetectsNpuBrandFromAuxModel) {
  // Test Qualcomm
  {
    std::string aux_payload = CreateMockNpuTfliteModel("qnn_partition_0");
    std::string litertlm_data = CreateTestLiteRTLMWithConfigs(
        "gemma", "gemma3",
        {
            {.model_type = "tf_lite_prefill_decode",
             .backend_constraint = "cpu,gpu"},
            {.model_type = "tf_lite_aux", .payload = aux_payload}
        });
    std::stringstream stream(litertlm_data);
    auto cap_or = InspectModel(stream);
    ASSERT_OK(cap_or.status());
    EXPECT_TRUE(cap_or->llm_capability->text_supported_backends.npu);
    EXPECT_EQ(cap_or->llm_capability->text_supported_backends.npu_brand,
              NpuBrand::kQualcomm);
  }

  // Test Google Tensor
  {
    std::string aux_payload = CreateMockNpuTfliteModel("subgraph_0_fn");
    std::string litertlm_data = CreateTestLiteRTLMWithConfigs(
        "gemma", "gemma3",
        {
            {.model_type = "tf_lite_prefill_decode",
             .backend_constraint = "cpu,gpu"},
            {.model_type = "tf_lite_aux", .payload = aux_payload}
        });
    std::stringstream stream(litertlm_data);
    auto cap_or = InspectModel(stream);
    ASSERT_OK(cap_or.status());
    EXPECT_TRUE(cap_or->llm_capability->text_supported_backends.npu);
    EXPECT_EQ(cap_or->llm_capability->text_supported_backends.npu_brand,
              NpuBrand::kGoogleTensor);
  }

  // Test MediaTek
  {
    std::string aux_payload = CreateMockNpuTfliteModel("Partition_0");
    std::string litertlm_data = CreateTestLiteRTLMWithConfigs(
        "gemma", "gemma3",
        {
            {.model_type = "tf_lite_prefill_decode",
             .backend_constraint = "cpu,gpu"},
            {.model_type = "tf_lite_aux", .payload = aux_payload}
        });
    std::stringstream stream(litertlm_data);
    auto cap_or = InspectModel(stream);
    ASSERT_OK(cap_or.status());
    EXPECT_TRUE(cap_or->llm_capability->text_supported_backends.npu);
    EXPECT_EQ(cap_or->llm_capability->text_supported_backends.npu_brand,
              NpuBrand::kMediaTek);
  }
}

TEST(CapabilitiesTest, InspectModel_ModalitySpecificSupportedBackends) {
  std::string aux_payload = CreateMockNpuTfliteModel("qnn_partition_0");
  std::string litertlm_data = CreateTestLiteRTLMWithConfigs(
      /*model_class=*/"", /*tf_hub_model_id=*/"",
      {
          {.model_type = "tf_lite_prefill_decode", .backend_constraint = "cpu"},
          {.model_type = "tf_lite_vision_adapter", .backend_constraint = "gpu"},
          {.model_type = "tf_lite_audio_adapter", .backend_constraint = "gpu"},
          {.model_type = "tf_lite_aux", .payload = aux_payload}
      });
  std::stringstream stream(litertlm_data);
  auto cap_or = InspectModel(stream);
  ASSERT_OK(cap_or.status());
  const auto& llm = *cap_or->llm_capability;

  // Text
  EXPECT_TRUE(llm.text_supported_backends.cpu);
  EXPECT_FALSE(llm.text_supported_backends.gpu);
  EXPECT_TRUE(llm.text_supported_backends.npu);
  EXPECT_EQ(llm.text_supported_backends.npu_brand, NpuBrand::kQualcomm);

  // Vision
  EXPECT_FALSE(llm.vision_supported_backends.cpu);
  EXPECT_TRUE(llm.vision_supported_backends.gpu);
  EXPECT_FALSE(llm.vision_supported_backends.npu);
  EXPECT_EQ(llm.vision_supported_backends.npu_brand, NpuBrand::kUnknown);

  // Audio
  EXPECT_FALSE(llm.audio_supported_backends.cpu);
  EXPECT_TRUE(llm.audio_supported_backends.gpu);
  EXPECT_FALSE(llm.audio_supported_backends.npu);
  EXPECT_EQ(llm.audio_supported_backends.npu_brand, NpuBrand::kUnknown);

  // Video (not present)
  EXPECT_FALSE(llm.video_supported_backends.cpu);
  EXPECT_FALSE(llm.video_supported_backends.gpu);
  EXPECT_FALSE(llm.video_supported_backends.npu);
}

}  // namespace
}  // namespace litert::lm::schema::capabilities
