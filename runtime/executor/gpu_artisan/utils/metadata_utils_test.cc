// Copyright 2024 The ODML Authors.
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

#include "runtime/executor/gpu_artisan/utils/metadata_utils.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT
#include "tflite/model_builder.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert

namespace odml::infra::llm_utils {
namespace {

using ::testing::status::StatusIs;

std::string BuildSyntheticTfliteModel(
    const std::vector<std::pair<std::string, uint32_t>>& metadata_entries) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));

  flatbuffers::Offset<
      flatbuffers::Vector<flatbuffers::Offset<tflite::Metadata>>>
      metadata_vec = 0;
  if (!metadata_entries.empty()) {
    std::vector<flatbuffers::Offset<tflite::Metadata>> metadata_offsets;
    metadata_offsets.reserve(metadata_entries.size());
    for (const auto& [name, buffer_idx] : metadata_entries) {
      metadata_offsets.push_back(
          tflite::CreateMetadataDirect(builder, name.c_str(), buffer_idx));
    }
    metadata_vec = builder.CreateVector(metadata_offsets);
  }

  auto buffers_vec = builder.CreateVector(buffers);
  auto description = builder.CreateString("synthetic_test_model");
  auto model = tflite::CreateModel(builder, /*version=*/3, /*operator_codes=*/0,
                                   /*subgraphs=*/0, description, buffers_vec,
                                   /*metadata_buffer=*/0, metadata_vec);
  tflite::FinishModelBuffer(builder, model);
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

TEST(MetadataUtilsTest, FailedGetLlmModelTypeFromEmptyMetadata) {
  std::string model_bytes = BuildSyntheticTfliteModel({});
  std::unique_ptr<tflite::FlatBufferModel> model_ptr =
      tflite::FlatBufferModel::BuildFromBuffer(model_bytes.data(),
                                               model_bytes.size());
  ASSERT_NE(model_ptr, nullptr);
  auto model_type = GetLlmModelType(*model_ptr);
  EXPECT_THAT(model_type, StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(model_type.status().message(),
            absl::StrCat("Failed to get LLM model type, missing ",
                         llm_utils::kLlmModelTypeName, " in tflite metadata"));
}

TEST(MetadataUtilsTest, FailedGetLlmModelTypeWhenUnrelatedMetadataPresent) {
  std::string model_bytes = BuildSyntheticTfliteModel({{"unrelated_key", 0}});
  std::unique_ptr<tflite::FlatBufferModel> model_ptr =
      tflite::FlatBufferModel::BuildFromBuffer(model_bytes.data(),
                                               model_bytes.size());
  ASSERT_NE(model_ptr, nullptr);
  EXPECT_THAT(GetLlmModelType(*model_ptr),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(MetadataUtilsTest, SucceededGetLlmModelType) {
  std::string model_bytes = BuildSyntheticTfliteModel(
      {{"unrelated_key", 0},
       {std::string(kLlmModelTypeName),
        static_cast<uint32_t>(proto::LLM_MODEL_TYPE_GEMMA_2B)}});
  std::unique_ptr<tflite::FlatBufferModel> model_ptr =
      tflite::FlatBufferModel::BuildFromBuffer(model_bytes.data(),
                                               model_bytes.size());
  ASSERT_NE(model_ptr, nullptr);
  ASSERT_OK_AND_ASSIGN(auto model_type, GetLlmModelType(*model_ptr));
  EXPECT_EQ(model_type, proto::LLM_MODEL_TYPE_GEMMA_2B);
}

TEST(MetadataUtilsTest, RequireBytesToUnicodeMapping) {
  EXPECT_TRUE(
      RequireBytesToUnicodeMapping(proto::LLM_MODEL_TYPE_STABLELM_4E1T_3B));
  EXPECT_TRUE(RequireBytesToUnicodeMapping(proto::LLM_MODEL_TYPE_FALCON_RW_1B));
  EXPECT_TRUE(RequireBytesToUnicodeMapping(proto::LLM_MODEL_TYPE_PHI_2));
  EXPECT_FALSE(RequireBytesToUnicodeMapping(proto::LLM_MODEL_TYPE_GEMMA_2B));
  EXPECT_FALSE(RequireBytesToUnicodeMapping(proto::LLM_MODEL_TYPE_GEMMA3_4B));
}

TEST(MetadataUtilsTest, RequireFp32ModelFalse) {
  constexpr std::array<proto::LlmModelType, 5> kNonFp32Models = {
      proto::LLM_MODEL_TYPE_GEMMA_2B,         proto::LLM_MODEL_TYPE_GEMMA2_2B,
      proto::LLM_MODEL_TYPE_GEMMA3_1B,        proto::LLM_MODEL_TYPE_GEMMA3_4B,
      proto::LLM_MODEL_TYPE_STABLELM_4E1T_3B,
  };
  for (const auto& model_type : kNonFp32Models) {
    EXPECT_FALSE(RequireFp32Model(model_type));
  }
}

TEST(MetadataUtilsTest, RequireFp32ModelTrue) {
  constexpr std::array<proto::LlmModelType, 2> kFp32Models = {
      proto::LLM_MODEL_TYPE_PHI_2,
      proto::LLM_MODEL_TYPE_FALCON_RW_1B,
  };
  for (const auto& model_type : kFp32Models) {
    EXPECT_TRUE(RequireFp32Model(model_type));
  }
}

}  // namespace
}  // namespace odml::infra::llm_utils
