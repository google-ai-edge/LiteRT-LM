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

#include "runtime/components/model_resources_streaming.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

class MemoryDataStream : public DataStream {
 public:
  explicit MemoryDataStream(std::string data) : data_(std::move(data)) {}

  absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                              uint64_t size) override {
    return ReadAndPreserve(buffer, offset, size);
  }

  absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                               uint64_t size) override {
    if (offset + size > data_.size()) {
      return absl::OutOfRangeError("Read beyond end of stream");
    }
    std::memcpy(buffer, data_.data() + offset, size);
    return absl::OkStatus();
  }

  absl::Status Discard(uint64_t offset, uint64_t size) override {
    return absl::OkStatus();
  }

 private:
  std::string data_;
};

TEST(ModelResourcesStreamingTest, GetTFLiteModelNotFoundWhenUnset) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetTFLiteModel(ModelType::kUnknown).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(ModelResourcesStreamingTest, SetAndGetModelBuffer) {
  ModelResourcesStreaming model_resources;
  std::string dummy_bytes = "dummy_tflite_model_content";
  model_resources.SetModelBuffer(
      ModelType::kTfLitePrefillDecode,
      std::vector<char>(dummy_bytes.begin(), dummy_bytes.end()));

  ASSERT_OK_AND_ASSIGN(auto buffer, model_resources.GetTFLiteModelBuffer(
                                        ModelType::kTfLitePrefillDecode));
  EXPECT_EQ(buffer, dummy_bytes);
}

TEST(ModelResourcesStreamingTest, GetScopedFileUnimplementedWhenUnset) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetScopedFile().status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST(ModelResourcesStreamingTest,
     GetWeightsSectionOffsetUnimplementedWhenUnset) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(
      model_resources.GetWeightsSectionOffset(ModelType::kUnknown)
          .status()
          .code(),
      absl::StatusCode::kUnimplemented);
}

TEST(ModelResourcesStreamingTest, GetTFLiteModelBackendConstraint) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(
      model_resources.GetTFLiteModelBackendConstraint(ModelType::kUnknown),
      std::nullopt);
}

TEST(ModelResourcesStreamingTest, GetTokenizerUnimplemented) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetTokenizer().status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST(ModelResourcesStreamingTest, SetAndGetLlmMetadata) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetLlmMetadata().status().code(),
            absl::StatusCode::kNotFound);

  proto::LlmMetadata metadata;
  metadata.set_max_num_tokens(2048);
  model_resources.SetLlmMetadata(metadata);

  ASSERT_OK_AND_ASSIGN(const auto* retrieved_metadata,
                       model_resources.GetLlmMetadata());
  EXPECT_EQ(retrieved_metadata->max_num_tokens(), 2048);
}

TEST(ModelResourcesStreamingTest, SetAndGetEmbeddingMetadata) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetEmbeddingMetadata().status().code(),
            absl::StatusCode::kNotFound);

  proto::EmbeddingMetadata metadata;
  metadata.mutable_bos_token()->set_token_str("<bos>");
  model_resources.SetEmbeddingMetadata(metadata);

  ASSERT_OK_AND_ASSIGN(const auto* retrieved_metadata,
                       model_resources.GetEmbeddingMetadata());
  EXPECT_EQ(retrieved_metadata->bos_token().token_str(), "<bos>");
}

TEST(ModelResourcesStreamingTest, SetAndGetWeightsFromStream) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(
      model_resources.GetWeightInMemoryMap(ModelType::kTfLiteVisionEncoder),
      nullptr);

  std::string weight_data = "raw_weights_bytes_12345678";
  MemoryDataStream stream(weight_data);

  ASSERT_TRUE(model_resources
                  .SetWeightsFromStream(ModelType::kTfLiteVisionEncoder, stream,
                                        weight_data.size())
                  .ok());

  const auto* model_map =
      model_resources.GetWeightInMemoryMap(ModelType::kTfLiteVisionEncoder);
  ASSERT_NE(model_map, nullptr);
  auto model_it = model_map->find("tflite_weights");
  ASSERT_NE(model_it, model_map->end());
  EXPECT_EQ(model_it->second.size(), weight_data.size());

  // Non-streamed model type should return nullptr.
  EXPECT_EQ(model_resources.GetWeightInMemoryMap(ModelType::kTfLiteEmbedder),
            nullptr);
}

}  // namespace
}  // namespace litert::lm
