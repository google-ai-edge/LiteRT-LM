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

#include "runtime/executor/gpu_artisan/utils/model_data.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/transformer_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/metadata_utils.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/model_builder.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert

namespace odml::infra::llm_utils {
namespace {

using ::litert::lm::MemoryMappedFile;
using ::litert::lm::ScopedFile;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

struct SyntheticModelOptions {
  bool include_llm_params = true;
  std::optional<proto::LlmModelType> model_type =
      proto::LLM_MODEL_TYPE_GEMMA_2B;
  std::optional<uint32_t> lora_rank = std::nullopt;
  std::optional<std::string> backend = "gpu";
  std::string weight_payload = "weight_bytes_1234";
};

// Builds a TFLite FlatBuffer with external-offset buffers appended at the end
// of the returned byte string, matching the on-disk Artisan TFLite layout.
std::string BuildSyntheticArtisanModel(const SyntheticModelOptions& opts = {}) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<std::string> payloads;
  // Buffer 0 is always the empty sentinel buffer in TFLite.
  payloads.emplace_back("");

  std::vector<std::pair<std::string, uint32_t>> metadata_entries;
  if (opts.include_llm_params) {
    proto::LlmParameters params;
    params.set_vocab_size(256000);
    params.set_start_token_id(2);
    params.mutable_transformer_parameters()->set_batch_size(1);
    uint32_t idx = payloads.size();
    payloads.push_back(params.SerializeAsString());
    metadata_entries.emplace_back("odml.infra.proto.LlmParameters", idx);
  }
  if (opts.backend.has_value()) {
    uint32_t idx = payloads.size();
    payloads.push_back(*opts.backend);
    metadata_entries.emplace_back(std::string(kLlmBackendName), idx);
  }
  if (opts.model_type.has_value()) {
    metadata_entries.emplace_back(std::string(kLlmModelTypeName),
                                  static_cast<uint32_t>(*opts.model_type));
  }
  if (opts.lora_rank.has_value()) {
    metadata_entries.emplace_back(std::string(kLoRARank), *opts.lora_rank);
  }

  uint32_t weight_buf_idx = payloads.size();
  payloads.push_back(opts.weight_payload);

  // Two-pass construction so buffer offsets point past the FlatBuffer header.
  auto build_with_base = [&](uint64_t base_offset) -> std::string {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    uint64_t cursor = base_offset;
    for (size_t i = 0; i < payloads.size(); ++i) {
      if (i == 0) {
        buffers.push_back(tflite::CreateBuffer(b));
      } else {
        buffers.push_back(
            tflite::CreateBuffer(b, /*data=*/0, cursor, payloads[i].size()));
        cursor += payloads[i].size();
      }
    }
    std::vector<int32_t> shape = {
        static_cast<int32_t>(opts.weight_payload.size())};
    auto tensor = tflite::CreateTensorDirect(
        b, &shape, tflite::TensorType_UINT8, weight_buf_idx, "test_weight");
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {tensor};
    auto subgraph = tflite::CreateSubGraphDirect(b, &tensors);
    std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {subgraph};

    std::vector<flatbuffers::Offset<tflite::Metadata>> metadata_offsets;
    for (const auto& [name, idx] : metadata_entries) {
      metadata_offsets.push_back(
          tflite::CreateMetadataDirect(b, name.c_str(), idx));
    }

    auto model = tflite::CreateModel(
        b, /*version=*/3, /*operator_codes=*/0, b.CreateVector(subgraphs),
        b.CreateString("artisan_test"), b.CreateVector(buffers),
        /*metadata_buffer=*/0, b.CreateVector(metadata_offsets));
    tflite::FinishModelBuffer(b, model);
    return std::string(reinterpret_cast<const char*>(b.GetBufferPointer()),
                       b.GetSize());
  };

  // Pad header to 4096 bytes so offsets are deterministic across passes.
  constexpr uint64_t kHeaderPaddedSize = 4096;
  std::string header = build_with_base(kHeaderPaddedSize);
  header.resize(kHeaderPaddedSize, '\0');
  for (size_t i = 1; i < payloads.size(); ++i) {
    header.append(payloads[i]);
  }
  return header;
}

std::filesystem::path WriteTempModelFile(absl::string_view filename,
                                         absl::string_view content) {
  auto path =
      std::filesystem::path(::testing::TempDir()) / std::string(filename);
  std::ofstream out(path, std::ios::binary);
  out.write(content.data(), content.size());
  const size_t alignment = MemoryMappedFile::GetOffsetAlignment();
  if (content.size() % alignment != 0) {
    std::string pad(alignment - (content.size() % alignment), '\0');
    out.write(pad.data(), pad.size());
  }
  out.close();
  return path;
}

TEST(ModelDataTest, FailedGetLlmParamsFromEmptyMetadata) {
  std::string bytes = BuildSyntheticArtisanModel({.include_llm_params = false,
                                                  .model_type = std::nullopt,
                                                  .backend = std::nullopt});
  std::shared_ptr<tflite::FlatBufferModel> model_ptr(
      tflite::FlatBufferModel::BuildFromBuffer(bytes.data(), bytes.size()));
  ASSERT_NE(model_ptr, nullptr);
  auto model_data = ModelData::Create(model_ptr);
  EXPECT_THAT(
      model_data,
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr("Failed to get metadata: odml.infra.proto.LlmParameters")));
}

TEST(ModelDataTest, InMemoryModelDataReadsParamsMetadataAndTensors) {
  SyntheticModelOptions opts;
  opts.lora_rank = 16;
  std::string bytes = BuildSyntheticArtisanModel(opts);
  std::shared_ptr<tflite::FlatBufferModel> model_ptr(
      tflite::FlatBufferModel::BuildFromBuffer(bytes.data(), bytes.size()));
  ASSERT_NE(model_ptr, nullptr);

  ASSERT_OK_AND_ASSIGN(auto model_data, ModelData::Create(model_ptr));
  const auto& llm_parameters = model_data->GetLlmParameters();
  EXPECT_TRUE(llm_parameters.has_transformer_parameters());
  EXPECT_EQ(llm_parameters.vocab_size(), 256000);
  EXPECT_EQ(llm_parameters.start_token_id(), 2);
  EXPECT_EQ(model_data->GetModelType(), proto::LLM_MODEL_TYPE_GEMMA_2B);
  EXPECT_EQ(model_data->LoRARank(), 16);

  ASSERT_OK_AND_ASSIGN(auto backend, model_data->ReadMetadata(kLlmBackendName));
  EXPECT_EQ(backend, "gpu");
  EXPECT_THAT(model_data->ReadMetadata("missing_key"),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_EQ(model_data->GetMaxTensorSize(), opts.weight_payload.size());
  EXPECT_EQ(model_data->GetTensorSize("test_weight"),
            opts.weight_payload.size());
  EXPECT_EQ(model_data->GetTensorSize("missing_tensor"), 0);

  ASSERT_OK_AND_ASSIGN(auto holder, model_data->ReadTensor("test_weight"));
  ASSERT_NE(holder, nullptr);
  absl::string_view read_str(
      reinterpret_cast<const char*>(holder->GetData().data()),
      holder->GetData().size());
  EXPECT_EQ(read_str, opts.weight_payload);

  ASSERT_OK_AND_ASSIGN(auto missing_holder,
                       model_data->ReadTensor("missing_tensor"));
  EXPECT_EQ(missing_holder, nullptr);
  model_data->Clear();
}

TEST(ModelDataTest, ReturnsNulloptWhenOptionalMetadataAbsent) {
  std::string bytes = BuildSyntheticArtisanModel({.model_type = std::nullopt,
                                                  .lora_rank = std::nullopt,
                                                  .backend = std::nullopt});
  std::shared_ptr<tflite::FlatBufferModel> model_ptr(
      tflite::FlatBufferModel::BuildFromBuffer(bytes.data(), bytes.size()));
  ASSERT_OK_AND_ASSIGN(auto model_data, ModelData::Create(model_ptr));
  EXPECT_EQ(model_data->GetModelType(), std::nullopt);
  EXPECT_EQ(model_data->LoRARank(), std::nullopt);
}

TEST(ModelDataTest, CreateModelDataFromScopedFileAndPathAndRange) {
  std::string bytes = BuildSyntheticArtisanModel();
  std::string standalone_bytes = bytes;
  standalone_bytes.resize(1024 * 1024, '\0');
  auto path = WriteTempModelFile("synthetic_artisan.tflite", standalone_bytes);

  // Create from path.
  ASSERT_OK_AND_ASSIGN(auto from_path, ModelData::Create(path.string()));
  from_path->SetMemoryMappedFileAccess(MemoryMappedFile::Access::kRead);
  ASSERT_OK_AND_ASSIGN(auto backend, from_path->ReadMetadata(kLlmBackendName));
  EXPECT_EQ(backend, "gpu");

  // Create from shared ScopedFile with explicit offset and size.
  constexpr size_t kPrefixPadding = 128;
  std::string padded(kPrefixPadding, 'P');
  padded.append(bytes);
  auto padded_path = WriteTempModelFile("padded_artisan.tflite", padded);
  ASSERT_OK_AND_ASSIGN(ScopedFile scoped_file,
                       ScopedFile::Open(padded_path.string()));
  auto shared_file = std::make_shared<const ScopedFile>(std::move(scoped_file));

  ASSERT_OK_AND_ASSIGN(auto from_range,
                       ModelData::Create(shared_file, /*offset=*/kPrefixPadding,
                                         /*size=*/bytes.size()));
  ASSERT_OK_AND_ASSIGN(auto weight, from_range->ReadTensor("test_weight"));
  ASSERT_NE(weight, nullptr);
  EXPECT_EQ(
      absl::string_view(reinterpret_cast<const char*>(weight->GetData().data()),
                        weight->GetData().size()),
      "weight_bytes_1234");
  EXPECT_GE(weight->GetRawData().size(), weight->GetData().size());
  from_range->Clear();
}

TEST(ModelDataTest, CreateModelDataFromReadDataFn) {
  std::string bytes = BuildSyntheticArtisanModel();
  // Pad to 1 MiB since ModelData::Create(ReadDataFn) reads kTfliteBaseSize (1
  // MiB).
  if (bytes.size() < 1024 * 1024) {
    bytes.resize(1024 * 1024, '\0');
  }
  bool discarded_all = false;
  ModelData::ReadDataFn fn = [&bytes, &discarded_all](uint64_t offset,
                                                      uint64_t size,
                                                      int mode) -> void* {
    if (mode == ModelData::ReadMode::DISCARD_ALL) {
      discarded_all = true;
      return nullptr;
    }
    if (offset + size > bytes.size()) {
      return nullptr;
    }
    void* out = malloc(size);
    memcpy(out, bytes.data() + offset, size);
    return out;
  };

  {
    ASSERT_OK_AND_ASSIGN(auto model_data, ModelData::Create(std::move(fn)));
    ASSERT_OK_AND_ASSIGN(auto backend,
                         model_data->ReadMetadata(kLlmBackendName));
    EXPECT_EQ(backend, "gpu");
    ASSERT_OK_AND_ASSIGN(auto weight, model_data->ReadTensor("test_weight"));
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(absl::string_view(
                  reinterpret_cast<const char*>(weight->GetData().data()),
                  weight->GetData().size()),
              "weight_bytes_1234");
  }
  EXPECT_TRUE(discarded_all);
}

}  // namespace
}  // namespace odml::infra::llm_utils
