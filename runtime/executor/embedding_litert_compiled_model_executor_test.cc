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

#include "runtime/executor/embedding_litert_compiled_model_executor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::litert::Dimensions;
using ::litert::ElementType;
using ::litert::Environment;
using ::litert::Layout;
using ::litert::Model;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::lm::Backend;
using ::litert::lm::EmbeddingExecutorSettings;
using ::litert::lm::ExecutorInputs;
using ::litert::lm::ExecutorTextData;
using ::litert::lm::ModelAssets;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

// Builds a minimal TFLite model FlatBuffer with given signature key and tensor
// shapes.
//
// The model computes `output = input + zeros`, where `zeros` is a constant of
// shape `output_dims`. ADD broadcasts its operands, so the output shape that
// the runtime computes when tensors are allocated is the broadcast of
// `input_dims` and `output_dims`. This lets a model expand a small input (e.g.
// a single token id of shape [1]) into a larger output (e.g. an embedding of
// shape [1, 1, 8]) whose runtime shape matches the declared `output_dims`.
//
// `input_type` and `output_type` must be the same 4-byte type (e.g. FLOAT32),
// since ADD requires matching operand types and the zero constant is sized
// with 4 bytes per element.
std::vector<uint8_t> BuildDummyTfLiteModelBuffer(
    absl::string_view signature_key, const std::vector<int32_t>& input_dims,
    tflite::TensorType input_type, const std::vector<int32_t>& output_dims,
    tflite::TensorType output_type) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({opcode});

  // Buffer 0 is the conventional empty buffer for non-constant tensors, and
  // buffer 1 holds the all-zero data of the constant ADD operand.
  size_t num_output_elements = 1;
  for (int32_t dim : output_dims) {
    num_output_elements *= dim;
  }
  const std::vector<uint8_t> zero_bytes(num_output_elements * sizeof(float),
                                        0);
  auto empty_buffer = tflite::CreateBuffer(builder);
  auto zeros_buffer =
      tflite::CreateBuffer(builder, builder.CreateVector(zero_bytes));
  auto buffers_vec = builder.CreateVector({empty_buffer, zeros_buffer});

  auto input_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(input_dims), input_type,
      /*buffer=*/0, builder.CreateString("input_tensor"));
  auto zeros_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(output_dims), output_type,
      /*buffer=*/1, builder.CreateString("zeros_tensor"));
  auto output_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(output_dims), output_type,
      /*buffer=*/0, builder.CreateString("output_tensor"));
  auto tensors_vec =
      builder.CreateVector({input_tensor, zeros_tensor, output_tensor});

  std::vector<int32_t> op_inputs = {0, 1};
  std::vector<int32_t> op_outputs = {2};
  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                   builder.CreateVector(op_inputs),
                                   builder.CreateVector(op_outputs));
  auto ops_vec = builder.CreateVector({op});

  // Only the input is a subgraph input; the zero constant is baked into the
  // model.
  std::vector<int32_t> sg_inputs = {0};
  std::vector<int32_t> sg_outputs = {2};
  auto subgraph = tflite::CreateSubGraph(
      builder, tensors_vec, builder.CreateVector(sg_inputs),
      builder.CreateVector(sg_outputs), ops_vec, builder.CreateString("main"));
  auto subgraphs_vec = builder.CreateVector({subgraph});

  auto input_map =
      tflite::CreateTensorMap(builder, builder.CreateString("input"), 0);
  auto output_map =
      tflite::CreateTensorMap(builder, builder.CreateString("output"), 2);
  auto inputs_map_vec = builder.CreateVector({input_map});
  auto outputs_map_vec = builder.CreateVector({output_map});
  auto sig_def = tflite::CreateSignatureDef(
      builder, inputs_map_vec, outputs_map_vec,
      builder.CreateString(std::string(signature_key)), /*subgraph_index=*/0);
  auto sig_defs_vec = builder.CreateVector({sig_def});

  auto model =
      tflite::CreateModel(builder, /*version=*/3, opcodes_vec, subgraphs_vec,
                          builder.CreateString("dummy_model"), buffers_vec,
                          /*metadata_buffer=*/0, /*metadata=*/0, sig_defs_vec);
  tflite::FinishModelBuffer(builder, model);

  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

std::vector<uint8_t> BuildDummyEncoderModelBuffer(
    absl::string_view signature_key,
    const std::vector<int32_t>& embeddings_dims,
    const std::vector<int32_t>& mask_dims,
    const std::vector<int32_t>& output_dims) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({opcode});

  auto embeddings_tensor =
      tflite::CreateTensor(builder, builder.CreateVector(embeddings_dims),
                           tflite::TensorType_FLOAT32,
                           /*buffer=*/0, builder.CreateString("embeddings"));
  auto mask_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(mask_dims), tflite::TensorType_FLOAT32,
      /*buffer=*/0, builder.CreateString("input_mask"));
  auto output_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(output_dims), tflite::TensorType_FLOAT32,
      /*buffer=*/0, builder.CreateString("output_tensor"));
  auto tensors_vec =
      builder.CreateVector({embeddings_tensor, mask_tensor, output_tensor});

  std::vector<int32_t> op_inputs = {0, 1};
  std::vector<int32_t> op_outputs = {2};
  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                   builder.CreateVector(op_inputs),
                                   builder.CreateVector(op_outputs));
  auto ops_vec = builder.CreateVector({op});

  std::vector<int32_t> sg_inputs = {0, 1};
  std::vector<int32_t> sg_outputs = {2};
  auto subgraph = tflite::CreateSubGraph(
      builder, tensors_vec, builder.CreateVector(sg_inputs),
      builder.CreateVector(sg_outputs), ops_vec, builder.CreateString("main"));
  auto subgraphs_vec = builder.CreateVector({subgraph});

  auto buffer = tflite::CreateBuffer(builder);
  auto buffers_vec = builder.CreateVector({buffer});

  auto embeddings_map =
      tflite::CreateTensorMap(builder, builder.CreateString("embeddings"), 0);
  auto mask_map =
      tflite::CreateTensorMap(builder, builder.CreateString("input_mask"), 1);
  auto output_map =
      tflite::CreateTensorMap(builder, builder.CreateString("output"), 2);
  auto inputs_map_vec = builder.CreateVector({embeddings_map, mask_map});
  auto outputs_map_vec = builder.CreateVector({output_map});
  auto sig_def = tflite::CreateSignatureDef(
      builder, inputs_map_vec, outputs_map_vec,
      builder.CreateString(std::string(signature_key)), /*subgraph_index=*/0);
  auto sig_defs_vec = builder.CreateVector({sig_def});

  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, subgraphs_vec,
      builder.CreateString("dummy_encoder_model"), buffers_vec,
      /*metadata_buffer=*/0, /*metadata=*/0, sig_defs_vec);
  tflite::FinishModelBuffer(builder, model);

  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

std::vector<uint8_t> BuildDummyMultiSignatureEncoderModelBuffer(
    const std::vector<std::string>& signature_keys,
    const std::vector<int32_t>& seq_lengths, int embedding_dim) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  for (size_t i = 0; i < signature_keys.size(); ++i) {
    int32_t seq_len = seq_lengths[i];
    std::vector<int32_t> embeddings_dims = {1, seq_len, embedding_dim};
    std::vector<int32_t> mask_dims = {1, seq_len, 1};
    std::vector<int32_t> output_dims = {1, seq_len, embedding_dim};

    auto embeddings_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(embeddings_dims),
        tflite::TensorType_FLOAT32, /*buffer=*/0,
        builder.CreateString(absl::StrCat("embeddings_", i)));
    auto mask_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(mask_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("input_mask_", i)));
    auto output_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(output_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("output_tensor_", i)));
    auto tensors_vec =
        builder.CreateVector({embeddings_tensor, mask_tensor, output_tensor});

    std::vector<int32_t> op_inputs = {0, 1};
    std::vector<int32_t> op_outputs = {2};
    auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                     builder.CreateVector(op_inputs),
                                     builder.CreateVector(op_outputs));
    auto ops_vec = builder.CreateVector({op});

    std::vector<int32_t> sg_inputs = {0, 1};
    std::vector<int32_t> sg_outputs = {2};
    auto subgraph = tflite::CreateSubGraph(
        builder, tensors_vec, builder.CreateVector(sg_inputs),
        builder.CreateVector(sg_outputs), ops_vec,
        builder.CreateString(absl::StrCat("subgraph_", i)));
    subgraphs.push_back(subgraph);

    auto embeddings_map =
        tflite::CreateTensorMap(builder, builder.CreateString("embeddings"), 0);
    auto mask_map =
        tflite::CreateTensorMap(builder, builder.CreateString("input_mask"), 1);
    auto output_map =
        tflite::CreateTensorMap(builder, builder.CreateString("output"), 2);
    auto inputs_map_vec = builder.CreateVector({embeddings_map, mask_map});
    auto outputs_map_vec = builder.CreateVector({output_map});

    auto sig_def = tflite::CreateSignatureDef(
        builder, inputs_map_vec, outputs_map_vec,
        builder.CreateString(signature_keys[i]), /*subgraph_index=*/i);
    sig_defs.push_back(sig_def);
  }

  auto subgraphs_vec = builder.CreateVector(subgraphs);
  auto sig_defs_vec = builder.CreateVector(sig_defs);

  auto buffer = tflite::CreateBuffer(builder);
  auto buffers_vec = builder.CreateVector({buffer});

  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, subgraphs_vec,
      builder.CreateString("dummy_multi_sig_encoder_model"), buffers_vec,
      /*metadata_buffer=*/0, /*metadata=*/0, sig_defs_vec);
  tflite::FinishModelBuffer(builder, model);

  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

// Fake ModelResources implementation providing in-memory TFLite models for
// testing.
class FakeEmbeddingModelResources : public ModelResources {
 public:
  using ModelResources::GetTFLiteModel;
  using ModelResources::GetTFLiteModelBuffer;
  using ModelResources::HasTFLiteModel;

  FakeEmbeddingModelResources(const litert::Model* embedder,
                              const litert::Model* encoder,
                              absl::string_view encoder_buffer = {})
      : embedder_(embedder),
        encoder_(encoder),
        encoder_buffer_(encoder_buffer) {}

  bool HasTFLiteModel(proto::LlmMetadata::TfLiteModelType model_type) override {
    return HasTFLiteModel(TfLiteModelTypeToWireString(model_type));
  }

  bool HasTFLiteModel(absl::string_view model_type_str) override {
    if (model_type_str == "tf_lite_embedder") {
      return embedder_ != nullptr;
    }
    if (model_type_str == "tf_lite_text_encoder") {
      return encoder_ != nullptr;
    }
    return false;
  }

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetTFLiteModel(TfLiteModelTypeToWireString(model_type));
  }

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      absl::string_view model_type_str) override {
    if (model_type_str == "tf_lite_embedder") {
      if (embedder_ != nullptr) return embedder_;
      return absl::NotFoundError("kTfLiteEmbedder model not found.");
    }
    if (model_type_str == "tf_lite_text_encoder") {
      if (encoder_ != nullptr) return encoder_;
      return absl::NotFoundError("kTfLiteTextEncoder model not found.");
    }
    return absl::NotFoundError("Model type not found.");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetTFLiteModelBuffer(TfLiteModelTypeToWireString(model_type));
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      absl::string_view model_type_str) override {
    if (model_type_str == "tf_lite_text_encoder" && !encoder_buffer_.empty()) {
      return encoder_buffer_;
    }
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("Unimplemented");
  }

 private:
  const litert::Model* embedder_;
  const litert::Model* encoder_;
  absl::string_view encoder_buffer_;
};

TEST(EmbeddingLiteRtCompiledModelExecutorTest, CreateExecutor_Success) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  auto encoder_buf = BuildDummyTfLiteModelBuffer(
      "encoder", {1, 1, 8}, tflite::TensorType_FLOAT32, {1, 8},
      tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));
  EXPECT_EQ(embedding_executor->ExecutorBackendName(), "CPU");
  EXPECT_NE(embedding_executor->GetEnvironment(), nullptr);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest, GetDimension_Success) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  auto encoder_buf = BuildDummyTfLiteModelBuffer(
      "encoder", {1, 1, 8}, tflite::TensorType_FLOAT32, {1, 8},
      tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  ASSERT_OK_AND_ASSIGN(int embedding_dim,
                       embedding_executor->GetEmbeddingDimension());
  EXPECT_EQ(embedding_dim, 8);

  ASSERT_OK_AND_ASSIGN(auto input_dims,
                       embedding_executor->GetExpectedInputDimension());
  EXPECT_EQ(input_dims, (std::vector<int>{1, 1, 8}));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest, ComputeEmbedding_Success) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  auto encoder_buf = BuildDummyTfLiteModelBuffer(
      "encoder", {1, 1, 8}, tflite::TensorType_FLOAT32, {1, 8},
      tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[] = {42};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({1}))),
          token_data, sizeof(token_data)));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::nullopt, std::nullopt);

  ASSERT_OK_AND_ASSIGN(auto output,
                       embedding_executor->ComputeEmbedding(inputs));
  EXPECT_EQ(output.embedding.size(), 8);
  EXPECT_EQ(output.input_length, 1);
  EXPECT_EQ(output.truncated_length, std::nullopt);
  EXPECT_EQ(output.num_chunks, 1);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest, ComputeEmbeddingBatch_Success) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  auto encoder_buf = BuildDummyTfLiteModelBuffer(
      "encoder", {1, 1, 8}, tflite::TensorType_FLOAT32, {1, 8},
      tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data1[] = {10};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer1,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({1}))),
          token_data1, sizeof(token_data1)));
  ExecutorInputs inputs1(ExecutorTextData(std::move(token_ids_buffer1)),
                         std::nullopt, std::nullopt);

  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data2[] = {20};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer2,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({1}))),
          token_data2, sizeof(token_data2)));
  ExecutorInputs inputs2(ExecutorTextData(std::move(token_ids_buffer2)),
                         std::nullopt, std::nullopt);

  std::vector<ExecutorInputs> batch_inputs;
  batch_inputs.push_back(std::move(inputs1));
  batch_inputs.push_back(std::move(inputs2));

  ASSERT_OK_AND_ASSIGN(auto batch_outputs,
                       embedding_executor->ComputeEmbeddingBatch(batch_inputs));
  EXPECT_EQ(batch_outputs.size(), 2);
  EXPECT_EQ(batch_outputs[0].embedding.size(), 8);
  EXPECT_EQ(batch_outputs[0].input_length, 1);
  EXPECT_EQ(batch_outputs[0].truncated_length, std::nullopt);
  EXPECT_EQ(batch_outputs[0].num_chunks, 1);
  EXPECT_EQ(batch_outputs[1].embedding.size(), 8);
  EXPECT_EQ(batch_outputs[1].input_length, 1);
  EXPECT_EQ(batch_outputs[1].truncated_length, std::nullopt);
  EXPECT_EQ(batch_outputs[1].num_chunks, 1);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_NullResources_ReturnsInternalError) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto embedding_executor =
      EmbeddingLiteRtCompiledModelExecutor::Create(settings, env, nullptr);
  EXPECT_THAT(embedding_executor,
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("ModelResources is null.")));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_MissingEmbedder_RejectsTokenInputs) {
  auto encoder_buf = BuildDummyTfLiteModelBuffer(
      "encoder", {1, 1, 8}, tflite::TensorType_FLOAT32, {1, 8},
      tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  // The embedder is optional: without it, only pre-computed input embeddings
  // are supported.
  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(nullptr, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[] = {42};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({1}))),
          token_data, sizeof(token_data)));
  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::nullopt, std::nullopt);
  EXPECT_THAT(embedding_executor->ComputeEmbedding(inputs),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("requires the embedder model")));

  std::vector<float> input_embeddings(8, 1.0f);
  ASSERT_OK_AND_ASSIGN(std::vector<float> output,
                       embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                           input_embeddings));
  EXPECT_EQ(output.size(), 8);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingFromInputEmbeddings_PadsAndMasksTheSequence) {
  // The encoder computes `output = embeddings + input_mask` for a sequence
  // length of 4 and an embedding dimension of 8.
  auto encoder_buf = BuildDummyEncoderModelBuffer(
      "encoder", /*embeddings_dims=*/{1, 4, 8}, /*mask_dims=*/{1, 4, 1},
      /*output_dims=*/{1, 4, 8});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(nullptr, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  // A sequence of 2 embeddings, padded to the signature length of 4.
  std::vector<float> input_embeddings(2 * 8);
  for (size_t i = 0; i < input_embeddings.size(); ++i) {
    input_embeddings[i] = 0.5f * i;
  }
  ASSERT_OK_AND_ASSIGN(std::vector<float> output,
                       embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                           input_embeddings));
  ASSERT_EQ(output.size(), 4 * 8);
  for (size_t i = 0; i < input_embeddings.size(); ++i) {
    // Active positions: the input embeddings plus a mask of 1.
    EXPECT_FLOAT_EQ(output[i], input_embeddings[i] + 1.0f) << "i=" << i;
  }
  for (size_t i = input_embeddings.size(); i < output.size(); ++i) {
    // Padded positions: zero embeddings plus a mask of 0.
    EXPECT_FLOAT_EQ(output[i], 0.0f) << "i=" << i;
  }

  // A full sequence reuses the same signature.
  std::vector<float> full_input_embeddings(4 * 8, 2.0f);
  ASSERT_OK_AND_ASSIGN(output,
                       embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                           full_input_embeddings));
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 3.0f);
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingFromInputEmbeddings_Cancellation) {
  auto encoder_buf = BuildDummyEncoderModelBuffer(
      "encoder", /*embeddings_dims=*/{1, 4, 8}, /*mask_dims=*/{1, 4, 1},
      /*output_dims=*/{1, 4, 8});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(nullptr, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));
  std::vector<float> input_embeddings(2 * 8, 1.0f);

  // Not cancelled.
  EXPECT_OK(embedding_executor->ComputeEmbeddingFromInputEmbeddings(
      input_embeddings, [] { return false; }));

  // Cancelled before the run.
  EXPECT_THAT(embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                  input_embeddings, [] { return true; }),
              StatusIs(absl::StatusCode::kCancelled));

  // Cancelled before the run: `should_cancel` is called once, then the result
  // is latched.
  int num_checks = 0;
  EXPECT_THAT(
      embedding_executor->ComputeEmbeddingFromInputEmbeddings(input_embeddings,
                                                              [&] {
                                                                ++num_checks;
                                                                return true;
                                                              }),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 1);

  // Cancelled during the run: the two checks before the run pass, then the
  // LiteRT interpreter requests cancellation after an op. `should_cancel` is
  // not called again once it returned true.
  num_checks = 0;
  auto result = embedding_executor->ComputeEmbeddingFromInputEmbeddings(
      input_embeddings, [&] { return ++num_checks > 2; });
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 3);

  // A one-shot (stateful) `should_cancel` that returns true only once, during
  // the run, still cancels the call: the cancellation is latched.
  num_checks = 0;
  result = embedding_executor->ComputeEmbeddingFromInputEmbeddings(
      input_embeddings, [&] { return ++num_checks == 3; });
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kCancelled));

  // The executor is usable after a cancelled run, and a call without a
  // cancellation callback is not affected by the previous one.
  ASSERT_OK_AND_ASSIGN(std::vector<float> output,
                       embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                           input_embeddings));
  EXPECT_EQ(output.size(), 4 * 8);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest, GetSupportedVariants) {
  // Model with two signatures: encoder_12 and encoder_24.
  auto encoder_buf = BuildDummyMultiSignatureEncoderModelBuffer(
      {"encoder_24", "encoder_12"}, {24, 12}, 8);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(nullptr, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  // Sorted by ascending sequence length, regardless of the signature order in
  // the model.
  EXPECT_THAT(
      embedding_executor->GetSupportedVariants(),
      ElementsAre(TextEncoderVariant{.batch_size = 1, .sequence_length = 12},
                  TextEncoderVariant{.batch_size = 1, .sequence_length = 24}));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     GetSupportedVariants_OnlyReportsSelectedSignatures) {
  auto encoder_buf = BuildDummyMultiSignatureEncoderModelBuffer(
      {"encoder_12", "encoder_24"}, {12, 24}, 8);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  settings.SetSelectedSignatures({"encoder_24"});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(nullptr, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  EXPECT_THAT(
      embedding_executor->GetSupportedVariants(),
      ElementsAre(TextEncoderVariant{.batch_size = 1, .sequence_length = 24}));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingFromInputEmbeddings_CancellationDoesNotLeakToOtherCalls) {
  auto encoder_buf = BuildDummyEncoderModelBuffer(
      "encoder", /*embeddings_dims=*/{1, 4, 8}, /*mask_dims=*/{1, 4, 1},
      /*output_dims=*/{1, 4, 8});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(nullptr, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));
  std::vector<float> input_embeddings(2 * 8, 1.0f);

  // A cancelled call ...
  int num_checks = 0;
  EXPECT_THAT(embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                  input_embeddings, [&] { return ++num_checks > 2; }),
              StatusIs(absl::StatusCode::kCancelled));
  const int num_checks_after_cancelled_call = num_checks;

  // ... does not affect the following calls: the latched state is reset and
  // the per-op check of the cancelled call is unregistered.
  ASSERT_OK_AND_ASSIGN(std::vector<float> output,
                       embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                           input_embeddings));
  EXPECT_EQ(num_checks, num_checks_after_cancelled_call);
  ASSERT_OK_AND_ASSIGN(std::vector<float> cancellable_output,
                       embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                           input_embeddings, [] { return false; }));
  EXPECT_EQ(cancellable_output, output);
}

// Tests that the cancellation state of a cancellable call (its
// CancellationLatch and `should_cancel`) is scoped to the call: it is neither
// used by nor leaked into the following calls.
class TextEncoderCancellationScopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_buffer_ = BuildDummyEncoderModelBuffer(
        "encoder", /*embeddings_dims=*/{1, 4, 8}, /*mask_dims=*/{1, 4, 1},
        /*output_dims=*/{1, 4, 8});
    LITERT_ASSERT_OK_AND_ASSIGN(
        model_, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                    model_buffer_.data(), model_buffer_.size())));
    ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                         ModelAssets::Create("dummy_model_path"));
    ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                         EmbeddingExecutorSettings::CreateDefault(
                             model_assets, /*backend=*/Backend::CPU));
    LITERT_ASSERT_OK_AND_ASSIGN(
        env_, Environment::Create(std::vector<Environment::Option>()));
    ASSERT_OK_AND_ASSIGN(
        executor_,
        EmbeddingLiteRtCompiledModelExecutor::Create(
            settings, *env_,
            std::make_unique<FakeEmbeddingModelResources>(nullptr, &*model_)));
    ASSERT_OK_AND_ASSIGN(
        expected_output_,
        executor_->ComputeEmbeddingFromInputEmbeddings(input_embeddings_));
  }

  // Checks that a plain call still works and produces the expected output.
  void ExpectPlainCallWorks() {
    EXPECT_THAT(
        executor_->ComputeEmbeddingFromInputEmbeddings(input_embeddings_),
        IsOkAndHolds(expected_output_));
  }

  std::vector<uint8_t> model_buffer_;
  std::optional<Model> model_;
  std::optional<Environment> env_;
  std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor> executor_;
  std::vector<float> input_embeddings_ = std::vector<float>(2 * 8, 1.0f);
  std::vector<float> expected_output_;
};

TEST_F(TextEncoderCancellationScopeTest, LaterCallsDontUseTheCallback) {
  bool cancel = false;
  int num_checks = 0;
  auto should_cancel = [&] {
    ++num_checks;
    return cancel;
  };
  EXPECT_THAT(executor_->ComputeEmbeddingFromInputEmbeddings(input_embeddings_,
                                                             should_cancel),
              IsOkAndHolds(expected_output_));
  ASSERT_GT(num_checks, 0);
  const int num_checks_after_call = num_checks;

  // The callback, still alive, would now cancel. Neither the pre-run checks
  // nor the per-op checks of the following plain calls call it.
  cancel = true;
  for (int i = 0; i < 3; ++i) {
    ExpectPlainCallWorks();
  }
  EXPECT_EQ(num_checks, num_checks_after_call);
}

TEST_F(TextEncoderCancellationScopeTest,
       CallbackCanBeDestroyedRightAfterTheCall) {
  // Both after a completed and after a cancelled call. If the latch or the
  // callback were still referenced (e.g. by the per-op check registered with
  // LiteRT), the following calls would use freed memory (caught by ASan).
  for (const bool cancel_during_run : {false, true}) {
    SCOPED_TRACE(cancel_during_run);
    int num_checks = 0;
    auto should_cancel = std::make_unique<std::function<bool()>>(
        [&] { return ++num_checks > 2 && cancel_during_run; });
    absl::StatusOr<std::vector<float>> result =
        executor_->ComputeEmbeddingFromInputEmbeddings(input_embeddings_,
                                                       *should_cancel);
    if (cancel_during_run) {
      EXPECT_THAT(result, StatusIs(absl::StatusCode::kCancelled));
    } else {
      EXPECT_THAT(result, IsOkAndHolds(expected_output_));
    }
    should_cancel.reset();

    ExpectPlainCallWorks();
    EXPECT_THAT(executor_->ComputeEmbeddingFromInputEmbeddings(
                    input_embeddings_, [] { return false; }),
                IsOkAndHolds(expected_output_));
  }
}

TEST_F(TextEncoderCancellationScopeTest, EachCallHasItsOwnLatch) {
  // A call cancelled during the run by a one-shot callback: its latch is
  // cancelled.
  int num_checks = 0;
  EXPECT_THAT(executor_->ComputeEmbeddingFromInputEmbeddings(
                  input_embeddings_, [&] { return ++num_checks == 3; }),
              StatusIs(absl::StatusCode::kCancelled));

  // The next call starts with a fresh latch, which calls its own callback (at
  // entry and before the run) instead of reporting the previous cancellation.
  num_checks = 0;
  EXPECT_THAT(executor_->ComputeEmbeddingFromInputEmbeddings(input_embeddings_,
                                                             [&] {
                                                               ++num_checks;
                                                               return false;
                                                             }),
              IsOkAndHolds(expected_output_));
  EXPECT_GE(num_checks, 2);

  // And a call cancelled at entry is cancelled by its own callback only.
  num_checks = 0;
  EXPECT_THAT(executor_->ComputeEmbeddingFromInputEmbeddings(input_embeddings_,
                                                             [&] {
                                                               ++num_checks;
                                                               return true;
                                                             }),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 1);
  ExpectPlainCallWorks();
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_NullCompiledTextEncoder_ReturnsError) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  EXPECT_THAT(
      EmbeddingLiteRtCompiledModelExecutor::Create(
          settings, env, /*resources=*/nullptr,
          /*embedding_lookup=*/nullptr,
          /*per_layer_embedding_lookup=*/nullptr, CompiledTextEncoderInfo{}),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("The compiled text encoder is null.")));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingFromInputEmbeddings_InvalidSizes) {
  auto encoder_buf = BuildDummyEncoderModelBuffer(
      "encoder", /*embeddings_dims=*/{1, 4, 8}, /*mask_dims=*/{1, 4, 1},
      /*output_dims=*/{1, 4, 8});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(nullptr, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  // Empty.
  EXPECT_THAT(embedding_executor->ComputeEmbeddingFromInputEmbeddings({}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Not a multiple of the embedding dimension.
  EXPECT_THAT(embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                  std::vector<float>(15, 1.0f)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("multiple of the embedding dimension")));
  // Longer than the longest signature.
  EXPECT_THAT(embedding_executor->ComputeEmbeddingFromInputEmbeddings(
                  std::vector<float>(5 * 8, 1.0f)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds maximum supported signature")));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_MissingTextEncoder_ReturnsNotFoundError) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(&embedder_model, nullptr);
  auto embedding_executor = EmbeddingLiteRtCompiledModelExecutor::Create(
      settings, env, std::move(fake_resources));
  EXPECT_THAT(embedding_executor,
              StatusIs(absl::StatusCode::kNotFound,
                       "kTfLiteTextEncoder model not found."));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbedding_NullTextData_ReturnsError) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  auto encoder_buf = BuildDummyTfLiteModelBuffer(
      "encoder", {1, 1, 8}, tflite::TensorType_FLOAT32, {1, 8},
      tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  // Inputs without text_data initialized
  ExecutorInputs empty_inputs;
  auto embedding = embedding_executor->ComputeEmbedding(empty_inputs);
  EXPECT_FALSE(embedding.ok());
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbedding_WithMask_Success) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  auto encoder_buf =
      BuildDummyEncoderModelBuffer("encoder", {1, 1, 8}, {1, 1}, {1, 8});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[] = {42};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({1}))),
          token_data, sizeof(token_data)));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::nullopt, std::nullopt);

  ASSERT_OK_AND_ASSIGN(auto output,
                       embedding_executor->ComputeEmbedding(inputs));
  EXPECT_EQ(output.embedding.size(), 8);
  EXPECT_EQ(output.input_length, 1);
  EXPECT_EQ(output.truncated_length, std::nullopt);
  EXPECT_EQ(output.num_chunks, 1);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbedding_MultipleSignatures_Success) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  // Model with two signatures: encoder_12 and encoder_24.
  auto encoder_buf = BuildDummyMultiSignatureEncoderModelBuffer(
      {"encoder_12", "encoder_24"}, {12, 24}, 8);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  // Case 1: Input size 8 -> should choose encoder_12.
  {
    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[8];
    std::fill_n(token_data, 8, 42);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto token_ids_buffer,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({8}))),
            token_data, sizeof(token_data)));

    ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                          std::nullopt, std::nullopt);

    ASSERT_OK_AND_ASSIGN(auto output,
                         embedding_executor->ComputeEmbedding(inputs));
    EXPECT_EQ(output.embedding.size(), 12 * 8);
    EXPECT_EQ(output.input_length, 8);
    EXPECT_EQ(output.truncated_length, std::nullopt);
    EXPECT_EQ(output.num_chunks, 1);
  }

  // Case 2: Input size 16 -> should choose encoder_24.
  {
    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[16];
    std::fill_n(token_data, 16, 42);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto token_ids_buffer,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({16}))),
            token_data, sizeof(token_data)));

    ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                          std::nullopt, std::nullopt);

    ASSERT_OK_AND_ASSIGN(auto output,
                         embedding_executor->ComputeEmbedding(inputs));
    EXPECT_EQ(output.embedding.size(), 24 * 8);
    EXPECT_EQ(output.input_length, 16);
    EXPECT_EQ(output.truncated_length, std::nullopt);
    EXPECT_EQ(output.num_chunks, 1);
  }

  // Case 3: Input size 48 with kChunkAndAverage strategy -> Chunk and Average
  // across 2 chunks of size 24 (both using encoder_24).
  {
    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[48];
    std::fill_n(token_data, 48, 42);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto token_ids_buffer,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({48}))),
            token_data, sizeof(token_data)));

    ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                          std::nullopt, std::nullopt);

    ComputeEmbeddingOptions options{
        .input_overflow_strategy = InputOverflowStrategy::kChunkAndAverage,
    };
    ASSERT_OK_AND_ASSIGN(auto output,
                         embedding_executor->ComputeEmbedding(inputs, options));
    EXPECT_EQ(output.embedding.size(), 24 * 8);
    EXPECT_EQ(output.input_length, 48);
    EXPECT_EQ(output.truncated_length, std::nullopt);
    EXPECT_EQ(output.num_chunks, 2);
  }

  // Case 4: Input size 48 with kTruncate strategy -> truncates to max signature
  // length (24).
  {
    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[48];
    std::fill_n(token_data, 48, 42);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto token_ids_buffer,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({48}))),
            token_data, sizeof(token_data)));

    ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                          std::nullopt, std::nullopt);

    ComputeEmbeddingOptions options{
        .input_overflow_strategy = InputOverflowStrategy::kTruncate,
    };
    ASSERT_OK_AND_ASSIGN(auto output,
                         embedding_executor->ComputeEmbedding(inputs, options));
    EXPECT_EQ(output.embedding.size(), 24 * 8);
    EXPECT_EQ(output.input_length, 48);
    EXPECT_EQ(output.truncated_length, 24);
    EXPECT_EQ(output.num_chunks, 1);
  }

  // Case 5: Input size 48 with default kError strategy -> returns
  // InvalidArgumentError.
  {
    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[48];
    std::fill_n(token_data, 48, 42);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto token_ids_buffer,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({48}))),
            token_data, sizeof(token_data)));

    ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                          std::nullopt, std::nullopt);

    auto result = embedding_executor->ComputeEmbedding(inputs);
    EXPECT_THAT(
        result,
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("exceeds maximum supported signature length")));
  }

  // Case 6: Input size 48 with kTruncate strategy and eos_token_ids ->
  // truncates to max signature length (24) and preserves EOS token.
  {
    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[48];
    for (int i = 0; i < 47; ++i) {
      token_data[i] = i;
    }
    token_data[47] = 999;  // EOS token
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto token_ids_buffer,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({48}))),
            token_data, sizeof(token_data)));

    ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                          std::nullopt, std::nullopt);

    ComputeEmbeddingOptions options{
        .input_overflow_strategy = InputOverflowStrategy::kTruncate,
        .eos_token_ids = {999},
    };
    ASSERT_OK_AND_ASSIGN(auto output,
                         embedding_executor->ComputeEmbedding(inputs, options));
    EXPECT_EQ(output.embedding.size(), 24 * 8);
    EXPECT_EQ(output.input_length, 48);
    EXPECT_EQ(output.truncated_length, 24);
    EXPECT_EQ(output.num_chunks, 1);
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest, Create_WithCache_Success) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  auto encoder_buf = BuildDummyTfLiteModelBuffer(
      "encoder", {1, 1, 8}, tflite::TensorType_FLOAT32, {1, 8},
      tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  settings.SetCacheDir("/tmp/test_cache_dir");
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));
  EXPECT_EQ(embedding_executor->ExecutorBackendName(), "CPU");
  EXPECT_NE(embedding_executor->GetEnvironment(), nullptr);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_WithSelectedSignatures_FiltersSignatures) {
  auto embedder_buf =
      BuildDummyTfLiteModelBuffer("embedder", {1}, tflite::TensorType_FLOAT32,
                                  {1, 1, 8}, tflite::TensorType_FLOAT32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  auto encoder_buf = BuildDummyMultiSignatureEncoderModelBuffer(
      {"encoder_12", "encoder_24"}, {12, 24}, /*embedding_dim=*/8);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, /*backend=*/Backend::CPU));
  settings.SetSelectedSignatures({"encoder_12"});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto fake_resources = std::make_unique<FakeEmbeddingModelResources>(
      &embedder_model, &encoder_model);
  ASSERT_OK_AND_ASSIGN(auto embedding_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env, std::move(fake_resources)));

  // Input of 24 tokens with truncate strategy should truncate to 12 since
  // only encoder_12 was selected.
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[24];
  for (int i = 0; i < 24; ++i) {
    token_data[i] = i + 1;
  }
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({24}))),
          token_data, sizeof(token_data)));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::nullopt, std::nullopt);

  ComputeEmbeddingOptions options{
      .input_overflow_strategy = InputOverflowStrategy::kTruncate,
  };
  ASSERT_OK_AND_ASSIGN(auto output,
                       embedding_executor->ComputeEmbedding(inputs, options));
  EXPECT_EQ(output.embedding.size(), 12 * 8);
  EXPECT_EQ(output.input_length, 24);
  EXPECT_EQ(output.truncated_length, 12);
}

// Builds an embedder model that converts INT32 token IDs `[1]` into FLOAT32
// embeddings `[1, 1, hidden_dim]` (`CAST(input) + ramp_const`).
std::vector<uint8_t> BuildCastAddEmbedderBuffer(int32_t hidden_dim) {
  tflite::ModelT model;
  model.version = 3;
  model.description = "cast_add_embedder";

  auto add_opcode = [&](tflite::BuiltinOperator code) -> uint32_t {
    auto oc = std::make_unique<tflite::OperatorCodeT>();
    oc->builtin_code = code;
    oc->deprecated_builtin_code =
        code < 127
            ? static_cast<int8_t>(code)
            : static_cast<int8_t>(
                  tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
    oc->version = 1;
    model.operator_codes.push_back(std::move(oc));
    return static_cast<uint32_t>(model.operator_codes.size() - 1);
  };
  const uint32_t op_cast = add_opcode(tflite::BuiltinOperator_CAST);
  const uint32_t op_add = add_opcode(tflite::BuiltinOperator_ADD);

  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto ramp_buf = std::make_unique<tflite::BufferT>();
  std::vector<float> ramp(hidden_dim);
  for (int32_t d = 0; d < hidden_dim; ++d) {
    ramp[d] = 0.1f * static_cast<float>(d + 1);
  }
  ramp_buf->data.resize(ramp.size() * sizeof(float));
  std::memcpy(ramp_buf->data.data(), ramp.data(), ramp_buf->data.size());
  model.buffers.push_back(std::move(ramp_buf));

  auto sg = std::make_unique<tflite::SubGraphT>();
  sg->name = "main";
  auto make_tensor = [&](std::vector<int32_t> shape, tflite::TensorType type,
                         uint32_t buffer, std::string name) -> int32_t {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->shape = std::move(shape);
    tensor->type = type;
    tensor->buffer = buffer;
    tensor->name = std::move(name);
    sg->tensors.push_back(std::move(tensor));
    return static_cast<int32_t>(sg->tensors.size() - 1);
  };

  const int32_t t_in =
      make_tensor({1}, tflite::TensorType_INT32, 0, "token_ids");
  const int32_t t_float =
      make_tensor({1}, tflite::TensorType_FLOAT32, 0, "token_float");
  const int32_t t_ramp =
      make_tensor({1, 1, hidden_dim}, tflite::TensorType_FLOAT32, 1, "ramp");
  const int32_t t_out =
      make_tensor({1, 1, hidden_dim}, tflite::TensorType_FLOAT32, 0, "output");

  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = op_cast;
    op->inputs = {t_in};
    op->outputs = {t_float};
    tflite::CastOptionsT cast_opts;
    cast_opts.in_data_type = tflite::TensorType_INT32;
    cast_opts.out_data_type = tflite::TensorType_FLOAT32;
    op->builtin_options.Set(cast_opts);
    sg->operators.push_back(std::move(op));
  }
  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = op_add;
    op->inputs = {t_float, t_ramp};
    op->outputs = {t_out};
    tflite::AddOptionsT add_opts;
    op->builtin_options.Set(add_opts);
    sg->operators.push_back(std::move(op));
  }

  sg->inputs = {t_in};
  sg->outputs = {t_out};
  model.subgraphs.push_back(std::move(sg));

  auto sig = std::make_unique<tflite::SignatureDefT>();
  sig->signature_key = "embedder";
  sig->subgraph_index = 0;
  auto in_map = std::make_unique<tflite::TensorMapT>();
  in_map->name = "token_ids";
  in_map->tensor_index = t_in;
  sig->inputs.push_back(std::move(in_map));
  auto out_map = std::make_unique<tflite::TensorMapT>();
  out_map->name = "embeddings";
  out_map->tensor_index = t_out;
  sig->outputs.push_back(std::move(out_map));
  model.signature_defs.push_back(std::move(sig));

  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

struct SyntheticEncoderOptions {
  std::vector<std::string> signature_keys = {"encoder_4"};
  std::vector<int32_t> seq_lengths = {4};
  int32_t hidden_dim = 4;
  int32_t emb_dim = 6;
  int32_t num_layers = 8;
  bool pre_expose_token_embeddings = false;
  bool add_pooling_bias = false;
  bool append_proj_weight_buffer = false;
};

// Hand-builds a synthetic EGv2-structured TFLite encoder FlatBuffer with:
//   - per-layer UNPACK -> MUL -> FULLY_CONNECTED -> STABLEHLO_COMPOSITE -> ADD
//     -> skip_scale MUL residual chain (`num_layers` layers),
//   - final STABLEHLO_COMPOSITE norm producing `h` `[1, seq_len, hidden_dim]`,
//   - masked mean pooling `MUL(h, input_mask) -> SUM -> DIV -> FULLY_CONNECTED`
//     producing output 0 `encodings` `[1, emb_dim]`.
std::vector<uint8_t> BuildSyntheticEgv2LikeEncoderBuffer(
    const SyntheticEncoderOptions& opts) {
  tflite::ModelT model;
  model.version = 3;
  model.description = "synthetic_egv2_encoder";

  auto add_opcode = [&](tflite::BuiltinOperator code) -> uint32_t {
    auto oc = std::make_unique<tflite::OperatorCodeT>();
    oc->builtin_code = code;
    oc->deprecated_builtin_code =
        code < 127
            ? static_cast<int8_t>(code)
            : static_cast<int8_t>(
                  tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
    oc->version = 1;
    model.operator_codes.push_back(std::move(oc));
    return static_cast<uint32_t>(model.operator_codes.size() - 1);
  };

  const uint32_t op_add = add_opcode(tflite::BuiltinOperator_ADD);
  const uint32_t op_mul = add_opcode(tflite::BuiltinOperator_MUL);
  const uint32_t op_div = add_opcode(tflite::BuiltinOperator_DIV);
  const uint32_t op_sum = add_opcode(tflite::BuiltinOperator_SUM);
  const uint32_t op_reshape = add_opcode(tflite::BuiltinOperator_RESHAPE);
  const uint32_t op_unpack = add_opcode(tflite::BuiltinOperator_UNPACK);
  const uint32_t op_fc = add_opcode(tflite::BuiltinOperator_FULLY_CONNECTED);
  const uint32_t op_composite =
      add_opcode(tflite::BuiltinOperator_STABLEHLO_COMPOSITE);

  model.buffers.push_back(std::make_unique<tflite::BufferT>());  // buffer 0

  auto add_float_buffer = [&](const std::vector<float>& vals) -> uint32_t {
    auto buf = std::make_unique<tflite::BufferT>();
    buf->data.resize(vals.size() * sizeof(float));
    std::memcpy(buf->data.data(), vals.data(), buf->data.size());
    model.buffers.push_back(std::move(buf));
    return static_cast<uint32_t>(model.buffers.size() - 1);
  };
  auto add_int32_buffer = [&](const std::vector<int32_t>& vals) -> uint32_t {
    auto buf = std::make_unique<tflite::BufferT>();
    buf->data.resize(vals.size() * sizeof(int32_t));
    std::memcpy(buf->data.data(), vals.data(), buf->data.size());
    model.buffers.push_back(std::move(buf));
    return static_cast<uint32_t>(model.buffers.size() - 1);
  };

  // Shared constant buffers across signatures.
  const uint32_t zeros_hidden_buf =
      add_float_buffer(std::vector<float>(opts.hidden_dim, 0.0f));
  const uint32_t ones_hidden_buf =
      add_float_buffer(std::vector<float>(opts.hidden_dim, 1.0f));
  const uint32_t half_hidden_buf =
      add_float_buffer(std::vector<float>(opts.hidden_dim, 0.5f));

  std::vector<float> ple_fc_weights(opts.hidden_dim * opts.hidden_dim, 0.0f);
  for (int32_t d = 0; d < opts.hidden_dim; ++d) {
    ple_fc_weights[d * opts.hidden_dim + d] = 0.25f;
  }
  const uint32_t ple_fc_buf = add_float_buffer(ple_fc_weights);

  std::vector<float> layer_bias(opts.num_layers * opts.hidden_dim);
  for (int32_t li = 0; li < opts.num_layers; ++li) {
    for (int32_t d = 0; d < opts.hidden_dim; ++d) {
      layer_bias[li * opts.hidden_dim + d] =
          0.2f * static_cast<float>(li + 1) + 0.01f * static_cast<float>(d);
    }
  }
  const uint32_t layer_bias_buf = add_float_buffer(layer_bias);

  std::vector<float> proj_weights(opts.emb_dim * opts.hidden_dim);
  for (int32_t o = 0; o < opts.emb_dim; ++o) {
    for (int32_t d = 0; d < opts.hidden_dim; ++d) {
      proj_weights[o * opts.hidden_dim + d] =
          0.1f * static_cast<float>((o + 1) * (d + 1));
    }
  }
  std::vector<uint8_t> appended_proj_bytes(
      reinterpret_cast<const uint8_t*>(proj_weights.data()),
      reinterpret_cast<const uint8_t*>(proj_weights.data() +
                                       proj_weights.size()));
  uint32_t proj_weight_buf = 0;
  if (opts.append_proj_weight_buffer) {
    auto buf = std::make_unique<tflite::BufferT>();
    buf->offset = 64;  // Placeholder > 1; stabilized below.
    buf->size = appended_proj_bytes.size();
    model.buffers.push_back(std::move(buf));
    proj_weight_buf = static_cast<uint32_t>(model.buffers.size() - 1);
  } else {
    proj_weight_buf = add_float_buffer(proj_weights);
  }

  uint32_t proj_bias_buf = 0;
  if (opts.add_pooling_bias) {
    proj_bias_buf = add_float_buffer(std::vector<float>(opts.emb_dim, 0.05f));
  }
  const uint32_t axis_1_buf = add_int32_buffer({1});

  // Valid minimal FlexBuffer map `{}` bytes: `[0x00, 0x24, 0x01]`.
  const std::vector<uint8_t> empty_flex_map = {0x00, 0x24, 0x01};

  // First create one identity decomposition subgraph per signature seq_len so
  // STABLEHLO_COMPOSITE runs cleanly on CPU.
  std::vector<int32_t> decomp_sg_indices;
  for (size_t s = 0; s < opts.signature_keys.size(); ++s) {
    const int32_t seq_len = opts.seq_lengths[s];
    auto dsg = std::make_unique<tflite::SubGraphT>();
    dsg->name = absl::StrCat("rms_norm_decomp_", seq_len);
    auto make_dtensor = [&](std::vector<int32_t> shape, uint32_t buffer,
                            std::string name) -> int32_t {
      auto tensor = std::make_unique<tflite::TensorT>();
      tensor->shape = std::move(shape);
      tensor->type = tflite::TensorType_FLOAT32;
      tensor->buffer = buffer;
      tensor->name = std::move(name);
      dsg->tensors.push_back(std::move(tensor));
      return static_cast<int32_t>(dsg->tensors.size() - 1);
    };
    const int32_t dx = make_dtensor({1, seq_len, opts.hidden_dim}, 0, "x");
    const int32_t dz =
        make_dtensor({1, 1, opts.hidden_dim}, zeros_hidden_buf, "zero");
    const int32_t dy = make_dtensor({1, seq_len, opts.hidden_dim}, 0, "y");
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = op_add;
    op->inputs = {dx, dz};
    op->outputs = {dy};
    tflite::AddOptionsT add_opts;
    op->builtin_options.Set(add_opts);
    dsg->operators.push_back(std::move(op));
    dsg->inputs = {dx};
    dsg->outputs = {dy};
    decomp_sg_indices.push_back(static_cast<int32_t>(model.subgraphs.size()));
    model.subgraphs.push_back(std::move(dsg));
  }

  for (size_t s = 0; s < opts.signature_keys.size(); ++s) {
    const int32_t seq_len = opts.seq_lengths[s];
    const int32_t decomp_idx = decomp_sg_indices[s];
    const uint32_t shape_4d_buf =
        add_int32_buffer({1, seq_len, 1, opts.hidden_dim});

    auto sg = std::make_unique<tflite::SubGraphT>();
    sg->name = absl::StrCat("main_", seq_len);

    auto make_tensor = [&](std::vector<int32_t> shape, tflite::TensorType type,
                           uint32_t buffer, std::string name) -> int32_t {
      auto tensor = std::make_unique<tflite::TensorT>();
      tensor->shape = std::move(shape);
      tensor->type = type;
      tensor->buffer = buffer;
      tensor->name = std::move(name);
      sg->tensors.push_back(std::move(tensor));
      return static_cast<int32_t>(sg->tensors.size() - 1);
    };

    auto add_composite_op = [&](int32_t in_idx, int32_t out_idx) {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_composite;
      op->inputs = {in_idx};
      op->outputs = {out_idx};
      tflite::StableHLOCompositeOptionsT comp_opts;
      comp_opts.name = "odml.rms_norm.impl";
      comp_opts.decomposition_subgraph_index = decomp_idx;
      comp_opts.composite_attributes = empty_flex_map;
      comp_opts.composite_attributes_format =
          tflite::CustomOptionsFormat_FLEXBUFFERS;
      comp_opts.version = 1;
      op->builtin_options_2.Set(comp_opts);
      sg->operators.push_back(std::move(op));
    };

    const int32_t t_emb =
        make_tensor({1, seq_len, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                    0, "embeddings");
    const int32_t t_mask = make_tensor(
        {1, seq_len, 1}, tflite::TensorType_FLOAT32, 0, "input_mask");
    const int32_t t_shape_4d =
        make_tensor({4}, tflite::TensorType_INT32, shape_4d_buf, "shape_4d");
    const int32_t t_emb_4d =
        make_tensor({1, seq_len, 1, opts.hidden_dim},
                    tflite::TensorType_FLOAT32, 0, "emb_4d");
    const int32_t t_layer_bias =
        make_tensor({1, 1, opts.num_layers, opts.hidden_dim},
                    tflite::TensorType_FLOAT32, layer_bias_buf, "layer_bias");
    const int32_t t_ple =
        make_tensor({1, seq_len, opts.num_layers, opts.hidden_dim},
                    tflite::TensorType_FLOAT32, 0, "per_layer_all");

    {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_reshape;
      op->inputs = {t_emb, t_shape_4d};
      op->outputs = {t_emb_4d};
      tflite::ReshapeOptionsT reshape_opts;
      reshape_opts.new_shape = {1, seq_len, 1, opts.hidden_dim};
      op->builtin_options.Set(reshape_opts);
      sg->operators.push_back(std::move(op));
    }
    {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_add;
      op->inputs = {t_emb_4d, t_layer_bias};
      op->outputs = {t_ple};
      tflite::AddOptionsT add_opts;
      op->builtin_options.Set(add_opts);
      sg->operators.push_back(std::move(op));
    }

    std::vector<int32_t> unpack_outs;
    unpack_outs.reserve(opts.num_layers);
    for (int32_t li = 0; li < opts.num_layers; ++li) {
      unpack_outs.push_back(make_tensor({1, seq_len, opts.hidden_dim},
                                        tflite::TensorType_FLOAT32, 0,
                                        absl::StrCat("ple_", li)));
    }
    {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_unpack;
      op->inputs = {t_ple};
      op->outputs = unpack_outs;
      tflite::UnpackOptionsT unpack_opts;
      unpack_opts.num = opts.num_layers;
      unpack_opts.axis = 2;
      op->builtin_options.Set(unpack_opts);
      sg->operators.push_back(std::move(op));
    }

    const int32_t t_ones =
        make_tensor({1, 1, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                    ones_hidden_buf, "ones");
    const int32_t t_half =
        make_tensor({1, 1, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                    half_hidden_buf, "half");
    const int32_t t_ple_w =
        make_tensor({opts.hidden_dim, opts.hidden_dim},
                    tflite::TensorType_FLOAT32, ple_fc_buf, "ple_fc_w");

    int32_t cur = t_emb;
    for (int32_t li = 0; li < opts.num_layers; ++li) {
      int32_t cur_pre = cur;
      if (li > 0) {
        cur_pre = make_tensor({1, seq_len, opts.hidden_dim},
                              tflite::TensorType_FLOAT32, 0,
                              absl::StrCat("pre_norm_", li));
        add_composite_op(cur, cur_pre);
      }

      const int32_t mul1_out =
          make_tensor({1, seq_len, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                      0, absl::StrCat("ple_mul_", li));
      {
        auto op = std::make_unique<tflite::OperatorT>();
        op->opcode_index = op_mul;
        op->inputs = {unpack_outs[li], t_ones};
        op->outputs = {mul1_out};
        tflite::MulOptionsT mul_opts;
        op->builtin_options.Set(mul_opts);
        sg->operators.push_back(std::move(op));
      }

      const int32_t fc_out =
          make_tensor({1, seq_len, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                      0, absl::StrCat("ple_fc_", li));
      {
        auto op = std::make_unique<tflite::OperatorT>();
        op->opcode_index = op_fc;
        op->inputs = {mul1_out, t_ple_w, -1};
        op->outputs = {fc_out};
        tflite::FullyConnectedOptionsT fc_opts;
        fc_opts.keep_num_dims = true;
        op->builtin_options.Set(fc_opts);
        sg->operators.push_back(std::move(op));
      }

      const int32_t norm_out =
          make_tensor({1, seq_len, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                      0, absl::StrCat("ple_norm_", li));
      add_composite_op(fc_out, norm_out);

      const int32_t add_out =
          make_tensor({1, seq_len, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                      0, absl::StrCat("res_add_", li));
      {
        auto op = std::make_unique<tflite::OperatorT>();
        op->opcode_index = op_add;
        op->inputs = {cur_pre, norm_out};
        op->outputs = {add_out};
        tflite::AddOptionsT add_opts;
        op->builtin_options.Set(add_opts);
        sg->operators.push_back(std::move(op));
      }

      const int32_t skip_out =
          make_tensor({1, seq_len, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                      0, absl::StrCat("skip_out_", li));
      {
        auto op = std::make_unique<tflite::OperatorT>();
        op->opcode_index = op_mul;
        op->inputs = {add_out, t_half};
        op->outputs = {skip_out};
        tflite::MulOptionsT mul_opts;
        op->builtin_options.Set(mul_opts);
        sg->operators.push_back(std::move(op));
      }
      cur = skip_out;
    }

    const int32_t t_final_h =
        make_tensor({1, seq_len, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                    0, "final_norm_h");
    add_composite_op(cur, t_final_h);

    const int32_t t_masked =
        make_tensor({1, seq_len, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                    0, "h_masked");
    {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_mul;
      op->inputs = {t_final_h, t_mask};
      op->outputs = {t_masked};
      tflite::MulOptionsT mul_opts;
      op->builtin_options.Set(mul_opts);
      sg->operators.push_back(std::move(op));
    }

    const int32_t t_axis1 =
        make_tensor({1}, tflite::TensorType_INT32, axis_1_buf, "axis_1");
    const int32_t t_sum = make_tensor({1, opts.hidden_dim},
                                      tflite::TensorType_FLOAT32, 0, "h_sum");
    {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_sum;
      op->inputs = {t_masked, t_axis1};
      op->outputs = {t_sum};
      tflite::ReducerOptionsT red_opts;
      red_opts.keep_dims = false;
      op->builtin_options.Set(red_opts);
      sg->operators.push_back(std::move(op));
    }

    const int32_t t_mask_sum =
        make_tensor({1, 1}, tflite::TensorType_FLOAT32, 0, "mask_sum");
    {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_sum;
      op->inputs = {t_mask, t_axis1};
      op->outputs = {t_mask_sum};
      tflite::ReducerOptionsT red_opts;
      red_opts.keep_dims = false;
      op->builtin_options.Set(red_opts);
      sg->operators.push_back(std::move(op));
    }

    const int32_t t_mean = make_tensor({1, opts.hidden_dim},
                                       tflite::TensorType_FLOAT32, 0, "h_mean");
    {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_div;
      op->inputs = {t_sum, t_mask_sum};
      op->outputs = {t_mean};
      tflite::DivOptionsT div_opts;
      op->builtin_options.Set(div_opts);
      sg->operators.push_back(std::move(op));
    }

    const int32_t t_proj_w =
        make_tensor({opts.emb_dim, opts.hidden_dim}, tflite::TensorType_FLOAT32,
                    proj_weight_buf, "proj_weight");
    int32_t t_proj_bias = -1;
    if (opts.add_pooling_bias) {
      t_proj_bias = make_tensor({opts.emb_dim}, tflite::TensorType_FLOAT32,
                                proj_bias_buf, "proj_bias");
    }
    const int32_t t_encodings = make_tensor(
        {1, opts.emb_dim}, tflite::TensorType_FLOAT32, 0, "encodings");
    {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_fc;
      op->inputs = {t_mean, t_proj_w, t_proj_bias};
      op->outputs = {t_encodings};
      tflite::FullyConnectedOptionsT fc_opts;
      fc_opts.keep_num_dims = false;
      op->builtin_options.Set(fc_opts);
      sg->operators.push_back(std::move(op));
    }

    sg->inputs = {t_emb, t_mask};
    sg->outputs = {t_encodings};

    int32_t t_pre_tok = -1;
    if (opts.pre_expose_token_embeddings) {
      t_pre_tok =
          make_tensor({1, seq_len, opts.emb_dim}, tflite::TensorType_FLOAT32, 0,
                      "pre_token_embeddings");
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = op_fc;
      op->inputs = {t_final_h, t_proj_w, -1};
      op->outputs = {t_pre_tok};
      tflite::FullyConnectedOptionsT fc_opts;
      fc_opts.keep_num_dims = true;
      op->builtin_options.Set(fc_opts);
      sg->operators.push_back(std::move(op));
      sg->outputs.push_back(t_pre_tok);
    }

    const uint32_t sg_idx = static_cast<uint32_t>(model.subgraphs.size());
    model.subgraphs.push_back(std::move(sg));

    auto sig = std::make_unique<tflite::SignatureDefT>();
    sig->signature_key = opts.signature_keys[s];
    sig->subgraph_index = sg_idx;
    {
      auto m = std::make_unique<tflite::TensorMapT>();
      m->name = "embeddings";
      m->tensor_index = t_emb;
      sig->inputs.push_back(std::move(m));
    }
    {
      auto m = std::make_unique<tflite::TensorMapT>();
      m->name = "input_mask";
      m->tensor_index = t_mask;
      sig->inputs.push_back(std::move(m));
    }
    {
      auto m = std::make_unique<tflite::TensorMapT>();
      m->name = "encodings";
      m->tensor_index = t_encodings;
      sig->outputs.push_back(std::move(m));
    }
    if (opts.pre_expose_token_embeddings) {
      auto m = std::make_unique<tflite::TensorMapT>();
      m->name = "token_embeddings";
      m->tensor_index = t_pre_tok;
      sig->outputs.push_back(std::move(m));
    }
    model.signature_defs.push_back(std::move(sig));
  }

  if (opts.append_proj_weight_buffer) {
    constexpr size_t kAlign = 64;
    uint64_t new_start = 0;
    for (int iter = 0; iter < 3; ++iter) {
      flatbuffers::FlatBufferBuilder probe;
      tflite::FinishModelBuffer(probe, tflite::Model::Pack(probe, &model));
      new_start = (probe.GetSize() + kAlign - 1) / kAlign * kAlign;
      model.buffers[proj_weight_buf]->offset = new_start;
    }
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
    std::vector<uint8_t> out(
        static_cast<size_t>(new_start) + appended_proj_bytes.size(), 0);
    std::memcpy(out.data(), builder.GetBufferPointer(), builder.GetSize());
    std::memcpy(out.data() + new_start, appended_proj_bytes.data(),
                appended_proj_bytes.size());
    return out;
  }

  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     PerTokenInjection_PooledEncodingsBitIdenticalAndMeanMatchesAnalytical) {
  constexpr int32_t kHiddenDim = 4;
  constexpr int32_t kEmbDim = 6;
  auto embedder_buf = BuildCastAddEmbedderBuffer(kHiddenDim);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  SyntheticEncoderOptions enc_opts;
  enc_opts.signature_keys = {"encoder_4", "encoder_8"};
  enc_opts.seq_lengths = {4, 8};
  enc_opts.hidden_dim = kHiddenDim;
  enc_opts.emb_dim = kEmbDim;
  enc_opts.num_layers = 8;
  auto encoder_buf = BuildSyntheticEgv2LikeEncoderBuffer(enc_opts);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  // Unpatched baseline executor.
  ASSERT_OK_AND_ASSIGN(
      EmbeddingExecutorSettings unpatched_settings,
      EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  unpatched_settings.SetNumThreads(1);
  ASSERT_OK_AND_ASSIGN(auto unpatched_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           unpatched_settings, env,
                           std::make_unique<FakeEmbeddingModelResources>(
                               &embedder_model, &encoder_model)));

  // Patched executor with `token_embeddings` and `layer_02` + `layer_06` taps.
  ASSERT_OK_AND_ASSIGN(
      EmbeddingExecutorSettings patched_settings,
      EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  patched_settings.SetNumThreads(1);
  patched_settings.SetPerTokenOutputs(PerTokenOutputsConfig{
      .token_embeddings = true, .hidden_state_layers = {6, 2, 6}});
  ASSERT_OK_AND_ASSIGN(auto patched_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           patched_settings, env,
                           std::make_unique<FakeEmbeddingModelResources>(
                               &embedder_model, &encoder_model,
                               absl::string_view(reinterpret_cast<const char*>(
                                                     encoder_buf.data()),
                                                 encoder_buf.size()))));

  // Feed 3 active tokens into the 4-token signature (1 padded slot).
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[] = {1, 3, 7};
  auto make_inputs = [&]() {
    auto buf = TensorBuffer::CreateFromHostMemory(
        env, RankedTensorType(ElementType::Int32, Layout(Dimensions({3}))),
        token_data, sizeof(token_data));
    return ExecutorInputs(ExecutorTextData(std::move(*buf)), std::nullopt,
                          std::nullopt);
  };

  ASSERT_OK_AND_ASSIGN(EmbeddingOutput baseline_out,
                       unpatched_executor->ComputeEmbedding(make_inputs()));
  ASSERT_OK_AND_ASSIGN(
      EmbeddingOutput patched_out,
      patched_executor->ComputeEmbedding(
          make_inputs(),
          ComputeEmbeddingOptions{.return_token_embeddings = true,
                                  .return_hidden_state_layers = {6, 2}}));

  // Pooled `encodings` must be bit-identical between unpatched and patched.
  EXPECT_THAT(patched_out.embedding, ElementsAreArray(baseline_out.embedding));

  // Only the 3 active tokens are returned (signature padding is stripped).
  EXPECT_EQ(patched_out.token_embedding_dim, kEmbDim);
  ASSERT_EQ(patched_out.token_embeddings.size(), 3 * kEmbDim);

  // Because pooling is bias-free linear projection of masked mean(h), the mean
  // of active `token_embeddings` rows must match `embedding` to FP tolerance.
  for (int32_t d = 0; d < kEmbDim; ++d) {
    const float mean_tok = (patched_out.token_embeddings[0 * kEmbDim + d] +
                            patched_out.token_embeddings[1 * kEmbDim + d] +
                            patched_out.token_embeddings[2 * kEmbDim + d]) /
                           3.0f;
    EXPECT_NEAR(mean_tok, patched_out.embedding[d], 1e-5f);
  }
  // Distinct tokens produce distinct token embedding rows.
  EXPECT_NE(patched_out.token_embeddings[0],
            patched_out.token_embeddings[kEmbDim]);

  // Verify `layer_06` and `layer_02` hidden state taps (`[3, kHiddenDim]`).
  EXPECT_EQ(patched_out.hidden_state_dim, kHiddenDim);
  ASSERT_TRUE(patched_out.hidden_states.contains(6));
  ASSERT_TRUE(patched_out.hidden_states.contains(2));
  ASSERT_EQ(patched_out.hidden_states.at(6).size(), 3 * kHiddenDim);
  ASSERT_EQ(patched_out.hidden_states.at(2).size(), 3 * kHiddenDim);
  EXPECT_NE(patched_out.hidden_states.at(6), patched_out.hidden_states.at(2));

  // Default options on the patched executor return empty per-token fields and
  // bit-identical pooled embedding.
  ASSERT_OK_AND_ASSIGN(EmbeddingOutput patched_default_out,
                       patched_executor->ComputeEmbedding(make_inputs()));
  EXPECT_TRUE(patched_default_out.token_embeddings.empty());
  EXPECT_TRUE(patched_default_out.hidden_states.empty());
  EXPECT_THAT(patched_default_out.embedding,
              ElementsAreArray(baseline_out.embedding));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     PerTokenInjection_ChunkAndAverageConcatenatesActiveTokensAcrossChunks) {
  constexpr int32_t kHiddenDim = 4;
  constexpr int32_t kEmbDim = 6;
  auto embedder_buf = BuildCastAddEmbedderBuffer(kHiddenDim);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  SyntheticEncoderOptions enc_opts;
  enc_opts.signature_keys = {"encoder_4"};
  enc_opts.seq_lengths = {4};
  enc_opts.hidden_dim = kHiddenDim;
  enc_opts.emb_dim = kEmbDim;
  enc_opts.num_layers = 8;
  auto encoder_buf = BuildSyntheticEgv2LikeEncoderBuffer(enc_opts);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(
      EmbeddingExecutorSettings settings,
      EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  settings.SetPerTokenOutputs(PerTokenOutputsConfig{
      .token_embeddings = true, .hidden_state_layers = {6}});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env,
                           std::make_unique<FakeEmbeddingModelResources>(
                               &embedder_model, &encoder_model,
                               absl::string_view(reinterpret_cast<const char*>(
                                                     encoder_buf.data()),
                                                 encoder_buf.size()))));

  // 6 tokens -> chunk 0 has 4 tokens, chunk 1 has 2 active tokens (padded to
  // 4).
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT)
      int32_t token_data[] = {1, 2, 3, 4, 5, 6};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({6}))),
          token_data, sizeof(token_data)));
  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::nullopt, std::nullopt);

  ASSERT_OK_AND_ASSIGN(
      EmbeddingOutput out,
      executor->ComputeEmbedding(
          inputs, ComputeEmbeddingOptions{
                      .input_overflow_strategy =
                          InputOverflowStrategy::kChunkAndAverage,
                      .return_token_embeddings = true,
                      .return_hidden_state_layers = {6},
                  }));
  EXPECT_EQ(out.num_chunks, 2);
  EXPECT_EQ(out.token_embedding_dim, kEmbDim);
  EXPECT_EQ(out.token_embeddings.size(), 6 * kEmbDim);
  EXPECT_EQ(out.hidden_state_dim, kHiddenDim);
  ASSERT_TRUE(out.hidden_states.contains(6));
  EXPECT_EQ(out.hidden_states.at(6).size(), 6 * kHiddenDim);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     PerTokenInjection_RelocatesAppendedBufferOffsetsAndCreatesDistinctCache) {
  constexpr int32_t kHiddenDim = 4;
  constexpr int32_t kEmbDim = 6;
  auto embedder_buf = BuildCastAddEmbedderBuffer(kHiddenDim);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  SyntheticEncoderOptions enc_opts;
  enc_opts.signature_keys = {"encoder_4"};
  enc_opts.seq_lengths = {4};
  enc_opts.hidden_dim = kHiddenDim;
  enc_opts.emb_dim = kEmbDim;
  enc_opts.num_layers = 8;
  enc_opts.append_proj_weight_buffer = true;
  auto encoder_buf = BuildSyntheticEgv2LikeEncoderBuffer(enc_opts);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  const std::string cache_dir =
      absl::StrCat(::testing::TempDir(), "/per_token_xnn_cache_test");
  std::filesystem::remove_all(cache_dir);
  std::filesystem::create_directories(cache_dir);

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("/tmp/synth_egv2.litertlm"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  ASSERT_OK_AND_ASSIGN(
      EmbeddingExecutorSettings base_settings,
      EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  base_settings.SetNumThreads(1);
  base_settings.SetCacheDir(cache_dir);
  ASSERT_OK_AND_ASSIGN(auto base_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           base_settings, env,
                           std::make_unique<FakeEmbeddingModelResources>(
                               &embedder_model, &encoder_model)));

  ASSERT_OK_AND_ASSIGN(
      EmbeddingExecutorSettings patched_settings,
      EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  patched_settings.SetNumThreads(1);
  patched_settings.SetCacheDir(cache_dir);
  patched_settings.SetPerTokenOutputs(PerTokenOutputsConfig{
      .token_embeddings = true, .hidden_state_layers = {6}});
  ASSERT_OK_AND_ASSIGN(auto patched_executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           patched_settings, env,
                           std::make_unique<FakeEmbeddingModelResources>(
                               &embedder_model, &encoder_model,
                               absl::string_view(reinterpret_cast<const char*>(
                                                     encoder_buf.data()),
                                                 encoder_buf.size()))));

  // Verify two distinct XNNPACK cache files exist (unpatched vs patched).
  int xnnpack_cache_files = 0;
  bool found_patched_tag = false;
  for (const auto& entry : std::filesystem::directory_iterator(cache_dir)) {
    const std::string filename = entry.path().filename().string();
    if (absl::StrContains(filename, ".xnnpack_cache")) {
      ++xnnpack_cache_files;
      if (absl::StrContains(filename, ".pt_te_l06.")) {
        found_patched_tag = true;
      }
    }
  }
  EXPECT_EQ(xnnpack_cache_files, 2);
  EXPECT_TRUE(found_patched_tag);

  // Verify relocated appended projection weights produce bit-identical pooled
  // output and valid non-zero token embeddings.
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[] = {2, 5};
  auto make_inputs = [&]() {
    auto buf = TensorBuffer::CreateFromHostMemory(
        env, RankedTensorType(ElementType::Int32, Layout(Dimensions({2}))),
        token_data, sizeof(token_data));
    return ExecutorInputs(ExecutorTextData(std::move(*buf)), std::nullopt,
                          std::nullopt);
  };
  ASSERT_OK_AND_ASSIGN(EmbeddingOutput base_out,
                       base_executor->ComputeEmbedding(make_inputs()));
  ASSERT_OK_AND_ASSIGN(
      EmbeddingOutput patched_out,
      patched_executor->ComputeEmbedding(
          make_inputs(),
          ComputeEmbeddingOptions{.return_token_embeddings = true,
                                  .return_hidden_state_layers = {6}}));
  EXPECT_THAT(patched_out.embedding, ElementsAreArray(base_out.embedding));
  ASSERT_EQ(patched_out.token_embeddings.size(), 2 * kEmbDim);
  EXPECT_GT(std::abs(patched_out.token_embeddings[0]), 0.0f);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     PerTokenInjection_SkipsRewriteWhenOutputsAlreadyExist) {
  constexpr int32_t kHiddenDim = 4;
  constexpr int32_t kEmbDim = 6;
  auto embedder_buf = BuildCastAddEmbedderBuffer(kHiddenDim);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  SyntheticEncoderOptions enc_opts;
  enc_opts.pre_expose_token_embeddings = true;
  auto encoder_buf = BuildSyntheticEgv2LikeEncoderBuffer(enc_opts);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  ASSERT_OK_AND_ASSIGN(
      EmbeddingExecutorSettings settings,
      EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  settings.SetPerTokenOutputs(PerTokenOutputsConfig{.token_embeddings = true});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  // Notice `encoder_buffer` is NOT supplied to `FakeEmbeddingModelResources`:
  // if `CompileTextEncoder` tried to rewrite, `GetTFLiteModelBuffer` would fail
  // with `UnimplementedError`.
  ASSERT_OK_AND_ASSIGN(auto executor,
                       EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env,
                           std::make_unique<FakeEmbeddingModelResources>(
                               &embedder_model, &encoder_model)));

  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[] = {3, 4};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({2}))),
          token_data, sizeof(token_data)));
  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::nullopt, std::nullopt);
  ASSERT_OK_AND_ASSIGN(
      EmbeddingOutput out,
      executor->ComputeEmbedding(
          inputs, ComputeEmbeddingOptions{.return_token_embeddings = true}));
  EXPECT_EQ(out.token_embedding_dim, kEmbDim);
  EXPECT_EQ(out.token_embeddings.size(), 2 * kEmbDim);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     PerTokenInjection_ValidationAndMissingOutputErrors) {
  auto embedder_buf = BuildCastAddEmbedderBuffer(4);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto embedder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               embedder_buf.data(), embedder_buf.size())));

  SyntheticEncoderOptions enc_opts;
  enc_opts.num_layers = 8;
  auto encoder_buf = BuildSyntheticEgv2LikeEncoderBuffer(enc_opts);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto encoder_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                              encoder_buf.data(), encoder_buf.size())));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("dummy_model_path"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  // 1. Out-of-range layer index at load time -> InvalidArgumentError.
  {
    ASSERT_OK_AND_ASSIGN(
        EmbeddingExecutorSettings bad_layer_settings,
        EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
    bad_layer_settings.SetPerTokenOutputs(
        PerTokenOutputsConfig{.hidden_state_layers = {8}});
    EXPECT_THAT(EmbeddingLiteRtCompiledModelExecutor::Create(
                    bad_layer_settings, env,
                    std::make_unique<FakeEmbeddingModelResources>(
                        &embedder_model, &encoder_model,
                        absl::string_view(
                            reinterpret_cast<const char*>(encoder_buf.data()),
                            encoder_buf.size()))),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("out of range")));
  }

  // 2. Biased pooling projection -> FailedPreconditionError.
  {
    SyntheticEncoderOptions biased_opts = enc_opts;
    biased_opts.add_pooling_bias = true;
    auto biased_buf = BuildSyntheticEgv2LikeEncoderBuffer(biased_opts);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto biased_model, Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
                               biased_buf.data(), biased_buf.size())));
    ASSERT_OK_AND_ASSIGN(
        EmbeddingExecutorSettings settings,
        EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
    settings.SetPerTokenOutputs(
        PerTokenOutputsConfig{.token_embeddings = true});
    EXPECT_THAT(EmbeddingLiteRtCompiledModelExecutor::Create(
                    settings, env,
                    std::make_unique<FakeEmbeddingModelResources>(
                        &embedder_model, &biased_model,
                        absl::string_view(
                            reinterpret_cast<const char*>(biased_buf.data()),
                            biased_buf.size()))),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("bias-free")));
  }

  // 3. Requesting un-injected outputs at inference time -> FailedPrecondition
  //    (and negative layer -> InvalidArgument).
  {
    ASSERT_OK_AND_ASSIGN(
        EmbeddingExecutorSettings unpatched_settings,
        EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
    ASSERT_OK_AND_ASSIGN(auto executor,
                         EmbeddingLiteRtCompiledModelExecutor::Create(
                             unpatched_settings, env,
                             std::make_unique<FakeEmbeddingModelResources>(
                                 &embedder_model, &encoder_model)));

    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[] = {1};
    auto make_inputs = [&]() {
      auto buf = TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Int32, Layout(Dimensions({1}))),
          token_data, sizeof(token_data));
      return ExecutorInputs(ExecutorTextData(std::move(*buf)), std::nullopt,
                            std::nullopt);
    };
    EXPECT_THAT(executor->ComputeEmbedding(
                    make_inputs(),
                    ComputeEmbeddingOptions{.return_token_embeddings = true}),
                StatusIs(absl::StatusCode::kFailedPrecondition));
    EXPECT_THAT(executor->ComputeEmbedding(
                    make_inputs(),
                    ComputeEmbeddingOptions{.return_hidden_state_layers = {6}}),
                StatusIs(absl::StatusCode::kFailedPrecondition));
    EXPECT_THAT(
        executor->ComputeEmbedding(
            make_inputs(),
            ComputeEmbeddingOptions{.return_hidden_state_layers = {-1}}),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 4. Truncated FlatBuffer with valid "TFL3" magic -> FailedPreconditionError
  //    (caught by flatbuffers::Verifier instead of reading out-of-bounds).
  {
    ASSERT_OK_AND_ASSIGN(
        EmbeddingExecutorSettings settings,
        EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
    settings.SetPerTokenOutputs(
        PerTokenOutputsConfig{.token_embeddings = true});
    constexpr absl::string_view kTruncatedTfl3 = "XXXXTFL3\x7f\x7f\x7f\x7f";
    EXPECT_THAT(EmbeddingLiteRtCompiledModelExecutor::Create(
                    settings, env,
                    std::make_unique<FakeEmbeddingModelResources>(
                        &embedder_model, &encoder_model, kTruncatedTfl3)),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("Invalid TFLite FlatBuffer")));
  }

  // 5. Appended buffer with out-of-bounds `offset + size` (even when minimum
  //    `data_start` is within bounds) -> FailedPreconditionError.
  {
    std::unique_ptr<tflite::ModelT> unpacked =
        tflite::UnPackModel(encoder_buf.data());
    auto ok_appended = std::make_unique<tflite::BufferT>();
    ok_appended->offset = 64;
    ok_appended->size = 16;
    unpacked->buffers.push_back(std::move(ok_appended));
    auto oob_appended = std::make_unique<tflite::BufferT>();
    oob_appended->offset = 64;
    oob_appended->size = 1ULL << 30;
    unpacked->buffers.push_back(std::move(oob_appended));
    flatbuffers::FlatBufferBuilder fbb;
    tflite::FinishModelBuffer(fbb, tflite::Model::Pack(fbb, unpacked.get()));
    std::vector<uint8_t> bad_appended_buf(fbb.GetSize() + 128, 0);
    std::memcpy(bad_appended_buf.data(), fbb.GetBufferPointer(), fbb.GetSize());

    ASSERT_OK_AND_ASSIGN(
        EmbeddingExecutorSettings settings,
        EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
    settings.SetPerTokenOutputs(
        PerTokenOutputsConfig{.token_embeddings = true});
    EXPECT_THAT(EmbeddingLiteRtCompiledModelExecutor::Create(
                    settings, env,
                    std::make_unique<FakeEmbeddingModelResources>(
                        &embedder_model, &encoder_model,
                        absl::string_view(reinterpret_cast<const char*>(
                                              bad_appended_buf.data()),
                                          bad_appended_buf.size()))),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("out-of-bounds appended buffer offset")));
  }

  // 6. Selected signatures: unselected malformed signature is ignored while
  //    selected signature is patched and executes cleanly.
  {
    SyntheticEncoderOptions multi_opts;
    multi_opts.signature_keys = {"encoder_4", "encoder_8"};
    multi_opts.seq_lengths = {4, 8};
    auto multi_buf = BuildSyntheticEgv2LikeEncoderBuffer(multi_opts);
    std::unique_ptr<tflite::ModelT> unpacked =
        tflite::UnPackModel(multi_buf.data());
    // Corrupt unselected `encoder_8` subgraph by clearing its operators so
    // patching it would fail if `selected_signatures` were not respected.
    unpacked->subgraphs[unpacked->signature_defs[1]->subgraph_index]
        ->operators.clear();
    flatbuffers::FlatBufferBuilder fbb;
    tflite::FinishModelBuffer(fbb, tflite::Model::Pack(fbb, unpacked.get()));
    std::vector<uint8_t> selective_buf(fbb.GetBufferPointer(),
                                       fbb.GetBufferPointer() + fbb.GetSize());
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto selective_model,
        Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
            selective_buf.data(), selective_buf.size())));

    ASSERT_OK_AND_ASSIGN(
        EmbeddingExecutorSettings settings,
        EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
    settings.SetSelectedSignatures({"encoder_4"});
    settings.SetPerTokenOutputs(PerTokenOutputsConfig{
        .token_embeddings = true, .hidden_state_layers = {6}});
    ASSERT_OK_AND_ASSIGN(
        auto executor, EmbeddingLiteRtCompiledModelExecutor::Create(
                           settings, env,
                           std::make_unique<FakeEmbeddingModelResources>(
                               &embedder_model, &selective_model,
                               absl::string_view(reinterpret_cast<const char*>(
                                                     selective_buf.data()),
                                                 selective_buf.size()))));
    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t tok[] = {2, 4};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto tok_buf,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({2}))),
            tok, sizeof(tok)));
    ASSERT_OK_AND_ASSIGN(
        EmbeddingOutput out,
        executor->ComputeEmbedding(
            ExecutorInputs(ExecutorTextData(std::move(tok_buf)), std::nullopt,
                           std::nullopt),
            ComputeEmbeddingOptions{.return_token_embeddings = true,
                                    .return_hidden_state_layers = {6}}));
    EXPECT_EQ(out.token_embeddings.size(), 2 * multi_opts.emb_dim);
    EXPECT_EQ(out.hidden_states.at(6).size(), 2 * multi_opts.hidden_dim);
  }
}

}  // namespace
}  // namespace litert::lm
