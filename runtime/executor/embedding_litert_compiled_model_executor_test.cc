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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/run_options_test_util.h"
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
  using ModelResources::HasTFLiteModel;

  FakeEmbeddingModelResources(const litert::Model* embedder,
                              const litert::Model* encoder,
                              const litert::Model* per_layer_embedder = nullptr)
      : embedder_(embedder),
        encoder_(encoder),
        per_layer_embedder_(per_layer_embedder) {}

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
    if (model_type_str == "tf_lite_per_layer_embedder") {
      return per_layer_embedder_ != nullptr;
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
    if (model_type_str == "tf_lite_per_layer_embedder") {
      if (per_layer_embedder_ != nullptr) return per_layer_embedder_;
      return absl::NotFoundError("kTfLitePerLayerEmbedder model not found.");
    }
    return absl::NotFoundError("Model type not found.");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      proto::LlmMetadata::TfLiteModelType model_type) override {
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
  const litert::Model* per_layer_embedder_;
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

TEST(EmbeddingLiteRtCompiledModelExecutorTest, ComputeEmbeddingWithRunOptions) {
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
  auto make_inputs = [&]() -> absl::StatusOr<ExecutorInputs> {
    LITERT_ASSIGN_OR_RETURN(
        auto token_ids_buffer,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({1}))),
            token_data, sizeof(token_data)));
    return ExecutorInputs(ExecutorTextData(std::move(token_ids_buffer)),
                          std::nullopt, std::nullopt);
  };
  auto compute = [&]() -> absl::StatusOr<std::vector<float>> {
    ABSL_ASSIGN_OR_RETURN(ExecutorInputs inputs, make_inputs());
    ABSL_ASSIGN_OR_RETURN(EmbeddingOutput output,
                          embedding_executor->ComputeEmbedding(inputs));
    return output.embedding;
  };
  ASSERT_OK_AND_ASSIGN(std::vector<float> expected, compute());

  // The CPU model ignores the run options, so the results don't change. Every
  // text encoder run gets new options.
  int num_calls = 0;
  ASSERT_OK(
      embedding_executor->SetRunOptions(CountingRunOptionsFactory(&num_calls)));
  EXPECT_THAT(compute(), IsOkAndHolds(expected));
  EXPECT_EQ(num_calls, 1);
  EXPECT_THAT(compute(), IsOkAndHolds(expected));
  EXPECT_EQ(num_calls, 2);

  // The model has no batched signature, so a batch runs its inputs one at a
  // time, each with new options. The batched run is covered by
  // ComputeEmbeddingBatch_BatchedSignatureUsesRunOptions.
  std::vector<ExecutorInputs> batch_inputs;
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK_AND_ASSIGN(ExecutorInputs inputs, make_inputs());
    batch_inputs.push_back(std::move(inputs));
  }
  ASSERT_OK_AND_ASSIGN(auto batch_outputs,
                       embedding_executor->ComputeEmbeddingBatch(batch_inputs));
  EXPECT_EQ(batch_outputs.size(), 2);
  EXPECT_EQ(num_calls, 4);

  // A failure to create the options fails the call.
  ASSERT_OK(embedding_executor->SetRunOptions(
      FailingRunOptionsFactory(absl::InvalidArgumentError("bad options"))));
  EXPECT_THAT(compute(),
              StatusIs(absl::StatusCode::kInvalidArgument, "bad options"));

  // An empty factory reverts to the compile-time options.
  ASSERT_OK(embedding_executor->SetRunOptions(nullptr));
  const int num_calls_before_reset = num_calls;
  EXPECT_THAT(compute(), IsOkAndHolds(expected));
  EXPECT_EQ(num_calls, num_calls_before_reset);
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

// Builds an embedder that maps an int32 token id `t` to a float embedding of
// shape [1, 1, embedding_dim] filled with `float(t) + offset`.
std::vector<uint8_t> BuildCastEmbedderModelBuffer(int embedding_dim,
                                                  float offset = 0.0f) {
  flatbuffers::FlatBufferBuilder builder;

  auto cast_opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_CAST);
  auto add_opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({cast_opcode, add_opcode});

  const std::vector<float> offsets(embedding_dim, offset);
  const auto* offset_bytes = reinterpret_cast<const uint8_t*>(offsets.data());
  auto empty_buffer = tflite::CreateBuffer(builder);
  auto offsets_buffer = tflite::CreateBuffer(
      builder,
      builder.CreateVector(offset_bytes, offsets.size() * sizeof(float)));
  auto buffers_vec = builder.CreateVector({empty_buffer, offsets_buffer});

  const std::vector<int32_t> token_dims = {1};
  const std::vector<int32_t> output_dims = {1, 1, embedding_dim};
  auto token_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(token_dims), tflite::TensorType_INT32,
      /*buffer=*/0, builder.CreateString("token"));
  auto token_float_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(token_dims), tflite::TensorType_FLOAT32,
      /*buffer=*/0, builder.CreateString("token_float"));
  auto offsets_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(output_dims), tflite::TensorType_FLOAT32,
      /*buffer=*/1, builder.CreateString("offsets"));
  auto output_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(output_dims), tflite::TensorType_FLOAT32,
      /*buffer=*/0, builder.CreateString("embedding"));
  auto tensors_vec = builder.CreateVector(
      {token_tensor, token_float_tensor, offsets_tensor, output_tensor});

  auto cast_op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                        builder.CreateVector<int32_t>({0}),
                                        builder.CreateVector<int32_t>({1}));
  auto add_op = tflite::CreateOperator(builder, /*opcode_index=*/1,
                                       builder.CreateVector<int32_t>({1, 2}),
                                       builder.CreateVector<int32_t>({3}));
  auto ops_vec = builder.CreateVector({cast_op, add_op});

  auto subgraph = tflite::CreateSubGraph(builder, tensors_vec,
                                         builder.CreateVector<int32_t>({0}),
                                         builder.CreateVector<int32_t>({3}),
                                         ops_vec, builder.CreateString("main"));
  auto subgraphs_vec = builder.CreateVector({subgraph});

  auto input_map =
      tflite::CreateTensorMap(builder, builder.CreateString("token"), 0);
  auto output_map =
      tflite::CreateTensorMap(builder, builder.CreateString("embedding"), 3);
  auto sig_def = tflite::CreateSignatureDef(
      builder, builder.CreateVector({input_map}),
      builder.CreateVector({output_map}), builder.CreateString("embedder"),
      /*subgraph_index=*/0);
  auto sig_defs_vec = builder.CreateVector({sig_def});

  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, subgraphs_vec,
      builder.CreateString("cast_embedder_model"), buffers_vec,
      /*metadata_buffer=*/0, /*metadata=*/0, sig_defs_vec);
  tflite::FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

// Options of `BuildBatchedSumEncoderModelBuffer` and `CreateSumModels`.
struct SumEncoderOptions {
  // 3 for an `input_mask` of shape [B, T, 1], 2 for [B, T].
  int mask_rank = 3;
  // Adds a `per_layer_embeddings` input of shape [B, T, embedding_dim], which
  // is added to the embeddings.
  bool with_per_layer_input = false;
  // Adds the `per_layer_embeddings` input only to batched (B > 1) signatures.
  bool with_batched_per_layer_input = false;
  // Gives the batched signatures (B > 1) an `input_mask` with B + 1 rows, so
  // that their tensors fail to allocate.
  bool broken_batched_signatures = false;
  // Gives the batched signatures (B > 1) an `input_mask` of shape
  // [B, T, embedding_dim], which runs but is not a valid mask.
  bool batched_mask_per_embedding_element = false;
  // Makes the batched signatures (B > 1) output the unpooled [B, T, D]
  // masked embeddings.
  bool batched_unpooled_output = false;
  // Gives every signature an empty list of outputs.
  bool no_outputs = false;
  // Offset that the cast embedder adds to every token, so that the default
  // (token 0) embedding used for padding is `embedder_offset`.
  float embedder_offset = 0.0f;
  // Also creates a per-layer embedder (a cast embedder without offset).
  bool with_per_layer_embedder = false;
};

// Builds a text encoder with one signature `encoder_{B}x{T}` per
// (batch size, sequence length) pair. Each signature computes
// `output[b, :] = sum_t(embeddings[b, t, :] + input_mask[b, t])`, i.e. the
// sum of the embeddings plus the number of unmasked tokens of each row, which
// does not depend on the padded length as long as padding embeddings are zero.
// A non-positive batch size or sequence length is dynamic (and 1 at run time).
std::vector<uint8_t> BuildBatchedSumEncoderModelBuffer(
    const std::vector<std::pair<int32_t, int32_t>>& batch_and_seq_lengths,
    int embedding_dim, const SumEncoderOptions& options = {}) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcodes_vec = builder.CreateVector({
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD),
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_SUM),
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_RESHAPE),
  });

  // Buffer 1 holds the reduction axis (1) of SUM.
  const int32_t axis = 1;
  const auto* axis_bytes = reinterpret_cast<const uint8_t*>(&axis);
  auto empty_buffer = tflite::CreateBuffer(builder);
  auto axis_buffer = tflite::CreateBuffer(
      builder, builder.CreateVector(axis_bytes, sizeof(axis)));
  auto buffers_vec = builder.CreateVector({empty_buffer, axis_buffer});

  // Tensor indices.
  constexpr int32_t kEmbeddings = 0, kMask = 1, kMasked = 2, kAxis = 3,
                    kEncodings = 4, kMask3d = 5, kPerLayer = 6,
                    kEmbeddingsWithMask = 7;

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;
  for (size_t i = 0; i < batch_and_seq_lengths.size(); ++i) {
    const auto [batch, seq_len] = batch_and_seq_lengths[i];
    const bool batched = batch > 1;
    const int32_t runtime_batch = batch <= 0 ? 1 : batch;
    const int32_t runtime_seq_len = seq_len <= 0 ? 1 : seq_len;
    const int32_t mask_batch =
        options.broken_batched_signatures && batched ? batch + 1 : batch;
    const int32_t runtime_mask_batch = mask_batch <= 0 ? 1 : mask_batch;
    const int32_t mask_last_dim =
        options.batched_mask_per_embedding_element && batched ? embedding_dim
                                                              : 1;
    const bool has_per_layer_input =
        options.with_per_layer_input ||
        (options.with_batched_per_layer_input && batched);
    const bool unpooled_output = options.batched_unpooled_output && batched;

    // Creates a float tensor of shape `dims`, where the first two dimensions
    // are the (possibly dynamic) `logical_batch` and `logical_seq_len`.
    auto float_tensor = [&](const std::vector<int32_t>& dims,
                            absl::string_view name, int32_t logical_batch,
                            std::optional<int32_t> logical_seq_len =
                                std::nullopt) {
      std::vector<int32_t> shape_signature = dims;
      bool dynamic = false;
      if (logical_batch <= 0) {
        shape_signature[0] = logical_batch;
        dynamic = true;
      }
      if (logical_seq_len.has_value() && *logical_seq_len <= 0) {
        shape_signature[1] = *logical_seq_len;
        dynamic = true;
      }
      return tflite::CreateTensor(
          builder, builder.CreateVector(dims), tflite::TensorType_FLOAT32,
          /*buffer=*/0, builder.CreateString(std::string(name)),
          /*quantization=*/0, /*is_variable=*/false, /*sparsity=*/0,
          dynamic ? builder.CreateVector(shape_signature) : 0,
          /*has_rank=*/dynamic);
    };
    const std::vector<int32_t> embeddings_dims = {
        runtime_batch, runtime_seq_len, embedding_dim};
    const std::vector<int32_t> mask_3d_dims = {runtime_mask_batch,
                                               runtime_seq_len, mask_last_dim};
    const std::vector<int32_t> mask_dims =
        options.mask_rank == 2
            ? std::vector<int32_t>{runtime_mask_batch, runtime_seq_len}
            : mask_3d_dims;
    auto tensors_vec = builder.CreateVector({
        float_tensor(embeddings_dims, "embeddings", batch, seq_len),
        float_tensor(mask_dims, "input_mask", mask_batch, seq_len),
        float_tensor(embeddings_dims, "masked", batch, seq_len),
        tflite::CreateTensor(builder, builder.CreateVector<int32_t>({1}),
                             tflite::TensorType_INT32, /*buffer=*/1,
                             builder.CreateString("axis")),
        float_tensor({runtime_batch, embedding_dim}, "encodings", batch),
        float_tensor(mask_3d_dims, "input_mask_3d", mask_batch, seq_len),
        float_tensor(embeddings_dims, "per_layer_embeddings", batch, seq_len),
        float_tensor(embeddings_dims, "embeddings_with_mask", batch, seq_len),
    });

    std::vector<flatbuffers::Offset<tflite::Operator>> ops;
    int32_t mask_for_add = kMask;
    if (options.mask_rank == 2) {
      ops.push_back(tflite::CreateOperator(
          builder, /*opcode_index=*/2, builder.CreateVector<int32_t>({kMask}),
          builder.CreateVector<int32_t>({kMask3d}),
          tflite::BuiltinOptions_ReshapeOptions,
          tflite::CreateReshapeOptions(builder,
                                       builder.CreateVector(mask_3d_dims))
              .Union()));
      mask_for_add = kMask3d;
    }
    // masked = embeddings + mask (+ per_layer_embeddings).
    ops.push_back(tflite::CreateOperator(
        builder, /*opcode_index=*/0,
        builder.CreateVector<int32_t>({kEmbeddings, mask_for_add}),
        builder.CreateVector<int32_t>(
            {has_per_layer_input ? kEmbeddingsWithMask : kMasked})));
    if (has_per_layer_input) {
      ops.push_back(tflite::CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector<int32_t>({kEmbeddingsWithMask, kPerLayer}),
          builder.CreateVector<int32_t>({kMasked})));
    }
    const int32_t output = unpooled_output ? kMasked : kEncodings;
    if (!unpooled_output) {
      ops.push_back(tflite::CreateOperator(
          builder, /*opcode_index=*/1,
          builder.CreateVector<int32_t>({kMasked, kAxis}),
          builder.CreateVector<int32_t>({kEncodings}),
          tflite::BuiltinOptions_ReducerOptions,
          tflite::CreateReducerOptions(builder, /*keep_dims=*/false).Union()));
    }

    std::vector<int32_t> sg_inputs = {kEmbeddings, kMask};
    std::vector<flatbuffers::Offset<tflite::TensorMap>> input_maps = {
        tflite::CreateTensorMap(builder, builder.CreateString("embeddings"),
                                kEmbeddings),
        tflite::CreateTensorMap(builder, builder.CreateString("input_mask"),
                                kMask),
    };
    if (has_per_layer_input) {
      sg_inputs.push_back(kPerLayer);
      input_maps.push_back(tflite::CreateTensorMap(
          builder, builder.CreateString("per_layer_embeddings"), kPerLayer));
    }

    subgraphs.push_back(tflite::CreateSubGraph(
        builder, tensors_vec, builder.CreateVector(sg_inputs),
        options.no_outputs ? builder.CreateVector<int32_t>({})
                           : builder.CreateVector<int32_t>({output}),
        builder.CreateVector(ops),
        builder.CreateString(absl::StrCat("subgraph_", i))));
    std::vector<flatbuffers::Offset<tflite::TensorMap>> output_maps;
    if (!options.no_outputs) {
      output_maps.push_back(tflite::CreateTensorMap(
          builder, builder.CreateString("encodings"), output));
    }
    const std::string signature_name =
        batch <= 0 ? absl::StrCat("encoder_", seq_len)
                   : absl::StrCat("encoder_", batch, "x", seq_len);
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder, builder.CreateVector(input_maps),
        builder.CreateVector(output_maps), builder.CreateString(signature_name),
        /*subgraph_index=*/i));
  }

  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector(subgraphs),
      builder.CreateString("batched_sum_encoder_model"), buffers_vec,
      /*metadata_buffer=*/0, /*metadata=*/0, builder.CreateVector(sig_defs));
  tflite::FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

// Returns the sum of `tokens` plus their count, i.e. each element of the
// output of the sum encoder for these tokens with the cast embedder.
float ExpectedSumEncoding(const std::vector<int32_t>& tokens) {
  float expected = tokens.size();
  for (int32_t token : tokens) expected += std::max(token, 0);
  return expected;
}

// Creates text-only executor inputs holding a copy of `tokens`.
absl::StatusOr<ExecutorInputs> CreateTokenInputs(
    Environment& env, const std::vector<int32_t>& tokens) {
  auto buffer = TensorBuffer::CreateManaged(
      env, ::litert::TensorBufferType::kHostMemory,
      RankedTensorType(ElementType::Int32,
                       Layout(Dimensions({static_cast<int>(tokens.size())}))),
      tokens.size() * sizeof(int32_t));
  if (!buffer.HasValue()) {
    return absl::InternalError(buffer.Error().Message());
  }
  if (auto written = buffer->Write(absl::MakeConstSpan(tokens));
      !written.HasValue()) {
    return absl::InternalError(written.Error().Message());
  }
  return ExecutorInputs(ExecutorTextData(std::move(*buffer)), std::nullopt,
                        std::nullopt);
}

// The cast embedder and a sum encoder, which must outlive the executors
// created from them.
struct SumModels {
  std::vector<uint8_t> embedder_buf;
  std::vector<uint8_t> encoder_buf;
  std::vector<uint8_t> per_layer_embedder_buf;
  std::optional<Model> embedder;
  std::optional<Model> encoder;
  std::optional<Model> per_layer_embedder;
};

constexpr int kSumEmbeddingDim = 8;

// Returns `value` repeated for every element of a sum encoder output.
std::vector<float> SumEncoderOutput(float value) {
  return std::vector<float>(kSumEmbeddingDim, value);
}

absl::StatusOr<std::unique_ptr<SumModels>> CreateSumModels(
    const std::vector<std::pair<int32_t, int32_t>>& batch_and_seq_lengths,
    const SumEncoderOptions& options = {}) {
  auto models = std::make_unique<SumModels>();
  models->embedder_buf =
      BuildCastEmbedderModelBuffer(kSumEmbeddingDim, options.embedder_offset);
  models->encoder_buf = BuildBatchedSumEncoderModelBuffer(
      batch_and_seq_lengths, kSumEmbeddingDim, options);
  auto embedder = Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
      models->embedder_buf.data(), models->embedder_buf.size()));
  auto encoder = Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
      models->encoder_buf.data(), models->encoder_buf.size()));
  if (!embedder.HasValue() || !encoder.HasValue()) {
    return absl::InternalError("Failed to create the test models.");
  }
  models->embedder = std::move(*embedder);
  models->encoder = std::move(*encoder);
  if (options.with_per_layer_embedder) {
    models->per_layer_embedder_buf =
        BuildCastEmbedderModelBuffer(kSumEmbeddingDim);
    auto per_layer_embedder = Model::CreateFromBuffer(
        litert::BufferRef<uint8_t>(models->per_layer_embedder_buf.data(),
                                   models->per_layer_embedder_buf.size()));
    if (!per_layer_embedder.HasValue()) {
      return absl::InternalError("Failed to create the per-layer embedder.");
    }
    models->per_layer_embedder = std::move(*per_layer_embedder);
  }
  return models;
}

absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
CreateSumExecutor(Environment& env, const SumModels& models,
                  std::vector<std::string> selected_signatures = {}) {
  ABSL_ASSIGN_OR_RETURN(ModelAssets model_assets,
                        ModelAssets::Create("dummy_model_path"));
  ABSL_ASSIGN_OR_RETURN(EmbeddingExecutorSettings settings,
                        EmbeddingExecutorSettings::CreateDefault(
                            model_assets, /*backend=*/Backend::CPU));
  settings.SetSelectedSignatures(std::move(selected_signatures));
  return EmbeddingLiteRtCompiledModelExecutor::Create(
      settings, env,
      std::make_unique<FakeEmbeddingModelResources>(
          &*models.embedder, &*models.encoder,
          models.per_layer_embedder.has_value() ? &*models.per_layer_embedder
                                                : nullptr));
}

absl::StatusOr<std::vector<ExecutorInputs>> CreateBatchInputs(
    Environment& env, const std::vector<std::vector<int32_t>>& requests) {
  std::vector<ExecutorInputs> batch_inputs;
  for (const auto& tokens : requests) {
    ABSL_ASSIGN_OR_RETURN(auto inputs, CreateTokenInputs(env, tokens));
    batch_inputs.push_back(std::move(inputs));
  }
  return batch_inputs;
}

// Runs `ComputeEmbeddingBatch` with profiling, and returns the outputs and the
// number of padded tokens that the text encoder processed.
absl::StatusOr<std::pair<std::vector<EmbeddingOutput>, int64_t>>
RunProfiledBatch(EmbeddingLiteRtCompiledModelExecutor& executor,
                 const std::vector<ExecutorInputs>& batch_inputs,
                 const ComputeEmbeddingOptions& options = {}) {
  ABSL_RETURN_IF_ERROR(executor.StartProfiling());
  ABSL_ASSIGN_OR_RETURN(auto outputs,
                        executor.ComputeEmbeddingBatch(batch_inputs, options));
  ABSL_ASSIGN_OR_RETURN(ExecutorStats stats, executor.StopProfiling());
  return std::make_pair(
      std::move(outputs),
      stats.GetMetricAsInt64("(e2e) Embedding num padded tokens"));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_BatchedSignature_MatchesSingleSequenceRuns) {
  // `encoder_1x8` and `encoder_2x8` share the sequence length 8, and must not
  // shadow each other.
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{1, 8}, {2, 4}, {2, 8}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));
  ASSERT_OK_AND_ASSIGN(auto input_dims, executor->GetExpectedInputDimension());
  EXPECT_EQ(input_dims, (std::vector<int>{1, 8, kSumEmbeddingDim}));

  const std::vector<std::vector<int32_t>> requests = {
      {1, 2, 3}, {5}, {7, 11, 13, 17}};
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  auto& [outputs, padded_tokens] = profiled_batch;
  // The two shortest requests run together on `encoder_2x4` (2 x 4 padded
  // tokens) and the longest one alone on `encoder_1x8` (8 padded tokens),
  // instead of 3 x 8 padded tokens without batching.
  EXPECT_EQ(padded_tokens, 16);

  ASSERT_EQ(outputs.size(), requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(
        outputs[i].embedding,
        std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[i])))
        << "request " << i;
    EXPECT_EQ(outputs[i].input_length, requests[i].size());
    EXPECT_EQ(outputs[i].num_chunks, 1);

    ASSERT_OK_AND_ASSIGN(auto single_output,
                         executor->ComputeEmbedding(batch_inputs[i]));
    EXPECT_EQ(outputs[i].embedding, single_output.embedding) << "request " << i;
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_TwoDimensionalMask_CascadesBatchSizes) {
  // EmbeddingGemma v2 exports an `input_mask` of shape [B, T].
  ASSERT_OK_AND_ASSIGN(auto models,
                       CreateSumModels({{1, 8}, {2, 4}, {4, 4}},
                                       SumEncoderOptions{.mask_rank = 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  const std::vector<std::vector<int32_t>> requests = {
      {1}, {2, 3}, {4, 5, 6}, {7, 8, 9, 10}, {11}, {12, 13}, {14, 15, 16}};
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  auto& [outputs, padded_tokens] = profiled_batch;
  // 4 requests on `encoder_4x4`, 2 on `encoder_2x4` and 1 on `encoder_1x8`.
  EXPECT_EQ(padded_tokens, 4 * 4 + 2 * 4 + 8);

  ASSERT_EQ(outputs.size(), requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(
        outputs[i].embedding,
        std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[i])))
        << "request " << i;
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_SkipsBatchedSignaturesWithMorePadding) {
  // Running the 4 short requests on `encoder_4x16` would process 64 padded
  // tokens instead of 4 x 4 on `encoder_1x4`.
  ASSERT_OK_AND_ASSIGN(auto models,
                       CreateSumModels({{1, 4}, {1, 16}, {4, 16}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  const std::vector<std::vector<int32_t>> requests = {
      {1, 2}, {3, 4}, {5, 6}, {7, 8}};
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  auto& [outputs, padded_tokens] = profiled_batch;
  EXPECT_EQ(padded_tokens, 4 * 4);
  ASSERT_EQ(outputs.size(), requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(
        outputs[i].embedding,
        std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[i])))
        << "request " << i;
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_MixedLengths_SkipsBatchedWhenTotalPaddingIncreases) {
  ASSERT_OK_AND_ASSIGN(auto models,
                       CreateSumModels({{1, 4}, {1, 16}, {4, 12}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  // 4 requests of lengths 2, 2, 2, 10: single-item padded tokens =
  // 4 + 4 + 4 + 16 = 28, whereas `encoder_4x12` would process 4 * 12 = 48.
  const std::vector<std::vector<int32_t>> mixed_requests = {
      {1, 2},
      {3, 4},
      {5, 6},
      {7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
  };
  ASSERT_OK_AND_ASSIGN(auto mixed_inputs,
                       CreateBatchInputs(env, mixed_requests));
  ASSERT_OK_AND_ASSIGN(auto mixed_profiled,
                       RunProfiledBatch(*executor, mixed_inputs));
  EXPECT_EQ(mixed_profiled.second, 4 + 4 + 4 + 16);
  ASSERT_EQ(mixed_profiled.first.size(), mixed_requests.size());
  for (size_t i = 0; i < mixed_requests.size(); ++i) {
    EXPECT_EQ(mixed_profiled.first[i].embedding,
              SumEncoderOutput(ExpectedSumEncoding(mixed_requests[i])))
        << "mixed request " << i;
  }

  // 4 requests of lengths 10, 10, 10, 10: 4 * 12 = 48 <= 16 + 16 + 16 + 16,
  // so `encoder_4x12` is dispatched, which the padded token count (48 instead
  // of 64) shows.
  const std::vector<std::vector<int32_t>> equal_requests = {
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
      {3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      {4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
  };
  ASSERT_OK_AND_ASSIGN(auto equal_inputs,
                       CreateBatchInputs(env, equal_requests));
  ASSERT_OK_AND_ASSIGN(auto equal_profiled,
                       RunProfiledBatch(*executor, equal_inputs));
  EXPECT_EQ(equal_profiled.second, 4 * 12);
  ASSERT_EQ(equal_profiled.first.size(), equal_requests.size());
  for (size_t i = 0; i < equal_requests.size(); ++i) {
    EXPECT_EQ(equal_profiled.first[i].embedding,
              SumEncoderOutput(ExpectedSumEncoding(equal_requests[i])))
        << "equal request " << i;
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_MixedInputsKeepTheirOrder) {
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{1, 8}, {2, 4}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  const std::vector<std::vector<int32_t>> requests = {
      {1, 2},
      // Too long for any signature: truncated to 8 tokens on `encoder_1x8`.
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {3, 4, 5},
      // A negative (multimodal placeholder) token is never batched.
      {6, -1},
  };
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(
      auto profiled_batch,
      RunProfiledBatch(
          *executor, batch_inputs,
          {.input_overflow_strategy = InputOverflowStrategy::kTruncate}));
  auto& [outputs, padded_tokens] = profiled_batch;
  // Requests 0 and 2 run together on `encoder_2x4`; 1 and 3 run alone on
  // `encoder_1x8`.
  EXPECT_EQ(padded_tokens, 2 * 4 + 8 + 8);

  ASSERT_EQ(outputs.size(), requests.size());
  EXPECT_EQ(
      outputs[0].embedding,
      std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[0])));
  EXPECT_EQ(outputs[1].embedding,
            std::vector<float>(kSumEmbeddingDim,
                               ExpectedSumEncoding({1, 2, 3, 4, 5, 6, 7, 8})));
  EXPECT_EQ(outputs[1].input_length, 10);
  EXPECT_EQ(outputs[1].truncated_length, 8);
  EXPECT_EQ(
      outputs[2].embedding,
      std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[2])));
  EXPECT_EQ(
      outputs[3].embedding,
      std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[3])));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_BatchedSignatureUsesRunOptions) {
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{1, 8}, {2, 4}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  // Both requests fit `encoder_2x4`, so they run together in one batched run.
  const std::vector<std::vector<int32_t>> requests = {{1, 2}, {3, 4, 5}};
  int num_calls = 0;
  ASSERT_OK(executor->SetRunOptions(CountingRunOptionsFactory(&num_calls)));
  {
    ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
    ASSERT_OK_AND_ASSIGN(auto outputs,
                         executor->ComputeEmbeddingBatch(batch_inputs));
    ASSERT_EQ(outputs.size(), requests.size());
    for (int i = 0; i < requests.size(); ++i) {
      EXPECT_EQ(outputs[i].embedding,
                std::vector<float>(kSumEmbeddingDim,
                                   ExpectedSumEncoding(requests[i])))
          << "request " << i;
    }
  }
  // The single batched run got new options.
  EXPECT_EQ(num_calls, 1);

  // A failure to create the options fails the batched run.
  ASSERT_OK(executor->SetRunOptions(
      FailingRunOptionsFactory(absl::InvalidArgumentError("bad options"))));
  {
    ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
    EXPECT_THAT(executor->ComputeEmbeddingBatch(batch_inputs),
                StatusIs(absl::StatusCode::kInvalidArgument, "bad options"));
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_LongInputsFallBackToSingleSequenceRuns) {
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{1, 4}, {1, 8}, {2, 4}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  // Both requests are longer than `encoder_2x4`, so they run one at a time on
  // `encoder_1x8`.
  const std::vector<std::vector<int32_t>> requests = {{1, 2, 3, 4, 5},
                                                      {6, 7, 8, 9, 10, 11}};
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  auto& [outputs, padded_tokens] = profiled_batch;
  EXPECT_EQ(padded_tokens, 16);
  ASSERT_EQ(outputs.size(), requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(
        outputs[i].embedding,
        std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[i])))
        << "request " << i;
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_UnselectedBatchedSignaturesAreNotUsed) {
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{1, 8}, {2, 4}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateSumExecutor(env, *models, /*selected_signatures=*/{"encoder_1x8"}));

  ASSERT_OK_AND_ASSIGN(auto batch_inputs,
                       CreateBatchInputs(env, {{1, 2}, {3, 4}}));
  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  auto& [outputs, padded_tokens] = profiled_batch;
  EXPECT_EQ(padded_tokens, 2 * 8);
  EXPECT_EQ(outputs.size(), 2);
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_DynamicBatchDimension_RunsAsBatchSizeOne) {
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{-1, 8}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  EXPECT_THAT(
      executor->GetSupportedVariants(),
      ElementsAre(TextEncoderVariant{.batch_size = 1, .sequence_length = 8}));
  ASSERT_OK_AND_ASSIGN(auto input_dims, executor->GetExpectedInputDimension());
  EXPECT_EQ(input_dims, (std::vector<int>{-1, 8, kSumEmbeddingDim}));

  const std::vector<std::vector<int32_t>> requests = {{1, 2, 3}, {4, 5}};
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(auto single_output,
                       executor->ComputeEmbedding(batch_inputs[0]));
  EXPECT_EQ(
      single_output.embedding,
      std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[0])));

  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  EXPECT_EQ(profiled_batch.second, 2 * 8);
  ASSERT_EQ(profiled_batch.first.size(), requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(
        profiled_batch.first[i].embedding,
        std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[i])))
        << "request " << i;
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_OnlyBatchedSignatures_ReturnsFailedPrecondition) {
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{2, 8}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  EXPECT_THAT(CreateSumExecutor(env, *models),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("batch size 1")));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_BatchedSignaturesFailToLoad_FallsBackToSingleSignatures) {
  ASSERT_OK_AND_ASSIGN(
      auto models,
      CreateSumModels({{1, 8}, {2, 4}},
                      SumEncoderOptions{.broken_batched_signatures = true}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  const std::vector<std::vector<int32_t>> requests = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  auto& [outputs, padded_tokens] = profiled_batch;
  // Both requests run alone on `encoder_1x8`.
  EXPECT_EQ(padded_tokens, 2 * 8);
  ASSERT_EQ(outputs.size(), requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(
        outputs[i].embedding,
        std::vector<float>(kSumEmbeddingDim, ExpectedSumEncoding(requests[i])))
        << "request " << i;
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_PerLayerInputWithoutEmbedder_ReturnsError) {
  ASSERT_OK_AND_ASSIGN(
      auto models,
      CreateSumModels({{1, 8}, {2, 4}},
                      SumEncoderOptions{.with_per_layer_input = true}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  ASSERT_OK_AND_ASSIGN(auto batch_inputs,
                       CreateBatchInputs(env, {{1, 2}, {3, 4}}));
  EXPECT_THAT(
      executor->ComputeEmbedding(batch_inputs[0]),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("per-layer")));
  EXPECT_THAT(
      executor->ComputeEmbeddingBatch(batch_inputs),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("per-layer")));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_BatchedSignaturesFailToLoad_InterleavedSignatures) {
  // The export interleaves the batch sizes, so the batch size 1 signatures
  // kept after the fallback have non-contiguous indices.
  ASSERT_OK_AND_ASSIGN(
      auto models,
      CreateSumModels({{1, 4}, {2, 4}, {1, 8}, {2, 8}},
                      SumEncoderOptions{.broken_batched_signatures = true}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));
  EXPECT_THAT(
      executor->GetSupportedVariants(),
      ElementsAre(TextEncoderVariant{.batch_size = 1, .sequence_length = 4},
                  TextEncoderVariant{.batch_size = 1, .sequence_length = 8}));

  const std::vector<std::vector<int32_t>> requests = {
      {1, 2}, {3, 4, 5, 6, 7, 8}, {9, 10}};
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  auto& [outputs, padded_tokens] = profiled_batch;
  // Each request runs alone, on `encoder_1x4` or `encoder_1x8`.
  EXPECT_EQ(padded_tokens, 4 + 8 + 4);
  ASSERT_EQ(outputs.size(), requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(outputs[i].embedding,
              SumEncoderOutput(ExpectedSumEncoding(requests[i])))
        << "request " << i;
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_ReusedBatchedSignature_RewritesPaddingAndMask) {
  // Padding uses the default (token 0) embedding, which is 100 here, so that
  // it differs from zeros and from stale embeddings of an earlier run.
  constexpr float kOffset = 100.0f;
  ASSERT_OK_AND_ASSIGN(
      auto models,
      CreateSumModels({{1, 8}, {2, 8}},
                      SumEncoderOptions{.embedder_offset = kOffset}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  // Every position of the 8 padded positions contributes the offset: tokens
  // as `t + offset` (plus 1 for the mask), padding as the default embedding.
  auto expected = [&](const std::vector<int32_t>& tokens) {
    return SumEncoderOutput(ExpectedSumEncoding(tokens) + 8 * kOffset);
  };

  // A long group, then a short group on the same `encoder_2x8` signature.
  for (const std::vector<std::vector<int32_t>>& requests :
       std::vector<std::vector<std::vector<int32_t>>>{
           {{1, 2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7, 8}}, {{3}, {4}}}) {
    ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
    ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                         RunProfiledBatch(*executor, batch_inputs));
    auto& [outputs, padded_tokens] = profiled_batch;
    EXPECT_EQ(padded_tokens, 2 * 8);
    ASSERT_EQ(outputs.size(), requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
      EXPECT_EQ(outputs[i].embedding, expected(requests[i]))
          << "request " << i << " of length " << requests[i].size();
      ASSERT_OK_AND_ASSIGN(auto single_output,
                           executor->ComputeEmbedding(batch_inputs[i]));
      EXPECT_EQ(outputs[i].embedding, single_output.embedding);
    }
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_BatchedSignatureWithPerLayerEmbeddings) {
  ASSERT_OK_AND_ASSIGN(
      auto models,
      CreateSumModels({{1, 8}, {2, 4}},
                      SumEncoderOptions{.with_per_layer_input = true,
                                        .with_per_layer_embedder = true}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  const std::vector<std::vector<int32_t>> requests = {{1, 2}, {3, 4, 5}};
  ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
  ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                       RunProfiledBatch(*executor, batch_inputs));
  auto& [outputs, padded_tokens] = profiled_batch;
  // Both requests run together on `encoder_2x4`.
  EXPECT_EQ(padded_tokens, 2 * 4);
  ASSERT_EQ(outputs.size(), requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    // The per-layer embeddings add the tokens a second time.
    float token_sum = 0;
    for (int32_t token : requests[i]) token_sum += token;
    EXPECT_EQ(outputs[i].embedding,
              SumEncoderOutput(ExpectedSumEncoding(requests[i]) + token_sum))
        << "request " << i;
    ASSERT_OK_AND_ASSIGN(auto single_output,
                         executor->ComputeEmbedding(batch_inputs[i]));
    EXPECT_EQ(outputs[i].embedding, single_output.embedding);
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_DropsBatchedSignaturesThatCannotRun) {
  const std::vector<std::pair<std::string, SumEncoderOptions>> cases = {
      {"mask_per_embedding_element",
       {.batched_mask_per_embedding_element = true}},
      {"unpooled_output", {.batched_unpooled_output = true}},
      {"per_layer_input_without_embedder",
       {.with_batched_per_layer_input = true}},
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  for (const auto& [name, options] : cases) {
    SCOPED_TRACE(name);
    ASSERT_OK_AND_ASSIGN(auto models,
                         CreateSumModels({{1, 8}, {2, 4}}, options));
    ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

    const std::vector<std::vector<int32_t>> requests = {{1, 2}, {3, 4}};
    ASSERT_OK_AND_ASSIGN(auto batch_inputs, CreateBatchInputs(env, requests));
    ASSERT_OK_AND_ASSIGN(auto profiled_batch,
                         RunProfiledBatch(*executor, batch_inputs));
    auto& [outputs, padded_tokens] = profiled_batch;
    // `encoder_2x4` is dropped, so both requests run alone on `encoder_1x8`.
    EXPECT_EQ(padded_tokens, 2 * 8);
    ASSERT_EQ(outputs.size(), requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
      EXPECT_EQ(outputs[i].embedding,
                SumEncoderOutput(ExpectedSumEncoding(requests[i])))
          << "request " << i;
    }
  }
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_SkipsSignaturesWithInvalidSequenceLength) {
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{1, -1}, {1, 8}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));
  EXPECT_THAT(
      executor->GetSupportedVariants(),
      ElementsAre(TextEncoderVariant{.batch_size = 1, .sequence_length = 8}));

  ASSERT_OK_AND_ASSIGN(auto inputs, CreateTokenInputs(env, {1, 2, 3}));
  ASSERT_OK_AND_ASSIGN(auto output, executor->ComputeEmbedding(inputs));
  EXPECT_EQ(output.embedding, SumEncoderOutput(ExpectedSumEncoding({1, 2, 3})));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     Create_SignatureWithoutOutputs_ReturnsFailedPrecondition) {
  ASSERT_OK_AND_ASSIGN(
      auto models,
      CreateSumModels({{1, 8}}, SumEncoderOptions{.no_outputs = true}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  EXPECT_THAT(CreateSumExecutor(env, *models),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("has no output tensors")));
}

TEST(EmbeddingLiteRtCompiledModelExecutorTest,
     ComputeEmbeddingBatch_OverflowError_FailsBeforeAnyRun) {
  ASSERT_OK_AND_ASSIGN(auto models, CreateSumModels({{1, 4}, {2, 4}}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor, CreateSumExecutor(env, *models));

  // The first two requests would run on `encoder_2x4`, but the last one is
  // too long.
  ASSERT_OK_AND_ASSIGN(
      auto batch_inputs,
      CreateBatchInputs(env, {{1, 2}, {3, 4}, {1, 2, 3, 4, 5, 6}}));
  ASSERT_OK(executor->StartProfiling());
  EXPECT_THAT(
      executor->ComputeEmbeddingBatch(
          batch_inputs,
          ComputeEmbeddingOptions{.input_overflow_strategy =
                                      InputOverflowStrategy::kError}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("exceeds maximum supported signature length")));
  ASSERT_OK_AND_ASSIGN(ExecutorStats stats, executor->StopProfiling());
  EXPECT_EQ(stats.GetMetricAsInt64("(e2e) Embedding num padded tokens"), 0);
}

}  // namespace
}  // namespace litert::lm
