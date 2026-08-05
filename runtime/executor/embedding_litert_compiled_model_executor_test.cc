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
#include "runtime/executor/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"  // from @org_tensorflow

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
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

// Builds a minimal TFLite model FlatBuffer with given signature key and tensor
// shapes.
std::vector<uint8_t> BuildDummyTfLiteModelBuffer(
    absl::string_view signature_key, const std::vector<int32_t>& input_dims,
    tflite::TensorType input_type, const std::vector<int32_t>& output_dims,
    tflite::TensorType output_type) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ABS);
  auto opcodes_vec = builder.CreateVector({opcode});

  auto input_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(input_dims), input_type,
      /*buffer=*/0, builder.CreateString("input_tensor"));
  auto output_tensor = tflite::CreateTensor(
      builder, builder.CreateVector(output_dims), output_type,
      /*buffer=*/0, builder.CreateString("output_tensor"));
  auto tensors_vec = builder.CreateVector({input_tensor, output_tensor});

  std::vector<int32_t> op_inputs = {0};
  std::vector<int32_t> op_outputs = {1};
  auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                   builder.CreateVector(op_inputs),
                                   builder.CreateVector(op_outputs));
  auto ops_vec = builder.CreateVector({op});

  std::vector<int32_t> sg_inputs = {0};
  std::vector<int32_t> sg_outputs = {1};
  auto subgraph = tflite::CreateSubGraph(
      builder, tensors_vec, builder.CreateVector(sg_inputs),
      builder.CreateVector(sg_outputs), ops_vec, builder.CreateString("main"));
  auto subgraphs_vec = builder.CreateVector({subgraph});

  auto buffer = tflite::CreateBuffer(builder);
  auto buffers_vec = builder.CreateVector({buffer});

  auto input_map =
      tflite::CreateTensorMap(builder, builder.CreateString("input"), 0);
  auto output_map =
      tflite::CreateTensorMap(builder, builder.CreateString("output"), 1);
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
  FakeEmbeddingModelResources(const litert::Model* embedder,
                              const litert::Model* encoder)
      : embedder_(embedder), encoder_(encoder) {}

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    if (model_type == ModelType::kTfLiteEmbedder) {
      if (embedder_ != nullptr) return embedder_;
      return absl::NotFoundError("kTfLiteEmbedder model not found.");
    }
    if (model_type == ModelType::kTfLiteTextEncoder) {
      if (encoder_ != nullptr) return encoder_;
      return absl::NotFoundError("kTfLiteTextEncoder model not found.");
    }
    return absl::NotFoundError("Model type not found.");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("Unimplemented");
  }

 private:
  const litert::Model* embedder_;
  const litert::Model* encoder_;
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

  ASSERT_OK_AND_ASSIGN(auto embedding,
                       embedding_executor->ComputeEmbedding(inputs));
  EXPECT_EQ(embedding.size(), 8);
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

  ASSERT_OK_AND_ASSIGN(auto batch_embeddings,
                       embedding_executor->ComputeEmbeddingBatch(batch_inputs));
  EXPECT_EQ(batch_embeddings.size(), 2);
  EXPECT_EQ(batch_embeddings[0].size(), 8);
  EXPECT_EQ(batch_embeddings[1].size(), 8);
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
     Create_MissingEmbedder_ReturnsNotFoundError) {
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

  auto fake_resources =
      std::make_unique<FakeEmbeddingModelResources>(nullptr, &encoder_model);
  auto embedding_executor = EmbeddingLiteRtCompiledModelExecutor::Create(
      settings, env, std::move(fake_resources));
  EXPECT_THAT(
      embedding_executor,
      StatusIs(absl::StatusCode::kNotFound,
               "kTfLiteEmbedder model not found in resources for embedding."));
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

  ASSERT_OK_AND_ASSIGN(auto embedding,
                       embedding_executor->ComputeEmbedding(inputs));
  EXPECT_EQ(embedding.size(), 8);
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

    ASSERT_OK_AND_ASSIGN(auto embedding,
                         embedding_executor->ComputeEmbedding(inputs));
    EXPECT_EQ(embedding.size(), 12 * 8);
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

    ASSERT_OK_AND_ASSIGN(auto embedding,
                         embedding_executor->ComputeEmbedding(inputs));
    EXPECT_EQ(embedding.size(), 24 * 8);
  }

  // Case 3: Input size 32 -> should fail because it exceeds max signature
  // length (24).
  {
    alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) int32_t token_data[32];
    std::fill_n(token_data, 32, 42);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto token_ids_buffer,
        TensorBuffer::CreateFromHostMemory(
            env, RankedTensorType(ElementType::Int32, Layout(Dimensions({32}))),
            token_data, sizeof(token_data)));

    ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                          std::nullopt, std::nullopt);

    auto embedding = embedding_executor->ComputeEmbedding(inputs);
    EXPECT_FALSE(embedding.ok());
    EXPECT_THAT(
        embedding.status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Max input length of the text encoder is 24. "
                           "Input length was 32")));
  }
}

}  // namespace
}  // namespace litert::lm
