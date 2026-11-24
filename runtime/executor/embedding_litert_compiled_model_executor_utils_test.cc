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

#include "runtime/executor/embedding_litert_compiled_model_executor_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::litert::BufferRef;
using ::litert::Dimensions;
using ::litert::ElementType;
using ::litert::Layout;
using ::litert::Model;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::Pair;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

struct SignatureSpec {
  std::string key;
  // Pairs of (input_name, shape_signature). Entries <= 0 are encoded as dynamic
  // dimensions in `shape_signature` with static dim 1.
  std::vector<std::pair<std::string, std::vector<int32_t>>> inputs;
  bool has_output = true;
};

std::vector<uint8_t> BuildSignatureTestModelBuffer(
    absl::Span<const SignatureSpec> specs) {
  flatbuffers::FlatBufferBuilder builder;
  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  for (size_t i = 0; i < specs.size(); ++i) {
    const SignatureSpec& spec = specs[i];
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
    std::vector<int32_t> sg_inputs;
    std::vector<flatbuffers::Offset<tflite::TensorMap>> inputs_map;

    for (size_t j = 0; j < spec.inputs.size(); ++j) {
      const auto& [name, shape_sig] = spec.inputs[j];
      std::vector<int32_t> dims = shape_sig;
      bool dynamic = false;
      for (int32_t& dim : dims) {
        if (dim <= 0) {
          dim = 1;
          dynamic = true;
        }
      }
      tensors.push_back(tflite::CreateTensor(
          builder, builder.CreateVector(dims), tflite::TensorType_FLOAT32,
          /*buffer=*/0,
          builder.CreateString(absl::StrCat(name, "_", i, "_", j)),
          /*quantization=*/0, /*is_variable=*/false, /*sparsity=*/0,
          dynamic ? builder.CreateVector(shape_sig) : 0,
          /*has_rank=*/dynamic));
      const int32_t tensor_idx = static_cast<int32_t>(j);
      sg_inputs.push_back(tensor_idx);
      inputs_map.push_back(tflite::CreateTensorMap(
          builder, builder.CreateString(name), tensor_idx));
    }

    std::vector<int32_t> sg_outputs;
    std::vector<flatbuffers::Offset<tflite::TensorMap>> outputs_map;
    if (spec.has_output) {
      const int32_t out_idx = static_cast<int32_t>(tensors.size());
      tensors.push_back(tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>({1, 4}),
          tflite::TensorType_FLOAT32, /*buffer=*/0,
          builder.CreateString(absl::StrCat("output_", i))));
      sg_outputs.push_back(out_idx);
      outputs_map.push_back(tflite::CreateTensorMap(
          builder, builder.CreateString("output"), out_idx));
    }

    std::vector<flatbuffers::Offset<tflite::Operator>> ops;
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, builder.CreateVector(tensors), builder.CreateVector(sg_inputs),
        builder.CreateVector(sg_outputs), builder.CreateVector(ops),
        builder.CreateString(absl::StrCat("subgraph_", i))));
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder, builder.CreateVector(inputs_map),
        builder.CreateVector(outputs_map), builder.CreateString(spec.key),
        /*subgraph_index=*/static_cast<uint32_t>(i)));
  }

  auto buffer = tflite::CreateBuffer(builder);
  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector(subgraphs),
      builder.CreateString("signature_test_model"),
      builder.CreateVector({buffer}), /*metadata_buffer=*/0, /*metadata=*/0,
      builder.CreateVector(sig_defs));
  tflite::FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

absl::StatusOr<TensorBuffer> MakeHostBuffer(ElementType element_type,
                                            Dimensions dims,
                                            size_t element_size_bytes) {
  size_t total_elements = 1;
  for (int dim : dims) {
    total_elements *= static_cast<size_t>(dim);
  }
  RankedTensorType type(element_type, Layout(std::move(dims)));
  auto buffer = TensorBuffer::CreateManagedHostMemory(
      type, total_elements * element_size_bytes);
  if (!buffer) {
    return absl::InternalError(buffer.Error().Message());
  }
  return std::move(*buffer);
}

TEST(GetTextEncoderSignatureMapsTest,
     GroupsSingleAndBatchedSignaturesAndIgnoresNonEncoder) {
  const std::vector<uint8_t> buf = BuildSignatureTestModelBuffer({
      {.key = "embedder", .inputs = {{"tokens", {1, 8}}}},
      {.key = "encoder_1x8",
       .inputs = {{"input_mask", {1, 8}}, {"embeddings", {1, 8, 4}}}},
      {.key = "encoder_1x16", .inputs = {{"embeddings", {1, 16, 4}}}},
      {.key = "encoder_4x8", .inputs = {{"embeddings", {4, 8, 4}}}},
      {.key = "encoder_2x16", .inputs = {{"embeddings", {2, 16, 4}}}},
  });
  auto model = Model::CreateFromBuffer(BufferRef<uint8_t>(buf));
  ASSERT_TRUE(model.HasValue()) << model.Error().Message();

  ASSERT_OK_AND_ASSIGN(TextEncoderSignatureMaps maps,
                       GetTextEncoderSignatureMaps(*model));
  EXPECT_THAT(maps.single, ElementsAre(Pair(8, 1), Pair(16, 2)));
  EXPECT_THAT(maps.batched, ElementsAre(Pair(2, ElementsAre(Pair(16, 4))),
                                        Pair(4, ElementsAre(Pair(8, 3)))));
}

TEST(GetTextEncoderSignatureMapsTest, FiltersBySelectedSignatures) {
  const std::vector<uint8_t> buf = BuildSignatureTestModelBuffer({
      {.key = "encoder_1x8", .inputs = {{"embeddings", {1, 8, 4}}}},
      {.key = "encoder_1x16", .inputs = {{"embeddings", {1, 16, 4}}}},
      {.key = "encoder_4x8", .inputs = {{"embeddings", {4, 8, 4}}}},
  });
  auto model = Model::CreateFromBuffer(BufferRef<uint8_t>(buf));
  ASSERT_TRUE(model.HasValue()) << model.Error().Message();

  const std::vector<std::string> selected = {"encoder_1x16", "encoder_4x8"};
  ASSERT_OK_AND_ASSIGN(TextEncoderSignatureMaps maps,
                       GetTextEncoderSignatureMaps(*model, selected));
  EXPECT_THAT(maps.single, ElementsAre(Pair(16, 1)));
  EXPECT_THAT(maps.batched, ElementsAre(Pair(4, ElementsAre(Pair(8, 2)))));

  const std::vector<std::string> batched_only = {"encoder_4x8"};
  EXPECT_THAT(GetTextEncoderSignatureMaps(*model, batched_only),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(GetTextEncoderSignatureMapsTest,
     SupportsRank2EmbeddingsAndFallbackToFirstInput) {
  const std::vector<uint8_t> buf = BuildSignatureTestModelBuffer({
      {.key = "encoder_rank2", .inputs = {{"tokens_in", {12, 4}}}},
  });
  auto model = Model::CreateFromBuffer(BufferRef<uint8_t>(buf));
  ASSERT_TRUE(model.HasValue()) << model.Error().Message();

  ASSERT_OK_AND_ASSIGN(TextEncoderSignatureMaps maps,
                       GetTextEncoderSignatureMaps(*model));
  EXPECT_THAT(maps.single, ElementsAre(Pair(12, 0)));
  EXPECT_THAT(maps.batched, IsEmpty());
}

TEST(GetTextEncoderSignatureMapsTest,
     MapsDynamicBatchToOneAndSkipsNonPositiveSequenceLength) {
  const std::vector<uint8_t> buf = BuildSignatureTestModelBuffer({
      {.key = "encoder_dyn_seq", .inputs = {{"embeddings", {1, -1, 4}}}},
      {.key = "encoder_dyn_batch", .inputs = {{"embeddings", {-1, 8, 4}}}},
  });
  auto model = Model::CreateFromBuffer(BufferRef<uint8_t>(buf));
  ASSERT_TRUE(model.HasValue()) << model.Error().Message();

  ASSERT_OK_AND_ASSIGN(TextEncoderSignatureMaps maps,
                       GetTextEncoderSignatureMaps(*model));
  EXPECT_THAT(maps.single, ElementsAre(Pair(8, 1)));
  EXPECT_THAT(maps.batched, IsEmpty());
}

TEST(GetTextEncoderSignatureMapsTest,
     RejectsEmptyInputsEmptyOutputsAndUnsupportedRank) {
  {
    const std::vector<uint8_t> buf = BuildSignatureTestModelBuffer({
        {.key = "encoder_no_inputs", .inputs = {}, .has_output = true},
    });
    auto model = Model::CreateFromBuffer(BufferRef<uint8_t>(buf));
    ASSERT_TRUE(model.HasValue()) << model.Error().Message();
    EXPECT_THAT(GetTextEncoderSignatureMaps(*model),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  {
    const std::vector<uint8_t> buf = BuildSignatureTestModelBuffer({
        {.key = "encoder_no_outputs",
         .inputs = {{"embeddings", {1, 8, 4}}},
         .has_output = false},
    });
    auto model = Model::CreateFromBuffer(BufferRef<uint8_t>(buf));
    ASSERT_TRUE(model.HasValue()) << model.Error().Message();
    EXPECT_THAT(GetTextEncoderSignatureMaps(*model),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  {
    const std::vector<uint8_t> buf = BuildSignatureTestModelBuffer({
        {.key = "encoder_rank4", .inputs = {{"embeddings", {1, 8, 4, 1}}}},
    });
    auto model = Model::CreateFromBuffer(BufferRef<uint8_t>(buf));
    ASSERT_TRUE(model.HasValue()) << model.Error().Message();
    EXPECT_THAT(GetTextEncoderSignatureMaps(*model),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
}

TEST(GetTextEncoderSignatureMapsTest,
     RejectsModelWithoutBatchSizeOneSignature) {
  const std::vector<uint8_t> buf = BuildSignatureTestModelBuffer({
      {.key = "encoder_4x8", .inputs = {{"embeddings", {4, 8, 4}}}},
  });
  auto model = Model::CreateFromBuffer(BufferRef<uint8_t>(buf));
  ASSERT_TRUE(model.HasValue()) << model.Error().Message();
  EXPECT_THAT(GetTextEncoderSignatureMaps(*model),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(GetEncoderInputIndicesTest, IdentifiesAllInputs) {
  const std::vector<std::string> names = {"input_mask", "per_layer_embeddings",
                                          "embeddings"};
  const EncoderInputIndices indices = GetEncoderInputIndices(names);
  EXPECT_EQ(indices.embeddings, 2u);
  EXPECT_THAT(indices.input_mask, Optional(0));
  EXPECT_THAT(indices.per_layer_embeddings, Optional(1));
}

TEST(GetEncoderInputIndicesTest, DefaultsAndOmitsMissingOptionals) {
  const std::vector<std::string> names = {"tokens_input"};
  const EncoderInputIndices indices = GetEncoderInputIndices(names);
  EXPECT_EQ(indices.embeddings, 0u);
  EXPECT_EQ(indices.input_mask, std::nullopt);
  EXPECT_EQ(indices.per_layer_embeddings, std::nullopt);
}

TEST(GetMaxBatchedSequenceLengthTest, ReturnsZeroForEmptyOrInvalidSignatures) {
  EXPECT_EQ(GetMaxBatchedSequenceLength({}, {{2, {{8, 0}}}}), 0);
  EXPECT_EQ(GetMaxBatchedSequenceLength({{8, 0}}, {}), 0);
  EXPECT_EQ(GetMaxBatchedSequenceLength({{-4, 0}, {0, 1}}, {{2, {{8, 2}}}}), 0);
  EXPECT_EQ(
      GetMaxBatchedSequenceLength({{8, 0}}, {{-1, {{8, 1}}}, {1, {{8, 2}}}}),
      0);
  EXPECT_EQ(GetMaxBatchedSequenceLength({{8, 0}}, {{2, {}}}), 0);
}

TEST(GetMaxBatchedSequenceLengthTest, CapsByMaxSingleSignatureLength) {
  EXPECT_EQ(GetMaxBatchedSequenceLength(
                {{8, 0}, {16, 1}}, {{2, {{4, 2}, {32, 3}}}, {4, {{12, 4}}}}),
            16);
  EXPECT_EQ(GetMaxBatchedSequenceLength({{8, 0}, {32, 1}},
                                        {{2, {{4, 2}}}, {4, {{16, 3}}}}),
            16);
}

TEST(GetFloat32ElementsTest, ReturnsElementCountForFloat32Buffer) {
  ASSERT_OK_AND_ASSIGN(TensorBuffer buffer,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 8}), sizeof(float)));
  EXPECT_THAT(GetFloat32Elements(buffer, "embeddings"), IsOkAndHolds(64));
}

TEST(GetFloat32ElementsTest, RejectsNonFloat32Buffer) {
  ASSERT_OK_AND_ASSIGN(
      TensorBuffer buffer,
      MakeHostBuffer(ElementType::Int32, Dimensions({2, 4}), sizeof(int32_t)));
  EXPECT_THAT(GetFloat32Elements(buffer, "embeddings"),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ValidateBatchedSignatureTest, AcceptsValid2DAnd3DMasksWithPerLayer) {
  ASSERT_OK_AND_ASSIGN(TensorBuffer emb,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 8}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(
      TensorBuffer mask_2d,
      MakeHostBuffer(ElementType::Float32, Dimensions({2, 4}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(TensorBuffer mask_3d,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 1}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(TensorBuffer per_layer,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 3}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(
      TensorBuffer out,
      MakeHostBuffer(ElementType::Float32, Dimensions({2, 6}), sizeof(float)));

  const EncoderInputIndices indices{
      .embeddings = 0, .input_mask = 1, .per_layer_embeddings = 2};
  std::vector<TensorBuffer> inputs_2d;
  inputs_2d.push_back(std::move(emb));
  inputs_2d.push_back(std::move(mask_2d));
  inputs_2d.push_back(std::move(per_layer));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out));

  EXPECT_OK(ValidateBatchedSignature(indices, inputs_2d, outputs,
                                     /*batch_size=*/2, /*sequence_length=*/4,
                                     /*embedding_floats_per_token=*/8,
                                     /*per_layer_floats_per_token=*/3,
                                     /*single_output_elements=*/6));

  inputs_2d[1] = std::move(mask_3d);
  EXPECT_OK(ValidateBatchedSignature(indices, inputs_2d, outputs,
                                     /*batch_size=*/2, /*sequence_length=*/4,
                                     /*embedding_floats_per_token=*/8,
                                     /*per_layer_floats_per_token=*/3,
                                     /*single_output_elements=*/6));
}

TEST(ValidateBatchedSignatureTest, RejectsInvalidBatchSizeOrSequenceLength) {
  const EncoderInputIndices indices;
  EXPECT_THAT(
      ValidateBatchedSignature(indices, {}, {}, /*batch_size=*/1,
                               /*sequence_length=*/4, 8, std::nullopt, 6),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      ValidateBatchedSignature(indices, {}, {}, /*batch_size=*/2,
                               /*sequence_length=*/0, 8, std::nullopt, 6),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValidateBatchedSignatureTest, RejectsMissingPerLayerEmbedder) {
  ASSERT_OK_AND_ASSIGN(TensorBuffer emb,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 8}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(TensorBuffer per_layer,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 3}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(
      TensorBuffer out,
      MakeHostBuffer(ElementType::Float32, Dimensions({2, 6}), sizeof(float)));
  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(emb));
  inputs.push_back(std::move(per_layer));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out));

  const EncoderInputIndices indices{.embeddings = 0, .per_layer_embeddings = 1};
  EXPECT_THAT(ValidateBatchedSignature(indices, inputs, outputs,
                                       /*batch_size=*/2, /*sequence_length=*/4,
                                       /*embedding_floats_per_token=*/8,
                                       /*per_layer_floats_per_token=*/
                                       std::nullopt,
                                       /*single_output_elements=*/6),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ValidateBatchedSignatureTest, RejectsInvalidInputMaskRank) {
  ASSERT_OK_AND_ASSIGN(TensorBuffer emb,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 8}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(TensorBuffer bad_mask,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 1, 1}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(
      TensorBuffer out,
      MakeHostBuffer(ElementType::Float32, Dimensions({2, 6}), sizeof(float)));
  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(emb));
  inputs.push_back(std::move(bad_mask));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out));

  const EncoderInputIndices indices{.embeddings = 0, .input_mask = 1};
  EXPECT_THAT(
      ValidateBatchedSignature(indices, inputs, outputs,
                               /*batch_size=*/2, /*sequence_length=*/4,
                               /*embedding_floats_per_token=*/8, std::nullopt,
                               /*single_output_elements=*/6),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValidateBatchedSignatureTest, RejectsEmptyOrMismatchedOutput) {
  ASSERT_OK_AND_ASSIGN(TensorBuffer emb,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 8}), sizeof(float)));
  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(emb));
  const EncoderInputIndices indices{.embeddings = 0};

  EXPECT_THAT(
      ValidateBatchedSignature(indices, inputs, /*output_buffers=*/{},
                               /*batch_size=*/2, /*sequence_length=*/4,
                               /*embedding_floats_per_token=*/8, std::nullopt,
                               /*single_output_elements=*/6),
      StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK_AND_ASSIGN(
      TensorBuffer wrong_out,
      MakeHostBuffer(ElementType::Float32, Dimensions({2, 5}), sizeof(float)));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(wrong_out));
  EXPECT_THAT(
      ValidateBatchedSignature(indices, inputs, outputs, /*batch_size=*/2,
                               /*sequence_length=*/4,
                               /*embedding_floats_per_token=*/8, std::nullopt,
                               /*single_output_elements=*/6),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValidateBatchedSignatureTest, RejectsInputShapeAndTypeMismatch) {
  ASSERT_OK_AND_ASSIGN(TensorBuffer wrong_type,
                       MakeHostBuffer(ElementType::Int32, Dimensions({2, 4, 8}),
                                      sizeof(int32_t)));
  ASSERT_OK_AND_ASSIGN(
      TensorBuffer out,
      MakeHostBuffer(ElementType::Float32, Dimensions({2, 6}), sizeof(float)));
  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(wrong_type));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out));
  const EncoderInputIndices indices{.embeddings = 0};

  EXPECT_THAT(
      ValidateBatchedSignature(indices, inputs, outputs, /*batch_size=*/2,
                               /*sequence_length=*/4,
                               /*embedding_floats_per_token=*/8, std::nullopt,
                               /*single_output_elements=*/6),
      StatusIs(absl::StatusCode::kUnimplemented));

  ASSERT_OK_AND_ASSIGN(TensorBuffer wrong_shape,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 7}), sizeof(float)));
  inputs[0] = std::move(wrong_shape);
  EXPECT_THAT(
      ValidateBatchedSignature(indices, inputs, outputs, /*batch_size=*/2,
                               /*sequence_length=*/4,
                               /*embedding_floats_per_token=*/8, std::nullopt,
                               /*single_output_elements=*/6),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Same element count (2 x 4 x 8), but the batch and sequence dimensions are
  // swapped.
  ASSERT_OK_AND_ASSIGN(TensorBuffer swapped_dims,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({4, 2, 8}), sizeof(float)));
  inputs[0] = std::move(swapped_dims);
  EXPECT_THAT(
      ValidateBatchedSignature(indices, inputs, outputs, /*batch_size=*/2,
                               /*sequence_length=*/4,
                               /*embedding_floats_per_token=*/8, std::nullopt,
                               /*single_output_elements=*/6),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Same element count, but of rank 1.
  ASSERT_OK_AND_ASSIGN(
      TensorBuffer rank1,
      MakeHostBuffer(ElementType::Float32, Dimensions({64}), sizeof(float)));
  inputs[0] = std::move(rank1);
  EXPECT_THAT(
      ValidateBatchedSignature(indices, inputs, outputs, /*batch_size=*/2,
                               /*sequence_length=*/4,
                               /*embedding_floats_per_token=*/8, std::nullopt,
                               /*single_output_elements=*/6),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValidateBatchedSignatureTest, RejectsOutOfRangeInputIndex) {
  ASSERT_OK_AND_ASSIGN(TensorBuffer emb,
                       MakeHostBuffer(ElementType::Float32,
                                      Dimensions({2, 4, 8}), sizeof(float)));
  ASSERT_OK_AND_ASSIGN(
      TensorBuffer out,
      MakeHostBuffer(ElementType::Float32, Dimensions({2, 6}), sizeof(float)));
  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(emb));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out));

  EXPECT_THAT(
      ValidateBatchedSignature(EncoderInputIndices{.embeddings = 1}, inputs,
                               outputs, /*batch_size=*/2,
                               /*sequence_length=*/4,
                               /*embedding_floats_per_token=*/8, std::nullopt,
                               /*single_output_elements=*/6),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(ValidateBatchedSignature(
                  EncoderInputIndices{.embeddings = 0, .input_mask = 3}, inputs,
                  outputs, /*batch_size=*/2, /*sequence_length=*/4,
                  /*embedding_floats_per_token=*/8, std::nullopt,
                  /*single_output_elements=*/6),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(GroupInputsForBatchedSignaturesTest, NoSignatures) {
  EXPECT_THAT(GroupInputsForBatchedSignatures({1, 2}, {}, {}), IsEmpty());
  EXPECT_THAT(GroupInputsForBatchedSignatures({1, 2}, {}, {{2, {{4, 1}}}}),
              IsEmpty());
  EXPECT_THAT(GroupInputsForBatchedSignatures({1, 2}, {{4, 0}}, {}), IsEmpty());
  EXPECT_THAT(GroupInputsForBatchedSignatures({1, 2}, {{4, 0}}, {{2, {}}}),
              IsEmpty());
}

TEST(GroupInputsForBatchedSignaturesTest, IgnoresInvalidBatchSizeOrLength) {
  EXPECT_THAT(GroupInputsForBatchedSignatures(
                  {1, 2}, /*single_signatures=*/{{4, 0}},
                  /*batched_signatures=*/
                  {{-1, {{4, 1}}}, {0, {{4, 2}}}, {1, {{4, 3}}}}),
              IsEmpty());
  EXPECT_THAT(GroupInputsForBatchedSignatures(
                  {1, 2}, /*single_signatures=*/{{4, 0}},
                  /*batched_signatures=*/{{2, {{-4, 1}, {0, 2}}}}),
              IsEmpty());
  EXPECT_THAT(GroupInputsForBatchedSignatures(
                  {1, 2}, /*single_signatures=*/{{-4, 0}, {0, 1}},
                  /*batched_signatures=*/{{2, {{4, 2}}}}),
              IsEmpty());
  // Valid batched entries still match alongside invalid ones.
  EXPECT_THAT(
      GroupInputsForBatchedSignatures(
          {1, 2}, /*single_signatures=*/{{4, 0}},
          /*batched_signatures=*/{{0, {{4, 1}}}, {1, {{4, 2}}}, {2, {{4, 3}}}}),
      ElementsAre(BatchedTextEncoderGroup{.signature_index = 3,
                                          .padded_sequence_length = 4,
                                          .inputs = {0, 1}}));
}

TEST(GroupInputsForBatchedSignaturesTest, SkipsGroupsWithMorePadding) {
  // The four 2-token inputs would process 4 x 16 padded tokens on
  // `encoder_4x16` instead of 4 x 4 alone, but the four 10-token inputs fill
  // `encoder_4x16` with as many padded tokens as alone.
  EXPECT_THAT(
      GroupInputsForBatchedSignatures({2, 10, 2, 10, 2, 10, 2, 10},
                                      /*single_signatures=*/{{4, 0}, {16, 1}},
                                      /*batched_signatures=*/{{4, {{16, 2}}}}),
      ElementsAre(BatchedTextEncoderGroup{.signature_index = 2,
                                          .padded_sequence_length = 16,
                                          .inputs = {1, 3, 5, 7}}));
}

TEST(GroupInputsForBatchedSignaturesTest, PrefersLargerBatchesOnTies) {
  // Input 1 is not batchable (0), and input 3 is longer than any batched
  // signature.
  EXPECT_THAT(
      GroupInputsForBatchedSignatures(
          {3, 0, 1, 9, 2, 4, 1}, /*single_signatures=*/{{8, 0}, {16, 4}},
          /*batched_signatures=*/{{2, {{4, 1}, {8, 3}}}, {4, {{4, 2}}}}),
      ElementsAre(BatchedTextEncoderGroup{.signature_index = 2,
                                          .padded_sequence_length = 4,
                                          .inputs = {2, 6, 4, 0}}));
}

TEST(GroupInputsForBatchedSignaturesTest, CascadesBatchSizes) {
  EXPECT_THAT(GroupInputsForBatchedSignatures(
                  {1, 2, 3, 4, 1, 2, 3}, /*single_signatures=*/{{8, 0}},
                  /*batched_signatures=*/{{2, {{4, 1}}}, {4, {{4, 2}}}}),
              ElementsAre(BatchedTextEncoderGroup{.signature_index = 2,
                                                  .padded_sequence_length = 4,
                                                  .inputs = {0, 4, 1, 5}},
                          BatchedTextEncoderGroup{.signature_index = 1,
                                                  .padded_sequence_length = 4,
                                                  .inputs = {2, 6}}));
}

}  // namespace
}  // namespace litert::lm
