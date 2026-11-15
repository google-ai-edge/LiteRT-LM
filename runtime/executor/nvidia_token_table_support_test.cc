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

#include "runtime/executor/nvidia_token_table_support.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

constexpr int32_t kVocab = 4;
constexpr int32_t kDim = 8;

// The describers read the embedders' graphs: models built as the converter
// shapes the embedders (a lookup of a quantized table, a multiply by a
// scalar), loaded through LiteRT as a model file would be.
class EmbedderBuilder {
 public:
  EmbedderBuilder() {
    buffers_.push_back(tflite::CreateBuffer(builder_));  // 0: no data
  }

  // An INT4 [vocab, dim] table with one FP32 scale per row: 0.5 (row + 1).
  int PerChannelTable(int32_t vocab, int32_t dim,
                      const std::vector<int64_t>& zero_points) {
    std::vector<float> scales(vocab);
    for (int32_t row = 0; row < vocab; ++row) scales[row] = 0.5f * (row + 1);
    const auto quantization = tflite::CreateQuantizationParameters(
        builder_, /*min=*/0, /*max=*/0, builder_.CreateVector(scales),
        builder_.CreateVector(zero_points), tflite::QuantizationDetails_NONE,
        /*details=*/0,
        /*quantized_dimension=*/0);
    return Tensor({vocab, dim}, tflite::TensorType_INT4,
                  Bytes(static_cast<size_t>(vocab) * dim / 2, 0x21),
                  quantization);
  }

  // An INT4 [vocab, dim] table with FP16 scales per block of `block`.
  int BlockwiseTable(int32_t vocab, int32_t dim, int32_t block) {
    const int scales =
        Tensor({vocab, dim / block}, tflite::TensorType_FLOAT16,
               Bytes(static_cast<size_t>(vocab) * (dim / block) * 2, 0x3c),
               /*quantization=*/0);
    const auto blockwise = tflite::CreateBlockwiseQuantization(
        builder_, scales, /*zero_points=*/-1, block);
    const auto quantization = tflite::CreateQuantizationParameters(
        builder_, 0, 0, 0, 0, tflite::QuantizationDetails_BlockwiseQuantization,
        blockwise.Union(), 0);
    return Tensor({vocab, dim}, tflite::TensorType_INT4,
                  Bytes(static_cast<size_t>(vocab) * dim / 2, 0x43),
                  quantization);
  }

  int Scalar(float value) {
    std::vector<uint8_t> bytes(sizeof(float));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return Tensor({}, tflite::TensorType_FLOAT32, bytes, 0);
  }

  // A lookup of `table` by a new INT32 ids input: its float32 output.
  int Lookup(int table, int32_t dim) {
    const int ids = Tensor({1}, tflite::TensorType_INT32, {}, 0);
    inputs_.push_back(ids);
    const int output = Tensor({1, dim}, tflite::TensorType_FLOAT32, {}, 0);
    Op(tflite::BuiltinOperator_EMBEDDING_LOOKUP, {ids, table}, output);
    return output;
  }

  // A binary op over `input` and `other`, its output of `dimensions`.
  int Binary(tflite::BuiltinOperator code, int input, int other,
             std::vector<int32_t> dimensions) {
    const int output = Tensor(dimensions, tflite::TensorType_FLOAT32, {}, 0);
    Op(code, {input, other}, output);
    return output;
  }

  int Concatenate(std::vector<int32_t> inputs, int32_t width) {
    const int output = Tensor({1, static_cast<int32_t>(inputs.size()) * width},
                              tflite::TensorType_FLOAT32, {}, 0);
    Op(tflite::BuiltinOperator_CONCATENATION, inputs, output);
    return output;
  }

  Model Finish(int output) {
    const auto subgraph = tflite::CreateSubGraph(
        builder_, builder_.CreateVector(tensors_),
        builder_.CreateVector(inputs_),
        builder_.CreateVector(std::vector<int32_t>{output}),
        builder_.CreateVector(ops_), builder_.CreateString("embedder"));
    const auto model = tflite::CreateModel(
        builder_, /*version=*/3, builder_.CreateVector(opcodes_),
        builder_.CreateVector({subgraph}), builder_.CreateString("test"),
        builder_.CreateVector(buffers_));
    tflite::FinishModelBuffer(builder_, model);
    file_.assign(builder_.GetBufferPointer(),
                 builder_.GetBufferPointer() + builder_.GetSize());
    auto loaded =
        Model::CreateFromBuffer(BufferRef<uint8_t>(file_.data(), file_.size()));
    EXPECT_TRUE(loaded.HasValue()) << loaded.Error().Message();
    return std::move(*loaded);
  }

  // The bytes of the table or scales tensor `index` was given.
  const std::vector<uint8_t>& BytesOf(int index) const {
    return bytes_of_[index];
  }

 private:
  static std::vector<uint8_t> Bytes(size_t count, uint8_t fill) {
    std::vector<uint8_t> bytes(count);
    for (size_t i = 0; i < count; ++i)
      bytes[i] = static_cast<uint8_t>(fill + i);
    return bytes;
  }

  int Tensor(std::vector<int32_t> shape, tflite::TensorType type,
             std::vector<uint8_t> bytes,
             flatbuffers::Offset<tflite::QuantizationParameters> quantization) {
    uint32_t buffer = 0;
    if (!bytes.empty()) {
      buffer = static_cast<uint32_t>(buffers_.size());
      buffers_.push_back(
          tflite::CreateBuffer(builder_, builder_.CreateVector(bytes)));
    }
    const int index = static_cast<int>(tensors_.size());
    tensors_.push_back(tflite::CreateTensor(
        builder_, builder_.CreateVector(shape), type, buffer,
        builder_.CreateString("t" + std::to_string(index)), quantization));
    bytes_of_.push_back(std::move(bytes));
    return index;
  }

  void Op(tflite::BuiltinOperator code, std::vector<int32_t> inputs,
          int output) {
    const uint32_t opcode = static_cast<uint32_t>(opcodes_.size());
    opcodes_.push_back(
        tflite::CreateOperatorCode(builder_, static_cast<int8_t>(code),
                                   /*custom_code=*/0, /*version=*/1, code));
    ops_.push_back(tflite::CreateOperator(
        builder_, opcode, builder_.CreateVector(inputs),
        builder_.CreateVector(std::vector<int32_t>{output})));
  }

  flatbuffers::FlatBufferBuilder builder_;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers_;
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors_;
  std::vector<flatbuffers::Offset<tflite::Operator>> ops_;
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes_;
  std::vector<int32_t> inputs_;
  std::vector<std::vector<uint8_t>> bytes_of_;
  std::vector<uint8_t> file_;  // the model's bytes, outliving the model
};

bool SameBytes(const void* data, size_t size,
               const std::vector<uint8_t>& expected) {
  return data != nullptr && size == expected.size() &&
         std::memcmp(data, expected.data(), size) == 0;
}

std::vector<float> RowScales(const void* scales, int32_t vocab) {
  std::vector<float> out(vocab);
  std::memcpy(out.data(), scales, sizeof(float) * vocab);
  return out;
}

TEST(NvidiaTokenTableSupportTest, DescribesAPerChannelTableAndItsMultiplier) {
  EmbedderBuilder builder;
  const int table = builder.PerChannelTable(kVocab, kDim, {0, 0, 0, 0});
  const int product =
      builder.Binary(tflite::BuiltinOperator_MUL, builder.Lookup(table, kDim),
                     builder.Scalar(2.5f), {1, kDim});
  const Model embedder = builder.Finish(product);

  auto described = DescribeNvidiaTokenTable(embedder);
  ASSERT_TRUE(described.ok()) << described.status();
  ASSERT_TRUE(described->has_value());
  const NvidiaTokenTable& found = **described;
  EXPECT_EQ(found.vocab, static_cast<uint32_t>(kVocab));
  EXPECT_EQ(found.dim, static_cast<uint32_t>(kDim));
  EXPECT_EQ(found.bits, 4u);
  EXPECT_TRUE(
      SameBytes(found.table, found.table_bytes, builder.BytesOf(table)));
  EXPECT_EQ(found.table_subgraph, 0u);
  EXPECT_EQ(found.table_tensor, static_cast<uint32_t>(table));
  ASSERT_EQ(found.scales_bytes, sizeof(float) * kVocab);
  EXPECT_EQ(RowScales(found.scales, kVocab),
            (std::vector<float>{0.5f, 1.0f, 1.5f, 2.0f}));
  EXPECT_EQ(found.scale_block, 0u);
  EXPECT_EQ(found.multiplier, 2.5f);
}

TEST(NvidiaTokenTableSupportTest, DescribesABlockwiseTableWithoutAMultiply) {
  EmbedderBuilder builder;
  const int table = builder.BlockwiseTable(kVocab, kDim, /*block=*/4);
  const Model embedder = builder.Finish(builder.Lookup(table, kDim));

  auto described = DescribeNvidiaTokenTable(embedder);
  ASSERT_TRUE(described.ok()) << described.status();
  ASSERT_TRUE(described->has_value());
  const NvidiaTokenTable& found = **described;
  EXPECT_TRUE(
      SameBytes(found.table, found.table_bytes, builder.BytesOf(table)));
  EXPECT_TRUE(SameBytes(found.scales, found.scales_bytes,
                        builder.BytesOf(table - 1)));  // the scales tensor
  EXPECT_EQ(found.scale_block, 4u);
  EXPECT_EQ(found.multiplier, 1.0f);  // the lookup is the output
}

TEST(NvidiaTokenTableSupportTest, DeclinesWhatIsNotAMultiplyAfterTheLookup) {
  EmbedderBuilder builder;
  const int table = builder.PerChannelTable(kVocab, kDim, {0, 0, 0, 0});
  const int sum =
      builder.Binary(tflite::BuiltinOperator_ADD, builder.Lookup(table, kDim),
                     builder.Scalar(1.0f), {1, kDim});
  const Model embedder = builder.Finish(sum);

  auto described = DescribeNvidiaTokenTable(embedder);
  ASSERT_TRUE(described.ok()) << described.status();
  EXPECT_FALSE(described->has_value());
}

TEST(NvidiaTokenTableSupportTest, DeclinesZeroPoints) {
  EmbedderBuilder builder;
  const int table = builder.PerChannelTable(kVocab, kDim, {0, 1, 0, 0});
  const Model embedder = builder.Finish(builder.Lookup(table, kDim));

  auto described = DescribeNvidiaTokenTable(embedder);
  ASSERT_TRUE(described.ok()) << described.status();
  EXPECT_FALSE(described->has_value());
}

TEST(NvidiaTokenTableSupportTest, DescribesPerLayerTablesAndTheirMultiplier) {
  constexpr int32_t kWidth = 6;
  EmbedderBuilder builder;
  const int first = builder.PerChannelTable(kVocab, kWidth, {0, 0, 0, 0});
  const int second = builder.PerChannelTable(kVocab, kWidth, {0, 0, 0, 0});
  const int joined = builder.Concatenate(
      {builder.Lookup(first, kWidth), builder.Lookup(second, kWidth)}, kWidth);
  const int product = builder.Binary(tflite::BuiltinOperator_MUL, joined,
                                     builder.Scalar(1.5f), {1, 2 * kWidth});
  const Model embedder = builder.Finish(product);

  auto described = DescribeNvidiaPerLayerTables(embedder);
  ASSERT_TRUE(described.ok()) << described.status();
  ASSERT_TRUE(described->has_value());
  const NvidiaPerLayerTables& found = **described;
  EXPECT_EQ(found.vocab, static_cast<uint32_t>(kVocab));
  EXPECT_EQ(found.width, static_cast<uint32_t>(kWidth));
  EXPECT_EQ(found.bits, 4u);
  ASSERT_EQ(found.tables.size(), 2u);
  EXPECT_TRUE(
      SameBytes(found.tables[0], found.table_bytes, builder.BytesOf(first)));
  EXPECT_TRUE(
      SameBytes(found.tables[1], found.table_bytes, builder.BytesOf(second)));
  ASSERT_EQ(found.row_scales.size(), 2u);
  EXPECT_EQ(RowScales(found.row_scales[0], kVocab),
            (std::vector<float>{0.5f, 1.0f, 1.5f, 2.0f}));
  EXPECT_EQ(RowScales(found.row_scales[1], kVocab),
            (std::vector<float>{0.5f, 1.0f, 1.5f, 2.0f}));
  EXPECT_EQ(found.multiplier, 1.5f);
}

TEST(NvidiaTokenTableSupportTest, DeclinesWhatIsNotAMultiplyAfterTheJoin) {
  constexpr int32_t kWidth = 6;
  EmbedderBuilder builder;
  const int first = builder.PerChannelTable(kVocab, kWidth, {0, 0, 0, 0});
  const int second = builder.PerChannelTable(kVocab, kWidth, {0, 0, 0, 0});
  const int joined = builder.Concatenate(
      {builder.Lookup(first, kWidth), builder.Lookup(second, kWidth)}, kWidth);
  const int sum = builder.Binary(tflite::BuiltinOperator_ADD, joined,
                                 builder.Scalar(1.0f), {1, 2 * kWidth});
  const Model embedder = builder.Finish(sum);

  auto described = DescribeNvidiaPerLayerTables(embedder);
  ASSERT_TRUE(described.ok()) << described.status();
  EXPECT_FALSE(described->has_value());
}

}  // namespace
}  // namespace litert::lm
