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

#include "omni/any2any/any2any_test_utils.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/mock_litert_runner.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "support/util/convert_tensor_buffer.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::testing::_;
using ::testing::NiceMock;

constexpr absl::string_view kTestdataDir =
    "litert_lm/runtime/testdata";

absl::StatusOr<std::string> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return absl::InternalError(absl::StrCat("Failed to open: ", path.string()));
  }
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

}  // namespace

std::string BuildMinimalTfliteModelBuffer(
    absl::Span<const TestTensorSpec> inputs,
    absl::Span<const TestTensorSpec> outputs) {
  flatbuffers::FlatBufferBuilder builder;

  flatbuffers::Offset<tflite::OperatorCode> opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  flatbuffers::Offset<
      flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>>
      opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  std::vector<int32_t> sg_inputs;
  std::vector<int32_t> sg_outputs;
  std::vector<flatbuffers::Offset<tflite::TensorMap>> inputs_map;
  std::vector<flatbuffers::Offset<tflite::TensorMap>> outputs_map;
  std::vector<flatbuffers::Offset<tflite::Operator>> ops;

  for (const auto& in : inputs) {
    int32_t idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(
        tflite::CreateTensor(builder, builder.CreateVector(in.dims), in.type,
                             /*buffer=*/0, builder.CreateString(in.name)));
    sg_inputs.push_back(idx);
    inputs_map.push_back(
        tflite::CreateTensorMap(builder, builder.CreateString(in.name), idx));
  }

  for (size_t j = 0; j < outputs.size(); ++j) {
    const auto& out = outputs[j];
    ABSL_CHECK_EQ(out.type, tflite::TensorType_FLOAT32);
    size_t num_elems = 1;
    for (int32_t d : out.dims) {
      num_elems *= static_cast<size_t>(d);
    }
    std::vector<uint8_t> zero_bytes(num_elems * sizeof(float), 0);
    uint32_t buf_a_idx = static_cast<uint32_t>(buffers.size());
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(zero_bytes)));
    uint32_t buf_b_idx = static_cast<uint32_t>(buffers.size());
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(zero_bytes)));

    int32_t const_a_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(out.dims), out.type, buf_a_idx,
        builder.CreateString(absl::StrCat("const_a_", j))));
    int32_t const_b_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(out.dims), out.type, buf_b_idx,
        builder.CreateString(absl::StrCat("const_b_", j))));
    int32_t out_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(
        tflite::CreateTensor(builder, builder.CreateVector(out.dims), out.type,
                             /*buffer=*/0, builder.CreateString(out.name)));
    sg_outputs.push_back(out_idx);
    outputs_map.push_back(tflite::CreateTensorMap(
        builder, builder.CreateString(out.name), out_idx));

    std::vector<int32_t> op_inputs = {const_a_idx, const_b_idx};
    std::vector<int32_t> op_outputs = {out_idx};
    ops.push_back(tflite::CreateOperator(builder, /*opcode_index=*/0,
                                         builder.CreateVector(op_inputs),
                                         builder.CreateVector(op_outputs)));
  }

  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors), builder.CreateVector(sg_inputs),
      builder.CreateVector(sg_outputs), builder.CreateVector(ops),
      builder.CreateString("main"));
  flatbuffers::Offset<tflite::SignatureDef> sig_def =
      tflite::CreateSignatureDef(builder, builder.CreateVector(inputs_map),
                                 builder.CreateVector(outputs_map),
                                 builder.CreateString("serving_default"),
                                 /*subgraph_index=*/0);
  flatbuffers::Offset<tflite::Model> model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector({subgraph}),
      builder.CreateString("test_model"), builder.CreateVector(buffers),
      /*metadata_buffer=*/0, /*metadata=*/0, builder.CreateVector({sig_def}));
  tflite::FinishModelBuffer(builder, model);

  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

absl::StatusOr<std::string> WriteAny2AnyTestBundle(
    absl::string_view filename, Any2AnyBundleOptions options) {
  const std::filesystem::path testdata_dir =
      std::filesystem::path(::testing::SrcDir()) / std::string(kTestdataDir);
  ABSL_ASSIGN_OR_RETURN(
      std::string llm_backbone_bytes,
      ReadFileBytes(testdata_dir / "checksum_fake_lm_embedding_input.tflite"));
  ABSL_ASSIGN_OR_RETURN(
      std::string text_embedder_bytes,
      ReadFileBytes(testdata_dir / "checksum_fake_lm_embedder.tflite"));

  lm::schema::ZlibBackendedSectionStream zlib_tok_stream(
      std::make_unique<lm::schema::StringBackedSectionStream>(
          std::move(options.tokenizer_json)));
  ABSL_RETURN_IF_ERROR(zlib_tok_stream.Prepare());
  std::stringstream tok_ss;
  tok_ss << zlib_tok_stream.GetStream().rdbuf();
  std::string compressed_tokenizer_bytes = tok_ss.str();

  std::vector<BundleSectionPayload> payloads;
  if (options.metadata.has_value()) {
    payloads.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_Any2AnyMetadataProto,
        .bytes = options.metadata->SerializeAsString(),
    });
  }
  payloads.push_back(BundleSectionPayload{
      .data_type = lm::schema::AnySectionDataType_HF_Tokenizer_Zlib,
      .bytes = std::move(compressed_tokenizer_bytes),
  });
  payloads.push_back(BundleSectionPayload{
      .data_type = lm::schema::AnySectionDataType_TFLiteModel,
      .model_type = "tf_lite_text_embedder",
      .backend_constraint = options.backbone_backend_constraint,
      .prefer_activation_type = options.backbone_prefer_activation_type,
      .bytes = std::move(text_embedder_bytes),
  });
  payloads.push_back(BundleSectionPayload{
      .data_type = lm::schema::AnySectionDataType_TFLiteModel,
      .model_type = "tf_lite_llm_backbone",
      .backend_constraint = std::move(options.backbone_backend_constraint),
      .prefer_activation_type =
          std::move(options.backbone_prefer_activation_type),
      .bytes = std::move(llm_backbone_bytes),
  });
  for (BundleSectionPayload& section : options.extra_sections) {
    payloads.push_back(std::move(section));
  }

  constexpr uint64_t kDataStartOffset = 16384;
  flatbuffers::FlatBufferBuilder builder(2048);
  std::vector<flatbuffers::Offset<lm::schema::SectionObject>> section_objects;
  uint64_t cursor = kDataStartOffset;
  for (const BundleSectionPayload& payload : payloads) {
    std::vector<flatbuffers::Offset<lm::schema::KeyValuePair>> items;
    if (!payload.model_type.empty()) {
      items.push_back(lm::schema::CreateKeyValuePair(builder, "model_type",
                                                     payload.model_type));
    }
    if (!payload.backend_constraint.empty()) {
      items.push_back(lm::schema::CreateKeyValuePair(
          builder, "backend_constraint", payload.backend_constraint));
    }
    if (!payload.prefer_activation_type.empty()) {
      items.push_back(lm::schema::CreateKeyValuePair(
          builder, "prefer_activation_type", payload.prefer_activation_type));
    }
    auto items_vec = items.empty() ? 0 : builder.CreateVector(items);
    const uint64_t begin = cursor;
    const uint64_t end = begin + payload.bytes.size();
    section_objects.push_back(lm::schema::CreateSectionObject(
        builder, items_vec, begin, end, payload.data_type));
    cursor = end;
  }
  auto section_metadata = lm::schema::CreateSectionMetadata(
      builder, builder.CreateVector(section_objects));
  auto metadata =
      lm::schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  const uint64_t header_end_offset = 32 + builder.GetSize();
  if (header_end_offset > kDataStartOffset) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Test bundle header size (", header_end_offset,
        ") exceeds reserved kDataStartOffset (", kDataStartOffset, ")."));
  }

  const std::filesystem::path out_path =
      std::filesystem::path(::testing::TempDir()) / std::string(filename);
  std::ofstream file(out_path, std::ios::binary);
  if (!file.good()) {
    return absl::InternalError("Failed to open test .litertlm file.");
  }
  file.write("LITERTLM", 8);
  const uint32_t major_version = 1;
  const uint32_t minor_version = 0;
  const uint32_t patch_version = 0;
  const uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&major_version), 4);
  file.write(reinterpret_cast<const char*>(&minor_version), 4);
  file.write(reinterpret_cast<const char*>(&patch_version), 4);
  file.write(reinterpret_cast<const char*>(&padding), 4);
  file.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  std::string header_pad(kDataStartOffset - header_end_offset, '\0');
  file.write(header_pad.data(), header_pad.size());
  for (const BundleSectionPayload& payload : payloads) {
    file.write(payload.bytes.data(), payload.bytes.size());
  }
  if (!file.good()) {
    return absl::InternalError("Failed to write test .litertlm file.");
  }
  return out_path.string();
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockVocoderRunner(
    int input_tokens, int output_samples,
    std::vector<std::vector<int32_t>>* recorded_windows,
    int samples_per_token) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault([input_tokens](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            TensorBuffer buf,
            support::CreateTensorBuffer<int32_t>({1, input_tokens}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([output_samples](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            TensorBuffer buf,
            support::CreateTensorBuffer<float>({1, output_samples}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [input_tokens, output_samples, recorded_windows, samples_per_token](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            LITERT_ASSIGN_OR_RETURN(
                std::vector<int32_t> codes,
                support::CopyFromTensorBuffer<int32_t>(inputs[0]));
            if (recorded_windows != nullptr) {
              recorded_windows->push_back(codes);
            }
            const int spt =
                samples_per_token > 0
                    ? samples_per_token
                    : ((input_tokens > 0 && output_samples >= input_tokens)
                           ? (output_samples / input_tokens)
                           : 1);
            std::vector<float> pcm(output_samples, 0.0f);
            for (int i = 0; i < output_samples; ++i) {
              const int tok_idx = i / spt;
              const int sample_in_tok = i % spt;
              const int code =
                  tok_idx < static_cast<int>(codes.size()) ? codes[tok_idx] : 0;
              pcm[i] = static_cast<float>(code * 10 + sample_in_tok);
            }
            TensorBuffer& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(pcm)));
            return absl::OkStatus();
          });
  return runner;
}

}  // namespace litert::omni::any2any
