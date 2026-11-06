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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif  // defined(_WIN32)

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_model.h"  // from @litert
#include "litert/c/litert_model_types.h"  // from @litert
#include "litert/c/litert_op_code.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/vendors/nvidia/dispatch/token_table_c_api.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/nvidia_mtp_drafter_support.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

#if defined(_WIN32)
// The dispatch reads a table the model keeps out of memory with pread from a
// CRT descriptor. The model file is a native handle on Windows: the
// descriptor is one of a duplicate of it, closed once the registration
// returns (the dispatch reads the table during the call).
struct CrtDescriptor {
  int fd = -1;
  ~CrtDescriptor() {
    if (fd >= 0) _close(fd);
  }
};

absl::Status OpenCrtDescriptor(HANDLE file, CrtDescriptor& descriptor) {
  HANDLE duplicate = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), file, GetCurrentProcess(),
                       &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
    return absl::InternalError(
        absl::StrCat("the model file's handle cannot be duplicated: error ",
                     GetLastError()));
  }
  descriptor.fd = _open_osfhandle(reinterpret_cast<intptr_t>(duplicate),
                                  _O_RDONLY | _O_BINARY);
  if (descriptor.fd < 0) {
    CloseHandle(duplicate);
    return absl::InternalError("the model file has no CRT descriptor");
  }
  return absl::OkStatus();
}
#endif  // defined(_WIN32)

absl::Status LiteRtError(absl::string_view what, LiteRtStatus status) {
  return absl::InternalError(
      absl::StrCat(what, ": LiteRT status ", static_cast<int>(status)));
}

// The ranked type of a tensor.
absl::StatusOr<LiteRtRankedTensorType> RankedType(LiteRtTensor tensor) {
  LiteRtTensorTypeId type_id;
  if (LiteRtStatus status = LiteRtGetTensorTypeId(tensor, &type_id);
      status != kLiteRtStatusOk) {
    return LiteRtError("tensor type id", status);
  }
  if (type_id != kLiteRtRankedTensorType) {
    return absl::InvalidArgumentError("the tensor is not ranked");
  }
  LiteRtRankedTensorType type;
  if (LiteRtStatus status = LiteRtGetRankedTensorType(tensor, &type);
      status != kLiteRtStatusOk) {
    return LiteRtError("ranked tensor type", status);
  }
  return type;
}

// The bytes of a tensor's weights: none without weights.
struct Bytes {
  const void* data = nullptr;
  size_t size = 0;
};
Bytes WeightBytes(LiteRtTensor tensor) {
  Bytes bytes;
  LiteRtWeights weights = nullptr;
  if (LiteRtGetTensorWeights(tensor, &weights) != kLiteRtStatusOk ||
      weights == nullptr ||
      LiteRtGetWeightsBytes(weights, &bytes.data, &bytes.size) !=
          kLiteRtStatusOk ||
      bytes.data == nullptr) {
    return Bytes();
  }
  return bytes;
}

// The multiplier of the tfl.mul that follows the lookup's output through
// reshapes (its other input a float32 scalar constant): 1 when nothing
// follows the lookup (the model's output); nothing when something else does,
// since the feed would not write what the embedder computes.
std::optional<float> LookupMultiplier(LiteRtTensor output) {
  LiteRtTensor value = output;
  for (int hops = 0; hops < 4; ++hops) {
    LiteRtParamIndex num_uses = 0;
    if (LiteRtGetNumTensorUses(value, &num_uses) != kLiteRtStatusOk) {
      return std::nullopt;
    }
    if (num_uses == 0) return 1.0f;
    if (num_uses != 1) return std::nullopt;
    LiteRtOp user = nullptr;
    LiteRtParamIndex arg = 0;
    LiteRtOpCode code;
    if (LiteRtGetTensorUse(value, 0, &user, &arg) != kLiteRtStatusOk ||
        LiteRtGetOpCode(user, &code) != kLiteRtStatusOk) {
      return std::nullopt;
    }
    if (code == kLiteRtOpCodeTflReshape) {
      if (LiteRtGetOpOutput(user, 0, &value) != kLiteRtStatusOk) {
        return std::nullopt;
      }
      continue;
    }
    if (code != kLiteRtOpCodeTflMul) return std::nullopt;
    LiteRtParamIndex num_inputs = 0;
    LiteRtTensor other = nullptr;
    if (LiteRtGetNumOpInputs(user, &num_inputs) != kLiteRtStatusOk ||
        num_inputs != 2 ||
        LiteRtGetOpInput(user, arg == 0 ? 1 : 0, &other) != kLiteRtStatusOk) {
      return std::nullopt;
    }
    const Bytes scalar = WeightBytes(other);
    auto type = RankedType(other);
    if (!type.ok() || type->element_type != kLiteRtElementTypeFloat32 ||
        scalar.size != sizeof(float)) {
      return std::nullopt;
    }
    float multiplier = 1.0f;
    std::memcpy(&multiplier, scalar.data, sizeof(multiplier));
    return multiplier;
  }
  return std::nullopt;
}

// Per-channel zero points, when there are any, must all be zero: the
// symmetric lookup the CPU embedder runs.
bool ZeroPointsAreZero(const LiteRtQuantizationPerChannel& q) {
  if (q.zero_points == nullptr) return true;
  for (uint64_t i = 0; i < q.num_channels; ++i) {
    if (q.zero_points[i] != 0) return false;
  }
  return true;
}

// The first tfl.embedding_lookup of the model, or null, and its subgraph.
absl::StatusOr<LiteRtOp> FindEmbeddingLookup(LiteRtModel model,
                                             uint32_t* subgraph_index) {
  LiteRtParamIndex num_subgraphs = 0;
  if (LiteRtStatus status = LiteRtGetNumModelSubgraphs(model, &num_subgraphs);
      status != kLiteRtStatusOk) {
    return LiteRtError("subgraph count", status);
  }
  for (LiteRtParamIndex s = 0; s < num_subgraphs; ++s) {
    LiteRtSubgraph subgraph = nullptr;
    LiteRtParamIndex num_ops = 0;
    if (LiteRtGetModelSubgraph(model, s, &subgraph) != kLiteRtStatusOk ||
        LiteRtGetNumSubgraphOps(subgraph, &num_ops) != kLiteRtStatusOk) {
      return absl::InternalError("the embedder's subgraphs cannot be read");
    }
    for (LiteRtParamIndex i = 0; i < num_ops; ++i) {
      LiteRtOp op = nullptr;
      LiteRtOpCode code;
      if (LiteRtGetSubgraphOp(subgraph, i, &op) != kLiteRtStatusOk ||
          LiteRtGetOpCode(op, &code) != kLiteRtStatusOk) {
        return absl::InternalError("the embedder's ops cannot be read");
      }
      if (code == kLiteRtOpCodeTflEmbeddingLookup) {
        *subgraph_index = static_cast<uint32_t>(s);
        return op;
      }
    }
  }
  return nullptr;
}

// Where the file holds a table the model keeps out of memory: the tensor's
// external buffer in the model's flatbuffer, within its group's section of
// the model file. The resources' file (duplicated) and the offset in it, or
// nullopt when the table is not such a buffer.
struct ExternalTable {
  ExternalWeightResources resources;  // holds the descriptor open
  uint64_t offset = 0;
};
absl::StatusOr<std::optional<ExternalTable>> FindExternalTable(
    ModelResources& resources, const NvidiaTokenTable& table) {
  const std::string wire =
      TfLiteModelTypeToWireString(proto::LlmMetadata::TF_LITE_EMBEDDER);
  auto buffer = resources.GetTFLiteModelBuffer(wire);
  if (!buffer.ok()) return std::nullopt;
  const tflite::Model* model = tflite::GetModel(buffer->data());
  if (model == nullptr || model->subgraphs() == nullptr ||
      table.table_subgraph >= model->subgraphs()->size()) {
    return std::nullopt;
  }
  const tflite::SubGraph* subgraph =
      model->subgraphs()->Get(table.table_subgraph);
  if (subgraph == nullptr || subgraph->tensors() == nullptr ||
      table.table_tensor >= subgraph->tensors()->size() ||
      model->external_buffers() == nullptr ||
      model->external_buffer_groups() == nullptr) {
    return std::nullopt;
  }
  const uint32_t id =
      subgraph->tensors()->Get(table.table_tensor)->external_buffer();
  const tflite::ExternalBuffer* external = nullptr;
  for (const tflite::ExternalBuffer* candidate : *model->external_buffers()) {
    if (candidate != nullptr && candidate->id() == id) external = candidate;
  }
  if (external == nullptr ||
      external->group() >= model->external_buffer_groups()->size() ||
      external->length() != table.table_bytes) {
    return std::nullopt;
  }
  const tflite::ExternalBufferGroup* group =
      model->external_buffer_groups()->Get(external->group());
  if (group == nullptr || group->name() == nullptr) return std::nullopt;
  ExternalTable found;
  ABSL_ASSIGN_OR_RETURN(found.resources,
                        GetExternalWeightResources(resources, wire));
  if (!found.resources.scoped_file.has_value()) return std::nullopt;
  const auto section = found.resources.sections.find(group->name()->str());
  if (section == found.resources.sections.end() ||
      external->offset() + external->length() > section->second.length) {
    return std::nullopt;
  }
  found.offset = section->second.offset + external->offset();
  return found;
}

}  // namespace

absl::StatusOr<std::optional<NvidiaTokenTable>> DescribeNvidiaTokenTable(
    const litert::Model& embedder) {
  uint32_t subgraph_index = 0;
  ABSL_ASSIGN_OR_RETURN(LiteRtOp lookup,
                        FindEmbeddingLookup(embedder.Get(), &subgraph_index));
  if (lookup == nullptr) return std::nullopt;
  LiteRtParamIndex num_inputs = 0;
  LiteRtTensor table = nullptr;
  LiteRtTensor output = nullptr;
  if (LiteRtGetNumOpInputs(lookup, &num_inputs) != kLiteRtStatusOk ||
      num_inputs != 2 ||
      LiteRtGetOpInput(lookup, 1, &table) != kLiteRtStatusOk ||
      LiteRtGetOpOutput(lookup, 0, &output) != kLiteRtStatusOk) {
    return absl::InternalError(
        "the embedding lookup's operands cannot be read");
  }
  ABSL_ASSIGN_OR_RETURN(const LiteRtRankedTensorType table_type,
                        RankedType(table));
  const uint32_t bits = table_type.element_type == kLiteRtElementTypeInt4   ? 4
                        : table_type.element_type == kLiteRtElementTypeInt2 ? 2
                                                                            : 0;
  if (bits == 0 || table_type.layout.rank != 2 ||
      table_type.layout.dimensions[0] <= 0 ||
      table_type.layout.dimensions[1] <= 0 ||
      (table_type.layout.dimensions[1] * bits) % 8 != 0) {
    ABSL_LOG(INFO) << "NVIDIA token table: the embedder's table is not INT4 "
                      "or INT2 [vocab, dim]";
    return std::nullopt;
  }
  NvidiaTokenTable result;
  result.vocab = static_cast<uint32_t>(table_type.layout.dimensions[0]);
  result.dim = static_cast<uint32_t>(table_type.layout.dimensions[1]);
  result.bits = bits;
  result.table_bytes =
      static_cast<size_t>(result.vocab) * result.dim * bits / 8;
  result.table_subgraph = subgraph_index;
  if (LiteRtStatus status = LiteRtGetTensorIndex(table, &result.table_tensor);
      status != kLiteRtStatusOk) {
    return LiteRtError("tensor index", status);
  }
  const Bytes table_bytes = WeightBytes(table);
  if (table_bytes.size == result.table_bytes) {
    result.table = table_bytes.data;
  } else if (table_bytes.size != 0) {
    ABSL_LOG(INFO) << "NVIDIA token table: the embedder's table ["
                   << result.vocab << ", " << result.dim << "] holds "
                   << table_bytes.size << " bytes, not " << result.table_bytes;
    return std::nullopt;
  }
  LiteRtQuantizationTypeId quantization;
  if (LiteRtStatus status = LiteRtGetQuantizationTypeId(table, &quantization);
      status != kLiteRtStatusOk) {
    return LiteRtError("quantization type", status);
  }
  if (quantization == kLiteRtQuantizationPerChannel) {
    LiteRtQuantizationPerChannel q;
    if (LiteRtStatus status = LiteRtGetPerChannelQuantization(table, &q);
        status != kLiteRtStatusOk) {
      return LiteRtError("per-channel quantization", status);
    }
    if (q.quantized_dimension != 0 || q.num_channels != result.vocab ||
        q.scales == nullptr || !ZeroPointsAreZero(q)) {
      ABSL_LOG(INFO) << "NVIDIA token table: the embedder's scales are not "
                        "one per row without zero points";
      return std::nullopt;
    }
    result.scales = q.scales;
    result.scales_bytes = static_cast<size_t>(result.vocab) * sizeof(float);
    result.scale_block = 0;
  } else if (quantization == kLiteRtQuantizationBlockWise) {
    LiteRtQuantizationBlockWise q;
    if (LiteRtStatus status = LiteRtGetBlockWiseQuantization(table, &q);
        status != kLiteRtStatusOk) {
      return LiteRtError("blockwise quantization", status);
    }
    if (q.scales == nullptr || q.zero_points != nullptr || q.block_size <= 0 ||
        result.dim % q.block_size != 0) {
      ABSL_LOG(INFO) << "NVIDIA token table: the embedder's blockwise scales "
                        "have zero points or an odd block";
      return std::nullopt;
    }
    ABSL_ASSIGN_OR_RETURN(const LiteRtRankedTensorType scales_type,
                          RankedType(q.scales));
    const Bytes scales = WeightBytes(q.scales);
    const size_t blocks = result.dim / q.block_size;
    if (scales_type.element_type != kLiteRtElementTypeFloat16 ||
        scales.size !=
            static_cast<size_t>(result.vocab) * blocks * sizeof(uint16_t)) {
      ABSL_LOG(INFO) << "NVIDIA token table: the embedder's blockwise scales "
                        "are not FP16 [vocab, dim / block]";
      return std::nullopt;
    }
    result.scales = scales.data;
    result.scales_bytes = scales.size;
    result.scale_block = static_cast<uint32_t>(q.block_size);
  } else {
    ABSL_LOG(INFO) << "NVIDIA token table: the embedder's table is neither "
                      "per-channel nor blockwise quantized";
    return std::nullopt;
  }
  const std::optional<float> multiplier = LookupMultiplier(output);
  if (!multiplier.has_value()) {
    ABSL_LOG(INFO) << "NVIDIA token table: what follows the embedder's lookup "
                      "is not a multiply by a scalar";
    return std::nullopt;
  }
  result.multiplier = *multiplier;
  return result;
}

absl::StatusOr<std::optional<NvidiaPerLayerTables>>
DescribeNvidiaPerLayerTables(const litert::Model& per_layer_embedder) {
  // The concatenation of the lookups' outputs, in the first subgraph.
  LiteRtModel model = per_layer_embedder.Get();
  LiteRtParamIndex num_subgraphs = 0;
  if (LiteRtStatus status = LiteRtGetNumModelSubgraphs(model, &num_subgraphs);
      status != kLiteRtStatusOk || num_subgraphs == 0) {
    return std::nullopt;
  }
  LiteRtSubgraph subgraph = nullptr;
  LiteRtParamIndex num_ops = 0;
  if (LiteRtGetModelSubgraph(model, 0, &subgraph) != kLiteRtStatusOk ||
      LiteRtGetNumSubgraphOps(subgraph, &num_ops) != kLiteRtStatusOk) {
    return absl::InternalError("the per-layer embedder cannot be read");
  }
  LiteRtOp concatenation = nullptr;
  for (LiteRtParamIndex i = 0; i < num_ops && concatenation == nullptr; ++i) {
    LiteRtOp op = nullptr;
    LiteRtOpCode code;
    if (LiteRtGetSubgraphOp(subgraph, i, &op) != kLiteRtStatusOk ||
        LiteRtGetOpCode(op, &code) != kLiteRtStatusOk) {
      return absl::InternalError("the per-layer embedder's ops cannot be read");
    }
    if (code == kLiteRtOpCodeTflConcatenation) concatenation = op;
  }
  if (concatenation == nullptr) return std::nullopt;
  LiteRtParamIndex layers = 0;
  if (LiteRtGetNumOpInputs(concatenation, &layers) != kLiteRtStatusOk ||
      layers == 0) {
    return std::nullopt;
  }
  NvidiaPerLayerTables result;
  for (LiteRtParamIndex layer = 0; layer < layers; ++layer) {
    // Each input: a lookup's output, the lookup's table its second input.
    LiteRtTensor joined = nullptr;
    bool has_lookup = false;
    LiteRtTensorDefiningOp defining{};
    LiteRtOpCode code;
    LiteRtParamIndex num_inputs = 0;
    LiteRtTensor table = nullptr;
    if (LiteRtGetOpInput(concatenation, layer, &joined) != kLiteRtStatusOk ||
        LiteRtGetTensorDefiningOp(joined, &has_lookup, &defining) !=
            kLiteRtStatusOk ||
        !has_lookup || defining.op == nullptr ||
        LiteRtGetOpCode(defining.op, &code) != kLiteRtStatusOk ||
        code != kLiteRtOpCodeTflEmbeddingLookup ||
        LiteRtGetNumOpInputs(defining.op, &num_inputs) != kLiteRtStatusOk ||
        num_inputs != 2 ||
        LiteRtGetOpInput(defining.op, 1, &table) != kLiteRtStatusOk) {
      ABSL_LOG(INFO) << "NVIDIA per-layer tables: input " << layer
                     << " of the concatenation is not a lookup's output";
      return std::nullopt;
    }
    ABSL_ASSIGN_OR_RETURN(const LiteRtRankedTensorType type, RankedType(table));
    const uint32_t bits = type.element_type == kLiteRtElementTypeInt4   ? 4
                          : type.element_type == kLiteRtElementTypeInt2 ? 2
                                                                        : 0;
    if (bits == 0 || type.layout.rank != 2 || type.layout.dimensions[0] <= 0 ||
        type.layout.dimensions[1] <= 0) {
      ABSL_LOG(INFO) << "NVIDIA per-layer tables: table " << layer
                     << " is not INT4 or INT2 [vocab, width]";
      return std::nullopt;
    }
    const uint32_t vocab = static_cast<uint32_t>(type.layout.dimensions[0]);
    const uint32_t width = static_cast<uint32_t>(type.layout.dimensions[1]);
    if (layer == 0) {
      result.vocab = vocab;
      result.width = width;
      result.bits = bits;
      result.table_bytes = static_cast<size_t>(vocab) * width * bits / 8;
    } else if (vocab != result.vocab || width != result.width ||
               bits != result.bits) {
      ABSL_LOG(INFO) << "NVIDIA per-layer tables: table " << layer
                     << " differs in shape or type";
      return std::nullopt;
    }
    const Bytes bytes = WeightBytes(table);
    LiteRtQuantizationTypeId quantization;
    LiteRtQuantizationPerChannel q;
    if (bytes.size != result.table_bytes ||
        LiteRtGetQuantizationTypeId(table, &quantization) != kLiteRtStatusOk ||
        quantization != kLiteRtQuantizationPerChannel ||
        LiteRtGetPerChannelQuantization(table, &q) != kLiteRtStatusOk ||
        q.quantized_dimension != 0 || q.num_channels != vocab ||
        q.scales == nullptr || !ZeroPointsAreZero(q)) {
      ABSL_LOG(INFO) << "NVIDIA per-layer tables: table " << layer
                     << " is not loaded, or its scales are not one per row "
                        "without zero points";
      return std::nullopt;
    }
    result.tables.push_back(bytes.data);
    result.row_scales.push_back(q.scales);
  }
  LiteRtTensor joined = nullptr;
  if (LiteRtGetOpOutput(concatenation, 0, &joined) != kLiteRtStatusOk) {
    return absl::InternalError("the concatenation's output cannot be read");
  }
  const std::optional<float> multiplier = LookupMultiplier(joined);
  if (!multiplier.has_value()) {
    ABSL_LOG(INFO) << "NVIDIA per-layer tables: what follows the "
                      "concatenation is not a multiply by a scalar";
    return std::nullopt;
  }
  result.multiplier = *multiplier;
  return result;
}

absl::StatusOr<bool> RegisterNvidiaTokenTable(
    const litert::Environment& env, ModelResources& resources,
    const proto::ExecutorMetadata* executor_metadata) {
  ABSL_ASSIGN_OR_RETURN(auto library, LoadedDispatchLibrary(env));
  if (!library.has_value()) return false;
  using RegisterFn = decltype(&LiteRtDispatchNvidiaRegisterTokenTable);
  auto register_table = library->LookupSymbol<RegisterFn>(
      "LiteRtDispatchNvidiaRegisterTokenTable");
  if (!register_table.HasValue()) {
    ABSL_LOG(INFO) << "NVIDIA dispatch library takes no token table";
    return false;
  }
  // The sliding window, so the steps ahead fill the local mask right from
  // the first position past it.
  const AttentionMaskParams mask_params =
      GetAttentionMaskParams(executor_metadata);
  if (mask_params.sliding_window_size.has_value() &&
      *mask_params.sliding_window_size > 0) {
    using WindowFn = decltype(&LiteRtDispatchNvidiaStepFeedSetLocalWindow);
    auto set_window = library->LookupSymbol<WindowFn>(
        "LiteRtDispatchNvidiaStepFeedSetLocalWindow");
    if (set_window.HasValue()) {
      LITERT_RETURN_IF_ERROR(set_window.Value()(
          static_cast<uint32_t>(*mask_params.sliding_window_size)));
      ABSL_LOG(INFO) << "NVIDIA step feed: the sliding window of "
                     << *mask_params.sliding_window_size << " told";
    }
  } else {
    ABSL_LOG(INFO) << "NVIDIA step feed: the executor metadata names no "
                      "sliding window; the dispatch learns it from the mask";
  }
  // The per-layer embedder's tables, for the decode steps that await them.
  if (auto per_layer_embedder = resources.GetTFLiteModel(
          proto::LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER);
      per_layer_embedder.ok()) {
    ABSL_ASSIGN_OR_RETURN(auto tables,
                          DescribeNvidiaPerLayerTables(**per_layer_embedder));
    using PerLayerFn =
        decltype(&LiteRtDispatchNvidiaRegisterPerLayerTokenTables);
    auto register_per_layer = library->LookupSymbol<PerLayerFn>(
        "LiteRtDispatchNvidiaRegisterPerLayerTokenTables");
    if (tables.has_value() && register_per_layer.HasValue()) {
      LiteRtDispatchNvidiaPerLayerTokenTables record;
      record.vocab = tables->vocab;
      record.width = tables->width;
      record.layers = static_cast<uint32_t>(tables->tables.size());
      record.bits = tables->bits;
      record.tables = tables->tables.data();
      record.table_bytes = tables->table_bytes;
      record.row_scales = tables->row_scales.data();
      record.multiplier = tables->multiplier;
      const LiteRtStatus status = register_per_layer.Value()(&record);
      if (status == kLiteRtStatusOk) {
        ABSL_LOG(INFO) << "NVIDIA per-layer tables: " << record.layers
                       << " of [" << record.vocab << ", " << record.width
                       << "] INT" << record.bits
                       << " registered for the decode step (multiplier "
                       << record.multiplier << ")";
      } else if (status != kLiteRtStatusErrorNotFound) {
        return LiteRtError("registering the per-layer tables", status);
      }
    }
  }
  auto embedder =
      resources.GetTFLiteModel(proto::LlmMetadata::TF_LITE_EMBEDDER);
  if (!embedder.ok()) return false;
  ABSL_ASSIGN_OR_RETURN(auto table, DescribeNvidiaTokenTable(**embedder));
  if (!table.has_value()) return false;
  LiteRtDispatchNvidiaTokenTable record;
  record.vocab = table->vocab;
  record.dim = table->dim;
  record.bits = table->bits;
  record.table = table->table;
  record.table_bytes = table->table_bytes;
  record.table_fd = -1;
  record.table_offset = 0;
  std::optional<ExternalTable> external;
#if defined(_WIN32)
  CrtDescriptor crt_descriptor;  // outlives the registration below
#endif
  if (record.table == nullptr) {
    ABSL_ASSIGN_OR_RETURN(external, FindExternalTable(resources, *table));
    if (!external.has_value()) {
      ABSL_LOG(INFO) << "NVIDIA token table: the embedder's table ["
                     << table->vocab << ", " << table->dim
                     << "] is neither in memory nor an external buffer of the "
                        "model file";
      return false;
    }
#if defined(_WIN32)
    ABSL_RETURN_IF_ERROR(OpenCrtDescriptor(
        external->resources.scoped_file->file(), crt_descriptor));
    record.table_fd = crt_descriptor.fd;
#else
    record.table_fd = external->resources.scoped_file->file();
#endif
    record.table_offset = external->offset;
    ABSL_LOG(INFO) << "NVIDIA token table: the embedder's table ["
                   << table->vocab << ", " << table->dim
                   << "] is an external "
                      "buffer of the model file at offset "
                   << external->offset << " (" << table->table_bytes
                   << " bytes)";
  }
  record.scales = table->scales;
  record.scales_bytes = table->scales_bytes;
  record.scale_block = table->scale_block;
  record.multiplier = table->multiplier;
  const LiteRtStatus status = register_table.Value()(&record);
  if (status == kLiteRtStatusErrorNotFound) {
    ABSL_LOG(INFO) << "NVIDIA token table [" << table->vocab << ", "
                   << table->dim << "]: no decode step awaits it";
    return false;
  }
  if (status != kLiteRtStatusOk) {
    return LiteRtError("registering the token table", status);
  }
  ABSL_LOG(INFO) << "NVIDIA token table [" << table->vocab << ", " << table->dim
                 << "] INT" << table->bits
                 << " registered for the decode step ("
                 << (table->scale_block == 0
                         ? "row scales"
                         : absl::StrCat("block ", table->scale_block,
                                        " scales"))
                 << ", multiplier " << table->multiplier << ")";
  return true;
}

void RegisterNvidiaTokenTables(
    const litert::Environment& env, ModelResources& resources,
    const proto::ExecutorMetadata* executor_metadata) {
  if (auto registered =
          RegisterNvidiaTokenTable(env, resources, executor_metadata);
      !registered.ok()) {
    ABSL_LOG(WARNING) << "NVIDIA token table not registered: "
                      << registered.status();
  }
}

}  // namespace litert::lm
