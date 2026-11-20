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
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/executor/executor_stats.h"
#if !defined(LITERT_DISABLE_NPU)
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif  // !defined(LITERT_DISABLE_NPU)
#include "absl/strings/match.h"  // from @com_google_absl
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/cancellation_latch.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/streamed_weights_uploader.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

constexpr absl::string_view kEmbeddingModuleName = "Embedding";

constexpr absl::string_view kEmbedderLookupLatency = "Embedder lookup";
constexpr absl::string_view kMaskPreparationLatency = "Mask preparation";
constexpr absl::string_view kPerLayerEmbedderLookupLatency =
    "Per-layer embedder lookup";
constexpr absl::string_view kTextEncoderInferenceLatency =
    "Text encoder inference";
constexpr absl::string_view kOutputCopyLatency = "Output copy";

constexpr absl::string_view kEmbeddingNumTokensMetric =
    "(e2e) Embedding num tokens";
constexpr absl::string_view kEmbeddingNumPaddedTokensMetric =
    "(e2e) Embedding num padded tokens";
constexpr absl::string_view kEmbeddingE2eTokensPerSecondMetric =
    "(e2e) Embedding tokens per second";
constexpr absl::string_view kEmbeddingE2ePaddedTokensPerSecondMetric =
    "(e2e) Embedding tokens per second (padded)";
constexpr absl::string_view kEmbeddingEncoderStackTokensPerSecondMetric =
    "(EncoderStackOnly) Embedding tokens per second";
constexpr absl::string_view kEmbeddingEncoderStackPaddedTokensPerSecondMetric =
    "(EncoderStackOnly) Embedding tokens per second (padded)";
constexpr absl::string_view kTextEncoderMetricsPrefix = "Text encoder ";

constexpr absl::string_view kEncoderSignatureRunner = "encoder";
constexpr absl::string_view kPerLayerEmbeddingsName = "per_layer";
constexpr absl::string_view kInputMaskName = "input_mask";
constexpr absl::string_view kEmbeddingsName = "embeddings";
constexpr absl::string_view kTokenEmbeddingsName = "token_embeddings";
constexpr size_t kAppendedBufferAlignment = 64;

std::string LayerTapOutputName(int layer) {
  return absl::StrFormat("layer_%02d", layer);
}

absl::StatusOr<std::vector<int>> CanonicalizeHiddenStateLayers(
    absl::Span<const int> layers) {
  std::vector<int> canonical(layers.begin(), layers.end());
  for (int layer : canonical) {
    if (layer < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Hidden state layer index must be non-negative, got: ", layer));
    }
  }
  absl::c_sort(canonical);
  canonical.erase(std::unique(canonical.begin(), canonical.end()),
                  canonical.end());
  return canonical;
}

std::string BuildPerTokenCacheTag(bool token_embeddings,
                                  absl::Span<const int> layers) {
  std::string tag = ".pt";
  if (token_embeddings) {
    absl::StrAppend(&tag, "_te");
  }
  for (int layer : layers) {
    absl::StrAppend(&tag, absl::StrFormat("_l%02d", layer));
  }
  return tag;
}

// Returns true if every considered encoder signature in `model` already exposes
// all requested per-token output names.
absl::StatusOr<bool> EncoderSignaturesAlreadyExposeOutputs(
    const litert::Model& model,
    absl::Span<const std::string> selected_signatures,
    bool want_token_embeddings, absl::Span<const int> want_layers) {
  bool found_encoder_sig = false;
  for (size_t i = 0; i < model.GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(i));
    absl::string_view key = signature.Key();
    if (!absl::StartsWith(key, kEncoderSignatureRunner) ||
        (!selected_signatures.empty() &&
         !absl::c_linear_search(selected_signatures, key))) {
      continue;
    }
    found_encoder_sig = true;
    const auto output_names = signature.OutputNames();
    if (want_token_embeddings &&
        !absl::c_linear_search(output_names, kTokenEmbeddingsName)) {
      return false;
    }
    for (int layer : want_layers) {
      if (!absl::c_linear_search(output_names, LayerTapOutputName(layer))) {
        return false;
      }
    }
  }
  return found_encoder_sig;
}

tflite::BuiltinOperator GetOpBuiltinCode(const tflite::ModelT& model,
                                         const tflite::OperatorT& op) {
  if (op.opcode_index >= model.operator_codes.size() ||
      model.operator_codes[op.opcode_index] == nullptr) {
    return tflite::BuiltinOperator_CUSTOM;
  }
  const auto& oc = *model.operator_codes[op.opcode_index];
  return static_cast<tflite::BuiltinOperator>(
      std::max(static_cast<int32_t>(oc.builtin_code),
               static_cast<int32_t>(oc.deprecated_builtin_code)));
}

struct FinalNormAndProj {
  int32_t h_idx = -1;
  int32_t weight_idx = -1;
  const tflite::OperatorT* proj_op = nullptr;
};

// Walks back from output 0 (`encodings`) of `sig` through masked mean pooling
// (`FULLY_CONNECTED(W) <- DIV <- SUM <- MUL(h, mask)`) to locate the final
// normalized hidden-state tensor `h` and the bias-free pooling projection `W`.
absl::StatusOr<FinalNormAndProj> FindFinalNormAndProjection(
    const tflite::ModelT& model, const tflite::SubGraphT& sg,
    const tflite::SignatureDefT& sig) {
  if (sig.outputs.empty() || sig.outputs[0] == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("Encoder signature '", sig.signature_key,
                     "' has no pooled output at index 0."));
  }
  const int32_t out_idx = static_cast<int32_t>(sig.outputs[0]->tensor_index);
  if (out_idx < 0 || static_cast<size_t>(out_idx) >= sg.tensors.size() ||
      sg.tensors[out_idx] == nullptr) {
    return absl::FailedPreconditionError(
        "Encoder output 0 references an invalid tensor.");
  }
  absl::flat_hash_map<int32_t, const tflite::OperatorT*> producer;
  for (const auto& op : sg.operators) {
    if (op == nullptr) continue;
    for (int32_t out : op->outputs) {
      if (out >= 0) producer[out] = op.get();
    }
  }

  auto proj_it = producer.find(out_idx);
  if (proj_it == producer.end() ||
      GetOpBuiltinCode(model, *proj_it->second) !=
          tflite::BuiltinOperator_FULLY_CONNECTED) {
    return absl::FailedPreconditionError(
        "Encoder pooled output is not produced by FULLY_CONNECTED.");
  }
  const tflite::OperatorT* proj_op = proj_it->second;
  if (proj_op->inputs.size() < 2 || proj_op->inputs[0] < 0 ||
      proj_op->inputs[1] < 0) {
    return absl::FailedPreconditionError(
        "Pooling projection FULLY_CONNECTED has invalid inputs.");
  }
  if (proj_op->inputs.size() > 2 && proj_op->inputs[2] >= 0) {
    return absl::FailedPreconditionError(
        "Pooling projection has a bias; per-token projection requires a "
        "bias-free linear pooling head.");
  }
  const int32_t weight_idx = proj_op->inputs[1];
  if (static_cast<size_t>(weight_idx) >= sg.tensors.size() ||
      sg.tensors[weight_idx] == nullptr ||
      sg.tensors[weight_idx]->shape.size() != 2) {
    return absl::FailedPreconditionError(
        "Pooling projection weight tensor does not have rank 2.");
  }
  const int32_t hidden_dim = sg.tensors[weight_idx]->shape[1];

  int32_t cur = proj_op->inputs[0];
  const tflite::OperatorT* mul_op = nullptr;
  for (tflite::BuiltinOperator expected :
       {tflite::BuiltinOperator_DIV, tflite::BuiltinOperator_SUM,
        tflite::BuiltinOperator_MUL}) {
    auto it = producer.find(cur);
    const size_t min_inputs = expected == tflite::BuiltinOperator_MUL ? 2 : 1;
    if (it == producer.end() ||
        GetOpBuiltinCode(model, *it->second) != expected ||
        it->second->inputs.size() < min_inputs) {
      return absl::FailedPreconditionError(
          "Unexpected op in masked mean pooling chain.");
    }
    mul_op = it->second;
    cur = mul_op->inputs[0];
  }
  if (mul_op == nullptr || mul_op->inputs.size() < 2) {
    return absl::FailedPreconditionError(
        "Unexpected op in masked mean pooling chain.");
  }

  for (int k = 0; k < 2; ++k) {
    const int32_t cand = mul_op->inputs[k];
    if (cand >= 0 && static_cast<size_t>(cand) < sg.tensors.size() &&
        sg.tensors[cand] != nullptr && sg.tensors[cand]->shape.size() == 3 &&
        sg.tensors[cand]->shape[2] == hidden_dim) {
      return FinalNormAndProj{
          .h_idx = cand, .weight_idx = weight_idx, .proj_op = proj_op};
    }
  }
  return absl::FailedPreconditionError(
      "Could not identify rank-3 final normalized hidden state tensor.");
}

// Finds the per-layer residual-stream output tensor indices (`0..num_layers-1`)
// right after each layer's post-PLE residual ADD and `skip_scale` MUL.
absl::StatusOr<std::vector<int32_t>> FindLayerOutputs(
    const tflite::ModelT& model, const tflite::SubGraphT& sg) {
  absl::flat_hash_map<int32_t, std::vector<size_t>> consumers;
  const tflite::OperatorT* unpack_op = nullptr;
  int num_unpack_ops = 0;
  for (size_t idx = 0; idx < sg.operators.size(); ++idx) {
    const auto& op = sg.operators[idx];
    if (op == nullptr) continue;
    for (int32_t inp : op->inputs) {
      if (inp >= 0) consumers[inp].push_back(idx);
    }
    if (GetOpBuiltinCode(model, *op) == tflite::BuiltinOperator_UNPACK &&
        !op->outputs.empty()) {
      unpack_op = op.get();
      ++num_unpack_ops;
    }
  }
  if (num_unpack_ops != 1 || unpack_op == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected exactly 1 per-layer UNPACK op in encoder subgraph, found ",
        num_unpack_ops, "."));
  }

  std::vector<int32_t> layer_outs;
  layer_outs.reserve(unpack_op->outputs.size());
  for (size_t li = 0; li < unpack_op->outputs.size(); ++li) {
    int32_t cur = unpack_op->outputs[li];
    for (tflite::BuiltinOperator expected :
         {tflite::BuiltinOperator_MUL, tflite::BuiltinOperator_FULLY_CONNECTED,
          tflite::BuiltinOperator_STABLEHLO_COMPOSITE,
          tflite::BuiltinOperator_ADD, tflite::BuiltinOperator_MUL}) {
      auto it = consumers.find(cur);
      if (it == consumers.end() || it->second.empty()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Layer ", li, " incomplete PLE residual chain."));
      }
      const tflite::OperatorT& next = *sg.operators[it->second[0]];
      if (GetOpBuiltinCode(model, next) != expected || next.outputs.empty() ||
          next.outputs[0] < 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Layer ", li, " unexpected op in PLE chain."));
      }
      cur = next.outputs[0];
    }
    if (static_cast<size_t>(cur) >= sg.tensors.size() ||
        sg.tensors[cur] == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("Layer ", li, " skip_scale tensor index is invalid."));
    }
    auto next_it = consumers.find(cur);
    const bool feeds_composite =
        next_it != consumers.end() &&
        absl::c_any_of(next_it->second, [&](size_t ci) {
          return GetOpBuiltinCode(model, *sg.operators[ci]) ==
                 tflite::BuiltinOperator_STABLEHLO_COMPOSITE;
        });
    if (!feeds_composite) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Layer ", li,
          " skip_scale output does not feed STABLEHLO_COMPOSITE."));
    }
    layer_outs.push_back(cur);
  }
  return layer_outs;
}

absl::Span<uint8_t> AllocateAlignedBuffer(size_t size,
                                          std::vector<uint8_t>& storage) {
  storage.assign(size + kAppendedBufferAlignment, 0);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(storage.data());
  const uintptr_t aligned = (addr + kAppendedBufferAlignment - 1) &
                            ~(uintptr_t{kAppendedBufferAlignment} - 1);
  return absl::MakeSpan(reinterpret_cast<uint8_t*>(aligned), size);
}

// Verifies and unpacks `tflite_bytes`, injects missing `token_embeddings`
// and/or `layer_<NN>` outputs into active encoder signatures, and repacks into
// a 64-byte-aligned buffer (relocating any FlatBuffer-appended buffers while
// preserving mod-64 alignment and external buffer groups).
absl::StatusOr<std::unique_ptr<PatchedTextEncoderModel>>
BuildPatchedEncoderModel(absl::string_view tflite_bytes,
                         absl::Span<const std::string> selected_signatures,
                         bool want_token_embeddings,
                         absl::Span<const int> want_layers) {
  constexpr size_t kMaxFlatBufferVerifyBytes =
      std::numeric_limits<int32_t>::max();
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(tflite_bytes.data()),
      std::min(tflite_bytes.size(), kMaxFlatBufferVerifyBytes));
  if (tflite_bytes.size() < 8 ||
      !tflite::ModelBufferHasIdentifier(tflite_bytes.data()) ||
      !tflite::VerifyModelBuffer(verifier)) {
    return absl::FailedPreconditionError(
        "Invalid TFLite FlatBuffer for text encoder.");
  }
  std::unique_ptr<tflite::ModelT> tfl_model =
      tflite::UnPackModel(tflite_bytes.data());
  if (tfl_model == nullptr) {
    return absl::InternalError("Failed to unpack text encoder FlatBuffer.");
  }

  bool found_encoder_sig = false;
  absl::flat_hash_map<uint32_t, int32_t> injected_tok_emb_by_subgraph;
  absl::flat_hash_map<uint32_t, std::vector<int32_t>> layer_outs_by_subgraph;

  for (auto& sig : tfl_model->signature_defs) {
    if (sig == nullptr ||
        !absl::StartsWith(sig->signature_key, kEncoderSignatureRunner) ||
        (!selected_signatures.empty() &&
         !absl::c_linear_search(selected_signatures, sig->signature_key))) {
      continue;
    }
    found_encoder_sig = true;
    if (sig->subgraph_index >= tfl_model->subgraphs.size() ||
        tfl_model->subgraphs[sig->subgraph_index] == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("Encoder signature '", sig->signature_key,
                       "' references invalid subgraph index."));
    }
    tflite::SubGraphT& sg = *tfl_model->subgraphs[sig->subgraph_index];

    absl::flat_hash_set<std::string> existing_names;
    for (const auto& out_map : sig->outputs) {
      if (out_map != nullptr) existing_names.insert(out_map->name);
    }

    std::vector<std::pair<std::string, int32_t>> extra_outs;
    if (want_token_embeddings &&
        !existing_names.contains(kTokenEmbeddingsName)) {
      int32_t tok_tensor_idx = -1;
      if (auto it = injected_tok_emb_by_subgraph.find(sig->subgraph_index);
          it != injected_tok_emb_by_subgraph.end()) {
        tok_tensor_idx = it->second;
      } else {
        ABSL_ASSIGN_OR_RETURN(FinalNormAndProj norm_proj,
                              FindFinalNormAndProjection(*tfl_model, sg, *sig));
        const tflite::TensorT& h_tensor = *sg.tensors[norm_proj.h_idx];
        const int32_t emb_dim = sg.tensors[norm_proj.weight_idx]->shape[0];
        const int32_t pooled_out_idx =
            static_cast<int32_t>(sig->outputs[0]->tensor_index);

        auto new_tensor = std::make_unique<tflite::TensorT>();
        new_tensor->shape = {h_tensor.shape[0], h_tensor.shape[1], emb_dim};
        if (h_tensor.shape_signature.size() == 3) {
          new_tensor->shape_signature = {h_tensor.shape_signature[0],
                                         h_tensor.shape_signature[1], emb_dim};
        }
        new_tensor->type = sg.tensors[pooled_out_idx]->type;
        new_tensor->buffer = 0;
        new_tensor->name =
            absl::StrCat(sig->signature_key, "_", kTokenEmbeddingsName);
        sg.tensors.push_back(std::move(new_tensor));
        tok_tensor_idx = static_cast<int32_t>(sg.tensors.size() - 1);

        auto new_op = std::make_unique<tflite::OperatorT>();
        new_op->opcode_index = norm_proj.proj_op->opcode_index;
        new_op->inputs = {norm_proj.h_idx, norm_proj.weight_idx, -1};
        new_op->outputs = {tok_tensor_idx};
        tflite::FullyConnectedOptionsT fc_opts;
        if (const auto* src_opts =
                norm_proj.proj_op->builtin_options.AsFullyConnectedOptions();
            src_opts != nullptr) {
          fc_opts = *src_opts;
        }
        fc_opts.keep_num_dims = true;
        new_op->builtin_options.Set(fc_opts);
        sg.operators.push_back(std::move(new_op));
        injected_tok_emb_by_subgraph[sig->subgraph_index] = tok_tensor_idx;
      }
      extra_outs.emplace_back(std::string(kTokenEmbeddingsName),
                              tok_tensor_idx);
    }

    const bool need_layer_walk = absl::c_any_of(want_layers, [&](int layer) {
      return !existing_names.contains(LayerTapOutputName(layer));
    });
    if (need_layer_walk) {
      auto layer_it = layer_outs_by_subgraph.find(sig->subgraph_index);
      if (layer_it == layer_outs_by_subgraph.end()) {
        ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> found_layers,
                              FindLayerOutputs(*tfl_model, sg));
        layer_it = layer_outs_by_subgraph
                       .emplace(sig->subgraph_index, std::move(found_layers))
                       .first;
      }
      for (int layer : want_layers) {
        if (layer < 0 ||
            static_cast<size_t>(layer) >= layer_it->second.size()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Requested hidden state layer ", layer, " is out of range [0, ",
              layer_it->second.size(), ")."));
        }
        std::string tap_name = LayerTapOutputName(layer);
        if (!existing_names.contains(tap_name)) {
          extra_outs.emplace_back(std::move(tap_name), layer_it->second[layer]);
        }
      }
    }

    for (const auto& [name, tensor_idx] : extra_outs) {
      if (!absl::c_linear_search(sg.outputs, tensor_idx)) {
        sg.outputs.push_back(tensor_idx);
      }
      auto tensor_map = std::make_unique<tflite::TensorMapT>();
      tensor_map->name = name;
      tensor_map->tensor_index = static_cast<uint32_t>(tensor_idx);
      sig->outputs.push_back(std::move(tensor_map));
    }
  }

  if (!found_encoder_sig) {
    return absl::FailedPreconditionError(absl::StrCat(
        "No signature found with prefix ", kEncoderSignatureRunner));
  }

  std::vector<std::pair<uint64_t*, uint64_t>> relocated_offsets;
  uint64_t data_start = std::numeric_limits<uint64_t>::max();
  auto record_appended = [&](uint64_t& offset_field,
                             uint64_t region_size) -> absl::Status {
    if (offset_field <= 1) return absl::OkStatus();
    if (offset_field > tflite_bytes.size() ||
        region_size > tflite_bytes.size() - offset_field) {
      return absl::FailedPreconditionError(
          "Text encoder FlatBuffer has out-of-bounds appended buffer offset.");
    }
    data_start = std::min(data_start, offset_field);
    relocated_offsets.emplace_back(&offset_field, offset_field);
    return absl::OkStatus();
  };
  for (auto& buf : tfl_model->buffers) {
    if (buf != nullptr) {
      ABSL_RETURN_IF_ERROR(record_appended(buf->offset, buf->size));
    }
  }
  for (auto& sg : tfl_model->subgraphs) {
    if (sg == nullptr) continue;
    for (auto& op : sg->operators) {
      if (op != nullptr) {
        ABSL_RETURN_IF_ERROR(record_appended(op->large_custom_options_offset,
                                             op->large_custom_options_size));
      }
    }
  }

  const size_t init_builder_bytes = std::min<size_t>(
      relocated_offsets.empty() ? tflite_bytes.size()
                                : static_cast<size_t>(data_start),
      1024 * 1024);
  std::vector<uint8_t> storage;
  absl::Span<uint8_t> model_span;

  if (relocated_offsets.empty()) {
    flatbuffers::FlatBufferBuilder builder(init_builder_bytes);
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, tfl_model.get()));
    model_span = AllocateAlignedBuffer(builder.GetSize(), storage);
    std::memcpy(model_span.data(), builder.GetBufferPointer(),
                builder.GetSize());
  } else {
    const uint64_t align_rem = data_start % kAppendedBufferAlignment;
    uint64_t new_start = 0;
    bool stabilized = false;
    for (int iter = 0; iter < 4; ++iter) {
      flatbuffers::FlatBufferBuilder builder(init_builder_bytes);
      tflite::FinishModelBuffer(builder,
                                tflite::Model::Pack(builder, tfl_model.get()));
      const uint64_t fb_size = builder.GetSize();
      const uint64_t aligned =
          fb_size <= align_rem
              ? align_rem
              : align_rem +
                    (fb_size - align_rem + kAppendedBufferAlignment - 1) /
                        kAppendedBufferAlignment * kAppendedBufferAlignment;
      if (iter > 0 && aligned == new_start) {
        const size_t tail_size =
            tflite_bytes.size() - static_cast<size_t>(data_start);
        model_span = AllocateAlignedBuffer(
            static_cast<size_t>(new_start) + tail_size, storage);
        std::memcpy(model_span.data(), builder.GetBufferPointer(),
                    builder.GetSize());
        std::memcpy(model_span.data() + new_start,
                    tflite_bytes.data() + data_start, tail_size);
        stabilized = true;
        break;
      }
      new_start = aligned;
      for (auto& [field_ptr, orig_offset] : relocated_offsets) {
        *field_ptr = orig_offset - data_start + new_start;
      }
    }
    if (!stabilized) {
      return absl::InternalError(
          "Failed to stabilize appended buffer offsets during FlatBuffer "
          "repack.");
    }
  }

  ABSL_ASSIGN_OR_RETURN(litert::Model rewritten_model,
                        CreateLiteRtModelFromBuffer(litert::BufferRef<uint8_t>(
                            model_span.data(), model_span.size())));
  return absl::WrapUnique(new PatchedTextEncoderModel{
      .storage = std::move(storage),
      .model = std::move(rewritten_model),
  });
}

absl::Status AppendPerTokenOutputSlice(const litert::TensorBuffer& buffer,
                                       absl::string_view output_name,
                                       size_t padded_length,
                                       size_t active_tokens,
                                       std::vector<float>& dst, int& dst_dim) {
  LITERT_ASSIGN_OR_RETURN(auto all_rows, CopyFromTensorBuffer<float>(buffer));
  if (padded_length == 0 || all_rows.empty() ||
      all_rows.size() % padded_length != 0) {
    return absl::InternalError(absl::StrCat(
        "Unexpected '", output_name, "' output size ", all_rows.size(),
        " for padded sequence length ", padded_length, "."));
  }
  if (active_tokens > padded_length) {
    return absl::InternalError(absl::StrCat(
        "Active token count (", active_tokens,
        ") exceeds padded sequence length (", padded_length, ")."));
  }
  const int dim = static_cast<int>(all_rows.size() / padded_length);
  if (dst_dim == 0) {
    dst_dim = dim;
  } else if (dst_dim != dim) {
    return absl::InternalError(absl::StrCat("Inconsistent '", output_name,
                                            "' dimension across chunks (",
                                            dst_dim, " vs ", dim, ")."));
  }
  dst.insert(dst.end(), all_rows.begin(),
             all_rows.begin() + active_tokens * static_cast<size_t>(dim));
  return absl::OkStatus();
}

class CompiledModelWrapper : public litert::CompiledModel {
 public:
  static Expected<CompiledModel> Create(litert::Environment& env,
                                        const LiteRtModel litert_model,
                                        Options& compilation_options) {
    return litert::CompiledModel::Create(env, litert_model,
                                         compilation_options);
  }
};

absl::StatusOr<std::map<int, size_t>> GetTextEncoderSignatureMap(
    const litert::Model& model,
    const std::vector<std::string>& selected_signatures = {}) {
  std::map<int, size_t> encoder_signatures;

  for (int i = 0; i < model.GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(i));
    absl::string_view key = signature.Key();
    if (absl::StartsWith(key, kEncoderSignatureRunner)) {
      if (!selected_signatures.empty() &&
          !absl::c_linear_search(selected_signatures, key)) {
        continue;
      }
      if (signature.InputNames().empty()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Text encoder signature has no input tensors: ", key));
      }

      // Find the embeddings input tensor type to get sequence length.
      std::optional<RankedTensorType> embeddings_tensor_type;
      for (auto name : signature.InputNames()) {
        if (absl::StrContains(name, kEmbeddingsName)) {
          LITERT_ASSIGN_OR_RETURN(embeddings_tensor_type,
                                  signature.InputTensorType(name));
          break;
        }
      }

      // Fallback to the first input tensor if kEmbeddingsName was not found.
      if (!embeddings_tensor_type.has_value()) {
        LITERT_ASSIGN_OR_RETURN(embeddings_tensor_type,
                                signature.InputTensorType(0));
      }

      int sequence_length = 0;
      if (embeddings_tensor_type->Layout().Rank() == 3) {
        // [batch_size, seq_len, embed_dim]
        sequence_length = embeddings_tensor_type->Layout().Dimensions()[1];
      } else if (embeddings_tensor_type->Layout().Rank() == 2) {
        // [seq_len, embed_dim]
        sequence_length = embeddings_tensor_type->Layout().Dimensions()[0];
      } else {
        return absl::FailedPreconditionError(
            absl::StrCat("Embedding input tensor has unsupported dimension: ",
                         embeddings_tensor_type->Layout().Rank()));
      }

      encoder_signatures[sequence_length] = i;
    }
  }

  if (encoder_signatures.empty()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "No signature found with prefix ", kEncoderSignatureRunner));
  }

  return encoder_signatures;
}

}  // namespace

// static
absl::StatusOr<CompiledTextEncoderInfo>
EmbeddingLiteRtCompiledModelExecutor::CompileTextEncoder(
    const EmbeddingExecutorSettings& executor_settings, Environment& env,
    ModelResources& resources, std::shared_ptr<DataStream> weights_stream) {
  ABSL_ASSIGN_OR_RETURN(
      const litert::Model* text_encoder_model,
      resources.GetTFLiteModel(proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER));
  RET_CHECK_NE(text_encoder_model, nullptr) << "TEXT_ENCODER model is null.";

  const PerTokenOutputsConfig& per_token_cfg =
      executor_settings.GetPerTokenOutputs();
  ABSL_ASSIGN_OR_RETURN(
      std::vector<int> canonical_layers,
      CanonicalizeHiddenStateLayers(per_token_cfg.hidden_state_layers));

  std::unique_ptr<PatchedTextEncoderModel> patched_model;
  std::string encoder_cache_name(EmbeddingExecutorSettings::kEncoderName);
  if (per_token_cfg.IsEnabled()) {
    ABSL_ASSIGN_OR_RETURN(
        bool already_exposed,
        EncoderSignaturesAlreadyExposeOutputs(
            *text_encoder_model, executor_settings.GetSelectedSignatures(),
            per_token_cfg.token_embeddings, canonical_layers));
    if (!already_exposed) {
      ABSL_ASSIGN_OR_RETURN(
          absl::string_view raw_model_bytes,
          resources.GetTFLiteModelBuffer(
              proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER));
      ABSL_ASSIGN_OR_RETURN(
          patched_model,
          BuildPatchedEncoderModel(
              raw_model_bytes, executor_settings.GetSelectedSignatures(),
              per_token_cfg.token_embeddings, canonical_layers));
      text_encoder_model = &patched_model->model;
      absl::StrAppend(&encoder_cache_name,
                      BuildPerTokenCacheTag(per_token_cfg.token_embeddings,
                                            canonical_layers));
    }
  }

  litert::Options options;
  switch (executor_settings.GetBackend()) {
    case Backend::CPU: {
      LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                              options.GetOptions<::litert::CpuOptions>());
      LITERT_RETURN_IF_ERROR(
          SetCpuOptions(cpu_options, executor_settings.GetNumThreads()));
      auto weight_cache_file = executor_settings.GetWeightCacheFile(
          absl::StrCat(encoder_cache_name,
                       ExecutorSettingsBase::kXnnpackCacheSuffix),
          /*check_and_clean=*/true);
      ABSL_RETURN_IF_ERROR(SetCpuCacheOptions(
          weight_cache_file,
          /*logging_prefix=*/encoder_cache_name, cpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                              options.GetOptions<::litert::GpuOptions>());
      ABSL_ASSIGN_OR_RETURN(
          const auto cache_files,
          GetGpuModelCacheData(executor_settings, encoder_cache_name));
      LITERT_RETURN_IF_ERROR(
          SetCommonGpuOptions(executor_settings, gpu_options));
      // The text encoder is compiled for the GPU only, so it is always fully
      // delegated to a single delegate. This lets TFLite skip preparing the
      // CPU kernels before delegation, which otherwise allocates a CPU arena
      // for every STABLEHLO_COMPOSITE decomposition subgraph that is never
      // released.
      gpu_options.SetHintFullyDelegatedToSingleDelegate(true);
      ABSL_RETURN_IF_ERROR(SetGpuCacheOptions(
          cache_files.weight_cache_file, cache_files.program_cache_file,
          cache_files.cache_key,
          /*logging_prefix=*/encoder_cache_name,
          /*cache_compiled_shaders_only=*/false, gpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
#if !defined(LITERT_DISABLE_NPU)
    case Backend::NPU: {
      LITERT_ASSIGN_OR_RETURN(auto& qualcomm_options,
                              options.GetOptions<qualcomm::QualcommOptions>());
      qualcomm_options.SetLogLevel(qualcomm::QualcommOptions::LogLevel::kOff);
      qualcomm_options.SetHtpFileReadMemoryBudgetMB(16);
      ABSL_RETURN_IF_ERROR(
          SetNpuOptions(executor_settings.GetNpuOptions(), options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kNpu |
                                      litert::HwAccelerators::kCpu);
      break;
    }
#endif  // !defined(LITERT_DISABLE_NPU)
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ",
                       GetBackendString(executor_settings.GetBackend())));
  }
  ABSL_RETURN_IF_ERROR(SetExternalWeightOptions(
      resources, proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER, options));

  if (!executor_settings.GetSelectedSignatures().empty()) {
    std::vector<absl::string_view> selected_signatures;
    selected_signatures.reserve(
        executor_settings.GetSelectedSignatures().size());
    for (const auto& sig : executor_settings.GetSelectedSignatures()) {
      selected_signatures.push_back(sig);
    }
    LITERT_ASSIGN_OR_RETURN(auto& runtime_options, options.GetRuntimeOptions());
    LITERT_RETURN_IF_ERROR(
        runtime_options.SetSelectedSignatures(selected_signatures));
  }

  // The uploader must outlive compilation. Destroying it afterwards drops the
  // weights stream, which discards whatever compilation didn't consume.
  std::unique_ptr<StreamedWeightsUploader> weights_uploader;
  if (weights_stream != nullptr) {
    weights_uploader =
        std::make_unique<StreamedWeightsUploader>(std::move(weights_stream));
    weights_uploader->SetOnOptions(options);
  }
  auto compiled_model_status_or =
      CompiledModelWrapper::Create(env, text_encoder_model->Get(), options);
  weights_uploader.reset();
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          std::move(compiled_model_status_or));
  auto compiled_model_ptr =
      std::make_unique<litert::CompiledModel>(std::move(compiled_model));

  ABSL_ASSIGN_OR_RETURN(
      auto encoder_signatures,
      GetTextEncoderSignatureMap(*text_encoder_model,
                                 executor_settings.GetSelectedSignatures()));
  size_t default_signature_index = encoder_signatures.begin()->second;

  LITERT_ASSIGN_OR_RETURN(
      auto input_tensor_type,
      text_encoder_model->GetInputTensorType(default_signature_index, 0));
  const auto& input_dims = input_tensor_type.Layout().Dimensions();
  std::vector<int> expected_input_dimension(input_dims.begin(),
                                            input_dims.end());

  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor_type,
      text_encoder_model->GetOutputTensorType(default_signature_index, 0));
  const auto& output_dims = output_tensor_type.Layout().Dimensions();
  RET_CHECK(!output_dims.empty()) << "Output dimensions cannot be empty.";
  int embedding_dimension = output_dims.back();

  absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
      input_buffers_cache;
  absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
      output_buffers_cache;
  for (const auto& [seq_len, signature_index] : encoder_signatures) {
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffers,
        compiled_model_ptr->CreateInputBuffers(signature_index));
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffers,
        compiled_model_ptr->CreateOutputBuffers(signature_index));

    input_buffers_cache[signature_index] = std::move(input_buffers);
    output_buffers_cache[signature_index] = std::move(output_buffers);
  }

  CompiledTextEncoderInfo info;
  info.patched_model = std::move(patched_model);
  info.compiled_model = std::move(compiled_model_ptr);
  info.expected_input_dimension = std::move(expected_input_dimension);
  info.embedding_dimension = embedding_dimension;
  info.encoder_signatures = std::move(encoder_signatures);
  info.input_buffers_cache = std::move(input_buffers_cache);
  info.output_buffers_cache = std::move(output_buffers_cache);
  return info;
}

// static
absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
EmbeddingLiteRtCompiledModelExecutor::Create(
    EmbeddingExecutorSettings executor_settings, Environment& env,
    std::shared_ptr<ModelResources> resources) {
  RET_CHECK_NE(resources, nullptr) << "ModelResources is null.";

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  ABSL_RETURN_IF_ERROR(InitializeEmbeddingLookups(
      env, *resources, embedding_lookup, per_layer_embedding_lookup));
  if (embedding_lookup == nullptr) {
    ABSL_LOG(INFO) << "kTfLiteEmbedder model not found in resources. The "
                      "executor only supports pre-computed input embeddings.";
  }

  ABSL_ASSIGN_OR_RETURN(auto compiled_text_encoder_info,
                        CompileTextEncoder(executor_settings, env, *resources));

  return Create(std::move(executor_settings), env, std::move(resources),
                std::move(embedding_lookup),
                std::move(per_layer_embedding_lookup),
                std::move(compiled_text_encoder_info));
}

// static
absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
EmbeddingLiteRtCompiledModelExecutor::Create(
    EmbeddingExecutorSettings executor_settings, Environment& env,
    std::shared_ptr<ModelResources> resources,
    std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
    CompiledTextEncoderInfo compiled_text_encoder_info) {
  RET_CHECK_NE(compiled_text_encoder_info.compiled_model, nullptr)
      << "The compiled text encoder is null.";
  return absl::WrapUnique(new EmbeddingLiteRtCompiledModelExecutor(
      std::move(executor_settings), env, std::move(resources),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      std::move(compiled_text_encoder_info.patched_model),
      std::move(compiled_text_encoder_info.compiled_model),
      std::move(compiled_text_encoder_info.expected_input_dimension),
      compiled_text_encoder_info.embedding_dimension,
      std::move(compiled_text_encoder_info.encoder_signatures),
      std::move(compiled_text_encoder_info.input_buffers_cache),
      std::move(compiled_text_encoder_info.output_buffers_cache)));
}

// static
absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
EmbeddingLiteRtCompiledModelExecutor::Create(
    EmbeddingExecutorSettings executor_settings, Environment& env) {
  LITERT_ASSIGN_OR_RETURN(
      auto resources,
      BuildLiteRtCompiledModelResources(executor_settings.GetModelAssets()));
  return Create(std::move(executor_settings), env, std::move(resources));
}

absl::StatusOr<EmbeddingLiteRtCompiledModelExecutor::EncoderRun>
EmbeddingLiteRtCompiledModelExecutor::PrepareEncoderRun(
    size_t sequence_length) {
  // Find the smallest signature larger than or equal to the input size.
  auto it = encoder_signatures_.lower_bound(sequence_length);
  if (it == encoder_signatures_.end()) {
    return absl::InternalError(absl::StrCat(
        "No suitable signature found for sequence length ", sequence_length));
  }
  EncoderRun run;
  run.signature_index = it->second;
  run.padded_sequence_length = it->first;

  // Lazily create input buffers.
  if (!input_buffers_cache_.contains(run.signature_index)) {
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffers,
        compiled_model_->CreateInputBuffers(run.signature_index));
    input_buffers_cache_[run.signature_index] = std::move(input_buffers);
  }

  // Lazily create output buffers.
  if (!output_buffers_cache_.contains(run.signature_index)) {
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffers,
        compiled_model_->CreateOutputBuffers(run.signature_index));
    output_buffers_cache_[run.signature_index] = std::move(output_buffers);
  }
  run.input_buffers = &input_buffers_cache_[run.signature_index];
  run.output_buffers = &output_buffers_cache_[run.signature_index];

  LITERT_ASSIGN_OR_RETURN(
      auto input_names,
      compiled_model_->GetSignatureInputNames(run.signature_index));
  for (size_t i = 0; i < input_names.size(); ++i) {
    absl::string_view name = input_names[i];
    if (absl::StrContains(name, kPerLayerEmbeddingsName)) {
      run.per_layer_embeddings_buffer_index = i;
    } else if (absl::StrContains(name, kInputMaskName)) {
      run.input_mask_buffer_index = i;
    } else if (absl::StrContains(name, kEmbeddingsName)) {
      run.embeddings_buffer_index = i;
    }
  }
  return run;
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::FinishEncoderRun(
    const EncoderRun& run, size_t sequence_length, CancellationLatch* latch,
    PerTokenOutputCollector* per_token_collector) {
  std::vector<litert::TensorBuffer>& input_buffers = *run.input_buffers;
  std::vector<litert::TensorBuffer>& output_buffers = *run.output_buffers;

  std::optional<size_t> token_embeddings_output_index;
  std::vector<std::pair<int, size_t>> hidden_state_output_indices;
  if (per_token_collector != nullptr &&
      (per_token_collector->collect_token_embeddings ||
       !per_token_collector->hidden_state_layers.empty())) {
    LITERT_ASSIGN_OR_RETURN(
        auto output_names,
        compiled_model_->GetSignatureOutputNames(run.signature_index));
    if (per_token_collector->collect_token_embeddings) {
      auto name_it = absl::c_find(output_names, kTokenEmbeddingsName);
      if (name_it == output_names.end()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "The text encoder has no '", kTokenEmbeddingsName,
            "' output; enable PerTokenOutputsConfig::token_embeddings at model "
            "load time."));
      }
      token_embeddings_output_index =
          static_cast<size_t>(std::distance(output_names.begin(), name_it));
    }
    hidden_state_output_indices.reserve(
        per_token_collector->hidden_state_layers.size());
    for (int layer : per_token_collector->hidden_state_layers) {
      const std::string tap_name = LayerTapOutputName(layer);
      auto name_it = absl::c_find(output_names, tap_name);
      if (name_it == output_names.end()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "The text encoder has no '", tap_name, "' output; add layer ",
            layer,
            " to PerTokenOutputsConfig::hidden_state_layers at model load "
            "time."));
      }
      hidden_state_output_indices.emplace_back(
          layer,
          static_cast<size_t>(std::distance(output_names.begin(), name_it)));
    }
  }

  // Compute mask for text encoder. The mask masks out tokens outside of the
  // input sequence length, which can be smaller than the static size of the
  // TFLite graph.
  if (run.input_mask_buffer_index.has_value()) {
    ScopedLatency scoped(latency_stats_, kMaskPreparationLatency);
    auto& mask_buffer = input_buffers[*run.input_mask_buffer_index];
    LITERT_ASSIGN_OR_RETURN(
        auto mask_lock_and_addr,
        litert::TensorBufferScopedLock::Create(
            mask_buffer, litert::TensorBuffer::LockMode::kWrite));
    float* mask_ptr = static_cast<float*>(mask_lock_and_addr.second);
    LITERT_ASSIGN_OR_RETURN(auto mask_size_bytes, mask_buffer.Size());
    size_t mask_elements = mask_size_bytes / sizeof(float);

    const size_t active_elements = std::min(sequence_length, mask_elements);
    std::fill_n(mask_ptr, active_elements, 1.0f);
    if (active_elements < mask_elements) {
      std::fill(mask_ptr + active_elements, mask_ptr + mask_elements, 0.0f);
    }
  }

  if (latch != nullptr && latch->IsCancelled()) {
    return absl::CancelledError("The text encoder run was cancelled.");
  }
  {
    ScopedLatency scoped(latency_stats_, kTextEncoderInferenceLatency);
    auto result = compiled_model_->Run(run.signature_index, input_buffers,
                                       output_buffers);
    // An interrupted run reports a generic error.
    if (!result.HasValue() && latch != nullptr && latch->IsCancelled()) {
      return absl::CancelledError("The text encoder run was cancelled.");
    }
    LITERT_RETURN_IF_ERROR(result);
  }

  std::vector<float> output_vector;
  {
    ScopedLatency scoped(latency_stats_, kOutputCopyLatency);
    LITERT_ASSIGN_OR_RETURN(output_vector,
                            CopyFromTensorBuffer<float>(output_buffers[0]));
    if (token_embeddings_output_index.has_value()) {
      ABSL_RETURN_IF_ERROR(AppendPerTokenOutputSlice(
          output_buffers[*token_embeddings_output_index], kTokenEmbeddingsName,
          run.padded_sequence_length, sequence_length,
          per_token_collector->token_embeddings,
          per_token_collector->token_embedding_dim));
    }
    for (const auto& [layer, output_index] : hidden_state_output_indices) {
      ABSL_RETURN_IF_ERROR(AppendPerTokenOutputSlice(
          output_buffers[output_index], LayerTapOutputName(layer),
          run.padded_sequence_length, sequence_length,
          per_token_collector->hidden_states[layer],
          per_token_collector->hidden_state_dim));
    }
  }

  AccumulateStat(latency_stats_, kEmbeddingNumTokensMetric,
                 static_cast<int64_t>(sequence_length));
  AccumulateStat(latency_stats_, kEmbeddingNumPaddedTokensMetric,
                 static_cast<int64_t>(run.padded_sequence_length));

  return output_vector;
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::RunEncoderForTokens(
    absl::Span<const int32_t> tokens,
    PerTokenOutputCollector* per_token_collector) {
  ScopedLatency scoped_total_latency(latency_stats_);
  RET_CHECK_NE(embedding_lookup_, nullptr);

  ABSL_ASSIGN_OR_RETURN(EncoderRun run, PrepareEncoderRun(tokens.size()));
  std::vector<litert::TensorBuffer>& input_buffers = *run.input_buffers;

  {
    ScopedLatency scoped(latency_stats_, kEmbedderLookupLatency);
    ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
        tokens, &input_buffers[run.embeddings_buffer_index],
        /*token_offset=*/0));
  }

  if (run.per_layer_embeddings_buffer_index.has_value() &&
      per_layer_embedding_lookup_ != nullptr) {
    ScopedLatency scoped(latency_stats_, kPerLayerEmbedderLookupLatency);
    ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
        tokens, &input_buffers[*run.per_layer_embeddings_buffer_index],
        /*token_offset=*/0));
  }

  return FinishEncoderRun(run, tokens.size(), /*latch=*/nullptr,
                          per_token_collector);
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbeddingFromInputEmbeddings(
    absl::Span<const float> input_embeddings) {
  return EncodeInputEmbeddings(input_embeddings, /*latch=*/nullptr);
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::EncodeInputEmbeddings(
    absl::Span<const float> input_embeddings, CancellationLatch* latch) {
  ScopedLatency scoped_total_latency(latency_stats_);

  if (expected_input_dimension_.empty() ||
      expected_input_dimension_.back() <= 0) {
    return absl::FailedPreconditionError(
        "The embedding dimension of the text encoder input is unknown.");
  }
  const size_t embed_dim = expected_input_dimension_.back();
  if (input_embeddings.empty() || input_embeddings.size() % embed_dim != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The size of the input embeddings (", input_embeddings.size(),
        ") must be a positive multiple of the embedding dimension (", embed_dim,
        ")."));
  }
  const size_t sequence_length = input_embeddings.size() / embed_dim;
  if (encoder_signatures_.empty()) {
    return absl::FailedPreconditionError("No encoder signatures available.");
  }
  const size_t max_seq_len = std::prev(encoder_signatures_.end())->first;
  if (sequence_length > max_seq_len) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input sequence length (", sequence_length,
        ") exceeds maximum supported signature length (", max_seq_len, ")."));
  }

  ABSL_ASSIGN_OR_RETURN(EncoderRun run, PrepareEncoderRun(sequence_length));
  if (run.per_layer_embeddings_buffer_index.has_value()) {
    return absl::UnimplementedError(
        "The text encoder requires per-layer embeddings, which are computed "
        "from token IDs. Use ComputeEmbedding instead.");
  }

  {
    ScopedLatency scoped(latency_stats_, kEmbedderLookupLatency);
    litert::TensorBuffer& embeddings_buffer =
        (*run.input_buffers)[run.embeddings_buffer_index];
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, embeddings_buffer.TensorType());
    if (tensor_type.ElementType() != litert::ElementType::Float32) {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported text encoder input element type: ",
                       static_cast<int>(tensor_type.ElementType()),
                       ". Only float32 input embeddings are supported."));
    }
    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create(
            embeddings_buffer, litert::TensorBuffer::LockMode::kWrite));
    float* embeddings_ptr = static_cast<float*>(lock_and_addr.second);
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size_bytes, embeddings_buffer.Size());
    const size_t buffer_elements = buffer_size_bytes / sizeof(float);
    RET_CHECK_LE(input_embeddings.size(), buffer_elements)
        << "The text encoder input buffer is too small.";
    absl::c_copy(input_embeddings, embeddings_ptr);
    std::fill(embeddings_ptr + input_embeddings.size(),
              embeddings_ptr + buffer_elements, 0.0f);
  }

  return FinishEncoderRun(run, sequence_length, latch);
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbeddingFromInputEmbeddings(
    absl::Span<const float> input_embeddings,
    absl::FunctionRef<bool()> should_cancel) {
  RET_CHECK_NE(compiled_model_, nullptr) << "The text encoder is not compiled.";
  CancellationLatch latch(should_cancel, {compiled_model_.get()});
  if (latch.IsCancelled()) {
    return absl::CancelledError("The text encoder run was cancelled.");
  }
  return EncodeInputEmbeddings(input_embeddings, &latch);
}

absl::StatusOr<EmbeddingOutput>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbedding(
    const ExecutorInputs& inputs, const ComputeEmbeddingOptions& options) {
  if (embedding_lookup_ == nullptr) {
    return absl::FailedPreconditionError(
        "Computing embeddings from token IDs requires the embedder model "
        "(kTfLiteEmbedder), which the model does not contain. Use "
        "ComputeEmbeddingFromInputEmbeddings instead.");
  }
  ABSL_RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  if (per_layer_embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(
        per_layer_embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }
  auto cleanup =
      absl::MakeCleanup([this]() {
        auto status = embedding_lookup_->CleanupMultiModalEmbeddings();
        if (!status.ok()) {
          ABSL_LOG(WARNING)
              << "Failed to cleanup multimodal embeddings: " << status;
        }
        if (per_layer_embedding_lookup_ != nullptr) {
          status = per_layer_embedding_lookup_->CleanupMultiModalEmbeddings();
          if (!status.ok()) {
            ABSL_LOG(WARNING)
                << "Failed to cleanup per layer multimodal embeddings: "
                << status;
          }
        }
      });

  ABSL_ASSIGN_OR_RETURN(auto text_data_ptr, inputs.GetTextDataPtr());
  RET_CHECK_NE(text_data_ptr, nullptr) << "TextData cannot be null.";

  LITERT_ASSIGN_OR_RETURN(
      auto token_ids_vec,
      CopyFromTensorBuffer<int32_t>(text_data_ptr->GetTokenIds()));

  int input_length = static_cast<int>(token_ids_vec.size());
  if (encoder_signatures_.empty()) {
    return absl::FailedPreconditionError("No encoder signatures available.");
  }

  size_t max_seq_len = std::prev(encoder_signatures_.end())->first;

  ABSL_ASSIGN_OR_RETURN(
      std::vector<int> canonical_layers,
      CanonicalizeHiddenStateLayers(options.return_hidden_state_layers));
  PerTokenOutputCollector collector{
      .collect_token_embeddings = options.return_token_embeddings,
      .hidden_state_layers = std::move(canonical_layers),
  };
  PerTokenOutputCollector* collector_ptr =
      (collector.collect_token_embeddings ||
       !collector.hidden_state_layers.empty())
          ? &collector
          : nullptr;

  std::vector<float> output_vector;
  std::optional<int> truncated_length = std::nullopt;
  int num_chunks = 1;

  if (token_ids_vec.size() > max_seq_len) {
    switch (options.input_overflow_strategy) {
      case InputOverflowStrategy::kError:
        return absl::InvalidArgumentError(
            absl::StrCat("Input sequence length (", token_ids_vec.size(),
                         ") exceeds maximum supported signature length (",
                         max_seq_len, ")."));
      case InputOverflowStrategy::kTruncate: {
        token_ids_vec.resize(max_seq_len);
        if (!options.eos_token_ids.empty() &&
            options.eos_token_ids.size() < max_seq_len) {
          std::copy(options.eos_token_ids.begin(), options.eos_token_ids.end(),
                    token_ids_vec.end() - options.eos_token_ids.size());
        }
        ABSL_ASSIGN_OR_RETURN(
            output_vector,
            RunEncoderForTokens(absl::MakeSpan(token_ids_vec), collector_ptr));
        truncated_length = static_cast<int>(max_seq_len);
        break;
      }
      case InputOverflowStrategy::kChunkAndAverage: {
        num_chunks = static_cast<int>((token_ids_vec.size() + max_seq_len - 1) /
                                      max_seq_len);
        for (int i = 0; i < num_chunks; ++i) {
          size_t start = i * max_seq_len;
          size_t end = std::min(start + max_seq_len, token_ids_vec.size());
          ABSL_ASSIGN_OR_RETURN(
              auto chunk_embedding,
              RunEncoderForTokens(
                  absl::MakeSpan(token_ids_vec).subspan(start, end - start),
                  collector_ptr));
          if (i == 0) {
            output_vector = std::move(chunk_embedding);
          } else {
            if (output_vector.size() != chunk_embedding.size()) {
              return absl::InternalError(
                  "Mismatch in embedding output dimensions between chunks.");
            }
            for (size_t j = 0; j < output_vector.size(); ++j) {
              output_vector[j] += chunk_embedding[j];
            }
          }
        }
        float scale = 1.0f / static_cast<float>(num_chunks);
        for (float& val : output_vector) {
          val *= scale;
        }
        break;
      }
    }
  } else {
    ABSL_ASSIGN_OR_RETURN(
        output_vector,
        RunEncoderForTokens(absl::MakeSpan(token_ids_vec), collector_ptr));
  }

  return EmbeddingOutput{
      .embedding = std::move(output_vector),
      .input_length = input_length,
      .truncated_length = truncated_length,
      .num_chunks = num_chunks,
      .token_embeddings = std::move(collector.token_embeddings),
      .token_embedding_dim = collector.token_embedding_dim,
      .hidden_states = std::move(collector.hidden_states),
      .hidden_state_dim = collector.hidden_state_dim,
  };
}

absl::StatusOr<std::vector<EmbeddingOutput>>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbeddingBatch(
    const std::vector<ExecutorInputs>& batch_inputs,
    const ComputeEmbeddingOptions& options) {
  std::vector<EmbeddingOutput> batch_outputs;
  batch_outputs.reserve(batch_inputs.size());
  // Batch inference isn't supported yet. Loop over the batch for now.
  for (const auto& inputs : batch_inputs) {
    ABSL_ASSIGN_OR_RETURN(auto output, ComputeEmbedding(inputs, options));
    batch_outputs.push_back(std::move(output));
  }
  return batch_outputs;
}

absl::string_view EmbeddingLiteRtCompiledModelExecutor::ExecutorBackendName()
    const {
  return backend_name_;
}

absl::StatusOr<std::vector<int>>
EmbeddingLiteRtCompiledModelExecutor::GetExpectedInputDimension() const {
  return expected_input_dimension_;
}

std::vector<TextEncoderVariant>
EmbeddingLiteRtCompiledModelExecutor::GetSupportedVariants() const {
  // The input embeddings are [batch, seq_len, embed_dim] or [seq_len,
  // embed_dim]. A dynamic (non-positive) batch dimension is run with a batch
  // size of 1.
  const int batch_size = (expected_input_dimension_.size() == 3 &&
                          expected_input_dimension_[0] > 0)
                             ? expected_input_dimension_[0]
                             : 1;
  std::vector<TextEncoderVariant> variants;
  variants.reserve(encoder_signatures_.size());
  // encoder_signatures_ is ordered by ascending sequence length.
  for (const auto& [sequence_length, signature_index] : encoder_signatures_) {
    variants.push_back(
        {.batch_size = batch_size, .sequence_length = sequence_length});
  }
  return variants;
}

absl::StatusOr<int>
EmbeddingLiteRtCompiledModelExecutor::GetEmbeddingDimension() const {
  return embedding_dimension_;
}

litert::Environment* EmbeddingLiteRtCompiledModelExecutor::GetEnvironment()
    const {
  return &env_;
}

absl::Status EmbeddingLiteRtCompiledModelExecutor::StartProfiling() {
  latency_stats_ = ExecutorStats();
  if (compiled_model_ != nullptr) {
    StartCompiledModelMetricsCollection(*compiled_model_);
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutorStats>
EmbeddingLiteRtCompiledModelExecutor::StopProfiling() {
  if (!latency_stats_.has_value()) {
    return absl::FailedPreconditionError("Profiling has not been started.");
  }
  ExecutorStats stats = *std::move(latency_stats_);
  latency_stats_ = std::nullopt;
  stats.module_name = kEmbeddingModuleName;

  if (compiled_model_ != nullptr) {
    CollectCompiledModelMetrics(*compiled_model_, stats,
                                kTextEncoderMetricsPrefix);
  }

  auto num_tokens_metric = stats.GetMetric(kEmbeddingNumTokensMetric);
  int64_t num_tokens = 0;
  if (num_tokens_metric.has_value() &&
      std::holds_alternative<int64_t>(*num_tokens_metric)) {
    num_tokens = std::get<int64_t>(*num_tokens_metric);
  }

  auto num_padded_tokens_metric =
      stats.GetMetric(kEmbeddingNumPaddedTokensMetric);
  int64_t num_padded_tokens = 0;
  if (num_padded_tokens_metric.has_value() &&
      std::holds_alternative<int64_t>(*num_padded_tokens_metric)) {
    num_padded_tokens = std::get<int64_t>(*num_padded_tokens_metric);
  }

  absl::Duration total_latency = stats.GetTotalLatency();
  if (total_latency > absl::ZeroDuration()) {
    double total_seconds = absl::ToDoubleSeconds(total_latency);
    if (total_seconds > 0.0) {
      if (num_tokens > 0) {
        stats.Accumulate(kEmbeddingE2eTokensPerSecondMetric,
                         static_cast<double>(num_tokens) / total_seconds);
      }
      if (num_padded_tokens > 0) {
        stats.Accumulate(
            kEmbeddingE2ePaddedTokensPerSecondMetric,
            static_cast<double>(num_padded_tokens) / total_seconds);
      }
    }
  }

  auto encoder_latency = stats.GetLatency(kTextEncoderInferenceLatency);
  if (encoder_latency.has_value() && *encoder_latency > absl::ZeroDuration()) {
    double encoder_seconds = absl::ToDoubleSeconds(*encoder_latency);
    if (encoder_seconds > 0.0) {
      if (num_tokens > 0) {
        stats.Accumulate(kEmbeddingEncoderStackTokensPerSecondMetric,
                         static_cast<double>(num_tokens) / encoder_seconds);
      }
      if (num_padded_tokens > 0) {
        stats.Accumulate(
            kEmbeddingEncoderStackPaddedTokensPerSecondMetric,
            static_cast<double>(num_padded_tokens) / encoder_seconds);
      }
    }
  }

  return stats;
}

EmbeddingLiteRtCompiledModelExecutor::EmbeddingLiteRtCompiledModelExecutor(
    EmbeddingExecutorSettings executor_settings, Environment& env,
    std::shared_ptr<ModelResources> resources,
    std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
    std::unique_ptr<PatchedTextEncoderModel> patched_model,
    std::unique_ptr<litert::CompiledModel> compiled_model,
    std::vector<int> expected_input_dimension, int embedding_dimension,
    std::map<int, size_t> encoder_signatures,
    absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
        input_buffers_cache,
    absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
        output_buffers_cache)
    : executor_settings_(std::move(executor_settings)),
      env_(env),
      resources_(std::move(resources)),
      embedding_lookup_(std::move(embedding_lookup)),
      per_layer_embedding_lookup_(std::move(per_layer_embedding_lookup)),
      patched_model_(std::move(patched_model)),
      compiled_model_(std::move(compiled_model)),
      expected_input_dimension_(std::move(expected_input_dimension)),
      embedding_dimension_(embedding_dimension),
      backend_name_(GetBackendString(executor_settings_.GetBackend())),
      encoder_signatures_(std::move(encoder_signatures)),
      input_buffers_cache_(std::move(input_buffers_cache)),
      output_buffers_cache_(std::move(output_buffers_cache)) {}

}  // namespace litert::lm
