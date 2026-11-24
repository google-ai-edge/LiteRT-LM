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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/btree_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {
namespace {

constexpr absl::string_view kEncoderSignatureRunner = "encoder";

}  // namespace

absl::StatusOr<TextEncoderSignatureMaps> GetTextEncoderSignatureMaps(
    const litert::Model& model,
    absl::Span<const std::string> selected_signatures) {
  TextEncoderSignatureMaps maps;

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
      if (signature.OutputNames().empty()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Text encoder signature has no output tensors: ", key));
      }

      // Find the embeddings input tensor type to get sequence length.
      std::optional<litert::RankedTensorType> embeddings_tensor_type;
      for (absl::string_view name : signature.InputNames()) {
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

      int batch_size = 1;
      int sequence_length = 0;
      if (embeddings_tensor_type->Layout().Rank() == 3) {
        // [batch_size, seq_len, embed_dim]
        batch_size = embeddings_tensor_type->Layout().Dimensions()[0];
        sequence_length = embeddings_tensor_type->Layout().Dimensions()[1];
      } else if (embeddings_tensor_type->Layout().Rank() == 2) {
        // [seq_len, embed_dim]
        sequence_length = embeddings_tensor_type->Layout().Dimensions()[0];
      } else {
        return absl::FailedPreconditionError(
            absl::StrCat("Embedding input tensor has unsupported dimension: ",
                         embeddings_tensor_type->Layout().Rank()));
      }

      if (sequence_length <= 0) {
        // A signature with a dynamic sequence length cannot be selected by
        // input length; skip it rather than failing for the whole model.
        ABSL_LOG(WARNING) << "Skipping text encoder signature " << key
                          << " with an invalid sequence length: "
                          << sequence_length;
        continue;
      }
      if (batch_size <= 0) {
        // A dynamic (non-positive) batch dimension is run with a batch size
        // of 1.
        batch_size = 1;
      }
      if (batch_size == 1) {
        maps.single[sequence_length] = i;
      } else {
        maps.batched[batch_size][sequence_length] = i;
      }
    }
  }

  if (maps.single.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("No signature with batch size 1 found with prefix ",
                     kEncoderSignatureRunner));
  }

  return maps;
}

absl::StatusOr<size_t> GetFloat32Elements(const litert::TensorBuffer& buffer,
                                          absl::string_view name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, buffer.TensorType());
  if (tensor_type.ElementType() != litert::ElementType::Float32) {
    return absl::UnimplementedError(absl::StrCat(
        "Unsupported text encoder ", name,
        " element type: ", static_cast<int>(tensor_type.ElementType()),
        ". Only float32 is supported."));
  }
  LITERT_ASSIGN_OR_RETURN(size_t packed_size_bytes, buffer.PackedSize());
  return packed_size_bytes / sizeof(float);
}

int GetMaxBatchedSequenceLength(
    const absl::btree_map<int, size_t>& single_signatures,
    const absl::btree_map<int, absl::btree_map<int, size_t>>&
        batched_signatures) {
  if (single_signatures.empty()) return 0;
  const int max_single_seq_len = std::prev(single_signatures.end())->first;
  if (max_single_seq_len <= 0) return 0;
  int max_batched_seq_len = 0;
  for (const auto& [batch_size, signatures] : batched_signatures) {
    if (batch_size <= 1 || signatures.empty()) continue;
    max_batched_seq_len =
        std::max(max_batched_seq_len, std::prev(signatures.end())->first);
  }
  // Longer inputs would need the input overflow handling of the
  // single-sequence path.
  return std::min(max_batched_seq_len, max_single_seq_len);
}

absl::Status ValidateBatchedSignature(
    const EncoderInputIndices& inputs,
    absl::Span<const litert::TensorBuffer> input_buffers,
    absl::Span<const litert::TensorBuffer> output_buffers, int batch_size,
    int sequence_length, size_t embedding_floats_per_token,
    std::optional<size_t> per_layer_floats_per_token,
    size_t single_output_elements) {
  if (batch_size <= 1 || sequence_length <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Batched text encoder signature must have batch_size > 1 and "
        "sequence_length > 0, got ",
        batch_size, " x ", sequence_length));
  }
  // Checks that the input `index` is float32 and of shape
  // `[batch_size, sequence_length, ...]` with `floats_per_token` floats per
  // token.
  auto check_input = [&](size_t index, absl::string_view name,
                         size_t floats_per_token) -> absl::Status {
    if (index >= input_buffers.size()) {
      return absl::FailedPreconditionError(
          absl::StrCat("The ", name, " input index ", index,
                       " is out of range for a signature with ",
                       input_buffers.size(), " inputs."));
    }
    const litert::TensorBuffer& buffer = input_buffers[index];
    ABSL_ASSIGN_OR_RETURN(size_t elements, GetFloat32Elements(buffer, name));
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, buffer.TensorType());
    const auto& dims = tensor_type.Layout().Dimensions();
    if (dims.size() < 2 || dims[0] != batch_size ||
        dims[1] != sequence_length ||
        elements != static_cast<size_t>(batch_size) * sequence_length *
                        floats_per_token) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unexpected ", name, " input shape: expected ", batch_size, " x ",
          sequence_length, " x ", floats_per_token, " floats, got ", elements,
          " floats of rank ", dims.size()));
    }
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(
      check_input(inputs.embeddings, "embeddings", embedding_floats_per_token));
  if (inputs.per_layer_embeddings.has_value()) {
    if (!per_layer_floats_per_token.has_value()) {
      return absl::FailedPreconditionError(
          "The signature requires per-layer embeddings, but the model has no "
          "per-layer embedder.");
    }
    ABSL_RETURN_IF_ERROR(check_input(*inputs.per_layer_embeddings,
                                     "per-layer embeddings",
                                     *per_layer_floats_per_token));
  }
  if (inputs.input_mask.has_value()) {
    // `[B, T]` or `[B, T, 1]`.
    ABSL_RETURN_IF_ERROR(
        check_input(*inputs.input_mask, "input mask", /*floats_per_token=*/1));
    LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                            input_buffers[*inputs.input_mask].TensorType());
    const size_t mask_rank = tensor_type.Layout().Rank();
    if (mask_rank != 2 && mask_rank != 3) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unexpected input mask rank: ", mask_rank));
    }
  }

  if (output_buffers.empty()) {
    return absl::InvalidArgumentError("The signature has no output.");
  }
  ABSL_ASSIGN_OR_RETURN(size_t output_elements,
                        GetFloat32Elements(output_buffers[0], "output"));
  if (output_elements != batch_size * single_output_elements) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected output size: expected ", batch_size, " x ",
                     single_output_elements,
                     " floats (the output of the batch size 1 "
                     "signatures), got ",
                     output_elements));
  }
  return absl::OkStatus();
}

std::vector<BatchedTextEncoderGroup> GroupInputsForBatchedSignatures(
    absl::Span<const size_t> lengths,
    const absl::btree_map<int, size_t>& single_signatures,
    const absl::btree_map<int, absl::btree_map<int, size_t>>&
        batched_signatures) {
  const size_t max_batched_seq_len = static_cast<size_t>(
      GetMaxBatchedSequenceLength(single_signatures, batched_signatures));
  std::vector<size_t> batchable;
  for (size_t i = 0; i < lengths.size(); ++i) {
    if (lengths[i] > 0 && lengths[i] <= max_batched_seq_len) {
      batchable.push_back(i);
    }
  }
  absl::c_stable_sort(
      batchable, [&](size_t a, size_t b) { return lengths[a] < lengths[b]; });

  // The padded length of a sequence run alone. Batchable lengths always fit
  // in a batch size 1 signature.
  auto single_padded_length = [&](size_t length) -> size_t {
    return single_signatures.lower_bound(static_cast<int>(length))->first;
  };

  std::vector<BatchedTextEncoderGroup> groups;
  size_t pos = 0;
  while (pos < batchable.size()) {
    const size_t remaining = batchable.size() - pos;
    const absl::btree_map<int, size_t>::value_type* best = nullptr;
    size_t best_batch_size = 0;
    for (const auto& [batch_size, signatures] : batched_signatures) {
      if (batch_size <= 1) continue;
      if (static_cast<size_t>(batch_size) > remaining) break;
      // Inputs are sorted by length, so the last one is the longest.
      const size_t max_len = lengths[batchable[pos + batch_size - 1]];
      auto sig_it = signatures.lower_bound(static_cast<int>(max_len));
      if (sig_it == signatures.end()) continue;
      size_t single_padded_tokens = 0;
      for (size_t j = 0; j < static_cast<size_t>(batch_size); ++j) {
        single_padded_tokens +=
            single_padded_length(lengths[batchable[pos + j]]);
      }
      if (static_cast<size_t>(batch_size) * sig_it->first >
          single_padded_tokens) {
        continue;
      }
      if (best == nullptr || sig_it->first <= best->first) {
        best = &*sig_it;
        best_batch_size = batch_size;
      }
    }
    if (best == nullptr) {
      // The shortest remaining input cannot start a batch; it runs alone.
      ++pos;
      continue;
    }
    BatchedTextEncoderGroup& group = groups.emplace_back();
    group.signature_index = best->second;
    group.padded_sequence_length = best->first;
    group.inputs.assign(batchable.begin() + pos,
                        batchable.begin() + pos + best_batch_size);
    pos += best_batch_size;
  }
  return groups;
}

}  // namespace litert::lm
