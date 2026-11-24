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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Input tensor name substrings used to identify text encoder inputs.
inline constexpr absl::string_view kPerLayerEmbeddingsName = "per_layer";
inline constexpr absl::string_view kInputMaskName = "input_mask";
inline constexpr absl::string_view kEmbeddingsName = "embeddings";

// The text encoder signatures of a model, grouped by batch size.
struct TextEncoderSignatureMaps {
  // Sequence length -> signature index, for signatures with batch size 1.
  absl::btree_map<int, size_t> single;
  // Batch size (> 1) -> sequence length -> signature index.
  absl::btree_map<int, absl::btree_map<int, size_t>> batched;
};

// Parses the text encoder signatures of `model` into batch-size-1 and batched
// (`B > 1`) maps. If `selected_signatures` is non-empty, only signatures whose
// keys appear in `selected_signatures` are included.
absl::StatusOr<TextEncoderSignatureMaps> GetTextEncoderSignatureMaps(
    const litert::Model& model,
    absl::Span<const std::string> selected_signatures = {});

// The indices of the inputs of a text encoder signature.
struct EncoderInputIndices {
  size_t embeddings = 0;
  std::optional<size_t> input_mask;
  std::optional<size_t> per_layer_embeddings;
};

// Maps text encoder signature input names to their slot indices by substring
// match (`kPerLayerEmbeddingsName` is checked before `kEmbeddingsName` so
// "per_layer_embeddings" is not misclassified as the main embeddings input).
// `embeddings` defaults to 0 if no input name matches `kEmbeddingsName`.
template <typename Names>
EncoderInputIndices GetEncoderInputIndices(const Names& input_names) {
  EncoderInputIndices indices;
  for (size_t i = 0; i < input_names.size(); ++i) {
    absl::string_view name = input_names[i];
    if (absl::StrContains(name, kPerLayerEmbeddingsName)) {
      indices.per_layer_embeddings = i;
    } else if (absl::StrContains(name, kInputMaskName)) {
      indices.input_mask = i;
    } else if (absl::StrContains(name, kEmbeddingsName)) {
      indices.embeddings = i;
    }
  }
  return indices;
}

// Returns the number of elements of `buffer`, which must hold float32 values.
// `name` names the tensor in error messages.
absl::StatusOr<size_t> GetFloat32Elements(const litert::TensorBuffer& buffer,
                                          absl::string_view name);

// Returns the length of the longest sequence that fits in a batched signature
// and in a batch size 1 signature, or 0 if there is none.
int GetMaxBatchedSequenceLength(
    const absl::btree_map<int, size_t>& single_signatures,
    const absl::btree_map<int, absl::btree_map<int, size_t>>&
        batched_signatures);

// Checks that the batched text encoder signature with the given buffers can be
// executed in batched mode: inputs are float32 with `batch_size` rows of
// `sequence_length` tokens, and the first output has `batch_size` rows of
// `single_output_elements` floats.
//
// `embedding_floats_per_token` and `per_layer_floats_per_token` are the
// number of floats per token of the embeddings and per-layer embeddings
// inputs; `per_layer_floats_per_token` is nullopt without a per-layer
// embedder.
absl::Status ValidateBatchedSignature(
    const EncoderInputIndices& inputs,
    absl::Span<const litert::TensorBuffer> input_buffers,
    absl::Span<const litert::TensorBuffer> output_buffers, int batch_size,
    int sequence_length, size_t embedding_floats_per_token,
    std::optional<size_t> per_layer_floats_per_token,
    size_t single_output_elements);

// A group of inputs that run together on one batched text encoder signature.
struct BatchedTextEncoderGroup {
  // The index of the batched signature.
  size_t signature_index = 0;
  // The sequence length of the batched signature.
  int padded_sequence_length = 0;
  // The indices of the inputs, one per row of the signature, in ascending
  // order of length.
  std::vector<size_t> inputs;

  bool operator==(const BatchedTextEncoderGroup& other) const = default;
};

// Groups inputs of similar length to run them on batched text encoder
// signatures with as little padding as possible.
//
// At each step, the shortest remaining inputs run on the batched signature
// with the smallest padded length that they fill entirely (preferring larger
// batches on ties), as long as the batched run does not process more padded
// tokens than running those inputs one at a time on the batch size 1
// signatures. Otherwise the shortest remaining input is left out and runs
// alone.
//
// Args:
//   lengths: The number of tokens of each input, or 0 for an input that must
//     not be batched. Inputs longer than the longest batched or batch size 1
//     signature are not batched either.
//   single_signatures: Sequence length -> signature index, for the signatures
//     with batch size 1.
//   batched_signatures: Batch size (> 1) -> sequence length -> signature
//     index.
//
// Returns:
//   The groups, each referring to inputs by their index in `lengths`. Inputs
//   in no group are meant to run one at a time.
std::vector<BatchedTextEncoderGroup> GroupInputsForBatchedSignatures(
    absl::Span<const size_t> lengths,
    const absl::btree_map<int, size_t>& single_signatures,
    const absl::btree_map<int, absl::btree_map<int, size_t>>&
        batched_signatures);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_
