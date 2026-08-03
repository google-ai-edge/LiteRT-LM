// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_MULTI_MODAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_MULTI_MODAL_H_

#include <sys/types.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup.h"

namespace litert::lm {

class EmbeddingLookupMultiModal : public EmbeddingLookup {
 public:
  ~EmbeddingLookupMultiModal() override = default;

  // Creates a EmbeddingLookupMultiModal instance.
  //
  // The embedding buffer will be used whenever the special token is present in
  // the input tokens to the Lookup functions. Whenever the special token is
  // present, a number of bytes will be read from the embedding
  // buffer into the Lookup function's output tensor.
  //
  // The output tensor will have the shape [B=1, T=tokens.size(), ...] so the
  // number of bytes that are read from this embedding_buffer per special token
  // is equal to the product of all the output_tensor dimensions starting with
  // the 2nd dimension (ie. if the shape of the output tensor is [1, 2, 4, 32],
  // the number of bytes read from embedding_buffer per special_token is
  // 4 * 32 = 128).
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupMultiModal>> Create(
      const ::litert::TensorBuffer* embedding_buffer, int special_token);

  // For a given token, looks up the embedding and stores it in the
  // provided vector. The caller is responsible for ensuring that the vector is
  // the correct size for the embedding.
  absl::Status LookupDecode(int token,
                            std::vector<float>& output_vector) override;

  // For a given token, looks up the embedding and stores it in the
  // output tensor.
  absl::Status LookupDecode(int token,
                            litert::TensorBuffer* output_tensor) override;

  // For a given token, looks up the embedding and stores it in the
  // provided vector. The caller is responsible for ensuring that the vector is
  // the correct size for the embedding.
  //
  // If the token is not the special token, then this function will return
  // without copying any data.
  //
  // This is used for the case where the llm_litert_executor needs to look up
  // embeddings for the current step and then use the result for the next step.
  // At that point, it does not have a TensorBuffer to store the result in.
  absl::Status LookupPrefill(int token,
                             std::vector<float>& output_vector) override;

  // For a given list of tokens, looks up the embeddings, concatenates them and
  // returns the result through the output tensor.
  //
  // This function supports the case where the output tensor's 0th dimension is
  // of size 1, its 1st dimension is greater than or equal to tokens.size(), and
  // its subsequent dimensions match the dimensions of the embedding output. In
  // other words, if the embedding output is [B=1, T=1, ...], then the output
  // tensor must be [1, >=tokens.size(), ...].
  //
  // bytes_offset is used to indicate what byte to start writing to in the
  // output_tensor. This is used in cases where the output_tensor has already
  // had some embeddings written to it.
  absl::Status LookupPrefill(absl::Span<const int> tokens,
                             litert::TensorBuffer* output_tensor,
                             size_t byte_offset) override;

  // Returns true if there are any embeddings left to be read.
  bool HasRemainingEmbeddings() const { return embedding_.size() > 0; }

 protected:
  absl::Status Initialize(const ::litert::TensorBuffer* embedding_buffer,
                          int special_token);

  absl::Span<float> embedding_;
  int special_token_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_MULTI_MODAL_H_
