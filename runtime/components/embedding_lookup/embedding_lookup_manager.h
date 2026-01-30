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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_MANAGER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_end_of_multi_modal.h"
#include "runtime/components/embedding_lookup/embedding_lookup_multi_modal.h"
#include "runtime/components/embedding_lookup/embedding_lookup_text.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

class EmbeddingLookupManager {
 public:
  // Creates an EmbeddingLookupManager.
  //
  // The end_of_multi_modal_embedding_models is a map of special tokens to the
  // corresponding embedding models. The special tokens are used to indicate
  // that the corresponding embedding model should be used.
  //
  // If fully_supports_multi_modal is true, the EmbeddingLookupManager will
  // handle multimodal tokens via the multimodal embedding lookup. Otherwise, it
  // default any multi-modal tokens to the text embedding value of entry 0.
  // If fully_supports_multi_modal is false, the
  // end_of_multi_modal_embedding_models must be empty.
  //
  // If the provide text_embedding_model has more than one signature, the
  // signature_key must be provided.
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupManager>> Create(
      const litert::Model* absl_nonnull text_embedding_model,
      absl::flat_hash_map<int, const litert::Model*>&
          end_of_multi_modal_embedding_models,
      bool fully_supports_multi_modal = true,
      std::optional<std::string> signature_key = std::nullopt);

  static absl::StatusOr<std::unique_ptr<EmbeddingLookupManager>> Create(
      const litert::Model* absl_nonnull text_embedding_model,
      bool fully_supports_multi_modal = true,
      std::optional<std::string> signature_key = std::nullopt);

  // Updates the multimodal embeddings for the given ExecutorInputs.
  // Intended to be called at the beginning of the prefill pass.
  //
  // If fully_supports_multi_modal_ is false, this function will return an error
  // if the ExecutorInputs contain any multimodal embeddings.
  absl::Status UpdateMultiModalEmbeddings(
      const ::litert::lm::ExecutorInputs& inputs);

  // Cleans up the multimodal embeddings and verifies that all the embeddings
  // have been used.
  // Intended to be called at the end of the prefill pass.
  absl::Status CleanupMultiModalEmbeddings();

  // For a given token, looks up the embedding and stores it in the output
  // vector.
  //
  // This is used for the case where the llm_litert_executor needs to look up
  // embeddings for the current step and then use the result for the next step.
  // At that point, it does not have a TfLiteTensor to store the result in.
  absl::Status LookupDecode(int token, std::vector<float>& output_vector);

  // For a given token, looks up the embedding and stores it in the
  // output tensor.
  absl::Status LookupDecode(int token, litert::TensorBuffer* output_tensor);

  // For a given token, looks up the embedding and stores it in the provided
  // vector. This function is responsible for setting the size of the vector to
  // the correct size and filling it with the embedding. Any data that was
  // previously in the vector will be overwritten.
  //
  // This is used for the case where the llm_litert_executor needs to look up
  // embeddings for the current step and then use the result for the next step.
  // At that point, it does not have a TfLiteTensor to store the result in.
  absl::Status LookupPrefill(int token, std::vector<float>& output_vector);

  // For a given list of tokens, looks up the embeddings, concatenates them and
  // returns the result through the output tensor.
  //
  // token_offset is used to indicate where to start writing to in the
  // output_tensor. This is used in cases where the output_tensor has already
  // had some embeddings written to it. If this is the first time embeddings are
  // being written to the output_tensor, token_offset should be 0.
  absl::Status LookupPrefill(absl::Span<const int> tokens,
                             litert::TensorBuffer* output_tensor,
                             size_t token_offset);

  EmbeddingLookupText* GetTextEmbeddingLookup() const {
    return text_embedding_lookup_.get();
  }

 protected:
  absl::Status Initialize(
      const litert::Model* absl_nonnull text_embedding_model,
      absl::flat_hash_map<int, const litert::Model*>&
          end_of_multi_modal_embedding_models,
      bool fully_supports_multi_modal,
      std::optional<std::string> signature_key);

  std::unique_ptr<EmbeddingLookupText> text_embedding_lookup_;
  std::vector<std::unique_ptr<EmbeddingLookupMultiModal>>
      multi_modal_embedding_lookups_;
  std::vector<std::unique_ptr<EndOfMultiModalEmbedding>>
      end_of_multi_modal_embedding_lookups_;

  // If true, the EmbeddingLookupManager will support multimodal embeddings.
  // Otherwise, it will default any multimodal tokens to the text embedding
  // value of entry 0.
  bool fully_supports_multi_modal_;

  // The default embedding vector to use when full_supports_multi_modal_=false.
  std::vector<float> default_embedding_vector_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_MANAGER_H_
