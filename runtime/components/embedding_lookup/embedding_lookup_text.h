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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_TEXT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_TEXT_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup.h"

namespace litert::lm {

// Class used for looking up text embeddings on the CPU.
//
// Ideally text embedding lookups should be a part of the main model but there
// are cases where the embedding lookup needs to be done separately for now. For
// example, large embedding tables may use too much memory on the accelerator
// and so they need to be placed on the CPU. Currently there is no mechanism
// to tell a delegate to move embedding lookups to the CPU.
class EmbeddingLookupText : public EmbeddingLookup {
 public:
  ~EmbeddingLookupText() = default;

  // Creates a EmbeddingLookupText instance. The reference of |model| is kept
  // in the returned instance, so the caller must ensure that |model| outlives
  // the returned instance.  If the model has more than one signature, and
  // signature_key is not provided, the first signature will be used by default.
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupText>> Create(
      const litert::Model* absl_nonnull model,
      std::optional<std::string> signature_key = std::nullopt);

  // For a given token, looks up the embedding and stores it in the
  // provided vector. The caller is responsible for ensuring that the vector is
  // the correct size for the embedding.
  //
  // This is used for the case where the llm_litert_executor needs to look up
  // embeddings for the current step and then use the result for the next step.
  // At that point, it does not have a LiteRtTensor to store the result in.
  absl::Status LookupDecode(int token,
                            std::vector<float>& decode_output_vector) override;

  // For a given token, looks up the embedding and stores it in the
  // output tensor.
  absl::Status LookupDecode(int token,
                            litert::TensorBuffer* decode_output) override;

  // For a given token, looks up the embedding and stores it in the
  // provided vector. The caller is responsible for ensuring that the vector is
  // the correct size for the embedding model output.
  //
  // This is used for the case where the llm_litert_executor needs to look up
  // embeddings for the current step and then use the result for the next step.
  // At that point, it does not have a LiteRtTensor to store the result in.
  absl::Status LookupPrefill(
      int token, std::vector<float>& prefill_output_vector) override;

  // For a given list of tokens, looks up the embeddings, concatenates them and
  // returns the result through the output tensor.
  //
  // Support is only partially implemented right now. This function only
  // supports the case where the output tensor's 0th dimension is of size
  // 1, its 1st dimension is greater than or equal to tokens.size(), and its
  // subsequent dimensions match the dimensions of the embedding model output.
  // In other words, if the embedding model outputs [B=1, T=1, N, H], then the
  // output tensor must be [1, >=tokens.size(), N, H].
  //
  // bytes_offset is used to indicate what byte to start writing to in the
  // output_tensor. This is used in cases where the output_tensor has already
  // had some embeddings written to it.
  absl::Status LookupPrefill(absl::Span<const int> tokens,
                             litert::TensorBuffer* prefill_output,
                             size_t byte_offset) override;

  // Returns number of floats per token in the output tensor.
  size_t GetFloatsPerToken();

  // Returns the default embedding vector to use when a token is not found in
  // the lookup table.
  std::vector<float> GetDefaultEmbeddingVector() const {
    return default_embedding_vector_;
  }

  // Returns the signature key to use for the embedding model.
  std::optional<litert::RankedTensorType> GetOutputBufferType() const {
    return output_buffer_type_;
  }

 protected:
  EmbeddingLookupText(litert::Environment env,
                      const litert::Model* absl_nonnull model,
                      std::optional<std::string> signature_key)
      : env_(std::move(env)), model_(*model), signature_key_(signature_key) {}

  // Loads the provided model. This must be called before Lookup.
  absl::Status Initialize();

  // Internal implementation of Lookup for both the single and multiple token
  // cases.
  absl::Status LookupInternal(int token, absl::Span<uint8_t> buffer);

  // The environment for the embedding lookup.
  litert::Environment env_;
  // The model for the embedding lookup. The actual model instance is owned by
  // the model resources.
  const litert::Model& model_;
  // The compiled model for the embedding model.
  std::optional<litert::CompiledModel> compiled_model_;

  // The input buffer for the embedding model.
  std::vector<litert::TensorBuffer> input_buffers_;

  // The output buffers for the embedding model.
  std::vector<litert::TensorBuffer> output_buffers_;
  // The output buffer type for the embedding model.
  std::optional<litert::RankedTensorType> output_buffer_type_;

  // The size of the output tensor needed for a single token.
  size_t floats_per_token_output_;

  // The default embedding vector to use when a token is not found in the
  // lookup table. This is set to the value of token id 0.
  std::vector<float> default_embedding_vector_;

  // The signature key to use for the embedding model. If not provided, the
  // first signature key will be used.
  std::optional<std::string> signature_key_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_TEXT_H_
