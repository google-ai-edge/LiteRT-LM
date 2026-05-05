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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_END_OF_MULTI_MODAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_END_OF_MULTI_MODAL_H_

#include <sys/types.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup.h"

namespace litert::lm {

// This class is used to look up the end of multi-modal embedding, such as the
// end of audio or end of image embeddings.
class EndOfMultiModalEmbedding : public EmbeddingLookup {
 public:
  ~EndOfMultiModalEmbedding() override = default;

  // Creates a EndOfMultiModalEmbedding instance.
  // Special token is the token that indicates when to insert the end of
  // multi-modal embedding. If the special token is not found in the tokens,
  // the end of multi-modal embedding will not be inserted.
  static absl::StatusOr<std::unique_ptr<EndOfMultiModalEmbedding>> Create(
      litert::Environment& env, const litert::Model* absl_nonnull model,
      int special_token);

  // Multimodal embeddings are not supported during decode.
  absl::Status LookupDecode(int token,
                            std::vector<float>& output_vector) override {
    return absl::UnimplementedError(
        "LookupDecode is not implemented for EndOfMultiModalEmbedding.");
  }

  // Multimodal embeddings are not supported during decode.
  absl::Status LookupDecode(int token,
                            litert::TensorBuffer* output_tensor) override {
    return absl::UnimplementedError(
        "LookupDecode is not implemented for EndOfMultiModalEmbedding.");
  }

  // If the token is the special token, looks up the end of multimodal
  // embedding and stores it in the provided vector.
  absl::Status LookupPrefill(int token,
                             std::vector<float>& output_vector) override;

  // If any of the tokens are the special token, looks up the end of multimodal
  // embedding and stores it in the provided output tensor in the correct place.
  absl::Status LookupPrefill(absl::Span<const int> tokens,
                             litert::TensorBuffer* prefill_output,
                             size_t byte_offset) override;

  litert::Expected<bool> IsFullyAccelerated() override {
    return is_fully_accelerated_;
  }

 protected:
  EndOfMultiModalEmbedding(litert::Environment& env, int special_token)
      : env_(env), special_token_(special_token) {}

  // Loads the provided model. This must be called before Lookup functions.
  absl::Status Initialize(const litert::Model& model);

  // The environment for the embedding lookup.
  litert::Environment& env_;

  // The layout of the output tensor from the embedding model.
  litert::Layout output_buffer_layout_;

  // The special token that indicates when to insert the end of multi-modal
  // embedding.
  int special_token_;

  // Contains the end of multi-modal embedding that was looked up from the
  // model.
  std::vector<float> end_of_multi_modal_embedding_;

  bool is_fully_accelerated_ = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_END_OF_MULTI_MODAL_H_
