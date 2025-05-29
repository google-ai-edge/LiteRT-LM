#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_H_

#include <sys/types.h>

#include <cstddef>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Virtual class for looking up embeddings.
//
// This can be subclassed for different embedding lookup implementations. Some
// implementations may require looking up from a .tflite model, while others
// may have already been pre-computed and can be looked up from a buffer.
class EmbeddingLookup {
 public:
  virtual ~EmbeddingLookup() = default;

  // For a given token, looks up the embedding and stores it in the
  // provided vector. The caller is responsible for ensuring that the vector is
  // the correct size for the embedding.
  //
  // This is used for the case where the llm_litert_executor needs to look up
  // embeddings for the current step and then use the result for the next step.
  // At that point, it does not have a TfLiteTensor to store the result in.
  virtual absl::Status LookupDecode(int token,
                                    std::vector<float>& output_vector) = 0;

  // For a given token, looks up the embedding and stores it in the
  // output tensor.
  virtual absl::Status LookupDecode(int token,
                                    litert::TensorBuffer* output_tensor) = 0;

  // For a given token, looks up the embedding and stores it in the
  // provided vector. The caller is responsible for ensuring that the vector is
  // the correct size for the embedding.
  //
  // This is used for the case where the llm_litert_executor needs to look up
  // embeddings for the current step and then use the result for the next step.
  // At that point, it does not have a TfLiteTensor to store the result in.
  virtual absl::Status LookupPrefill(int token,
                                     std::vector<float>& output_vector) = 0;

  // For a given list of tokens, looks up the embeddings, concatenates them and
  // returns the result through the output tensor.
  //
  // bytes_offset is used to indicate what byte to start writing to in the
  // output_tensor. This is used in cases where the output_tensor has already
  // had some embeddings written to it.
  virtual absl::Status LookupPrefill(absl::Span<const int> tokens,
                                     litert::TensorBuffer* output_tensor,
                                     size_t byte_offset) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_H_
