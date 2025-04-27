#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOKENIZER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  // Encodes the given text into a sequence of token ids.
  virtual absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) = 0;

  // Encodes the given text into a sequence of token ids stored in a 1D
  // litert::TensorBuffer of shape [batch_size(==1), num_tokens].
  // The prepend_token_ids and postpend_token_ids are appended to the beginning
  // and end of the token ids respectively if provided.
  absl::StatusOr<TensorBuffer> TextToTensorBuffer(
      absl::string_view text, const std::vector<int>& prepend_token_ids = {},
      const std::vector<int>& postpend_token_ids = {}) {
    auto ids_or = this->TextToTokenIds(text);
    if (!ids_or.ok()) {
      return ids_or.status();
    }
    if (!prepend_token_ids.empty()) {
      ids_or.value().insert(ids_or.value().begin(), prepend_token_ids.begin(),
                            prepend_token_ids.end());
    }
    if (!postpend_token_ids.empty()) {
      ids_or.value().insert(ids_or.value().end(), postpend_token_ids.begin(),
                            postpend_token_ids.end());
    }
    LITERT_ASSIGN_OR_RETURN(auto tensor,
                            CopyToTensorBuffer<int>(
                                absl::MakeConstSpan(ids_or.value()),
                                {1, static_cast<int>(ids_or.value().size())}));
    return tensor;
  }

  // Decodes the given sequence of token ids into a string.
  virtual absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) = 0;

  // Decodes the given sequence of token ids into a string. The input is a 2D
  // litert::TensorBuffer shape [batch_size, decode_steps]. The output is a
  // vector of strings, each of which is a decoded string of the corresponding
  // batch.
  absl::StatusOr<std::vector<std::string>> TensorBufferToText(
      const TensorBuffer& token_ids_tensor) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, token_ids_tensor.TensorType());
    auto dims = tensor_type.Layout().Dimensions();
    const int batch_size = dims[0];
    if (dims.size() != 2) {
      return absl::InvalidArgumentError(
          "The input tensor must have 2 dimensions.");
    }
    auto token_ids_or =
        CopyFromTensorBuffer2D<int>(token_ids_tensor);
    std::vector<std::string> decoded_strings(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      const std::vector<int>& token_ids = (*token_ids_or)[i];
      auto text_or = this->TokenIdsToText(token_ids);
      if (!text_or.ok()) {
        return text_or.status();
      }
      decoded_strings[i] = text_or.value();
    }
    return decoded_strings;
  }
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOKENIZER_H_
