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

typedef std::vector<int> TokenIds;

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  // Encodes the given text into a sequence of token ids.
  virtual absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) = 0;

  // Returns BOS id.
  virtual absl::StatusOr<int> BosId() const {
    return absl::UnimplementedError("BosId is not implemented.");
  };

  // Returns EOS id.
  virtual absl::StatusOr<int> EosId() const {
    return absl::UnimplementedError("EosId is not implemented.");
  };

  // Helper function to convert a vector of token ids into a 1D
  // litert::TensorBuffer of shape [batch_size(==1), num_tokens].
  absl::StatusOr<TensorBuffer> TokenIdsToTensorBuffer(
      const TokenIds& token_ids) {
    LITERT_ASSIGN_OR_RETURN(auto tensor,
                            CopyToTensorBuffer<int>(
                                absl::MakeConstSpan(token_ids),
                                {1, static_cast<int>(token_ids.size())}));
    return tensor;
  }

  // Decodes the given sequence of token ids into a string.
  // Returns absl::DataLossError if any of the tokens are part of an incomplete
  // BPE sequence.
  virtual absl::StatusOr<std::string> TokenIdsToText(
      const TokenIds& token_ids) = 0;

  // Converts a tensor buffer of token ids into a vector of token ids. The input
  // is a 2D litert::TensorBuffer shape [batch_size, decode_steps].
  static absl::StatusOr<std::vector<TokenIds>> TensorBufferToTokenIds(
      const TensorBuffer& token_ids_tensor) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, token_ids_tensor.TensorType());
    auto dims = tensor_type.Layout().Dimensions();
    if (dims.size() != 2) {
      return absl::InvalidArgumentError(
          "The input tensor must have 2 dimensions.");
    }
    auto token_ids = CopyFromTensorBuffer2D<int>(token_ids_tensor);
    return token_ids.Value();
  }

  // Merges the previous and next token ids, by appending each next token
  // id to the corresponding previous token id row by row.
  static absl::StatusOr<std::vector<TokenIds>> MergeTokenIds(
      const std::vector<TokenIds>& previous_token_ids,
      const std::vector<TokenIds>& next_token_ids) {
    std::vector<TokenIds> merged_token_ids(next_token_ids.size());
    if (previous_token_ids.size() != next_token_ids.size()) {
      return absl::InvalidArgumentError(
          "The previous and next token ids must have the same size.");
    }
    for (uint8_t i = 0; i < previous_token_ids.size(); ++i) {
      merged_token_ids[i] = previous_token_ids[i];
      merged_token_ids[i].insert(merged_token_ids[i].end(),
                                 next_token_ids[i].begin(),
                                 next_token_ids[i].end());
    }
    return merged_token_ids;
  }

  // Decodes the given sequence of token ids into a string. The input is a 2D
  // vector of token ids, each of which is a sequence of token ids. The output
  // Tokenizer is a vector of strings, each of which is a decoded string of the
  // corresponding batch.
  // Returns absl::DataLossError if any of the tokens are part of an incomplete
  // BPE sequence.
  absl::StatusOr<std::vector<std::string>> TokenIdsToTexts(
      int batch_size, const std::vector<TokenIds>& token_ids) {
    std::vector<std::string> decoded_strings(batch_size);
    if (token_ids.size() != batch_size) {
      return absl::InvalidArgumentError(
          "The token ID vector must have the same number of rows as the batch "
          "size.");
    }
    for (int i = 0; i < batch_size; ++i) {
      auto text = this->TokenIdsToText(token_ids[i]);
      if (!text.ok()) {
        return text.status();
      }
      decoded_strings[i] = text.value();
    }
    return decoded_strings;
  }

  template <typename T>
  static bool IsIncompleteBpeSequence(const absl::StatusOr<T>& result) {
    return result.status().code() == absl::StatusCode::kDataLoss;
  }
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOKENIZER_H_
