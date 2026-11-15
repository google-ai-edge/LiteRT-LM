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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/util/convert_tensor_buffer.h"

namespace litert::support {

typedef std::vector<int> TokenIds;

// Enum representing the type of tokenizer.
enum class TokenizerType {
  kUnspecified,
  kSentencePiece,
  kHuggingFace,
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  // Returns the type of the tokenizer.
  virtual TokenizerType GetTokenizerType() const = 0;

  // Encodes the given input text to token ids. Includes tokenizer pre/post
  // processing.
  virtual absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) = 0;

  // Converts a token string to its token id. This is a raw token look up,
  // without any tokenizer pre/post processing. The implementation is expected
  // to return absl::NotFoundError if the token is not found.
  virtual absl::StatusOr<int> TokenToId(absl::string_view token) = 0;

  // Helper function to convert a vector of token ids into a 1D
  // litert::TensorBuffer of shape [batch_size(==1), num_tokens].
  static absl::StatusOr<TensorBuffer> TokenIdsToTensorBuffer(
      absl::Span<const int> token_ids) {
    LITERT_ASSIGN_OR_RETURN(
        auto tensor, CopyToTensorBuffer<int>(
                         token_ids, {1, static_cast<int>(token_ids.size())}));
    return tensor;
  }

  // Decodes the given sequence of token ids into a string.
  // Returns absl::DataLossError if any of the tokens are part of an incomplete
  // BPE sequence.
  // If skip_special_tokens is true, special tokens (BOS, EOS, PAD, UNK, etc.)
  // will be skipped in the decoding process.
  virtual absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool skip_special_tokens) = 0;

  // Decodes the given sequence of token ids into a string including special
  // tokens.
  absl::StatusOr<std::string> TokenIdsToText(absl::Span<const int> token_ids) {
    return TokenIdsToText(token_ids, /*skip_special_tokens=*/false);
  }

  // Returns the list of tokens in the tokenizer.
  virtual std::vector<std::string> GetTokens() const = 0;

  // Returns the size of the vocabulary.
  virtual int GetVocabSize() const = 0;

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
      absl::Span<const TokenIds> previous_token_ids,
      absl::Span<const TokenIds> next_token_ids) {
    std::vector<TokenIds> merged_token_ids(next_token_ids.size());
    if (previous_token_ids.size() != next_token_ids.size()) {
      return absl::InvalidArgumentError(
          "The previous and next token ids must have the same size.");
    }
    for (int i = 0; i < previous_token_ids.size(); ++i) {
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
  // corresponding batch or absl::DataLossError if an incomplete BPE sequence.
  absl::StatusOr<std::vector<absl::StatusOr<std::string>>> TokenIdsToTexts(
      int batch_size, absl::Span<const TokenIds> token_ids) {
    if (token_ids.size() != batch_size) {
      return absl::InvalidArgumentError(
          "The token ID vector must have the same number of rows as the batch "
          "size.");
    }
    std::vector<absl::StatusOr<std::string>> decoded_strings(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      decoded_strings[i] = this->TokenIdsToText(token_ids[i]);
    }
    return decoded_strings;
  }
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZER_H_
