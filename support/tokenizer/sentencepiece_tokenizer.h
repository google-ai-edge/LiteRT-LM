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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_SENTENCEPIECE_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_SENTENCEPIECE_TOKENIZER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizer.h"
#include "sentencepiece_model.pb.h"  // from @sentencepiece
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::support {

// A Tokenizer implementation using SentencePiece.
class SentencePieceTokenizer : public Tokenizer {
 public:
  // Creates a SentencePieceTokenizer from the given model path.
  // Note that the model path can only be a local file path but not a CNS path.
  static absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>> CreateFromFile(
      absl::string_view model_path);

  // Creates a SentencePieceTokenizer from a preloaded model buffer.
  static absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
  CreateFromBuffer(absl::string_view model_buffer);

  // Creates a SentencePieceTokenizer from a model proto.
  static absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
  CreateFromProto(std::unique_ptr<sentencepiece::ModelProto> model_proto);

  TokenizerType GetTokenizerType() const override {
    return TokenizerType::kSentencePiece;
  }

  // Encodes the given text into a sequence of token ids.
  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override;

  // Converts a token string to its token id. Uses SentencePiece's
  // PieceToId method.
  absl::StatusOr<int> TokenToId(absl::string_view token) override;

  // Decodes the given sequence of token ids into a string.
  using Tokenizer::TokenIdsToText;  // For skip_special_tokens=false case.
  absl::StatusOr<std::string> TokenIdsToText(absl::Span<const int> token_ids,
                                             bool skip_special_tokens) override;

  // Returns the tokens in the SentencePiece model.
  std::vector<std::string> GetTokens() const override;

  // Returns the size of the vocabulary.
  int GetVocabSize() const override;

  const sentencepiece::SentencePieceProcessor& GetProcessor() const {
    return *processor_;
  }

 private:
  // Constructor.
  explicit SentencePieceTokenizer(
      std::unique_ptr<sentencepiece::SentencePieceProcessor> processor)
      : processor_(std::move(processor)),
        vocab_size_(processor_->GetPieceSize()) {};

  // SentencePiece processor.
  std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;

  // The size of the vocabulary. Used to avoid decoding the invalid IDs that are
  // out of the range of the vocabulary.
  int vocab_size_;
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_SENTENCEPIECE_TOKENIZER_H_
