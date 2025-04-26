#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SENTENCEPIECE_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SENTENCEPIECE_TOKENIZER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::lm {

// A Tokenizer implementation using SentencePiece.
class SentencePieceTokenizer : public Tokenizer {
 public:
  // Creates a SentencePieceTokenizer from the given model path.
  // Note that the model path can only be a local file path but not a CNS path.
  static absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>> CreateFromFile(
      absl::string_view model_path);

  // Creates a SentencePieceTokenizer from a pre-loaded model buffer.
  static absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
  CreateFromBuffer(absl::string_view model_buffer);

  // Encodes the given text into a sequence of token ids.
  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override;

  // Decodes the given sequence of token ids into a string.
  absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) override;

 private:
  // Constructor.
  explicit SentencePieceTokenizer(
      std::unique_ptr<sentencepiece::SentencePieceProcessor> processor)
      : processor_(std::move(processor)) {};

  // SentencePiece processor.
  std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SENTENCEPIECE_TOKENIZER_H_
