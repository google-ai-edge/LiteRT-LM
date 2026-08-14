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

#include "support/tokenizer/sentencepiece_tokenizer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "sentencepiece_model.pb.h"  // from @sentencepiece
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::support {

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromFile(absl::string_view model_path) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->Load(model_path);
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromBuffer(absl::string_view model_buffer) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->LoadFromSerializedProto(model_buffer);
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromProto(
    std::unique_ptr<sentencepiece::ModelProto> model_proto) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->Load(std::move(model_proto));
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

// Encodes the given text into a TensorBuffer of token ids.
absl::StatusOr<std::vector<int>> SentencePieceTokenizer::TextToTokenIds(
    absl::string_view text) {
  std::vector<int> ids;
  auto status = processor_->Encode(text, &ids);
  if (!status.ok()) {
    return status;
  }
  return ids;
}

absl::StatusOr<int> SentencePieceTokenizer::TokenToId(absl::string_view token) {
  int id = processor_->PieceToId(token);
  if (id == processor_->unk_id()) {
    return absl::NotFoundError(absl::StrCat("Unknown token: ", token));
  }
  return id;
}

// Decodes the given TensorBuffer of token ids into a string.
absl::StatusOr<std::string> SentencePieceTokenizer::TokenIdsToText(
    absl::Span<const int> token_ids, bool skip_special_tokens) {
  if (skip_special_tokens) {
    return absl::InvalidArgumentError(
        "SentencePieceTokenizer does not support skipping special tokens. "
        "Special tokens are handled by the SentencePiece buildenormalizer.");
  }

  for (const auto& token_id : token_ids) {
    if (token_id >= vocab_size_ || token_id < 0) {
      return absl::NotFoundError(
          absl::StrCat("Token id ", token_id,
                       " is out of range. Vocab size is ", vocab_size_));
    }
  }

  // We need special handling for control tokens like BOS and EOS.
  if (token_ids.size() == 1 && processor_->IsControl(token_ids[0])) {
    return processor_->IdToPiece(token_ids[0]);
  }

  std::string text;
  auto status = processor_->Decode(token_ids, &text);
  if (!status.ok()) {
    return status;
  }
  return text;
}

std::vector<std::string> SentencePieceTokenizer::GetTokens() const {
  std::vector<std::string> tokens;
  for (const auto& piece : processor_->model_proto().pieces()) {
    tokens.push_back(piece.piece());
  }
  return tokens;
}

int SentencePieceTokenizer::GetVocabSize() const {
  return processor_->GetPieceSize();
}

}  // namespace litert::support
