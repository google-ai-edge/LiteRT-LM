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

#include "support/tokenizer/huggingface_tokenizer.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizers.h"
#include "support/util/memory_mapped_file.h"
#include "support/util/status_macros.h"  // NOLINT

namespace litert::support {

HuggingFaceTokenizer::HuggingFaceTokenizer(
    absl_nonnull std::unique_ptr<::tokenizers::Tokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)) {}

HuggingFaceTokenizer::~HuggingFaceTokenizer() = default;

absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>>
HuggingFaceTokenizer::CreateFromFile(absl::string_view json_path) {
  ASSIGN_OR_RETURN(auto memory_mapped_file,  // NOLINT
                   MemoryMappedFile::Create(json_path));
  std::string json_data(memory_mapped_file->length(), '\0');
  memcpy(json_data.data(), memory_mapped_file->data(),
         memory_mapped_file->length());
  return CreateFromJson(json_data);
}

absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>>
HuggingFaceTokenizer::CreateFromJson(const std::string& json) {
  ASSIGN_OR_RETURN(::tokenizers::Tokenizer tokenizer,
                   ::tokenizers::Tokenizer::FromStr(json));
  return absl::WrapUnique(new HuggingFaceTokenizer(
      std::make_unique<::tokenizers::Tokenizer>(std::move(tokenizer))));
}

// Encodes the given text into a TensorBuffer of token ids.
absl::StatusOr<std::vector<int>> HuggingFaceTokenizer::TextToTokenIds(
    absl::string_view text) {
  ASSIGN_OR_RETURN(::tokenizers::Encoding encoding,
                   tokenizer_->Encode(text, /*add_special_tokens=*/false));
  const std::vector<uint32_t>& ids = encoding.GetIds();
  return std::vector<int>(ids.begin(), ids.end());
}

absl::StatusOr<int> HuggingFaceTokenizer::TokenToId(absl::string_view token) {
  std::optional<uint32_t> id = tokenizer_->TokenToId(token);
  if (!id.has_value()) {
    return absl::NotFoundError(absl::StrCat("Unknown token: ", token));
  }
  return static_cast<int>(*id);
}

// Decodes the given TensorBuffer of token ids into a vector of strings.
absl::StatusOr<std::string> HuggingFaceTokenizer::TokenIdsToText(
    absl::Span<const int> token_ids, bool skip_special_tokens) {
  std::vector<uint32_t> ids(token_ids.begin(), token_ids.end());
  return tokenizer_->Decode(ids, skip_special_tokens);
}

std::vector<std::string> HuggingFaceTokenizer::GetTokens() const {
  std::vector<std::string> tokens;
  int vocab_size = GetVocabSize();
  tokens.reserve(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    tokens.push_back(
        tokenizer_->IdToToken(static_cast<uint32_t>(i)).value_or(""));
  }
  return tokens;
}

int HuggingFaceTokenizer::GetVocabSize() const {
  return static_cast<int>(tokenizer_->GetVocabSize(/*with_added_tokens=*/true));
}

}  // namespace litert::support
