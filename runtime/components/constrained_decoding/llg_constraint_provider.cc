// Copyright 2026 The ODML Authors.
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

#include "runtime/components/constrained_decoding/llg_constraint_provider.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/llg_constraint.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "llguidance.h"

namespace litert::lm {
namespace {

// SentencePiece stores a space as U+2581 ("▁").
constexpr absl::string_view kSentencePieceSpace = "\xe2\x96\x81";

// Returns the bytes that the vocabulary entry `token` decodes to. LLGuidance
// matches grammars against these bytes, and they must also agree with the
// tokens that `tokenize_fn` returns. SentencePiece-style entries differ from
// their bytes in two ways, which this undoes like LLGuidance's own
// tokenizer.json loader does:
//   - A byte-fallback piece "<0xXX>" is the single byte 0xXX.
//   - "▁" is a space.
// Not for special tokens, which keep their name.
std::string TokenToBytes(absl::string_view token) {
  uint32_t byte = 0;
  if (token.size() == 6 && absl::StartsWith(token, "<0x") && token[5] == '>' &&
      absl::ascii_isxdigit(token[3]) && absl::ascii_isxdigit(token[4]) &&
      absl::SimpleHexAtoi(token.substr(3, 2), &byte)) {
    return std::string(1, static_cast<char>(byte));
  }
  return absl::StrReplaceAll(token, {{kSentencePieceSpace, " "}});
}

// Returns whether the tokens `tokenizer` returns for a text decode to exactly
// that text, which LLGuidance assumes when it forces tokens. Some SentencePiece
// models break this: add_dummy_prefix adds a leading "▁", and
// remove_extra_whitespaces or NFKC normalization change the text. Probes the
// tokenizer can't encode are skipped. `token_bytes` are the tokens' bytes as
// LLGuidance stores them (special tokens start with 0xFF).
// TODO: Remove once LLGuidance checks tokenize_fn results itself.
bool TokenizesVerbatim(Tokenizer& tokenizer,
                       absl::Span<const std::string> token_bytes) {
  for (absl::string_view probe : {"a", "1", "x y", "a  b", " a", "\n"}) {
    absl::StatusOr<std::vector<int>> ids = tokenizer.TextToTokenIds(probe);
    if (!ids.ok() || ids->empty()) {
      continue;
    }
    std::string decoded;
    for (int id : *ids) {
      if (id < 0 || id >= token_bytes.size()) {
        return false;
      }
      absl::StrAppend(&decoded, token_bytes[id]);
    }
    if (decoded != probe) {
      return false;
    }
  }
  return true;
}

::LlgConstraint* CreateLlgConstraint(LlgConstraintInit* llg_constraint_init,
                                     absl::string_view constraint_string,
                                     LlgConstraintType constraint_type) {
  switch (constraint_type) {
    case LlgConstraintType::kRegex:
      return llg_new_constraint_regex(llg_constraint_init,
                                      constraint_string.data());
    case LlgConstraintType::kJsonSchema:
      return llg_new_constraint_json(llg_constraint_init,
                                     constraint_string.data());
    case LlgConstraintType::kLark:
      return llg_new_constraint_lark(llg_constraint_init,
                                     constraint_string.data());
    case LlgConstraintType::kLlGuidanceInternal:
      return llg_new_constraint(llg_constraint_init, constraint_string.data());
  }
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<ConstraintProvider>>
LlgConstraintProvider::Create(const Tokenizer& tokenizer,
                              LlGuidanceConfig llg_config) {
  if (llg_config.eos_ids.empty()) {
    return absl::InvalidArgumentError(
        "LlGuidanceConfig::eos_ids must not be empty.");
  }

  std::vector<std::string> tokens = tokenizer.GetTokens();

  absl::flat_hash_set<absl::string_view> special_tokens_set(
      llg_config.special_tokens.begin(), llg_config.special_tokens.end());
  std::vector<bool> is_special(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    is_special[i] = special_tokens_set.contains(tokens[i]);
  }
  // EOS tokens are special too, so that the constraint can't match them as
  // text. Otherwise text could contain an EOS token, which ends generation
  // before the constraint allows the output to end.
  for (uint32_t eos_id : llg_config.eos_ids) {
    if (eos_id >= tokens.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("EOS token ID ", eos_id, " is out of range for a ",
                       tokens.size(), "-token vocabulary."));
    }
    is_special[eos_id] = true;
  }

  // `tokens` becomes the bytes LLGuidance stores for each token.
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (is_special[i]) {
      // Special tokens need to be prefixed with 0xFF.
      // https://github.com/guidance-ai/llguidance/blob/main/docs/special_tokens.md
      tokens[i].insert(0, 1, '\xff');
    } else {
      tokens[i] = TokenToBytes(tokens[i]);
    }
  }

  std::vector<uint32_t> token_lens;
  std::vector<uint8_t> token_bytes;
  size_t total_size = 0;

  token_lens.reserve(tokens.size());
  for (const std::string& token : tokens) {
    token_lens.push_back(token.size());
    total_size += token.size();
  }

  token_bytes.reserve(total_size);
  for (const std::string& token : tokens) {
    token_bytes.insert(token_bytes.end(), token.begin(), token.end());
  }

  auto tokenize_fn = [](const void* user_data, const uint8_t* bytes,
                        size_t bytes_len, uint32_t* output_tokens,
                        size_t output_tokens_len) -> size_t {
    absl::string_view text(reinterpret_cast<const char*>(bytes), bytes_len);

    // The tokenizer is passed as `user_data` to tokenize_fn. It needs to be
    // cast back into a Tokenizer*.
    Tokenizer* tokenizer =
        static_cast<Tokenizer*>(const_cast<void*>(user_data));
    auto token_ids = tokenizer->TextToTokenIds(text);
    if (!token_ids.ok()) {
      return 0;
    }
    if (output_tokens_len > 0) {
      memcpy(output_tokens, token_ids->data(),
             std::min(output_tokens_len, token_ids->size()) * sizeof(uint32_t));
    }
    return token_ids->size();
  };

  // llguidance takes the first EOS token separately from the others.
  const uint32_t* extra_eos_ids =
      llg_config.eos_ids.size() > 1 ? llg_config.eos_ids.data() + 1 : nullptr;
  LlgTokenizerInitV2 tok_init = {
      .struct_size = sizeof(LlgTokenizerInitV2),
      .vocab_size = static_cast<uint32_t>(tokens.size()),
      .tok_eos = llg_config.eos_ids[0],
      .token_lens = token_lens.data(),
      .token_bytes = token_bytes.data(),
      // Tokenizer::TextToTokenIds() expects UTF-8 (SentencePiece replaces
      // invalid bytes with U+FFFD), so LLGuidance tokenizes invalid UTF-8
      // itself, using the byte tokens.
      .tokenize_assumes_string = true,
      .tokenize_fn = tokenize_fn,
      .tokenize_user_data = &tokenizer,
      .tok_eos_extra = extra_eos_ids,
      .tok_eos_extra_count =
          static_cast<uint32_t>(llg_config.eos_ids.size() - 1),
  };
  // Without a tokenize_fn, LLGuidance doesn't force tokens. The masks still
  // allow exactly the tokens that match the constraint.
  if (!TokenizesVerbatim(const_cast<Tokenizer&>(tokenizer), tokens)) {
    ABSL_LOG(WARNING) << "The tokenizer changes the text it encodes (e.g. adds "
                         "a leading space), so constrained decoding won't "
                         "force canonical tokens.";
    tok_init.tokenize_fn = nullptr;
    tok_init.tokenize_user_data = nullptr;
    tok_init.tokenize_assumes_string = false;
    tok_init.use_approximate_greedy_tokenize_fn = true;
  }

  char error_buf[128];
  LlgTokenizer* llg_tokenizer =
      llg_new_tokenizer_v2(&tok_init, error_buf, sizeof(error_buf));
  if (llg_tokenizer == nullptr) {
    return absl::InternalError(error_buf);
  }

  return std::make_unique<LlgConstraintProvider>(
      std::move(token_lens), std::move(token_bytes), llg_tokenizer, llg_config);
}

LlgConstraintProvider::~LlgConstraintProvider() {
  llg_free_tokenizer(llg_tokenizer_);
}

absl::StatusOr<std::unique_ptr<Constraint>>
LlgConstraintProvider::CreateConstraint(ConstraintArg constraint_arg) const {
  if (!std::holds_alternative<LlGuidanceConstraintArg>(constraint_arg)) {
    return absl::InvalidArgumentError(
        "LlgConstraintProvider only supports LlGuidanceConstraintArg.");
  }
  const auto& llg_arg = std::get<LlGuidanceConstraintArg>(constraint_arg);

  LlgConstraintInit llg_constraint_init;
  llg_constraint_init_set_defaults(&llg_constraint_init, llg_tokenizer_);
  ::LlgConstraint* llg_constraint = CreateLlgConstraint(
      &llg_constraint_init, llg_arg.constraint_string, llg_arg.constraint_type);

  if (llg_get_error(llg_constraint)) {
    std::string error_message = llg_get_error(llg_constraint);
    llg_free_constraint(llg_constraint);
    return absl::InternalError(absl::StrCat(
        "Failed to create LLGuidance constraint: ", error_message));
  }

  return std::make_unique<LlgConstraint>(
      llg_constraint, static_cast<int>(token_lens_.size()),
      std::vector<int>(llg_config_.eos_ids.begin(), llg_config_.eos_ids.end()));
}

absl::StatusOr<std::unique_ptr<ConstraintProvider>>
CreateLlgToolCallConstraintProvider(
    const Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    absl::string_view fallback_eos_token,
    std::vector<std::string> special_tokens) {
  if (tokenizer == nullptr) {
    return absl::InvalidArgumentError(
        "Tokenizer must not be null when constrained decoding is enabled.");
  }
  LlGuidanceConfig llg_config;
  AddStopTokensAsEosIds(stop_token_ids, llg_config);
  if (llg_config.eos_ids.empty()) {
    if (absl::StatusOr<int> eos_id =
            const_cast<Tokenizer*>(tokenizer)->TokenToId(fallback_eos_token);
        eos_id.ok() && *eos_id >= 0) {
      llg_config.eos_ids.push_back(static_cast<uint32_t>(*eos_id));
    }
  }
  llg_config.special_tokens = std::move(special_tokens);
  absl::StatusOr<std::unique_ptr<ConstraintProvider>> provider =
      LlgConstraintProvider::Create(*tokenizer, std::move(llg_config));
  if (!provider.ok()) {
    return absl::Status(
        provider.status().code(),
        absl::StrCat("Failed to create the LLGuidance tool call constraint "
                     "provider: ",
                     provider.status().message()));
  }
  ABSL_LOG(INFO) << "Created the LLGuidance tool call constraint provider.";
  return provider;
}

}  // namespace litert::lm
