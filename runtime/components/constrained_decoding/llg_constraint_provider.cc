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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/llg_constraint.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/tokenizer.h"
#include "llguidance.h"

namespace litert::lm {
namespace {

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
  if (!llg_config.eos_id.has_value()) {
    return absl::InvalidArgumentError("LlGuidanceConfig::eos_id must be set.");
  }

  std::vector<std::string> tokens = tokenizer.GetTokens();

  std::vector<uint32_t> token_lens;
  std::vector<uint8_t> token_bytes;
  size_t total_size = 0;

  token_lens.reserve(tokens.size());
  for (const auto& token : tokens) {
    token_lens.push_back(token.size());
    total_size += token.size();
  }

  token_bytes.reserve(total_size);
  for (const auto& token : tokens) {
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

  LlgTokenizerInit tok_init = {
      .vocab_size = static_cast<uint32_t>(tokens.size()),
      .tok_eos = *llg_config.eos_id,
      .token_lens = token_lens.data(),
      .token_bytes = token_bytes.data(),
      .tokenize_assumes_string = false,
      .tokenize_fn = tokenize_fn,
      .tokenize_user_data = &tokenizer,
  };

  char error_buf[128];
  LlgTokenizer* llg_tokenizer =
      llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));
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

  return std::make_unique<LlgConstraint>(llg_constraint,
                                         static_cast<int>(token_lens_.size()),
                                         *llg_config_.eos_id);
}

}  // namespace litert::lm
