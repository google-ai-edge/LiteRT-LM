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

#include "runtime/executor/llm_executor_processed_tokens.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// tokens_ must have at least one element.
ProcessedTokens::ProcessedTokens() : tokens_(1) {}

int ProcessedTokens::TokenCount() const {
  return GetStep() + (HasPendingInputToken() ? 1 : 0);
}

absl::Status ProcessedTokens::ReduceTokenCandidates(size_t index) {
  if (index >= tokens_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("index must be less than tokens_.size(), got ", index,
                     " vs ", tokens_.size()));
  }
  if (index > 0) {
    std::swap(tokens_[0], tokens_[index]);
  }
  tokens_.resize(1);
  return absl::OkStatus();
}

absl::Status ProcessedTokens::BroadcastTokenCandidates(size_t size) {
  if (tokens_.size() != 1) {
    return absl::FailedPreconditionError(
        "ExpandTokenCandidates called with tokens_.size() != 1.");
  }
  if (size < tokens_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("size must be greater than or equal to 1, got ", size));
  }

  tokens_.reserve(size);
  for (size_t i = 1; i < size; ++i) {
    tokens_.push_back(tokens_[0]);
  }
  return absl::OkStatus();
}

ProcessedTokens::StepAndToken ProcessedTokens::GetNextUnprocessedToken() const {
  return StepAndToken{.step = GetStep(), .token = GetPendingInputToken()};
}

void ProcessedTokens::AddProcessedTokens(absl::Span<const int> token_ids) {
  for (auto& t : tokens_) {
    t.token_ids.insert(t.token_ids.end(), token_ids.begin(), token_ids.end());
  }
}

absl::Status ProcessedTokens::AddPendingInputToken(
    const std::vector<std::shared_ptr<TokenData>>& token) {
  if (token.size() != tokens_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Token size must be equal to tokens_.size(), got ",
                     token.size(), " vs ", tokens_.size()));
  }
  if (HasPendingInputToken()) {
    return absl::AlreadyExistsError(
        "AddPendingInputToken called with an existing pending token.");
  }
  for (int i = 0; i < token.size(); ++i) {
    tokens_[i].pending_input_token = token[i];
  }
  return absl::OkStatus();
}

absl::Status ProcessedTokens::RollBackToStep(int new_step) {
  if (new_step < 0) {
    return absl::InternalError(
        absl::StrCat("new_step must be non-negative, got ", new_step));
  }
  if (new_step > TokenCount()) {
    return absl::InternalError(absl::StrCat(
        "new_step must be less than or equal to TokenCount(), got ", new_step,
        " vs ", TokenCount()));
  }

  if (new_step == TokenCount()) {
    return absl::OkStatus();
  }

  for (auto& t : tokens_) {
    t.token_ids.resize(new_step);
    t.pending_input_token = nullptr;
  }
  return absl::OkStatus();
}

std::vector<int> ProcessedTokens::GetTokenAtStep(int step) const {
  std::vector<int> token;
  if (step < 0 || step >= TokenCount()) {
    return token;
  }

  token.reserve(tokens_.size());
  if (step == GetStep() && HasPendingInputToken()) {
    for (const auto& t : tokens_) {
      token.push_back(t.pending_input_token->id());
    }
    return token;
  }

  for (const auto& t : tokens_) {
    token.push_back(t.token_ids[step]);
  }
  return token;
}

absl::Status ProcessedTokens::MarkPendingInputTokenAsProcessed() {
  if (!HasPendingInputToken()) {
    return absl::NotFoundError(
        "MarkPendingInputTokenAsProcessed called with no pending token.");
  }

  for (auto& t : tokens_) {
    t.token_ids.push_back(t.pending_input_token->id());
    t.pending_input_token = nullptr;
  }
  return absl::OkStatus();
}

std::vector<std::vector<int>> ProcessedTokens::GetCopyOfTokens() const {
  std::vector<std::vector<int>> copy_of_tokens;
  copy_of_tokens.reserve(tokens_.size());
  for (const auto& t : tokens_) {
    copy_of_tokens.push_back(t.token_ids);
    if (t.pending_input_token) {
      copy_of_tokens.back().push_back(t.pending_input_token->id());
    }
  }
  return copy_of_tokens;
}

const std::vector<int>& ProcessedTokens::GetTokensUnsafe() const {
  ABSL_CHECK_EQ(tokens_[0].pending_input_token, nullptr);
  return tokens_[0].token_ids;
}

void ProcessedTokens::InvalidatePendingInputToken() {
  for (auto& t : tokens_) {
    t.pending_input_token = nullptr;
  }
}

int ProcessedTokens::GetStep() const { return tokens_[0].token_ids.size(); }

bool ProcessedTokens::HasPendingInputToken() const {
  return tokens_[0].pending_input_token != nullptr;
}

std::vector<std::shared_ptr<TokenData>> ProcessedTokens::GetPendingInputToken()
    const {
  std::vector<std::shared_ptr<TokenData>> token;
  if (HasPendingInputToken()) {
    token.reserve(tokens_.size());
    for (const auto& t : tokens_) {
      token.push_back(t.pending_input_token);
    }
  }
  return token;
}

}  // namespace litert::lm
