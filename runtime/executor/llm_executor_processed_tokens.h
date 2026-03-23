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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_PROCESSED_TOKENS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_PROCESSED_TOKENS_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Information which is used to process a token.
class TokenData {
 public:
  explicit TokenData(int token_id) : id_(token_id) {}

  TokenData(int token_id, std::vector<float> token_embedding,
            std::vector<float> token_per_layer_embedding)
      : id_(token_id),
        embedding_(std::move(token_embedding)),
        per_layer_embedding_(std::move(token_per_layer_embedding)) {}

  int id() const { return id_; }

  absl::Span<const float> embedding() const { return embedding_; }
  std::vector<float>& mutable_embedding() { return embedding_; }

  absl::Span<const float> per_layer_embedding() const {
    return per_layer_embedding_;
  }
  std::vector<float>& mutable_per_layer_embedding() {
    return per_layer_embedding_;
  }

 private:
  // The token id that is to be processed.
  const int id_;

  // May contain the embedding corresponding to the token id.
  std::vector<float> embedding_;

  // May contain the per-layer embedding corresponding to the token id.
  std::vector<float> per_layer_embedding_;
};

// Keeps track of processed tokens during the LLM execution.
//
// This class is used by `ProcessedContext` to store the sequence of tokens
// that have been processed so far. It keeps track of both the processed tokens
// and a pending input token, if any, which may be used by backends which
// require an input token to be provided during Decode.
//
// During prefill, one set of processed tokens are maintained.
// During decode, output batch size (or number of output candidates) sets of
// processed tokens are maintained.
class ProcessedTokens {
 public:
  // Tokens and their corresponding step. Number of tokens will be:
  // - Empty if the step does not correspond to the tokens in this
  //   ProcessedTokens.
  // - One if the step is for prefill.
  // - Size of the output batch if the steps corresponds to decode.
  struct StepAndToken {
    int step;
    std::vector<std::shared_ptr<TokenData>> token;
  };

  ProcessedTokens();

  ProcessedTokens(const ProcessedTokens&) = default;
  ProcessedTokens(ProcessedTokens&&) noexcept = default;
  ProcessedTokens& operator=(const ProcessedTokens&) = default;
  ProcessedTokens& operator=(ProcessedTokens&&) noexcept = default;

  // Returns the number of processed tokens inclusive of the pending input
  // token, if any.
  int TokenCount() const;

  // Reduces the token candidates to 1 with one of given index.
  // It will be called when LLM switches from decode to prefill.
  absl::Status ReduceTokenCandidates(size_t index);

  // Broadcasts the token candidates to the given size.
  // It will be called when LLM switches from prefill to decode.
  absl::Status BroadcastTokenCandidates(size_t size);

  // Returns `pending_input_token_` and its step, if it exists; otherwise,
  // the step after the last processed token.
  StepAndToken GetNextUnprocessedToken() const;

  // Appends the given tokens to the list of processed tokens.
  void AddProcessedTokens(absl::Span<const int> token_ids);

  // Add token (or tokens during Decode) as "pending" input token(s), which
  // indicates that the token has not yet been processed by the LLM, but is part
  // of the current context and is to be processed during the next Prefill or
  // Decode step. This may be used by backends which require an input token to
  // be provided during Decode.
  absl::Status AddPendingInputToken(
      const std::vector<std::shared_ptr<TokenData>>& token);

  // Reverts the processed tokens to the given step. This new step must be
  // non-negative and smaller than the current token count.
  absl::Status RollBackToStep(int new_step);

  // Returns the token at the given `step` or empty if the step does not
  // correspond to a token. It may contains tokens more than one during decode
  // when decode batch size is greater than one.
  std::vector<int> GetTokenAtStep(int step) const;

  // Marks the pending input token as processed.
  // Returns kNotFoundError if there is no pending input token.
  absl::Status MarkPendingInputTokenAsProcessed();

  // Returns a deep copy of the complete list of processed tokens, inclusive of
  // the pending input token, if any.
  std::vector<std::vector<int>> GetCopyOfTokens() const;

  // WARNING: This function returns a reference to the internal `tokens_`
  // directly, which may not include the pending input token. This method MUST
  // NOT be used in code that runs a backend which uses a pending input token.
  const std::vector<int>& GetTokensUnsafe() const;

  // Invalidates the pending input token, if any.
  void InvalidatePendingInputToken();

 private:
  int GetStep() const;
  bool HasPendingInputToken() const;
  std::vector<std::shared_ptr<TokenData>> GetPendingInputToken() const;

  struct Tokens {
    std::vector<int> token_ids;
    std::shared_ptr<TokenData> pending_input_token;
  };

  // tokens_.size() is 1 if prefill or output batch size if decode.
  std::vector<Tokens> tokens_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_PROCESSED_TOKENS_H_
