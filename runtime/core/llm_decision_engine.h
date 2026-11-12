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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_LLM_DECISION_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_LLM_DECISION_ENGINE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/decision_engine.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"

namespace litert::lm {

namespace internal {
// Per-model decision settings of `LlmDecisionEngine`, defined in the .cc file.
struct LlmDecisionModelConfig;
}  // namespace internal

// `DecisionEngine` that scores the choices with an LLM, built only on the
// public `Engine` / `SessionInterface` API so that `Conversation`,
// `SessionInterface` and `LlmDecisionEngine` can share one loaded `Engine`
// (model weights and backend resources).
//
// Each question is answered by scoring single-letter choice labels rather than
// by generating and parsing text:
//   1. Each question becomes one prompt ("branch") whose options are labeled
//      `'A'..'Z'`. The shared input is placed before the question, so all
//      branches of a request share a long token prefix.
//   2. Each branch is rendered with the model's chat template and tokenized as
//      a whole, then prefilled with `PrefillPreprocessedContents`. On Gemma 4
//      the shared prefix is prefilled once per `Decide()` call and the session
//      is rewound to it with `RewindToStep` before each branch. Other models,
//      and Gemma 4 sessions that cannot be rewound, use a fresh session per
//      branch.
//   3. Each label is scored as `log P(label | branch prompt)` with
//      `RunTextScoring` on a `Clone()` of the prefilled session, the same way
//      `Conversation::RunTextScoring` scores text without changing the
//      conversation. This costs one decode step per label. The label text
//      is `" A"` after the assistant prefix (`"Answer:"`), or `"A"` if there is
//      no prefix.
//   4. A temperature-scaled softmax over the labels gives the option
//      probabilities, `selected_key`, `expected_score` and `confidence`.
//
// Thread-safe: concurrent calls to `Decide()` are serialized by `mutex_`.
class LlmDecisionEngine : public DecisionEngine {
 public:
  // Creates an `LlmDecisionEngine` that shares `engine` with other users (e.g.
  // `Conversation` or `SessionInterface`). `engine` must outlive the returned
  // object. The `Engine` runs the tasks of all its sessions one at a time, so
  // `Decide()` and other sessions on the same `Engine` may wait for each other.
  static absl::StatusOr<std::unique_ptr<LlmDecisionEngine>> Create(
      Engine& engine);

  // Creates an `LlmDecisionEngine` that owns a default `Engine` created from
  // `engine_settings`.
  static absl::StatusOr<std::unique_ptr<LlmDecisionEngine>> Create(
      EngineSettings engine_settings);

  ~LlmDecisionEngine() override;

  absl::StatusOr<DecisionResponse> Decide(
      const DecisionRequest& request) override;

 private:
  LlmDecisionEngine(
      std::unique_ptr<Engine> owned_engine, Engine& engine,
      SessionConfig session_config, PromptTemplate prompt_template,
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      bool supports_step_rewind,
      std::unique_ptr<const internal::LlmDecisionModelConfig> model_config);

  // Renders `user_turn_text` with the model's chat template, prepends the
  // start token if any, and tokenizes the result.
  absl::StatusOr<std::vector<int>> RenderAndTokenizeBranch(
      absl::string_view user_turn_text) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Prefills each branch and scores its choice labels. Returns one vector of
  // label log-probabilities per branch.
  absl::StatusOr<std::vector<std::vector<double>>> ExecuteAndScoreBranches(
      absl::Span<const std::vector<int>> branch_tokens,
      absl::Span<const size_t> branch_option_counts)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Returns `log P(label | prompt)` for labels `'A' .. 'A' + num_options - 1`
  // given the prompt prefilled in `session`. `session` is left unchanged.
  absl::StatusOr<std::vector<double>> ScoreLabels(SessionInterface& session,
                                                  size_t num_options)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Renders `user_turn_text` through the model's chat template with thinking
  // disabled and appends the assistant prefix.
  absl::StatusOr<std::string> RenderBranchPrompt(
      absl::string_view user_turn_text) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Synchronously prefills pre-tokenized `tokens` on `session`.
  absl::Status PrefillTokens(SessionInterface& session,
                             absl::Span<const int> tokens) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Turns an executor "not enough state entries" error from prefilling
  // `num_tokens` tokens at `start_step` into a `ResourceExhausted` error.
  // Other statuses are returned unchanged.
  absl::Status AnnotateCapacityError(absl::Status status, size_t start_step,
                                     size_t num_tokens) const;

  // Set only by `Create(EngineSettings)`; `engine_` refers to it then.
  std::unique_ptr<Engine> owned_engine_;
  Engine& engine_;
  SessionConfig session_config_;
  PromptTemplate prompt_template_;
  std::unique_ptr<ModelDataProcessor> model_data_processor_;
  const int start_token_id_;
  const std::string bos_string_;
  const bool supports_step_rewind_;
  const std::unique_ptr<const internal::LlmDecisionModelConfig> model_config_;
  // Text scored for each choice label `'A'..'Z'`, or why it cannot be scored.
  const std::vector<absl::StatusOr<std::string>> label_texts_;

  // Serializes `Decide()` calls.
  mutable absl::Mutex mutex_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_LLM_DECISION_ENGINE_H_
