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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_

#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "support/util/io_types.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constraint.h"
#include "runtime/components/logits_processor/repetition_penalty_config.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

using InputText = ::litert::support::InputText;
using InputImage = ::litert::support::InputImage;
using InputImageEnd = ::litert::support::InputImageEnd;
using InputAudio = ::litert::support::InputAudio;
using InputAudioEnd = ::litert::support::InputAudioEnd;
using InputData = ::litert::support::InputData;
using VisionExecutorProperties = ::litert::support::VisionExecutorProperties;
using AudioExecutorProperties = ::litert::support::AudioExecutorProperties;

using ::litert::support::CreateInputDataCopy;
using ::litert::support::CreateInputDataVectorCopy;

// A struct that holds the scoring output for a single option.
struct ScorerOutput {
  // The score of the option text.
  // NOTE: this is the sum of the scores for each token in the option text.
  double score;
  // Character length of the option text.
  std::optional<int> option_text_char_length;
  // Token length of the option text.
  std::optional<int> option_text_token_length;
};

// The state of the task.
enum class TaskState {
  kUnknown,                 // The task is in an unknown state.
  kCreated,                 // The task is created and waiting for other
                            // dependent tasks.
                            // For example, the decode task is waiting for the
                            // prefill task to be done.
  kQueued,                  // The task is queued to be processed.
                            // For example, the decode task is queued to be
                            // processed after the prefill task is done.
  kProcessing,              // The task is being processed.
  kDone,                    // The task is done.
  kMaxNumTokensReached,     // The task is done because the max number of tokens
                            // is reached.
  kFailed,                  // The task is failed.
  kDependentTaskFailed,     // The task was cancelled because a dependent task
                            // failed.
  kCancelled,               // The task is cancelled.
  kDependentTaskCancelled,  // The task was cancelled because a dependent task
                            // was cancelled.
  kLastCallbackQueued,      // The last callback is queued to be called.
                            // This is internal state, will not be called to the
                            // user callback.
};
std::ostream& operator<<(std::ostream& os, const TaskState& task_state);

bool IsTaskEndState(const TaskState& task_state);

// A container to host the model responses.
class Responses {
 public:
  explicit Responses(TaskState task_state,
                     std::vector<std::string> response_texts = {},
                     std::vector<float> scores = {},
                     std::vector<int> token_lengths = {},
                     std::vector<std::vector<int>> token_ids = {})
      : task_state_(task_state),
        response_texts_(std::move(response_texts)),
        scores_(std::move(scores)),
        token_ids_(std::move(token_ids)) {
    if (!token_lengths.empty()) {
      token_lengths_ = std::move(token_lengths);
    }
  };

  // Returns the task state.
  const TaskState& GetTaskState() const { return task_state_; }

  // Sets the task state.
  void SetTaskState(TaskState task_state) { task_state_ = task_state; }

  // Returns the const texts vector.
  // The returned vector contains the response texts for each candidate output
  // string. In most cases, the candidate number is 1, the vector will contain a
  // single string. It requires the model to support batch size > 1 to have more
  // than one candidate.
  const std::vector<std::string>& GetTexts() const { return response_texts_; }

  // Returns the const scores vector.
  const std::vector<float>& GetScores() const { return scores_; }

  // Returns the mutable texts vector.
  std::vector<std::string>& GetMutableTexts() { return response_texts_; };

  // Returns the mutable scores vector.
  std::vector<float>& GetMutableScores() { return scores_; };

  // Returns the const token lengths vector.
  const std::optional<std::vector<int>>& GetTokenLengths() const {
    return token_lengths_;
  }

  // Returns the mutable token lengths vector.
  std::optional<std::vector<int>>& GetMutableTokenLengths() {
    return token_lengths_;
  };

  // Returns the const token ids vector.
  // The returned vector contains the token ids for each candidate output ids.
  // In most cases, the candidate number is 1, the vector will contain a single
  // vector of token ids. It requires the model to support batch size > 1 to
  // have more than one candidate.
  const std::vector<std::vector<int>>& GetTokenIds() const {
    return token_ids_;
  }

  // Returns the mutable token ids vector.
  std::vector<std::vector<int>>& GetMutableTokenIds() { return token_ids_; };

  // Returns the const token scores vector.
  const std::optional<std::vector<std::vector<float>>>& GetTokenScores() const {
    return token_scores_;
  }

  // Returns the mutable token scores vector.
  std::optional<std::vector<std::vector<float>>>& GetMutableTokenScores() {
    return token_scores_;
  };

 private:
  // The state of the task.
  TaskState task_state_;

  // The output vector of response tokens (as strings).
  std::vector<std::string> response_texts_;

  // The output vector of scores for each response text. The "score" is pulled
  // from the probability of the last token in the response text.
  std::vector<float> scores_;

  // The output vector of token lengths for each response text. Optional.
  std::optional<std::vector<int>> token_lengths_;

  // The output vector of token scores for each response text. Optional.
  std::optional<std::vector<std::vector<float>>> token_scores_;

  // The output vector of token ids for each response text.
  std::vector<std::vector<int>> token_ids_;
};
std::ostream& operator<<(std::ostream& os, const Responses& responses);

// Class to store the data for a single turn of the benchmark. A "turn" is
// defined as a single RunPrefill or RunDecode call.
struct BenchmarkTurnData {
  absl::Duration duration;  // Duration of this entire operation/turn.
  uint64_t num_tokens;      // The number of tokens processed in this turn.
  BenchmarkTurnData(uint64_t tokens, absl::Duration dur);
};
std::ostream& operator<<(std::ostream& os, const BenchmarkTurnData& data);

// Class to store and manage comprehensive performance benchmark information for
// LLMs.
class BenchmarkInfo {
 public:
  explicit BenchmarkInfo(const proto::BenchmarkParams& benchmark_params);
  const proto::BenchmarkParams& GetBenchmarkParams() const;

  // --- Methods to record data ---

  enum class InitPhase {
    kModelAssets,
    kLlmMetadata,
    kExecutor,
    kTokenizer,
    kSession,
    kConversation,
    kTotal,
  };
  static constexpr absl::string_view InitPhaseToString(InitPhase phase) {
    switch (phase) {
      case InitPhase::kModelAssets:
        return "Init Model assets";
      case InitPhase::kLlmMetadata:
        return "Init LLM metadata";
      case InitPhase::kExecutor:
        return "Init Executor";
      case InitPhase::kTokenizer:
        return "Init Tokenizer";
      case InitPhase::kSession:
        return "Init Session";
      case InitPhase::kConversation:
        return "Init Conversation";
      case InitPhase::kTotal:
        return "Init Total";
    }
  }

  // Time the start and end of an init phase. The method will return an error
  // if the methods are called out of order (i.e. one end after one start).
  // Each phase can only be timed once, and the subsequent calls will return
  // error.
  absl::Status TimeInitPhaseStart(InitPhase phase);
  absl::Status TimeInitPhaseEnd(InitPhase phase);

  // An alternative to TimeInitPhaseStart and TimeInitPhaseEnd. Allows directly
  // recording the duration of a phase. This is useful when the BenchmarkInfo
  // object is not available to mark the start time as needed.
  absl::Status InitPhaseRecord(InitPhase phase, absl::Duration duration);

  // Time the start and end of a prefill/decode turn. The num_prefill_tokens
  // should be the number of tokens processed in this turn. The method will
  // return an error if the methods are called out of order (i.e. one end after
  // one start).
  absl::Status TimePrefillTurnStart();
  absl::Status TimePrefillTurnEnd(uint64_t num_prefill_tokens);
  absl::Status TimeDecodeTurnStart();
  absl::Status TimeDecodeTurnEnd(uint64_t num_decode_tokens);
  absl::Status TimeTextToTokenIdsStart();
  absl::Status TimeTextToTokenIdsEnd(uint64_t num_tokens);
  // Time the duration between two consecutive marks. Useful for profiling the
  // pipeline at a specific point. For example:
  //   RETURN_IF_ERROR(benchmark_info.TimeMarkDelta("sampling"));
  //   ... actual sampling logics ...
  //   RETURN_IF_ERROR(benchmark_info.TimeMarkDelta("sampling"));
  //
  // The method will return the duration as the time delta between the two
  // TimeMarkDelta("sampling") calls. The duration will be stored / recorded for
  // each unique mark name.
  absl::Status TimeMarkDelta(const std::string& mark_name);

  // --- Getters for raw data ---
  const std::map<std::string, absl::Duration>& GetInitPhases() const;
  const std::map<std::string, absl::Duration>& GetMarkDurations() const;

  // --- Calculated metrics and getters for Prefill ---
  uint64_t GetTotalPrefillTurns() const;
  absl::StatusOr<BenchmarkTurnData> GetPrefillTurn(int turn_index) const;
  double GetPrefillTokensPerSec(int turn_index) const;

  // --- Calculated metrics and getters for Decode ---
  uint64_t GetTotalDecodeTurns() const;
  absl::StatusOr<BenchmarkTurnData> GetDecodeTurn(int turn_index) const;
  double GetDecodeTokensPerSec(int turn_index) const;

  // --- Calculated metrics and getters for TextToTokenIds ---
  uint64_t GetTotalTextToTokenIdsTurns() const;
  absl::StatusOr<BenchmarkTurnData> GetTextToTokenIdsTurn(int turn_index) const;

  // --- Gets the time to the first token ---
  // Note that the first time to token doesn't include the time for
  // initialization. It is the sum of the prefill time for the first turn and
  // the time spent for decoding the first token.
  double GetTimeToFirstToken() const;

 private:
  proto::BenchmarkParams benchmark_params_;

  // Map of phase names to start time.
  std::map<std::string, absl::Time> start_time_map_;
  std::map<std::string, absl::Time> mark_time_map_;
  // The current index of the prefill / decode / text_to_token_ids turn.
  int prefill_turn_index_ = 0;
  int decode_turn_index_ = 0;
  int text_to_token_ids_turn_index_ = 0;

  std::map<std::string, absl::Duration> init_phases_;
  std::map<std::string, absl::Duration> mark_durations_;
  std::vector<BenchmarkTurnData> prefill_turns_;
  std::vector<BenchmarkTurnData> decode_turns_;
  std::vector<BenchmarkTurnData> text_to_token_ids_turns_;
};
std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info);

// Configurations used for a single decode request.
class DecodeConfig {
 public:
  // Creates a default DecodeConfig.
  static DecodeConfig CreateDefault();

  // Sets the repetition penalty config to penalize repetitive tokens during
  // decoding.
  void SetRepetitionPenaltyConfig(
      const RepetitionPenaltyConfig& repetition_penalty_config) {
    repetition_penalty_config_ = repetition_penalty_config;
  }

  // Returns the repetition penalty config.
  RepetitionPenaltyConfig GetRepetitionPenaltyConfig() const {
    return repetition_penalty_config_;
  }

  // Sets the optional constraint used to guide the generation.
  // `DecodeConfig` does not take ownership of the `constraint`, which must
  // outlives the single generation process.
  void SetConstraint(Constraint* absl_nullable constraint) {
    constraint_ = constraint;
  }

  // Returns a pointer to the constraint, or nullptr if no constraint is set.
  Constraint* absl_nullable GetConstraint() const { return constraint_; }

  // Sets the max output tokens.
  void SetMaxOutputTokens(int max_output_tokens) {
    max_output_tokens_ = max_output_tokens;
  }

  // Returns the max output tokens.
  std::optional<int> GetMaxOutputTokens() const { return max_output_tokens_; }

 private:
  DecodeConfig() = default;

  RepetitionPenaltyConfig repetition_penalty_config_ =
      RepetitionPenaltyConfig::Default();
  Constraint* absl_nullable constraint_ = nullptr;
  std::optional<int> max_output_tokens_ = std::nullopt;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
