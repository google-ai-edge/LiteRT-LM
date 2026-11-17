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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_END_OF_TURN_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_END_OF_TURN_H_

#include <optional>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/executor/executor_stats.h"

namespace litert::lm {

// The message that ends a turn of `Conversation::SendMessageAsync` looks like
//
//   {"role": "assistant", "finish_reason": "stop", "executor_stats": {...}}
//
// `finish_reason` says why the turn ended; content chunks never have it, and a
// null `finish_reason` counts as missing.
// `executor_stats` holds the executor stats of the turn, or null if it has
// none (e.g. an appended message, which isn't decoded).
inline constexpr absl::string_view kFinishReasonKey = "finish_reason";
inline constexpr absl::string_view kExecutorStatsKey = "executor_stats";

// Why a turn ended. More reasons may be added, so consumers must treat any
// `finish_reason` they don't know as the end of the turn.
enum class FinishReason {
  // The model finished its response.
  kStop,
  // The response reached the maximum number of tokens.
  kMaxTokens,
  // The message was appended to the conversation without a response.
  kAppended,
  // The turn was cancelled. Reserved: cancelled turns end with
  // `absl::CancelledError` today.
  kCancelled,
};

// Returns the `finish_reason` string of `reason`, e.g. "stop".
absl::string_view FinishReasonToString(FinishReason reason);

// Returns whether `message` ends a turn, i.e. has a non-null `finish_reason`.
// A null message also ends a turn, as the end-of-turn marker used to be a null
// `Message()`.
bool IsEndOfTurnMessage(const Message& message);

// Returns the message that ends a turn for `reason`, with `executor_stats`
// (null if `std::nullopt`).
Message CreateEndOfTurnMessage(
    FinishReason reason, const std::optional<ExecutorStats>& executor_stats);

// Converts `stats` to JSON:
//
//   {"module_name": "LLM", "metrics": {"plain_steps": 1, ...},
//    "latencies_us": {"plain_step_time": 51234, ...}, "substats": [...]}
//
// Integer metrics become JSON integers and double metrics JSON floats; NaN or
// infinite double metrics are left out. Latencies are whole microseconds.
nlohmann::ordered_json ExecutorStatsToJson(const ExecutorStats& stats);

// Converts JSON written by `ExecutorStatsToJson()` back to `ExecutorStats`.
// Missing fields are left empty. Returns an error if a field has the wrong
// type.
absl::StatusOr<ExecutorStats> ExecutorStatsFromJson(
    const nlohmann::ordered_json& json);

// Returns the executor stats of the end-of-turn message `message`, or
// `std::nullopt` if it has none.
absl::StatusOr<std::optional<ExecutorStats>> GetExecutorStats(
    const Message& message);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_END_OF_TURN_H_
