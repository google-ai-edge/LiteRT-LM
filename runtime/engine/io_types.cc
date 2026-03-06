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

#include "runtime/engine/io_types.h"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

absl::StatusOr<absl::string_view> InputText::GetRawTextString() const {
  if (std::holds_alternative<std::string>(data_)) {
    return absl::string_view(std::get<std::string>(data_));
  }
  return absl::FailedPreconditionError(
      "The text is preprocessed and does not have raw text bytes.");
}

absl::StatusOr<const TensorBuffer*> InputText::GetPreprocessedTextTensor()
    const {
  if (std::holds_alternative<TensorBuffer>(data_)) {
    return &std::get<TensorBuffer>(data_);
  }
  return absl::FailedPreconditionError(
      "The text is not preprocessed and does not have a tensor.");
}

absl::StatusOr<InputText> InputText::CreateCopy() const {
  if (std::holds_alternative<std::string>(data_)) {
    return InputText(std::move(std::get<std::string>(data_)));
  } else if (std::holds_alternative<TensorBuffer>(data_)) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_clone,
                            std::get<TensorBuffer>(data_).Duplicate());
    return InputText(std::move(tensor_buffer_clone));
  }
  return absl::FailedPreconditionError(
      "The data_ is not a string or a TensorBuffer.");
}

absl::StatusOr<absl::string_view> InputImage::GetRawImageBytes() const {
  if (std::holds_alternative<std::string>(data_)) {
    return absl::string_view(std::get<std::string>(data_));
  }
  if (std::holds_alternative<absl::string_view>(data_)) {
    return std::get<absl::string_view>(data_);
  }
  return absl::FailedPreconditionError(
      "The image is preprocessed and does not have raw image bytes.");
}

absl::StatusOr<const TensorBuffer*> InputImage::GetPreprocessedImageTensor()
    const {
  if (std::holds_alternative<TensorBuffer>(data_)) {
    return &std::get<TensorBuffer>(data_);
  }
  return absl::FailedPreconditionError(
      "The image is not preprocessed and does not have a tensor.");
}

absl::StatusOr<const absl::flat_hash_map<std::string, TensorBuffer>*>
InputImage::GetPreprocessedImageTensorMap() const {
  if (std::holds_alternative<absl::flat_hash_map<std::string, TensorBuffer>>(
          data_)) {
    return &std::get<absl::flat_hash_map<std::string, TensorBuffer>>(data_);
  }
  return absl::FailedPreconditionError(
      "The image is not preprocessed and does not have a tensor map.");
}

absl::StatusOr<InputImage> InputImage::CreateCopy() const {
  if (std::holds_alternative<std::string>(data_)) {
    return InputImage(std::move(std::get<std::string>(data_)));
  } else if (std::holds_alternative<absl::string_view>(data_)) {
    // Deep copy the string view.
    return InputImage(std::string(std::get<absl::string_view>(data_)));
  } else if (std::holds_alternative<TensorBuffer>(data_)) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_clone,
                            std::get<TensorBuffer>(data_).Duplicate());
    return InputImage(std::move(tensor_buffer_clone));
  } else if (std::holds_alternative<
                 absl::flat_hash_map<std::string, TensorBuffer>>(data_)) {
    const auto& tensor_buffer_map =
        std::get<absl::flat_hash_map<std::string, TensorBuffer>>(data_);
    absl::flat_hash_map<std::string, TensorBuffer> tensor_buffer_map_copy;
    for (const auto& [key, value] : tensor_buffer_map) {
      LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_clone, value.Duplicate());
      tensor_buffer_map_copy[key] = std::move(tensor_buffer_clone);
    }
    return InputImage(std::move(tensor_buffer_map_copy));
  }
  return absl::FailedPreconditionError(
      "The data_ is not a string or a TensorBuffer.");
}

absl::StatusOr<absl::string_view> InputAudio::GetRawAudioBytes() const {
  if (std::holds_alternative<std::string>(data_)) {
    return absl::string_view(std::get<std::string>(data_));
  }
  return absl::FailedPreconditionError("The audio is not raw audio bytes.");
}

absl::StatusOr<const TensorBuffer*> InputAudio::GetPreprocessedAudioTensor()
    const {
  if (std::holds_alternative<TensorBuffer>(data_)) {
    return &std::get<TensorBuffer>(data_);
  }
  return absl::FailedPreconditionError(
      "The audio is not a preprocessed tensor.");
}

absl::StatusOr<absl::Span<const float>> InputAudio::GetPcmFrames() const {
  if (std::holds_alternative<std::vector<float>>(data_)) {
    return std::get<std::vector<float>>(data_);
  }
  return absl::FailedPreconditionError("The audio is not a float vector.");
}

absl::StatusOr<InputAudio> InputAudio::CreateCopy() const {
  if (std::holds_alternative<std::string>(data_)) {
    return InputAudio(std::move(std::get<std::string>(data_)));
  } else if (std::holds_alternative<TensorBuffer>(data_)) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_clone,
                            std::get<TensorBuffer>(data_).Duplicate());
    return InputAudio(std::move(tensor_buffer_clone));
  } else if (std::holds_alternative<std::vector<float>>(data_)) {
    return InputAudio(std::get<std::vector<float>>(data_));
  }
  return absl::FailedPreconditionError(
      "The data_ is not a string, TensorBuffer, or float vector.");
}

std::ostream& operator<<(std::ostream& os, const TaskState& task_state) {
  switch (task_state) {
    case TaskState::kCreated:
      os << "Created";
      break;
    case TaskState::kQueued:
      os << "Queued";
      break;
    case TaskState::kProcessing:
      os << "Processing";
      break;
    case TaskState::kDone:
      os << "Done";
      break;
    case TaskState::kMaxNumTokensReached:
      os << "MaxNumTokensReached";
      break;
    case TaskState::kFailed:
      os << "Failed";
      break;
    case TaskState::kDependentTaskFailed:
      os << "DependentTaskFailed";
      break;
    case TaskState::kCancelled:
      os << "Cancelled";
      break;
    case TaskState::kDependentTaskCancelled:
      os << "DependentTaskCancelled";
      break;
    case TaskState::kLastCallbackQueued:
      os << "LastCallbackQueued";
      break;
    default:
      os << "Unknown";
      break;
  }
  return os;
}

bool IsTaskEndState(const TaskState& task_state) {
  return task_state == TaskState::kDone ||
         task_state == TaskState::kMaxNumTokensReached ||
         task_state == TaskState::kFailed ||
         task_state == TaskState::kDependentTaskFailed ||
         task_state == TaskState::kCancelled ||
         task_state == TaskState::kDependentTaskCancelled;
}

std::ostream& operator<<(std::ostream& os, const Responses& responses) {
  os << "Task State: " << responses.GetTaskState() << std::endl;
  if (responses.GetTexts().empty()) {
    os << " No reponses." << std::endl;
    return os;
  }
  os << "Total candidates: " << responses.GetTexts().size() << ":" << std::endl;

  for (int i = 0; i < responses.GetTexts().size(); ++i) {
    if (responses.GetScores().size() > i) {
      os << "  Candidate " << i << " (score: " << responses.GetScores()[i]
         << "):" << std::endl;
    } else {
      os << "  Candidate " << i << " (score: N/A):" << std::endl;
    }

    os << "    Text: \"" << responses.GetTexts()[i] << "\"" << std::endl;
  }
  return os;  // Return the ostream to allow chaining
}

// --- BenchmarkTurnData Method Definitions ---
BenchmarkTurnData::BenchmarkTurnData(uint64_t tokens, absl::Duration dur)
    : duration(dur), num_tokens(tokens) {}

BenchmarkInfo::BenchmarkInfo(const proto::BenchmarkParams& benchmark_params)
    : benchmark_params_(benchmark_params) {};

const proto::BenchmarkParams& BenchmarkInfo::GetBenchmarkParams() const {
  return benchmark_params_;
}

absl::Status BenchmarkInfo::TimeInitPhaseStart(InitPhase phase) {
  std::string phase_name = std::string(InitPhaseToString(phase));
  if (start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Phase ", phase_name, " already started."));
  }
  start_time_map_[phase_name] = absl::Now();
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimeInitPhaseEnd(InitPhase phase) {
  std::string phase_name = std::string(InitPhaseToString(phase));
  if (!start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Phase ", phase_name, " not started."));
  }
  init_phases_[phase_name] = absl::Now() - start_time_map_[phase_name];
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::InitPhaseRecord(InitPhase phase,
                                            absl::Duration duration) {
  std::string phase_name = std::string(InitPhaseToString(phase));
  if (init_phases_.contains(phase_name)) {
    return absl::InternalError(absl::StrCat("Phase ", phase_name,
                                            " already recorded with duration ",
                                            init_phases_[phase_name], "."));
  }
  init_phases_[phase_name] = duration;
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimeMarkDelta(const std::string& mark_name) {
  if (mark_time_map_.contains(mark_name)) {
    mark_durations_[mark_name] = absl::Now() - mark_time_map_[mark_name];
  }
  mark_time_map_[mark_name] = absl::Now();
  return absl::OkStatus();
}

const std::map<std::string, absl::Duration>& BenchmarkInfo::GetMarkDurations()
    const {
  return mark_durations_;
}

absl::Status BenchmarkInfo::TimePrefillTurnStart() {
  const std::string phase_name = absl::StrCat("prefill:", prefill_turn_index_);
  if (start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Prefill turn ", phase_name, " already started."));
  }
  start_time_map_[phase_name] = absl::Now();
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimePrefillTurnEnd(uint64_t num_prefill_tokens) {
  const std::string phase_name = absl::StrCat("prefill:", prefill_turn_index_);
  if (!start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Prefill turn ", phase_name, " not started."));
  }
  prefill_turns_.emplace_back(num_prefill_tokens,
                              absl::Now() - start_time_map_[phase_name]);
  prefill_turn_index_++;
  return absl::OkStatus();
}

absl::StatusOr<BenchmarkTurnData> BenchmarkInfo::GetPrefillTurn(
    int turn_index) const {
  if (turn_index < 0 ||
      static_cast<size_t>(turn_index) >= prefill_turns_.size()) {
    return absl::OutOfRangeError(absl::StrCat(
        "Prefill turn index ", turn_index,
        " is out of bounds. Total prefill turns: ", prefill_turns_.size()));
  }
  return prefill_turns_[turn_index];
}

absl::Status BenchmarkInfo::TimeDecodeTurnStart() {
  const std::string phase_name = absl::StrCat("decode:", decode_turn_index_);
  if (start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Decode turn ", phase_name, " already started."));
  }
  start_time_map_[phase_name] = absl::Now();
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimeDecodeTurnEnd(uint64_t num_decode_tokens) {
  const std::string phase_name = absl::StrCat("decode:", decode_turn_index_);
  if (!start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Decode turn ", phase_name, " not started."));
  }
  decode_turns_.emplace_back(num_decode_tokens,
                             absl::Now() - start_time_map_[phase_name]);
  decode_turn_index_++;
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimeTextToTokenIdsStart() {
  const std::string phase_name =
      absl::StrCat("text_to_token_ids:", text_to_token_ids_turn_index_);
  if (start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("TextToTokenIds turn ", phase_name, " already started."));
  }
  start_time_map_[phase_name] = absl::Now();
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimeTextToTokenIdsEnd(uint64_t num_tokens) {
  const std::string phase_name =
      absl::StrCat("text_to_token_ids:", text_to_token_ids_turn_index_);
  if (!start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("TextToTokenIds turn ", phase_name, " not started."));
  }
  text_to_token_ids_turns_.emplace_back(
      num_tokens, absl::Now() - start_time_map_[phase_name]);
  text_to_token_ids_turn_index_++;
  return absl::OkStatus();
}

absl::StatusOr<BenchmarkTurnData> BenchmarkInfo::GetDecodeTurn(
    int turn_index) const {
  if (turn_index < 0 ||
      static_cast<size_t>(turn_index) >= decode_turns_.size()) {
    return absl::OutOfRangeError(absl::StrCat(
        "Decode turn index ", turn_index,
        " is out of bounds. Total decode turns: ", decode_turns_.size()));
  }
  return decode_turns_[turn_index];
}

const std::map<std::string, absl::Duration>& BenchmarkInfo::GetInitPhases()
    const {
  return init_phases_;
}

uint64_t BenchmarkInfo::GetTotalPrefillTurns() const {
  return prefill_turns_.size();
}

double BenchmarkInfo::GetPrefillTokensPerSec(int turn_index) const {
  if (turn_index < 0 ||
      static_cast<size_t>(turn_index) >= prefill_turns_.size()) {
    return 0.0;
  }
  const auto& turn = prefill_turns_[turn_index];
  if (turn.duration <= absl::ZeroDuration()) {
    return 0.0;  // Avoid division by zero or negative duration
  }
  double turn_seconds = absl::ToDoubleSeconds(turn.duration);
  if (turn_seconds <= 0.0) {  // Additional check for very small durations
    return 0.0;
  }
  return static_cast<double>(turn.num_tokens) / turn_seconds;
}

uint64_t BenchmarkInfo::GetTotalTextToTokenIdsTurns() const {
  return text_to_token_ids_turns_.size();
}

absl::StatusOr<BenchmarkTurnData> BenchmarkInfo::GetTextToTokenIdsTurn(
    int turn_index) const {
  if (turn_index < 0 ||
      static_cast<size_t>(turn_index) >= text_to_token_ids_turns_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("TextToTokenIds turn index ", turn_index,
                     " is out of bounds. Total text_to_token_ids turns: ",
                     text_to_token_ids_turns_.size()));
  }
  return text_to_token_ids_turns_[turn_index];
}

uint64_t BenchmarkInfo::GetTotalDecodeTurns() const {
  return decode_turns_.size();
}

// Interpreted as Generated Tokens Per Second for the specified turn_index.
// The "Avg" in the name might be a misnomer if it's for a specific turn,
// but implementing based on the header's declaration.
double BenchmarkInfo::GetDecodeTokensPerSec(int turn_index) const {
  if (turn_index < 0 ||
      static_cast<size_t>(turn_index) >= decode_turns_.size()) {
    // Consider logging an error or throwing std::out_of_range
    return 0.0;
  }
  const auto& turn = decode_turns_[turn_index];
  if (turn.duration <= absl::ZeroDuration()) {
    return 0.0;  // Avoid division by zero or negative duration
  }
  double turn_seconds = absl::ToDoubleSeconds(turn.duration);
  if (turn_seconds <= 0.0) {  // Additional check for very small durations
    return 0.0;
  }
  // This calculates tokens/sec for the specific turn.
  // If "turns/sec" for a specific turn was intended, the logic would be
  // different (1.0 / turn_seconds). Given the name and typical metrics,
  // tokens/sec for the turn seems more likely.
  return static_cast<double>(turn.num_tokens) / turn_seconds;
}

double BenchmarkInfo::GetTimeToFirstToken() const {
  if (prefill_turns_.empty() || decode_turns_.empty() ||
      prefill_turns_[0].num_tokens == 0 || decode_turns_[0].num_tokens == 0) {
    return 0.0;  // No valid data to calculate time to first token.
  }
  double first_decode_token_seconds = absl::ToDoubleSeconds(
      decode_turns_[0].duration / decode_turns_[0].num_tokens);
  double first_prefill_token_seconds =
      absl::ToDoubleSeconds(prefill_turns_[0].duration);
  return first_decode_token_seconds + first_prefill_token_seconds;
}

std::ostream& operator<<(std::ostream& os, const BenchmarkTurnData& data) {
  os << "Processed " << data.num_tokens << " tokens in " << data.duration
     << " duration." << std::endl;
  return os;
}

std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info) {
  os << std::fixed << std::setprecision(2);

  os << "BenchmarkInfo:" << std::endl;
  os << "  Init Phases (" << info.GetInitPhases().size() << "):" << std::endl;
  if (info.GetInitPhases().empty()) {
    os << "    No init phases recorded." << std::endl;
  } else {
    const auto& init_phases = info.GetInitPhases();
    bool has_conversation_creation = false;
    std::string conversation_creation_phase_name =
        std::string(BenchmarkInfo::InitPhaseToString(
            BenchmarkInfo::InitPhase::kConversation));
    std::string session_creation_phase_name = std::string(
        BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kSession));
    for (const auto& [phase_name, phase_duration] : init_phases) {
      if (phase_name == conversation_creation_phase_name) {
        has_conversation_creation = true;
        break;
      }
    }
    for (const auto& phase : init_phases) {
      if (has_conversation_creation &&
          phase.first == session_creation_phase_name) {
        // Session creation time is included in conversation creation time,
        // so skip it.
        continue;
      }
      os << "    - " << phase.first << ": "
         << absl::ToDoubleMilliseconds(phase.second) << " ms" << std::endl;
    }
  }

  os << "--------------------------------------------------" << std::endl;
  os << "  Time to first token: " << info.GetTimeToFirstToken() << " s"
     << std::endl;

  os << "--------------------------------------------------" << std::endl;
  os << "  Prefill Turns (Total " << info.GetTotalPrefillTurns()
     << " turns):" << std::endl;
  if (info.GetTotalPrefillTurns() == 0) {
    os << "    No prefill turns recorded." << std::endl;
  } else {
    for (uint64_t i = 0; i < info.GetTotalPrefillTurns(); ++i) {
      os << "    Prefill Turn " << i + 1 << ": " << *info.GetPrefillTurn(i);
      os << "      Prefill Speed: "
         << info.GetPrefillTokensPerSec(static_cast<int>(i)) << " tokens/sec."
         << std::endl;
    }
  }

  os << "--------------------------------------------------" << std::endl;
  os << "  Decode Turns (Total " << info.GetTotalDecodeTurns()
     << " turns):" << std::endl;
  if (info.GetTotalDecodeTurns() == 0) {
    os << "    No decode turns recorded." << std::endl;
  } else {
    for (uint64_t i = 0; i < info.GetTotalDecodeTurns(); ++i) {
      os << "    Decode Turn " << i + 1 << ": " << *info.GetDecodeTurn(i);
      os << "      Decode Speed: "
         << info.GetDecodeTokensPerSec(static_cast<int>(i)) << " tokens/sec."
         << std::endl;
    }
  }
  os << "--------------------------------------------------" << std::endl;
  os << "  TextToTokenIds Turns (Total " << info.GetTotalTextToTokenIdsTurns()
     << " turns):" << std::endl;
  if (info.GetTotalTextToTokenIdsTurns() == 0) {
    os << "    No text to token ids turns recorded." << std::endl;
  } else {
    for (uint64_t i = 0; i < info.GetTotalTextToTokenIdsTurns(); ++i) {
      auto turn = info.GetTextToTokenIdsTurn(i);
      if (turn.ok()) {
        os << "    Turn " << i + 1 << ": " << turn->duration << ", "
           << turn->num_tokens << " tokens" << std::endl;
      }
    }
  }

  os << "--------------------------------------------------" << std::endl;

  if (!info.GetMarkDurations().empty()) {
    os << "  Mark Durations (" << info.GetMarkDurations().size()
       << "):" << std::endl;
    for (const auto& [mark_name, duration] : info.GetMarkDurations()) {
      os << "    - " << mark_name << ": " << duration << std::endl;
    }
  }
  os << "--------------------------------------------------" << std::endl;
  return os;
}

DecodeConfig DecodeConfig::CreateDefault() { return DecodeConfig(); }

std::ostream& operator<<(std::ostream& os,
                         const VisionExecutorProperties& properties) {
  os << "num_tokens_per_image: " << properties.num_tokens_per_image
     << std::endl;
  os << "patch_num_shrink_factor: "
     << (properties.patch_num_shrink_factor.has_value()
             ? absl::StrCat(properties.patch_num_shrink_factor.value())
             : "not set")
     << std::endl;
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const AudioExecutorProperties& properties) {
  os << "is_streaming_model: " << properties.is_streaming_model << std::endl;
  os << "streaming_chunk_size: " << properties.streaming_chunk_size
     << std::endl;
  os << "streaming_chunk_overlap_size: "
     << properties.streaming_chunk_overlap_size << std::endl;
  os << "audio_shrink_factor: " << properties.audio_shrink_factor << std::endl;
  return os;
}

}  // namespace litert::lm
