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

#include "runtime/components/stop_token_detector.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

// Prints a sequence of integers.
inline std::string PrintSequence(const std::vector<int>& sequence) {
  std::string existing_sequence_str = "{";
  for (size_t i = 0; i < sequence.size(); ++i) {
    absl::StrAppend(&existing_sequence_str, sequence[i]);
    if (i < sequence.size() - 1) {
      absl::StrAppend(&existing_sequence_str, ", ");
    }
  }
  absl::StrAppend(&existing_sequence_str, "}");
  return existing_sequence_str;
}

}  // namespace

StopTokenDetector::StopTokenDetector(size_t batch_size) {
  ABSL_CHECK_GT(batch_size, 0) << "Batch size must be greater than 0.";
  ResetBatch(batch_size);
}

absl::Status StopTokenDetector::AddStopTokenSequence(
    const std::vector<int>& stop_sequence) {
  if (stop_sequence.empty()) {
    return absl::InvalidArgumentError(
        "Cannot add an empty stop token sequence.");
  }

  // Check if the sequence already exists
  if (std::find(stop_sequences_storage_.begin(), stop_sequences_storage_.end(),
                stop_sequence) != stop_sequences_storage_.end()) {
    ABSL_VLOG(1) << absl::StrFormat(
        "Stop token sequence %s already exists. Skipping "
        "adding the stop token sequence.",
        PrintSequence(stop_sequence));
    return absl::OkStatus();
  }

  stop_sequences_storage_.push_back(stop_sequence);
  max_stop_sequence_length_ =
      std::max(max_stop_sequence_length_, stop_sequence.size());
  return absl::OkStatus();
}

void StopTokenDetector::ResetBatch(size_t batch_size) {
  int new_batch_size = batch_size == 0 ? stop_token_found_.size() : batch_size;
  stop_token_found_.assign(new_batch_size, false);
  max_batch_item_match_progress_.assign(new_batch_size, 0);
  batch_item_token_history_.assign(new_batch_size, std::vector<int>());
  matched_stop_sequence_length_.assign(new_batch_size, 0);
}

// Processes the latest incoming token for each sequence in the batch.
absl::Status StopTokenDetector::ProcessTokens(
    absl::Span<const int> latest_tokens) {
  if (latest_tokens.size() != stop_token_found_.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Size of latest_tokens (%d) does not match configured batch size (%d).",
        latest_tokens.size(), stop_token_found_.size()));
  }
  if (stop_sequences_storage_.empty()) {  // No stop sequences to check against.
    return absl::InvalidArgumentError(
        "No stop sequences to check against. Did you forget to call "
        "AddStopTokenSequence()?");
  }

  for (size_t i = 0; i < latest_tokens.size(); ++i) {
    if (stop_token_found_[i]) {
      // Already stopped, but increase the length of the matched stop sequence.
      matched_stop_sequence_length_[i]++;
      continue;
    }

    int current_token_id = latest_tokens[i];
    auto& history = batch_item_token_history_[i];
    history.push_back(current_token_id);
    if (history.size() > max_stop_sequence_length_) {
      history.erase(history.begin());
    }

    max_batch_item_match_progress_[i] = 0;
    for (const auto& stop_seq : stop_sequences_storage_) {
      // Check for full match.
      if (history.size() >= stop_seq.size()) {
        bool full_match = true;
        for (size_t idx = 0; idx < stop_seq.size(); ++idx) {
          if (history[history.size() - stop_seq.size() + idx] !=
              stop_seq[idx]) {
            full_match = false;
            break;
          }
        }
        if (full_match) {
          stop_token_found_[i] = true;
          matched_stop_sequence_length_[i] = stop_seq.size();
          max_batch_item_match_progress_[i] = stop_seq.size();
          break;  // Stop token found, no need to check other sequences.
        }
      }

      // Check for partial match (only if stop token is not found yet).
      if (!stop_token_found_[i]) {
        for (int len = std::min(history.size(), stop_seq.size() - 1);
             len > max_batch_item_match_progress_[i]; --len) {
          bool partial_match = true;
          for (int idx = 0; idx < len; ++idx) {
            if (history[history.size() - len + idx] != stop_seq[idx]) {
              partial_match = false;
              break;
            }
          }
          if (partial_match) {
            max_batch_item_match_progress_[i] = len;
            break;
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status StopTokenDetector::ProcessTokens(
    const std::vector<std::vector<int>>& latest_tokens) {
  if (latest_tokens.size() != stop_token_found_.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Size of latest_tokens (%d) does not match configured batch size (%d).",
        latest_tokens.size(), stop_token_found_.size()));
  }
  std::vector<int> flattened_tokens;
  flattened_tokens.reserve(latest_tokens.size());
  for (auto& tokens : latest_tokens) {
    RET_CHECK_EQ(tokens.size(), 1)
        << "The current implementation of ProcessTokens() requires that "
           "latest_tokens must contain only single tokens.";
    flattened_tokens.push_back(tokens[0]);
  }
  return ProcessTokens(flattened_tokens);
}

int StopTokenDetector::MaxPartialStopTokenLength(int index) const {
  return max_batch_item_match_progress_[index];
}

const std::vector<int>& StopTokenDetector::GetStepsBeforeStopTokens() const {
  return matched_stop_sequence_length_;
}

absl::StatusOr<bool> StopTokenDetector::AllDone() const {
  if (stop_token_found_.empty()) {
    return absl::FailedPreconditionError(
        "The Detector is not initialized with non-zero batch size. Did you "
        "forget to call ResetBatch() or AddStopTokenSequence() ??");
  }
  return std::all_of(stop_token_found_.begin(), stop_token_found_.end(),
                     [](bool found) { return found; });
}

}  // namespace litert::lm
