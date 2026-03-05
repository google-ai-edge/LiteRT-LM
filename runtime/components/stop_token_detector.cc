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
    ABSL_LOG(INFO) << absl::StrFormat(
        "Stop token sequence %s already exists. Skipping "
        "adding the stop token sequence.",
        PrintSequence(stop_sequence));
    return absl::OkStatus();
  }

  stop_sequences_storage_.push_back(stop_sequence);

  // Add a progress tracker for the new stop sequence for each batch item.
  for (auto& progress_vector_for_item : batch_item_match_progress_) {
    progress_vector_for_item.push_back(0);
  }
  return absl::OkStatus();
}

void StopTokenDetector::ResetBatch(size_t batch_size) {
  int new_batch_size = batch_size == 0 ? stop_token_found_.size() : batch_size;
  stop_token_found_.assign(new_batch_size, false);
  max_batch_item_match_progress_.assign(new_batch_size, 0);
  // Initialize progress for each batch item for all currently defined stop
  // sequences.
  batch_item_match_progress_.assign(
      new_batch_size, std::vector<int>(stop_sequences_storage_.size(), 0));
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
    max_batch_item_match_progress_[i] = 0;
    int current_token_id = latest_tokens[i];
    for (size_t k = 0; k < stop_sequences_storage_.size(); ++k) {
      const auto& stop_seq_k =
          stop_sequences_storage_[k];  // Guaranteed non-empty
      int& current_match_len_for_k = batch_item_match_progress_[i][k];

      if (current_match_len_for_k < stop_seq_k.size() &&
          stop_seq_k[current_match_len_for_k] == current_token_id) {
        current_match_len_for_k++;
      } else {
        // Mismatch or sequence completed; reset progress for this stop_seq_k.
        // Check if current token starts stop_seq_k anew.
        if (stop_seq_k[0] == current_token_id) {
          current_match_len_for_k = 1;
        } else {
          current_match_len_for_k = 0;
        }
      }

      if (current_match_len_for_k > 0 &&
          current_match_len_for_k == stop_seq_k.size()) {
        stop_token_found_[i] = true;
        matched_stop_sequence_length_[i] = stop_seq_k.size();
      }
      max_batch_item_match_progress_[i] =
          std::max(max_batch_item_match_progress_[i], current_match_len_for_k);
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
