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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_STOP_TOKEN_DETECTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_STOP_TOKEN_DETECTOR_H_

#include <cstddef>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Detects stop token sequences in a batch of token streams.
// Tracks match progress for each potential stop sequence independently per
// batch item. Stop sequences can be added dynamically. Example usage:
//
//   StopTokenDetector detector(batch_size);
//   RETURN_IF_ERROR(detector.AddStopTokenSequence({1}));
//   RETURN_IF_ERROR(detector.AddStopTokenSequence({4,5,3,2}));
//   // ... add more stop sequences as needed ...
//   // ... process tokens ...
//   RETURN_IF_ERROR(detector.ProcessTokens(token_stream_1));
//   RETURN_IF_ERROR(detector.ProcessTokens(token_stream_2));
//   // ... process more tokens ...
//   // Check if stop token was found...
//   ASIGN_OR_RETURN(bool done, detector.AllDone());
//   if (done) {
//     // Stop token found...
//   }
//
class StopTokenDetector {
 public:
  // Constructs the detector for a given batch size.
  // No stop sequences are defined initially; use AddStopTokenSequence().
  //   - batch_size: The number of sequences to track in the batch.
  explicit StopTokenDetector(size_t batch_size);

  // Adds a new stop token sequence.
  //   - stop_sequence: The token ID sequence to add. Must not be empty.
  //   - InvalidArgumentError if sequence is empty or added before.
  absl::Status AddStopTokenSequence(const std::vector<int>& stop_sequence);

  // Resets detector state for a new batch size or clears existing state. Note
  // that this does not clear the stop sequences themselves.
  //   - batch_size: The new number of sequences in the batch. If zeros is
  //     passed, the detector will be reset with the same batch size.
  void ResetBatch(size_t batch_size = 0);

  // Processes the latest incoming token for each sequence in the batch.
  //   - latest_tokens Span of token IDs, one per batch sequence. Size must
  //     match batch_size.
  // Returns an error status on precondition failure.
  absl::Status ProcessTokens(absl::Span<const int> latest_tokens);

  // Processes the latest incoming token for each sequence in the batch.
  //   - latest_tokens Span of token IDs, one per batch sequence. Size must
  //     match batch_size.
  // Returns an error status on precondition failure.
  absl::Status ProcessTokens(
      const std::vector<std::vector<int>>& latest_tokens);

  // Returns a const reference to the vector containing the lengths of the
  // matched stop token sequences for all batch items. If a batch item has not
  // yet matched a stop sequence, its corresponding value in the vector will be
  // 0 (or its value from the last match if ResetBatch hasn't been called).
  // Returns a const reference to the vector of matched stop sequence lengths.
  const std::vector<int>& GetStepsBeforeStopTokens() const;

  // Checks if all sequences in the current batch have found a stop token.
  // Returns True if all sequences are done or batch is empty.
  absl::StatusOr<bool> AllDone() const;

  // Returns a const reference to the vector indicating whether a stop token
  // has been found for each batch item.
  const std::vector<bool>& GetStopTokensFound() const {
    return stop_token_found_;
  }

  // Returns the maximum length of the partial stop token sequence found for the
  // given batch index. zero if no partial stop token sequence is found or -1 if
  // the stop token is already found.
  int MaxPartialStopTokenLength(int index) const;

 private:
  // Stores all added stop sequences.
  std::vector<std::vector<int>> stop_sequences_storage_;

  // batch_item_match_progress_[i][k]: current match length for batch item 'i'
  // against stop_sequences_storage_[k].
  std::vector<std::vector<int>> batch_item_match_progress_;

  // max_batch_item_match_progress_[i]: maximum match length for batch item 'i'
  // against all stop_sequences_storage_[k].
  std::vector<int> max_batch_item_match_progress_;

  // stop_token_found_[i]: true if batch item 'i' has matched a stop sequence.
  std::vector<bool> stop_token_found_;

  // matched_stop_sequence_length_[i]: length of the token ids the detokenizer
  // should ignore. This includes the length of the detected stop sequence plus
  // (if batch_size > 1) the additional length until the other batch items
  // also match the stop sequence.
  std::vector<int> matched_stop_sequence_length_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_STOP_TOKEN_DETECTOR_H_
