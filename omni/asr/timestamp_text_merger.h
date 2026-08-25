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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TIMESTAMP_TEXT_MERGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TIMESTAMP_TEXT_MERGER_H_

#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/detokenizer.h"
#include "omni/asr/text_merger.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {

// Aligns and merges overlapping word streams from consecutive audio chunks
// based on word timestamps and windowed Levenshtein distance.
// It works better for live audio streaming than LevenshteinTextMerger.
class TimestampTextMerger : public TextMerger {
 public:
  explicit TimestampTextMerger(
      Stage<std::vector<Detokenizer::Word>>* absl_nonnull detokenizer,
      float overlap_ratio = 0.5f, int search_window = 4,
      int max_levenshtein_distance = 3, float pivot_factor = 0.6f);

  // Flushes remaining unconfirmed text into output queue at end of stream.
  // Returns error if called when Schedule() is in progress.
  absl::Status Flush() override;

  // Returns cached unconfirmed words.
  // It's only used for testing. Not thread-safe.
  absl::Span<const std::string> unconfirmed_words_for_testing() const {
    return prev_words_;
  }

 private:
  void ResetInternal() override;
  absl::Status ScheduleInternal() override;

  absl::Status Execute();
  void LogAndPushOutput(MergeResult output);
  void UpdateStreamTimestampOffset(std::optional<int> max_chunk_timestamp_ms,
                                   absl::Span<const int> curr_timestamps);

  const float overlap_ratio_;
  const int search_window_;
  const int max_levenshtein_distance_;
  const float pivot_factor_;

  std::vector<std::string> prev_words_;
  std::vector<int> prev_timestamps_;
  std::vector<std::string> last_confirmed_words_;
  int prev_word_index_of_unconfirmed_ = -1;
  int prev_word_index_of_pivot_ = -1;
  int stream_timestamp_offset_ms_ = 0;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TIMESTAMP_TEXT_MERGER_H_
