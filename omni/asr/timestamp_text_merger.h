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

// Aligns and merges overlapping word streams from consecutive audio chunks.
//
// When words carry timestamps (TDT / CTC decoders) the merge is time-based:
// words of the previous chunk that lie before the current chunk are
// confirmed; inside the overlap of the two windows the words are aligned by
// text and time, the previous chunk's words are kept up to a pivot
// (`pivot_factor` of the overlap, where they still had context on both sides)
// and the current chunk's words are used after it. Because words are placed
// by time rather than by count, a hallucinated burst at the end of the
// previous window cannot shift the seam, and the current chunk's new words are
// never discarded wholesale. Without timestamps a windowed Levenshtein
// alignment on word counts is used instead.
// It works better for live audio streaming than LevenshteinTextMerger.
class TimestampTextMerger : public TextMerger {
 public:
  explicit TimestampTextMerger(
      Stage<std::vector<Detokenizer::Word>>* absl_nonnull detokenizer,
      float overlap_ratio = 0.5f, int search_window = 4,
      int max_levenshtein_distance = 3, float pivot_factor = 0.6f);

  // Returns cached unconfirmed words.
  // It's only used for testing. Not thread-safe.
  absl::Span<const std::string> unconfirmed_words_for_testing() const {
    return prev_words_;
  }
  // Returns the absolute start of the next chunk in timestamp units.
  // It's only used for testing. Not thread-safe.
  double stream_timestamp_offset_for_testing() const {
    return stream_timestamp_offset_ms_;
  }

 private:
  absl::Status FlushInternal() override;
  void ResetInternal() override;
  absl::Status ScheduleInternal() override;

  absl::Status Execute();
  // Time-based merge of `curr_words` (absolute `curr_timestamps`, one per
  // word) into the previous chunk's unconfirmed words; updates `prev_words_`,
  // `prev_timestamps_` and `last_confirmed_words_`.
  void MergeByTime(absl::Span<const std::string> curr_words,
                   absl::Span<const int> curr_timestamps);
  // Legacy merge by word counts and windowed Levenshtein alignment, used when
  // timestamps are unavailable.
  void MergeByWordCount(absl::Span<const std::string> curr_words,
                        absl::Span<const int> curr_timestamps);
  void LogAndPushOutput(MergeResult output);
  // Advances the stream offset by the hop of a chunk ending at `chunk_end_ms`
  // (relative to its start).
  void UpdateStreamTimestampOffset(int chunk_end_ms);

  const float overlap_ratio_;
  const int search_window_;
  const int max_levenshtein_distance_;
  const float pivot_factor_;

  std::vector<std::string> prev_words_;
  std::vector<int> prev_timestamps_;
  std::vector<std::string> last_confirmed_words_;
  int prev_word_index_of_unconfirmed_ = -1;
  int prev_word_index_of_pivot_ = -1;
  // Absolute start of the current chunk (sum of the previous chunks' steps).
  // Kept as a double: the step is `end_of_chunk * (1 - overlap_ratio)`, which
  // is fractional in encoder frames.
  double stream_timestamp_offset_ms_ = 0.0;
  // Absolute end of the previous chunk's window (-1 if unknown).
  double prev_chunk_end_ms_ = -1.0;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TIMESTAMP_TEXT_MERGER_H_
