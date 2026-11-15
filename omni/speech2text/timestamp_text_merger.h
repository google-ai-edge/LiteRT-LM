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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_TIMESTAMP_TEXT_MERGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_TIMESTAMP_TEXT_MERGER_H_

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/speech2text/detokenizer.h"
#include "omni/speech2text/text_merger.h"

namespace litert::omni::asr {

// Aligns and merges overlapping word streams from consecutive audio chunks.
//
// When words carry timestamps (TDT / CTC decoders) the merge is time-based:
// words of the previous chunk that lie before the current chunk are
// confirmed; inside the overlap of the two windows the words are aligned by
// text and time, the previous chunk's words are kept up to a pivot
// (`pivot_factor` of the overlap, where they still had context on both sides)
// and the current chunk's words are used after it. Without timestamps a
// seam-run Levenshtein alignment on word counts is used instead.
// It works better for live audio streaming than LevenshteinTextMerger.
class TimestampTextMerger : public TextMerger {
 public:
  explicit TimestampTextMerger(
      Stage<std::vector<Detokenizer::Word>>* absl_nonnull detokenizer,
      float overlap_ratio = 0.5f, int search_window = 4,
      float pivot_factor = 0.6f);

  // Returns cached unconfirmed words.
  // It's only used for testing. Not thread-safe.
  absl::Span<const std::string> unconfirmed_words_for_testing() const {
    return absl::MakeConstSpan(prev_words_)
        .subspan(std::min(confirmed_prefix_size_, prev_words_.size()));
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
  // Merge by word counts and seam-run alignment, used when timestamps are
  // unavailable. Re-seeds `prev_timestamps_` when `curr_timestamps` are usable.
  void MergeByWordCount(absl::Span<const std::string> curr_words,
                        absl::Span<const int> curr_timestamps);
  void LogAndPushOutput(MergeResult output);
  // Advances the stream offset by the hop of a chunk ending at `chunk_end_ms`
  // (relative to its start).
  void UpdateStreamTimestampOffset(int chunk_end_ms);

  const float overlap_ratio_;
  const int search_window_;
  const float pivot_factor_;

  std::vector<std::string> prev_words_;
  std::vector<int> prev_timestamps_;
  std::vector<std::string> last_confirmed_words_;
  // Absolute start of the current chunk (sum of the previous chunks' steps).
  // Kept as a double: the step is `end_of_chunk * (1 - overlap_ratio)`, which
  // is fractional in encoder frames.
  double stream_timestamp_offset_ms_ = 0.0;
  // Absolute end of the previous chunk's window (-1 if unknown).
  double prev_chunk_end_ms_ = -1.0;
  // Duration of the most recent non-empty chunk, used to advance
  // `stream_timestamp_offset_ms_` across silent chunks that carry no sentinel.
  int last_chunk_end_ms_ = 0;
  // Whether `prev_words_` currently holds a progressive warm-start prefix that
  // should be replaced rather than merged by the next chunk.
  bool prev_is_warm_start_ = false;
  // Number of consecutive non-warm-start empty chunks observed since the last
  // chunk with spoken words.
  int consecutive_empty_chunks_ = 0;
  // Number of leading words in `prev_words_` that have already been emitted in
  // `confirmed_text` (while remaining in `prev_words_` for overlap alignment on
  // the next window).
  size_t confirmed_prefix_size_ = 0;
  // Upper timestamp bound of words already finalized during a live silence
  // pause, used to avoid re-emitting pre-pause words that remain in the left
  // overlap prefix of a subsequent window.
  int confirmed_until_ms_ = -1;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_TIMESTAMP_TEXT_MERGER_H_
