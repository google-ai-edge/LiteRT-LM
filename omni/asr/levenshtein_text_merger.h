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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_TEXT_MERGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_TEXT_MERGER_H_

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
// using Levenshtein distance sequence alignment.
class LevenshteinTextMerger : public TextMerger {
 public:
  explicit LevenshteinTextMerger(
      Stage<std::vector<Detokenizer::Word>>* absl_nonnull detokenizer)
      : TextMerger(detokenizer) {}

  // Flushes remaining unconfirmed text into output queue at end of stream.
  // Returns error if called when Schedule() is in progress.
  absl::Status Flush() override;

  // Returns cached unconfirmed words.
  // It's only used for testing. Not thread-safe.
  absl::Span<const std::string> unconfirmed_words_for_testing() const {
    return unconfirmed_words_;
  }

 private:
  // SingleThreadedStageWithDeque<MergeResult> implementation:
  absl::Status ScheduleInternal() override;

  // Resets internal cached state for a new audio stream.
  void ResetInternal() override;

  absl::Status Execute();

  std::vector<std::string> unconfirmed_words_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_TEXT_MERGER_H_
