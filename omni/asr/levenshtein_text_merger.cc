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

#include "omni/asr/levenshtein_text_merger.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/detokenizer.h"
#include "omni/asr/levenshtein_align.h"
#include "omni/asr/text_merger.h"

namespace litert::omni::asr {

void LevenshteinTextMerger::ResetInternal() {
  unconfirmed_words_.clear();
}

absl::Status LevenshteinTextMerger::ScheduleInternal() {
  SetState(State::kRunning);
  auto status = Execute();
  SetState(State::kIdle);
  return status;
}

absl::Status LevenshteinTextMerger::Execute() {
  ABSL_ASSIGN_OR_RETURN(auto curr_chunk_words, detokenizer_.GetOutput());

  std::vector<std::string> curr_strings;
  curr_strings.reserve(curr_chunk_words.size());
  for (const auto& word : curr_chunk_words) {
    if (word.IsEndOfChunk()) {
      break;
    }
    curr_strings.push_back(word.text);
  }

  if (unconfirmed_words_.empty()) {
    // Initial chunk: cache words as unconfirmed state.
    unconfirmed_words_ = std::move(curr_strings);
    PushOutput({"", absl::StrJoin(unconfirmed_words_, " ")});
    return absl::OkStatus();
  }

  auto align_codes = AlignTokens(unconfirmed_words_, curr_strings);

  size_t first_matching_hyp_idx = curr_strings.size();
  size_t first_matching_ref_idx = unconfirmed_words_.size();

  size_t ref_idx = 0;
  size_t hyp_idx = 0;
  for (const auto& code : align_codes) {
    if (code == AlignCode::kCorrect) {
      if (first_matching_hyp_idx == curr_strings.size()) {
        first_matching_hyp_idx = hyp_idx;
        first_matching_ref_idx = ref_idx;
      }
    }
    if (code == AlignCode::kDeletion || code == AlignCode::kSubstitution ||
        code == AlignCode::kCorrect) {
      ref_idx++;
    }
    if (code == AlignCode::kInsertion || code == AlignCode::kSubstitution ||
        code == AlignCode::kCorrect) {
      hyp_idx++;
    }
  }

  std::vector<std::string> confirmed;
  confirmed.reserve(first_matching_ref_idx);
  if (first_matching_ref_idx < unconfirmed_words_.size()) {
    for (size_t i = 0; i < first_matching_ref_idx; ++i) {
      confirmed.push_back(unconfirmed_words_[i]);
    }
  } else {
    confirmed = unconfirmed_words_;
  }

  std::vector<std::string> new_unconfirmed;
  if (first_matching_hyp_idx < curr_strings.size()) {
    new_unconfirmed.reserve(curr_strings.size() - first_matching_hyp_idx);
    for (size_t i = first_matching_hyp_idx; i < curr_strings.size(); ++i) {
      new_unconfirmed.push_back(curr_strings[i]);
    }
  } else {
    new_unconfirmed = curr_strings;
  }

  unconfirmed_words_ = std::move(new_unconfirmed);
  PushOutput(
      {absl::StrJoin(confirmed, " "), absl::StrJoin(unconfirmed_words_, " ")});
  return absl::OkStatus();
}

absl::Status LevenshteinTextMerger::Flush() {
  {
    absl::MutexLock lock(mutex_);
    if (state_ != State::kIdle) {
      return absl::FailedPreconditionError(
          "Flush() called while Schedule() is in progress.");
    }
    state_ = State::kRunning;
  }

  if (!unconfirmed_words_.empty()) {
    MergeResult result = {absl::StrJoin(unconfirmed_words_, " "), ""};
    unconfirmed_words_.clear();
    PushOutput(std::move(result));
  }

  SetState(State::kIdle);
  return absl::OkStatus();
}

}  // namespace litert::omni::asr
