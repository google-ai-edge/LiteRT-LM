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

#include "omni/speech2text/levenshtein_text_merger.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/speech2text/detokenizer.h"
#include "omni/speech2text/levenshtein_align.h"
#include "omni/speech2text/text_merger.h"
#include "omni/utils/text_utils.h"

namespace litert::omni::asr {
namespace {

std::vector<std::string> ExtractChunkWords(
    absl::Span<const Detokenizer::Word> chunk_words) {
  std::vector<std::string> words;
  words.reserve(chunk_words.size());
  for (const auto& word : chunk_words) {
    if (word.IsEndOfChunk()) {
      break;
    }
    words.push_back(word.text);
  }
  return words;
}

}  // namespace

void LevenshteinTextMerger::ResetInternal() { unconfirmed_words_.clear(); }

absl::Status LevenshteinTextMerger::ScheduleInternal() {
  SetState(State::kRunning);
  auto status = Execute();
  SetState(State::kIdle);
  return status;
}

absl::Status LevenshteinTextMerger::Execute() {
  ABSL_ASSIGN_OR_RETURN(auto curr_chunk_words, detokenizer_.GetOutput());

  std::vector<std::string> curr_strings =
      SplitCjkCharacters(ExtractChunkWords(curr_chunk_words));
  if (curr_strings.empty()) {
    PushOutput(MergeResult{"", JoinWithCjkAwareSpacing(unconfirmed_words_)});
    return absl::OkStatus();
  }

  if (unconfirmed_words_.empty()) {
    // Initial chunk: cache words as unconfirmed state.
    unconfirmed_words_ = std::move(curr_strings);
    PushOutput(MergeResult{"", JoinWithCjkAwareSpacing(unconfirmed_words_)});
    return absl::OkStatus();
  }

  const std::vector<WordIndexPair> best_run = FindBestSeamMatchRun(
      unconfirmed_words_, 0, curr_strings, curr_strings.size());
  const size_t first_matching_ref_idx =
      best_run.empty() ? unconfirmed_words_.size() : best_run.front().first;
  const size_t first_matching_hyp_idx =
      best_run.empty() ? curr_strings.size() : best_run.front().second;

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
  PushOutput(MergeResult{JoinWithCjkAwareSpacing(confirmed),
                         JoinWithCjkAwareSpacing(unconfirmed_words_)});
  return absl::OkStatus();
}

absl::Status LevenshteinTextMerger::FlushInternal() {
  if (!unconfirmed_words_.empty()) {
    MergeResult result = {JoinWithCjkAwareSpacing(unconfirmed_words_), ""};
    unconfirmed_words_.clear();
    PushOutput(std::move(result));
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::asr
