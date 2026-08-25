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

#include "omni/asr/timestamp_text_merger.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/asr/detokenizer.h"
#include "omni/asr/levenshtein_align.h"
#include "omni/asr/text_merger.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {
namespace {

std::vector<int> MergeTimestamps(absl::Span<const int> prev_timestamps,
                                 absl::Span<const int> curr_timestamps,
                                 int prev_word_index_of_unconfirmed,
                                 int prev_word_index_of_pivot,
                                 size_t num_merged_words) {
  if (num_merged_words == 0) {
    return {};
  }

  if (prev_timestamps.empty() || prev_word_index_of_unconfirmed == -1 ||
      prev_word_index_of_unconfirmed >=
          static_cast<int>(prev_timestamps.size()) ||
      prev_word_index_of_unconfirmed == prev_word_index_of_pivot) {
    return std::vector<int>(curr_timestamps.begin(), curr_timestamps.end());
  }

  size_t num_words_from_prev =
      prev_word_index_of_pivot - prev_word_index_of_unconfirmed;
  if (num_words_from_prev >
      prev_timestamps.size() - prev_word_index_of_unconfirmed) {
    num_words_from_prev =
        prev_timestamps.size() - prev_word_index_of_unconfirmed;
  }

  std::vector<int> result(
      prev_timestamps.begin() + prev_word_index_of_unconfirmed,
      prev_timestamps.begin() + prev_word_index_of_unconfirmed +
          num_words_from_prev);
  if (num_merged_words > num_words_from_prev) {
    size_t num_words_from_curr = num_merged_words - num_words_from_prev;
    if (num_words_from_curr <= curr_timestamps.size()) {
      result.insert(result.end(), curr_timestamps.end() - num_words_from_curr,
                    curr_timestamps.end());
    } else {
      result.insert(result.end(), curr_timestamps.begin(),
                    curr_timestamps.end());
    }
  }
  return result;
}

std::vector<std::string> MergeIntoUnconfirmedText(
    absl::Span<const std::string> prev_words,
    absl::Span<const int> prev_timestamps,
    absl::Span<const std::string> curr_words,
    absl::Span<const int> curr_timestamps, float overlap_ratio,
    float pivot_factor, int prev_word_index_of_unconfirmed,
    int prev_word_index_of_pivot) {
  if (prev_word_index_of_unconfirmed == -1 ||
      prev_word_index_of_unconfirmed >= static_cast<int>(prev_words.size())) {
    return std::vector<std::string>(curr_words.begin(), curr_words.end());
  }

  bool use_timestamps = !curr_timestamps.empty() && !prev_timestamps.empty() &&
                        (prev_timestamps.back() > prev_timestamps.front() ||
                         curr_timestamps.back() > curr_timestamps.front());
  size_t num_words_before_pivot_in_current = 0;
  if (use_timestamps) {
    int timestamp_at_pivot =
        (prev_word_index_of_pivot >= static_cast<int>(prev_timestamps.size()))
            ? prev_timestamps.back() + 1
            : prev_timestamps[prev_word_index_of_pivot];
    for (int ts : curr_timestamps) {
      if (ts < timestamp_at_pivot) {
        num_words_before_pivot_in_current++;
      }
    }
  } else {
    int num_words_overlap_in_current = std::max<int>(
        std::ceil(curr_words.size() * overlap_ratio),
        static_cast<int>(prev_words.size()) - prev_word_index_of_unconfirmed);
    num_words_before_pivot_in_current = std::min<size_t>(
        static_cast<size_t>(num_words_overlap_in_current * pivot_factor),
        curr_words.size());
  }

  size_t num_words_in_prev_unconfirmed = std::max<int>(
      0, prev_word_index_of_pivot - prev_word_index_of_unconfirmed);
  num_words_before_pivot_in_current = std::min<size_t>(
      num_words_before_pivot_in_current, num_words_in_prev_unconfirmed);

  size_t num_words_after_pivot_in_current =
      curr_words.size() - num_words_before_pivot_in_current;
  if (num_words_after_pivot_in_current <
      prev_words.size() - prev_word_index_of_pivot) {
    return std::vector<std::string>(
        prev_words.begin() + prev_word_index_of_unconfirmed, prev_words.end());
  }

  std::vector<std::string> result;
  for (int i = prev_word_index_of_unconfirmed; i < prev_word_index_of_pivot;
       ++i) {
    result.push_back(prev_words[i]);
  }
  std::vector<std::string> curr_tail(
      curr_words.begin() + num_words_before_pivot_in_current, curr_words.end());
  std::vector<std::string> deduped_tail =
      DedupWords(absl::MakeSpan(result), absl::MakeSpan(curr_tail));
  result.insert(result.end(), deduped_tail.begin(), deduped_tail.end());
  return result;
}

}  // namespace

TimestampTextMerger::TimestampTextMerger(
    Stage<std::vector<Detokenizer::Word>>* absl_nonnull detokenizer,
    float overlap_ratio, int search_window, int max_levenshtein_distance,
    float pivot_factor)
    : TextMerger(detokenizer),
      overlap_ratio_(overlap_ratio),
      search_window_(search_window),
      max_levenshtein_distance_(max_levenshtein_distance),
      pivot_factor_(pivot_factor) {}

void TimestampTextMerger::ResetInternal() {
  prev_words_.clear();
  prev_timestamps_.clear();
  last_confirmed_words_.clear();
  prev_word_index_of_unconfirmed_ = -1;
  prev_word_index_of_pivot_ = -1;
  stream_timestamp_offset_ms_ = 0;
}

absl::Status TimestampTextMerger::ScheduleInternal() {
  SetState(State::kRunning);
  auto status = Execute();
  SetState(State::kIdle);
  return status;
}

absl::Status TimestampTextMerger::Execute() {
  LITERT_ASSIGN_OR_RETURN(auto curr_chunk_words, detokenizer_.GetOutput());
  if (curr_chunk_words.empty()) {
    LogAndPushOutput(
        {"", prev_words_.empty() ? "" : absl::StrJoin(prev_words_, " ")});
    return absl::OkStatus();
  }

  std::vector<std::string> curr_words;
  std::vector<int> curr_timestamps;
  std::optional<int> max_chunk_timestamp_ms;
  curr_words.reserve(curr_chunk_words.size());
  for (const auto& word : curr_chunk_words) {
    if (word.IsEndOfChunk()) {
      if (word.timestamp_ms.has_value()) {
        max_chunk_timestamp_ms = word.timestamp_ms.value();
      }
      continue;
    }
    curr_words.push_back(word.text);
    if (word.timestamp_ms.has_value()) {
      curr_timestamps.push_back(word.timestamp_ms.value() +
                                stream_timestamp_offset_ms_);
    }
  }

  if (overlap_ratio_ == 0.0f) {
    LogAndPushOutput({absl::StrJoin(curr_words, " "), ""});
    return absl::OkStatus();
  }

  if (prev_words_.empty()) {
    prev_words_ = curr_words;
    prev_timestamps_ = curr_timestamps;
    UpdateStreamTimestampOffset(max_chunk_timestamp_ms, curr_timestamps);
    LogAndPushOutput({"", absl::StrJoin(prev_words_, " ")});
    return absl::OkStatus();
  }

  int middle_index_to_search = 0;
  if (!prev_timestamps_.empty() && !curr_timestamps.empty() &&
      (prev_timestamps_.back() > prev_timestamps_.front() ||
       curr_timestamps.back() > curr_timestamps.front())) {
    int target_timestamp = curr_timestamps.front();
    int found_idx = -1;
    for (size_t i = 0; i < prev_timestamps_.size(); ++i) {
      if (prev_timestamps_[i] >= target_timestamp) {
        found_idx = static_cast<int>(i);
        break;
      }
    }
    middle_index_to_search =
        (found_idx != -1)
            ? found_idx
            : static_cast<int>(prev_words_.size() * (1 - overlap_ratio_));
  } else {
    middle_index_to_search =
        static_cast<int>(prev_words_.size() * (1 - overlap_ratio_));
  }

  int min_distance = max_levenshtein_distance_;
  prev_word_index_of_unconfirmed_ = std::min<int>(
      middle_index_to_search + search_window_ + 1, prev_words_.size());

  for (int i = -search_window_; i <= search_window_; ++i) {
    int start_index =
        std::clamp<int>(middle_index_to_search + i, 0, prev_words_.size());
    int end_index =
        std::min<int>(start_index + curr_words.size(), prev_words_.size());
    if (start_index >= end_index) continue;

    auto prev_sub = absl::MakeSpan(prev_words_)
                        .subspan(start_index, end_index - start_index);
    auto curr_sub = absl::MakeSpan(curr_words)
                        .subspan(0, std::min<size_t>(end_index - start_index,
                                                     curr_words.size()));

    int dist = ComputeLevenshteinDistanceWords(CanonicalizeWords(prev_sub),
                                               CanonicalizeWords(curr_sub));
    if (dist < min_distance) {
      min_distance = dist;
      prev_word_index_of_unconfirmed_ = start_index;
    }
  }

  int unconfirmed_until_pivot = std::ceil(
      (prev_words_.size() - prev_word_index_of_unconfirmed_) * pivot_factor_);
  prev_word_index_of_pivot_ =
      std::min<int>(prev_word_index_of_unconfirmed_ + unconfirmed_until_pivot,
                    prev_words_.size());

  last_confirmed_words_ = std::vector<std::string>(
      prev_words_.begin(),
      prev_words_.begin() + prev_word_index_of_unconfirmed_);

  std::vector<std::string> unconfirmed_merged = MergeIntoUnconfirmedText(
      prev_words_, prev_timestamps_, curr_words, curr_timestamps,
      overlap_ratio_, pivot_factor_, prev_word_index_of_unconfirmed_,
      prev_word_index_of_pivot_);

  prev_timestamps_ = MergeTimestamps(
      prev_timestamps_, curr_timestamps, prev_word_index_of_unconfirmed_,
      prev_word_index_of_pivot_, unconfirmed_merged.size());
  prev_words_ = unconfirmed_merged;
  UpdateStreamTimestampOffset(max_chunk_timestamp_ms, curr_timestamps);

  LogAndPushOutput({absl::StrJoin(last_confirmed_words_, " "),
                    absl::StrJoin(prev_words_, " ")});
  return absl::OkStatus();
}

absl::Status TimestampTextMerger::Flush() {
  {
    absl::MutexLock lock(mutex_);
    if (state_ != State::kIdle) {
      return absl::FailedPreconditionError(
          "Flush() called while Schedule() is in progress.");
    }
    state_ = State::kRunning;
  }

  if (!prev_words_.empty()) {
    MergeResult result = {absl::StrJoin(prev_words_, " "), ""};
    prev_words_.clear();
    prev_timestamps_.clear();
    last_confirmed_words_.clear();
    prev_word_index_of_unconfirmed_ = -1;
    prev_word_index_of_pivot_ = -1;
    LogAndPushOutput(std::move(result));
  }

  SetState(State::kIdle);
  return absl::OkStatus();
}

void TimestampTextMerger::UpdateStreamTimestampOffset(
    std::optional<int> max_chunk_timestamp_ms,
    absl::Span<const int> curr_timestamps) {
  if (overlap_ratio_ <= 0.0f) return;

  int max_ts =
      max_chunk_timestamp_ms.has_value()
          ? max_chunk_timestamp_ms.value()
          : (curr_timestamps.empty()
                 ? 0
                 : curr_timestamps.back() - stream_timestamp_offset_ms_);
  int timestamp_step = static_cast<int>(max_ts * (1.0f - overlap_ratio_));
  if (timestamp_step > 0) {
    stream_timestamp_offset_ms_ += timestamp_step;
  }
}

void TimestampTextMerger::LogAndPushOutput(MergeResult output) {
  ABSL_VLOG(1) << "Merged text: confirmed=" << output.confirmed_text;
  ABSL_VLOG(1) << "Merged text: unconfirmed=" << output.unconfirmed_text;
  PushOutput(std::move(output));
}

}  // namespace litert::omni::asr
