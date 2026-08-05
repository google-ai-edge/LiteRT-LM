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
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/asr/detokenizer.h"
#include "omni/asr/text_merger.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {
namespace {

int ComputeLevenshteinDistance(absl::string_view s1, absl::string_view s2) {
  const size_t len1 = s1.size();
  const size_t len2 = s2.size();
  std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1, 0));

  for (size_t i = 0; i <= len1; ++i) dp[i][0] = static_cast<int>(i);
  for (size_t j = 0; j <= len2; ++j) dp[0][j] = static_cast<int>(j);

  for (size_t i = 1; i <= len1; ++i) {
    for (size_t j = 1; j <= len2; ++j) {
      int deletion_cost = dp[i - 1][j] + 1;
      int insertion_cost = dp[i][j - 1] + 1;
      int match_or_sub_cost =
          dp[i - 1][j - 1] + ((s1[i - 1] == s2[j - 1]) ? 0 : 1);
      dp[i][j] = std::min({deletion_cost, insertion_cost, match_or_sub_cost});
    }
  }
  return dp[len1][len2];
}

std::string Canonicalize(absl::Span<const std::string> words) {
  std::string joined = absl::AsciiStrToLower(absl::StrJoin(words, " "));
  std::string result;
  result.reserve(joined.size());
  for (char c : joined) {
    if (!std::ispunct(static_cast<unsigned char>(c))) {
      result.push_back(c);
    }
  }
  return result;
}

bool CloseEnoughStrings(absl::string_view s1, absl::string_view s2) {
  const size_t min_len = std::min(s1.size(), s2.size());
  const absl::string_view prefix1 = s1.substr(0, min_len);
  const absl::string_view prefix2 = s2.substr(0, min_len);
  if (min_len <= 3) {
    return prefix1 == prefix2;
  }
  return ComputeLevenshteinDistance(prefix1, prefix2) <= 1;
}

bool CloseEnough(absl::Span<const std::string> l1,
                 absl::Span<const std::string> l2) {
  if (l1.size() != l2.size()) return false;
  for (size_t i = 0; i < l1.size(); ++i) {
    if (!CloseEnoughStrings(l1[i], l2[i])) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> DedupWords(absl::Span<const std::string> prev_words,
                                    absl::Span<const std::string> curr_words,
                                    int search_window = 2) {
  const int max_search = std::min<int>(
      search_window, std::min(prev_words.size(), curr_words.size()));
  for (int i = max_search; i >= 1; --i) {
    auto prev_tail = prev_words.subspan(prev_words.size() - i, i);
    auto curr_head = curr_words.subspan(0, i);
    if (CloseEnough(prev_tail, curr_head)) {
      return std::vector<std::string>(curr_words.begin() + i, curr_words.end());
    }
  }
  return std::vector<std::string>(curr_words.begin(), curr_words.end());
}

std::vector<std::string> MergeIntoUnconfirmedText(
    absl::Span<const std::string> prev_words,
    absl::Span<const int> prev_timestamps,
    absl::Span<const std::string> curr_words,
    absl::Span<const int> curr_timestamps, float overlap_ratio,
    float pivot_factor, int prev_word_index_of_unconfirmed,
    int prev_word_index_of_pivot) {
  if (prev_word_index_of_unconfirmed == -1) {
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

void TimestampTextMerger::Reset() {
  WaitForStateThenSetState(State::kIdle, State::kRunning);
  prev_words_.clear();
  prev_timestamps_.clear();
  last_confirmed_words_.clear();
  prev_word_index_of_unconfirmed_ = -1;
  prev_word_index_of_pivot_ = -1;
  ClearOutputsThenSetState(State::kIdle);
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
    PushOutput(
        {"", prev_words_.empty() ? "" : absl::StrJoin(prev_words_, " ")});
    return absl::OkStatus();
  }

  std::vector<std::string> curr_words;
  std::vector<int> curr_timestamps;
  curr_words.reserve(curr_chunk_words.size());
  for (const auto& word : curr_chunk_words) {
    curr_words.push_back(word.text);
    if (word.timestamp_ms.has_value()) {
      curr_timestamps.push_back(word.timestamp_ms.value());
    }
  }

  if (overlap_ratio_ == 0.0f) {
    PushOutput({absl::StrJoin(curr_words, " "), ""});
    return absl::OkStatus();
  }

  if (prev_words_.empty()) {
    prev_words_ = curr_words;
    prev_timestamps_ = curr_timestamps;
    PushOutput({"", absl::StrJoin(prev_words_, " ")});
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

    int dist = ComputeLevenshteinDistance(Canonicalize(prev_sub),
                                          Canonicalize(curr_sub));
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

  prev_words_ = unconfirmed_merged;
  prev_timestamps_ = curr_timestamps;
  if (!prev_timestamps_.empty() && overlap_ratio_ > 0.0f) {
    int max_ts = prev_timestamps_.back();
    int timestamp_step = static_cast<int>(max_ts * (1.0f - overlap_ratio_));
    if (timestamp_step > 0) {
      for (int& ts : prev_timestamps_) {
        ts -= timestamp_step;
      }
    }
  }

  PushOutput({absl::StrJoin(last_confirmed_words_, " "),
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
    PushOutput(std::move(result));
  }

  SetState(State::kIdle);
  return absl::OkStatus();
}

}  // namespace litert::omni::asr
