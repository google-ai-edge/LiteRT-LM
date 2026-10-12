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
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/absl_vlog_is_on.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/asr/detokenizer.h"
#include "omni/asr/levenshtein_align.h"
#include "omni/asr/text_merger.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {
namespace {

// Alignment costs. A substitution (same acoustic event, different words) is
// cheaper than an insertion plus a deletion but costlier than a match plus an
// indel, so a matching word is never traded for two substitutions. Pairing
// words whose timestamps disagree costs more than an indel pair, so it never
// happens.
constexpr int kMatchCost = 0;
constexpr int kIndelCost = 2;
constexpr int kSubstitutionCost = 3;
constexpr int kFarPairCost = 5;

enum class AlignOp { kPair, kPrevOnly, kCurrOnly };

// Timestamp tolerances of a merge, in timestamp units, scaled from the length
// of the overlap. Emission times of the same word differ between the two
// windows: near the end of its window the previous chunk tends to emit early
// (up to ~40% of the overlap has been observed), the current chunk rarely
// emits earlier than the previous one did.
struct Tolerances {
  // Max |dt| for two different words to be the same acoustic event.
  double substitution;
  // Max (prev - curr) for close-enough words to be the same word.
  double match_early;
  // Max (curr - prev) for close-enough words to be the same word.
  double match_late;
};

struct TimedWord {
  std::string word;
  // Stream timestamp as reported by the decoder (stored for the next merge).
  int timestamp;
  // Timestamp in the previous chunk's time base.
  double time;
  bool from_prev;
};

bool HasUsableTimestamps(absl::Span<const std::string> words,
                         absl::Span<const int> timestamps) {
  return !words.empty() && timestamps.size() == words.size() &&
         absl::c_is_sorted(timestamps);
}

bool HasDistinctTimestamps(absl::Span<const int> timestamps) {
  return !timestamps.empty() && timestamps.back() > timestamps.front();
}

// Returns the shift that best maps the current chunk's timestamps onto the
// previous chunk's: the most frequent `curr - prev` difference among pairs of
// close-enough words within `max_shift`, with ties broken towards 0. At least
// two supporting pairs are required, otherwise 0 is returned. This absorbs a
// systematic emission delay between the two windows.
int EstimateTimestampShift(absl::Span<const std::string> prev_words,
                           absl::Span<const int> prev_timestamps,
                           absl::Span<const std::string> curr_words,
                           absl::Span<const int> curr_timestamps,
                           double max_shift) {
  std::map<int, int> support;
  for (size_t i = 0; i < prev_words.size(); ++i) {
    for (size_t j = 0; j < curr_words.size(); ++j) {
      const int shift = curr_timestamps[j] - prev_timestamps[i];
      if (std::abs(shift) > max_shift) continue;
      if (CloseEnoughStr(prev_words[i], curr_words[j])) ++support[shift];
    }
  }
  int best_shift = 0;
  int best_support = 1;
  for (const auto& [shift, count] : support) {
    if (count > best_support ||
        (count == best_support && std::abs(shift) < std::abs(best_shift))) {
      best_shift = shift;
      best_support = count;
    }
  }
  return best_shift;
}

int PairCost(absl::string_view prev_word, double prev_time,
             absl::string_view curr_word, double curr_time,
             const Tolerances& tol) {
  const double dt = curr_time - prev_time;
  if (CloseEnoughStr(prev_word, curr_word)) {
    return (dt >= -tol.match_early && dt <= tol.match_late) ? kMatchCost
                                                            : kFarPairCost;
  }
  return std::fabs(dt) <= tol.substitution ? kSubstitutionCost : kFarPairCost;
}

// Monotone alignment of the overlapping words of the two chunks: a Levenshtein
// alignment in which words may only be paired when their timestamps agree
// (see `Tolerances`). Words that are far apart in time are never paired.
std::vector<AlignOp> AlignOverlap(absl::Span<const std::string> prev_words,
                                  absl::Span<const int> prev_timestamps,
                                  absl::Span<const std::string> curr_words,
                                  absl::Span<const double> curr_times,
                                  const Tolerances& tol) {
  const size_t num_prev = prev_words.size();
  const size_t num_curr = curr_words.size();
  auto pair_cost = [&](size_t i, size_t j) {
    return PairCost(prev_words[i], prev_timestamps[i], curr_words[j],
                    curr_times[j], tol);
  };

  std::vector<std::vector<int>> cost(num_prev + 1,
                                     std::vector<int>(num_curr + 1, 0));
  for (size_t i = 0; i <= num_prev; ++i) cost[i][0] = i * kIndelCost;
  for (size_t j = 0; j <= num_curr; ++j) cost[0][j] = j * kIndelCost;
  for (size_t i = 1; i <= num_prev; ++i) {
    for (size_t j = 1; j <= num_curr; ++j) {
      cost[i][j] =
          std::min({cost[i - 1][j - 1] + pair_cost(i - 1, j - 1),
                    cost[i - 1][j] + kIndelCost, cost[i][j - 1] + kIndelCost});
    }
  }

  std::vector<AlignOp> ops;
  ops.reserve(num_prev + num_curr);
  size_t i = num_prev;
  size_t j = num_curr;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 &&
        cost[i][j] == cost[i - 1][j - 1] + pair_cost(i - 1, j - 1)) {
      ops.push_back(AlignOp::kPair);
      --i;
      --j;
      continue;
    }
    const bool prev_only_ok =
        i > 0 && cost[i][j] == cost[i - 1][j] + kIndelCost;
    const bool curr_only_ok =
        j > 0 && cost[i][j] == cost[i][j - 1] + kIndelCost;
    // When both are optimal, emit (walking backwards) the later word first so
    // that unmatched words of the two chunks interleave in time order.
    if (prev_only_ok &&
        (!curr_only_ok || prev_timestamps[i - 1] >= curr_times[j - 1])) {
      ops.push_back(AlignOp::kPrevOnly);
      --i;
    } else {
      ops.push_back(AlignOp::kCurrOnly);
      --j;
    }
  }
  std::reverse(ops.begin(), ops.end());
  return ops;
}

// Legacy helpers of the word-count based merge.
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
        static_cast<size_t>(
            std::ceil(num_words_overlap_in_current * pivot_factor)),
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
  stream_timestamp_offset_ms_ = 0.0;
  prev_chunk_end_ms_ = -1.0;
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

  const int chunk_start_ms =
      static_cast<int>(std::lround(stream_timestamp_offset_ms_));
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
      curr_timestamps.push_back(word.timestamp_ms.value() + chunk_start_ms);
    }
  }
  // End of the chunk relative to its start; the last word's timestamp when
  // the decoder does not report the end of the chunk.
  const int chunk_end_ms = max_chunk_timestamp_ms.value_or(
      curr_timestamps.empty() ? 0 : curr_timestamps.back() - chunk_start_ms);

  if (curr_words.empty()) {
    UpdateStreamTimestampOffset(chunk_end_ms);
    LogAndPushOutput(
        {"", prev_words_.empty() ? "" : absl::StrJoin(prev_words_, " ")});
    return absl::OkStatus();
  }

  if (overlap_ratio_ == 0.0f) {
    LogAndPushOutput({absl::StrJoin(curr_words, " "), ""});
    return absl::OkStatus();
  }

  if (prev_words_.empty()) {
    prev_words_ = curr_words;
    prev_timestamps_ = curr_timestamps;
    prev_chunk_end_ms_ = stream_timestamp_offset_ms_ + chunk_end_ms;
    UpdateStreamTimestampOffset(chunk_end_ms);
    LogAndPushOutput({"", absl::StrJoin(prev_words_, " ")});
    return absl::OkStatus();
  }

  // Chunks whose words all share one timestamp carry no timing information
  // unless the decoder reports the end of the chunk.
  const bool use_time = prev_chunk_end_ms_ >= 0.0 &&
                        HasUsableTimestamps(prev_words_, prev_timestamps_) &&
                        HasUsableTimestamps(curr_words, curr_timestamps) &&
                        (max_chunk_timestamp_ms.has_value() ||
                         HasDistinctTimestamps(prev_timestamps_) ||
                         HasDistinctTimestamps(curr_timestamps));
  if (use_time) {
    MergeByTime(curr_words, curr_timestamps);
  } else {
    MergeByWordCount(curr_words, curr_timestamps);
  }
  prev_chunk_end_ms_ = stream_timestamp_offset_ms_ + chunk_end_ms;
  UpdateStreamTimestampOffset(chunk_end_ms);

  LogAndPushOutput({absl::StrJoin(last_confirmed_words_, " "),
                    absl::StrJoin(prev_words_, " ")});
  return absl::OkStatus();
}

void TimestampTextMerger::MergeByTime(absl::Span<const std::string> curr_words,
                                      absl::Span<const int> curr_timestamps) {
  const double prev_end = prev_chunk_end_ms_;
  if (prev_end <= stream_timestamp_offset_ms_) {
    // The window that produced the previous words ends before this chunk
    // (e.g. an empty chunk in between): the previous words are final.
    last_confirmed_words_ = std::move(prev_words_);
    prev_words_.assign(curr_words.begin(), curr_words.end());
    prev_timestamps_.assign(curr_timestamps.begin(), curr_timestamps.end());
    return;
  }

  const double overlap_len = prev_end - stream_timestamp_offset_ms_;
  const Tolerances tol = {
      .substitution = std::max(1.0, 0.15 * overlap_len),
      .match_early = std::max(2.0, 0.25 * overlap_len),
      .match_late = std::max(2.0, 0.5 * overlap_len),
  };

  // Work in the previous chunk's time base.
  const int shift =
      EstimateTimestampShift(prev_words_, prev_timestamps_, curr_words,
                             curr_timestamps, tol.match_early);
  const double chunk_start = stream_timestamp_offset_ms_ - shift;
  const double pivot = chunk_start + pivot_factor_ * (prev_end - chunk_start);
  std::vector<double> curr_times;
  curr_times.reserve(curr_timestamps.size());
  for (int ts : curr_timestamps) curr_times.push_back(ts - shift);

  // Previous words before the current chunk are final.
  size_t num_confirmed = 0;
  while (num_confirmed < prev_timestamps_.size() &&
         prev_timestamps_[num_confirmed] < chunk_start) {
    ++num_confirmed;
  }
  last_confirmed_words_.assign(prev_words_.begin(),
                               prev_words_.begin() + num_confirmed);
  const auto prev_ov_words =
      absl::MakeConstSpan(prev_words_).subspan(num_confirmed);
  const auto prev_ov_ts =
      absl::MakeConstSpan(prev_timestamps_).subspan(num_confirmed);

  // Current words well beyond the previous window are new; the margin lets
  // the last words of the previous window, emitted early, find their pair.
  size_t num_overlap = 0;
  while (num_overlap < curr_times.size() &&
         curr_times[num_overlap] < prev_end + tol.match_late) {
    ++num_overlap;
  }

  auto curr_word_within = [&](double time, double tolerance) {
    for (double t : curr_times) {
      if (std::fabs(t - time) <= tolerance) return true;
    }
    return false;
  };
  const std::vector<AlignOp> ops = AlignOverlap(
      prev_ov_words, prev_ov_ts, curr_words.subspan(0, num_overlap),
      absl::MakeConstSpan(curr_times).subspan(0, num_overlap), tol);

  std::vector<bool> prev_ov_matched(prev_ov_words.size(), false);
  {
    size_t pi = 0;
    size_t cj = 0;
    for (AlignOp op : ops) {
      if (op == AlignOp::kPair) {
        if (CloseEnoughStr(prev_ov_words[pi], curr_words[cj])) {
          prev_ov_matched[pi] = true;
        }
        ++pi;
        ++cj;
      } else if (op == AlignOp::kPrevOnly) {
        ++pi;
      } else {
        ++cj;
      }
    }
  }

  // A current-only word before the pivot is kept only where the previous
  // chunk heard nothing: it is dropped when the previous chunk has an unpaired
  // or substituted word at about the same time, or when it starts the window
  // while the previous chunk was mid-speech at the boundary (the tail of a
  // word cut by the window start). Words in `prev` that are already matched to
  // their counterpart in `curr` only cover their own frame (`<= 1.0`), so short
  // function words between matched words are not dropped.
  auto covered_by_prev = [&](double time) {
    for (size_t idx = 0; idx < num_confirmed; ++idx) {
      if (std::fabs(prev_timestamps_[idx] - time) <= tol.substitution) {
        return true;
      }
    }
    for (size_t idx = 0; idx < prev_ov_ts.size(); ++idx) {
      const double limit = prev_ov_matched[idx] ? 1.0 : tol.substitution;
      if (std::fabs(prev_ov_ts[idx] - time) <= limit) return true;
    }
    return false;
  };
  auto cut_at_window_start = [&](double time) {
    if (time > chunk_start + tol.substitution) return false;
    for (int ts : prev_timestamps_) {
      if (ts >= chunk_start - tol.match_early &&
          ts <= time + tol.substitution &&
          (time - ts <= tol.match_early || time <= chunk_start + 1.0)) {
        return true;
      }
    }
    return false;
  };
  // A previous-only word after the pivot is a word the alignment could not
  // pair. It is dropped when it is implausible — part of a burst of words
  // emitted at one timestamp, a hallucination at the end of the window — or
  // when the current chunk heard something else at about the same time (the
  // current chunk wins after the pivot). It is kept where the current chunk
  // heard nothing, as the current chunk may have a gap.
  auto in_burst = [&](size_t index) {
    constexpr int kMinBurstWords = 3;
    int count = 0;
    for (int ts : prev_ov_ts) {
      if (ts == prev_ov_ts[index]) ++count;
    }
    return count >= kMinBurstWords;
  };
  auto drop_prev_tail_word = [&](size_t index) {
    const bool truncated_final_word =
        index + 1 == prev_ov_ts.size() && prev_ov_ts[index] >= prev_end - 4.0 &&
        !curr_times.empty() &&
        curr_times.front() + tol.substitution < prev_ov_ts[index] &&
        curr_times.back() > prev_ov_ts[index];
    return in_burst(index) || truncated_final_word ||
           curr_word_within(prev_ov_ts[index], tol.substitution);
  };

  std::vector<TimedWord> merged;
  merged.reserve(prev_ov_words.size() + curr_words.size());
  size_t i = 0;
  size_t j = 0;
  auto push_prev = [&]() {
    merged.push_back({prev_ov_words[i], prev_ov_ts[i],
                      static_cast<double>(prev_ov_ts[i]), /*from_prev=*/true});
  };
  auto push_curr = [&]() {
    merged.push_back({curr_words[j], curr_timestamps[j], curr_times[j],
                      /*from_prev=*/false});
  };
  for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
    const AlignOp op = ops[op_idx];
    switch (op) {
      case AlignOp::kPair: {
        // Before the pivot the previous chunk had more context, after it the
        // current one has. However, if a substitution before the pivot is
        // immediately followed by a current-only word at or after the pivot
        // (e.g., a compound word in `prev` split into two words in `curr`
        // straddling the pivot), keep `curr`'s first word so both halves come
        // from `curr` rather than splicing `prev`'s compound word with `curr`'s
        // second word.
        const bool is_sub = !CloseEnoughStr(prev_ov_words[i], curr_words[j]);
        const bool split_straddles_pivot =
            is_sub && op_idx + 1 < ops.size() &&
            ops[op_idx + 1] == AlignOp::kCurrOnly &&
            j + 1 < curr_times.size() && curr_times[j + 1] >= pivot;
        // If the final word of `prev` was truncated to a single consonant at
        // the end of `prev`'s window (e.g. "coyote b" vs "coyote bill" in
        // `curr`), prefer `curr`'s full word even if the onset was just before
        // `pivot`. Do not apply this in the middle of `prev_ov_words`, where
        // single-letter tokens before `pivot` may be spelled-out initialisms
        // (e.g. "d e p t i d").
        bool prev_is_fragment = false;
        if (is_sub && i + 1 == prev_ov_words.size()) {
          const std::string c_prev = CanonicalizeStr(prev_ov_words[i]);
          const std::string c_curr = CanonicalizeStr(curr_words[j]);
          prev_is_fragment = c_prev.size() == 1 && c_prev != "a" &&
                             c_prev != "i" && c_curr.size() >= 2 &&
                             c_curr.rfind(c_prev, 0) == 0;
        }
        if (!prev_is_fragment && prev_ov_ts[i] < pivot - 1e-4 &&
            !split_straddles_pivot) {
          push_prev();
        } else {
          push_curr();
        }
        ++i;
        ++j;
        break;
      }
      case AlignOp::kPrevOnly:
        if (prev_ov_ts[i] < pivot || !drop_prev_tail_word(i)) push_prev();
        ++i;
        break;
      case AlignOp::kCurrOnly:
        if (curr_times[j] >= pivot || (!covered_by_prev(curr_times[j]) &&
                                       !cut_at_window_start(curr_times[j]))) {
          push_curr();
        }
        ++j;
        break;
    }
  }
  for (; j < curr_words.size(); ++j) push_curr();

  ABSL_VLOG(2) << "MergeByTime: shift=" << shift
               << " chunk_start=" << chunk_start << " prev_end=" << prev_end
               << " pivot=" << pivot << " confirmed=" << num_confirmed
               << " overlap prev/curr=" << prev_ov_words.size() << "/"
               << num_overlap;

  // The same word heard by both chunks but not paired by the alignment would
  // be duplicated at the seam: keep the copy with more context. Also drop a
  // trailing compound/acronym suffix duplicated right after `prev`'s word
  // (e.g. `"two-stage"` followed by `"stage"`, or `"rt."` followed by `"t."`).
  prev_words_.clear();
  prev_timestamps_.clear();
  for (size_t k = 0; k < merged.size(); ++k) {
    const TimedWord& w = merged[k];
    if (k + 1 < merged.size()) {
      const TimedWord& next = merged[k + 1];
      if (w.from_prev != next.from_prev) {
        const TimedWord& prev_copy = w.from_prev ? w : next;
        const TimedWord& curr_copy = w.from_prev ? next : w;
        const double dt = curr_copy.time - prev_copy.time;
        if (dt >= -tol.match_early && dt <= tol.match_late) {
          if (CloseEnoughStr(w.word, next.word)) {
            const bool keep_prev_copy = prev_copy.time < pivot;
            if (keep_prev_copy != w.from_prev) continue;  // Skip this copy.
            ++k;                                          // Skip the next copy.
          } else if (w.from_prev && !next.from_prev) {
            const auto canon = CanonicalizeWords({w.word, next.word});
            if (canon.size() == 2) {
              const std::string& cw = canon[0];
              const std::string& cnext = canon[1];
              if (!cnext.empty() && cw.size() > cnext.size() &&
                  cw.compare(cw.size() - cnext.size(), cnext.size(), cnext) ==
                      0 &&
                  (absl::StrContains(w.word, '-') ||
                   (cw.size() <= 3 && cnext.size() == 1))) {
                ++k;  // Skip the duplicated suffix word from `curr`.
              }
            }
          }
        }
      }
    }
    prev_words_.push_back(w.word);
    prev_timestamps_.push_back(w.timestamp);
  }
  if (ABSL_VLOG_IS_ON(2)) {
    std::string words;
    for (size_t k = 0; k < prev_words_.size(); ++k) {
      absl::StrAppend(&words, k == 0 ? "" : " ", prev_words_[k], "(",
                      prev_timestamps_[k], ")");
    }
    ABSL_VLOG(2) << "MergeByTime: unconfirmed=" << words;
  }
}

void TimestampTextMerger::MergeByWordCount(
    absl::Span<const std::string> curr_words,
    absl::Span<const int> curr_timestamps) {
  const bool has_timestamps =
      !prev_timestamps_.empty() && !curr_timestamps.empty() &&
      (prev_timestamps_.back() > prev_timestamps_.front() ||
       curr_timestamps.back() > curr_timestamps.front());
  if (!has_timestamps && !prev_words_.empty() && !curr_words.empty()) {
    const size_t n_prev = prev_words_.size();
    const size_t n_curr = curr_words.size();
    const size_t min_prev =
        (n_prev > 8)
            ? std::max<size_t>(
                  1, static_cast<size_t>(std::max<int>(
                         0, static_cast<int>(n_prev * (1.0f - overlap_ratio_)) -
                                search_window_ - 2)))
            : 0;
    const size_t max_curr = std::min<size_t>(
        n_curr, static_cast<size_t>(std::ceil(n_curr * overlap_ratio_)) +
                    search_window_ + 4);
    auto align_codes =
        AlignTokens(absl::MakeConstSpan(prev_words_).subspan(min_prev),
                    curr_words.subspan(0, max_curr));
    std::vector<std::pair<size_t, size_t>> best_run;
    std::vector<std::pair<size_t, size_t>> cur_run;
    std::vector<std::pair<size_t, size_t>> all_matches;
    size_t r_idx = 0;
    size_t h_idx = 0;
    for (AlignCode code : align_codes) {
      if (code == AlignCode::kCorrect) {
        all_matches.emplace_back(min_prev + r_idx, h_idx);
        cur_run.emplace_back(min_prev + r_idx, h_idx);
        if (cur_run.size() > best_run.size() ||
            (cur_run.size() == best_run.size() && cur_run.size() >= 2)) {
          best_run = cur_run;
        }
      } else {
        cur_run.clear();
      }
      if (code == AlignCode::kDeletion || code == AlignCode::kSubstitution ||
          code == AlignCode::kCorrect) {
        ++r_idx;
      }
      if (code == AlignCode::kInsertion || code == AlignCode::kSubstitution ||
          code == AlignCode::kCorrect) {
        ++h_idx;
      }
    }
    if (best_run.size() < 2) {
      for (size_t r = min_prev; r < n_prev; ++r) {
        for (size_t h = 0; h < max_curr; ++h) {
          size_t len = 0;
          while (r + len < n_prev && h + len < max_curr &&
                 CloseEnoughStr(prev_words_[r + len], curr_words[h + len])) {
            ++len;
          }
          if (len >= 2 && (h <= 3 || r + len + 2 >= n_prev) &&
              len >= best_run.size()) {
            best_run.clear();
            for (size_t k = 0; k < len; ++k) {
              best_run.emplace_back(r + k, h + k);
            }
          }
        }
      }
      if (best_run.size() < 2 && all_matches.size() >= 2) {
        for (size_t m = 0; m + 1 < all_matches.size(); ++m) {
          const auto [p0, c0] = all_matches[m];
          const auto [p1, c1] = all_matches[m + 1];
          if (p1 > p0 && c1 > c0 && (p1 - p0) + (c1 - c0) <= 5 &&
              (c0 <= 2 || p1 + 2 >= n_prev) &&
              std::max(CanonicalizeStr(prev_words_[p0]).size(),
                       CanonicalizeStr(prev_words_[p1]).size()) >= 6) {
            best_run = {all_matches[m], all_matches[m + 1]};
          }
        }
      }
    }
    const bool valid_single_match =
        best_run.size() == 1 &&
        ((best_run[0].first + 1 == n_prev && best_run[0].second == 0) ||
         (best_run[0].first + 3 >= n_prev && best_run[0].second <= 1 &&
          CanonicalizeStr(curr_words[best_run[0].second]).size() >= 5));
    if (best_run.size() >= 2 || valid_single_match) {
      const auto [p0, c0] = best_run.front();
      const auto [p_piv, c_piv] = best_run[best_run.size() / 2];
      prev_word_index_of_unconfirmed_ = std::clamp<int>(
          static_cast<int>(p0) - static_cast<int>(c0),
          static_cast<int>(min_prev), static_cast<int>(p0));
      prev_word_index_of_pivot_ = static_cast<int>(p_piv);
      last_confirmed_words_.assign(
          prev_words_.begin(),
          prev_words_.begin() + prev_word_index_of_unconfirmed_);
      std::vector<std::string> unconfirmed_merged(
          prev_words_.begin() + prev_word_index_of_unconfirmed_,
          prev_words_.begin() + prev_word_index_of_pivot_);
      if (n_curr - c_piv >= n_prev - p_piv) {
        unconfirmed_merged.insert(unconfirmed_merged.end(),
                                  curr_words.begin() + c_piv, curr_words.end());
      } else {
        unconfirmed_merged.insert(unconfirmed_merged.end(),
                                  prev_words_.begin() + p_piv,
                                  prev_words_.end());
      }
      prev_timestamps_ = MergeTimestamps(
          prev_timestamps_, curr_timestamps, prev_word_index_of_unconfirmed_,
          prev_word_index_of_pivot_, unconfirmed_merged.size());
      prev_words_ = std::move(unconfirmed_merged);
      return;
    }
  }

  int middle_index_to_search = 0;
  if (has_timestamps) {
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

  const int min_start_index = (prev_words_.size() > 2) ? 1 : 0;
  int min_distance = max_levenshtein_distance_;
  prev_word_index_of_unconfirmed_ =
      has_timestamps
          ? std::min<int>(middle_index_to_search + search_window_ + 1,
                          prev_words_.size())
          : static_cast<int>(prev_words_.size());

  for (int i = -search_window_; i <= search_window_; ++i) {
    int start_index = std::clamp<int>(middle_index_to_search + i,
                                      min_start_index, prev_words_.size());
    int end_index =
        std::min<int>(start_index + curr_words.size(), prev_words_.size());
    if (start_index >= end_index) continue;

    auto prev_sub = absl::MakeSpan(prev_words_)
                        .subspan(start_index, end_index - start_index);
    auto curr_sub = curr_words.subspan(
        0, std::min<size_t>(end_index - start_index, curr_words.size()));

    int dist = ComputeLevenshteinDistanceWords(CanonicalizeWords(prev_sub),
                                               CanonicalizeWords(curr_sub));
    if (dist < min_distance &&
        (has_timestamps || dist * 2 <= static_cast<int>(prev_sub.size()))) {
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
  prev_words_ = std::move(unconfirmed_merged);
}

absl::Status TimestampTextMerger::FlushInternal() {
  if (!prev_words_.empty()) {
    MergeResult result = {absl::StrJoin(prev_words_, " "), ""};
    prev_words_.clear();
    prev_timestamps_.clear();
    last_confirmed_words_.clear();
    prev_word_index_of_unconfirmed_ = -1;
    prev_word_index_of_pivot_ = -1;
    prev_chunk_end_ms_ = -1.0;
    LogAndPushOutput(std::move(result));
  }
  return absl::OkStatus();
}

void TimestampTextMerger::UpdateStreamTimestampOffset(int chunk_end_ms) {
  if (overlap_ratio_ <= 0.0f) return;
  // Kept fractional: with encoder-frame timestamps the step is not a whole
  // number of frames, and truncating it drifts by a frame every few chunks.
  if (chunk_end_ms > 0) {
    stream_timestamp_offset_ms_ += chunk_end_ms * (1.0 - overlap_ratio_);
  }
}

void TimestampTextMerger::LogAndPushOutput(MergeResult output) {
  ABSL_VLOG(1) << "Merged text: confirmed=" << output.confirmed_text;
  ABSL_VLOG(1) << "Merged text: unconfirmed=" << output.unconfirmed_text;
  PushOutput(std::move(output));
}

}  // namespace litert::omni::asr
