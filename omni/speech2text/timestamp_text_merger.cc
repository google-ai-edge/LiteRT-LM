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

#include "omni/speech2text/timestamp_text_merger.h"

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
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/absl_vlog_is_on.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/stage.h"
#include "omni/speech2text/detokenizer.h"
#include "omni/speech2text/levenshtein_align.h"
#include "omni/speech2text/text_merger.h"
#include "omni/utils/text_utils.h"

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

bool HasUsableTimestamps(absl::Span<const std::string> words,
                         absl::Span<const int> timestamps) {
  return !words.empty() && timestamps.size() == words.size() &&
         absl::c_is_sorted(timestamps);
}

bool HasDistinctTimestamps(absl::Span<const int> timestamps) {
  return !timestamps.empty() && timestamps.back() > timestamps.front();
}

std::vector<std::vector<bool>> BuildCloseEnoughMatrix(
    absl::Span<const std::string> canon_prev,
    absl::Span<const std::string> canon_curr) {
  std::vector<std::vector<bool>> matrix(
      canon_prev.size(), std::vector<bool>(canon_curr.size(), false));
  for (size_t i = 0; i < canon_prev.size(); ++i) {
    for (size_t j = 0; j < canon_curr.size(); ++j) {
      matrix[i][j] = CloseEnoughCanonical(canon_prev[i], canon_curr[j]);
    }
  }
  return matrix;
}

// Returns the shift that best maps the current chunk's timestamps onto the
// previous chunk's: the most frequent `curr - prev` difference among pairs of
// close-enough words (>= 2 canonical chars) in the overlap region within
// `max_shift`, with ties broken towards 0. At least two supporting pairs are
// required, otherwise 0 is returned. This absorbs a systematic emission delay
// between the two windows.
int EstimateTimestampShift(absl::Span<const std::string> canon_prev,
                           absl::Span<const int> prev_timestamps,
                           absl::Span<const std::string> canon_curr,
                           absl::Span<const int> curr_timestamps,
                           double stream_offset_ms, double prev_end_ms,
                           double max_shift) {
  std::map<int, int> support;
  const double min_prev_ts = stream_offset_ms - 2.0 * max_shift;
  const double max_curr_ts = prev_end_ms + 2.0 * max_shift;
  for (size_t i = 0; i < canon_prev.size(); ++i) {
    if (canon_prev[i].size() < 2 || prev_timestamps[i] < min_prev_ts) continue;
    for (size_t j = 0; j < canon_curr.size(); ++j) {
      if (canon_curr[j].size() < 2 || curr_timestamps[j] > max_curr_ts) {
        continue;
      }
      const int shift = curr_timestamps[j] - prev_timestamps[i];
      if (std::abs(shift) > max_shift) continue;
      if (CloseEnoughCanonical(canon_prev[i], canon_curr[j])) ++support[shift];
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

int PairCost(bool is_close_enough, double prev_time, double curr_time,
             const Tolerances& tol) {
  const double dt = curr_time - prev_time;
  if (is_close_enough) {
    return (dt >= -tol.match_early && dt <= tol.match_late) ? kMatchCost
                                                            : kFarPairCost;
  }
  return std::fabs(dt) <= tol.substitution ? kSubstitutionCost : kFarPairCost;
}

// Monotone alignment of the overlapping words of the two chunks: a Levenshtein
// alignment in which words may only be paired when their timestamps agree
// (see `Tolerances`). Words that are far apart in time are never paired.
std::vector<AlignOp> AlignOverlap(
    const std::vector<std::vector<bool>>& close_matrix,
    absl::Span<const int> prev_timestamps, absl::Span<const double> curr_times,
    const Tolerances& tol) {
  const size_t num_prev = prev_timestamps.size();
  const size_t num_curr = curr_times.size();

  std::vector<std::vector<int>> cost(num_prev + 1,
                                     std::vector<int>(num_curr + 1, 0));
  for (size_t i = 0; i <= num_prev; ++i) cost[i][0] = i * kIndelCost;
  for (size_t j = 0; j <= num_curr; ++j) cost[0][j] = j * kIndelCost;
  for (size_t i = 1; i <= num_prev; ++i) {
    for (size_t j = 1; j <= num_curr; ++j) {
      const int pair_cost =
          PairCost(close_matrix[i - 1][j - 1], prev_timestamps[i - 1],
                   curr_times[j - 1], tol);
      cost[i][j] =
          std::min({cost[i - 1][j - 1] + pair_cost, cost[i - 1][j] + kIndelCost,
                    cost[i][j - 1] + kIndelCost});
    }
  }

  std::vector<AlignOp> ops;
  ops.reserve(num_prev + num_curr);
  size_t i = num_prev;
  size_t j = num_curr;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0) {
      const int pair_cost =
          PairCost(close_matrix[i - 1][j - 1], prev_timestamps[i - 1],
                   curr_times[j - 1], tol);
      if (cost[i][j] == cost[i - 1][j - 1] + pair_cost) {
        ops.push_back(AlignOp::kPair);
        --i;
        --j;
        continue;
      }
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

bool HasWordWithin(absl::Span<const double> times, double time,
                   double tolerance) {
  for (double t : times) {
    if (std::fabs(t - time) <= tolerance) return true;
  }
  return false;
}

// Marks which words in `prev_ov_words` were paired with a close-enough word in
// `curr_words` by `ops`.
std::vector<bool> FindMatchedPrevOverlapWords(
    const std::vector<std::vector<bool>>& close_matrix,
    absl::Span<const AlignOp> ops) {
  std::vector<bool> prev_ov_matched(close_matrix.size(), false);
  size_t pi = 0;
  size_t cj = 0;
  for (AlignOp op : ops) {
    if (op == AlignOp::kPair) {
      if (close_matrix[pi][cj]) {
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
  return prev_ov_matched;
}

// A current-only word before the pivot is kept only where the previous chunk
// heard nothing: it is dropped when the previous chunk has an unpaired or
// substituted overlap word at about the same time. Words in `prev` that are
// already matched to their counterpart in `curr` only cover their own frame
// (`<= 1.0`), so short function words between matched words are not dropped.
bool IsCoveredByPrev(absl::Span<const int> prev_ov_ts,
                     const std::vector<bool>& prev_ov_matched, double time,
                     const Tolerances& tol) {
  for (size_t idx = 0; idx < prev_ov_ts.size(); ++idx) {
    const double limit = prev_ov_matched[idx] ? 1.0 : tol.substitution;
    if (std::fabs(prev_ov_ts[idx] - time) <= limit) return true;
  }
  return false;
}

// Returns true if `canon_curr` matches the spoken number or phonetic suffix of
// a numeric `canon_prev` token (e.g. "five" after "105" or "twenty" after
// "20").
bool MatchesSpokenNumberSuffix(absl::string_view canon_prev,
                               absl::string_view canon_curr) {
  if (canon_prev.empty() || canon_prev.back() < '0' ||
      canon_prev.back() > '9') {
    return false;
  }

  static const absl::NoDestructor<
      absl::flat_hash_map<std::string, std::vector<std::string>>>
      kSpokenNumberMap([] {
        constexpr absl::string_view kOnes[] = {"zero",  "one",  "two", "three",
                                               "four",  "five", "six", "seven",
                                               "eight", "nine"};
        constexpr absl::string_view kTeens[] = {
            "ten",     "eleven",  "twelve",    "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
        constexpr absl::string_view kTens[] = {
            "",      "",      "twenty",  "thirty", "forty",
            "fifty", "sixty", "seventy", "eighty", "ninety"};

        absl::flat_hash_map<std::string, std::vector<std::string>> map = {
            {"0", {"zero", "oh"}},
            {"00", {"hundred"}},
            {"000", {"thousand"}},
        };
        for (int d = 1; d <= 9; ++d) {
          map[absl::StrCat(d)] = {std::string(kOnes[d])};
          map[absl::StrCat("0", d)] = {absl::StrCat("oh", kOnes[d]),
                                       std::string(kOnes[d])};
        }
        for (int t = 0; t <= 9; ++t) {
          map[absl::StrCat(10 + t)] = {std::string(kTeens[t])};
        }
        for (int tens = 2; tens <= 9; ++tens) {
          map[absl::StrCat(tens, 0)] = {std::string(kTens[tens])};
          for (int ones = 1; ones <= 9; ++ones) {
            map[absl::StrCat(tens, ones)] = {
                absl::StrCat(kTens[tens], kOnes[ones]),
                std::string(kTens[tens])};
          }
        }
        return map;
      }());

  for (size_t suffix_len = 1;
       suffix_len <= std::min<size_t>(3, canon_prev.size()); ++suffix_len) {
    const absl::string_view suffix =
        canon_prev.substr(canon_prev.size() - suffix_len);
    const auto it = kSpokenNumberMap->find(suffix);
    if (it == kSpokenNumberMap->end()) continue;
    for (const std::string& cand : it->second) {
      if (canon_curr == cand || CloseEnoughCanonical(cand, canon_curr) ||
          absl::EndsWith(cand, canon_curr)) {
        return true;
      }
    }
  }
  return false;
}

// Returns true if `canon_curr` at the start of the current chunk looks like a
// truncated fragment, phonetic variant, or repeat of `canon_prev` that started
// just before the window boundary (e.g. a 1-2 character fragment, a suffix or
// close-enough match, a spoken number suffix matching a numeric token, an
// overlap where a prefix of `canon_curr` of length >= 3 matches a suffix of
// `canon_prev`, or both words having length >= 4 with matching 2-character
// suffix).
bool IsPartialOrMatchingBoundaryWord(absl::string_view canon_prev,
                                     absl::string_view canon_curr) {
  if (canon_curr.size() <= 2) return true;
  if (absl::EndsWith(canon_prev, canon_curr) ||
      CloseEnoughCanonical(canon_prev, canon_curr)) {
    return true;
  }
  if (MatchesSpokenNumberSuffix(canon_prev, canon_curr)) {
    return true;
  }
  for (size_t len = std::min(canon_prev.size(), canon_curr.size()); len >= 3;
       --len) {
    if (absl::EndsWith(canon_prev, canon_curr.substr(0, len))) {
      return true;
    }
  }
  return canon_prev.size() >= 4 && canon_curr.size() >= 4 &&
         canon_prev.substr(canon_prev.size() - 2) ==
             canon_curr.substr(canon_curr.size() - 2);
}

// Returns true when a current-only word at `time` starts the window while the
// previous chunk was mid-speech at the boundary (the tail of a word that
// started before `chunk_start` and was cut by the window start). Only previous
// words that started before `chunk_start` can cut the current word at the
// window start, and only when `canon_curr_word` looks like a partial
// suffix/fragment or repeat of that word, so genuinely new words missed by the
// previous chunk are preserved.
bool IsCutAtWindowStart(absl::Span<const std::string> canon_prev_words,
                        absl::Span<const int> prev_timestamps,
                        absl::string_view canon_curr_word, double chunk_start,
                        double time, const Tolerances& tol) {
  if (time > chunk_start + tol.substitution) return false;
  for (size_t idx = 0; idx < prev_timestamps.size(); ++idx) {
    const int ts = prev_timestamps[idx];
    if (ts >= chunk_start - tol.match_early &&
        ts < chunk_start + tol.substitution && ts < time &&
        IsPartialOrMatchingBoundaryWord(canon_prev_words[idx],
                                        canon_curr_word)) {
      return true;
    }
  }
  return false;
}

// Appends `word` and a monotone (non-decreasing) timestamp to `merged_words`
// and `merged_timestamps`, skipping `word` when it duplicates the trailing
// component of an immediately preceding hyphenated or apostrophe-joined word
// (e.g. "five" after "twenty-five" or "s" after "it's") within
// `max_suffix_gap`.
void AppendMergedWord(std::vector<std::string>& merged_words,
                      std::vector<int>& merged_timestamps,
                      absl::string_view word, int timestamp_ms,
                      double max_suffix_gap = -1.0) {
  if (!merged_words.empty() && merged_words.back().size() > word.size() &&
      absl::EndsWithIgnoreCase(merged_words.back(), word)) {
    const char delim =
        merged_words.back()[merged_words.back().size() - word.size() - 1];
    if (delim == '-' || delim == '\'') {
      if (max_suffix_gap < 0 ||
          std::abs(timestamp_ms - merged_timestamps.back()) <= max_suffix_gap) {
        return;
      }
    }
  }
  const int monotone_ts =
      merged_timestamps.empty()
          ? timestamp_ms
          : std::max(timestamp_ms, merged_timestamps.back());
  merged_words.emplace_back(word);
  merged_timestamps.push_back(monotone_ts);
}

// Walks the overlap alignment `ops` around `pivot` and appends the surviving
// unconfirmed words and non-decreasing timestamps into `merged_words` and
// `merged_timestamps`:
// - Paired words (`kPair`) come from `prev` before `pivot` and `curr` after it.
// - Unpaired `prev` words (`kPrevOnly`) are kept before `pivot`, or after
//   `pivot` when `curr` heard nothing at about the same time.
// - Unpaired `curr` words (`kCurrOnly`) are kept after `pivot`, or before
//   `pivot` when `prev` neither covered that time nor cut a word at the window
//   start.
void MergeAlignedOverlap(absl::Span<const std::string> canon_prev_words,
                         absl::Span<const int> prev_timestamps,
                         absl::Span<const std::string> prev_ov_words,
                         absl::Span<const int> prev_ov_ts,
                         absl::Span<const std::string> curr_words,
                         absl::Span<const std::string> canon_curr_words,
                         absl::Span<const int> curr_timestamps,
                         absl::Span<const double> curr_times,
                         const std::vector<std::vector<bool>>& close_matrix,
                         absl::Span<const AlignOp> ops, double chunk_start,
                         double pivot, double prev_end_ms,
                         const Tolerances& tol,
                         std::vector<std::string>& merged_words,
                         std::vector<int>& merged_timestamps) {
  const std::vector<bool> prev_ov_matched =
      FindMatchedPrevOverlapWords(close_matrix, ops);

  merged_words.reserve(prev_ov_words.size() + curr_words.size());
  merged_timestamps.reserve(prev_ov_words.size() + curr_words.size());
  size_t i = 0;
  size_t j = 0;
  for (AlignOp op : ops) {
    switch (op) {
      case AlignOp::kPair: {
        // Before the pivot the previous chunk had more context, after it the
        // current one has.
        const bool prefer_prev_numeric =
            !close_matrix[i][j] && prev_ov_ts[i] <= pivot + tol.substitution &&
            !prev_ov_words[i].empty() && prev_ov_words[i].front() >= '0' &&
            prev_ov_words[i].front() <= '9';
        if (prev_ov_ts[i] < pivot - 1e-4 || prefer_prev_numeric) {
          AppendMergedWord(merged_words, merged_timestamps, prev_ov_words[i],
                           prev_ov_ts[i], tol.match_late);
        } else {
          AppendMergedWord(merged_words, merged_timestamps, curr_words[j],
                           curr_timestamps[j], tol.match_late);
        }
        ++i;
        ++j;
        break;
      }
      case AlignOp::kPrevOnly: {
        const bool is_unconfirmed_prev_tail =
            i + 1 == prev_ov_words.size() &&
            prev_ov_ts[i] >= prev_end_ms - tol.substitution &&
            prev_ov_words[i].size() <= 3 && !curr_times.empty();
        if (prev_ov_ts[i] < pivot ||
            (!is_unconfirmed_prev_tail &&
             !HasWordWithin(curr_times, prev_ov_ts[i], tol.substitution))) {
          AppendMergedWord(merged_words, merged_timestamps, prev_ov_words[i],
                           prev_ov_ts[i], tol.match_late);
        }
        ++i;
        break;
      }
      case AlignOp::kCurrOnly:
        if (curr_times[j] >= pivot) {
          AppendMergedWord(merged_words, merged_timestamps, curr_words[j],
                           curr_timestamps[j], tol.match_late);
        } else if (!IsCoveredByPrev(prev_ov_ts, prev_ov_matched, curr_times[j],
                                    tol) &&
                   !IsCutAtWindowStart(canon_prev_words, prev_timestamps,
                                       canon_curr_words[j], chunk_start,
                                       curr_times[j], tol)) {
          AppendMergedWord(merged_words, merged_timestamps, curr_words[j],
                           static_cast<int>(std::lround(curr_times[j])),
                           tol.match_late);
        }
        ++j;
        break;
    }
  }
  for (; j < curr_words.size(); ++j) {
    AppendMergedWord(merged_words, merged_timestamps, curr_words[j],
                     curr_timestamps[j], tol.match_late);
  }
}

}  // namespace

TimestampTextMerger::TimestampTextMerger(
    Stage<std::vector<Detokenizer::Word>>* absl_nonnull detokenizer,
    float overlap_ratio, int search_window, float pivot_factor)
    : TextMerger(detokenizer),
      overlap_ratio_(overlap_ratio),
      search_window_(search_window),
      pivot_factor_(pivot_factor) {}

void TimestampTextMerger::ResetInternal() {
  prev_words_.clear();
  prev_timestamps_.clear();
  last_confirmed_words_.clear();
  stream_timestamp_offset_ms_ = 0.0;
  prev_chunk_end_ms_ = -1.0;
  last_chunk_end_ms_ = 0;
  prev_is_warm_start_ = false;
  consecutive_empty_chunks_ = 0;
  confirmed_prefix_size_ = 0;
  confirmed_until_ms_ = -1;
}

absl::Status TimestampTextMerger::ScheduleInternal() {
  SetState(State::kRunning);
  auto status = Execute();
  SetState(State::kIdle);
  return status;
}

absl::Status TimestampTextMerger::Execute() {
  LITERT_ASSIGN_OR_RETURN(auto curr_chunk_words, detokenizer_.GetOutput());

  const int chunk_start_ms =
      static_cast<int>(std::lround(stream_timestamp_offset_ms_));
  std::vector<std::string> curr_words;
  std::vector<int> curr_timestamps;
  std::optional<int> max_chunk_timestamp_ms;
  bool is_warm_start_chunk = false;
  // Whether a word had to be split into several units below.
  bool split_word = false;
  curr_words.reserve(curr_chunk_words.size());
  for (const auto& word : curr_chunk_words) {
    if (word.IsEndOfChunk()) {
      is_warm_start_chunk = word.is_warm_start;
      if (word.timestamp_ms.has_value()) {
        max_chunk_timestamp_ms = word.timestamp_ms.value();
      }
      continue;
    }
    // Align CJK text character by character (see `SplitCjkCharacters`).
    // `TokenizerDetokenizer` already emits one word per CJK character; this
    // only splits words from detokenizers that do not.
    std::vector<std::string> units = SplitCjkCharacters({word.text});
    split_word |= units.size() > 1;
    for (std::string& unit : units) {
      if (word.timestamp_ms.has_value()) {
        const int abs_ts = word.timestamp_ms.value() + chunk_start_ms;
        if (confirmed_until_ms_ >= 0 && abs_ts <= confirmed_until_ms_) {
          continue;
        }
        curr_timestamps.push_back(abs_ts);
      }
      curr_words.push_back(std::move(unit));
    }
  }
  // End of the chunk relative to its start; falls back to the last word's
  // timestamp (or the previous chunk's duration for a silent chunk) when the
  // decoder does not report the end of the chunk.
  const int chunk_end_ms = max_chunk_timestamp_ms.value_or(
      curr_timestamps.empty() ? last_chunk_end_ms_
                              : curr_timestamps.back() - chunk_start_ms);
  if (chunk_end_ms > 0) {
    last_chunk_end_ms_ = chunk_end_ms;
  }
  // A split word's timestamp is the onset of its first unit only; using it for
  // every unit would confirm the whole word as soon as its onset falls before
  // the next chunk, so such chunks are merged by word count instead.
  if (split_word) curr_timestamps.clear();

  if (curr_words.empty()) {
    std::string confirmed;
    if (!is_warm_start_chunk) {
      ++consecutive_empty_chunks_;
      const bool is_gated_followup_silence =
          !max_chunk_timestamp_ms.has_value() && last_chunk_end_ms_ > 0 &&
          prev_chunk_end_ms_ > static_cast<double>(last_chunk_end_ms_);
      if (!prev_words_.empty() && !prev_is_warm_start_ &&
          ((prev_chunk_end_ms_ >= 0.0 &&
            prev_chunk_end_ms_ <= stream_timestamp_offset_ms_) ||
           consecutive_empty_chunks_ >= 2 || is_gated_followup_silence)) {
        if (!prev_timestamps_.empty()) {
          confirmed_until_ms_ =
              prev_timestamps_.back() +
              std::max(1, static_cast<int>(0.08 * last_chunk_end_ms_));
        }
        const size_t prefix =
            std::min(confirmed_prefix_size_, prev_words_.size());
        confirmed = JoinWithCjkAwareSpacing(
            absl::MakeConstSpan(prev_words_).subspan(prefix));
        prev_words_.clear();
        prev_timestamps_.clear();
        confirmed_prefix_size_ = 0;
      }
      UpdateStreamTimestampOffset(chunk_end_ms);
    }
    const size_t prefix = std::min(confirmed_prefix_size_, prev_words_.size());
    LogAndPushOutput(
        {std::move(confirmed),
         prefix >= prev_words_.size()
             ? ""
             : JoinWithCjkAwareSpacing(
                   absl::MakeConstSpan(prev_words_).subspan(prefix))});
    return absl::OkStatus();
  }
  consecutive_empty_chunks_ = 0;

  if (overlap_ratio_ == 0.0f) {
    LogAndPushOutput({JoinWithCjkAwareSpacing(curr_words), ""});
    return absl::OkStatus();
  }

  if (prev_words_.empty() || prev_is_warm_start_) {
    prev_words_ = curr_words;
    prev_timestamps_ = curr_timestamps;
    prev_is_warm_start_ = is_warm_start_chunk;
    confirmed_prefix_size_ = 0;
    if (!is_warm_start_chunk) {
      prev_chunk_end_ms_ = stream_timestamp_offset_ms_ + chunk_end_ms;
      UpdateStreamTimestampOffset(chunk_end_ms);
    }
    LogAndPushOutput({"", JoinWithCjkAwareSpacing(prev_words_)});
    return absl::OkStatus();
  }

  // Chunks whose words all share one timestamp carry no timing information
  // unless the decoder reports the end of the chunk.
  const bool curr_has_usable_ts =
      HasUsableTimestamps(curr_words, curr_timestamps);
  const bool use_time = prev_chunk_end_ms_ >= 0.0 &&
                        HasUsableTimestamps(prev_words_, prev_timestamps_) &&
                        curr_has_usable_ts &&
                        (max_chunk_timestamp_ms.has_value() ||
                         HasDistinctTimestamps(prev_timestamps_) ||
                         HasDistinctTimestamps(curr_timestamps));
  if (use_time) {
    MergeByTime(curr_words, curr_timestamps);
  } else {
    if (curr_has_usable_ts) {
      ABSL_LOG_FIRST_N(INFO, 5)
          << "TimestampTextMerger: falling back to MergeByWordCount "
          << "(prev_words=" << prev_words_.size()
          << ", prev_timestamps=" << prev_timestamps_.size() << ")";
    }
    MergeByWordCount(curr_words, curr_timestamps);
  }
  prev_chunk_end_ms_ = stream_timestamp_offset_ms_ + chunk_end_ms;
  UpdateStreamTimestampOffset(chunk_end_ms);

  const size_t prefix = std::min(confirmed_prefix_size_, prev_words_.size());
  LogAndPushOutput(
      {JoinWithCjkAwareSpacing(last_confirmed_words_),
       prefix >= prev_words_.size()
           ? ""
           : JoinWithCjkAwareSpacing(
                 absl::MakeConstSpan(prev_words_).subspan(prefix))});
  return absl::OkStatus();
}

void TimestampTextMerger::MergeByTime(absl::Span<const std::string> curr_words,
                                      absl::Span<const int> curr_timestamps) {
  if (prev_chunk_end_ms_ <= stream_timestamp_offset_ms_) {
    // The window that produced the previous words ends before this chunk
    // (e.g. an empty chunk in between): the previous words are final.
    const size_t already_confirmed =
        std::min(confirmed_prefix_size_, prev_words_.size());
    last_confirmed_words_.assign(prev_words_.begin() + already_confirmed,
                                 prev_words_.end());
    confirmed_prefix_size_ = 0;
    prev_words_.assign(curr_words.begin(), curr_words.end());
    prev_timestamps_.assign(curr_timestamps.begin(), curr_timestamps.end());
    return;
  }

  const double overlap_len = prev_chunk_end_ms_ - stream_timestamp_offset_ms_;
  const double tol_scale =
      last_chunk_end_ms_ > 0
          ? std::min(overlap_len, 0.5 * static_cast<double>(last_chunk_end_ms_))
          : overlap_len;
  const Tolerances tol = {
      .substitution = std::max(1.0, 0.15 * tol_scale),
      .match_early = std::max(2.0, 0.25 * tol_scale),
      .match_late = std::max(2.0, 0.5 * tol_scale),
  };

  const std::vector<std::string> canon_prev_words =
      CanonicalizeWords(prev_words_);
  const std::vector<std::string> canon_curr_words =
      CanonicalizeWords(curr_words);

  // Express the current chunk's word timestamps and both overlap window bounds
  // in the previous chunk's word clock by subtracting `shift` (`curr - prev`).
  const int shift = EstimateTimestampShift(
      canon_prev_words, prev_timestamps_, canon_curr_words, curr_timestamps,
      stream_timestamp_offset_ms_, prev_chunk_end_ms_, tol.match_early);
  const double chunk_start = stream_timestamp_offset_ms_ - shift;
  const double prev_end = prev_chunk_end_ms_ - shift;
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
  const size_t already_confirmed =
      std::min(confirmed_prefix_size_, prev_words_.size());
  num_confirmed = std::max(num_confirmed, already_confirmed);
  last_confirmed_words_.assign(prev_words_.begin() + already_confirmed,
                               prev_words_.begin() + num_confirmed);
  confirmed_prefix_size_ = 0;
  const auto prev_ov_words =
      absl::MakeConstSpan(prev_words_).subspan(num_confirmed);
  const auto canon_prev_ov_words =
      absl::MakeConstSpan(canon_prev_words).subspan(num_confirmed);
  const auto prev_ov_ts =
      absl::MakeConstSpan(prev_timestamps_).subspan(num_confirmed);

  // Current words well beyond the previous window are new; the margin lets
  // the last words of the previous window, emitted early, find their pair.
  size_t num_overlap = 0;
  while (num_overlap < curr_times.size() &&
         curr_times[num_overlap] < prev_end + tol.match_late) {
    ++num_overlap;
  }

  const auto close_matrix = BuildCloseEnoughMatrix(
      canon_prev_ov_words,
      absl::MakeConstSpan(canon_curr_words).subspan(0, num_overlap));
  const std::vector<AlignOp> ops = AlignOverlap(
      close_matrix, prev_ov_ts,
      absl::MakeConstSpan(curr_times).subspan(0, num_overlap), tol);

  std::vector<std::string> merged_words;
  std::vector<int> merged_timestamps;
  MergeAlignedOverlap(canon_prev_words, prev_timestamps_, prev_ov_words,
                      prev_ov_ts, curr_words, canon_curr_words, curr_timestamps,
                      curr_times, close_matrix, ops, chunk_start, pivot,
                      prev_chunk_end_ms_, tol, merged_words, merged_timestamps);

  ABSL_VLOG(2) << "MergeByTime: shift=" << shift
               << " chunk_start=" << chunk_start << " prev_end=" << prev_end
               << " pivot=" << pivot << " confirmed=" << num_confirmed
               << " overlap prev/curr=" << prev_ov_words.size() << "/"
               << num_overlap;

  prev_words_ = std::move(merged_words);
  prev_timestamps_ = std::move(merged_timestamps);
  if (overlap_ratio_ >= 0.7f) {
    const double early_confirm_limit = pivot - tol.substitution;
    while (confirmed_prefix_size_ < prev_timestamps_.size() &&
           prev_timestamps_[confirmed_prefix_size_] < early_confirm_limit) {
      last_confirmed_words_.push_back(prev_words_[confirmed_prefix_size_]);
      ++confirmed_prefix_size_;
    }
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
  if (confirmed_prefix_size_ > 0) {
    const size_t drop = std::min(confirmed_prefix_size_, prev_words_.size());
    prev_words_.erase(prev_words_.begin(), prev_words_.begin() + drop);
    if (prev_timestamps_.size() >= drop) {
      prev_timestamps_.erase(prev_timestamps_.begin(),
                             prev_timestamps_.begin() + drop);
    } else {
      prev_timestamps_.clear();
    }
    confirmed_prefix_size_ = 0;
  }
  const bool curr_has_usable_ts =
      HasUsableTimestamps(curr_words, curr_timestamps);
  if (!prev_words_.empty() && !curr_words.empty()) {
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
    const std::vector<WordIndexPair> best_run =
        FindBestSeamMatchRun(prev_words_, min_prev, curr_words, max_curr);
    if (!best_run.empty()) {
      const auto [p0, c0] = best_run.front();
      const auto [p_piv, c_piv] = best_run[best_run.size() / 2];
      const int unconfirmed_idx =
          std::clamp<int>(static_cast<int>(p0) - static_cast<int>(c0),
                          static_cast<int>(min_prev), static_cast<int>(p0));
      last_confirmed_words_.assign(prev_words_.begin(),
                                   prev_words_.begin() + unconfirmed_idx);
      const size_t prefix_len = p_piv - static_cast<size_t>(unconfirmed_idx);
      std::vector<std::string> unconfirmed_merged(
          prev_words_.begin() + unconfirmed_idx, prev_words_.begin() + p_piv);
      if (n_curr - c_piv >= n_prev - p_piv) {
        unconfirmed_merged.insert(unconfirmed_merged.end(),
                                  curr_words.begin() + c_piv, curr_words.end());
      } else {
        unconfirmed_merged.insert(unconfirmed_merged.end(),
                                  prev_words_.begin() + p_piv,
                                  prev_words_.end());
      }
      prev_words_ = std::move(unconfirmed_merged);
      prev_timestamps_.clear();
      if (curr_has_usable_ts) {
        prev_timestamps_.reserve(prev_words_.size());
        for (size_t k = 0; k < prev_words_.size(); ++k) {
          const int c_idx =
              std::clamp<int>(static_cast<int>(c_piv) + static_cast<int>(k) -
                                  static_cast<int>(prefix_len),
                              0, static_cast<int>(n_curr) - 1);
          prev_timestamps_.push_back(curr_timestamps[c_idx]);
        }
      }
      return;
    }
  }

  last_confirmed_words_ = std::move(prev_words_);
  prev_words_.assign(curr_words.begin(), curr_words.end());
  if (curr_has_usable_ts) {
    prev_timestamps_.assign(curr_timestamps.begin(), curr_timestamps.end());
  } else {
    prev_timestamps_.clear();
  }
}

absl::Status TimestampTextMerger::FlushInternal() {
  const size_t prefix = std::min(confirmed_prefix_size_, prev_words_.size());
  if (prefix < prev_words_.size()) {
    MergeResult result = {JoinWithCjkAwareSpacing(
                              absl::MakeConstSpan(prev_words_).subspan(prefix)),
                          ""};
    prev_words_.clear();
    prev_timestamps_.clear();
    last_confirmed_words_.clear();
    confirmed_prefix_size_ = 0;
    confirmed_until_ms_ = -1;
    prev_chunk_end_ms_ = -1.0;
    last_chunk_end_ms_ = 0;
    LogAndPushOutput(std::move(result));
  } else {
    prev_words_.clear();
    prev_timestamps_.clear();
    last_confirmed_words_.clear();
    confirmed_prefix_size_ = 0;
    confirmed_until_ms_ = -1;
    prev_chunk_end_ms_ = -1.0;
    last_chunk_end_ms_ = 0;
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
