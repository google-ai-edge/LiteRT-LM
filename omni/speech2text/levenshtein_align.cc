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

#include "omni/speech2text/levenshtein_align.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/utils/text_utils.h"

namespace litert::omni::asr {

namespace {

// Extracts the longest contiguous `kCorrect` run (preferring later runs of
// length >= 2 on ties) and all individual `kCorrect` index pairs from the
// Levenshtein alignment of `prev_words[min_prev..]` and
// `curr_words[0..max_curr)`.
void CollectAlignedMatches(absl::Span<const std::string> prev_words,
                           size_t min_prev,
                           absl::Span<const std::string> curr_words,
                           size_t max_curr,
                           std::vector<WordIndexPair>& best_run,
                           std::vector<WordIndexPair>& all_matches) {
  const auto align_codes = AlignTokens(prev_words.subspan(min_prev),
                                       curr_words.subspan(0, max_curr));
  std::vector<WordIndexPair> cur_run;
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
}

// Fallback when global Levenshtein alignment (`AlignTokens`) did not find a
// contiguous match run of length >= 2: because a substitution has the same
// cost as an insertion + deletion (cost 2 = 1 + 1), a non-overlapping prefix
// in `prev_words` and suffix in `curr_words` of similar length can cause the
// global DP to pair non-overlapping words as substitutions and break up a
// genuine >= 2-word overlap run at the seam. Scans directly for the longest
// contiguous `CloseEnoughStr` run of length >= 2 near the chunk boundary
// (starting within the first 4 words of `curr_words` or ending within the last
// 2 words of `prev_words`), preferring later runs in `prev_words` on ties.
void FindDirectSeamMatchRun(absl::Span<const std::string> prev_words,
                            size_t min_prev,
                            absl::Span<const std::string> curr_words,
                            size_t max_curr,
                            std::vector<WordIndexPair>& best_run) {
  const size_t n_prev = prev_words.size();
  for (size_t r = min_prev; r < n_prev; ++r) {
    for (size_t h = 0; h < max_curr; ++h) {
      size_t len = 0;
      while (r + len < n_prev && h + len < max_curr &&
             CloseEnoughStr(prev_words[r + len], curr_words[h + len])) {
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
}

// Finds a pair of aligned matches separated by at most a small gap near the
// chunk boundary where at least one word is a content word (>= 6 chars).
void FindGappedSeamMatchPair(absl::Span<const std::string> prev_words,
                             absl::Span<const WordIndexPair> all_matches,
                             std::vector<WordIndexPair>& best_run) {
  const size_t n_prev = prev_words.size();
  for (size_t m = 0; m + 1 < all_matches.size(); ++m) {
    const auto [p0, c0] = all_matches[m];
    const auto [p1, c1] = all_matches[m + 1];
    if (p1 > p0 && c1 > c0 && (p1 - p0) + (c1 - c0) <= 5 &&
        (c0 <= 2 || p1 + 2 >= n_prev) &&
        std::max(CanonicalizeStr(prev_words[p0]).size(),
                 CanonicalizeStr(prev_words[p1]).size()) >= 6) {
      best_run = {all_matches[m], all_matches[m + 1]};
    }
  }
}

bool IsValidSingleWordSeamMatch(absl::Span<const WordIndexPair> best_run,
                                size_t n_prev,
                                absl::Span<const std::string> curr_words) {
  return best_run.size() == 1 &&
         ((best_run[0].first + 1 == n_prev && best_run[0].second == 0) ||
          (best_run[0].first + 3 >= n_prev && best_run[0].second <= 1 &&
           CanonicalizeStr(curr_words[best_run[0].second]).size() >= 5));
}

}  // namespace

// Based on original implementation in speech/common/levenshtein.cc.
std::vector<AlignCode> AlignTokens(absl::Span<const std::string> ref_tokens,
                                   absl::Span<const std::string> hyp_tokens) {
  const size_t r_len = ref_tokens.size();
  const size_t h_len = hyp_tokens.size();

  if (r_len == 0 && h_len == 0) {
    return {};
  }

  // Costs: ins = 1, del = 1, match = 0, sub = 2
  std::vector<std::vector<int>> dp(r_len + 1, std::vector<int>(h_len + 1, 0));
  std::vector<std::vector<AlignCode>> backptr(
      r_len + 1, std::vector<AlignCode>(h_len + 1, AlignCode::kDeletion));

  for (size_t i = 1; i <= r_len; ++i) {
    dp[i][0] = i;
    backptr[i][0] = AlignCode::kDeletion;
  }
  for (size_t j = 1; j <= h_len; ++j) {
    dp[0][j] = j;
    backptr[0][j] = AlignCode::kInsertion;
  }

  for (size_t i = 1; i <= r_len; ++i) {
    for (size_t j = 1; j <= h_len; ++j) {
      const int del_cost = dp[i - 1][j] + 1;
      const int ins_cost = dp[i][j - 1] + 1;
      const bool is_match =
          CloseEnoughStr(ref_tokens[i - 1], hyp_tokens[j - 1]);
      const int sub_cost = dp[i - 1][j - 1] + (is_match ? 0 : 2);

      int min_cost = sub_cost;
      AlignCode code =
          is_match ? AlignCode::kCorrect : AlignCode::kSubstitution;

      if (ins_cost < min_cost) {
        min_cost = ins_cost;
        code = AlignCode::kInsertion;
      }
      if (del_cost < min_cost) {
        min_cost = del_cost;
        code = AlignCode::kDeletion;
      }

      dp[i][j] = min_cost;
      backptr[i][j] = code;
    }
  }

  std::vector<AlignCode> alignment;
  size_t i = r_len;
  size_t j = h_len;
  while (i > 0 || j > 0) {
    AlignCode code = backptr[i][j];
    alignment.push_back(code);
    if (code == AlignCode::kInsertion) {
      --j;
    } else if (code == AlignCode::kDeletion) {
      --i;
    } else {
      --i;
      --j;
    }
  }

  std::reverse(alignment.begin(), alignment.end());
  return alignment;
}

int ComputeLevenshteinDistanceStr(absl::string_view s1, absl::string_view s2) {
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

int ComputeLevenshteinDistanceWords(absl::Span<const std::string> words1,
                                    absl::Span<const std::string> words2) {
  const size_t len1 = words1.size();
  const size_t len2 = words2.size();
  std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1, 0));

  for (size_t i = 0; i <= len1; ++i) dp[i][0] = static_cast<int>(i);
  for (size_t j = 0; j <= len2; ++j) dp[0][j] = static_cast<int>(j);

  for (size_t i = 1; i <= len1; ++i) {
    for (size_t j = 1; j <= len2; ++j) {
      int deletion_cost = dp[i - 1][j] + 1;
      int insertion_cost = dp[i][j - 1] + 1;
      int match_or_sub_cost =
          dp[i - 1][j - 1] +
          (CloseEnoughStr(words1[i - 1], words2[j - 1]) ? 0 : 1);
      dp[i][j] = std::min({deletion_cost, insertion_cost, match_or_sub_cost});
    }
  }
  return dp[len1][len2];
}

std::string CanonicalizeStr(absl::string_view word) {
  std::string lower = absl::AsciiStrToLower(word);
  std::string result;
  result.reserve(lower.size());
  size_t pos = 0;
  while (pos < lower.size()) {
    const size_t start = pos;
    if (IsPunctuation(DecodeUtf8CodePoint(lower, pos))) continue;
    result.append(lower, start, pos - start);
  }
  return result;
}

std::vector<std::string> CanonicalizeWords(
    absl::Span<const std::string> words) {
  std::vector<std::string> result;
  result.reserve(words.size());
  for (const auto& word : words) {
    result.push_back(CanonicalizeStr(word));
  }
  return result;
}

bool CloseEnoughCanonical(absl::string_view c1, absl::string_view c2) {
  const size_t min_len = std::min(c1.size(), c2.size());
  if (min_len <= 3) {
    return c1 == c2;
  }
  return ComputeLevenshteinDistanceStr(c1, c2) <= 1;
}

bool CloseEnoughStr(absl::string_view s1, absl::string_view s2) {
  return CloseEnoughCanonical(CanonicalizeStr(s1), CanonicalizeStr(s2));
}

bool CloseEnoughWords(absl::Span<const std::string> l1,
                      absl::Span<const std::string> l2) {
  if (l1.size() != l2.size()) return false;
  for (size_t i = 0; i < l1.size(); ++i) {
    if (!CloseEnoughStr(l1[i], l2[i])) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> DedupWords(absl::Span<const std::string> prev_words,
                                    absl::Span<const std::string> curr_words,
                                    int search_window) {
  const int max_search = std::min<int>(
      search_window, std::min(prev_words.size(), curr_words.size()));
  for (int i = max_search; i >= 1; --i) {
    auto prev_tail = prev_words.subspan(prev_words.size() - i, i);
    auto curr_head = curr_words.subspan(0, i);
    if (CloseEnoughWords(prev_tail, curr_head)) {
      return std::vector<std::string>(curr_words.begin() + i, curr_words.end());
    }
  }
  return std::vector<std::string>(curr_words.begin(), curr_words.end());
}

std::vector<WordIndexPair> FindBestSeamMatchRun(
    absl::Span<const std::string> prev_words, size_t min_prev,
    absl::Span<const std::string> curr_words, size_t max_curr) {
  std::vector<WordIndexPair> best_run;
  std::vector<WordIndexPair> all_matches;
  CollectAlignedMatches(prev_words, min_prev, curr_words, max_curr, best_run,
                        all_matches);
  if (best_run.size() < 2) {
    FindDirectSeamMatchRun(prev_words, min_prev, curr_words, max_curr,
                           best_run);
    if (best_run.size() < 2 && all_matches.size() >= 2) {
      FindGappedSeamMatchPair(prev_words, all_matches, best_run);
    }
  }
  if (best_run.size() >= 2 ||
      IsValidSingleWordSeamMatch(best_run, prev_words.size(), curr_words)) {
    return best_run;
  }
  return {};
}

}  // namespace litert::omni::asr
