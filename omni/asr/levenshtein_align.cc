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

#include "omni/asr/levenshtein_align.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::asr {

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
  for (char c : lower) {
    if (!std::ispunct(static_cast<unsigned char>(c))) {
      result.push_back(c);
    }
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

bool CloseEnoughStr(absl::string_view s1, absl::string_view s2) {
  std::string c1 = CanonicalizeStr(s1);
  std::string c2 = CanonicalizeStr(s2);
  const size_t min_len = std::min(c1.size(), c2.size());
  if (min_len <= 3) {
    return c1 == c2;
  }
  return ComputeLevenshteinDistanceStr(c1, c2) <= 1;
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

}  // namespace litert::omni::asr
