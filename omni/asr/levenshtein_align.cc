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
#include <cstddef>
#include <string>
#include <vector>

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
      const bool is_match = (ref_tokens[i - 1] == hyp_tokens[j - 1]);
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

}  // namespace litert::omni::asr
