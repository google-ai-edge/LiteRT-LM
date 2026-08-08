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

#include "support/tokenizer/buffered_streaming_detokenizer.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizer.h"

namespace litert::support {

namespace {

size_t GetLongestCommonPrefixLength(std::string_view a, std::string_view b) {
  size_t min_len = std::min(a.length(), b.length());
  for (size_t i = 0; i < min_len; ++i) {
    if (a[i] != b[i]) {
      return i;
    }
  }
  return min_len;
}

std::string_view TrimTrailingReplacement(std::string_view text) {
  constexpr std::string_view kReplacementChar = "\xef\xbf\xbd";
  while (text.ends_with(kReplacementChar)) {
    text.remove_suffix(kReplacementChar.length());
  }
  return text;
}

}  // namespace

BufferedStreamingDetokenizer::BufferedStreamingDetokenizer(
    Tokenizer* absl_nonnull tokenizer, int output_heads, size_t lookback_tokens)
    : tokenizer_(tokenizer),
      output_heads_(output_heads),
      lookback_tokens_(lookback_tokens) {
  Reset();
}

absl::StatusOr<std::vector<DetokenizedStep>>
BufferedStreamingDetokenizer::ProcessStep(
    const std::vector<std::vector<int>>& token_ids) {
  if (token_ids.size() != output_heads_) {
    return absl::InvalidArgumentError("Input batch size does not match.");
  }

  std::vector<DetokenizedStep> released_steps(output_heads_);
  for (int i = 0; i < output_heads_; ++i) {
    size_t prev_token_count = accumulated_token_ids_[i].size();

    // 1. Append new tokens.
    accumulated_token_ids_[i].insert(accumulated_token_ids_[i].end(),
                                     token_ids[i].begin(), token_ids[i].end());

    // 2. Decode the new accumulated tokens.
    std::string decoded;
    if (token_ids[i].empty()) {
      decoded = last_decoded_texts_[i];
    } else {
      ABSL_ASSIGN_OR_RETURN(
          decoded, tokenizer_->TokenIdsToText(accumulated_token_ids_[i]));
    }

    std::string released_text;
    if (last_decoded_texts_[i].empty()) {
      // First valid decode. We don't release anything yet to lag by 1 step.
      released_text = "";
    } else {
      std::string_view decoded_trimmed = TrimTrailingReplacement(decoded);
      size_t stable_length =
          GetLongestCommonPrefixLength(last_decoded_texts_[i], decoded_trimmed);
      size_t released_len = released_lengths_[i];
      if (stable_length < released_len) {
        // This shouldn't happen assuming detokenization is monotonic after
        // trimming.
        return absl::InternalError("Stable text shrunk compared to released.");
      }
      released_text =
          decoded_trimmed.substr(released_len, stable_length - released_len);
      released_lengths_[i] = stable_length;
    }
    last_decoded_texts_[i] = decoded;

    released_steps[i].text = released_text;
    if (!released_text.empty()) {
      size_t next_released_token_index = prev_token_count;
      if (next_released_token_index > released_token_indices_[i]) {
        released_steps[i].token_ids.assign(
            accumulated_token_ids_[i].begin() + released_token_indices_[i],
            accumulated_token_ids_[i].begin() + next_released_token_index);
        released_token_indices_[i] = next_released_token_index;
      }
    }

    // 3. Prune old released tokens outside the lookback window.
    if (released_token_indices_[i] > lookback_tokens_) {
      size_t prune_tokens = released_token_indices_[i] - lookback_tokens_;

      accumulated_token_ids_[i].erase(
          accumulated_token_ids_[i].begin(),
          accumulated_token_ids_[i].begin() + prune_tokens);
      released_token_indices_[i] -= prune_tokens;

      // Re-decode the remaining token window to synchronize last_decoded_texts_
      // and released_lengths_.
      ABSL_ASSIGN_OR_RETURN(
          std::string new_decoded,
          tokenizer_->TokenIdsToText(accumulated_token_ids_[i]));

      size_t unreleased_text_len = decoded.length() - released_lengths_[i];
      size_t new_released_len = new_decoded.length() > unreleased_text_len
                                    ? new_decoded.length() - unreleased_text_len
                                    : 0;

      last_decoded_texts_[i] = new_decoded;
      released_lengths_[i] = new_released_len;
    }
  }

  return released_steps;
}

absl::StatusOr<std::vector<DetokenizedStep>>
BufferedStreamingDetokenizer::Flush() {
  std::vector<DetokenizedStep> released_steps(output_heads_);

  for (int i = 0; i < output_heads_; ++i) {
    if (accumulated_token_ids_[i].empty()) {
      released_steps[i] = {"", {}};
      continue;
    }

    const std::string& decoded = last_decoded_texts_[i];
    size_t released_len = released_lengths_[i];
    if (decoded.length() < released_len) {
      return absl::InternalError(
          "Decoded text shrunk compared to released during flush.");
    }
    released_steps[i].text = decoded.substr(released_len);
    released_lengths_[i] = decoded.length();

    if (accumulated_token_ids_[i].size() > released_token_indices_[i]) {
      released_steps[i].token_ids.assign(
          accumulated_token_ids_[i].begin() + released_token_indices_[i],
          accumulated_token_ids_[i].end());
    }
  }

  Reset();
  return released_steps;
}

void BufferedStreamingDetokenizer::Reset() {
  accumulated_token_ids_.assign(output_heads_, std::vector<int>());
  released_lengths_.assign(output_heads_, 0);
  last_decoded_texts_.assign(output_heads_, "");
  released_token_indices_.assign(output_heads_, 0);
}

}  // namespace litert::support
