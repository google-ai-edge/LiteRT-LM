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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_BUFFERED_STREAMING_DETOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_BUFFERED_STREAMING_DETOKENIZER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizer.h"

namespace litert::support {

// Result of a single detokenization step or flush operation for a sequence.
struct DetokenizedStep {
  // Newly released decoded text for this step.
  std::string text;

  // Token IDs corresponding to the newly released text.
  std::vector<int> token_ids;

  bool operator==(const DetokenizedStep& other) const {
    return text == other.text && token_ids == other.token_ids;
  }
};

class BufferedStreamingDetokenizer {
 public:
  // Creates a BufferedStreamingDetokenizer.
  // `tokenizer` must outlive this object.
  // `output_heads` is the number of heads (sequences) to process in parallel.
  // `lookback_tokens` is the number of released tokens to preserve in the
  // active buffer for context.
  BufferedStreamingDetokenizer(Tokenizer* absl_nonnull tokenizer,
                               int output_heads, size_t lookback_tokens = 8);

  // Processes a single decode step for all heads.
  // `token_ids[i]` contains the newly accepted tokens for head `i`.
  // Returns the newly released text and token IDs for each head.
  absl::StatusOr<std::vector<DetokenizedStep>> ProcessStep(
      const std::vector<std::vector<int>>& token_ids);

  // Flushes all remaining withheld text and token IDs for all heads.
  // Typically called when the session reaches EOS.
  // This includes resetting the internal state and buffers for all heads.
  absl::StatusOr<std::vector<DetokenizedStep>> Flush();

  // Resets the internal state and buffers.
  void Reset();

 private:
  Tokenizer* absl_nonnull tokenizer_;
  int output_heads_;
  size_t lookback_tokens_;

  // Accumulated token IDs for each head.
  std::vector<std::vector<int>> accumulated_token_ids_;

  // Length of the text released so far for each head (in bytes).
  std::vector<size_t> released_lengths_;

  // Decoded text of the accumulated tokens from the previous step.
  std::vector<std::string> last_decoded_texts_;

  // Index of the first unreleased token in accumulated_token_ids_ for each
  // head.
  std::vector<size_t> released_token_indices_;
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_BUFFERED_STREAMING_DETOKENIZER_H_
