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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_NO_REPEAT_NGRAM_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_NO_REPEAT_NGRAM_PROCESSOR_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/logits_processor.h"
#include "runtime/components/logits_processor/no_repeat_ngram_config.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

// `NoRepeatNgramProcessor` modifies logits to ban previously seen n-grams. If
// `no_repeat_ngram_size` > 0, all n-grams of that size can only occur once.
//
// To enforce an n-gram ban, we map the `n-1` preceding generated tokens
// (the "prefix") to the final token that completed the n-gram. If `n` is 1, all
// tokens are banned from repeating.
//
// `UpdateState` removes expired n-grams and their hashes from the history and
// adds a new n-gram and its hash to the history.
//
// `ProcessLogits` checks the prefix hash history for any previous
// occurrences of the current prefix and bans those tokens from being generated.
//
// For example, if the vocabulary is {A, B}, the original sequence is
// [A, A, A, A], the window size is 3, and the configured n-gram size is 3
// (trigram), we are banning identical 3-token sequence blocks from occurring,
// so the generated sequence should be [A, A, A, B]. The process is as follows:
// 1. `ProcessLogits(index 0)`: no existing prefix hash for a trigram, so no
//    bans.
// 2. `UpdateState(index 0)`: store token A at index 0 in the token history.
// 3. `ProcessLogits(index 1)`: no existing prefix hash for a trigram, so no
//    bans.
// 4. `UpdateState(index 1)`: store token A at index 1 in the token history,
//    {hash(A, A), 2} in `prefix_hash_to_end_indices`, and hash(A, A) at index 2
//    in the prefix hash history.
// 5. `ProcessLogits(index 2)`: found hash(A, A) at index 2 in the prefix hash
//    history. As there's no other hash(A, A) in `prefix_hash_to_end_indices`,
//    no bans are applied.
// 6. `UpdateState(index 2)`: store token A at index 2 in the token history,
//    {hash(A, A), 3} in `prefix_hash_to_end_indices`, and hash(A, A) at index 3
//    in the prefix hash history.
// 7. `ProcessLogits(index 3)`: found hash(A, A) at index 3 in the prefix hash
//    history. As there's another hash(A, A) at index 2 in
//    `prefix_hash_to_end_indices`, we ban the token `A` at index 3 from being
//    generated, and the only valid token is `B`.
// 8. `UpdateState(index 3)`: store token B at index 3 in the token history,
//    {hash(A, B), 4} in `prefix_hash_to_end_indices`, hash(A, B) at index 4
//    in the prefix hash history, and remove the expired {hash(A, A), 2} from
//    `prefix_hash_to_end_indices` and the expired hash(A, A) from the prefix
//    hash history.
class NoRepeatNgramProcessor : public LogitsProcessor {
 public:
  // Creates a NoRepeatNgramProcessor for a given vocabulary size and batch
  // size.
  //
  // @param batch_size The number of sequences processed in parallel.
  // @param vocab_size The vocabulary size of the model.
  // @param config The configuration containing all penalty parameters.
  explicit NoRepeatNgramProcessor(int batch_size, int vocab_size,
                                  NoRepeatNgramConfig config);

  ~NoRepeatNgramProcessor() override = default;

  absl::Status ProcessLogits(::litert::TensorBuffer& logits) override;

  absl::Status ProcessLogits(
      absl::Span<float> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims) override;

  absl::Status ProcessLogits(
      absl::Span<tflite::half> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims) override;

  absl::Status UpdateState(
      const ::litert::TensorBuffer& next_token_ids) override;

  absl::Status UpdateState(absl::Span<const int> next_token_ids) override;

  // For testing only.
  int GetTokenHistorySizeForTesting(int batch_idx) const;
  int GetPrefixHashHistorySizeForTesting(int batch_idx) const;
  size_t GetLatestPrefixHashForTesting(int batch_idx) const;
  void SetLatestPrefixHashForTesting(int batch_idx, size_t prefix_hash);
  int GetPrefixHashCollisionCountForTesting() const;

 private:
  using PrefixHash = size_t;

  // Stateful information tracked for an individual sequence in the batch.
  struct BatchState {
    // If `config_.window_size() > 0`, this is a circular buffer of size
    // `window_size`. Otherwise, it stores all tokens generated so far for this
    // sequence in the batch.
    std::vector<int> token_history;

    // This field is only used when `config_.no_repeat_ngram_size() == 1`. This
    // stores the set of tokens that have been generated so far, and their
    // frequency (within the window if `config_.window_size() > 0`).
    absl::flat_hash_map<int, int> banned_tokens;

    // This field is only used when `config_.no_repeat_ngram_size() > 1`. This
    // stores the hashes of all prefixes of length `no_repeat_ngram_size - 1` to
    // the list of absolute indices where the *end* of the ngram occurred (i.e.
    // the index belonging to the banned token itself). For example, if
    // `token_history` is [A, B, C, D] and `no_repeat_ngram_size` is 3, then
    // `prefix_hash_to_end_indices` will be {hash(A, B): {2}, hash(B, C): {3}}.
    absl::flat_hash_map<PrefixHash, absl::flat_hash_set<int>>
        prefix_hash_to_end_indices;

    // This field is only used when `config_.no_repeat_ngram_size() > 1` and
    // `config_.window_size() == 0`. This stores the hash of the latest prefix
    // of length `no_repeat_ngram_size - 1`. We don't need to store all the
    // hashes in a circular buffer, because we never expire old prefixes.
    PrefixHash latest_prefix_hash;

    // This field is only used when `config_.no_repeat_ngram_size() > 1` and
    // `config_.window_size() > 0`. This stores the hashes of the prefixes of
    // length `no_repeat_ngram_size - 1` for tokens in `token_history`. This
    // allows us to efficiently find expired n-grams when removing old tokens.
    //
    // For example, if `token_history` is [A, B, C, D] and
    // `no_repeat_ngram_size` is 3, then `prefix_history` will be
    // [index 2: hash(A, B), index 3: hash(B, C)].
    std::vector<PrefixHash> prefix_hash_history;
  };

  // Helper to compute a hash for a prefix of tokens. Do not call this if
  // `config_.no_repeat_ngram_size() == 1`. `token_history_index_` is the
  // current index in `token_history` that's being updated during `UpdateState`.
  //
  // @param batch_state The batch state to compute the prefix hash for.
  // @return The prefix hash.
  PrefixHash ComputePrefixHash(BatchState& batch_state) const;

  // Helper to check if two prefixes are identical. `end_index1` and
  // `end_index2` are the indices representing the token immediately following
  // the prefix in `token_history`.
  //
  // @param batch_state The batch state to check.
  // @param end_index1 The index of the token following the first prefix.
  // @param end_index2 The index of the token following the second prefix.
  // @return True if the prefixes are identical, false otherwise.
  bool PrefixesMatch(const BatchState& batch_state, int end_index1,
                     int end_index2) const;

  // Template implementation of ProcessLogits to support different numeric types
  // for logits.
  template <typename T>
  absl::Status ProcessLogitsImpl(
      absl::Span<T> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims);

  const int vocab_size_;
  const NoRepeatNgramConfig config_;
  const int prefix_length_;

  std::vector<BatchState> batch_states_;

  // The number of tokens that have been updated to this processor by
  // `UpdateState`.
  int64_t num_tokens_updated_ = 0;

  // The index in `token_history` where the next generated token will be
  // written. All batch states share the same index.
  int token_history_index_ = 0;

  // This is only used when `config_.window_size() > 0` and
  // `config_.no_repeat_ngram_size() > 1`.
  // A temporary buffer to store the prefix of tokens that are used to compute
  // the prefix hash. This is to avoid repeating heap allocations for each
  // call to `ComputePrefixHash`.
  mutable std::vector<int> prefix_buffer_for_hash_;

  // The number of times a prefix hash collision has occurred. This is used for
  // testing and debugging purposes only.
  int prefix_hash_collision_count_ = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_NO_REPEAT_NGRAM_PROCESSOR_H_
