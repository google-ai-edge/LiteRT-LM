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

#include "runtime/components/logits_processor/no_repeat_ngram_processor.h"

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/no_repeat_ngram_config.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

NoRepeatNgramProcessor::NoRepeatNgramProcessor(int batch_size, int vocab_size,
                                               NoRepeatNgramConfig config)
    : vocab_size_(vocab_size),
      config_(std::move(config)),
      prefix_length_(config_.no_repeat_ngram_size() - 1),
      batch_states_(batch_size) {
  if (config_.enabled() && config_.window_size() > 0) {
    for (BatchState& batch_state : batch_states_) {
      batch_state.token_history.resize(config_.window_size());
    }

    if (config_.no_repeat_ngram_size() > 1) {
      for (BatchState& batch_state : batch_states_) {
        batch_state.prefix_hash_history.resize(config_.window_size());
      }
      prefix_buffer_for_hash_.resize(prefix_length_);
    }
  }
}

absl::Status NoRepeatNgramProcessor::ProcessLogits(
    absl::Span<float> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  return ProcessLogitsImpl(logits, logits_dims);
}

absl::Status NoRepeatNgramProcessor::ProcessLogits(
    absl::Span<tflite::half> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  return ProcessLogitsImpl(logits, logits_dims);
}

absl::Status NoRepeatNgramProcessor::ProcessLogits(
    ::litert::TensorBuffer& logits) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, logits.TensorType());
  auto element_type = tensor_type.ElementType();
  auto layout = tensor_type.Layout();

  if (element_type == ::litert::ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(
        auto span, ::litert::lm::ReferTensorBufferAsSpan<float>(logits));
    return ProcessLogits(absl::MakeSpan(span.data(), span.size()),
                         absl::MakeConstSpan(layout.Dimensions()));
  } else if (element_type == ::litert::ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(
        auto span, ::litert::lm::ReferTensorBufferAsSpan<tflite::half>(logits));
    return ProcessLogits(absl::MakeSpan(span.data(), span.size()),
                         absl::MakeConstSpan(layout.Dimensions()));
  }

  return absl::InvalidArgumentError(
      "Unsupported logits tensor format element type.");
}

absl::Status NoRepeatNgramProcessor::UpdateState(
    const ::litert::TensorBuffer& next_token_ids) {
  LITERT_ASSIGN_OR_RETURN(
      auto next_token_ids_span,
      ::litert::lm::ReferTensorBufferAsSpan<int>(next_token_ids));

  return UpdateState(
      absl::MakeSpan(next_token_ids_span.data(), next_token_ids_span.size()));
}

absl::Status NoRepeatNgramProcessor::UpdateState(
    absl::Span<const int> next_token_ids) {
  if (next_token_ids.size() != batch_states_.size()) {
    return absl::InvalidArgumentError(
        "next_token_ids size must match batch_size");
  }

  for (int token_id : next_token_ids) {
    if (token_id < 0 || token_id >= vocab_size_) {
      return absl::InvalidArgumentError(
          "next_token_ids contains invalid token id");
    }
  }

  if (!config_.enabled()) {
    return absl::OkStatus();
  }

  int next_token_history_index =
      config_.window_size() > 0
          ? (token_history_index_ + 1) % config_.window_size()
          : token_history_index_ + 1;

  int expired_prefix_hash_history_index = 0;
  if (config_.window_size() > 0 && config_.no_repeat_ngram_size() > 1) {
    // For example, assume the token updated sequence is [0, 1, 2, 3, 0, 1], the
    // window is 5, and the ngram size is 3. For the 6th update, before storing
    // the new token (1), the expired prefix hash is hash(0, 1) at index 2.
    expired_prefix_hash_history_index =
        (token_history_index_ + prefix_length_) % config_.window_size();
  }

  for (int batch_idx = 0; batch_idx < batch_states_.size(); ++batch_idx) {
    BatchState& batch_state = batch_states_[batch_idx];
    int token_id = next_token_ids[batch_idx];

    if (config_.window_size() > 0 &&
        num_tokens_updated_ >= config_.window_size()) {
      if (prefix_length_ == 0) {
        // Unigram banning, decrement the count of the expired token and
        // remove it if it reaches 0.
        int expired_token = batch_state.token_history[token_history_index_];
        auto it = batch_state.banned_tokens.find(expired_token);
        if (it != batch_state.banned_tokens.end()) {
          --it->second;
          if (it->second == 0) {
            batch_state.banned_tokens.erase(it);
          }
        }
      } else {
        // Remove the expired prefix hash from `prefix_hash_to_end_indices`.
        PrefixHash expired_prefix_hash =
            batch_state.prefix_hash_history[expired_prefix_hash_history_index];
        auto it =
            batch_state.prefix_hash_to_end_indices.find(expired_prefix_hash);

        if (it != batch_state.prefix_hash_to_end_indices.end()) {
          it->second.erase(expired_prefix_hash_history_index);
          if (it->second.empty()) {
            batch_state.prefix_hash_to_end_indices.erase(it);
          }
        }
      }
    }

    // Add the new token to the `token_history`.
    if (config_.window_size() == 0) {
      batch_state.token_history.push_back(token_id);
    } else {
      batch_state.token_history[token_history_index_] = token_id;
    }

    // Add the new token to the appropriate banning history.
    if (prefix_length_ == 0) {
      // Unigram banning, just add the current token to the set of banned
      // tokens.
      batch_state.banned_tokens[token_id]++;
    } else if (num_tokens_updated_ + 1 >= prefix_length_) {
      // N-gram banning, compute the prefix hash and store it in the
      // `prefix_hash_to_end_indices` and `prefix_hash_history`.
      PrefixHash prefix_hash = ComputePrefixHash(batch_state);

      batch_state.prefix_hash_to_end_indices[prefix_hash].insert(
          next_token_history_index);

      if (config_.window_size() == 0) {
        batch_state.latest_prefix_hash = prefix_hash;
      } else {
        batch_state.prefix_hash_history[next_token_history_index] = prefix_hash;
      }
    }
  }

  ++num_tokens_updated_;
  token_history_index_ = next_token_history_index;

  return absl::OkStatus();
}

int NoRepeatNgramProcessor::GetTokenHistorySizeForTesting(int batch_idx) const {
  return batch_states_[batch_idx].token_history.size();
}

int NoRepeatNgramProcessor::GetPrefixHashHistorySizeForTesting(
    int batch_idx) const {
  return batch_states_[batch_idx].prefix_hash_history.size();
}

size_t NoRepeatNgramProcessor::GetLatestPrefixHashForTesting(
    int batch_idx) const {
  if (config_.window_size() == 0) {
    return batch_states_[batch_idx].latest_prefix_hash;
  } else {
    return batch_states_[batch_idx].prefix_hash_history[token_history_index_];
  }
}

void NoRepeatNgramProcessor::SetLatestPrefixHashForTesting(int batch_idx,
                                                           size_t prefix_hash) {
  if (config_.window_size() == 0) {
    batch_states_[batch_idx].latest_prefix_hash = prefix_hash;
  } else {
    batch_states_[batch_idx].prefix_hash_history[token_history_index_] =
        prefix_hash;
  }
}

int NoRepeatNgramProcessor::GetPrefixHashCollisionCountForTesting() const {
  return prefix_hash_collision_count_;
}

NoRepeatNgramProcessor::PrefixHash NoRepeatNgramProcessor::ComputePrefixHash(
    BatchState& batch_state) const {
  int start_idx = (token_history_index_ + 1) - prefix_length_;

  // If the start index is negative, then we need to wrap around the circular
  // buffer. Note that `start_idx` will always be less than
  // `config_.window_size()` because `prefix_length_` is always greater than 0
  // and `token_history_index_` is always less than `config_.window_size()`.
  //
  // Also, since configuration clamps `window_size` to be at least
  // `no_repeat_ngram_size`, which is `prefix_length + 1`, a single integer
  // addition is guaranteed to bring `start_idx` into the valid non-negative
  // range.
  if (start_idx < 0) {
    start_idx += config_.window_size();
  }

  if (start_idx + prefix_length_ <= batch_state.token_history.size()) {
    auto span = absl::MakeConstSpan(&batch_state.token_history[start_idx],
                                    prefix_length_);
    return absl::Hash<decltype(span)>()(span);
  }

  int prefix_idx = 0;

  // Copy the end of the circular buffer first.
  for (int token_idx = start_idx; token_idx < batch_state.token_history.size();
       ++prefix_idx, ++token_idx) {
    prefix_buffer_for_hash_[prefix_idx] = batch_state.token_history[token_idx];
  }

  // Copy the beginning of the circular buffer.
  for (int token_idx = 0; token_idx <= token_history_index_;
       ++prefix_idx, ++token_idx) {
    prefix_buffer_for_hash_[prefix_idx] = batch_state.token_history[token_idx];
  }

  auto span = absl::MakeConstSpan(prefix_buffer_for_hash_);
  return absl::Hash<decltype(span)>()(span);
}

bool NoRepeatNgramProcessor::PrefixesMatch(const BatchState& batch_state,
                                           int end_index1,
                                           int end_index2) const {
  int start_index1 = end_index1 - prefix_length_;
  if (start_index1 < 0) {
    start_index1 += config_.window_size();
  }

  int start_index2 = end_index2 - prefix_length_;
  if (start_index2 < 0) {
    start_index2 += config_.window_size();
  }

  // If both prefixes are within the linear range of the token history, then we
  // can just compare the spans directly.
  if (start_index1 + prefix_length_ <= batch_state.token_history.size() &&
      start_index2 + prefix_length_ <= batch_state.token_history.size()) {
    auto span1 = absl::MakeConstSpan(&batch_state.token_history[start_index1],
                                     prefix_length_);
    auto span2 = absl::MakeConstSpan(&batch_state.token_history[start_index2],
                                     prefix_length_);
    return span1 == span2;
  }

  // Note: When `config_.window_size() == 0` (unbounded history), the loop below
  // is never entered because the short-circuit condition above
  // `start_index1 <= end_index1 && start_index2 <= end_index2` always evaluates
  // to true for linear history. Thus, we will never accidentally evaluate
  // `idx1 -= config_.window_size()` to `idx1 -= 0`.
  for (int i = 0; i < prefix_length_; ++i) {
    int idx1 = start_index1 + i;
    if (idx1 >= config_.window_size()) {
      idx1 -= config_.window_size();
    }

    int idx2 = start_index2 + i;
    if (idx2 >= config_.window_size()) {
      idx2 -= config_.window_size();
    }

    if (batch_state.token_history[idx1] != batch_state.token_history[idx2]) {
      return false;
    }
  }

  return true;
}

template <typename T>
absl::Status NoRepeatNgramProcessor::ProcessLogitsImpl(
    absl::Span<T> logits, absl::Span<const ::litert::Layout::Dim> logits_dims) {
  if (logits_dims.size() != 3 || logits_dims[0] != batch_states_.size() ||
      logits_dims[1] != 1 || logits_dims[2] != vocab_size_) {
    return absl::InvalidArgumentError(
        "Logits dimensions must be [batch_size, 1, vocab_size].");
  }

  if (logits.size() != vocab_size_ * batch_states_.size()) {
    return absl::InvalidArgumentError("Logits span size incorrectly mapped.");
  }

  if (!std::is_same_v<T, float> && !std::is_same_v<T, tflite::half>) {
    return absl::InvalidArgumentError(
        "Unsupported logits tensor format element type.");
  }

  if (!config_.enabled()) {
    return absl::OkStatus();
  }

  if (num_tokens_updated_ < config_.no_repeat_ngram_size()) {
    // Not enough tokens have been generated yet to ban any n-grams.
    return absl::OkStatus();
  }

  T min;
  if constexpr (std::is_same_v<T, float>) {
    min = -std::numeric_limits<float>::infinity();
  } else {
    min = tflite::half::min();
  }

  for (int batch_idx = 0; batch_idx < batch_states_.size(); ++batch_idx) {
    BatchState& batch_state = batch_states_[batch_idx];

    if (prefix_length_ == 0) {
      // Special case for unigram banning.
      for (const auto& [token_id, _] : batch_state.banned_tokens) {
        logits[batch_idx * vocab_size_ + token_id] = min;
      }

      continue;
    }

    PrefixHash prefix_hash =
        (config_.window_size() == 0)
            ? batch_state.latest_prefix_hash
            : batch_state.prefix_hash_history[token_history_index_];

    // N-gram banning, ban all tokens that end a previously seen n-gram.
    if (auto it = batch_state.prefix_hash_to_end_indices.find(prefix_hash);
        it != batch_state.prefix_hash_to_end_indices.end()) {
      for (int end_index : it->second) {
        // The current token is not stored in `token_history` yet, so we don't
        // need to ban it.
        if (end_index == token_history_index_) {
          continue;
        }

        // Verify that the prefix hashes actually represent identical prefixes
        // to avoid incorrect bans due to hash collisions.
        if (!PrefixesMatch(batch_state, token_history_index_, end_index)) {
          ++prefix_hash_collision_count_;
          continue;
        }

        int token_id = batch_state.token_history[end_index];
        logits[batch_idx * vocab_size_ + token_id] = min;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::lm
