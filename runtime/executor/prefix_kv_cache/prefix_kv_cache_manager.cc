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

#include "runtime/executor/prefix_kv_cache/prefix_kv_cache_manager.h"

#include <algorithm>
#include <vector>

namespace litert::lm {

// static
absl::StatusOr<std::unique_ptr<PrefixKVCacheManager>>
PrefixKVCacheManager::Create(PrefixKVCacheConfig config) {
  if (config.max_cached_tokens < 0) {
    return absl::InvalidArgumentError("max_cached_tokens must be >= 0");
  }
  if (config.lru_evict_ratio <= 0.0 || config.lru_evict_ratio > 1.0) {
    return absl::InvalidArgumentError("lru_evict_ratio must be in (0, 1]");
  }
  
  return std::unique_ptr<PrefixKVCacheManager>(
      new PrefixKVCacheManager(std::move(config)));
}

PrefixKVCacheManager::PrefixKVCacheManager(PrefixKVCacheConfig config)
    : config_(std::move(config)) {}

PrefixCacheHit PrefixKVCacheManager::Lookup(absl::Span<const int> tokens,
                                            BackendType backend) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto [matched_len, node] = tree_.FindLongestPrefixMatch(tokens, backend);
  
  if (matched_len == 0 || !node->checkpoint.has_value()) {
    return {0, nullptr};
  }
  
  // Update LRU timestamp
  const_cast<KVCheckpoint&>(node->checkpoint.value()).last_access_time =
      ++monotonic_clock_;
  
  return {matched_len, &node->checkpoint.value()};
}

void PrefixKVCacheManager::Store(absl::Span<const int> tokens,
                                 std::string serialized_kv,
                                 BackendType backend) {
  if (tokens.empty()) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Check if we need to evict
  MaybeEvict();
  
  // Create checkpoint
  KVCheckpoint checkpoint;
  checkpoint.serialized_kv = std::move(serialized_kv);
  checkpoint.num_tokens = static_cast<int>(tokens.size());
  checkpoint.backend = backend;
  checkpoint.last_access_time = ++monotonic_clock_;
  
  // Insert into tree
  tree_.Insert(tokens, std::move(checkpoint));
  current_token_count_ += static_cast<int>(tokens.size());
}

void PrefixKVCacheManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  tree_ = RadixTree();
  current_token_count_ = 0;
  monotonic_clock_ = 0;
}

PrefixKVCacheManager::Stats PrefixKVCacheManager::GetStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {
      .total_cached_tokens = tree_.GetTotalCachedTokens(),
      .total_nodes = tree_.GetTotalNodes(),
  };
}

void PrefixKVCacheManager::MaybeEvict() {
  // Unlimited cache
  if (config_.max_cached_tokens == 0) {
    return;
  }
  
  if (current_token_count_ <= config_.max_cached_tokens) {
    return;
  }
  
  EvictLRU();
}

void PrefixKVCacheManager::EvictLRU() {
  auto leaves = tree_.GetAllLeafNodes();
  
  if (leaves.empty()) {
    return;
  }
  
  // Sort by last_access_time ascending
  std::sort(leaves.begin(), leaves.end(),
            [](const RadixNode* a, const RadixNode* b) {
              int64_t time_a =
                  a->checkpoint.has_value() ? a->checkpoint->last_access_time : 0;
              int64_t time_b =
                  b->checkpoint.has_value() ? b->checkpoint->last_access_time : 0;
              return time_a < time_b;
            });
  
  int64_t tokens_to_evict =
      static_cast<int64_t>(current_token_count_ * config_.lru_evict_ratio);
  int64_t tokens_evicted = 0;
  
  for (RadixNode* leaf : leaves) {
    if (tokens_evicted >= tokens_to_evict) {
      break;
    }
    
    int tokens_in_subtree = CalculateSubtreeTokens(leaf);
    tree_.RemoveSubtree(leaf);
    current_token_count_ -= tokens_in_subtree;
    tokens_evicted += tokens_in_subtree;
  }
}

int PrefixKVCacheManager::CalculateSubtreeTokens(const RadixNode* node) const {
  if (!node) {
    return 0;
  }
  
  int total = static_cast<int>(node->token_ids.size());
  
  for (const auto& [_, child] : node->children) {
    total += CalculateSubtreeTokens(child);
  }
  
  return total;
}

}  // namespace litert::lm
