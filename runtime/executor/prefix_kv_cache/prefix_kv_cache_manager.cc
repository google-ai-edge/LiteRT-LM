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
  
  // Insert into tree (tree_.Insert updates total_cached_tokens_ internally)
  tree_.Insert(tokens, std::move(checkpoint));
}

void PrefixKVCacheManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  tree_ = RadixTree();
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
  
  if (tree_.GetTotalCachedTokens() <= config_.max_cached_tokens) {
    return;
  }
  
  EvictLRU();
}

void PrefixKVCacheManager::EvictLRU() {
  // Collect all nodes with checkpoints (not just leaves)
  // This prevents evicting leaf nodes that don't have checkpoints while
  // accidentally deleting intermediate nodes that DO have checkpoints
  auto checkpoint_nodes = tree_.GetAllCheckpointNodes();
  
  if (checkpoint_nodes.empty()) {
    return;
  }
  
  // Sort by last_access_time ascending (LRU order)
  std::sort(checkpoint_nodes.begin(), checkpoint_nodes.end(),
            [](const RadixNode* a, const RadixNode* b) {
              int64_t time_a =
                  a->checkpoint.has_value() ? a->checkpoint->last_access_time : 0;
              int64_t time_b =
                  b->checkpoint.has_value() ? b->checkpoint->last_access_time : 0;
              return time_a < time_b;
            });
  
  int64_t tokens_to_evict =
      static_cast<int64_t>(tree_.GetCurrentCachedTokens() * config_.lru_evict_ratio);
  int64_t tokens_evicted = 0;
  
  for (RadixNode* node : checkpoint_nodes) {
    if (tokens_evicted >= tokens_to_evict) {
      break;
    }
    
    // Distinguish between leaf nodes and internal nodes
    int tokens_removed = 0;
    
    if (node->children.empty()) {
      // Leaf node: remove the entire subtree (just this node)
      // RemoveSubtreeAndReturnTokens updates total_cached_tokens_ internally
      tokens_removed = tree_.RemoveSubtreeAndReturnTokens(node);
    } else {
      // Internal node: only clear the checkpoint data, keep the subtree intact
      // This prevents cascading deletion of child checkpoints
      // ClearCheckpoint returns node->token_ids.size() (the tokens this node contributes)
      // but does NOT modify total_cached_tokens_ (tree structure is unchanged)
      tokens_removed = tree_.ClearCheckpoint(node);
    }
    
    // Note: We don't need to manually update any counter here because:
    // - RemoveSubtreeAndReturnTokens already updated total_cached_tokens_
    // - ClearCheckpoint doesn't change total_cached_tokens_ (node still exists)
    // - tree_.GetCurrentCachedTokens() always returns the current total_cached_tokens_
    tokens_evicted += tokens_removed;
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
