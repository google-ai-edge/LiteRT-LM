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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_PREFIX_KV_CACHE_PREFIX_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_PREFIX_KV_CACHE_PREFIX_KV_CACHE_MANAGER_H_

#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/executor/prefix_kv_cache/radix_tree.h"

namespace litert::lm {

// Configuration for prefix KV cache
struct PrefixKVCacheConfig {
  int max_cached_tokens = 8192;    // 0 = unlimited
  double lru_evict_ratio = 0.3;    // Evict 30% of least recently used nodes
};

// Result of a cache lookup
struct PrefixCacheHit {
  int matched_len;
  const KVCheckpoint* checkpoint;  // nullptr if miss
};

// Prefix KV Cache Manager with LRU eviction
class PrefixKVCacheManager {
 public:
  static absl::StatusOr<std::unique_ptr<PrefixKVCacheManager>> Create(
      PrefixKVCacheConfig config);
  
  // Lookup longest prefix match
  PrefixCacheHit Lookup(absl::Span<const int> tokens, BackendType backend);
  
  // Store a new KV checkpoint
  void Store(absl::Span<const int> tokens, std::string serialized_kv,
             BackendType backend);
  
  // Clear all cached data
  void Clear();
  
  // Get cache statistics
  struct Stats {
    int total_cached_tokens;
    int total_nodes;
  };
  Stats GetStats() const;

 private:
  explicit PrefixKVCacheManager(PrefixKVCacheConfig config);
  
  void MaybeEvict();
  void EvictLRU();
  int CalculateSubtreeTokens(const RadixNode* node) const;
  
  RadixTree tree_;
  PrefixKVCacheConfig config_;
  int current_token_count_ = 0;
  int64_t monotonic_clock_ = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_PREFIX_KV_CACHE_PREFIX_KV_CACHE_MANAGER_H_
