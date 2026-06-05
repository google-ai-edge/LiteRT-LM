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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_PREFIX_KV_CACHE_RADIX_TREE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_PREFIX_KV_CACHE_RADIX_TREE_H_

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Backend type enum for cache isolation
enum class BackendType {
  CPU,
  CPU_ARTISAN,
  GPU,
  GPU_ARTISAN,
  NPU,
  GOOGLE_TENSOR_ARTISAN,
};

// KV checkpoint stored at each node
struct KVCheckpoint {
  std::string serialized_kv;  // Serialized KV cache data
  int num_tokens;             // Number of tokens this checkpoint represents
  BackendType backend;        // Backend type for isolation
  int64_t last_access_time;   // LRU tracking
};

// Radix tree node
struct RadixNode {
  std::vector<int> token_ids;  // Token sequence from parent to this node
  int depth;                   // Cumulative token count from root to this node
  RadixNode* parent;
  std::optional<KVCheckpoint> checkpoint;  // Each node has a checkpoint
  std::unordered_map<int, RadixNode*> children;  // Key = first token

  RadixNode() : depth(0), parent(nullptr) {}
  
  explicit RadixNode(std::vector<int> tokens, int node_depth, RadixNode* parent_node)
      : token_ids(std::move(tokens)),
        depth(node_depth),
        parent(parent_node) {}
  
  ~RadixNode();
  
  // Disable copy
  RadixNode(const RadixNode&) = delete;
  RadixNode& operator=(const RadixNode&) = delete;
};

// Radix tree for prefix KV cache
class RadixTree {
 public:
  RadixTree();
  ~RadixTree();
  
  // Find longest prefix match
  // Returns: {matched_length, pointer to the matched node}
  std::pair<int, const RadixNode*> FindLongestPrefixMatch(
      absl::Span<const int> tokens, BackendType backend) const;
  
  // Insert a new token sequence with its KV checkpoint
  // Returns: the number of NEW tokens added to the cache (0 if just updating existing checkpoint)
  int Insert(absl::Span<const int> tokens, KVCheckpoint checkpoint);
  
  // Remove subtree rooted at node
  void RemoveSubtree(RadixNode* node);
  
  // Remove subtree rooted at node and return the number of tokens removed
  int RemoveSubtreeAndReturnTokens(RadixNode* node);
  
  // Clear checkpoint from a node (keeps the node and its children intact)
  // Returns the number of tokens that were associated with this checkpoint
  int ClearCheckpoint(RadixNode* node);
  
  // Get all leaf nodes
  std::vector<RadixNode*> GetAllLeafNodes() const;
  
  // Get all nodes with valid checkpoints
  std::vector<RadixNode*> GetAllCheckpointNodes() const;
  
  // Get total number of nodes
  int GetTotalNodes() const;
  
  // Get total cached tokens
  int GetTotalCachedTokens() const { return total_cached_tokens_; }
  
  // Get current cached tokens (same as total_cached_tokens_)
  int GetCurrentCachedTokens() const { return total_cached_tokens_; }

 private:
  RadixNode* root_;
  int total_cached_tokens_ = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_PREFIX_KV_CACHE_RADIX_TREE_H_
