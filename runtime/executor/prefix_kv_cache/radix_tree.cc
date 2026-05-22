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

#include "runtime/executor/prefix_kv_cache/radix_tree.h"

#include <algorithm>
#include <functional>
#include <utility>

namespace litert::lm {

RadixNode::~RadixNode() {
  for (auto& [_, child] : children) {
    delete child;
  }
}

RadixTree::RadixTree() : root_(new RadixNode()) {}

RadixTree::~RadixTree() {
  delete root_;
}

std::pair<int, const RadixNode*> RadixTree::FindLongestPrefixMatch(
    absl::Span<const int> tokens, BackendType backend) const {
  const RadixNode* node = root_;
  int matched = 0;
  const RadixNode* best_node = root_;
  
  for (int token : tokens) {
    auto it = node->children.find(token);
    if (it == node->children.end()) {
      break;
    }
    
    const RadixNode* child = it->second;
    
    // Check if child has a checkpoint and backend matches
    if (!child->checkpoint.has_value()) {
      break;
    }
    
    if (child->checkpoint->backend != backend) {
      break;
    }
    
    // Move to child and update matched count
    matched += static_cast<int>(child->token_ids.size());
    node = child;
    best_node = node;
  }
  
  return {matched, best_node};
}

void RadixTree::Insert(absl::Span<const int> tokens, KVCheckpoint checkpoint) {
  RadixNode* node = root_;
  int token_idx = 0;
  
  while (token_idx < tokens.size()) {
    int first_token = tokens[token_idx];
    auto it = node->children.find(first_token);
    
    if (it == node->children.end()) {
      // Create new node with remaining tokens
      std::vector<int> remaining_tokens(tokens.begin() + token_idx, tokens.end());
      int new_depth = node->depth + static_cast<int>(remaining_tokens.size());
      
      auto* new_node = new RadixNode(std::move(remaining_tokens), new_depth, node);
      new_node->checkpoint = std::move(checkpoint);
      node->children[first_token] = new_node;
      
      total_cached_tokens_ += static_cast<int>(new_node->token_ids.size());
      return;
    }
    
    // Child exists, check if we need to split or continue
    RadixNode* child = it->second;
    
    // Find how many tokens match
    int match_len = 0;
    while (match_len < child->token_ids.size() &&
           token_idx + match_len < tokens.size() &&
           child->token_ids[match_len] == tokens[token_idx + match_len]) {
      ++match_len;
    }
    
    if (match_len == child->token_ids.size()) {
      // Full match, continue to next level
      if (token_idx + match_len == tokens.size()) {
        // Exact match, update checkpoint
        child->checkpoint = std::move(checkpoint);
        return;
      }
      
      // Continue down the tree
      node = child;
      token_idx += match_len;
    } else {
      // Partial match, need to split
      std::vector<int> child_remaining(child->token_ids.begin() + match_len,
                                       child->token_ids.end());
      int new_child_depth = child->depth;
      
      // Create new intermediate node
      std::vector<int> shared_tokens(child->token_ids.begin(),
                                     child->token_ids.begin() + match_len);
      int shared_depth = node->depth + match_len;
      
      auto* intermediate_node = new RadixNode(std::move(shared_tokens), shared_depth, node);
      
      // Move existing child's children to intermediate node
      intermediate_node->children = std::move(child->children);
      intermediate_node->children[child_remaining.front()] = child;
      child->parent = intermediate_node;
      child->token_ids = std::move(child_remaining);
      child->depth = new_child_depth;
      
      // Update parent to point to intermediate node
      node->children[first_token] = intermediate_node;
      
      // If we have more tokens to insert, create another child
      if (token_idx + match_len < tokens.size()) {
        std::vector<int> new_tokens(tokens.begin() + token_idx + match_len, tokens.end());
        int new_token_depth = shared_depth + static_cast<int>(new_tokens.size());
        
        auto* new_child = new RadixNode(std::move(new_tokens), new_token_depth, intermediate_node);
        new_child->checkpoint = std::move(checkpoint);
        intermediate_node->children[new_tokens.front()] = new_child;
        
        total_cached_tokens_ += static_cast<int>(new_child->token_ids.size());
      } else {
        // Update intermediate node's checkpoint
        intermediate_node->checkpoint = std::move(checkpoint);
      }
      
      total_cached_tokens_ += match_len;
      return;
    }
  }
}

void RadixTree::RemoveSubtree(RadixNode* node) {
  if (!node || node == root_) {
    return;
  }
  
  // Calculate tokens to remove
  int tokens_to_remove = 0;
  std::function<void(RadixNode*)> count_tokens = [&](RadixNode* n) {
    if (!n || n == root_) return;
    tokens_to_remove += static_cast<int>(n->token_ids.size());
    for (auto& [_, child] : n->children) {
      count_tokens(child);
    }
  };
  
  count_tokens(node);
  total_cached_tokens_ -= tokens_to_remove;
  
  // Remove from parent
  RadixNode* parent = node->parent;
  if (parent) {
    for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
      if (it->second == node) {
        parent->children.erase(it);
        break;
      }
    }
  }
  
  // Delete node and its children (destructor handles recursive deletion)
  delete node;
}

std::vector<RadixNode*> RadixTree::GetAllLeafNodes() const {
  std::vector<RadixNode*> leaves;
  
  std::function<void(const RadixNode*)> traverse = [&](const RadixNode* node) {
    if (!node) return;
    
    if (node->children.empty()) {
      leaves.push_back(const_cast<RadixNode*>(node));
      return;
    }
    
    for (const auto& [_, child] : node->children) {
      traverse(child);
    }
  };
  
  // Start from root's children
  for (const auto& [_, child] : root_->children) {
    traverse(child);
  }
  
  return leaves;
}

int RadixTree::GetTotalNodes() const {
  int count = 0;
  
  std::function<void(const RadixNode*)> traverse = [&](const RadixNode* node) {
    if (!node) return;
    ++count;
    for (const auto& [_, child] : node->children) {
      traverse(child);
    }
  };
  
  // Start from root's children
  for (const auto& [_, child] : root_->children) {
    traverse(child);
  }
  
  return count;
}

}  // namespace litert::lm
