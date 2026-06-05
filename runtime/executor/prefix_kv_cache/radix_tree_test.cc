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

#include <gtest/gtest.h>

#include <vector>

namespace litert::lm {
namespace {

TEST(RadixTreeTest, BasicMatch) {
  RadixTree tree;
  
  // Insert [1, 2, 3]
  KVCheckpoint cp1;
  cp1.num_tokens = 3;
  cp1.backend = BackendType::CPU;
  cp1.last_access_time = 1;
  tree.Insert({1, 2, 3}, cp1);
  
  // Insert [1, 2, 4]
  KVCheckpoint cp2;
  cp2.num_tokens = 3;
  cp2.backend = BackendType::CPU;
  cp2.last_access_time = 2;
  tree.Insert({1, 2, 4}, cp2);
  
  // Query [1, 2, 3, 5] - should match 3 tokens
  auto [matched_len, node] = tree.FindLongestPrefixMatch({1, 2, 3, 5}, BackendType::CPU);
  EXPECT_EQ(matched_len, 3);
  EXPECT_NE(node, nullptr);
  EXPECT_EQ(node->depth, 3);
}

TEST(RadixTreeTest, NoMatch) {
  RadixTree tree;
  
  // Insert [1, 2, 3]
  KVCheckpoint cp;
  cp.num_tokens = 3;
  cp.backend = BackendType::CPU;
  cp.last_access_time = 1;
  tree.Insert({1, 2, 3}, cp);
  
  // Query [4, 5, 6] - should match 0 tokens
  auto [matched_len, node] = tree.FindLongestPrefixMatch({4, 5, 6}, BackendType::CPU);
  EXPECT_EQ(matched_len, 0);
  EXPECT_EQ(node, nullptr);  // Root node is null or we track it differently
}

TEST(RadixTreeTest, BackendIsolation) {
  RadixTree tree;
  
  // Insert [1, 2] with CPU backend
  KVCheckpoint cp_cpu;
  cp_cpu.num_tokens = 2;
  cp_cpu.backend = BackendType::CPU;
  cp_cpu.last_access_time = 1;
  tree.Insert({1, 2}, cp_cpu);
  
  // Query with GPU backend - should not match
  auto [matched_len, node] = tree.FindLongestPrefixMatch({1, 2, 3}, BackendType::GPU);
  EXPECT_EQ(matched_len, 0);
}

TEST(RadixTreeTest, PartialMatch) {
  RadixTree tree;
  
  // Insert [1, 2, 3, 4]
  KVCheckpoint cp;
  cp.num_tokens = 4;
  cp.backend = BackendType::CPU;
  cp.last_access_time = 1;
  tree.Insert({1, 2, 3, 4}, cp);
  
  // Query [1, 2, 5] - should match 2 tokens
  auto [matched_len, node] = tree.FindLongestPrefixMatch({1, 2, 5}, BackendType::CPU);
  EXPECT_EQ(matched_len, 2);
  EXPECT_EQ(node->depth, 2);
}

TEST(RadixTreeTest, SingleNode) {
  RadixTree tree;
  
  // Insert [1, 2, 3]
  KVCheckpoint cp;
  cp.num_tokens = 3;
  cp.backend = BackendType::CPU;
  cp.last_access_time = 1;
  tree.Insert({1, 2, 3}, cp);
  
  // Query exact match [1, 2, 3]
  auto [matched_len, node] = tree.FindLongestPrefixMatch({1, 2, 3}, BackendType::CPU);
  EXPECT_EQ(matched_len, 3);
  EXPECT_EQ(node->depth, 3);
  
  // Query longer [1, 2, 3, 4, 5] - should still match 3
  auto [matched_len2, node2] = tree.FindLongestPrefixMatch({1, 2, 3, 4, 5}, BackendType::CPU);
  EXPECT_EQ(matched_len2, 3);
}

TEST(RadixTreeTest, GetLeafNodes) {
  RadixTree tree;
  
  // Insert multiple sequences
  KVCheckpoint cp;
  cp.backend = BackendType::CPU;
  cp.last_access_time = 1;
  
  cp.num_tokens = 3;
  tree.Insert({1, 2, 3}, cp);
  
  cp.num_tokens = 3;
  tree.Insert({1, 2, 4}, cp);
  
  cp.num_tokens = 2;
  tree.Insert({5, 6}, cp);
  
  auto leaves = tree.GetAllLeafNodes();
  EXPECT_EQ(leaves.size(), 3);
}

TEST(RadixTreeTest, RemoveSubtree) {
  RadixTree tree;
  
  // Insert [1, 2, 3]
  KVCheckpoint cp;
  cp.num_tokens = 3;
  cp.backend = BackendType::CPU;
  cp.last_access_time = 1;
  tree.Insert({1, 2, 3}, cp);
  
  EXPECT_EQ(tree.GetTotalCachedTokens(), 3);
  
  // Find the leaf node
  auto [matched_len, node] = tree.FindLongestPrefixMatch({1, 2, 3}, BackendType::CPU);
  EXPECT_EQ(matched_len, 3);
  
  // Remove it
  tree.RemoveSubtree(const_cast<RadixNode*>(node));
  
  EXPECT_EQ(tree.GetTotalCachedTokens(), 0);
  
  // Query again - should not match
  auto [matched_len2, node2] = tree.FindLongestPrefixMatch({1, 2, 3}, BackendType::CPU);
  EXPECT_EQ(matched_len2, 0);
}

TEST(RadixTreeTest, TotalNodes) {
  RadixTree tree;
  
  KVCheckpoint cp;
  cp.backend = BackendType::CPU;
  cp.last_access_time = 1;
  
  // Insert [1, 2, 3] - creates 1 node
  cp.num_tokens = 3;
  tree.Insert({1, 2, 3}, cp);
  EXPECT_EQ(tree.GetTotalNodes(), 1);
  
  // Insert [1, 2, 4] - should split and create more nodes
  cp.num_tokens = 3;
  tree.Insert({1, 2, 4}, cp);
  // Structure: root -> [1,2] -> [3] and [4]
  // Total nodes: [1,2] + [3] + [4] = 3
  EXPECT_EQ(tree.GetTotalNodes(), 3);
}

}  // namespace
}  // namespace litert::lm
