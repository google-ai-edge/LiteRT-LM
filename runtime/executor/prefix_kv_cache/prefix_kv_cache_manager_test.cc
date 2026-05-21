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

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace litert::lm {
namespace {

TEST(PrefixKVCacheManagerTest, BasicStoreLookup) {
  PrefixKVCacheConfig config;
  config.max_cached_tokens = 1000;
  
  auto manager = PrefixKVCacheManager::Create(config);
  ASSERT_TRUE(manager.ok());
  
  // Store [1, 2, 3]
  (*manager)->Store({1, 2, 3}, "kv_data_123", BackendType::CPU);
  
  // Lookup [1, 2, 3, 4] - should match 3 tokens
  auto hit = (*manager)->Lookup({1, 2, 3, 4}, BackendType::CPU);
  EXPECT_EQ(hit.matched_len, 3);
  EXPECT_NE(hit.checkpoint, nullptr);
  EXPECT_EQ(hit.checkpoint->serialized_kv, "kv_data_123");
}

TEST(PrefixKVCacheManagerTest, Eviction) {
  PrefixKVCacheConfig config;
  config.max_cached_tokens = 10;
  config.lru_evict_ratio = 0.5;
  
  auto manager = PrefixKVCacheManager::Create(config);
  ASSERT_TRUE(manager.ok());
  
  // Store multiple sequences
  (*manager)->Store({1, 2, 3}, "data1", BackendType::CPU);
  (*manager)->Store({4, 5, 6}, "data2", BackendType::CPU);
  (*manager)->Store({7, 8, 9}, "data3", BackendType::CPU);
  
  // Stats should show cached tokens
  auto stats = (*manager)->GetStats();
  EXPECT_GT(stats.total_cached_tokens, 0);
}

TEST(PrefixKVCacheManagerTest, Unlimited) {
  PrefixKVCacheConfig config;
  config.max_cached_tokens = 0;  // Unlimited
  
  auto manager = PrefixKVCacheManager::Create(config);
  ASSERT_TRUE(manager.ok());
  
  // Store many sequences - should not evict
  for (int i = 0; i < 100; ++i) {
    (*manager)->Store({i, i + 1, i + 2}, "data", BackendType::CPU);
  }
  
  auto stats = (*manager)->GetStats();
  EXPECT_GT(stats.total_cached_tokens, 100);
}

TEST(PrefixKVCacheManagerTest, Clear) {
  PrefixKVCacheConfig config;
  config.max_cached_tokens = 1000;
  
  auto manager = PrefixKVCacheManager::Create(config);
  ASSERT_TRUE(manager.ok());
  
  (*manager)->Store({1, 2, 3}, "data", BackendType::CPU);
  
  // Clear
  (*manager)->Clear();
  
  auto stats = (*manager)->GetStats();
  EXPECT_EQ(stats.total_cached_tokens, 0);
  EXPECT_EQ(stats.total_nodes, 0);
  
  // Lookup should miss
  auto hit = (*manager)->Lookup({1, 2, 3}, BackendType::CPU);
  EXPECT_EQ(hit.matched_len, 0);
  EXPECT_EQ(hit.checkpoint, nullptr);
}

TEST(PrefixKVCacheManagerTest, BackendIsolation) {
  PrefixKVCacheConfig config;
  config.max_cached_tokens = 1000;
  
  auto manager = PrefixKVCacheManager::Create(config);
  ASSERT_TRUE(manager.ok());
  
  // Store with CPU backend
  (*manager)->Store({1, 2, 3}, "cpu_data", BackendType::CPU);
  
  // Lookup with GPU backend - should miss
  auto hit = (*manager)->Lookup({1, 2, 3}, BackendType::GPU);
  EXPECT_EQ(hit.matched_len, 0);
  EXPECT_EQ(hit.checkpoint, nullptr);
}

}  // namespace
}  // namespace litert::lm
