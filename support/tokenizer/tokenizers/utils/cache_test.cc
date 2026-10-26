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
//
// Tests for the C++ port of `src/utils/cache.rs` from the HuggingFace
// `tokenizers` crate v0.23.2.

#include "support/tokenizer/tokenizers/utils/cache.h"

#include <cstddef>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers {
namespace {

TEST(CacheTest, BasicGetSetAndHeterogeneousLookup) {
  Cache<std::string, int> cache;
  EXPECT_EQ(cache.capacity(), kDefaultCacheCapacity);
  EXPECT_EQ(kDefaultCacheCapacity, 10000u);
  EXPECT_EQ(kMaxLength, 256u);

  EXPECT_EQ(cache.Get(absl::string_view("hello")), std::nullopt);
  cache.Set("hello", 42);
  EXPECT_EQ(cache.Get(std::string("hello")), std::optional<int>(42));
  EXPECT_EQ(cache.Get(absl::string_view("hello")), std::optional<int>(42));

  // Equality always returns true, matching Rust `PartialEq for Cache`.
  Cache<std::string, int> other(5);
  EXPECT_EQ(cache, other);
  EXPECT_FALSE(cache != other);
}

TEST(CacheTest, CapacityLimitAndSetValues) {
  Cache<std::string, int> cache(3);
  std::vector<std::pair<std::string, int>> batch = {
      {"a", 1},
      {"b", 2},
      {"c", 3},
      {"d", 4},
  };
  cache.SetValues(batch);

  // Only the first `capacity` (= 3) entries are inserted.
  EXPECT_EQ(cache.Get("a"), std::optional<int>(1));
  EXPECT_EQ(cache.Get("b"), std::optional<int>(2));
  EXPECT_EQ(cache.Get("c"), std::optional<int>(3));
  EXPECT_EQ(cache.Get("d"), std::nullopt);

  // At capacity, additional `Set` calls are no-ops.
  cache.Set("e", 5);
  EXPECT_EQ(cache.Get("e"), std::nullopt);

  std::vector<std::string> keys = {"a", "d", "c"};
  std::optional<std::vector<std::optional<int>>> got = cache.GetValues(keys);
  ASSERT_TRUE(got.has_value());
  const std::vector<std::optional<int>> expected = {1, std::nullopt, 3};
  EXPECT_EQ(*got, expected);
}

TEST(CacheTest, ClearResizeFreshCopyAndMove) {
  Cache<std::string, int> cache(4);
  cache.Set("x", 10);
  cache.Set("y", 20);

  // Fresh() and copy construction produce an empty cache with the same
  // capacity.
  Cache<std::string, int> fresh = cache.Fresh();
  EXPECT_EQ(fresh.capacity(), 4u);
  EXPECT_EQ(fresh.Get("x"), std::nullopt);

  Cache<std::string, int> copied(cache);
  EXPECT_EQ(copied.capacity(), 4u);
  EXPECT_EQ(copied.Get("x"), std::nullopt);

  // Move preserves entries.
  Cache<std::string, int> moved(std::move(cache));
  EXPECT_EQ(moved.capacity(), 4u);
  EXPECT_EQ(moved.Get("x"), std::optional<int>(10));

  moved.Resize(2);
  EXPECT_EQ(moved.capacity(), 2u);
  EXPECT_EQ(moved.Get("x"), std::optional<int>(10));
  // Now at capacity 2 with 2 entries ("x", "y"), so a new entry is ignored.
  moved.Set("z", 30);
  EXPECT_EQ(moved.Get("z"), std::nullopt);

  moved.Clear();
  EXPECT_EQ(moved.Get("x"), std::nullopt);
  moved.Set("z", 30);
  EXPECT_EQ(moved.Get("z"), std::optional<int>(30));
}

TEST(CacheTest, ConcurrentGetSetClear) {
  Cache<int, int> cache(128);
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 500; ++i) {
        const int key = (t * 1000 + i) % 200;
        cache.Set(key, key * 2);
        if (std::optional<int> v = cache.Get(key); v.has_value()) {
          EXPECT_EQ(*v, key * 2);
        }
        if (i % 127 == 0) {
          cache.Clear();
        }
      }
    });
  }
  for (auto& th : threads) th.join();
}

}  // namespace
}  // namespace tokenizers
