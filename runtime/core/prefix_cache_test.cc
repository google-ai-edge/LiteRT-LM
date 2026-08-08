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

#include "runtime/core/prefix_cache.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {
namespace {

TEST(PrefixCacheTest, InitialStateIsEmpty) {
  PrefixCache cache;
  EXPECT_EQ(cache.Size(), 0);
  EXPECT_EQ(cache.TokenLength(), 0);
}

TEST(PrefixCacheTest, AppendTokensIncreasesSizeAndLength) {
  PrefixCache cache;
  std::vector<int> tokens = {10, 20, 30};
  cache.AppendTokens(tokens);
  EXPECT_EQ(cache.Size(), 3);
  EXPECT_EQ(cache.TokenLength(), 3);
}

TEST(PrefixCacheTest, AppendElementWithMediaIncreasesSizeAndLength) {
  PrefixCache cache;
  MediaHash media_hash{123, 100};
  cache.AppendElement(media_hash);
  EXPECT_EQ(cache.Size(), 1);
  EXPECT_EQ(cache.TokenLength(), 100);
}

TEST(PrefixCacheTest, AppendElementAppendsCorrectVariant) {
  PrefixCache cache;

  cache.AppendElement(10);
  EXPECT_EQ(cache.Size(), 1);
  EXPECT_EQ(cache.TokenLength(), 1);

  MediaHash media_hash{456, 50};
  cache.AppendElement(media_hash);
  EXPECT_EQ(cache.Size(), 2);
  EXPECT_EQ(cache.TokenLength(), 51);
}

TEST(PrefixCacheTest, AppendElementsAppendsMultipleVariants) {
  PrefixCache cache;
  std::vector<CacheElement> elements = {
      10,
      MediaHash{1, 10},
      20,
  };
  cache.AppendElements(elements);
  EXPECT_EQ(cache.Size(), 3);
  EXPECT_EQ(cache.TokenLength(), 12);  // 1 + 10 + 1 = 12
}

TEST(PrefixCacheTest, FindLongestCommonPrefixHandlesEmptyCache) {
  PrefixCache cache;
  std::vector<CacheElement> incoming = {10, 20};
  PrefixCache::MatchResult result = cache.FindLongestCommonPrefix(incoming);
  EXPECT_EQ(result.matched_elements, 0);
  EXPECT_EQ(result.matched_tokens, 0);
}

TEST(PrefixCacheTest, FindLongestCommonPrefixHandlesFullMatch) {
  PrefixCache cache;
  cache.AppendTokens({10, 20, 30});

  std::vector<CacheElement> incoming = {10, 20, 30};
  PrefixCache::MatchResult result = cache.FindLongestCommonPrefix(incoming);
  EXPECT_EQ(result.matched_elements, 3);
  EXPECT_EQ(result.matched_tokens, 3);
}

TEST(PrefixCacheTest, FindLongestCommonPrefixHandlesPartialMatch) {
  PrefixCache cache;
  cache.AppendTokens({10, 20, 30});

  std::vector<CacheElement> incoming = {10, 20, 40, 50};
  PrefixCache::MatchResult result = cache.FindLongestCommonPrefix(incoming);
  EXPECT_EQ(result.matched_elements, 2);
  EXPECT_EQ(result.matched_tokens, 2);
}

TEST(PrefixCacheTest, FindLongestCommonPrefixHandlesMixedElements) {
  PrefixCache cache;
  cache.AppendElement(10);
  cache.AppendElement(MediaHash{1, 50});
  cache.AppendElement(20);

  std::vector<CacheElement> incoming = {10, MediaHash{1, 50}, 20, 30};
  PrefixCache::MatchResult result = cache.FindLongestCommonPrefix(incoming);
  EXPECT_EQ(result.matched_elements, 3);
  EXPECT_EQ(result.matched_tokens, 52);  // 1 + 50 + 1
}

TEST(PrefixCacheTest, FindLongestCommonPrefixMismatchedMediaHash) {
  PrefixCache cache;
  cache.AppendElement(10);
  cache.AppendElement(MediaHash{1, 50});
  cache.AppendElement(20);

  std::vector<CacheElement> incoming = {10, MediaHash{2, 50}, 20};
  PrefixCache::MatchResult result = cache.FindLongestCommonPrefix(incoming);
  EXPECT_EQ(result.matched_elements, 1);
  EXPECT_EQ(result.matched_tokens, 1);
}

TEST(PrefixCacheTest, FindLongestCommonPrefixMismatchedMediaTokenLength) {
  PrefixCache cache;
  cache.AppendElement(10);
  cache.AppendElement(MediaHash{1, 50});
  cache.AppendElement(20);

  std::vector<CacheElement> incoming = {
      10, MediaHash{1, 100},  // same hash, different token length
      20};
  PrefixCache::MatchResult result = cache.FindLongestCommonPrefix(incoming);
  EXPECT_EQ(result.matched_elements, 1);
  EXPECT_EQ(result.matched_tokens, 1);
}

TEST(PrefixCacheTest, TruncateReducesSizeAndLengthCorrectly) {
  PrefixCache cache;
  cache.AppendElement(10);
  cache.AppendElement(MediaHash{1, 50});
  cache.AppendElement(20);

  // Initial size 3, length 52
  ASSERT_EQ(cache.Size(), 3);
  ASSERT_EQ(cache.TokenLength(), 52);

  // Truncate to 2 elements (should keep 10 and media1)
  cache.Truncate(2);
  EXPECT_EQ(cache.Size(), 2);
  EXPECT_EQ(cache.TokenLength(), 51);  // 1 + 50

  // Truncate to 1 element (should keep only 10)
  cache.Truncate(1);
  EXPECT_EQ(cache.Size(), 1);
  EXPECT_EQ(cache.TokenLength(), 1);

  // Truncate to 0 elements (should clear all)
  cache.Truncate(0);
  EXPECT_EQ(cache.Size(), 0);
  EXPECT_EQ(cache.TokenLength(), 0);
}

TEST(PrefixCacheTest, TruncateNoOpIfLengthGreaterThanOrEqual) {
  PrefixCache cache;
  cache.AppendTokens({10, 20});

  cache.Truncate(2);
  EXPECT_EQ(cache.Size(), 2);
  EXPECT_EQ(cache.TokenLength(), 2);

  cache.Truncate(5);
  EXPECT_EQ(cache.Size(), 2);
  EXPECT_EQ(cache.TokenLength(), 2);
}

TEST(PrefixCacheTest, ClearResetsCache) {
  PrefixCache cache;
  cache.AppendTokens({10, 20});
  cache.AppendElement(MediaHash{1, 100});

  ASSERT_EQ(cache.Size(), 3);
  ASSERT_EQ(cache.TokenLength(), 102);

  cache.Clear();
  EXPECT_EQ(cache.Size(), 0);
  EXPECT_EQ(cache.TokenLength(), 0);
}

}  // namespace
}  // namespace litert::lm
