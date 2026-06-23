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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_PREFIX_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_PREFIX_CACHE_H_

#include <string>
#include <variant>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Represents a hash of a media input (image/audio) and its token length.
struct MediaHash {
  std::string hash;
  int token_length;

  bool operator==(const MediaHash& other) const {
    return hash == other.hash && token_length == other.token_length;
  }
  bool operator!=(const MediaHash& other) const { return !(*this == other); }
};

// An element in the cache: either a token ID or a media hash.
using CacheElement = std::variant<int, MediaHash>;

class PrefixCache {
 public:
  struct MatchResult {
    // How many CacheElements matched.
    int matched_elements = 0;

    // The equivalent number of tokens (steps) this match represents.
    int matched_tokens = 0;
  };

  PrefixCache() = default;
  ~PrefixCache() = default;

  // Finds the longest common prefix between the cached elements and the
  // incoming elements.
  MatchResult FindLongestCommonPrefix(
      const std::vector<CacheElement>& incoming_elements) const;

  // Appends new tokens to the cache (e.g., after prefill or decode).
  void AppendTokens(absl::Span<const int> tokens);

  // Appends a media hash to the cache.
  void AppendMedia(const MediaHash& media_hash);

  // Appends a single cache element (token or media).
  void AppendElement(const CacheElement& element);

  // Appends multiple cache elements.
  void AppendElements(absl::Span<const CacheElement> elements);

  // Truncates the cache to a specific element length (used during
  // rewinds/checkpoints).
  void Truncate(int element_length);

  // Clears the cache.
  void Clear();

  // Returns the current size in elements.
  int Size() const { return elements_.size(); }

  // Returns the equivalent total token length of the cache.
  int TokenLength() const { return total_token_length_; }

  // Returns the cached elements.
  const std::vector<CacheElement>& GetElements() const { return elements_; }

 private:
  std::vector<CacheElement> elements_;
  int total_token_length_ = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_PREFIX_CACHE_H_
