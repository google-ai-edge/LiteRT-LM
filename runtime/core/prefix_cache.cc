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

#include <algorithm>
#include <variant>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

namespace {

int GetTokenLength(const CacheElement& element) {
  if (std::holds_alternative<MediaHash>(element)) {
    return std::get<MediaHash>(element).token_length;
  }
  return 1;
}

}  // namespace

PrefixCache::MatchResult PrefixCache::FindLongestCommonPrefix(
    const std::vector<CacheElement>& incoming_elements) const {
  MatchResult result;
  int min_len = std::min(elements_.size(), incoming_elements.size());

  for (int i = 0; i < min_len; ++i) {
    if (elements_[i] != incoming_elements[i]) {
      break;
    }
    ++result.matched_elements;
    result.matched_tokens += GetTokenLength(elements_[i]);
  }
  return result;
}

void PrefixCache::AppendTokens(absl::Span<const int> tokens) {
  for (int token : tokens) {
    elements_.push_back(token);
    total_token_length_ += 1;
  }
}

void PrefixCache::AppendMedia(const MediaHash& media_hash) {
  elements_.push_back(media_hash);
  total_token_length_ += media_hash.token_length;
}

void PrefixCache::AppendElement(const CacheElement& element) {
  elements_.push_back(element);
  total_token_length_ += GetTokenLength(element);
}

void PrefixCache::AppendElements(absl::Span<const CacheElement> elements) {
  for (const auto& element : elements) {
    AppendElement(element);
  }
}

void PrefixCache::Truncate(int element_length) {
  if (element_length >= elements_.size()) {
    return;
  }

  // Calculate new total token length
  int new_total_token_length = 0;
  for (int i = 0; i < element_length; ++i) {
    new_total_token_length += GetTokenLength(elements_[i]);
  }

  elements_.resize(element_length);
  total_token_length_ = new_total_token_length;
}

void PrefixCache::Clear() {
  elements_.clear();
  total_token_length_ = 0;
}

}  // namespace litert::lm
