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

#include "runtime/components/constrained_decoding/logit_mask.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

namespace {

template <typename T>
absl::Status ApplyBitmapImpl(absl::Span<const uint64_t> words, int vocab_size,
                             absl::Span<T> logits) {
  T min_val;
  if constexpr (std::is_same_v<T, float>) {
    min_val = -std::numeric_limits<float>::infinity();
  } else {
    min_val = tflite::half::min();
  }

  const int limit = std::min<int>(logits.size(), vocab_size);
  const int num_words = words.size();

  for (int word_idx = 0; word_idx < num_words; ++word_idx) {
    const int base_token = word_idx * 64;
    if (base_token >= limit) break;

    const uint64_t word = words[word_idx];
    if (word == ~uint64_t{0} && base_token + 64 <= limit) {
      // All 64 tokens in this word are allowed, skip to next word.
      continue;
    }
    if (word == 0) {
      // All tokens in this word are disallowed.
      const int count = std::min(64, limit - base_token);
      for (int bit = 0; bit < count; ++bit) {
        logits[base_token + bit] = min_val;
      }
      continue;
    }

    const uint64_t disallowed_bits = ~word;
    const int count = std::min(64, limit - base_token);
    for (int bit = 0; bit < count; ++bit) {
      if ((disallowed_bits >> bit) & 1) {
        logits[base_token + bit] = min_val;
      }
    }
  }

  // Any logits beyond vocab_size (e.g. padding) are disallowed.
  for (int i = vocab_size; i < static_cast<int>(logits.size()); ++i) {
    logits[i] = min_val;
  }

  return absl::OkStatus();
}

template <typename T>
absl::Status ApplySparseImpl(absl::Span<const SparseLogitMask::Entry> entries,
                             absl::Span<T> logits) {
  for (const auto& entry : entries) {
    if (entry.token_id < 0 ||
        entry.token_id >= static_cast<int>(logits.size())) {
      continue;
    }
    float val = static_cast<float>(logits[entry.token_id]);
    if constexpr (std::is_same_v<T, float>) {
      if ((std::isinf(val) && val < 0.0f) ||
          (val <= std::numeric_limits<float>::lowest())) {
        // Disallowed by a hard constraint, preserve -inf.
        continue;
      }
    } else {
      if (logits[entry.token_id] == tflite::half::min() ||
          (std::isinf(val) && val < 0.0f)) {
        // Disallowed by a hard constraint, preserve half::min().
        continue;
      }
    }
    if (entry.sign_dependent_weight) {
      if (val > 0.0f) {
        val = val * entry.weight + entry.bias;
      } else {
        val = (entry.weight != 0.0f ? (val / entry.weight) : val) + entry.bias;
      }
    } else {
      val = val * entry.weight + entry.bias;
    }
    logits[entry.token_id] = static_cast<T>(val);
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status ApplyCompositeImpl(
    absl::Span<const std::unique_ptr<LogitMask>> masks, absl::Span<T> logits) {
  if (masks.empty()) {
    return absl::OkStatus();
  }

  std::vector<const BitmapLogitMask*> bitmap_masks;
  std::vector<const SparseLogitMask*> sparse_masks;
  std::vector<const LogitMask*> other_masks;

  bitmap_masks.reserve(masks.size());
  sparse_masks.reserve(masks.size());
  other_masks.reserve(masks.size());

  for (const auto& mask : masks) {
    if (!mask) continue;
    switch (mask->GetType()) {
      case MaskType::kBitmap:
        bitmap_masks.push_back(static_cast<const BitmapLogitMask*>(mask.get()));
        break;
      case MaskType::kSparse:
        sparse_masks.push_back(static_cast<const SparseLogitMask*>(mask.get()));
        break;
      default:
        other_masks.push_back(mask.get());
        break;
    }
  }

  // 1 & 2. Fused bitmap masks application
  if (!bitmap_masks.empty()) {
    T min_val;
    if constexpr (std::is_same_v<T, float>) {
      min_val = -std::numeric_limits<float>::infinity();
    } else {
      min_val = tflite::half::min();
    }

    int min_vocab_size = bitmap_masks[0]->vocab_size();
    for (size_t i = 1; i < bitmap_masks.size(); ++i) {
      min_vocab_size = std::min(min_vocab_size, bitmap_masks[i]->vocab_size());
    }

    const int limit = std::min<int>(logits.size(), min_vocab_size);
    const int num_words = (min_vocab_size > 0) ? (min_vocab_size + 63) / 64 : 0;

    for (int word_idx = 0; word_idx < num_words; ++word_idx) {
      const int base_token = word_idx * 64;
      if (base_token >= limit) break;

      uint64_t fused_word = ~uint64_t{0};
      for (const auto* bm : bitmap_masks) {
        fused_word &= bm->words()[word_idx];
        if (fused_word == 0) {
          break;
        }
      }

      if (fused_word == ~uint64_t{0} && base_token + 64 <= limit) {
        continue;
      }
      if (fused_word == 0) {
        const int count = std::min(64, limit - base_token);
        for (int bit = 0; bit < count; ++bit) {
          logits[base_token + bit] = min_val;
        }
        continue;
      }

      const uint64_t disallowed_bits = ~fused_word;
      const int count = std::min(64, limit - base_token);
      for (int bit = 0; bit < count; ++bit) {
        if ((disallowed_bits >> bit) & 1) {
          logits[base_token + bit] = min_val;
        }
      }
    }

    for (int i = min_vocab_size; i < static_cast<int>(logits.size()); ++i) {
      logits[i] = min_val;
    }
  }

  // 3. Other/custom/nested masks
  for (const auto* mask : other_masks) {
    auto status = mask->Apply(logits);
    if (!status.ok()) {
      return status;
    }
  }

  // 4. Sparse masks
  for (const auto* sm : sparse_masks) {
    auto status = sm->Apply(logits);
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

}  // namespace

// =============================================================================
// BitmapLogitMask Implementation
// =============================================================================

BitmapLogitMask::BitmapLogitMask(int vocab_size, std::vector<uint64_t> words)
    : vocab_size_(vocab_size), words_(std::move(words)) {
  size_t required_words = (vocab_size_ > 0) ? (vocab_size_ + 63) / 64 : 0;
  if (words_.size() < required_words) {
    words_.resize(required_words, 0);
  }
}

BitmapLogitMask::BitmapLogitMask(int vocab_size,
                                 absl::Span<const uint64_t> words)
    : BitmapLogitMask(vocab_size,
                      std::vector<uint64_t>(words.begin(), words.end())) {}

BitmapLogitMask::BitmapLogitMask(int vocab_size,
                                 absl::Span<const uint32_t> words_u32)
    : vocab_size_(vocab_size) {
  size_t required_words = (vocab_size_ > 0) ? (vocab_size_ + 63) / 64 : 0;
  words_.resize(required_words, 0);
  for (size_t i = 0; i < words_u32.size(); ++i) {
    size_t u64_idx = i / 2;
    if (u64_idx >= required_words) break;
    if (i % 2 == 0) {
      words_[u64_idx] |= static_cast<uint64_t>(words_u32[i]);
    } else {
      words_[u64_idx] |= (static_cast<uint64_t>(words_u32[i]) << 32);
    }
  }
}

BitmapLogitMask::BitmapLogitMask(int vocab_size,
                                 const std::vector<bool>& allowed_mask)
    : vocab_size_(vocab_size) {
  size_t required_words = (vocab_size_ > 0) ? (vocab_size_ + 63) / 64 : 0;
  words_.resize(required_words, 0);
  size_t limit = std::min<size_t>(vocab_size_, allowed_mask.size());
  for (size_t i = 0; i < limit; ++i) {
    if (allowed_mask[i]) {
      words_[i / 64] |= (uint64_t{1} << (i % 64));
    }
  }
}

BitmapLogitMask::BitmapLogitMask(int vocab_size,
                                 absl::Span<const bool> allowed_mask)
    : vocab_size_(vocab_size) {
  size_t required_words = (vocab_size_ > 0) ? (vocab_size_ + 63) / 64 : 0;
  words_.resize(required_words, 0);
  size_t limit = std::min<size_t>(vocab_size_, allowed_mask.size());
  for (size_t i = 0; i < limit; ++i) {
    if (allowed_mask[i]) {
      words_[i / 64] |= (uint64_t{1} << (i % 64));
    }
  }
}

bool BitmapLogitMask::IsAllowed(int token_id) const {
  if (token_id < 0 || token_id >= vocab_size_) {
    return false;
  }
  size_t word_idx = token_id / 64;
  if (word_idx >= words_.size()) {
    return false;
  }
  return (words_[word_idx] & (uint64_t{1} << (token_id % 64))) != 0;
}

absl::Status BitmapLogitMask::Apply(absl::Span<float> logits) const {
  return ApplyBitmapImpl(words_, vocab_size_, logits);
}

absl::Status BitmapLogitMask::Apply(absl::Span<tflite::half> logits) const {
  return ApplyBitmapImpl(words_, vocab_size_, logits);
}

std::unique_ptr<BitmapLogitMask> BitmapLogitMask::CreateAllAllowed(
    int vocab_size) {
  size_t num_words = (vocab_size > 0) ? (vocab_size + 63) / 64 : 0;
  std::vector<uint64_t> words(num_words, ~uint64_t{0});
  if (vocab_size > 0 && vocab_size % 64 != 0 && num_words > 0) {
    size_t valid_bits = vocab_size % 64;
    words.back() = (uint64_t{1} << valid_bits) - 1;
  }
  return std::make_unique<BitmapLogitMask>(vocab_size, std::move(words));
}

std::unique_ptr<BitmapLogitMask> BitmapLogitMask::CreateAllDisallowed(
    int vocab_size) {
  size_t num_words = (vocab_size > 0) ? (vocab_size + 63) / 64 : 0;
  std::vector<uint64_t> words(num_words, 0);
  return std::make_unique<BitmapLogitMask>(vocab_size, std::move(words));
}

std::unique_ptr<BitmapLogitMask> BitmapLogitMask::CreateSingleAllowedToken(
    int vocab_size, int token_id) {
  size_t num_words = (vocab_size > 0) ? (vocab_size + 63) / 64 : 0;
  std::vector<uint64_t> words(num_words, 0);
  if (token_id >= 0 && token_id < vocab_size) {
    words[token_id / 64] |= (uint64_t{1} << (token_id % 64));
  }
  return std::make_unique<BitmapLogitMask>(vocab_size, std::move(words));
}

std::unique_ptr<BitmapLogitMask> BitmapLogitMask::CreateFromDisallowedTokens(
    int vocab_size, absl::Span<const int> disallowed_tokens) {
  size_t num_words = (vocab_size > 0) ? (vocab_size + 63) / 64 : 0;
  std::vector<uint64_t> words(num_words, ~uint64_t{0});
  if (vocab_size > 0 && vocab_size % 64 != 0 && num_words > 0) {
    size_t valid_bits = vocab_size % 64;
    words.back() = (uint64_t{1} << valid_bits) - 1;
  }
  for (int token_id : disallowed_tokens) {
    if (token_id >= 0 && token_id < vocab_size) {
      words[token_id / 64] &= ~(uint64_t{1} << (token_id % 64));
    }
  }
  return std::make_unique<BitmapLogitMask>(vocab_size, std::move(words));
}

std::unique_ptr<BitmapLogitMask> BitmapLogitMask::CreateFromAllowedTokens(
    int vocab_size, absl::Span<const int> allowed_tokens) {
  size_t num_words = (vocab_size > 0) ? (vocab_size + 63) / 64 : 0;
  std::vector<uint64_t> words(num_words, 0);
  for (int token_id : allowed_tokens) {
    if (token_id >= 0 && token_id < vocab_size) {
      words[token_id / 64] |= (uint64_t{1} << (token_id % 64));
    }
  }
  return std::make_unique<BitmapLogitMask>(vocab_size, std::move(words));
}

// =============================================================================
// SparseLogitMask Implementation
// =============================================================================

absl::Status SparseLogitMask::Apply(absl::Span<float> logits) const {
  return ApplySparseImpl(entries_, logits);
}

absl::Status SparseLogitMask::Apply(absl::Span<tflite::half> logits) const {
  return ApplySparseImpl(entries_, logits);
}

std::unique_ptr<SparseLogitMask> SparseLogitMask::CreateBiases(
    absl::Span<const std::pair<int, float>> token_biases) {
  std::vector<Entry> entries;
  entries.reserve(token_biases.size());
  for (const auto& [token_id, bias] : token_biases) {
    entries.push_back({.token_id = token_id, .weight = 1.0f, .bias = bias});
  }
  return std::make_unique<SparseLogitMask>(std::move(entries));
}

std::unique_ptr<SparseLogitMask> SparseLogitMask::CreateWeights(
    absl::Span<const std::pair<int, float>> token_weights) {
  std::vector<Entry> entries;
  entries.reserve(token_weights.size());
  for (const auto& [token_id, weight] : token_weights) {
    entries.push_back({.token_id = token_id, .weight = weight, .bias = 0.0f});
  }
  return std::make_unique<SparseLogitMask>(std::move(entries));
}

// =============================================================================
// CompositeLogitMask Implementation
// =============================================================================

absl::Status CompositeLogitMask::Apply(absl::Span<float> logits) const {
  return ApplyCompositeImpl(masks_, logits);
}

absl::Status CompositeLogitMask::Apply(absl::Span<tflite::half> logits) const {
  return ApplyCompositeImpl(masks_, logits);
}

}  // namespace litert::lm
