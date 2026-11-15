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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LOGIT_MASK_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LOGIT_MASK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

enum class MaskType {
  kBitmap,
  kSparse,
  kComposite,
  kCustom,
};

// Base class for immutable logit masks that can be applied to a logits span.
class LogitMask {
 public:
  virtual ~LogitMask() = default;

  // Returns the concrete type tag of this mask.
  virtual MaskType GetType() const { return MaskType::kCustom; }

  // Applies the mask in-place to the given float logits span.
  virtual absl::Status Apply(absl::Span<float> logits) const = 0;

  // Applies the mask in-place to the given half-precision logits span.
  virtual absl::Status Apply(absl::Span<tflite::half> logits) const = 0;
};

// Immutable hard constraint mask represented as a bitset of allowed tokens.
// Bits set to 1 indicate allowed tokens; bits set to 0 indicate disallowed
// tokens. Applying the mask sets disallowed tokens to -inf (or half::min()).
class BitmapLogitMask : public LogitMask {
 public:
  // Constructs a BitmapLogitMask with given vocab size and 64-bit word bitset.
  explicit BitmapLogitMask(int vocab_size, std::vector<uint64_t> words);

  // Constructs a BitmapLogitMask copying words from a span.
  explicit BitmapLogitMask(int vocab_size, absl::Span<const uint64_t> words);

  // Constructs a BitmapLogitMask from 32-bit words (e.g. LLGuidance mask).
  explicit BitmapLogitMask(int vocab_size,
                           absl::Span<const uint32_t> words_u32);

  // Constructs a BitmapLogitMask from a std::vector<bool>.
  explicit BitmapLogitMask(int vocab_size,
                           const std::vector<bool>& allowed_mask);

  // Constructs a BitmapLogitMask from a boolean span.
  explicit BitmapLogitMask(int vocab_size, absl::Span<const bool> allowed_mask);

  MaskType GetType() const override { return MaskType::kBitmap; }

  int vocab_size() const { return vocab_size_; }
  absl::Span<const uint64_t> words() const { return words_; }

  // Returns true if token_id is within vocab_size and allowed by the mask.
  bool IsAllowed(int token_id) const;

  absl::Status Apply(absl::Span<float> logits) const override;
  absl::Status Apply(absl::Span<tflite::half> logits) const override;

  // Factory methods:
  static std::unique_ptr<BitmapLogitMask> CreateAllAllowed(int vocab_size);
  static std::unique_ptr<BitmapLogitMask> CreateAllDisallowed(int vocab_size);
  static std::unique_ptr<BitmapLogitMask> CreateSingleAllowedToken(
      int vocab_size, int token_id);
  static std::unique_ptr<BitmapLogitMask> CreateFromDisallowedTokens(
      int vocab_size, absl::Span<const int> disallowed_tokens);
  static std::unique_ptr<BitmapLogitMask> CreateFromAllowedTokens(
      int vocab_size, absl::Span<const int> allowed_tokens);

 private:
  int vocab_size_;
  std::vector<uint64_t> words_;
};

// Immutable soft constraint mask representing sparse token adjustments:
// z'_i = z_i * w_i + b_i
class SparseLogitMask : public LogitMask {
 public:
  struct Entry {
    int token_id;
    float weight = 1.0f;  // Multiplicative factor (default: 1.0)
    float bias = 0.0f;    // Additive bias (default: 0.0)
    // When true, positive logits are multiplied by weight (e.g. z * (1/p)), and
    // negative logits are divided by weight (e.g. z / (1/p) = z * p).
    bool sign_dependent_weight = false;

    bool operator==(const Entry& other) const {
      return token_id == other.token_id && weight == other.weight &&
             bias == other.bias &&
             sign_dependent_weight == other.sign_dependent_weight;
    }
  };

  explicit SparseLogitMask(std::vector<Entry> entries)
      : entries_(std::move(entries)) {}

  MaskType GetType() const override { return MaskType::kSparse; }

  absl::Span<const Entry> entries() const { return entries_; }

  absl::Status Apply(absl::Span<float> logits) const override;
  absl::Status Apply(absl::Span<tflite::half> logits) const override;

  // Factory methods for convenience:
  static std::unique_ptr<SparseLogitMask> CreateBiases(
      absl::Span<const std::pair<int, float>> token_biases);
  static std::unique_ptr<SparseLogitMask> CreateWeights(
      absl::Span<const std::pair<int, float>> token_weights);

 private:
  std::vector<Entry> entries_;
};

// Composite mask that explicitly chains and evaluates multiple masks:
// 1. Fuses all BitmapLogitMask bitsets via bitwise AND on-the-fly.
// 2. Applies b_i = -inf to disallowed indices in a single pass over logits.
// 3. Applies other / custom masks.
// 4. Applies sparse weight/bias adjustments.
class CompositeLogitMask : public LogitMask {
 public:
  CompositeLogitMask() = default;
  explicit CompositeLogitMask(std::vector<std::unique_ptr<LogitMask>> masks)
      : masks_(std::move(masks)) {}

  MaskType GetType() const override { return MaskType::kComposite; }

  void AddMask(std::unique_ptr<LogitMask> mask) {
    if (mask) {
      masks_.push_back(std::move(mask));
    }
  }

  absl::Span<const std::unique_ptr<LogitMask>> masks() const { return masks_; }
  size_t size() const { return masks_.size(); }
  bool empty() const { return masks_.empty(); }

  absl::Status Apply(absl::Span<float> logits) const override;
  absl::Status Apply(absl::Span<tflite::half> logits) const override;

 private:
  std::vector<std::unique_ptr<LogitMask>> masks_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LOGIT_MASK_H_
