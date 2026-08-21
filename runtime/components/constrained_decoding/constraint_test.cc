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

#include "runtime/components/constrained_decoding/constraint.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

// A legacy constraint that only overrides ComputeBitmap.
class LegacyBitmapConstraint : public Constraint {
 public:
  class SimpleState : public State {};

  explicit LegacyBitmapConstraint(int vocab_size,
                                  std::vector<int> allowed_tokens)
      : vocab_size_(vocab_size), allowed_tokens_(std::move(allowed_tokens)) {}

  std::unique_ptr<State> Start() const override {
    return std::make_unique<SimpleState>();
  }

  bool IsEnded(const State& state) const override { return false; }

  int GetVocabularySize() const override { return vocab_size_; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    return std::make_unique<SimpleState>();
  }

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override {
    auto bitmap = std::make_unique<SingleAllowedTokenBitmap>(
        allowed_tokens_.empty() ? 0 : allowed_tokens_[0]);
    return bitmap;
  }

 private:
  int vocab_size_;
  std::vector<int> allowed_tokens_;
};

// A modern constraint that only overrides ComputeMask.
class ModernMaskConstraint : public Constraint {
 public:
  class SimpleState : public State {};

  explicit ModernMaskConstraint(int vocab_size, int allowed_token)
      : vocab_size_(vocab_size), allowed_token_(allowed_token) {}

  std::unique_ptr<State> Start() const override {
    return std::make_unique<SimpleState>();
  }

  bool IsEnded(const State& state) const override { return false; }

  int GetVocabularySize() const override { return vocab_size_; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    return std::make_unique<SimpleState>();
  }

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override {
    return BitmapLogitMask::CreateSingleAllowedToken(vocab_size_,
                                                     allowed_token_);
  }

 private:
  int vocab_size_;
  int allowed_token_;
};

// A constraint that returns an error in ComputeBitmap.
class ErrorBitmapConstraint : public Constraint {
 public:
  class SimpleState : public State {};

  std::unique_ptr<State> Start() const override {
    return std::make_unique<SimpleState>();
  }

  bool IsEnded(const State& state) const override { return false; }

  int GetVocabularySize() const override { return 10; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    return std::make_unique<SimpleState>();
  }

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override {
    return absl::InternalError("Bitmap computation failed");
  }
};

TEST(ConstraintTest, ComputeMaskDefaultDelegatesToComputeBitmap) {
  LegacyBitmapConstraint constraint(/*vocab_size=*/8, /*allowed_tokens=*/{3});
  auto state = constraint.Start();

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  ASSERT_NE(mask, nullptr);
  EXPECT_EQ(mask->GetType(), MaskType::kBitmap);

  auto* bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_EQ(bitmap_mask->vocab_size(), 8);
  EXPECT_FALSE(bitmap_mask->IsAllowed(0));
  EXPECT_FALSE(bitmap_mask->IsAllowed(1));
  EXPECT_FALSE(bitmap_mask->IsAllowed(2));
  EXPECT_TRUE(bitmap_mask->IsAllowed(3));
  EXPECT_FALSE(bitmap_mask->IsAllowed(4));
}

TEST(ConstraintTest, ComputeMaskPropagatesComputeBitmapError) {
  ErrorBitmapConstraint constraint;
  auto state = constraint.Start();

  EXPECT_THAT(
      constraint.ComputeMask(*state),
      StatusIs(absl::StatusCode::kInternal, "Bitmap computation failed"));
}

TEST(ConstraintTest, ComputeBitmapReturnsUnimplementedByDefault) {
  ModernMaskConstraint constraint(/*vocab_size=*/8, /*allowed_token=*/5);
  auto state = constraint.Start();

  EXPECT_THAT(constraint.ComputeBitmap(*state),
              StatusIs(absl::StatusCode::kUnimplemented));

  // ComputeMask works directly
  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  ASSERT_NE(mask, nullptr);
  auto* bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_TRUE(bitmap_mask->IsAllowed(5));
  EXPECT_FALSE(bitmap_mask->IsAllowed(0));
}

}  // namespace
}  // namespace litert::lm
