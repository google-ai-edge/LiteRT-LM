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

#include "runtime/executor/gpu_artisan/utils/constraint_adapter.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace odml::infra::llm_utils {
namespace {

using ::testing::_;
using ::testing::Return;
using ::testing::status::StatusIs;

class MockBitmap : public litert::lm::Bitmap {
 public:
  MOCK_METHOD(bool, Get, (int index), (const, override));
};

class MockState : public litert::lm::Constraint::State {
 public:
  explicit MockState(int id) : id_(id) {}
  int id() const { return id_; }

 private:
  int id_;
};

class MockConstraint : public litert::lm::Constraint {
 public:
  MOCK_METHOD(std::unique_ptr<State>, Start, (), (const, override));
  MOCK_METHOD(bool, IsEnded, (const State& state), (const, override));
  MOCK_METHOD(int, GetVocabularySize, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<State>>, ComputeNext,
              (const State& state, int token), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<litert::lm::Bitmap>>,
              ComputeBitmap, (const State& state), (const, override));
};

TEST(ConstraintAdapterTest, AdaptsConstraintAndBitmap) {
  AllAllowedBitmap all_allowed;
  EXPECT_TRUE(all_allowed.Get(0));
  EXPECT_TRUE(all_allowed.Get(42));

  testing::StrictMock<MockConstraint> mock_constraint;

  EXPECT_CALL(mock_constraint, Start()).WillOnce([]() {
    return std::make_unique<MockState>(0);
  });

  EXPECT_CALL(mock_constraint, IsEnded(_))
      .WillOnce([](const litert::lm::Constraint::State& state) {
        EXPECT_EQ(static_cast<const MockState&>(state).id(), 0);
        return false;
      })
      .WillOnce([](const litert::lm::Constraint::State& state) {
        EXPECT_EQ(static_cast<const MockState&>(state).id(), 1);
        return true;
      });

  EXPECT_CALL(mock_constraint, ComputeBitmap(_))
      .WillOnce([](const litert::lm::Constraint::State& state) {
        EXPECT_EQ(static_cast<const MockState&>(state).id(), 0);
        auto mock_bitmap = std::make_unique<testing::StrictMock<MockBitmap>>();
        EXPECT_CALL(*mock_bitmap, Get(1)).WillOnce(Return(true));
        EXPECT_CALL(*mock_bitmap, Get(2)).WillOnce(Return(false));
        return absl::StatusOr<std::unique_ptr<litert::lm::Bitmap>>(
            std::move(mock_bitmap));
      });

  EXPECT_CALL(mock_constraint, ComputeNext(_, 1))
      .WillOnce([](const litert::lm::Constraint::State& state, int token) {
        EXPECT_EQ(static_cast<const MockState&>(state).id(), 0);
        return absl::StatusOr<std::unique_ptr<litert::lm::Constraint::State>>(
            std::make_unique<MockState>(1));
      });

  ConstraintAdapter adapter(&mock_constraint);
  EXPECT_EQ(adapter.GetConstraint(), &mock_constraint);

  auto state = adapter.Start();
  EXPECT_FALSE(adapter.IsEnded(*state));

  ASSERT_OK_AND_ASSIGN(auto bitmap, adapter.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(1));
  EXPECT_FALSE(bitmap->Get(2));

  ASSERT_OK_AND_ASSIGN(auto next_state, adapter.ComputeNext(*state, 1));
  EXPECT_TRUE(adapter.IsEnded(*next_state));
}

TEST(ConstraintAdapterTest, PropagatesErrors) {
  testing::StrictMock<MockConstraint> mock_constraint;
  EXPECT_CALL(mock_constraint, Start()).WillOnce([]() {
    return std::make_unique<MockState>(0);
  });
  EXPECT_CALL(mock_constraint, ComputeBitmap(_))
      .WillOnce(Return(absl::InternalError("bitmap failure")));
  EXPECT_CALL(mock_constraint, ComputeNext(_, 7))
      .WillOnce(Return(absl::InvalidArgumentError("bad token")));

  ConstraintAdapter adapter(&mock_constraint);
  auto state = adapter.Start();
  EXPECT_THAT(adapter.ComputeBitmap(*state),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(adapter.ComputeNext(*state, 7),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace odml::infra::llm_utils
