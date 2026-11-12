// Copyright 2024 The MediaPipe Authors.
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

#include "runtime/executor/gpu_artisan/ml_drift/group_size.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"

namespace odml::infra::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(GroupSizeTest, GetGroupSizesEmptyGroup) {
  EXPECT_THAT(GetGroupSizesForInput(1, {}), IsOkAndHolds(ElementsAre()));
}

TEST(GroupSizeTest, GetGroupSizesSingleGroup) {
  EXPECT_THAT(GetGroupSizesForInput(1, {GroupSize{1, 0.0f}}),
              IsOkAndHolds(ElementsAre(1)));
  EXPECT_THAT(GetGroupSizesForInput(5, {GroupSize{1, 0.0f}}),
              IsOkAndHolds(ElementsAre(1, 1, 1, 1, 1)));
  EXPECT_THAT(GetGroupSizesForInput(6, {GroupSize{5, 0.0f}}),
              IsOkAndHolds(ElementsAre(5, 5)));
}

TEST(GroupSizeTest, GetGroupSizesTwoGroups) {
  std::vector<GroupSize> groups = {GroupSize{5, 3.9f}, GroupSize{1, 1.0f}};
  EXPECT_THAT(GetGroupSizesForInput(1, groups), IsOkAndHolds(ElementsAre(1)));
  EXPECT_THAT(GetGroupSizesForInput(2, groups),
              IsOkAndHolds(ElementsAre(1, 1)));
  EXPECT_THAT(GetGroupSizesForInput(3, groups),
              IsOkAndHolds(ElementsAre(1, 1, 1)));
  EXPECT_THAT(GetGroupSizesForInput(4, groups), IsOkAndHolds(ElementsAre(5)));
  EXPECT_THAT(GetGroupSizesForInput(5, groups), IsOkAndHolds(ElementsAre(5)));
  EXPECT_THAT(GetGroupSizesForInput(6, groups),
              IsOkAndHolds(ElementsAre(5, 1)));
  EXPECT_THAT(GetGroupSizesForInput(8, groups),
              IsOkAndHolds(ElementsAre(5, 1, 1, 1)));
  EXPECT_THAT(GetGroupSizesForInput(9, groups),
              IsOkAndHolds(ElementsAre(5, 5)));
  EXPECT_THAT(GetGroupSizesForInput(22, groups),
              IsOkAndHolds(ElementsAre(5, 5, 5, 5, 1, 1)));
}

TEST(GroupSizeTest, GetGroupSizesManyGroups) {
  std::vector<GroupSize> groups = {GroupSize{100, 49.9f}, GroupSize{10, 7.0f},
                       GroupSize{5, 3.9f}, GroupSize{1, 1.0f}};
  EXPECT_THAT(GetGroupSizesForInput(1, groups), IsOkAndHolds(ElementsAre(1)));
  EXPECT_THAT(GetGroupSizesForInput(3, groups),
              IsOkAndHolds(ElementsAre(1, 1, 1)));
  EXPECT_THAT(GetGroupSizesForInput(4, groups), IsOkAndHolds(ElementsAre(5)));
  EXPECT_THAT(GetGroupSizesForInput(6, groups),
              IsOkAndHolds(ElementsAre(5, 1)));
  EXPECT_THAT(GetGroupSizesForInput(8, groups),
              IsOkAndHolds(ElementsAre(5, 1, 1, 1)));
  EXPECT_THAT(GetGroupSizesForInput(9, groups), IsOkAndHolds(ElementsAre(10)));
  EXPECT_THAT(GetGroupSizesForInput(16, groups),
              IsOkAndHolds(ElementsAre(10, 5, 1)));
  EXPECT_THAT(GetGroupSizesForInput(22, groups),
              IsOkAndHolds(ElementsAre(10, 10, 1, 1)));
  EXPECT_THAT(GetGroupSizesForInput(36, groups),
              IsOkAndHolds(ElementsAre(10, 10, 10, 5, 1)));
  EXPECT_THAT(GetGroupSizesForInput(39, groups),
              IsOkAndHolds(ElementsAre(10, 10, 10, 10)));
  EXPECT_THAT(GetGroupSizesForInput(55, groups),
              IsOkAndHolds(ElementsAre(10, 10, 10, 10, 10, 5)));
  EXPECT_THAT(GetGroupSizesForInput(70, groups),
              IsOkAndHolds(ElementsAre(10, 10, 10, 10, 10, 10, 10)));
  EXPECT_THAT(GetGroupSizesForInput(71, groups),
              IsOkAndHolds(ElementsAre(100)));
  EXPECT_THAT(GetGroupSizesForInput(231, groups),
              IsOkAndHolds(ElementsAre(100, 100, 10, 10, 10, 1)));
}

TEST(GroupSizeTest, GetGroupSizesInvalidGroups) {
  EXPECT_THAT(
      GetGroupSizesForInput(1, {GroupSize{1, 0.0f}, GroupSize{2, 0.0f}}),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(GetGroupSizesForInput(1, {GroupSize{3, 0.0f}, GroupSize{2, 0.0f},
                                        GroupSize{3, 0.0f}}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(GetGroupSizesForInput(1, {GroupSize{3, 0.0f}, GroupSize{2, 0.0f},
                                        GroupSize{2, 0.0f}}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      GetGroupSizesForInput(1, {GroupSize{2, 10.0f}, GroupSize{1, 1.0f}}),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace odml::infra::gpu
