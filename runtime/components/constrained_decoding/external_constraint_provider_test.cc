// Copyright 2025 The ODML Authors.
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

#include "runtime/components/constrained_decoding/external_constraint_provider.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/external_constraint_config.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/constrained_decoding/fst_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

TEST(ExternalConstraintProviderTest, Create) {
  auto provider = std::make_unique<ExternalConstraintProvider>();
  EXPECT_NE(provider, nullptr);
}

TEST(ExternalConstraintProviderTest, CreateConstraintSuccess) {
  auto provider = std::make_unique<ExternalConstraintProvider>();

  std::vector<int> tokens = {1, 2, 3};
  auto fake_constraint =
      std::make_unique<FakeConstraint>(tokens, /*vocab_size=*/10);
  Constraint* raw_fake_constraint = fake_constraint.get();

  ExternalConstraintArg external_constraint{std::move(fake_constraint)};
  ASSERT_OK_AND_ASSIGN(auto constraint,
                       provider->CreateConstraint(
                           ConstraintArg(std::move(external_constraint))));

  EXPECT_EQ(constraint.get(), raw_fake_constraint);
}

TEST(ExternalConstraintProviderTest, CreateConstraintFailsWithWrongArg) {
  auto provider = std::make_unique<ExternalConstraintProvider>();

  EXPECT_THAT(
      provider->CreateConstraint(LlGuidanceConstraintArg{}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "ExternalConstraintProvider only supports ExternalConstraintArg."));

  EXPECT_THAT(
      provider->CreateConstraint(FstConstraintArg{}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "ExternalConstraintProvider only supports ExternalConstraintArg."));
}

}  // namespace
}  // namespace litert::lm
