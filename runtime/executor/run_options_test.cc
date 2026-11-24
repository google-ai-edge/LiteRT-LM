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

#include "runtime/executor/run_options.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_options.h"  // from @litert
#include "runtime/executor/run_options_test_util.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::status::StatusIs;

TEST(RunOptionsTest, EmptyFactoryCreatesNoOptions) {
  ASSERT_OK_AND_ASSIGN(std::optional<::litert::Options> run_options,
                       CreateRunOptions(RunOptionsFactory()));
  EXPECT_FALSE(run_options.has_value());
  EXPECT_THAT(GetRunOptionsPtr(run_options), IsNull());
}

TEST(RunOptionsTest, FactoryCreatesNewOptionsForEveryCall) {
  int num_calls = 0;
  const RunOptionsFactory factory =
      [&num_calls]() -> absl::StatusOr<::litert::Options> {
    ++num_calls;
    return ::litert::Options();
  };

  ASSERT_OK_AND_ASSIGN(std::optional<::litert::Options> first,
                       CreateRunOptions(factory));
  ASSERT_OK_AND_ASSIGN(std::optional<::litert::Options> second,
                       CreateRunOptions(factory));

  EXPECT_EQ(num_calls, 2);
  ASSERT_THAT(GetRunOptionsPtr(first), NotNull());
  ASSERT_THAT(GetRunOptionsPtr(second), NotNull());
  EXPECT_NE(GetRunOptionsPtr(first), GetRunOptionsPtr(second));
}

TEST(RunOptionsTest, FactoryErrorIsReturned) {
  EXPECT_THAT(CreateRunOptions(FailingRunOptionsFactory(
                  absl::InvalidArgumentError("bad run options"))),
              StatusIs(absl::StatusCode::kInvalidArgument, "bad run options"));
}

}  // namespace
}  // namespace litert::lm
