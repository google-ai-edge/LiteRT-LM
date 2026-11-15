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

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/mock_litert_lm_runner.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::omni {
namespace {

using ::absl_testing::IsOk;
using ::testing::_;
using ::testing::Return;

TEST(MockLiteRtLmRunnerTest, MockMethodsCanBeInvoked) {
  MockLiteRtLmRunner runner;

  EXPECT_CALL(runner, Reset()).WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(runner.Reset(), IsOk());

  EXPECT_CALL(runner, Prefill(_)).WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(runner.Prefill(lm::ExecutorInputs{}), IsOk());

  TensorBuffer fake_tb;
  EXPECT_CALL(runner, Decode(_)).WillOnce(Return(std::move(fake_tb)));
  EXPECT_THAT(runner.Decode(lm::ExecutorInputs{}), IsOk());
}

}  // namespace
}  // namespace litert::omni
