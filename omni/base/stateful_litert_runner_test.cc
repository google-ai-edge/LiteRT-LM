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

#include "omni/base/stateful_litert_runner.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/base/mock_litert_runner.h"

namespace litert::omni {
namespace {

using ::absl_testing::IsOk;
using ::testing::_;
using ::testing::Return;

class StatefulLiteRtRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto env_status = Environment::Create({});
    ASSERT_TRUE(env_status.HasValue());
    env_ = std::make_unique<Environment>(std::move(*env_status));
  }

  absl::StatusOr<TensorBuffer> CreateFloatBuffer(const std::vector<int>& dims) {
    RankedTensorType type(ElementType::Float32,
                          Layout(Dimensions(dims.begin(), dims.end())));
    size_t num_elements = 1;
    for (int d : dims) num_elements *= d;
    LITERT_ASSIGN_OR_RETURN(
        auto buf, TensorBuffer::CreateManaged(
                      *env_, TensorBufferType::kHostMemory, std::move(type),
                      num_elements * sizeof(float)));
    return buf;
  }

  std::unique_ptr<Environment> env_;
};

TEST_F(StatefulLiteRtRunnerTest, StepAndAutoCommit) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();

  // 1 non-state input, 1 state input
  // 1 non-state output, 1 state output
  EXPECT_CALL(*mock_runner, CreateInputBuffers(absl::string_view("main")))
      .WillRepeatedly([this](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        auto b1 = CreateFloatBuffer({1, 4});
        auto b2 = CreateFloatBuffer({1, 4});
        if (b1.ok()) {
          bufs.push_back(std::move(*b1));
        }
        if (b2.ok()) {
          bufs.push_back(std::move(*b2));
        }
        return bufs;
      });

  EXPECT_CALL(*mock_runner, CreateOutputBuffers(absl::string_view("main")))
      .WillRepeatedly([this](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        auto b1 = CreateFloatBuffer({1, 4});
        auto b2 = CreateFloatBuffer({1, 4});
        if (b1.ok()) {
          bufs.push_back(std::move(*b1));
        }
        if (b2.ok()) {
          bufs.push_back(std::move(*b2));
        }
        return bufs;
      });

  int run_count = 0;
  EXPECT_CALL(*mock_runner, Run(absl::string_view("main"), _, _))
      .WillRepeatedly([&run_count](absl::string_view,
                                   absl::Span<const TensorBuffer> inputs,
                                   absl::Span<const TensorBuffer> outputs) {
        EXPECT_EQ(inputs.size(), 2);
        EXPECT_EQ(outputs.size(), 2);
        ++run_count;
        return absl::OkStatus();
      });

  auto runner_or = StatefulLiteRtRunnerImpl::Create(
      mock_runner.get(), "main", /*num_non_state_inputs=*/1,
      /*num_non_state_outputs=*/1);
  ASSERT_THAT(runner_or, IsOk());
  auto runner = std::move(*runner_or);

  auto in_buf = CreateFloatBuffer({1, 4});
  ASSERT_THAT(in_buf, IsOk());
  std::vector<TensorBuffer> in_bufs;
  in_bufs.push_back(std::move(*in_buf));

  // Step 1
  auto step1_or = runner->Step(in_bufs, /*auto_commit_state=*/true);
  ASSERT_THAT(step1_or, IsOk());
  EXPECT_EQ(step1_or->size(), 1);
  EXPECT_EQ(run_count, 1);

  // Step 2
  auto step2_or = runner->Step(in_bufs, /*auto_commit_state=*/true);
  ASSERT_THAT(step2_or, IsOk());
  EXPECT_EQ(run_count, 2);
}

TEST_F(StatefulLiteRtRunnerTest, ManualCommitAndReset) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();

  EXPECT_CALL(*mock_runner, CreateInputBuffers(absl::string_view("decode")))
      .WillRepeatedly([this](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        auto b1 = CreateFloatBuffer({1, 4});
        auto b2 = CreateFloatBuffer({1, 4});
        if (b1.ok()) {
          bufs.push_back(std::move(*b1));
        }
        if (b2.ok()) {
          bufs.push_back(std::move(*b2));
        }
        return bufs;
      });

  EXPECT_CALL(*mock_runner, CreateOutputBuffers(absl::string_view("decode")))
      .WillRepeatedly([this](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        auto b1 = CreateFloatBuffer({1, 4});
        auto b2 = CreateFloatBuffer({1, 4});
        if (b1.ok()) {
          bufs.push_back(std::move(*b1));
        }
        if (b2.ok()) {
          bufs.push_back(std::move(*b2));
        }
        return bufs;
      });

  EXPECT_CALL(*mock_runner, Run(absl::string_view("decode"), _, _))
      .WillRepeatedly(Return(absl::OkStatus()));

  auto runner_or = StatefulLiteRtRunnerImpl::Create(
      mock_runner.get(), "decode", /*num_non_state_inputs=*/1,
      /*num_non_state_outputs=*/1);
  ASSERT_THAT(runner_or, IsOk());
  auto runner = std::move(*runner_or);

  auto in_buf = CreateFloatBuffer({1, 4});
  ASSERT_THAT(in_buf, IsOk());
  std::vector<TensorBuffer> in_bufs;
  in_bufs.push_back(std::move(*in_buf));

  // Step with auto_commit = false
  auto step1_or = runner->Step(in_bufs, /*auto_commit_state=*/false);
  EXPECT_THAT(step1_or, IsOk());

  // Manually commit
  EXPECT_THAT(runner->CommitState(), IsOk());

  // Reset
  EXPECT_THAT(runner->Reset(), IsOk());
}

TEST_F(StatefulLiteRtRunnerTest, CreateWithOwnedRunner) {
  auto mock_runner = std::make_unique<MockLiteRtRunner>();

  EXPECT_CALL(*mock_runner, CreateInputBuffers(absl::string_view("main")))
      .WillRepeatedly([this](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        auto b1 = CreateFloatBuffer({1, 4});
        auto b2 = CreateFloatBuffer({1, 4});
        if (b1.ok()) bufs.push_back(std::move(*b1));
        if (b2.ok()) bufs.push_back(std::move(*b2));
        return bufs;
      });

  EXPECT_CALL(*mock_runner, CreateOutputBuffers(absl::string_view("main")))
      .WillRepeatedly([this](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        auto b1 = CreateFloatBuffer({1, 4});
        auto b2 = CreateFloatBuffer({1, 4});
        if (b1.ok()) bufs.push_back(std::move(*b1));
        if (b2.ok()) bufs.push_back(std::move(*b2));
        return bufs;
      });

  auto runner_or = StatefulLiteRtRunnerImpl::Create(
      std::move(mock_runner), "main", /*num_non_state_inputs=*/1,
      /*num_non_state_outputs=*/1);
  ASSERT_THAT(runner_or, IsOk());
  EXPECT_NE(*runner_or, nullptr);
}

}  // namespace
}  // namespace litert::omni
