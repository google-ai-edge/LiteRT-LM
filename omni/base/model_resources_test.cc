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

#include "omni/base/model_resources.h"

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_lm_runner.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

static_assert(!std::is_copy_constructible_v<ModelResources>);
static_assert(!std::is_copy_assignable_v<ModelResources>);
static_assert(std::is_move_constructible_v<ModelResources>);
static_assert(std::is_move_assignable_v<ModelResources>);

class TrackingLmRunner : public LiteRtLmRunner {
 public:
  explicit TrackingLmRunner(std::function<void()> on_destroy)
      : on_destroy_(std::move(on_destroy)) {}

  ~TrackingLmRunner() override {
    if (on_destroy_) {
      on_destroy_();
    }
  }

  absl::Status Prefill(const lm::ExecutorInputs& /*inputs*/) override {
    return absl::OkStatus();
  }

  absl::StatusOr<TensorBuffer> Decode(
      const lm::ExecutorInputs& /*inputs*/) override {
    return absl::UnimplementedError("not implemented");
  }

  absl::Status Reset() override { return absl::OkStatus(); }

 private:
  std::function<void()> on_destroy_;
};

TEST(ModelResourcesTest, AddAndGetCompiledModel) {
  ModelResources resources;
  EXPECT_FALSE(resources.HasCompiledModel("test_model"));
  EXPECT_THAT(resources.GetCompiledModel("test_model"),
              StatusIs(absl::StatusCode::kNotFound));

  // Reject adding null pointer
  EXPECT_THAT(resources.AddCompiledModel("test_model", nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelResourcesTest, SetAndGetEnvironment) {
  ModelResources resources;
  EXPECT_EQ(resources.environment(), nullptr);

  auto env_status = Environment::Create({});
  if (env_status.HasValue()) {
    auto shared_env = std::make_shared<Environment>(std::move(*env_status));
    resources.SetEnvironment(shared_env);
    EXPECT_EQ(resources.environment(), shared_env);
  }
}

TEST(ModelResourcesTest, AddAndGetLmRunner) {
  ModelResources resources;
  EXPECT_FALSE(resources.HasLmRunner("test_lm"));
  EXPECT_THAT(resources.GetLmRunner("test_lm"),
              StatusIs(absl::StatusCode::kNotFound));

  // Reject adding null pointer
  EXPECT_THAT(resources.AddLmRunner("test_lm", nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelResourcesTest, MoveConstructionTransfersOwnershipBeforeEnvDestroy) {
  std::vector<std::string> destroy_order;
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  std::shared_ptr<Environment> shared_env(new Environment(std::move(*env)),
                                          [&](Environment* ptr) {
                                            delete ptr;
                                            destroy_order.push_back("env");
                                          });

  ModelResources source;
  auto runner = std::make_shared<TrackingLmRunner>(
      [&] { destroy_order.push_back("runner"); });
  std::weak_ptr<LiteRtLmRunner> runner_weak = runner;
  ASSERT_OK(source.AddLmRunner("llm_backbone", std::move(runner)));

  {
    ModelResources dest(std::move(source));
    dest.SetEnvironment(std::move(shared_env));
    EXPECT_TRUE(dest.HasLmRunner("llm_backbone"));
    EXPECT_FALSE(runner_weak.expired());
  }

  // `dest` must release its runners before its Environment, without `source`
  // retaining stale runner references across `dest`'s destruction.
  EXPECT_TRUE(runner_weak.expired());
  EXPECT_THAT(destroy_order, ElementsAre("runner", "env"));
}

TEST(ModelResourcesTest, MoveAssignmentDestroysOldRunnersBeforeOldEnv) {
  std::vector<std::string> destroy_order;
  auto env_a = Environment::Create({});
  ASSERT_TRUE(env_a.HasValue());
  std::shared_ptr<Environment> shared_env_a(new Environment(std::move(*env_a)),
                                            [&](Environment* ptr) {
                                              delete ptr;
                                              destroy_order.push_back("env_a");
                                            });

  auto env_b = Environment::Create({});
  ASSERT_TRUE(env_b.HasValue());
  std::shared_ptr<Environment> shared_env_b(new Environment(std::move(*env_b)),
                                            [&](Environment* ptr) {
                                              delete ptr;
                                              destroy_order.push_back("env_b");
                                            });

  ModelResources source(std::move(shared_env_b));
  auto runner_b = std::make_shared<TrackingLmRunner>(
      [&] { destroy_order.push_back("runner_b"); });
  std::weak_ptr<LiteRtLmRunner> runner_b_weak = runner_b;
  ASSERT_OK(source.AddLmRunner("runner_b", std::move(runner_b)));

  {
    ModelResources dest(std::move(shared_env_a));
    auto runner_a = std::make_shared<TrackingLmRunner>(
        [&] { destroy_order.push_back("runner_a"); });
    std::weak_ptr<LiteRtLmRunner> runner_a_weak = runner_a;
    ASSERT_OK(dest.AddLmRunner("runner_a", std::move(runner_a)));

    dest = std::move(source);
    EXPECT_TRUE(runner_a_weak.expired());
    EXPECT_FALSE(runner_b_weak.expired());
    EXPECT_THAT(destroy_order, ElementsAre("runner_a", "env_a"));
    EXPECT_TRUE(dest.HasLmRunner("runner_b"));
  }

  EXPECT_TRUE(runner_b_weak.expired());
  EXPECT_THAT(destroy_order,
              ElementsAre("runner_a", "env_a", "runner_b", "env_b"));
}

}  // namespace
}  // namespace litert::omni
