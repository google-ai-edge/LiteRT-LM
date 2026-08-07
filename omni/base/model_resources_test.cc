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

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni {
namespace {

using ::absl_testing::StatusIs;

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

}  // namespace
}  // namespace litert::omni
