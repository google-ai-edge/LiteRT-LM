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

#include "runtime/executor/executor_backend_registry.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {
namespace {

TEST(ExecutorBackendRegistryTest, DynamicBackendNameRegistrationAndLookup) {
  BackendTraits custom_traits{
      .use_external_sampler = true,
      .default_sampler_backend = Backend::CPU,
  };
  const Backend assigned_backend = ExecutorBackendRegistry::Instance().Register(
      "custom_test_backend",
      [](const EngineSettings&) -> absl::StatusOr<BackendInstance> {
        return absl::UnimplementedError("Test stub creator");
      },
      custom_traits);

  EXPECT_TRUE(
      ExecutorBackendRegistry::Instance().IsRegistered("custom_test_backend"));
  EXPECT_TRUE(
      ExecutorBackendRegistry::Instance().IsRegistered("CUSTOM_TEST_BACKEND"));
  EXPECT_TRUE(
      ExecutorBackendRegistry::Instance().IsRegistered(assigned_backend));

  ASSERT_OK_AND_ASSIGN(const Backend resolved_backend,
                       GetBackendFromString("custom_test_backend"));
  EXPECT_EQ(resolved_backend, assigned_backend);
  EXPECT_EQ(GetBackendString(assigned_backend), "CUSTOM_TEST_BACKEND");

  std::optional<BackendTraits> traits =
      ExecutorBackendRegistry::Instance().GetTraits(assigned_backend);
  ASSERT_TRUE(traits.has_value());
  EXPECT_TRUE(traits->use_external_sampler);
  EXPECT_EQ(traits->default_sampler_backend, Backend::CPU);

  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create("dummy.sbs"));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, assigned_backend));
  EXPECT_EQ(executor_settings.GetBackend(), assigned_backend);
}

}  // namespace
}  // namespace litert::lm
