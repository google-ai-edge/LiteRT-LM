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

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

TEST(EngineTest, CreateEngine) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ModelAssets model_assets;
  model_assets.model_paths.push_back(task_path.string());
  LlmExecutorSettings executor_settings(model_assets);
  executor_settings.SetBackend(Backend::CPU);
  executor_settings.SetBackendConfig(CpuConfig());
  executor_settings.SetMaxNumTokens(160);
  EngineSettings llm_settings(executor_settings);

  absl::StatusOr<std::unique_ptr<Engine>> llm =
      Engine::CreateEngine(llm_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession();
  ABSL_CHECK_OK(session);

  absl::Status status = (*session)->RunPrefill("Hello world!");
  ABSL_CHECK_OK(status);

  auto responses = (*session)->RunDecode();

  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  EXPECT_TRUE((responses->GetResponseTextAt(0))->starts_with("<unused185>"));
  EXPECT_OK(llm);
}

// TODO (b/397975034): Add more tests for Engine.

}  // namespace
}  // namespace litert::lm
