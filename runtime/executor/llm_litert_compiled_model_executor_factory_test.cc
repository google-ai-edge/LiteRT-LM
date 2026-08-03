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

#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr char kTestStaticModelPath[] =
    "litert_lm/runtime/testdata/test_lm.litertlm";

absl::StatusOr<std::unique_ptr<ModelResources>> CreateExecutorModelResources(
    absl::string_view model_path) {
  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(model_path));
  ASSIGN_OR_RETURN(auto loader, LitertLmLoader::Create(std::move(scoped_file)));
  return ModelResourcesLitertLm::Create(std::move(loader));
}

TEST(LlmLiteRtCompiledModelExecutorFactoryTest, CanCreateStaticModelExecutor) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  EXPECT_OK(CreateLlmLiteRtCompiledModelExecutor(executor_settings, env,
                                                 *model_resources));
}

}  // namespace
}  // namespace litert::lm
