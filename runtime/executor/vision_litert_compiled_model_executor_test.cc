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

#include "runtime/executor/vision_litert_compiled_model_executor.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::lm::Backend;
using ::litert::lm::LitertLmLoader;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelResourcesLitertLm;
using ::litert::lm::VisionExecutorSettings;
using ::testing::status::StatusIs;

TEST(VisionLiteRtCompiledModelExecutorTest, CreateExecutorTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";

  ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(scoped_file)));
  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesLitertLm::Create(std::move(loader)));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path.string()));

  ASSERT_OK_AND_ASSIGN(VisionExecutorSettings settings,
                       VisionExecutorSettings::CreateDefault(
                           model_assets,
                           /*encoder_backend=*/Backend::GPU,
                           /*adapter_backend=*/Backend::GPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto vision_executor =
      VisionLiteRtCompiledModelExecutor::Create(settings, env);
  EXPECT_THAT(vision_executor,
              StatusIs(absl::StatusCode::kNotFound,
                       "TF_LITE_VISION_ENCODER not found in the model."));
}

}  // namespace
}  // namespace litert::lm
