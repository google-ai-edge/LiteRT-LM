// Copyright 2026 Google LLC.
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

#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include "runtime/components/model_resources.h"
#include "runtime/engine/litert_lm_lib.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/util/test_utils.h"

namespace litert::lm {
namespace {

TEST(SignatureSelectionTest, ForwardsSettingsWithoutChangingDefault) {
  LiteRtLmSettings settings;
  settings.model_path =
      (std::filesystem::path(::testing::SrcDir()) /
       "litert_lm/runtime/testdata/test_lm.litertlm")
          .string();
  settings.backend = "cpu";
  auto defaults = CreateEngineSettings(settings);
  ASSERT_TRUE(defaults.ok()) << defaults.status();
  EXPECT_TRUE(
      defaults->GetMainExecutorSettings().GetSelectedSignatures().empty());
  settings.selected_signatures = {"decode", "prefill"};
  auto selected = CreateEngineSettings(settings);
  ASSERT_TRUE(selected.ok()) << selected.status();
  EXPECT_EQ(selected->GetMainExecutorSettings().GetSelectedSignatures(),
            settings.selected_signatures);
}

TEST(SignatureSelectionTest, PrefillRunnersExcludeInactiveSignatures) {
  auto path = std::filesystem::path(::testing::SrcDir()) /
              "litert_lm/runtime/testdata/test_lm.task";
  auto assets = ModelAssets::Create(path.string());
  ASSERT_TRUE(assets.ok()) << assets.status();
  auto resources = BuildLiteRtCompiledModelResources(*assets);
  ASSERT_TRUE(resources.ok()) << resources.status();
  auto model = (*resources)->GetTFLiteModel(ModelType::kTfLitePrefillDecode);
  ASSERT_TRUE(model.ok()) << model.status();
  auto defaults = GetPrefillRunnerSetFromModel(**model, "prefill", "input_pos");
  ASSERT_TRUE(defaults.ok()) << defaults.status();
  ASSERT_EQ(defaults->size(), 1);
  auto selected = GetPrefillRunnerSetFromModel(**model, "prefill", "input_pos",
                                               {"decode", "prefill"});
  ASSERT_TRUE(selected.ok()) << selected.status();
  EXPECT_EQ(*selected, *defaults);
  auto excluded =
      GetPrefillRunnerSetFromModel(**model, "prefill", "input_pos", {"decode"});
  ASSERT_TRUE(excluded.ok()) << excluded.status();
  EXPECT_TRUE(excluded->empty());
}

}  // namespace
}  // namespace litert::lm
