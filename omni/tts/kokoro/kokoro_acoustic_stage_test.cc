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

#include "omni/tts/kokoro/kokoro_acoustic_stage.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/stream_text_source.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

TEST(KokoroAcousticStageTest, CreateRejectsMissingCompiledModel) {
  KokoroModelConfig config;
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  StreamTextSource text_source({});

  // ModelResources has no compiled model registered under "kokoro_acoustic".
  EXPECT_THAT(
      KokoroAcousticStage::Create(&text_source, config, "/tmp", resources)
          .status(),
      testing::status::StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace litert::omni::tts
