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

#include "omni/tts/kokoro/kokoro_factory.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/text_chunk_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

TEST(KokoroFactoryTest, InitKokoroResourcesFailsWithNonexistentModels) {
  KokoroModelConfig config;
  config.acoustic_file = "nonexistent_acoustic.tflite";
  config.vocoder_file = "nonexistent_vocoder.tflite";

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  ModelResources resources(shared_env);

  auto status =
      InitKokoroResources(config, "/tmp/invalid_path", "", lm::Backend::CPU, 1,
                          *shared_env, resources);
  EXPECT_FALSE(status.ok());
}

TEST(KokoroFactoryTest, CreateKokoroComponentsRejectsMissingModels) {
  KokoroModelConfig config;
  config.target_bucket = 128;

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  TextChunkConfig chunk_config;
  chunk_config.max_buffer_size = 0;

  // Since ModelResources does not have compiled models loaded,
  // CreateKokoroComponents should propagate the error.
  auto components =
      CreateKokoroComponents(config, "/tmp", chunk_config, resources);
  EXPECT_FALSE(components.ok());
}

}  // namespace
}  // namespace litert::omni::tts
