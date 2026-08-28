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

#include "omni/tts/kokoro/kokoro_vocoder_stage.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/kokoro/kokoro_io_types.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

class DummyAcousticStage
    : public SingleThreadedStageWithDeque<KokoroAcousticOutput> {
 public:
  void Reset() override {}

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TEST(KokoroVocoderStageTest, CreateRejectsMissingCompiledModel) {
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  DummyAcousticStage acoustic_stage;

  // ModelResources has no compiled model registered under "kokoro_vocoder".
  EXPECT_THAT(KokoroVocoderStage::Create(&acoustic_stage, resources).status(),
              testing::status::StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace litert::omni::tts
