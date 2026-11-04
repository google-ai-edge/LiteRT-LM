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

#include "omni/text2speech/kokoro/kokoro_acoustic_stage.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/text2speech/kokoro/common.h"
#include "omni/text2speech/kokoro/kokoro_model_config.h"
#include "omni/text2speech/kokoro/phonemizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {

class KokoroAcousticStageTestPeer {
 public:
  static std::unique_ptr<KokoroAcousticStage> Create(
      Stage<std::string>* text_source, KokoroModelConfig config,
      std::shared_ptr<ModelResources> resources) {
    auto stage = std::unique_ptr<KokoroAcousticStage>(new KokoroAcousticStage(
        text_source, std::move(config), "", std::move(resources)));
    stage->model_capacity_ = 512;
    stage->frame_capacity_ = 1000;
    stage->voice_pack_.resize(kokoro::kVoicePackSize, 0.0f);
    return stage;
  }

  static void SetPhonemizer(KokoroAcousticStage& stage,
                            std::unique_ptr<KokoroPhonemizer> phonemizer) {
    stage.phonemizer_ = std::move(phonemizer);
  }

  static bool NeedSchedule(const KokoroAcousticStage& stage) {
    return stage.NeedScheduleInternal();
  }

  static absl::Status Schedule(KokoroAcousticStage& stage) {
    return stage.ScheduleInternal();
  }

  static void Reset(KokoroAcousticStage& stage) { stage.ResetInternal(); }

  static void AddPendingSlice(KokoroAcousticStage& stage,
                              kokoro::TokenSlice slice) {
    stage.pending_slices_.push_back(std::move(slice));
    stage.num_pending_slices_.store(stage.pending_slices_.size(),
                                    std::memory_order_release);
  }

  static size_t NumPendingSlices(const KokoroAcousticStage& stage) {
    return stage.num_pending_slices_.load(std::memory_order_acquire);
  }
};

class FakeTextSource : public SingleThreadedStageWithDeque<std::string> {
 public:
  void Push(std::string text) { PushOutput(std::move(text)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

namespace {

using ::absl_testing::StatusIs;

TEST(KokoroAcousticStageTest, CreateRejectsMissingCompiledModel) {
  KokoroModelConfig config;
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  FakeTextSource text_source;

  // ModelResources has no compiled model registered under "kokoro_acoustic".
  EXPECT_THAT(
      KokoroAcousticStage::Create(&text_source, config, "/tmp", resources)
          .status(),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(KokoroAcousticStageTest, NeedScheduleReturnsTrueForPendingSlicesOrText) {
  KokoroModelConfig config;
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  FakeTextSource text_source;
  auto stage =
      KokoroAcousticStageTestPeer::Create(&text_source, config, resources);

  EXPECT_FALSE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  text_source.Push("Hello");
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));
  auto text = text_source.GetOutput();
  ASSERT_TRUE(text.ok());
  EXPECT_FALSE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  KokoroAcousticStageTestPeer::AddPendingSlice(*stage, kokoro::TokenSlice{});
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));
}

TEST(KokoroAcousticStageTest, ResetClearsPendingSlices) {
  KokoroModelConfig config;
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  FakeTextSource text_source;
  auto stage =
      KokoroAcousticStageTestPeer::Create(&text_source, config, resources);

  KokoroAcousticStageTestPeer::AddPendingSlice(*stage, kokoro::TokenSlice{});
  KokoroAcousticStageTestPeer::AddPendingSlice(*stage, kokoro::TokenSlice{});
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 2);
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  KokoroAcousticStageTestPeer::Reset(*stage);
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 0);
  EXPECT_FALSE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));
}

TEST(KokoroAcousticStageTest, ScheduleInternalWhenEmptyReturnsOk) {
  KokoroModelConfig config;
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  FakeTextSource text_source;
  auto stage =
      KokoroAcousticStageTestPeer::Create(&text_source, config, resources);

  EXPECT_TRUE(KokoroAcousticStageTestPeer::Schedule(*stage).ok());
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 0);
}

TEST(KokoroAcousticStageTest, ScheduleInternalSlicesAndRunsSlice) {
  KokoroModelConfig config;
  config.language = "en-us";
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  FakeTextSource text_source;
  auto stage =
      KokoroAcousticStageTestPeer::Create(&text_source, config, resources);

  auto phonemizer = KokoroPhonemizer::Create(
      "", "en-us", {{"hello", "həlˈO"}, {"world", "wˈɜːld"}});
  ASSERT_TRUE(phonemizer.ok());
  KokoroAcousticStageTestPeer::SetPhonemizer(*stage, std::move(*phonemizer));

  text_source.Push("hello world");
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  // ScheduleInternal slices "hello world", enqueues the slices into
  // pending_slices_, and pops the first slice to execute RunSlice.
  // Because no CompiledModel was attached, RunSlice safely returns
  // FailedPreconditionError.
  EXPECT_THAT(KokoroAcousticStageTestPeer::Schedule(*stage),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 0);
}

}  // namespace
}  // namespace litert::omni::tts
