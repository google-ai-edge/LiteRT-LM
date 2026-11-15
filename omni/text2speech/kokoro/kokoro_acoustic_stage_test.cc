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
      std::shared_ptr<ModelResources> resources, int model_capacity = 512) {
    auto stage = std::unique_ptr<KokoroAcousticStage>(new KokoroAcousticStage(
        text_source, std::move(config), "", std::move(resources)));
    stage->model_capacity_ = model_capacity;
    stage->frame_capacity_ = 1000;
    stage->voice_pack_.resize(kokoro::kVoicePackSize, 0.0f);
    return stage;
  }

  static void InitFake(KokoroAcousticStage& stage, int model_capacity = 512) {
    stage.model_capacity_ = model_capacity;
    stage.frame_capacity_ = 1000;
    stage.voice_pack_.resize(kokoro::kVoicePackSize, 0.0f);
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
  }

  static size_t NumPendingSlices(const KokoroAcousticStage& stage) {
    return stage.pending_slices_.size();
  }
};

namespace {

using ::absl_testing::StatusIs;

class FakeTextSource : public SingleThreadedStageWithDeque<std::string> {
 public:
  void Push(std::string text) { PushOutput(std::move(text)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

class FakeRunSliceAcousticStage : public KokoroAcousticStage {
 public:
  FakeRunSliceAcousticStage(Stage<std::string>* text_source,
                            KokoroModelConfig config,
                            std::shared_ptr<ModelResources> resources,
                            int model_capacity = 512)
      : KokoroAcousticStage(text_source, std::move(config), "",
                            std::move(resources)) {
    KokoroAcousticStageTestPeer::InitFake(*this, model_capacity);
  }

  absl::Status RunSlice(const kokoro::TokenSlice& slice) override {
    executed_slices_.push_back(slice);
    return run_slice_status_;
  }

  void SetRunSliceStatus(absl::Status status) {
    run_slice_status_ = std::move(status);
  }

  const std::vector<kokoro::TokenSlice>& executed_slices() const {
    return executed_slices_;
  }

 private:
  absl::Status run_slice_status_ = absl::OkStatus();
  std::vector<kokoro::TokenSlice> executed_slices_;
};

class KokoroAcousticStageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto env = ::litert::Environment::Create({});
    ASSERT_TRUE(env.HasValue());
    shared_env_ = std::make_shared<::litert::Environment>(std::move(*env));
    resources_ = std::make_shared<ModelResources>(shared_env_);
  }

  absl::StatusOr<std::unique_ptr<KokoroPhonemizer>> CreateTestPhonemizer() {
    return KokoroPhonemizer::Create("", "en-us",
                                    {{"hello", "həlˈO"}, {"world", "wˈɜːld"}});
  }

  std::shared_ptr<::litert::Environment> shared_env_;
  std::shared_ptr<ModelResources> resources_;
  FakeTextSource text_source_;
};

TEST_F(KokoroAcousticStageTest, CreateRejectsMissingCompiledModel) {
  KokoroModelConfig config;

  // ModelResources has no compiled model registered under "kokoro_acoustic".
  EXPECT_THAT(
      KokoroAcousticStage::Create(&text_source_, config, "/tmp", resources_)
          .status(),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(KokoroAcousticStageTest, NeedScheduleReturnsTrueForPendingSlicesOrText) {
  KokoroModelConfig config;
  auto stage =
      KokoroAcousticStageTestPeer::Create(&text_source_, config, resources_);

  EXPECT_FALSE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  text_source_.Push("Hello");
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));
  auto text = text_source_.GetOutput();
  ASSERT_TRUE(text.ok());
  EXPECT_FALSE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  KokoroAcousticStageTestPeer::AddPendingSlice(*stage, kokoro::TokenSlice{});
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));
}

TEST_F(KokoroAcousticStageTest, ResetClearsPendingSlices) {
  KokoroModelConfig config;
  auto stage =
      KokoroAcousticStageTestPeer::Create(&text_source_, config, resources_);

  KokoroAcousticStageTestPeer::AddPendingSlice(*stage, kokoro::TokenSlice{});
  KokoroAcousticStageTestPeer::AddPendingSlice(*stage, kokoro::TokenSlice{});
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 2);
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  KokoroAcousticStageTestPeer::Reset(*stage);
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 0);
  EXPECT_FALSE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));
}

TEST_F(KokoroAcousticStageTest, ScheduleInternalWhenEmptyReturnsOk) {
  KokoroModelConfig config;
  auto stage =
      KokoroAcousticStageTestPeer::Create(&text_source_, config, resources_);

  EXPECT_TRUE(KokoroAcousticStageTestPeer::Schedule(*stage).ok());
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 0);
}

TEST_F(KokoroAcousticStageTest, ScheduleRunsOneSlicePerStep) {
  KokoroModelConfig config;
  config.language = "en-us";
  // Capacity 8 forces "hello world" to split into multiple slices.
  auto stage = std::make_unique<FakeRunSliceAcousticStage>(
      &text_source_, config, resources_, /*model_capacity=*/8);
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer());
  KokoroAcousticStageTestPeer::SetPhonemizer(*stage, std::move(phonemizer));

  text_source_.Push("hello world");
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  // First step: slices text, queues slices, and executes the first slice.
  ASSERT_TRUE(KokoroAcousticStageTestPeer::Schedule(*stage).ok());
  EXPECT_EQ(stage->executed_slices().size(), 1);
  const size_t remaining_slices =
      KokoroAcousticStageTestPeer::NumPendingSlices(*stage);
  ASSERT_GE(remaining_slices, 1);
  // text_source_ is now consumed, but NeedSchedule remains true because slices
  // are pending.
  EXPECT_FALSE(text_source_.HasOutput());
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  // Second step: executes the next slice from the queue.
  ASSERT_TRUE(KokoroAcousticStageTestPeer::Schedule(*stage).ok());
  EXPECT_EQ(stage->executed_slices().size(), 2);
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage),
            remaining_slices - 1);
  EXPECT_EQ(stage->executed_slices()[0].join_after,
            stage->executed_slices()[1].join_before);
}

TEST_F(KokoroAcousticStageTest, ResetInMiddleOfChunkClearsPendingSlices) {
  KokoroModelConfig config;
  config.language = "en-us";
  auto stage = std::make_unique<FakeRunSliceAcousticStage>(
      &text_source_, config, resources_, /*model_capacity=*/8);
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer());
  KokoroAcousticStageTestPeer::SetPhonemizer(*stage, std::move(phonemizer));

  text_source_.Push("hello world");
  ASSERT_TRUE(KokoroAcousticStageTestPeer::Schedule(*stage).ok());
  ASSERT_GE(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 1);

  KokoroAcousticStageTestPeer::Reset(*stage);
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 0);
  EXPECT_FALSE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));
}

TEST_F(KokoroAcousticStageTest,
       ScheduleInternalClearsPendingSlicesOnSliceFailure) {
  KokoroModelConfig config;
  config.language = "en-us";
  auto stage = std::make_unique<FakeRunSliceAcousticStage>(
      &text_source_, config, resources_, /*model_capacity=*/8);
  stage->SetRunSliceStatus(absl::InternalError("simulated slice failure"));
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer());
  KokoroAcousticStageTestPeer::SetPhonemizer(*stage, std::move(phonemizer));

  text_source_.Push("hello world");
  EXPECT_TRUE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));

  // Schedule slices the chunk, pops the first slice, RunSlice fails,
  // and the remaining slices are cleared to prevent partial sentence output.
  EXPECT_THAT(KokoroAcousticStageTestPeer::Schedule(*stage),
              StatusIs(absl::StatusCode::kInternal, "simulated slice failure"));
  EXPECT_EQ(KokoroAcousticStageTestPeer::NumPendingSlices(*stage), 0);
  EXPECT_FALSE(KokoroAcousticStageTestPeer::NeedSchedule(*stage));
}

}  // namespace
}  // namespace litert::omni::tts
