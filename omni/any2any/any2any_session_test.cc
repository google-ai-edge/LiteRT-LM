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

#include "omni/any2any/any2any_session.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::any2any {

class Any2AnySessionTest : public ::testing::Test {};

namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

// Fake Any2Any processing stage that pulls `Input` items from `InputSource`
// until `EndOfInputTurn`, and emits `TextOutput`, optionally `AudioOutput`, and
// `EndOfOutputTurn`.
class FakeAny2AnyStage : public SingleThreadedStageWithDeque<Output> {
 public:
  explicit FakeAny2AnyStage(Stage<Input>* input_source, bool emit_audio = false)
      : input_source_(*input_source), emit_audio_(emit_audio) {}

 protected:
  bool NeedScheduleInternal() const override {
    return !is_finished_ &&
           (input_source_.NeedSchedule() || input_source_.HasOutput());
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);

    while (!is_finished_) {
      if (!input_source_.HasOutput()) {
        if (!input_source_.NeedSchedule()) {
          break;
        }
        absl::Status status = input_source_.Schedule();
        if (absl::IsOutOfRange(status)) {
          is_finished_ = true;
          break;
        }
        if (absl::IsNotFound(status)) {
          break;
        }
        ABSL_RETURN_IF_ERROR(status);
        if (!input_source_.HasOutput()) {
          break;
        }
      }

      ABSL_ASSIGN_OR_RETURN(Input input, input_source_.GetOutput());
      if (std::holds_alternative<EndOfInput>(input)) {
        is_finished_ = true;
        break;
      }
      if (std::holds_alternative<EndOfInputTurn>(input)) {
        EmitTurnOutput();
        SetState(State::kIdle);
        return absl::OkStatus();
      }
      if (const auto* aud_meta = std::get_if<AudioInputMetadata>(&input)) {
        current_sample_rate_ = aud_meta->sample_rate_hz;
      } else if (const auto* txt = std::get_if<TextInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "text:", txt->text);
      } else if (const auto* img = std::get_if<ImageInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "img:", img->image_bytes.size(), "x",
                        img->width, "x", img->height);
      } else if (const auto* aud = std::get_if<AudioInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "aud:", aud->pcm_samples.size(), "@",
                        current_sample_rate_);
      }
    }

    if (is_finished_) {
      if (!summary_.empty()) {
        EmitTurnOutput();
      }
      SetState(State::kIdle);
      if (HasOutput()) {
        return absl::OkStatus();
      }
      return absl::OutOfRangeError("End of input reached.");
    }

    SetState(State::kIdle);
    return absl::OkStatus();
  }

  void ResetInternal() override {
    summary_.clear();
    current_sample_rate_ = kDefaultInputSampleRateHz;
    is_finished_ = false;
  }

 private:
  void EmitTurnOutput() {
    PushOutput(TextOutput{.confirmed_text = std::move(summary_)});
    summary_.clear();
    if (emit_audio_) {
      PushOutput(AudioOutput{
          .pcm_samples = {0.25f, -0.25f},
          .sample_rate_hz = 24000,
      });
    }
    PushOutput(EndOfOutputTurn{});
  }

  Stage<Input>& input_source_;
  bool emit_audio_ = false;
  std::string summary_;
  int current_sample_rate_ = kDefaultInputSampleRateHz;
  bool is_finished_ = false;
};

TEST_F(Any2AnySessionTest,
       PushInputSourceEndOfTurnEnablesMultiTurnWithoutReset) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto any2any_stage = std::make_unique<FakeAny2AnyStage>(input_source.get(),
                                                          /*emit_audio=*/true);
  Stage<Output>* raw_out = any2any_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(any2any_stage));

  ASSERT_OK_AND_ASSIGN(auto session,
                       MultiStagedSession::Create(std::move(stages), raw_out));

  // Before EndOfInputTurn or Finish(), ProcessNext() waits for turn completion
  // and returns kNotFound.
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Describe "}));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "this:"}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  ASSERT_OK(raw_input->PushInput(OmniSession::ImageInput{
      .image_bytes = {255, 0, 0, 0, 255, 0},
      .width = 2,
      .height = 1,
      .channels = 3,
  }));
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInputMetadata{.sample_rate_hz = 24000}));
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {0.1f, 0.2f, 0.3f}}));

  // Signal end of turn with EndOfInputTurn control input.
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  // First ProcessNext() yields the TextOutput for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text,
            "text:Describe |text:this:|img:6x2x1|aud:3@24000");

  // Second ProcessNext() yields the AudioOutput for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out2, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  EXPECT_EQ(std::get<AudioOutput>(out2).sample_rate_hz, 24000);
  EXPECT_THAT(std::get<AudioOutput>(out2).pcm_samples,
              ElementsAre(0.25f, -0.25f));

  // Third ProcessNext() yields EndOfOutputTurn for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out_turn1, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(out_turn1));

  // Turn 2 on the SAME session (no Reset() called):
  ASSERT_OK(raw_input->PushInput(
      OmniSession::TextInput{.text = "Followup question"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out3, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out3));
  EXPECT_EQ(std::get<TextOutput>(out3).confirmed_text,
            "text:Followup question");
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out4, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out4));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out_turn2, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(out_turn2));

  // Calling Finish() ends the session.
  raw_input->Finish();
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST_F(Any2AnySessionTest, PushInputSourceProcessAsyncStreamsTextAndAudio) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto any2any_stage = std::make_unique<FakeAny2AnyStage>(input_source.get(),
                                                          /*emit_audio=*/true);
  Stage<Output>* raw_out = any2any_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(any2any_stage));

  ::litert::lm::ThreadPool pool("any2any_test_pool", 2);
  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), raw_out, &pool));

  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Async"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));
  raw_input->Finish();

  std::vector<std::string> texts;
  std::vector<float> audio_samples;
  int end_of_turn_count = 0;
  absl::Status final_status;
  absl::Notification done;

  ASSERT_OK(session->ProcessAsync(
      [&](absl::StatusOr<OmniSession::Output> res) -> absl::Status {
        if (!res.ok()) {
          final_status = res.status();
          done.Notify();
          return res.status();
        }
        if (const auto* txt = std::get_if<TextOutput>(&*res)) {
          texts.push_back(txt->confirmed_text);
        } else if (const auto* aud = std::get_if<AudioOutput>(&*res)) {
          audio_samples.insert(audio_samples.end(), aud->pcm_samples.begin(),
                               aud->pcm_samples.end());
        } else if (std::holds_alternative<OmniSession::EndOfOutputTurn>(*res)) {
          ++end_of_turn_count;
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  EXPECT_THAT(final_status, StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(texts, ElementsAre("text:Async"));
  EXPECT_THAT(audio_samples, ElementsAre(0.25f, -0.25f));
  EXPECT_EQ(end_of_turn_count, 1);
}

class FakeAny2AnySessionFactory : public Any2AnySessionFactory {
 public:
  FakeAny2AnySessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Any2AnyMetadata& metadata,
      std::vector<std::string> available_voices = {},
      std::shared_ptr<ModelResources> absl_nullable model_resources = nullptr)
      : Any2AnySessionFactory(settings, session_config, metadata,
                              std::move(available_voices),
                              std::move(model_resources)) {}

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithInputSource(
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override {
    auto any2any_stage = std::make_unique<FakeAny2AnyStage>(
        input_source.get(), session_config.generate_audio());
    Stage<Output>* raw_output_stage = any2any_stage.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(input_source));
    stages.push_back(std::move(any2any_stage));
    return MultiStagedSession::Create(std::move(stages), raw_output_stage,
                                      thread_pool());
  }
};

TEST_F(Any2AnySessionTest, FactoryMetadataAndConfigMerging) {
  lm::proto::Any2AnyMetadata metadata;
  metadata.add_supported_languages("en-US");
  metadata.add_supported_languages("zh-CN");
  metadata.set_input_sample_rate_hz(48000);

  lm::proto::Any2AnySessionConfig base_cfg;
  base_cfg.set_language("en-US");
  base_cfg.set_generate_audio(true);
  base_cfg.set_temperature(0.7f);

  OmniEngine::Settings settings;
  settings.num_threads = 2;

  FakeAny2AnySessionFactory factory(settings, &base_cfg, metadata,
                                    {"voice_a", "voice_b"},
                                    /*model_resources=*/nullptr);

  EXPECT_THAT(factory.GetAvailableLanguages(), ElementsAre("en-US", "zh-CN"));
  EXPECT_THAT(factory.available_voices(), ElementsAre("voice_a", "voice_b"));
  EXPECT_TRUE(factory.HasVoice("voice_a"));
  EXPECT_TRUE(factory.HasVoice("voice_b"));
  EXPECT_FALSE(factory.HasVoice("unknown_voice"));
  EXPECT_EQ(factory.GetDefaultVoice("en-US"), "voice_a");
  EXPECT_EQ(factory.GetInputSampleRateHz(), 48000);
  EXPECT_EQ(factory.engine_settings().num_threads, 2);
  EXPECT_EQ(factory.model_resources(), nullptr);

  lm::proto::Any2AnySessionConfig override_cfg;
  override_cfg.set_generate_audio(false);
  override_cfg.set_temperature(0.2f);
  lm::proto::Any2AnySessionConfig merged =
      factory.MergeSessionConfig(override_cfg);
  EXPECT_EQ(merged.language(), "en-US");
  EXPECT_FALSE(merged.generate_audio());
  EXPECT_FLOAT_EQ(merged.temperature(), 0.2f);
}

TEST_F(Any2AnySessionTest, FactoryDefaultsWhenEmpty) {
  lm::proto::Any2AnyMetadata metadata;
  FakeAny2AnySessionFactory factory(OmniEngine::Settings{},
                                    /*session_config=*/nullptr, metadata);

  EXPECT_TRUE(factory.GetAvailableLanguages().empty());
  EXPECT_TRUE(factory.available_voices().empty());
  EXPECT_FALSE(factory.HasVoice("voice_a"));
  EXPECT_EQ(factory.GetDefaultVoice("en-US"), "");
  EXPECT_EQ(factory.GetInputSampleRateHz(), kDefaultInputSampleRateHz);
  EXPECT_EQ(factory.session_config().language(), "");
}

TEST_F(Any2AnySessionTest, FactoryCreateAndExecuteSession) {
  lm::proto::Any2AnyMetadata metadata;
  lm::proto::Any2AnySessionConfig session_config;
  session_config.set_generate_audio(true);

  FakeAny2AnySessionFactory factory(OmniEngine::Settings{}, &session_config,
                                    metadata, {"voice_a"});

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();

  ASSERT_OK_AND_ASSIGN(auto session, factory.Create(std::move(input_source)));

  ASSERT_OK(
      raw_input->PushInput(OmniSession::TextInput{.text = "Test via Factory"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(text_out));
  EXPECT_EQ(std::get<TextOutput>(text_out).confirmed_text,
            "text:Test via Factory");

  ASSERT_OK_AND_ASSIGN(OmniSession::Output audio_out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));
}

}  // namespace
}  // namespace litert::omni::any2any
