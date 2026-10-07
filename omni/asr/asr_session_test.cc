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

#include "omni/asr/asr_session.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/detokenizer.h"
#include "omni/asr/levenshtein_text_merger.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_session.h"
#include "runtime/framework/threadpool.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::asr {

class AsrSessionTest : public ::testing::Test {
 public:
  static std::unique_ptr<AudioSource> CreateAudioInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      int sample_rate_hz, int num_channels, int samples_per_interval,
      int overlap_samples) {
    return AsrSessionFactory::CreateAudioInputSource(
        std::move(input_source), sample_rate_hz, num_channels,
        samples_per_interval, overlap_samples);
  }
};

namespace {

using ::absl_testing::StatusIs;

class FakeAudioPreprocessor : public AudioPreprocessor {
 public:
  explicit FakeAudioPreprocessor(AudioSource* source)
      : AudioPreprocessor(source) {}

 protected:
  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    auto pcm_samples = audio_source_.GetOutput();
    if (absl::IsNotFound(pcm_samples.status())) {
      SetState(State::kIdle);
      return absl::OkStatus();
    }
    if (!pcm_samples.ok()) {
      SetState(State::kIdle);
      return pcm_samples.status();
    }
    PushOutput(std::move(*pcm_samples));
    SetState(State::kIdle);
    return absl::OkStatus();
  }
};

class FakeSpeechRecognizer : public SpeechRecognizer {
 public:
  explicit FakeSpeechRecognizer(AudioPreprocessor* preprocessor)
      : SpeechRecognizer(preprocessor) {}

 protected:
  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    auto mel_features = audio_preprocessor_.GetOutput();
    if (absl::IsNotFound(mel_features.status())) {
      SetState(State::kIdle);
      return absl::OkStatus();
    }
    if (!mel_features.ok()) {
      SetState(State::kIdle);
      return mel_features.status();
    }
    std::vector<SpeechRecognizer::DecodedToken> tokens;
    for (float val : *mel_features) {
      tokens.push_back({static_cast<int>(val), 100});
    }
    PushOutput(std::move(tokens));
    SetState(State::kIdle);
    return absl::OkStatus();
  }
};

class FakeDetokenizer : public Detokenizer {
 public:
  explicit FakeDetokenizer(SpeechRecognizer* recognizer)
      : Detokenizer(recognizer) {}

 protected:
  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    auto tokens = speech_recognizer_.GetOutput();
    if (absl::IsNotFound(tokens.status())) {
      SetState(State::kIdle);
      return absl::OkStatus();
    }
    if (!tokens.ok()) {
      SetState(State::kIdle);
      return tokens.status();
    }
    std::vector<Detokenizer::Word> words;
    for (const auto& tok : *tokens) {
      words.push_back({"w_" + std::to_string(tok.token_id), tok.timestamp_ms});
    }
    PushOutput(std::move(words));
    SetState(State::kIdle);
    return absl::OkStatus();
  }
};

absl::StatusOr<std::unique_ptr<MultiStagedSession>> BuildFakeAsrSession(
    std::unique_ptr<AudioSource> audio_source,
    ::litert::lm::ThreadPool* pool = nullptr) {
  auto preprocessor =
      std::make_unique<FakeAudioPreprocessor>(audio_source.get());
  auto recognizer = std::make_unique<FakeSpeechRecognizer>(preprocessor.get());
  auto detokenizer = std::make_unique<FakeDetokenizer>(recognizer.get());
  auto text_merger = std::make_unique<LevenshteinTextMerger>(detokenizer.get());
  Stage<Output>* raw_merger = text_merger.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(audio_source));
  stages.push_back(std::move(preprocessor));
  stages.push_back(std::move(recognizer));
  stages.push_back(std::move(detokenizer));
  stages.push_back(std::move(text_merger));
  return MultiStagedSession::Create(std::move(stages), raw_merger, pool);
}

TEST_F(AsrSessionTest, AudioInputSourceValidatesMetadataAndInputTypes) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto audio_source = CreateAudioInputSource(
      std::move(input_source), /*sample_rate_hz=*/16000, /*num_channels=*/1,
      /*samples_per_interval=*/2, /*overlap_samples=*/0);
  ASSERT_OK_AND_ASSIGN(auto session,
                       BuildFakeAsrSession(std::move(audio_source)));

  // Mismatched sample_rate_hz fails.
  ASSERT_OK(raw_input->PushInput(OmniSession::AudioInputMetadata{
      .sample_rate_hz = 8000, .num_channels = 1}));
  EXPECT_THAT(session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched num_channels fails.
  ASSERT_OK(raw_input->PushInput(OmniSession::AudioInputMetadata{
      .sample_rate_hz = 16000, .num_channels = 2}));
  EXPECT_THAT(session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // TextInput fails on an ASR session.
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "invalid"}));
  EXPECT_THAT(session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(AsrSessionTest, OverlapAndZeroPadRemainderOnFlushAndReset) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  // Interval = 3 samples, overlap = 1 sample (step = 2 samples).
  auto audio_source = CreateAudioInputSource(
      std::move(input_source), /*sample_rate_hz=*/16000, /*num_channels=*/1,
      /*samples_per_interval=*/3, /*overlap_samples=*/1);
  ASSERT_OK_AND_ASSIGN(auto session,
                       BuildFakeAsrSession(std::move(audio_source)));

  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f, 3.0f, 4.0f}}));

  // First interval consumes [1, 2, 3], retaining overlap [3] + remaining [4].
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).unconfirmed_text, "w_1 w_2 w_3");

  // Flush() zero-pads the remaining [3, 4] to [3, 4, 0] and flushes the merger.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(flushed));
  EXPECT_EQ(std::get<TextOutput>(flushed).confirmed_text,
            "w_1 w_2 w_3 w_4 w_0");

  // Reset() allows a new stream on the same session.
  session->Reset();
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {7.0f, 8.0f, 9.0f}}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed2, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(flushed2));
  EXPECT_EQ(std::get<TextOutput>(flushed2).confirmed_text,
            "w_7 w_8 w_9 w_0 w_0");
}

TEST_F(AsrSessionTest, ProcessAsyncWithEndOfInputAndRemainderPadding) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto audio_source = CreateAudioInputSource(
      std::move(input_source), /*sample_rate_hz=*/16000, /*num_channels=*/1,
      /*samples_per_interval=*/3, /*overlap_samples=*/0);

  ::litert::lm::ThreadPool pool("asr_omni_test_pool", 2);
  ASSERT_OK_AND_ASSIGN(auto session,
                       BuildFakeAsrSession(std::move(audio_source), &pool));

  // Push 2 samples (< interval=3) followed by EndOfInput. ScheduleInternal()
  // pads the remainder to [5, 6, 0], and EOS flushes the text merger.
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {5.0f, 6.0f}}));
  raw_input->Finish();

  std::vector<TextOutput> outputs;
  absl::Status terminal_status;
  absl::Notification done;

  ASSERT_OK(session->ProcessAsync(
      [&](absl::StatusOr<OmniSession::Output> res) -> absl::Status {
        if (!res.ok()) {
          terminal_status = res.status();
          done.Notify();
          return res.status();
        }
        if (const auto* text = std::get_if<TextOutput>(&*res)) {
          outputs.push_back(*text);
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  EXPECT_THAT(terminal_status, StatusIs(absl::StatusCode::kOutOfRange));
  ASSERT_FALSE(outputs.empty());
  EXPECT_EQ(outputs.back().confirmed_text, "w_5 w_6 w_0");
}

}  // namespace
}  // namespace litert::omni::asr
