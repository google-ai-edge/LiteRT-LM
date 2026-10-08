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

#include "omni/tts/tts_session.h"

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
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_session.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/text_source.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {

class TtsSessionTest : public ::testing::Test {
 public:
  static std::unique_ptr<StreamTextSource> CreateTextInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      TextChunkConfig config = {}) {
    return TtsSessionFactory::CreateTextInputSource(std::move(input_source),
                                                    std::move(config));
  }
};

namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

// Fake Vocoder that encodes the length of each text chunk as a single float
// sample in `AudioOutput`.
class FakeLengthVocoder : public Vocoder {
 public:
  explicit FakeLengthVocoder(TextSource* text_source)
      : text_source_(*text_source) {}

 protected:
  bool NeedScheduleInternal() const override {
    return text_source_.HasOutput();
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    auto text = text_source_.GetOutput();
    if (absl::IsNotFound(text.status())) {
      SetState(State::kIdle);
      return absl::OkStatus();
    }
    if (!text.ok()) {
      SetState(State::kIdle);
      return text.status();
    }
    PushOutput(AudioOutput{
        .pcm_samples = {static_cast<float>(text->size())},
        .sample_rate_hz = 24000,
    });
    SetState(State::kIdle);
    return absl::OkStatus();
  }

 private:
  TextSource& text_source_;
};

TEST_F(TtsSessionTest, SequentialSynthesizeCallsWithResetAndFlush) {
  auto text_source = std::make_unique<StreamTextSource>();
  StreamTextSource* raw_text_source = text_source.get();
  auto vocoder = std::make_unique<FakeLengthVocoder>(raw_text_source);
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ASSERT_OK_AND_ASSIGN(
      auto session, MultiStagedSession::Create(std::move(stages), raw_vocoder));

  // First synthesize call (mirrors JNI nativeTtsSessionSynthesize & mediapipe).
  session->Reset();
  ASSERT_OK(raw_text_source->PushText("Hello"));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out1, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out1));
  EXPECT_THAT(std::get<AudioOutput>(out1).pcm_samples, ElementsAre(5.0f));

  // Second synthesize call on the same session after Reset().
  session->Reset();
  ASSERT_OK(raw_text_source->PushText("Second call!"));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out2, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  EXPECT_THAT(std::get<AudioOutput>(out2).pcm_samples, ElementsAre(12.0f));
}

TEST_F(TtsSessionTest, TextInputSourceProcessNextFlushAndReset) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto text_source = CreateTextInputSource(std::move(input_source));
  auto vocoder = std::make_unique<FakeLengthVocoder>(text_source.get());
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ASSERT_OK_AND_ASSIGN(
      auto session, MultiStagedSession::Create(std::move(stages), raw_vocoder));

  // Non-TextInput is rejected with InvalidArgumentError.
  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {0.1f}}));
  EXPECT_THAT(session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // First sentence has a delimiter; trailing fragment does not.
  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "Hi. Tail"}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output chunk1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(chunk1));
  EXPECT_THAT(std::get<AudioOutput>(chunk1).pcm_samples, ElementsAre(3.0f));

  // Flush() finishes TextInputSource and synthesizes " Tail" (length 5).
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(flushed));
  EXPECT_THAT(std::get<AudioOutput>(flushed).pcm_samples, ElementsAre(5.0f));

  // Reset() clears the finished state so a new stream can run on the session.
  session->Reset();
  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "Again"}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed2, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(flushed2));
  EXPECT_THAT(std::get<AudioOutput>(flushed2).pcm_samples, ElementsAre(5.0f));
}

TEST_F(TtsSessionTest, TextInputSourceProcessAsyncStreamsAndFlushesTail) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto text_source = CreateTextInputSource(std::move(input_source));
  auto vocoder = std::make_unique<FakeLengthVocoder>(text_source.get());
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ::litert::lm::ThreadPool pool("tts_omni_test_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      MultiStagedSession::Create(std::move(stages), raw_vocoder, &pool));

  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "One. Two"}));
  raw_input_source->Finish();

  std::vector<float> lengths;
  absl::Status final_status;
  absl::Notification done;

  ASSERT_OK(session->ProcessAsync(
      [&](absl::StatusOr<OmniSession::Output> res) -> absl::Status {
        if (!res.ok()) {
          final_status = res.status();
          done.Notify();
          return res.status();
        }
        if (const auto* audio = std::get_if<AudioOutput>(&*res)) {
          lengths.insert(lengths.end(), audio->pcm_samples.begin(),
                         audio->pcm_samples.end());
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  EXPECT_THAT(final_status, StatusIs(absl::StatusCode::kOutOfRange));
  // "One." (4 chars) and " Two" (4 chars).
  EXPECT_THAT(lengths, ElementsAre(4.0f, 4.0f));
}

}  // namespace
}  // namespace litert::omni::tts
