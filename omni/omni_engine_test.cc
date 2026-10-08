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

#include "omni/omni_engine.h"

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
#include "omni/asr/asr_session.h"
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/detokenizer.h"
#include "omni/asr/levenshtein_text_merger.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_session.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/text_source.h"
#include "omni/tts/tts_session.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {

class OmniSessionTest : public ::testing::Test {
 public:
  static std::unique_ptr<asr::AudioSource> CreateAudioInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      int sample_rate_hz, int num_channels, int samples_per_interval,
      int overlap_samples) {
    return asr::AsrSessionFactory::CreateAudioInputSource(
        std::move(input_source), sample_rate_hz, num_channels,
        samples_per_interval, overlap_samples);
  }

  static std::unique_ptr<tts::StreamTextSource> CreateTextInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      tts::TextChunkConfig config = {}) {
    return tts::TtsSessionFactory::CreateTextInputSource(
        std::move(input_source), std::move(config));
  }
};

namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class FakeAudioPreprocessor : public asr::AudioPreprocessor {
 public:
  explicit FakeAudioPreprocessor(asr::AudioSource* source)
      : asr::AudioPreprocessor(source) {}

 protected:
  absl::Status ScheduleInternal() override {
    auto pcm_samples = audio_source_.GetOutput();
    if (absl::IsNotFound(pcm_samples.status())) {
      return absl::OkStatus();
    } else if (!pcm_samples.ok()) {
      return pcm_samples.status();
    }
    PushOutput(std::move(*pcm_samples));
    SetState(State::kIdle);
    return absl::OkStatus();
  }
};

class FakeSpeechRecognizer : public asr::SpeechRecognizer {
 public:
  explicit FakeSpeechRecognizer(asr::AudioPreprocessor* preprocessor)
      : asr::SpeechRecognizer(preprocessor) {}

 protected:
  absl::Status ScheduleInternal() override {
    SetState(State::kIdle);
    auto mel_features = audio_preprocessor_.GetOutput();
    if (absl::IsNotFound(mel_features.status())) {
      return absl::OkStatus();
    } else if (!mel_features.ok()) {
      return mel_features.status();
    }
    std::vector<asr::SpeechRecognizer::DecodedToken> tokens;
    for (float val : *mel_features) {
      tokens.push_back({static_cast<int>(val), 100});
    }
    PushOutput(std::move(tokens));
    return absl::OkStatus();
  }
};

class FakeDetokenizer : public asr::Detokenizer {
 public:
  explicit FakeDetokenizer(asr::SpeechRecognizer* recognizer)
      : asr::Detokenizer(recognizer) {}

 protected:
  absl::Status ScheduleInternal() override {
    SetState(State::kIdle);
    auto tokens = speech_recognizer_.GetOutput();
    if (absl::IsNotFound(tokens.status())) {
      return absl::OkStatus();
    } else if (!tokens.ok()) {
      return tokens.status();
    }
    std::vector<asr::Detokenizer::Word> words;
    for (const auto& tok : *tokens) {
      words.push_back({"w_" + std::to_string(tok.token_id), tok.timestamp_ms});
    }
    PushOutput(std::move(words));
    return absl::OkStatus();
  }
};

class FakeVocoder : public tts::Vocoder {
 public:
  explicit FakeVocoder(tts::TextSource* text_source)
      : text_source_(text_source) {}

 protected:
  bool NeedScheduleInternal() const override {
    return text_source_->HasOutput();
  }
  absl::Status ScheduleInternal() override {
    SetState(State::kIdle);
    auto text = text_source_->GetOutput();
    if (absl::IsNotFound(text.status())) {
      return absl::OkStatus();
    } else if (!text.ok()) {
      return text.status();
    }
    PushOutput(
        AudioOutput{.pcm_samples = {0.25f, 0.5f}, .sample_rate_hz = 24000});
    return absl::OkStatus();
  }

 private:
  tts::TextSource* text_source_;
};

class FakeOmniSessionFactory : public OmniSessionFactory {
 public:
  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override {
    auto text_source =
        OmniSessionTest::CreateTextInputSource(std::move(input_source));
    auto vocoder = std::make_unique<FakeVocoder>(text_source.get());
    Stage<Output>* raw_output_stage = vocoder.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(text_source));
    stages.push_back(std::move(vocoder));
    return MultiStagedSession::Create(std::move(stages), raw_output_stage,
                                      &pool_);
  }

 private:
  ::litert::lm::ThreadPool pool_{"fake_factory_pool", 1};
};

TEST(PushInputSourceTest, PushScheduleAndReset) {
  PushInputSource source;
  EXPECT_FALSE(source.NeedSchedule());
  ASSERT_OK(source.Schedule());

  ASSERT_OK(source.PushInput(OmniSession::TextInput{.text = "First"}));
  ASSERT_OK(source.PushInput(OmniSession::TextInput{.text = "Second"}));
  source.Finish();
  EXPECT_FALSE(source.NeedSchedule());

  ASSERT_TRUE(source.HasOutput());
  auto out1 = source.GetOutput();
  ASSERT_OK(out1);
  EXPECT_EQ(std::get<OmniSession::TextInput>(*out1).text, "First");

  auto out2 = source.GetOutput();
  ASSERT_OK(out2);
  EXPECT_EQ(std::get<OmniSession::TextInput>(*out2).text, "Second");

  auto out3 = source.GetOutput();
  ASSERT_OK(out3);
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfInput>(*out3));

  source.Reset();
  EXPECT_FALSE(source.NeedSchedule());
  EXPECT_FALSE(source.HasOutput());
}

TEST(OmniEngineTest, CreateSessionDelegatesToFactory) {
  auto engine = OmniEngine::Create("test-model",
                                   std::make_unique<FakeOmniSessionFactory>());
  ASSERT_OK(engine);
  EXPECT_EQ((*engine)->model_name(), "test-model");

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto session = (*engine)->CreateSession(std::move(input_source));
  ASSERT_OK(session);

  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "Hello."}));
  auto out = (*session)->ProcessNext();
  ASSERT_OK(out);
  EXPECT_TRUE(std::holds_alternative<AudioOutput>(*out));
}

TEST(OmniEngineTest, ResolvesAsrModelAndForwardsOptions) {
  OmniEngine::Options options{
      .backend = OmniEngine::Options::Backend::kGpu,
      .cache_dir = "/custom/asr_cache",
      .num_threads = 8,
  };
  EXPECT_THAT(OmniEngine::Create("moonshine-tiny", options),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("/custom/asr_cache/moonshine-tiny.litertlm")));
}

TEST(OmniEngineTest, ResolvesTtsModelAndForwardsOptions) {
  OmniEngine::Options options{
      .backend = OmniEngine::Options::Backend::kNpu,
      .cache_dir = "/custom/tts_cache",
      .num_threads = 6,
  };
  EXPECT_THAT(OmniEngine::Create("kokoro", options),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("/custom/tts_cache/kokoro")));
}

TEST(OmniEngineTest, ResolvesText2ImageModelAndForwardsOptions) {
  OmniEngine::Options options{
      .backend = OmniEngine::Options::Backend::kGpu,
      .cache_dir = "/custom/text2image_cache",
      .num_threads = 4,
  };
  EXPECT_THAT(OmniEngine::Create("bonsai-flux2", options),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("/custom/text2image_cache/bonsai-flux2")));
}

TEST(OmniEngineTest, RejectsUnknownModel) {
  EXPECT_THAT(
      OmniEngine::Create("unknown-model-xyz"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("unknown-model-xyz")));
}

TEST_F(OmniSessionTest, AsrSessionProcessNextAndFlushWithAudioInput) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto audio_source =
      CreateAudioInputSource(std::move(input_source),
                             /*sample_rate_hz=*/16000, /*num_channels=*/1,
                             /*samples_per_interval=*/2, /*overlap_samples=*/0);
  auto preprocessor =
      std::make_unique<FakeAudioPreprocessor>(audio_source.get());
  auto recognizer = std::make_unique<FakeSpeechRecognizer>(preprocessor.get());
  auto detokenizer = std::make_unique<FakeDetokenizer>(recognizer.get());
  auto text_merger =
      std::make_unique<asr::LevenshteinTextMerger>(detokenizer.get());
  Stage<Output>* raw_output_stage = text_merger.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(audio_source));
  stages.push_back(std::move(preprocessor));
  stages.push_back(std::move(recognizer));
  stages.push_back(std::move(detokenizer));
  stages.push_back(std::move(text_merger));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSession> omni_session,
      MultiStagedSession::Create(std::move(stages), raw_output_stage));

  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "hello"}));
  EXPECT_THAT(omni_session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK(raw_input_source->PushInput(OmniSession::AudioInputMetadata{
      .sample_rate_hz = 16000, .num_channels = 1}));
  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f}}));
  auto chunk = omni_session->ProcessNext();
  ASSERT_OK(chunk);
  ASSERT_TRUE(std::holds_alternative<OmniSession::TextOutput>(*chunk));
  EXPECT_EQ(std::get<OmniSession::TextOutput>(*chunk).unconfirmed_text,
            "w_1 w_2");

  auto flushed = omni_session->Flush();
  ASSERT_OK(flushed);
  ASSERT_TRUE(std::holds_alternative<OmniSession::TextOutput>(*flushed));
  EXPECT_EQ(std::get<OmniSession::TextOutput>(*flushed).confirmed_text,
            "w_1 w_2");
}

TEST_F(OmniSessionTest, AsrSessionProcessAsyncWithAudioInput) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto audio_source =
      CreateAudioInputSource(std::move(input_source),
                             /*sample_rate_hz=*/16000, /*num_channels=*/1,
                             /*samples_per_interval=*/2, /*overlap_samples=*/0);
  auto preprocessor =
      std::make_unique<FakeAudioPreprocessor>(audio_source.get());
  auto recognizer = std::make_unique<FakeSpeechRecognizer>(preprocessor.get());
  auto detokenizer = std::make_unique<FakeDetokenizer>(recognizer.get());
  auto text_merger =
      std::make_unique<asr::LevenshteinTextMerger>(detokenizer.get());
  Stage<Output>* raw_output_stage = text_merger.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(audio_source));
  stages.push_back(std::move(preprocessor));
  stages.push_back(std::move(recognizer));
  stages.push_back(std::move(detokenizer));
  stages.push_back(std::move(text_merger));

  ::litert::lm::ThreadPool pool("test_asr_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSession> omni_session,
      MultiStagedSession::Create(std::move(stages), raw_output_stage, &pool));

  ASSERT_OK(raw_input_source->PushInput(OmniSession::AudioInputMetadata{
      .sample_rate_hz = 16000, .num_channels = 1}));
  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f}}));

  absl::Notification done;
  std::vector<OmniSession::TextOutput> outputs;
  ASSERT_OK(omni_session->ProcessAsync(
      [&](absl::StatusOr<OmniSession::Output> res) {
        if (absl::IsOutOfRange(res.status())) {
          done.Notify();
          return res.status();
        }
        if (res.ok() && std::holds_alternative<OmniSession::TextOutput>(*res)) {
          outputs.push_back(std::get<OmniSession::TextOutput>(*res));
        }
        return absl::OkStatus();
      }));
  done.WaitForNotification();
  ASSERT_FALSE(outputs.empty());
  EXPECT_EQ(outputs.back().confirmed_text, "w_1 w_2");
}

TEST_F(OmniSessionTest, TtsSessionProcessNextAndFlushWithTextInput) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto text_source = CreateTextInputSource(std::move(input_source));
  auto vocoder = std::make_unique<FakeVocoder>(text_source.get());
  Stage<Output>* raw_output_stage = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ::litert::lm::ThreadPool pool("test_tts_pool", 1);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSession> omni_session,
      MultiStagedSession::Create(std::move(stages), raw_output_stage, &pool));

  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f}}));
  EXPECT_THAT(omni_session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::TextInput{.text = "Hello world. Trailing"}));
  auto audio_out = omni_session->ProcessNext();
  ASSERT_OK(audio_out);
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(*audio_out));
  EXPECT_FALSE(std::get<AudioOutput>(*audio_out).pcm_samples.empty());

  auto flushed = omni_session->Flush();
  ASSERT_OK(flushed);
  EXPECT_TRUE(std::holds_alternative<AudioOutput>(*flushed));

  auto empty_flushed = omni_session->Flush();
  ASSERT_OK(empty_flushed);
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutput>(*empty_flushed));
}

TEST_F(OmniSessionTest, MultiStagedSessionFlushDrainsStages) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto text_source = CreateTextInputSource(std::move(input_source));
  auto vocoder = std::make_unique<FakeVocoder>(text_source.get());
  Stage<Output>* raw_output_stage = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ::litert::lm::ThreadPool pool("test_multi_staged_pool", 1);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      MultiStagedSession::Create(std::move(stages), raw_output_stage, &pool));

  // Push text without a sentence delimiter so `TextInputSource` will only emit
  // it once `Flush()` is called.
  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::TextInput{.text = "Trailing text without delimiter"}));

  ASSERT_OK_AND_ASSIGN(auto flushed, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(flushed));
  EXPECT_FALSE(std::get<AudioOutput>(flushed).pcm_samples.empty());
  for (const auto& stage : session->stages()) {
    EXPECT_TRUE(stage->IsIdle());
    EXPECT_FALSE(stage->NeedSchedule());
  }
}

TEST_F(OmniSessionTest, MultiStagedSessionAsrFlushDrainsStages) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto audio_source =
      CreateAudioInputSource(std::move(input_source),
                             /*sample_rate_hz=*/16000, /*num_channels=*/1,
                             /*samples_per_interval=*/4, /*overlap_samples=*/0);
  auto preprocessor =
      std::make_unique<FakeAudioPreprocessor>(audio_source.get());
  auto recognizer = std::make_unique<FakeSpeechRecognizer>(preprocessor.get());
  auto detokenizer = std::make_unique<FakeDetokenizer>(recognizer.get());
  auto text_merger =
      std::make_unique<asr::LevenshteinTextMerger>(detokenizer.get());
  Stage<Output>* raw_output_stage = text_merger.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(audio_source));
  stages.push_back(std::move(preprocessor));
  stages.push_back(std::move(recognizer));
  stages.push_back(std::move(detokenizer));
  stages.push_back(std::move(text_merger));

  ::litert::lm::ThreadPool pool("test_multi_staged_asr_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      MultiStagedSession::Create(std::move(stages), raw_output_stage, &pool));

  // Push partial audio (2 samples < samples_per_interval=4). `Flush()` causes
  // `AudioInputSource` to zero-pad and emit the final chunk, then schedules
  // and flushes all stages until idle.
  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f}}));

  ASSERT_OK_AND_ASSIGN(auto flushed, session->Flush());
  ASSERT_TRUE(std::holds_alternative<OmniSession::TextOutput>(flushed));
  EXPECT_EQ(std::get<OmniSession::TextOutput>(flushed).confirmed_text,
            "w_1 w_2 w_0 w_0");
  for (const auto& stage : session->stages()) {
    EXPECT_TRUE(stage->IsIdle());
    EXPECT_FALSE(stage->NeedSchedule());
  }
}

TEST_F(OmniSessionTest, ProcessAsyncFailsWithoutThreadPool) {
  auto input_source = std::make_unique<PushInputSource>();
  auto text_source = CreateTextInputSource(std::move(input_source));
  auto vocoder = std::make_unique<FakeVocoder>(text_source.get());
  Stage<Output>* raw_output_stage = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), raw_output_stage));
  EXPECT_THAT(session->ProcessAsync(
                  [](absl::StatusOr<Output>) { return absl::OkStatus(); }),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace litert::omni
