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
#include "omni/asr/asr_omni_session.h"
#include "omni/asr/asr_session.h"
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/detokenizer.h"
#include "omni/asr/levenshtein_text_merger.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/text_source.h"
#include "omni/tts/tts_omni_session.h"
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
    return asr::AsrOmniSessionFactory::CreateAudioInputSource(
        std::move(input_source), sample_rate_hz, num_channels,
        samples_per_interval, overlap_samples);
  }

  static std::unique_ptr<tts::StreamTextSource> CreateTextInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      tts::TextChunkConfig config = {}) {
    return tts::TtsOmniSessionFactory::CreateTextInputSource(
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
  absl::Status Flush() override { return absl::OkStatus(); }

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
    tts::TtsSession::Components components{
        .text_source = std::move(text_source),
        .vocoder = std::move(vocoder),
    };
    return tts::TtsSession::Create(std::move(components), &pool_);
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
  EXPECT_FALSE(source.NeedSchedule());

  ASSERT_TRUE(source.HasOutput());
  auto out = source.GetOutput();
  ASSERT_OK(out);
  EXPECT_EQ(std::get<OmniSession::TextInput>(*out).text, "First");

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
                       HasSubstr("/custom/asr_cache/moonshine-tiny.tflite")));
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

  asr::AsrSession::Components components{
      .audio_source = std::move(audio_source),
      .preprocessor = std::move(preprocessor),
      .speech_recognizer = std::move(recognizer),
      .detokenizer = std::move(detokenizer),
      .text_merger = std::move(text_merger),
  };
  auto asr_session = asr::AsrSession::Create(std::move(components));
  ASSERT_OK(asr_session);
  std::unique_ptr<OmniSession> omni_session = *std::move(asr_session);

  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "hello"}));
  EXPECT_THAT(omni_session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f},
                              .sample_rate_hz = 16000,
                              .num_channels = 1}));
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

  asr::AsrSession::Components components{
      .audio_source = std::move(audio_source),
      .preprocessor = std::move(preprocessor),
      .speech_recognizer = std::move(recognizer),
      .detokenizer = std::move(detokenizer),
      .text_merger = std::move(text_merger),
  };
  ::litert::lm::ThreadPool pool("test_asr_pool", 2);
  auto asr_session = asr::AsrSession::Create(std::move(components), &pool);
  ASSERT_OK(asr_session);
  std::unique_ptr<OmniSession> omni_session = *std::move(asr_session);

  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f},
                              .sample_rate_hz = 16000,
                              .num_channels = 1}));

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
  tts::TtsSession::Components components{
      .text_source = std::move(text_source),
      .vocoder = std::move(vocoder),
  };
  ::litert::lm::ThreadPool pool("test_tts_pool", 1);
  auto tts_session = tts::TtsSession::Create(std::move(components), &pool);
  ASSERT_OK(tts_session);
  std::unique_ptr<OmniSession> omni_session = *std::move(tts_session);

  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f}}));
  EXPECT_THAT(omni_session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::TextInput{.text = "Hello world."}));
  auto audio_out = omni_session->ProcessNext();
  ASSERT_OK(audio_out);
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(*audio_out));
  EXPECT_FALSE(std::get<AudioOutput>(*audio_out).pcm_samples.empty());

  auto flushed = omni_session->Flush();
  ASSERT_OK(flushed);
  EXPECT_TRUE(std::holds_alternative<AudioOutput>(*flushed));
}

}  // namespace
}  // namespace litert::omni
