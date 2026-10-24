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

#include "omni/speech2text/asr_session.h"

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
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/speech2text/audio_preprocessor.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/detokenizer.h"
#include "omni/speech2text/levenshtein_text_merger.h"
#include "omni/speech2text/speech_recognizer.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "support/tokenizer/tokenizer.h"
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
using ::testing::IsSupersetOf;

TEST_F(AsrSessionTest, AllDefaultModelsAreRegistered) {
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              IsSupersetOf({"parakeet-tdt-0.6b-v3",
                            "parakeet-tdt-0.6b-v3-int8",
                            "parakeet-tdt",
                            "parakeet-tdt-int8",
                            "parakeet-ctc-0.6b",
                            "parakeet-ctc-0.6b-int8",
                            "parakeet-ctc",
                            "parakeet-ctc-int8",
                            "moonshine-tiny",
                            "moonshine-tiny-int8",
                            "moonshine",
                            "moonshine-int8",
                            "whisper-tiny",
                            "whisper-tiny-int8",
                            "whisper",
                            "whisper-int8",
                            "qwen3-asr-0.6b",
                            "qwen3-asr-0.6b-int8",
                            "qwen3-asr",
                            "qwen3-asr-int8",
                            "tinygemma-asr",
                            "tinygemma-asr-int8"}));

  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("parakeet-tdt-0.6b-v3"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("parakeet-ctc-0.6b"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("moonshine-tiny"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("whisper-tiny"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("qwen3-asr-0.6b"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("tinygemma-asr"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("unknown-model"));
}

TEST_F(AsrSessionTest, RejectsUnknownModelOnCreate) {
  EXPECT_THAT(OmniEngine::Create("unknown-model"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(AsrSessionTest, DynamicRegisterAndUnregisterModel) {
  ASSERT_OK(
      OmniRegistry::Get().RegisterSessionFactory(OmniRegistry::RegistrySpec{
          .model_names = {"custom-test-asr-ctc"},
          .get_session_factory =
              [](absl::string_view /*model_name*/,
                 const OmniEngine::Settings& /*settings*/,
                 const MessageLite* absl_nullable /*session_config*/,
                 const MessageLite* absl_nullable /*metadata*/,
                 ::litert::Environment& /*env*/, ::litert::Options& /*options*/,
                 std::shared_ptr<::litert::lm::ModelResources> absl_nullable
                 /*lm_resources*/)
              -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
            return absl::UnimplementedError("Custom session factory invoked");
          },
          .session_config_type_id = "Speech2Text",
      }));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("custom-test-asr-ctc"));

  EXPECT_TRUE(OmniRegistry::Get().UnregisterModel("custom-test-asr-ctc"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("custom-test-asr-ctc"));
}

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
    PushOutput({.features = std::move(*pcm_samples),
                .is_warm_start = IsLastInputWarmStart()});
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
    for (float val : mel_features->features) {
      tokens.push_back({static_cast<int>(val), 100});
    }
    if (mel_features->is_warm_start) {
      tokens.push_back(
          {.token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
           .is_warm_start = true});
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
      if (tok.IsEndOfChunk()) {
        words.push_back({"", tok.timestamp_ms, tok.is_warm_start});
      } else {
        words.push_back(
            {"w_" + std::to_string(tok.token_id), tok.timestamp_ms});
      }
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

  // Non-positive sample_rate_hz fails.
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInputMetadata{.sample_rate_hz = 0, .num_channels = 1}));
  EXPECT_THAT(session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-positive num_channels fails.
  ASSERT_OK(raw_input->PushInput(OmniSession::AudioInputMetadata{
      .sample_rate_hz = 16000, .num_channels = 0}));
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

TEST_F(AsrSessionTest, AudioInputSourceResamplesFromAudioInputMetadata) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  // Target config: 16 kHz, 4 samples per interval, 2 overlap samples.
  auto audio_source = CreateAudioInputSource(
      std::move(input_source), /*sample_rate_hz=*/16000, /*num_channels=*/1,
      /*samples_per_interval=*/4, /*overlap_samples=*/2);
  AudioSource* raw_audio_source = audio_source.get();
  ASSERT_OK_AND_ASSIGN(auto session,
                       BuildFakeAsrSession(std::move(audio_source)));
  EXPECT_EQ(raw_audio_source->GetSampleRateHz(), 16000);

  // Configure 24 kHz input via AudioInputMetadata: AudioInputSource resamples
  // incoming 24 kHz chunks to 16 kHz before buffering into 4-sample intervals.
  ASSERT_OK(raw_input->PushInput(OmniSession::AudioInputMetadata{
      .sample_rate_hz = 24000, .num_channels = 1}));

  // 3 samples at 24 kHz resample to 2 samples at 16 kHz (< 4 interval).
  ASSERT_OK(raw_input->PushInput(OmniSession::AudioInput{{1.0f, 1.0f, 1.0f}}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(raw_audio_source->GetSampleRateHz(), 16000);

  // Pushing 3 more samples at 24 kHz yields 2 more resampled samples (total 4),
  // completing the 4-sample interval.
  ASSERT_OK(raw_input->PushInput(OmniSession::AudioInput{{1.0f, 1.0f, 1.0f}}));
  ASSERT_OK(session->ProcessNext());

  // Reset restores the default 16 kHz input sample rate.
  session->Reset();
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{{1.0f, 2.0f, 3.0f, 4.0f}}));
  ASSERT_OK(session->ProcessNext());

  // Configure 8 kHz input after reset: 2 samples at 8 kHz upsample to 4 samples
  // at 16 kHz, completing the 4-sample interval.
  session->Reset();
  ASSERT_OK(raw_input->PushInput(OmniSession::AudioInputMetadata{
      .sample_rate_hz = 8000, .num_channels = 1}));
  ASSERT_OK(raw_input->PushInput(OmniSession::AudioInput{{1.0f, 1.0f}}));
  ASSERT_OK(session->ProcessNext());
}

TEST_F(AsrSessionTest, OverlapAndUnpaddedRemainderOnFlushAndReset) {
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

  // Flush() emits the remaining [3, 4] unpadded (the preprocessor is
  // responsible for padding to the model window) and flushes the merger,
  // without marking the stream finished if EndOfInput was not sent.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(flushed));
  EXPECT_EQ(std::get<TextOutput>(flushed).confirmed_text, "w_1 w_2 w_3 w_4");

  // Pushing more audio after Flush() continues to work without Reset().
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {5.0f, 6.0f, 7.0f}}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out2, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out2));
  EXPECT_EQ(std::get<TextOutput>(out2).unconfirmed_text, "w_5 w_6 w_7");

  // Reset() allows a new stream on the same session even after Finish().
  raw_input->Finish();
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed_after_finish,
                       session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(flushed_after_finish));
  session->Reset();
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {7.0f, 8.0f, 9.0f}}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed2, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(flushed2));
  EXPECT_EQ(std::get<TextOutput>(flushed2).confirmed_text, "w_7 w_8 w_9");
}

TEST_F(AsrSessionTest,
       PartialFragmentsAndCoalescedInputsWaitUntilIntervalOrEndOfInput) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto audio_source = CreateAudioInputSource(
      std::move(input_source), /*sample_rate_hz=*/16000, /*num_channels=*/1,
      /*samples_per_interval=*/3, /*overlap_samples=*/0);
  ASSERT_OK_AND_ASSIGN(auto session,
                       BuildFakeAsrSession(std::move(audio_source)));

  // Partial chunk (< 3 samples) before EndOfInput returns NotFoundError rather
  // than prematurely padding or returning OutOfRangeError.
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{.pcm_samples = {1.0f}}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  // Multiple consecutive chunks pushed before ProcessNext() are coalesced up to
  // samples_per_interval_ without eagerly draining subsequent queued chunks.
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{.pcm_samples = {2.0f}}));
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {3.0f, 4.0f}}));
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{.pcm_samples = {5.0f}}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).unconfirmed_text, "w_1 w_2 w_3");
  EXPECT_TRUE(raw_input->HasOutput());

  // Remaining [4.0f] + [5.0f] (< 3 samples) still waits for more input until
  // Finish().
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_FALSE(raw_input->HasOutput());

  raw_input->Finish();
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out2, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out2));
  EXPECT_EQ(std::get<TextOutput>(out2).unconfirmed_text, "w_4 w_5");

  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));
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
  // emits the unpadded remainder [5, 6], and EOS flushes the text merger.
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
  EXPECT_EQ(outputs.back().confirmed_text, "w_5 w_6");
}

TEST_F(AsrSessionTest,
       FinishWithoutNewSamplesAfterFullIntervalDoesNotEmitOverlapRemainder) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  // Interval = 3 samples, overlap = 1 sample.
  auto audio_source = CreateAudioInputSource(
      std::move(input_source), /*sample_rate_hz=*/16000, /*num_channels=*/1,
      /*samples_per_interval=*/3, /*overlap_samples=*/1);
  ASSERT_OK_AND_ASSIGN(auto session,
                       BuildFakeAsrSession(std::move(audio_source)));

  // Push exactly one full interval (3 samples), then Finish() with no new
  // samples. The retained overlap sample [3] must not be emitted as a phantom
  // remainder chunk at EOS.
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f, 3.0f}}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).unconfirmed_text, "w_1 w_2 w_3");

  raw_input->Finish();
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output eos_flush, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(eos_flush));
  EXPECT_EQ(std::get<TextOutput>(eos_flush).confirmed_text, "w_1 w_2 w_3");
  EXPECT_EQ(std::get<TextOutput>(eos_flush).unconfirmed_text, "");
}

class FakeTokenizer : public ::litert::support::Tokenizer {  // NOLINT
 public:
  ::litert::support::TokenizerType GetTokenizerType()
      const override {                                        // NOLINT
    return ::litert::support::TokenizerType::kSentencePiece;  // NOLINT
  }
  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view /*text*/) override {
    return std::vector<int>{1};
  }
  absl::StatusOr<int> TokenToId(absl::string_view /*token*/) override {
    return 1;
  }
  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> /*token_ids*/,  // NOLINT
      bool /*skip_special_tokens*/) override {
    return "tok";
  }
  std::vector<std::string> GetTokens() const override { return {"<unk>", "a"}; }
  int GetVocabSize() const override { return 2; }
};

class FakeAsrSessionFactory : public AsrSessionFactory {
 public:
  FakeAsrSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const MessageLite* absl_nullable metadata, ::litert::Environment env,
      std::shared_ptr<lm::ModelResources> absl_nullable model_resources,
      std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer,
      bool enable_warm_start = false)
      : AsrSessionFactory(settings, session_config, metadata, std::move(env),
                          std::move(model_resources), std::move(tokenizer)),
        enable_warm_start_(enable_warm_start) {}

 protected:
  bool ShouldEnableWarmStart() const override { return enable_warm_start_; }

  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithAudioSource(
      std::unique_ptr<AudioSource> absl_nonnull audio_source) override {
    return BuildFakeAsrSession(std::move(audio_source));
  }

 private:
  bool enable_warm_start_ = false;
};

TEST_F(AsrSessionTest, WarmStartEmitsProgressivePrefixesBeforeFullInterval) {
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  OmniEngine::Settings settings{.num_threads = 2};
  lm::proto::Speech2TextMetadata metadata;
  metadata.set_sample_rate_hz(8000);
  metadata.set_input_milliseconds(1);  // 8 samples per interval
  lm::proto::Speech2TextSessionConfig session_config;
  session_config.set_overlap_ratio(0.75f);  // overlap = 6, step = 2 samples
  session_config.set_text_merger_type(
      lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP);

  FakeAsrSessionFactory factory(
      settings, &session_config, &metadata, std::move(*env),
      /*model_resources=*/nullptr, std::make_unique<FakeTokenizer>(),
      /*enable_warm_start=*/true);
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(auto session, factory.Create(std::move(input_source)));

  // 1 sample (< step of 2) -> not enough for first warm-start prefix yet.
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{.pcm_samples = {1.0f}}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  // 2 samples -> first warm-start prefix [1, 2]
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{.pcm_samples = {2.0f}}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output ws1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(ws1));
  EXPECT_EQ(std::get<TextOutput>(ws1).unconfirmed_text, "w_1 w_2");

  // 2 more samples (total 4) -> second warm-start prefix [1, 2, 3, 4]
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {3.0f, 4.0f}}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output ws2, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(ws2));

  // If Finish() is called right after a warm-start emission with no new
  // samples, no redundant remainder chunk is emitted.
  raw_input->Finish();
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));

  // Resetting restores the initial warm-start step threshold (2 samples).
  session->Reset();
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{.pcm_samples = {5.0f}}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{.pcm_samples = {6.0f}}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output ws_reset, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(ws_reset));
  EXPECT_EQ(std::get<TextOutput>(ws_reset).unconfirmed_text, "w_5 w_6");
}

TEST_F(AsrSessionTest, WarmStartIsDisabledForLevenshteinTextMerger) {
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  OmniEngine::Settings settings{.num_threads = 2};
  lm::proto::Speech2TextMetadata metadata;
  metadata.set_sample_rate_hz(8000);
  metadata.set_input_milliseconds(1);  // 8 samples per interval
  lm::proto::Speech2TextSessionConfig session_config;
  session_config.set_overlap_ratio(0.75f);
  session_config.set_text_merger_type(
      lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_LEVENSHTEIN);

  FakeAsrSessionFactory factory(
      settings, &session_config, &metadata, std::move(*env),
      /*model_resources=*/nullptr, std::make_unique<FakeTokenizer>(),
      /*enable_warm_start=*/true);
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(auto session, factory.Create(std::move(input_source)));

  // With Levenshtein merger, warm-start is gated off even when the factory
  // opts in via ShouldEnableWarmStart().
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {1.0f, 2.0f}}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  raw_input->Finish();
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out));
  EXPECT_EQ(std::get<TextOutput>(out).unconfirmed_text, "w_1 w_2");
}

}  // namespace
}  // namespace litert::omni::asr
