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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/detokenizer.h"
#include "omni/asr/levenshtein_text_merger.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/asr/text_merger.h"
#include "omni/base/stage.h"
#include "runtime/framework/threadpool.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

// Dummy AudioSource returning pre-configured PCM audio chunks via
// Schedule/GetOutput.
class DummyAudioSource : public AudioSource {
 public:
  explicit DummyAudioSource(std::vector<std::vector<float>> chunks)
      : chunks_(std::move(chunks)) {}

  int GetSampleRateHz() const override { return 16000; }
  int GetNumChannels() const override { return 1; }

 protected:
  void ResetInternal() override { chunk_index_ = 0; }

  bool NeedScheduleInternal() const override {
    return chunk_index_ < chunks_.size();
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    if (chunk_index_ >= chunks_.size()) {
      return absl::OutOfRangeError("End of audio stream reached.");
    }
    PushOutput(chunks_[chunk_index_++]);
    return absl::OkStatus();
  }

 private:
  std::vector<std::vector<float>> chunks_;
  size_t chunk_index_ = 0;
};

// Dummy AudioPreprocessor pulling PCM samples from audio source.
class DummyAudioPreprocessor : public AudioPreprocessor {
 public:
  explicit DummyAudioPreprocessor(
      Stage<std::vector<float>>* absl_nonnull audio_source)
      : AudioPreprocessor(audio_source) {}

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

// Dummy SpeechRecognizer pulling mel features from preprocessor.
class DummySpeechRecognizer : public SpeechRecognizer {
 public:
  explicit DummySpeechRecognizer(
      Stage<std::vector<float>>* absl_nonnull audio_preprocessor)
      : SpeechRecognizer(audio_preprocessor) {}

 protected:
  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

    auto mel_features = audio_preprocessor_.GetOutput();
    if (absl::IsNotFound(mel_features.status())) {
      return absl::OkStatus();
    } else if (!mel_features.ok()) {
      return mel_features.status();
    }

    std::vector<SpeechRecognizer::DecodedToken> tokens;
    tokens.reserve(mel_features->size());
    for (size_t i = 0; i < mel_features->size(); ++i) {
      tokens.push_back({static_cast<int>((*mel_features)[i]), 100});
    }
    PushOutput(std::move(tokens));
    return absl::OkStatus();
  }
};

// Dummy Detokenizer pulling decoded tokens from recognizer.
class DummyDetokenizer : public Detokenizer {
 public:
  explicit DummyDetokenizer(
      Stage<std::vector<SpeechRecognizer::DecodedToken>>* absl_nonnull
          recognizer)
      : Detokenizer(recognizer) {}

 protected:
  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

    auto tokens = speech_recognizer_.GetOutput();
    if (absl::IsNotFound(tokens.status())) {
      return absl::OkStatus();
    } else if (!tokens.ok()) {
      return tokens.status();
    }

    std::vector<Detokenizer::Word> words;
    words.reserve(tokens->size());
    for (const auto& tok : *tokens) {
      words.push_back({"w_" + std::to_string(tok.token_id), tok.timestamp_ms});
    }
    PushOutput(std::move(words));
    return absl::OkStatus();
  }
};

TEST(AsrSessionTest, FullSessionEndToEndFlow) {
  std::vector<std::vector<float>> chunks = {
      {1.0f, 2.0f},
      {2.0f, 3.0f},
  };

  auto audio_source = std::make_unique<DummyAudioSource>(chunks);
  auto preprocessor =
      std::make_unique<DummyAudioPreprocessor>(audio_source.get());
  auto speech_recognizer =
      std::make_unique<DummySpeechRecognizer>(preprocessor.get());
  auto detokenizer =
      std::make_unique<DummyDetokenizer>(speech_recognizer.get());
  auto text_merger = std::make_unique<LevenshteinTextMerger>(detokenizer.get());

  AsrSession::Components components;
  components.audio_source = std::move(audio_source);
  components.preprocessor = std::move(preprocessor);
  components.speech_recognizer = std::move(speech_recognizer);
  components.detokenizer = std::move(detokenizer);
  components.text_merger = std::move(text_merger);

  auto session_status = AsrSession::Create(std::move(components));
  ASSERT_OK(session_status);
  auto session = std::move(*session_status);

  // Process Chunk 1: "w_1 w_2"
  auto res1 = session->ProcessNextChunk();
  ASSERT_OK(res1);
  EXPECT_EQ(res1->confirmed_text, "");
  EXPECT_EQ(res1->unconfirmed_text, "w_1 w_2");

  // Process Chunk 2: "w_2 w_3" (overlaps at w_2)
  auto res2 = session->ProcessNextChunk();
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "w_1");
  EXPECT_EQ(res2->unconfirmed_text, "w_2 w_3");

  // Stream End returns OutOfRange error
  auto res3 = session->ProcessNextChunk();
  EXPECT_TRUE(absl::IsOutOfRange(res3.status()));

  // Flush remaining
  auto res_flush = session->Flush();
  ASSERT_OK(res_flush);
  EXPECT_EQ(res_flush->confirmed_text, "w_2 w_3");
  EXPECT_EQ(res_flush->unconfirmed_text, "");
}

TEST(AsrSessionTest, ProcessAsyncFlow) {
  std::vector<std::vector<float>> chunks = {
      {1.0f, 2.0f},
      {2.0f, 3.0f},
  };

  auto audio_source = std::make_unique<DummyAudioSource>(chunks);
  auto preprocessor =
      std::make_unique<DummyAudioPreprocessor>(audio_source.get());
  auto speech_recognizer =
      std::make_unique<DummySpeechRecognizer>(preprocessor.get());
  auto detokenizer =
      std::make_unique<DummyDetokenizer>(speech_recognizer.get());
  auto text_merger = std::make_unique<LevenshteinTextMerger>(detokenizer.get());

  AsrSession::Components components;
  components.audio_source = std::move(audio_source);
  components.preprocessor = std::move(preprocessor);
  components.speech_recognizer = std::move(speech_recognizer);
  components.detokenizer = std::move(detokenizer);
  components.text_merger = std::move(text_merger);

  auto session_status = AsrSession::Create(std::move(components));
  ASSERT_OK(session_status);
  auto session = std::move(*session_status);

  ::litert::lm::ThreadPool pool("test_pool", 4);
  std::vector<TextMerger::MergeResult> results;
  absl::Notification done;

  ASSERT_TRUE(session
                  ->ProcessAsync(pool,
                                 [&results, &done](auto res) -> absl::Status {
                                   if (!res.ok()) {
                                     done.Notify();
                                     return res.status();
                                   }
                                   results.push_back(*res);
                                   return absl::OkStatus();
                                 })
                  .ok());

  done.WaitForNotification();

  ASSERT_EQ(results.size(), 3);
  EXPECT_EQ(results[0].confirmed_text, "");
  EXPECT_EQ(results[0].unconfirmed_text, "w_1 w_2");
  EXPECT_EQ(results[1].confirmed_text, "w_1");
  EXPECT_EQ(results[1].unconfirmed_text, "w_2 w_3");
  EXPECT_EQ(results[2].confirmed_text, "w_2 w_3");
  EXPECT_EQ(results[2].unconfirmed_text, "");
}

TEST(AsrSessionTest, MultipleProcessAsyncCallsInSequence) {
  std::vector<std::vector<float>> chunks = {
      {1.0f, 2.0f},
  };

  auto audio_source = std::make_unique<DummyAudioSource>(chunks);
  auto preprocessor =
      std::make_unique<DummyAudioPreprocessor>(audio_source.get());
  auto speech_recognizer =
      std::make_unique<DummySpeechRecognizer>(preprocessor.get());
  auto detokenizer =
      std::make_unique<DummyDetokenizer>(speech_recognizer.get());
  auto text_merger = std::make_unique<LevenshteinTextMerger>(detokenizer.get());

  AsrSession::Components components;
  components.audio_source = std::move(audio_source);
  components.preprocessor = std::move(preprocessor);
  components.speech_recognizer = std::move(speech_recognizer);
  components.detokenizer = std::move(detokenizer);
  components.text_merger = std::move(text_merger);

  auto session_status = AsrSession::Create(std::move(components));
  ASSERT_OK(session_status);
  auto session = std::move(*session_status);

  ::litert::lm::ThreadPool pool("test_pool", 4);

  // First run
  absl::Notification done1;
  ASSERT_TRUE(session
                  ->ProcessAsync(pool,
                                 [&done1](auto res) -> absl::Status {
                                   if (!res.ok()) {
                                     done1.Notify();
                                     return res.status();
                                   }
                                   return absl::OkStatus();
                                 })
                  .ok());
  done1.WaitForNotification();

  // Reset session and run second time
  session->Reset();
  absl::Notification done2;
  ASSERT_TRUE(session
                  ->ProcessAsync(pool,
                                 [&done2](auto res) -> absl::Status {
                                   if (!res.ok()) {
                                     done2.Notify();
                                     return res.status();
                                   }
                                   return absl::OkStatus();
                                 })
                  .ok());
  done2.WaitForNotification();
}

TEST(AsrSessionTest, RejectsConcurrentProcessAsyncCalls) {
  std::vector<std::vector<float>> chunks = {
      {1.0f, 2.0f},
  };

  auto audio_source = std::make_unique<DummyAudioSource>(chunks);
  auto preprocessor =
      std::make_unique<DummyAudioPreprocessor>(audio_source.get());
  auto speech_recognizer =
      std::make_unique<DummySpeechRecognizer>(preprocessor.get());
  auto detokenizer =
      std::make_unique<DummyDetokenizer>(speech_recognizer.get());
  auto text_merger = std::make_unique<LevenshteinTextMerger>(detokenizer.get());

  AsrSession::Components components;
  components.audio_source = std::move(audio_source);
  components.preprocessor = std::move(preprocessor);
  components.speech_recognizer = std::move(speech_recognizer);
  components.detokenizer = std::move(detokenizer);
  components.text_merger = std::move(text_merger);

  auto session_status = AsrSession::Create(std::move(components));
  ASSERT_OK(session_status);
  auto session = std::move(*session_status);

  ::litert::lm::ThreadPool pool("test_pool", 4);

  absl::Notification done;
  ASSERT_TRUE(session
                  ->ProcessAsync(pool,
                                 [&done](auto res) -> absl::Status {
                                   if (!res.ok()) {
                                     done.Notify();
                                     return res.status();
                                   }
                                   return absl::OkStatus();
                                 })
                  .ok());

  // Second call while first is running should fail
  auto status2 = session->ProcessAsync(
      pool, [](auto res) -> absl::Status { return absl::OkStatus(); });
  EXPECT_FALSE(status2.ok());

  done.WaitForNotification();
}

TEST(AsrSessionTest, DestroySessionSafelyDuringProcessAsync) {
  std::vector<std::vector<float>> chunks = {
      {1.0f, 2.0f},
      {2.0f, 3.0f},
  };

  auto audio_source = std::make_unique<DummyAudioSource>(chunks);
  auto preprocessor =
      std::make_unique<DummyAudioPreprocessor>(audio_source.get());
  auto speech_recognizer =
      std::make_unique<DummySpeechRecognizer>(preprocessor.get());
  auto detokenizer =
      std::make_unique<DummyDetokenizer>(speech_recognizer.get());
  auto text_merger = std::make_unique<LevenshteinTextMerger>(detokenizer.get());

  AsrSession::Components components;
  components.audio_source = std::move(audio_source);
  components.preprocessor = std::move(preprocessor);
  components.speech_recognizer = std::move(speech_recognizer);
  components.detokenizer = std::move(detokenizer);
  components.text_merger = std::move(text_merger);

  auto session_status = AsrSession::Create(std::move(components));
  ASSERT_OK(session_status);
  auto session = std::move(*session_status);

  ::litert::lm::ThreadPool pool("test_pool", 4);

  // Return error on callback to stop, then destroy session immediately.
  ASSERT_TRUE(session
                  ->ProcessAsync(pool,
                                 [](auto res) -> absl::Status {
                                   return absl::InternalError("Stop requested");
                                 })
                  .ok());

  // Destroy session immediately while worker tasks may be in flight
  session.reset();
  EXPECT_TRUE(pool.WaitUntilIdle(absl::Seconds(1)).ok());
}

TEST(AsrSessionTest, FailsWhenMissingComponent) {
  AsrSession::Components components;
  components.audio_source =
      std::make_unique<DummyAudioSource>(std::vector<std::vector<float>>{});
  // Intentionally leave preprocessor null

  auto session_status = AsrSession::Create(std::move(components));
  EXPECT_FALSE(session_status.ok());
}

}  // namespace
}  // namespace litert::omni::asr
