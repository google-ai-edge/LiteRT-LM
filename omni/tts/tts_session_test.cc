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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/latent_decoder.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_frontend.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

class DummyTextFrontend : public TextFrontend {
 public:
  explicit DummyTextFrontend(Stage<std::string>* absl_nonnull text_source)
      : TextFrontend(text_source) {}

  void Reset() override {
    absl::MutexLock lock(mutex_);
    outputs_.clear();
  }

 protected:
  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto text_chunk = text_source_.GetOutput();
    if (absl::IsNotFound(text_chunk.status())) {
      return absl::OkStatus();
    } else if (!text_chunk.ok()) {
      return text_chunk.status();
    }
    FrontendOutput out;
    out.token_ids = {10, 20};
    PushOutput(std::move(out));
    return absl::OkStatus();
  }
};

class DummyAcousticPredictor : public AcousticPredictor {
 public:
  explicit DummyAcousticPredictor(
      Stage<TextFrontend::FrontendOutput>* absl_nonnull text_frontend)
      : AcousticPredictor(text_frontend) {}

  void Reset() override {
    absl::MutexLock lock(mutex_);
    outputs_.clear();
  }

 protected:
  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto frontend_out = text_frontend_.GetOutput();
    if (absl::IsNotFound(frontend_out.status())) {
      return absl::OkStatus();
    } else if (!frontend_out.ok()) {
      return frontend_out.status();
    }
    AcousticOutput out;
    out.rvq_frames.push_back({1, 2, 3});
    PushOutput(std::move(out));
    return absl::OkStatus();
  }
};

class DummyLatentDecoder : public LatentDecoder {
 public:
  explicit DummyLatentDecoder(
      Stage<AcousticPredictor::AcousticOutput>* absl_nonnull acoustic_predictor)
      : LatentDecoder(acoustic_predictor) {}

  void Reset() override {
    absl::MutexLock lock(mutex_);
    outputs_.clear();
  }

 protected:
  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto acoustic_out = acoustic_predictor_.GetOutput();
    if (absl::IsNotFound(acoustic_out.status())) {
      return absl::OkStatus();
    } else if (!acoustic_out.ok()) {
      return acoustic_out.status();
    }
    LatentOutput out;
    out.codec_features = {0.1f, 0.2f};
    out.rvq_frames = acoustic_out->rvq_frames;
    PushOutput(std::move(out));
    return absl::OkStatus();
  }
};

class DummyVocoder : public Vocoder {
 public:
  explicit DummyVocoder(
      Stage<LatentDecoder::LatentOutput>* absl_nonnull latent_decoder)
      : Vocoder(latent_decoder) {}

  void Reset() override {
    absl::MutexLock lock(mutex_);
    outputs_.clear();
    has_pending_audio_ = false;
  }

  absl::Status Flush() override {
    absl::MutexLock lock(mutex_);
    if (has_pending_audio_) {
      outputs_.push_back({{0.5f, -0.5f}, 24000});
      has_pending_audio_ = false;
    }
    return absl::OkStatus();
  }

 protected:
  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto latent_out = latent_decoder_.GetOutput();
    if (absl::IsNotFound(latent_out.status())) {
      return absl::OkStatus();
    } else if (!latent_out.ok()) {
      return latent_out.status();
    }
    AudioOutput out;
    out.pcm_samples = {0.1f, 0.2f, 0.3f};
    out.sample_rate_hz = 24000;
    has_pending_audio_ = true;
    PushOutput(std::move(out));
    return absl::OkStatus();
  }

 private:
  bool has_pending_audio_ = false;
};

TtsSession::Components CreateStreamComponents() {
  auto source = std::make_unique<StreamTextSource>();
  auto frontend = std::make_unique<DummyTextFrontend>(source.get());
  auto acoustic = std::make_unique<DummyAcousticPredictor>(frontend.get());
  auto latent = std::make_unique<DummyLatentDecoder>(acoustic.get());
  auto vocoder = std::make_unique<DummyVocoder>(latent.get());

  return TtsSession::Components{
      std::move(source), std::move(frontend), std::move(acoustic),
      std::move(latent), std::move(vocoder),
  };
}

TEST(TtsSessionTest, Synthesize) {
  ::litert::lm::ThreadPool thread_pool("tts_test_pool", 4);
  ASSERT_OK_AND_ASSIGN(
      auto session, TtsSession::Create(CreateStreamComponents(), &thread_pool));

  ASSERT_OK_AND_ASSIGN(auto audio, session->Synthesize("Hello world."));
  EXPECT_EQ(audio.sample_rate_hz, 24000);
  EXPECT_FALSE(audio.pcm_samples.empty());
}

TEST(TtsSessionTest, SynthesizeAsync) {
  ::litert::lm::ThreadPool thread_pool("tts_test_pool", 4);
  ASSERT_OK_AND_ASSIGN(
      auto session, TtsSession::Create(CreateStreamComponents(), &thread_pool));

  absl::Notification done;
  int chunk_count = 0;
  absl::Status final_status;

  absl::Status status = session->SynthesizeAsync(
      "Hello world.", [&](absl::StatusOr<AudioOutput> result) -> absl::Status {
        if (!result.ok()) {
          final_status = result.status();
          done.Notify();
          return result.status();
        }
        chunk_count++;
        return absl::OkStatus();
      });
  ASSERT_TRUE(status.ok());

  done.WaitForNotification();
  EXPECT_TRUE(absl::IsOutOfRange(final_status))
      << "final_status was: " << final_status;
  EXPECT_GT(chunk_count, 0);
}

TEST(TtsSessionTest, SequentialSynthesizeCalls) {
  ::litert::lm::ThreadPool thread_pool("tts_test_pool", 4);
  ASSERT_OK_AND_ASSIGN(
      auto session, TtsSession::Create(CreateStreamComponents(), &thread_pool));

  ASSERT_OK_AND_ASSIGN(auto audio1, session->Synthesize("First chunk."));
  EXPECT_FALSE(audio1.pcm_samples.empty());

  ASSERT_OK_AND_ASSIGN(auto audio2, session->Synthesize("Second chunk."));
  EXPECT_FALSE(audio2.pcm_samples.empty());
}

TEST(TtsSessionTest, ResetAndFlush) {
  ::litert::lm::ThreadPool thread_pool("tts_test_pool", 4);
  ASSERT_OK_AND_ASSIGN(
      auto session, TtsSession::Create(CreateStreamComponents(), &thread_pool));

  session->Reset();
  ASSERT_OK_AND_ASSIGN(auto audio, session->Flush());
  EXPECT_TRUE(audio.pcm_samples.empty());
}

}  // namespace
}  // namespace litert::omni::tts
