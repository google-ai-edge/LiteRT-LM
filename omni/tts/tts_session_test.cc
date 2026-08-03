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

#include <cstddef>
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
#include "support/util/test_utils.h"  // from @litert  // IWYU pragma: keep
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/latent_decoder.h"
#include "omni/tts/text_frontend.h"
#include "omni/tts/text_source.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::tts {
namespace {

class DummyTextSource : public TextSource {
 public:
  explicit DummyTextSource(std::vector<std::string> chunks)
      : chunks_(std::move(chunks)) {}

  void Reset() override {
    chunk_index_ = 0;
    absl::MutexLock lock(mutex_);
    outputs_.clear();
  }

 protected:
  bool NeedScheduleInternal() const override {
    return chunk_index_ < chunks_.size();
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    if (chunk_index_ >= chunks_.size()) {
      return absl::OutOfRangeError("End of text stream reached.");
    }
    PushOutput(chunks_[chunk_index_++]);
    return absl::OkStatus();
  }

 private:
  std::vector<std::string> chunks_;
  size_t chunk_index_ = 0;
};

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

TEST(TtsSessionTest, ProcessNextChunkSync) {
  std::vector<std::string> chunks = {"Hello ", "world!"};
  auto source = std::make_unique<DummyTextSource>(std::move(chunks));
  auto frontend = std::make_unique<DummyTextFrontend>(source.get());
  auto acoustic = std::make_unique<DummyAcousticPredictor>(frontend.get());
  auto latent = std::make_unique<DummyLatentDecoder>(acoustic.get());
  auto vocoder = std::make_unique<DummyVocoder>(latent.get());

  TtsSession::Components components{
      std::move(source), std::move(frontend), std::move(acoustic),
      std::move(latent), std::move(vocoder),
  };
  ASSERT_OK_AND_ASSIGN(auto session, TtsSession::Create(std::move(components)));

  // Process chunk 1
  ASSERT_OK_AND_ASSIGN(auto chunk1, session->ProcessNextChunk());
  EXPECT_EQ(chunk1.pcm_samples.size(), 3);

  // Process chunk 2
  ASSERT_OK_AND_ASSIGN(auto chunk2, session->ProcessNextChunk());
  EXPECT_EQ(chunk2.pcm_samples.size(), 3);

  // End of stream
  auto chunk3 = session->ProcessNextChunk();
  EXPECT_TRUE(absl::IsOutOfRange(chunk3.status()));
}

TEST(TtsSessionTest, ProcessAsyncWithThreadPool) {
  std::vector<std::string> chunks = {"Hello ", "world!"};
  auto source = std::make_unique<DummyTextSource>(std::move(chunks));
  auto frontend = std::make_unique<DummyTextFrontend>(source.get());
  auto acoustic = std::make_unique<DummyAcousticPredictor>(frontend.get());
  auto latent = std::make_unique<DummyLatentDecoder>(acoustic.get());
  auto vocoder = std::make_unique<DummyVocoder>(latent.get());

  TtsSession::Components components{
      std::move(source), std::move(frontend), std::move(acoustic),
      std::move(latent), std::move(vocoder),
  };
  ASSERT_OK_AND_ASSIGN(auto session, TtsSession::Create(std::move(components)));

  ::litert::lm::ThreadPool thread_pool("tts_test_pool", 4);
  absl::Notification done;
  int chunk_count = 0;
  absl::Status final_status;

  absl::Status status = session->ProcessAsync(
      thread_pool, [&](absl::StatusOr<AudioOutput> result) -> absl::Status {
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
  // 2 normal chunks + 1 flush chunk from Flush() on EOS = 3 chunks
  EXPECT_EQ(chunk_count, 3);
}

}  // namespace
}  // namespace litert::omni::tts
