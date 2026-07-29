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
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/async_stage_scheduler.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<TtsSession>> TtsSession::Create(
    Components components) {
  if (components.text_source == nullptr) {
    return absl::InvalidArgumentError("TextSource component is required.");
  }
  if (components.text_frontend == nullptr) {
    return absl::InvalidArgumentError("TextFrontend component is required.");
  }
  if (components.acoustic_predictor == nullptr) {
    return absl::InvalidArgumentError(
        "AcousticPredictor component is required.");
  }
  if (components.latent_decoder == nullptr) {
    return absl::InvalidArgumentError("LatentDecoder component is required.");
  }
  if (components.vocoder == nullptr) {
    return absl::InvalidArgumentError("Vocoder component is required.");
  }
  return std::unique_ptr<TtsSession>(new TtsSession(std::move(components)));
}

TtsSession::TtsSession(Components components)
    : components_(std::move(components)) {}

TtsSession::~TtsSession() { ResetAsyncScheduler(); }

void TtsSession::ResetAsyncScheduler() {
  absl::MutexLock lock(mutex_);
  if (async_scheduler_) {
    absl::Status status = async_scheduler_->Stop(absl::Seconds(3));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to stop async scheduler: " << status;
    }
  }
}

void TtsSession::Reset() {
  ResetAsyncScheduler();
  components_.text_source->Reset();
  components_.text_frontend->Reset();
  components_.acoustic_predictor->Reset();
  components_.latent_decoder->Reset();
  components_.vocoder->Reset();
}

absl::StatusOr<AudioOutput> TtsSession::ProcessNextChunk() {
  ABSL_RETURN_IF_ERROR(components_.text_source->Schedule());

  if (components_.text_frontend->NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(components_.text_frontend->Schedule());
  }

  if (components_.acoustic_predictor->NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(components_.acoustic_predictor->Schedule());
  }

  if (components_.latent_decoder->NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(components_.latent_decoder->Schedule());
  }

  if (components_.vocoder->NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(components_.vocoder->Schedule());
  }

  return components_.vocoder->GetOutput();
}

absl::Status TtsSession::ProcessAsync(::litert::lm::ThreadPool& thread_pool,
                                      AsyncCallback callback) {
  absl::MutexLock lock( mutex_ );
  if (async_scheduler_ != nullptr) {
    if (async_scheduler_->IsRunning()) {
      return absl::AlreadyExistsError("Async processing is already active.");
    }
    ABSL_RETURN_IF_ERROR(async_scheduler_->Stop(absl::Seconds(3)));
  }

  std::vector<internal::StageBase*> stages = {
      components_.text_source.get(),
      components_.text_frontend.get(),
      components_.acoustic_predictor.get(),
      components_.latent_decoder.get(),
      components_.vocoder.get(),
  };
  auto callback_with_flush_on_eos =
      [callback = std::move(callback),
       this](absl::StatusOr<AudioOutput> result) mutable -> absl::Status {
    if (absl::IsOutOfRange(result.status())) {
      ABSL_RETURN_IF_ERROR(components_.vocoder->Flush());
      ABSL_RETURN_IF_ERROR(callback(components_.vocoder->GetOutput()));
    }
    return callback(std::move(result));
  };
  async_scheduler_ = std::make_unique<AsyncStageScheduler<AudioOutput>>(
      std::move(stages), components_.vocoder.get(), &thread_pool,
      std::move(callback_with_flush_on_eos));
  return async_scheduler_->Start();
}

absl::StatusOr<AudioOutput> TtsSession::Flush() {
  ABSL_RETURN_IF_ERROR(components_.vocoder->Flush());
  return components_.vocoder->GetOutput();
}

}  // namespace litert::omni::tts
