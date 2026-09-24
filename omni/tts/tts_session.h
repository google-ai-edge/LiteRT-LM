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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_SESSION_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "omni/base/async_stage_scheduler.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::tts {

// Orchestrates component pipeline execution for TTS speech synthesis streams.
class TtsSession : public OmniSession {
 public:
  struct Components {
    std::unique_ptr<StreamTextSource> text_source;
    std::vector<std::unique_ptr<internal::StageBase>> intermediate_stages;
    std::unique_ptr<Vocoder> vocoder;
  };

  // Creates a TtsSession instance taking ownership of configured components
  // and reference to the ThreadPool (owned by TtsEngine).
  static absl::StatusOr<std::unique_ptr<TtsSession>> Create(
      Components components, ::litert::lm::ThreadPool* thread_pool);

  ~TtsSession() override;

  // Resets session and component state for a new synthesis stream.
  void Reset() override;

  // Flushes remaining synthesized audio at stream end.
  absl::StatusOr<Output> Flush() override;

  // Processes the next synthesized audio chunk synchronously from `text_source`
  // and returns `OmniSession::Output` (`AudioOutput`).
  // Returns absl::OutOfRangeError when synthesis stream ends.
  absl::StatusOr<Output> ProcessNext() override;

  // Processes the TTS stream asynchronously using the session's thread pool and
  // emits `OmniSession::Output` (`AudioOutput`) chunks to `callback`.
  absl::Status ProcessAsync(OutputCallback callback) override;

  // Returns the session's `StreamTextSource` stage.
  StreamTextSource& text_source() { return *components_.text_source; }

 private:
  explicit TtsSession(Components components,
                      ::litert::lm::ThreadPool* thread_pool);

  void ResetAsyncScheduler();
  void WaitForIdleOrStopped();

  Components components_;
  ::litert::lm::ThreadPool* thread_pool_ = nullptr;

  mutable absl::Mutex mutex_;
  std::unique_ptr<AsyncStageScheduler<AudioOutput>> async_scheduler_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_SESSION_H_
