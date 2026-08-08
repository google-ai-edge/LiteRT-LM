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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_SESSION_H_

#include <memory>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/detokenizer.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/asr/text_merger.h"
#include "omni/base/async_stage_scheduler.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::asr {

// Orchestrates component pipeline execution for ASR speech recognition streams.
class AsrSession {
 public:
  struct Components {
    std::unique_ptr<AudioSource> audio_source;
    std::unique_ptr<AudioPreprocessor> preprocessor;
    std::unique_ptr<SpeechRecognizer> speech_recognizer;
    std::unique_ptr<Detokenizer> detokenizer;
    std::unique_ptr<TextMerger> text_merger;
  };

  // Creates an AsrSession instance taking ownership of configured components.
  static absl::StatusOr<std::unique_ptr<AsrSession>> Create(
      Components components);

  ~AsrSession();

  // Resets session and component state for a new audio stream.
  void Reset();

  // Processes the next audio chunk from AudioSource synchronously.
  // Returns absl::OutOfRangeError when audio stream ends.
  absl::StatusOr<TextMerger::MergeResult> ProcessNextChunk();

  // Processes the audio stream asynchronously using the provided thread pool.
  // Returns absl::AlreadyExistsError if async processing is already active.
  // Schedules asr session stages on the thread pool to execute concurrently and
  // passes results to `callback` until `callback` returns an error status.
  using AsyncCallback =
      absl::AnyInvocable<absl::Status(absl::StatusOr<TextMerger::MergeResult>)>;
  absl::Status ProcessAsync(::litert::lm::ThreadPool& thread_pool,
                            AsyncCallback callback);

  // Flushes remaining unconfirmed text at stream end.
  absl::StatusOr<TextMerger::MergeResult> Flush();

  const Components& components() const { return components_; }

 private:
  explicit AsrSession(Components components);

  void ResetAsyncScheduler();

  Components components_;

  mutable absl::Mutex mutex_;
  std::unique_ptr<AsyncStageScheduler<TextMerger::MergeResult>> async_scheduler_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_SESSION_H_
