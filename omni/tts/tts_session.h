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

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "omni/base/async_stage_scheduler.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/latent_decoder.h"
#include "omni/tts/text_frontend.h"
#include "omni/tts/text_source.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"

namespace litert_lm::omni::tts {

// Orchestrates component pipeline execution for TTS speech synthesis streams.
class TtsSession {
 public:
  struct Components {
    std::unique_ptr<TextSource> text_source;
    std::unique_ptr<TextFrontend> text_frontend;
    std::unique_ptr<AcousticPredictor> acoustic_predictor;
    std::unique_ptr<LatentDecoder> latent_decoder;
    std::unique_ptr<Vocoder> vocoder;
  };

  // Creates a TtsSession instance taking ownership of configured components.
  static absl::StatusOr<std::unique_ptr<TtsSession>> Create(
      Components components);

  ~TtsSession();

  // Resets session and component state for a new synthesis stream.
  void Reset();

  // Processes the next audio chunk from Vocoder synchronously.
  // Returns absl::OutOfRangeError when audio stream ends.
  absl::StatusOr<Vocoder::AudioOutput> ProcessNextChunk();

  // Processes the TTS stream asynchronously using the provided thread pool.
  // Returns absl::AlreadyExistsError if async processing is already active.
  // Schedules TTS session stages on the thread pool to execute concurrently and
  // passes results to `callback` until `callback` returns an error status.
  using AsyncCallback =
      absl::AnyInvocable<absl::Status(absl::StatusOr<Vocoder::AudioOutput>)>;
  absl::Status ProcessAsync(::litert::lm::ThreadPool& thread_pool,
                            AsyncCallback callback);

  // Flushes remaining synthesized audio at stream end.
  absl::StatusOr<Vocoder::AudioOutput> Flush();

  const Components& components() const { return components_; }

 private:
  explicit TtsSession(Components components);

  void ResetAsyncScheduler();

  Components components_;

  mutable absl::Mutex mutex_;
  std::unique_ptr<AsyncStageScheduler<Vocoder::AudioOutput>> async_scheduler_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert_lm::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_SESSION_H_
