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

#include "omni/tts/tts_engine.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/tts_session.h"
#include "runtime/framework/threadpool.h"

namespace litert_lm::omni::tts {

absl::StatusOr<std::unique_ptr<TtsEngine>> TtsEngine::CreateWithComponents(
    const TtsEngineSettings& settings, TtsSession::Components components) {
  if (components.text_source == nullptr) {
    components.text_source =
        std::make_unique<StreamTextSource>(settings.text_chunk_config);
  }
  StreamTextSource* stream_text_source =
      dynamic_cast<StreamTextSource*>(components.text_source.get());
  if (stream_text_source == nullptr) {
    return absl::InvalidArgumentError(
        "TtsEngine requires components.text_source to be a StreamTextSource.");
  }
  ABSL_ASSIGN_OR_RETURN(auto session,
                        TtsSession::Create(std::move(components)));
  auto thread_pool = std::make_unique<::litert::lm::ThreadPool>(
      "tts_engine_pool", settings.num_threads);
  return std::unique_ptr<TtsEngine>(new TtsEngine(settings, std::move(session),
                                                  stream_text_source,
                                                  std::move(thread_pool)));
}

absl::StatusOr<std::unique_ptr<TtsEngine>> TtsEngine::Create(
    const TtsEngineSettings& settings) {
  TtsSession::Components components;
  return CreateWithComponents(settings, std::move(components));
}

TtsEngine::TtsEngine(const TtsEngineSettings& settings,
                     std::unique_ptr<TtsSession> session,
                     StreamTextSource* stream_text_source,
                     std::unique_ptr<::litert::lm::ThreadPool> thread_pool)
    : settings_(settings),
      session_(std::move(session)),
      stream_text_source_(stream_text_source),
      thread_pool_(std::move(thread_pool)) {}

void TtsEngine::Reset() { session_->Reset(); }

absl::StatusOr<AudioOutput> TtsEngine::Flush() {
  stream_text_source_->Finish();
  auto result = session_->Flush();
  if (absl::IsNotFound(result.status())) {
    return AudioOutput();
  }
  return result;
}

absl::StatusOr<AudioOutput> TtsEngine::Synthesize(absl::string_view text) {
  Reset();
  ABSL_RETURN_IF_ERROR(stream_text_source_->PushText(text));
  stream_text_source_->Finish();

  AudioOutput result;
  while (true) {
    auto chunk = session_->ProcessNextChunk();
    if (absl::IsOutOfRange(chunk.status()) ||
        absl::IsNotFound(chunk.status())) {
      break;
    }
    if (!chunk.ok()) {
      return chunk.status();
    }
    result.sample_rate_hz = chunk->sample_rate_hz;
    result.pcm_samples.insert(result.pcm_samples.end(),
                              chunk->pcm_samples.begin(),
                              chunk->pcm_samples.end());
  }

  auto flush_chunk = session_->Flush();
  if (flush_chunk.ok()) {
    result.sample_rate_hz = flush_chunk->sample_rate_hz;
    result.pcm_samples.insert(result.pcm_samples.end(),
                              flush_chunk->pcm_samples.begin(),
                              flush_chunk->pcm_samples.end());
  }
  return result;
}

absl::Status TtsEngine::SynthesizeAsync(absl::string_view text,
                                        AsyncCallback callback) {
  ABSL_RETURN_IF_ERROR(stream_text_source_->PushText(text));
  absl::Status status =
      session_->ProcessAsync(*thread_pool_, std::move(callback));
  if (absl::IsAlreadyExists(status)) {
    return absl::OkStatus();
  }
  return status;
}

}  // namespace litert_lm::omni::tts
