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

#include "omni/tts/stream_text_source.h"

#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "omni/tts/text_chunk_utils.h"

namespace litert::omni::tts {

StreamTextSource::StreamTextSource(TextChunkConfig config)
    : config_(std::move(config)) {}

absl::Status StreamTextSource::PushText(absl::string_view text) {
  absl::MutexLock lock(mutex_);
  if (state_ != State::kIdle) {
    return absl::FailedPreconditionError(
        "Cannot push text while stage is not running.");
  } else if (is_finished_) {
    return absl::FailedPreconditionError(
        "Cannot push text after Finish() has been called.");
  }
  buffer_.append(text);
  return absl::OkStatus();
}

void StreamTextSource::Finish() {
  absl::MutexLock lock(mutex_);
  is_finished_ = true;
}

bool StreamTextSource::IsFinished() const {
  absl::MutexLock lock(mutex_);
  return is_finished_;
}

void StreamTextSource::ResetInternal() {
  {
    absl::MutexLock lock(mutex_);
    is_finished_ = false;
  }
  buffer_.clear();
  buffer_start_index_ = 0;
}

bool StreamTextSource::NeedScheduleInternal() const {
  absl::string_view remaining =
      absl::string_view(buffer_).substr(buffer_start_index_);
  return ShouldSchedule(remaining, is_finished_, config_);
}

absl::Status StreamTextSource::ScheduleInternal() {
  SetState(State::kRunning);
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  auto chunk =
      ExtractNextChunk(buffer_, buffer_start_index_, is_finished_, config_);
  if (chunk.has_value()) {
    PushOutput(std::string(chunk->chunk));
    buffer_start_index_ = chunk->next_start_index;
    if (buffer_start_index_ >= buffer_.size()) {
      buffer_.clear();
      buffer_start_index_ = 0;
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::tts
