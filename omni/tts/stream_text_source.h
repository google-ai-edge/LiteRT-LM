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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_STREAM_TEXT_SOURCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_STREAM_TEXT_SOURCE_H_

#include <cstddef>
#include <string>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/text_source.h"

namespace litert::omni::tts {

// TextSource implementation that accepts streaming text fragments from callers
// via PushText() and splits them into delimiter-aligned chunks for TTS
// synthesis.
class StreamTextSource : public TextSource {
 public:
  // Constructs a StreamTextSource with optional text chunk configuration.
  //
  // args
  // - config: Text chunk configuration options for stream chunking.
  explicit StreamTextSource(TextChunkConfig config = {});

  ~StreamTextSource() override = default;

  // Appends a text fragment to the streaming input buffer.
  //
  // args
  // - text: Incoming text fragment string view from caller/user.
  //
  // returns
  // - absl::OkStatus() on success, or absl::FailedPreconditionError if Finish()
  //   has already been called on this stream.
  absl::Status PushText(absl::string_view text);

  // Marks the end of the text input stream.
  // Any remaining un-emitted text in the buffer will be scheduled and flushed.
  void Finish();

  // Returns true if Finish() has been called to complete the stream.
  bool IsFinished() const;

  // Accessor for the active text chunk configuration.
  const TextChunkConfig& config() const { return config_; }

 protected:
  // Resets stream state and buffers for a new session.
  void ResetInternal() override;

  // Determines if ScheduleInternal should be called based on delimiter match or
  // Finish().
  bool NeedScheduleInternal() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) override;

  // Extracts text chunks from the stream buffer into the stage output deque.
  absl::Status ScheduleInternal() ABSL_NO_THREAD_SAFETY_ANALYSIS override;

 private:
  // Text chunk configuration for stream chunking.
  const TextChunkConfig config_;

  // Internal buffer accumulating text pushed by callers.
  std::string buffer_;

  // Read offset index within buffer_.
  size_t buffer_start_index_ = 0;

  // True if Finish() has been called on this stream source.
  bool is_finished_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_STREAM_TEXT_SOURCE_H_
