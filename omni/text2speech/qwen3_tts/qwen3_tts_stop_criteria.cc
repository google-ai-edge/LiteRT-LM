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

#include "omni/text2speech/qwen3_tts/qwen3_tts_stop_criteria.h"

#include <algorithm>

namespace litert::omni::tts {

Qwen3TtsStopCriteria::Qwen3TtsStopCriteria(int text_tokens, int max_frames)
    : text_tokens_(text_tokens),
      frame_limit_(std::min(
          max_frames,
          kMaxFramesPerTextToken * std::max(text_tokens, 0) + kExtraFrames)) {}

int Qwen3TtsStopCriteria::FrameLimit() const { return frame_limit_; }

bool Qwen3TtsStopCriteria::ShouldStop(int frame, int cb0) {
  repeated_frames_ = cb0 == last_cb0_ ? repeated_frames_ + 1 : 1;
  last_cb0_ = cb0;
  if (frame + 1 >= frame_limit_) {
    return true;
  }
  return frame >= text_tokens_ && repeated_frames_ >= kMaxRepeatedFrames;
}

}  // namespace litert::omni::tts
