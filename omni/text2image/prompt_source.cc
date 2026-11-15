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

#include "omni/text2image/prompt_source.h"

#include <string>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::omni::text2image {

absl::Status PushPromptSource::PushPrompt(absl::string_view text) {
  if (text.empty()) {
    return absl::InvalidArgumentError("Prompt text must not be empty.");
  }
  absl::MutexLock lock(mutex_);
  if (is_finished_) {
    return absl::FailedPreconditionError(
        "Cannot push prompt after Finish() has been called.");
  }
  outputs_.push_back(
      Text2ImagePrompt{.metadata = metadata_, .text = std::string(text)});
  return absl::OkStatus();
}

void PushPromptSource::Finish() {
  {
    absl::MutexLock lock(mutex_);
    is_finished_ = true;
  }
  // TODO(b/568027544): Return absl::Status from Finish().
  ABSL_CHECK_OK(Flush());
}

void PushPromptSource::ResetInternal() {
  absl::MutexLock lock(mutex_);
  is_finished_ = false;
}

bool PushPromptSource::NeedScheduleInternal() const { return false; }

absl::Status PushPromptSource::ScheduleInternal() {
  SetState(State::kIdle);
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image
