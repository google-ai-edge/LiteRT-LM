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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_PROMPT_SOURCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_PROMPT_SOURCE_H_

#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/stage.h"

namespace litert::omni::text2image {

// Prompt request payload for a text-to-image generation pipeline.
struct Text2ImagePrompt {
  ImageGenInputMetadata metadata;
  std::string text;
};

// Base stage that provides Text2ImagePrompt items to downstream stages.
class PromptSource : public SingleThreadedStageWithDeque<Text2ImagePrompt> {
 public:
  ~PromptSource() override = default;

  // Signals that no more prompts will be pushed to the source.
  virtual void Finish() = 0;
};

// Thread-safe PromptSource implementation that queues Text2ImagePrompt requests
// pushed by the caller.
class PushPromptSource : public PromptSource {
 public:
  explicit PushPromptSource(ImageGenInputMetadata metadata = {})
      : metadata_(std::move(metadata)) {}
  ~PushPromptSource() override = default;

  // Enqueues a text prompt using the source's default generation metadata.
  absl::Status PushPrompt(absl::string_view text);

  // Signals that no more prompts will be pushed to the stream.
  void Finish() override;

 protected:
  void ResetInternal() override;

  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  const ImageGenInputMetadata metadata_;
  bool is_finished_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_PROMPT_SOURCE_H_
