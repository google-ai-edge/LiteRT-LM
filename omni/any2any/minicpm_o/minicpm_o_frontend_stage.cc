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

#include "omni/any2any/minicpm_o/minicpm_o_frontend_stage.h"

#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/fan_in_stage.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

MiniCpmOFrontendStage::MiniCpmOFrontendStage(
    Stage<Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable text_stage,
    Stage<Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable audio_stage,
    Stage<Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable image_stage,
    const MiniCpmOModelConfig& config)
    : config_(config), generate_audio_(config.generate_audio) {
  if (text_stage != nullptr) {
    fan_in_.AddInputStage(text_stage);
  }
  if (audio_stage != nullptr) {
    fan_in_.AddInputStage(audio_stage);
  }
  if (image_stage != nullptr) {
    fan_in_.AddInputStage(image_stage);
  }
}

bool MiniCpmOFrontendStage::NeedScheduleInternal() const {
  return fan_in_.NeedSchedule() || fan_in_.HasOutput();
}

absl::Status MiniCpmOFrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (fan_in_.NeedSchedule()) {
    absl::Status status = fan_in_.Schedule();
    if (!status.ok() && !absl::IsNotFound(status)) {
      return status;
    }
  }

  bool processed_any = false;
  while (fan_in_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment segment,
                          fan_in_.GetOutput());
    processed_any = true;
    AppendSegment(std::move(segment));
  }

  if (!processed_any && !this->HasOutput()) {
    return absl::NotFoundError(
        "MiniCpmOFrontendStage has no ready sequence items.");
  }
  return absl::OkStatus();
}

void MiniCpmOFrontendStage::AppendSegment(MiniCpmOEmbeddingSegment segment) {
  prompt_embeddings_.insert(prompt_embeddings_.end(),
                            segment.embeddings.begin(),
                            segment.embeddings.end());
  token_ids_.insert(token_ids_.end(), segment.token_ids.begin(),
                    segment.token_ids.end());
  if (segment.is_turn_end) {
    FlushTurn();
  }
}

absl::Status MiniCpmOFrontendStage::FlushInternal() {
  if (fan_in_.NeedSchedule()) {
    absl::Status status = fan_in_.Schedule();
    if (!status.ok() && !absl::IsNotFound(status)) {
      return status;
    }
  }
  while (fan_in_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment segment,
                          fan_in_.GetOutput());
    AppendSegment(std::move(segment));
  }
  FlushTurn();
  return absl::OkStatus();
}

void MiniCpmOFrontendStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  fan_in_.Reset();
  ResetTurn();
}

void MiniCpmOFrontendStage::FlushTurn() {
  if (!prompt_embeddings_.empty() || !token_ids_.empty()) {
    const int prompt_len =
        config_.llm_hidden_size > 0
            ? static_cast<int>(prompt_embeddings_.size() /
                               config_.llm_hidden_size)
            : static_cast<int>(token_ids_.size());
    this->PushOutput(MiniCpmOFrontendOutput{
        .prompt_embeddings = std::move(prompt_embeddings_),
        .prompt_len = prompt_len,
        .token_ids = std::move(token_ids_),
        .generate_audio = generate_audio_,
    });
  }
  ResetTurn();
}

void MiniCpmOFrontendStage::ResetTurn() {
  prompt_embeddings_.clear();
  token_ids_.clear();
}

}  // namespace litert::omni::any2any
