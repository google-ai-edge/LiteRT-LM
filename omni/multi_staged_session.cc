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

#include "omni/multi_staged_session.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/async_stage_scheduler.h"
#include "omni/base/stage.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni {

absl::StatusOr<std::unique_ptr<MultiStagedSession>> MultiStagedSession::Create(
    std::vector<std::unique_ptr<internal::StageBase>> stages,
    Stage<Output>* absl_nonnull output_stage,
    ::litert::lm::ThreadPool* absl_nullable thread_pool) {
  if (stages.empty()) {
    return absl::InvalidArgumentError("stages must not be empty.");
  }
  for (const auto& stage : stages) {
    if (stage == nullptr) {
      return absl::InvalidArgumentError("stages contains null stage pointer.");
    }
  }
  if (output_stage == nullptr) {
    return absl::InvalidArgumentError("output_stage is required.");
  }
  if (stages.back().get() != output_stage) {
    return absl::InvalidArgumentError(
        "output_stage must be the last stage in stages.");
  }
  return std::unique_ptr<MultiStagedSession>(
      new MultiStagedSession(std::move(stages), output_stage, thread_pool));
}

MultiStagedSession::~MultiStagedSession() { ResetAsyncScheduler(); }

void MultiStagedSession::Reset() {
  ResetAsyncScheduler();
  for (auto& stage : stages_) {
    stage->Reset();
  }
}

absl::StatusOr<MultiStagedSession::Output> MultiStagedSession::ProcessNext() {
  ResetAsyncScheduler();
  if (output_stage_->HasOutput()) {
    return output_stage_->GetOutput();
  }

  do {
    bool any_downstream_needs_schedule = false;
    for (size_t i = 1; i < stages_.size(); ++i) {
      if (stages_[i]->NeedSchedule()) {
        any_downstream_needs_schedule = true;
        break;
      }
    }
    if (!any_downstream_needs_schedule) {
      ABSL_RETURN_IF_ERROR(stages_[0]->Schedule());
    }

    for (size_t i = 1; i < stages_.size(); ++i) {
      if (stages_[i]->NeedSchedule()) {
        ABSL_RETURN_IF_ERROR(stages_[i]->Schedule());
      }
    }
  } while (!output_stage_->HasOutput() && AnyStageNeedsSchedule());

  return output_stage_->GetOutput();
}

absl::Status MultiStagedSession::ProcessAsync(OutputCallback callback) {
  if (thread_pool_ == nullptr) {
    return absl::FailedPreconditionError("ThreadPool is null.");
  }
  absl::MutexLock lock(mutex_);
  if (async_scheduler_ != nullptr) {
    if (async_scheduler_->IsRunning()) {
      return absl::AlreadyExistsError("Async processing is already active.");
    }
    ABSL_RETURN_IF_ERROR(async_scheduler_->Stop(absl::Seconds(3)));
  }

  std::vector<internal::StageBase*> raw_stages;
  raw_stages.reserve(stages_.size());
  for (const auto& stage : stages_) {
    raw_stages.push_back(stage.get());
  }

  // Note: When `AsyncStageScheduler` reaches end-of-stream (`OutOfRangeError`),
  // this callback runs on a thread-pool worker thread and flushes all stages
  // before forwarding the terminal status to `callback`.
  auto callback_with_flush_on_eos =
      [callback = std::move(callback),
       this](absl::StatusOr<Output> result) mutable -> absl::Status {
    if (absl::IsOutOfRange(result.status())) {
      if (absl::Status status = FlushAllStages(); !status.ok()) {
        return callback(status);
      }
      while (output_stage_->HasOutput()) {
        auto out = output_stage_->GetOutput();
        if (out.ok()) {
          ABSL_RETURN_IF_ERROR(callback(std::move(*out)));
        } else if (!absl::IsNotFound(out.status())) {
          return callback(out.status());
        }
      }
      return callback(result.status());
    }
    return callback(std::move(result));
  };

  async_scheduler_ = std::make_unique<AsyncStageScheduler<Output>>(
      std::move(raw_stages), output_stage_, thread_pool_,
      std::move(callback_with_flush_on_eos));
  return async_scheduler_->Start();
}

absl::StatusOr<MultiStagedSession::Output> MultiStagedSession::Flush() {
  ResetAsyncScheduler();
  ABSL_RETURN_IF_ERROR(FlushAllStages());
  if (!output_stage_->HasOutput()) {
    return Output{};
  }
  ABSL_ASSIGN_OR_RETURN(Output result, output_stage_->GetOutput());
  while (output_stage_->HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(Output next, output_stage_->GetOutput());
    ABSL_RETURN_IF_ERROR(CombineOutput(result, std::move(next)));
  }
  return result;
}

absl::Status MultiStagedSession::CombineOutput(Output& accumulated,
                                               Output&& next) {
  if (std::holds_alternative<EndOfOutput>(next)) {
    return absl::OkStatus();
  }
  if (std::holds_alternative<EndOfOutput>(accumulated)) {
    accumulated = std::move(next);
    return absl::OkStatus();
  }
  if (accumulated.index() != next.index()) {
    return absl::InternalError(
        "Cannot combine outputs of different variant types.");
  }
  if (auto* acc_audio = std::get_if<AudioOutput>(&accumulated)) {
    const auto& next_audio = std::get<AudioOutput>(next);
    if (acc_audio->sample_rate_hz == 0) {
      acc_audio->sample_rate_hz = next_audio.sample_rate_hz;
    } else if (next_audio.sample_rate_hz != 0 &&
               acc_audio->sample_rate_hz != next_audio.sample_rate_hz) {
      return absl::InternalError(
          "Cannot combine AudioOutputs with mismatched sample rates.");
    }
    acc_audio->pcm_samples.insert(acc_audio->pcm_samples.end(),
                                  next_audio.pcm_samples.begin(),
                                  next_audio.pcm_samples.end());
    return absl::OkStatus();
  }
  if (auto* acc_text = std::get_if<TextOutput>(&accumulated)) {
    auto& next_text = std::get<TextOutput>(next);
    if (!next_text.confirmed_text.empty()) {
      if (!acc_text->confirmed_text.empty()) {
        // Join confirmed text segments with a space, matching TextMerger's
        // space-delimited word joining convention.
        acc_text->confirmed_text += ' ';
      }
      acc_text->confirmed_text += next_text.confirmed_text;
    }
    acc_text->unconfirmed_text = std::move(next_text.unconfirmed_text);
    return absl::OkStatus();
  }
  return absl::InternalError("Unsupported output type for CombineOutput.");
}

absl::Status MultiStagedSession::DrainStageUntilIdle(internal::StageBase& stage,
                                                     bool is_source_stage) {
  while (!stage.IsIdle() || stage.NeedSchedule()) {
    if (stage.NeedSchedule()) {
      absl::Status status = stage.Schedule();
      if (absl::IsNotFound(status) ||
          (is_source_stage && absl::IsOutOfRange(status))) {
        break;
      }
      ABSL_RETURN_IF_ERROR(status);
    } else if (!stage.IsIdle()) {
      absl::SleepFor(absl::Milliseconds(1));
    }
  }
  return absl::OkStatus();
}

bool MultiStagedSession::AnyStageNeedsSchedule() const {
  return std::any_of(stages_.begin(), stages_.end(),
                     [](const std::unique_ptr<internal::StageBase>& stage) {
                       return stage->NeedSchedule();
                     });
}

absl::Status MultiStagedSession::FlushAllStages() {
  for (size_t i = 0; i < stages_.size(); ++i) {
    const bool is_source_stage = (i == 0);
    ABSL_RETURN_IF_ERROR(DrainStageUntilIdle(*stages_[i], is_source_stage));
    ABSL_RETURN_IF_ERROR(stages_[i]->Flush());
    ABSL_RETURN_IF_ERROR(DrainStageUntilIdle(*stages_[i], is_source_stage));
  }
  return absl::OkStatus();
}

void MultiStagedSession::ResetAsyncScheduler() {
  absl::MutexLock lock(mutex_);
  if (async_scheduler_) {
    ABSL_CHECK_OK(async_scheduler_->Stop(absl::Seconds(3)));
  }
}

}  // namespace litert::omni
