// Copyright 2025 The ODML Authors.
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

#include "runtime/framework/resource_management/context_handler/context_handler.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<int>
ContextHandler::SharedProcessedContext::LongestHandlerTimeStep(
    LlmExecutor& llm_executor) const {
  absl::MutexLock lock(handlers_mutex_);
  // If this is no longer referenced by a ContextHandler, the handlers may
  // be empty here, thus nullptr might be returned. This can happen if the
  // last ContextHandler is deleted when this is the active
  // processed_context, and the manager has already taken a reference.
  int max_tokens = 0;
  for (const auto& handler : handlers_) {
    int current_step;
    // If the runtime_config_ is nullptr, it means the executor is currently
    // using the handler.
    if (handler->runtime_config_ == nullptr) {
      ASSIGN_OR_RETURN(current_step, llm_executor.GetCurrentStep());
    } else {
      current_step = handler->runtime_state_->current_step;
    }
    max_tokens = std::max(max_tokens, current_step);
  }
  return max_tokens;
}

// static
absl::StatusOr<std::unique_ptr<ContextHandler>> ContextHandler::Create(
    std::unique_ptr<LlmContext> llm_context,
    std::unique_ptr<AudioContext> audio_context) {
  RET_CHECK_NE(llm_context, nullptr) << "The llm_context is null.";
  ASSIGN_OR_RETURN(auto processed_context,
                   llm_context->RetrieveProcessedContext());
  auto shared_processed_context =
      std::make_shared<SharedProcessedContext>(std::move(processed_context));
  ASSIGN_OR_RETURN(auto runtime_config, llm_context->RetrieveRuntimeConfig());
  ASSIGN_OR_RETURN(auto runtime_state, llm_context->RetrieveRuntimeState());
  return Bundle(shared_processed_context, std::move(runtime_config),
                std::move(runtime_state), std::move(audio_context));
}

// static
absl::StatusOr<std::unique_ptr<ContextHandler>> ContextHandler::Bundle(
    std::shared_ptr<SharedProcessedContext> shared_processed_context,
    std::unique_ptr<RuntimeConfig> runtime_config,
    std::unique_ptr<RuntimeState> runtime_state,
    std::unique_ptr<AudioContext> audio_context) {
  RET_CHECK_NE(shared_processed_context, nullptr)
      << "The shared_processed_context is null.";
  auto handler = std::unique_ptr<ContextHandler>(
      new ContextHandler(shared_processed_context, std::move(runtime_config),
                         std::move(runtime_state), std::move(audio_context)));
  return handler;
}

ContextHandler::~ContextHandler() {
  shared_processed_context_->RemoveHandler(this);
  if (audio_context_ != nullptr) {
    audio_context_.reset();
  }
}

absl::Status ContextHandler::UpdateSharedProcessedContext(
    std::shared_ptr<SharedProcessedContext> new_shared_processed_context) {
  if (shared_processed_context_ == new_shared_processed_context) {
    return absl::OkStatus();
  }
  shared_processed_context_->RemoveHandler(this);
  shared_processed_context_ = new_shared_processed_context;
  shared_processed_context_->AddHandler(this);
  return absl::OkStatus();
}

ContextHandler::ContextHandler(
    std::shared_ptr<SharedProcessedContext> shared_processed_context,
    std::unique_ptr<RuntimeConfig> runtime_config,
    std::unique_ptr<RuntimeState> runtime_state,
    std::unique_ptr<AudioContext> audio_context)
    : shared_processed_context_(shared_processed_context),
      runtime_config_(std::move(runtime_config)),
      runtime_state_(std::move(runtime_state)),
      audio_context_(std::move(audio_context)) {
  shared_processed_context_->AddHandler(this);
}

}  // namespace litert::lm
