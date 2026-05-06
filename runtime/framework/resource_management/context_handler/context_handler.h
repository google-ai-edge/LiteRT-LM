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

#ifndef THIRD_PARTY_ODML_LITERT_LM_FRAMEWORK_RESOURCE_MANAGEMENT_CONTEXT_HANDLER_CONTEXT_HANDLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_FRAMEWORK_RESOURCE_MANAGEMENT_CONTEXT_HANDLER_CONTEXT_HANDLER_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

class ContextHandler {
 public:
  class SharedProcessedContext;

  // Creates a new ContextHandler from a provided LlmContext.
  static absl::StatusOr<std::unique_ptr<ContextHandler>> Create(
      std::unique_ptr<LlmContext> llm_context,
      std::unique_ptr<AudioContext> audio_context = nullptr);

  // Wraps the shared processed context, runtime config, and runtime state into
  // a ContextHandler.
  static absl::StatusOr<std::unique_ptr<ContextHandler>> Bundle(
      std::shared_ptr<SharedProcessedContext> shared_processed_context,
      std::unique_ptr<RuntimeConfig> runtime_config,
      std::unique_ptr<RuntimeState> runtime_state,
      std::unique_ptr<AudioContext> audio_context = nullptr);

  ~ContextHandler();

  // Assign and switch the shared processed context to point to a new one. This
  // will also update the handlers vector in the previous and the new shared
  // processed context.
  absl::Status UpdateSharedProcessedContext(
      std::shared_ptr<SharedProcessedContext> new_shared_processed_context);

  // Returns the shared processed context.
  std::shared_ptr<SharedProcessedContext> shared_processed_context() const {
    return shared_processed_context_;
  }

  // Returns true if the runtime config is set.
  bool HasRuntimeConfig() const { return runtime_config_ != nullptr; }

  // Sets the runtime config.
  absl::Status SetRuntimeConfig(std::unique_ptr<RuntimeConfig> runtime_config) {
    runtime_config_ = std::move(runtime_config);
    return absl::OkStatus();
  }

  // Retrieves the runtime config, the caller will take the ownership of the
  // returned runtime config and it will no longer be available in the
  // ContextHandler.
  absl::StatusOr<std::unique_ptr<RuntimeConfig>> RetrieveRuntimeConfig() {
    RET_CHECK(HasRuntimeConfig()) << "Runtime config not found.";
    return std::move(runtime_config_);
  };

  // Gets a copy of the current runtime configuration.
  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const {
    RET_CHECK(HasRuntimeConfig()) << "Runtime config not found.";
    return *runtime_config_;
  };

  // Returns true if the runtime state is set.
  bool HasRuntimeState() const { return runtime_state_ != nullptr; }

  // Sets the runtime state.
  absl::Status SetRuntimeState(std::unique_ptr<RuntimeState> runtime_state) {
    runtime_state_ = std::move(runtime_state);
    return absl::OkStatus();
  }

  // Retrieves the runtime state, the caller will take the ownership of the
  // returned runtime state and it will no longer be available in the
  // ContextHandler.
  absl::StatusOr<std::unique_ptr<RuntimeState>> RetrieveRuntimeState() {
    RET_CHECK(HasRuntimeState()) << "Runtime state not found.";
    return std::move(runtime_state_);
  }

  // Gets a copy of the current runtime state.
  absl::StatusOr<RuntimeState> GetRuntimeState() const {
    RET_CHECK(HasRuntimeState()) << "Runtime state not found.";
    return *runtime_state_;
  }

  // Returns true if the audio context is set.
  bool HasAudioContext() const { return audio_context_ != nullptr; }

  // Retrieves the audio context, the caller will take the ownership of the
  // returned audio context and it will no longer be available in the
  // ContextHandler.
  absl::StatusOr<std::unique_ptr<AudioContext>> RetrieveAudioContext() {
    RET_CHECK(HasAudioContext()) << "Audio context not found.";
    return std::move(audio_context_);
  }

  // Gets a reference of the current audio context.
  const AudioContext& GetAudioContext() const { return *audio_context_; }

  // Sets the audio context.
  absl::Status SetAudioContext(std::unique_ptr<AudioContext> audio_context) {
    audio_context_ = std::move(audio_context);
    return absl::OkStatus();
  }

 private:
  ContextHandler(
      std::shared_ptr<SharedProcessedContext> shared_processed_context,
      std::unique_ptr<RuntimeConfig> runtime_config,
      std::unique_ptr<RuntimeState> runtime_state,
      std::unique_ptr<AudioContext> audio_context);

  // The shared processed context.
  std::shared_ptr<SharedProcessedContext> shared_processed_context_;

  // The runtime config.
  std::unique_ptr<RuntimeConfig> runtime_config_;

  // The runtime state.
  std::unique_ptr<RuntimeState> runtime_state_;

  // The audio context.
  std::unique_ptr<AudioContext> audio_context_;
};

// Holds the real ProcessedContext and handlers any operations on it.
// ContextHandler will hold a reference to this to allow copy on write
// behavior of ProcessedContext.
class ContextHandler::SharedProcessedContext {
 public:
  explicit SharedProcessedContext(
      std::unique_ptr<ProcessedContext> processed_context)
      : processed_context_(std::move(processed_context)) {}

  // Adds a handler to this SharedProcessedContext.
  void AddHandler(ContextHandler* handler) {
    absl::MutexLock lock(&handlers_mutex_);
    handlers_.push_back(handler);
  }

  // Removes a handler from this SharedProcessedContext.
  void RemoveHandler(ContextHandler* handler) {
    absl::MutexLock lock(&handlers_mutex_);
    handlers_.erase(std::remove(handlers_.begin(), handlers_.end(), handler),
                    handlers_.end());
  }

  // Returns the number of tokens in the longest handler.
  absl::StatusOr<int> LongestHandlerTimeStep(LlmExecutor& llm_executor) const;

  // Returns true if the processed context is set.
  bool HasProcessedContext() const { return processed_context_ != nullptr; }

  // Sets the processed context.
  absl::Status SetProcessedContext(
      std::unique_ptr<ProcessedContext> processed_context) {
    absl::MutexLock lock(&processed_context_mutex_);
    RET_CHECK(!HasProcessedContext())
        << "The processed context is already set.";
    processed_context_ = std::move(processed_context);
    return absl::OkStatus();
  }

  // Retrieves the processed context, the caller will take the ownership of the
  // returned processed context and it will no longer be available in the
  // SharedProcessedContext.
  absl::StatusOr<std::unique_ptr<ProcessedContext>> RetrieveProcessedContext() {
    absl::MutexLock lock(&processed_context_mutex_);
    return std::move(processed_context_);
  }

 private:
  // Handlers can be removed outside of the runner lock, so lock them
  // separately.
  mutable absl::Mutex handlers_mutex_;

  // A chain of `ContextHandler`s that share this
  // `SharedProcessedContext`, where handlers[i] is copied from
  // handlers[i-1].
  std::vector<ContextHandler*> handlers_ ABSL_GUARDED_BY(handlers_mutex_);

  // Protects the processed context.
  mutable absl::Mutex processed_context_mutex_;

  // The actual Processed Context.
  std::unique_ptr<ProcessedContext> processed_context_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_FRAMEWORK_RESOURCE_MANAGEMENT_CONTEXT_HANDLER_CONTEXT_HANDLER_H_
