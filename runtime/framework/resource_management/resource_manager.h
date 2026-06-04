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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_RESOURCE_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_RESOURCE_MANAGER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"

namespace litert::lm {

// The ResourceManager provides thread-safe access to shared resources such
// as the LlmExecutor, enabling multiple sessions to utilize it concurrently.
class ResourceManager {
 public:
  explicit ResourceManager(
      ModelResources* absl_nullable model_resources,
      std::unique_ptr<LlmExecutor> llm_executor,
      std::unique_ptr<VisionExecutorSettings> vision_executor_settings,
      std::unique_ptr<AudioExecutorSettings> audio_executor_settings,
      LlmExecutorSettings llm_executor_settings,
      ::litert::Environment* absl_nullable litert_env,
      std::unique_ptr<AudioExecutor> audio_executor = nullptr)
      :  // dummy comment to prevent clang-format from moving the next line here
        llm_executor_(std::move(llm_executor)),
        vision_executor_settings_(std::move(vision_executor_settings)),
        audio_executor_(std::move(audio_executor)),
        audio_executor_settings_(std::move(audio_executor_settings)),
        litert_env_(litert_env),
        llm_executor_settings_(std::move(llm_executor_settings)) {}

  // Creates a ResourceManager with the provided llm_executor.
  // Note that the audio_executor is used for testing only (dependency
  // injection)
  static absl::StatusOr<std::unique_ptr<ResourceManager>> Create(
      ModelResources* absl_nullable model_resources,
      std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
      std::unique_ptr<VisionExecutorSettings> absl_nullable
          vision_executor_settings,
      std::unique_ptr<AudioExecutorSettings> absl_nullable
          audio_executor_settings,
      ::litert::Environment* absl_nullable litert_env,
      std::unique_ptr<AudioExecutor> absl_nullable audio_executor = nullptr);

  ~ResourceManager() = default;

  // Assigns the lora id from the given lora path or scoped file. If no lora is
  // used, will return std::nullopt instead of an uint32_t id.
  // If lora_path is not empty, it will be treated as the hash key, retrieving
  // the corresponding lora id from the lora_hash_to_id_ map if it exists.
  // Otherwise, a unique lora id will be assigned.
  // If lora_path is empty and has_scoped_lora_file is true, a unique lora id
  // will be assigned. Scoped file should be provided under
  // session_config_struct.scoped_lora_file, and the lora will be loaded while
  // calling CreateContextHandler.
  // If lora_path is empty and has_scoped_lora_file is false, std::nullopt will
  // be returned.
  // Noticed: If you intend to reuse a LoRA loaded via a scoped file, please
  // assign a unique, custom lora_path. This lora_path serves as the identifier
  // for the LoRA across all sessions referencing that scoped file.
  std::optional<uint32_t> AssignLoraId(std::string lora_path,
                                       bool has_scoped_lora_file);

  // Creates a new context handler from the provided session config struct.
  // If a session specific lora is provided, the lora will be loaded and the
  // corresponding lora id will be assigned.
  absl::StatusOr<std::unique_ptr<ContextHandler>> CreateContextHandler(
      const SessionConfig& session_config);

  // Clones the context handler.
  // The cloned context handler will have the same shared processed context as
  // the original context handler.
  // The cloned context handler's runtime config and runtime state however will
  // be copied from the original context handler, thus the values will initially
  // be the same, but can be different afterward.
  absl::StatusOr<std::unique_ptr<ContextHandler>> CloneContextHandler(
      std::shared_ptr<const ContextHandler> llm_context_handler);

  // Acquires the executor without any context handler. This function should
  // only be called when the usage of the returned executor does not involve any
  // state updates, e.g. CreateContext, GetCurrentStep(), etc.
  absl::StatusOr<std::unique_ptr<LlmExecutor>> AcquireExecutor()
      ABSL_LOCKS_EXCLUDED(executor_mutex_);

  // Acquires the executor after loading the provided context handle.
  // Typically, this function is called instead of AcquireExecutor() when the
  // usage of the returned executor involves any state updates, e.g. prefill,
  // decode, etc.
  // Note the method try to lock llm_executor_mutex_ and audio_executor_mutex_
  // in order to clone the audio context if needed, thus other methods should
  // not try to acquire the audio executor within the llm executor mutex.
  // TODO(b/483136581): Refactor the locking mechanism.
  absl::StatusOr<std::unique_ptr<LlmExecutor>>
  AcquireExecutorWithContextHandler(
      std::shared_ptr<ContextHandler> new_context_handle)
      ABSL_LOCKS_EXCLUDED(executor_mutex_)
          ABSL_LOCKS_EXCLUDED(audio_executor_mutex_);

  // Try to load the vision executor if the vision executor is not loaded.
  absl::Status TryLoadingVisionExecutor()
      ABSL_LOCKS_EXCLUDED(vision_executor_mutex_);

  // Acquires the vision executor.
  absl::StatusOr<std::unique_ptr<VisionExecutor>> AcquireVisionExecutor()
      ABSL_LOCKS_EXCLUDED(vision_executor_mutex_);

  // Try to load the audio executor if the audio executor is not loaded.
  absl::Status TryLoadingAudioExecutor()
      ABSL_LOCKS_EXCLUDED(audio_executor_mutex_);

  // Acquires the audio executor.
  absl::StatusOr<std::unique_ptr<AudioExecutor>> AcquireAudioExecutor()
      ABSL_LOCKS_EXCLUDED(audio_executor_mutex_);

  // Returns the audio executor properties.
  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      ABSL_LOCKS_EXCLUDED(audio_executor_mutex_);

  // Returns the vision executor properties.
  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      ABSL_LOCKS_EXCLUDED(vision_executor_mutex_);

 private:
  // Creates the litert environment if it is not created yet.
  absl::Status MaybeCreateLitertEnv();

  // Guards the llm_executor_.
  absl::Mutex executor_mutex_;

  // Maintains the gpu executor.
  std::shared_ptr<LlmExecutor> llm_executor_ ABSL_GUARDED_BY(executor_mutex_);

  // Maintains the current llm context.
  std::shared_ptr<ContextHandler> current_handler_
      ABSL_GUARDED_BY(executor_mutex_);

  // Map lora id from hash. If lora is provided by lora path, lora path will be
  // treated as the hash key.
  absl::flat_hash_map<std::string, uint32_t> lora_hash_to_id_;

  // The mutex lock for the vision executor.
  absl::Mutex vision_executor_mutex_;

  std::shared_ptr<VisionExecutor> vision_executor_
      ABSL_GUARDED_BY(vision_executor_mutex_);

  // The vision executor options, needed for loading the vision executor.
  std::unique_ptr<VisionExecutorSettings> vision_executor_settings_;

  // The mutex lock for the audio executor.
  absl::Mutex audio_executor_mutex_;

  std::shared_ptr<AudioExecutor> audio_executor_
      ABSL_GUARDED_BY(audio_executor_mutex_);

  // The audio executor options, needed for loading the audio executor.
  std::unique_ptr<AudioExecutorSettings> audio_executor_settings_;

  // The potential litert compiled model environment for the vision and audio
  // executor.
  ::litert::Environment* absl_nullable litert_env_;

  // The backup litert compiled model environment for the vision and audio
  // executor, created if litert_env is not provided when resource manager is
  // created.
  std::unique_ptr<::litert::Environment> backup_litert_env_;

  // The llm executor settings.
  std::optional<LlmExecutorSettings> llm_executor_settings_;

  friend class LockedLlmExecutor;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_RESOURCE_MANAGER_H_
