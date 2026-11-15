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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EXECUTOR_BACKEND_REGISTRY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EXECUTOR_BACKEND_REGISTRY_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Declares orchestration and sampling capabilities of a dynamically registered
// executor backend so EngineSettings and SessionConfig do not need to branch on
// specific Backend enum values.
struct BackendTraits {
  // Whether SessionAdvanced should instantiate an external Sampler and pass
  // output_logits / decoded_ids to Decode().
  bool use_external_sampler = true;
  // Default sampler backend when SessionConfig does not explicitly set one.
  Backend default_sampler_backend = Backend::CPU;
};

// Registry for custom `ModelResources` file-loader factories keyed by file
// extension (e.g. ".sbs") and optional magic header bytes.
//
// Custom file formats registered via `LITERT_LM_REGISTER_MODEL_RESOURCES` or
// `ModelResourcesRegistry::Register` / `TryRegister` are automatically
// dispatched during `Engine` initialization when `CanLoad(model_assets)`
// matches, allowing custom backends to supply `Tokenizer`, `LlmMetadata`, and
// backend-specific model resources through the `ModelResources` interface.
class ModelResourcesRegistry {
 public:
  using CreatorFunc =
      std::function<absl::StatusOr<std::unique_ptr<ModelResources>>(
          const ModelAssets& model_assets)>;

  static ModelResourcesRegistry& Instance();

  // Registers a `ModelResources` factory for `extension` (e.g. ".sbs",
  // case-insensitive) and optional `magic_bytes`. Overwrites any existing
  // registration for the same extension. Intended for static registration via
  // `LITERT_LM_REGISTER_MODEL_RESOURCES`.
  void Register(absl::string_view extension, CreatorFunc creator,
                absl::string_view magic_bytes = "") ABSL_LOCKS_EXCLUDED(mutex_);

  // Runtime registration. Returns `AlreadyExistsError` if a loader for the same
  // extension is already registered, or `InvalidArgumentError` if `extension`
  // or `creator` is empty.
  absl::Status TryRegister(absl::string_view extension, CreatorFunc creator,
                           absl::string_view magic_bytes = "")
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Removes a registered `ModelResources` factory for `extension`. Returns
  // `NotFoundError` if not registered.
  absl::Status Unregister(absl::string_view extension)
      ABSL_LOCKS_EXCLUDED(mutex_);

  bool IsRegistered(absl::string_view extension) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns true if `model_assets` matches any registered extension or magic
  // header bytes.
  bool CanLoad(const ModelAssets& model_assets) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Creates `ModelResources` using the registered loader that matches
  // `model_assets` (by file extension or magic header bytes).
  absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      const ModelAssets& model_assets) const ABSL_LOCKS_EXCLUDED(mutex_);

  // Creates `ModelResources` using the loader registered for `extension`.
  absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      absl::string_view extension, const ModelAssets& model_assets) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  ModelResourcesRegistry() = default;

 private:
  struct Entry {
    std::string extension;  // Lowercase with leading '.', e.g. ".sbs".
    std::string magic_bytes;
    CreatorFunc creator;
  };

  std::optional<Entry> FindMatchingEntry(const ModelAssets& model_assets) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  mutable absl::Mutex mutex_;
  std::vector<Entry> entries_ ABSL_GUARDED_BY(mutex_);
};

// Registry for backend executor factories keyed by `Backend` or backend name.
//
// Two registration patterns are supported:
//
// 1. Static registration at program load time via
//    `LITERT_LM_REGISTER_EXECUTOR_BACKEND(...)` in a `.cc` file linked with
//    `alwayslink = 1`. This uses `Register()`, which overwrites any existing
//    entry for the same backend.
//
// 2. Dynamic registration at runtime (e.g. from `main()`, an application
//    embedding LiteRT-LM, a plugin loaded via `dlopen`, or a test) via
//    `TryRegister()` / `Unregister()`. Dynamic registration fails with
//    `AlreadyExistsError` instead of silently replacing an existing backend.
//
// Backend IDs assigned to backend names are stable for the lifetime of the
// process: unregistering and re-registering the same name yields the same
// `Backend` ID. A backend must be registered before `Engine` creation for that
// backend; unregistering it does not affect engines that were already created.
class ExecutorBackendRegistry {
 public:
  using CreatorFunc =
      std::function<absl::StatusOr<std::unique_ptr<LlmExecutor>>(
          const LlmExecutorSettings& executor_settings,
          ModelResources& model_resources)>;

  static ExecutorBackendRegistry& Instance();

  // Registers a backend by string name (e.g. "gemma_cpp", "llama_cpp"),
  // dynamically assigning a `Backend` ID via `RegisterCustomBackend`, and
  // returns the assigned `Backend` ID. Overwrites any existing registration.
  // Intended for static registration via LITERT_LM_REGISTER_EXECUTOR_BACKEND.
  Backend Register(absl::string_view backend_name, CreatorFunc creator,
                   BackendTraits traits = {}) ABSL_LOCKS_EXCLUDED(mutex_);

  void Register(Backend backend, CreatorFunc creator, BackendTraits traits = {})
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Runtime registration. Returns the assigned `Backend` ID, or
  // `AlreadyExistsError` if a backend with the same name is already registered,
  // or `InvalidArgumentError` if `creator` is empty.
  absl::StatusOr<Backend> TryRegister(absl::string_view backend_name,
                                      CreatorFunc creator,
                                      BackendTraits traits = {})
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status TryRegister(Backend backend, CreatorFunc creator,
                           BackendTraits traits = {})
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Removes a registered backend. Returns `NotFoundError` if not registered.
  absl::Status Unregister(Backend backend) ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status Unregister(absl::string_view backend_name)
      ABSL_LOCKS_EXCLUDED(mutex_);

  bool IsRegistered(Backend backend) const ABSL_LOCKS_EXCLUDED(mutex_);

  bool IsRegistered(absl::string_view backend_name) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  std::optional<BackendTraits> GetTraits(Backend backend) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<std::unique_ptr<LlmExecutor>> Create(
      Backend backend, const LlmExecutorSettings& executor_settings,
      ModelResources& model_resources) const ABSL_LOCKS_EXCLUDED(mutex_);

  ExecutorBackendRegistry() = default;

 private:
  struct Entry {
    CreatorFunc creator;
    BackendTraits traits;
  };

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<Backend, Entry> entries_ ABSL_GUARDED_BY(mutex_);
};

#define LITERT_LM_REGISTER_EXECUTOR_BACKEND_CONCAT_INNER(x, y) x##y
#define LITERT_LM_REGISTER_EXECUTOR_BACKEND_CONCAT(x, y) \
  LITERT_LM_REGISTER_EXECUTOR_BACKEND_CONCAT_INNER(x, y)

#define LITERT_LM_REGISTER_MODEL_RESOURCES(extension_or_format, ...)   \
  [[maybe_unused]] static const bool                                   \
  LITERT_LM_REGISTER_EXECUTOR_BACKEND_CONCAT(kRegisterModelResources_, \
                                             __COUNTER__) = []() {     \
    ::litert::lm::ModelResourcesRegistry::Instance().Register(         \
        extension_or_format, __VA_ARGS__);                             \
    return true;                                                       \
  }()

#define LITERT_LM_REGISTER_EXECUTOR_BACKEND(backend_name_or_enum, ...)  \
  [[maybe_unused]] static const bool                                    \
  LITERT_LM_REGISTER_EXECUTOR_BACKEND_CONCAT(kRegisterExecutorBackend_, \
                                             __COUNTER__) = []() {      \
    ::litert::lm::ExecutorBackendRegistry::Instance().Register(         \
        backend_name_or_enum, __VA_ARGS__);                             \
    return true;                                                        \
  }()

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EXECUTOR_BACKEND_REGISTRY_H_
