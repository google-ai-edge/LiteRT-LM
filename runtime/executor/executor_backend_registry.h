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
#include <utility>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

class EngineSettings;

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

// Encapsulates the backend executor and associated model artifacts produced by
// a registered backend factory. This decouples Engine orchestration from
// LiteRT-specific ModelResources/Environment initialization.
struct BackendInstance {
  std::unique_ptr<LlmExecutor> llm_executor;
  std::unique_ptr<support::Tokenizer> tokenizer;
  std::unique_ptr<proto::LlmMetadata> llm_metadata;
  // Optional: Only populated by backends that still rely on ModelResources.
  std::unique_ptr<ModelResources> model_resources = nullptr;
};

// Registry for backend executor factories keyed by `Backend` or backend name.
class ExecutorBackendRegistry {
 public:
  using CreatorFunc = std::function<absl::StatusOr<BackendInstance>(
      const EngineSettings& engine_settings)>;

  static ExecutorBackendRegistry& Instance();

  // Registers a backend by string name (e.g. "gemma_cpp", "llama_cpp"),
  // dynamically assigning a `Backend` ID via `RegisterCustomBackend`, and
  // returns the assigned `Backend` ID.
  Backend Register(absl::string_view backend_name, CreatorFunc creator,
                   BackendTraits traits = {}) ABSL_LOCKS_EXCLUDED(mutex_);

  void Register(Backend backend, CreatorFunc creator, BackendTraits traits = {})
      ABSL_LOCKS_EXCLUDED(mutex_);

  bool IsRegistered(Backend backend) const ABSL_LOCKS_EXCLUDED(mutex_);

  bool IsRegistered(absl::string_view backend_name) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  std::optional<BackendTraits> GetTraits(Backend backend) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<BackendInstance> Create(
      Backend backend, const EngineSettings& engine_settings) const
      ABSL_LOCKS_EXCLUDED(mutex_);

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

#define LITERT_LM_REGISTER_EXECUTOR_BACKEND(backend_name_or_enum, ...) \
  static const bool LITERT_LM_REGISTER_EXECUTOR_BACKEND_CONCAT(        \
      kRegisterExecutorBackend_, __COUNTER__) = []() {                 \
    ::litert::lm::ExecutorBackendRegistry::Instance().Register(        \
        backend_name_or_enum, __VA_ARGS__);                            \
    return true;                                                       \
  }()

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EXECUTOR_BACKEND_REGISTRY_H_
