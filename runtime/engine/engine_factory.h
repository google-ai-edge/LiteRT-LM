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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_FACTORY_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/cpu_affinity_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

// Factory class to create Engine instances.
// The factory is a singleton class that is used to create Engine instances.
// It is used to decouple the engine creation logic from the engine
// implementation. This allows us to register different engine types and create
// them at runtime.
//
// Example usage:
//   // Create an engine instance of type kLiteRTCompiledModel.
//   auto engine = EngineFactory::Create(
//       EngineFactory::EngineType::kLiteRTCompiledModel,
//       std::move(engine_settings));
//   CHECK_OK(engine);
//
// Note:
//   - Ensure the desired engine type is registered (i.e., add the
//   implementation library as a dependency)
//
class EngineFactory {
 public:
  // Function signature to create an Engine instance.
  using Creator = absl::AnyInvocable<absl::StatusOr<std::unique_ptr<Engine>>(
      EngineSettings settings, absl::string_view input_prompt_as_hint)>;

  // The type of engine to create.
  enum class EngineType {
    kAdvancedLiteRTCompiledModel,
    kLiteRTCompiledModel,
    kLegacyTfLite,
    kAdvancedLegacyTfLite,
  };

  // Returns the string representation of the engine type.
  static std::string EngineTypeToString(EngineType engine_type) {
    switch (engine_type) {
      case EngineType::kLiteRTCompiledModel:
        return "kLiteRTCompiledModel";
      case EngineType::kLegacyTfLite:
        return "kLegacyTfLite";
      case EngineType::kAdvancedLegacyTfLite:
        return "kAdvancedLegacyTfLite";
      case EngineType::kAdvancedLiteRTCompiledModel:
        return "kAdvancedLiteRTCompiledModel";
      default:
        return "Unknown";
    }
  }

  // Creates a default Engine based on the given EngineSetting and the preferred
  // engines map. It picks the first available (registered) engine type mapped
  // to the backend specified in settings.
  static absl::StatusOr<std::unique_ptr<Engine>> CreateDefault(
      EngineSettings settings, absl::string_view input_prompt_as_hint = "") {
    auto& instance = Instance();
    Backend backend = settings.GetMutableMainExecutorSettings().GetBackend();

    auto it = instance.preferred_engines_.find(backend);
    if (it != instance.preferred_engines_.end()) {
      for (auto engine_type : it->second) {
        if (instance.registry_.contains(engine_type)) {
          return Create(engine_type, std::move(settings), input_prompt_as_hint);
        }
      }
    }

    std::string error_msg = absl::StrCat("No available engine for backend: ",
                                         GetBackendString(backend));

    if (it != instance.preferred_engines_.end()) {
      absl::StrAppend(&error_msg, ". Preferred engine types: [");
      for (size_t i = 0; i < it->second.size(); ++i) {
        absl::StrAppend(&error_msg, EngineTypeToString(it->second[i]));
        if (i < it->second.size() - 1) absl::StrAppend(&error_msg, ", ");
      }
      absl::StrAppend(&error_msg, "]");
    } else {
      absl::StrAppend(&error_msg,
                      ". No preferred engine types defined for this backend");
    }

    absl::StrAppend(&error_msg, ". Available (registered) engine types: [");
    auto available_types_or = instance.ListEngineTypes();
    if (available_types_or.ok()) {
      for (size_t i = 0; i < available_types_or->size(); ++i) {
        absl::StrAppend(&error_msg,
                        EngineTypeToString((*available_types_or)[i]));
        if (i < available_types_or->size() - 1)
          absl::StrAppend(&error_msg, ", ");
      }
    }
    absl::StrAppend(&error_msg, "]");

    return absl::NotFoundError(error_msg);
  }

  // Creates an Engine instance of the given type.
  static absl::StatusOr<std::unique_ptr<Engine>> Create(
      EngineType engine_type, EngineSettings settings,
      absl::string_view input_prompt_as_hint = "") {
    if (IsPixelTensorDevice()) {
      auto cores = GetPixelPerformanceCores();
      auto status = SetCpuAffinity(cores);
      if (!status.ok()) {
        ABSL_LOG(WARNING) << "Failed to set CPU affinity: " << status;
      }
    }

    auto& instance = Instance();
    auto it = instance.registry_.find(engine_type);
    if (it == instance.registry_.end()) {
      return absl::NotFoundError(absl::StrCat("Engine type not found: ",
                                              EngineTypeToString(engine_type)));
    }
    return it->second(std::move(settings), input_prompt_as_hint);
  };

  // Returns the singleton instance of the EngineFactory.
  static EngineFactory& Instance() {
    static EngineFactory* instance = new EngineFactory();
    return *instance;
  }

  // Registers a creator for the given engine type. Please use the
  // LITERT_LM_REGISTER_ENGINE macro to register an engine type.
  absl::Status Register(EngineType engine_type, Creator creator) {
    if (registry_.contains(engine_type)) {
      return absl::AlreadyExistsError(
          absl::StrCat("Engine type already exists: ", engine_type));
    }
    registry_[engine_type] = std::move(creator);
    return absl::OkStatus();
  }

  // Unregisters the creator for the given engine type.
  absl::Status Unregister(EngineType engine_type) {
    if (!registry_.contains(engine_type)) {
      return absl::NotFoundError(
          absl::StrCat("Engine type not found: ", engine_type));
    }
    registry_.erase(engine_type);
    return absl::OkStatus();
  }

  // Lists all registered engine types.
  absl::StatusOr<std::vector<EngineType>> ListEngineTypes() const {
    std::vector<EngineType> engine_types;
    for (const auto& [engine_type, creator] : registry_) {
      engine_types.push_back(engine_type);
    }
    return engine_types;
  }

 private:
  EngineFactory()
      : preferred_engines_({
            {Backend::CPU,
             {
                 EngineType::kAdvancedLiteRTCompiledModel,
                 EngineType::kLiteRTCompiledModel,
             }},
            {Backend::GPU,
             {
                 EngineType::kAdvancedLiteRTCompiledModel,
                 EngineType::kLiteRTCompiledModel,
             }},
            {Backend::GPU_ARTISAN,
             {
                 EngineType::kAdvancedLegacyTfLite,
                 EngineType::kLegacyTfLite,
             }},
            {Backend::NPU,
             {
                 EngineType::kAdvancedLiteRTCompiledModel,
                 EngineType::kLiteRTCompiledModel,
             }},
        }) {}

  // Use std::unordered_map instead of absl::flat_hash_map to avoid Windows
  // build/linker issues across DLL boundaries (see cl/403158423 and
  // b/508692203). std::unordered_map can also be faster when the number of
  // elements is low.
  std::unordered_map<EngineType, Creator> registry_;

  // Map of preferred engine types for each backend. The first available
  // (registered) engine type in the list will be selected by CreateDefault().
  // Use std::unordered_map for consistency with registry_ and to avoid similar
  // Windows issues.
  std::unordered_map<Backend, std::vector<EngineType>> preferred_engines_;
};

// Helper struct to register an engine type with the EngineFactory.
struct EngineRegisterer {
  EngineRegisterer(EngineFactory::EngineType engine_type,
                   EngineFactory::Creator creator) {
    absl::Status status =
        EngineFactory::Instance().Register(engine_type, std::move(creator));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to register engine: " << status;
    }
  }
};

#define LITERT_LM_CONCAT_INNER(x, y) x##y
#define LITERT_LM_CONCAT(x, y) LITERT_LM_CONCAT_INNER(x, y)
// Macro to register an engine type with the EngineFactory.
#define LITERT_LM_REGISTER_ENGINE(engine_type, creator)         \
  static const ::litert::lm::EngineRegisterer LITERT_LM_CONCAT( \
      _engine_registerer_, __COUNTER__)(engine_type, creator);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_FACTORY_H_
