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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_RESOURCES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_RESOURCES_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert

namespace litert::omni {

// Generic key-value container for shared heavy resources (such as LiteRT
// CompiledModel instances and Environment) used across Omni pipeline sessions.
class ModelResources {
 public:
  ModelResources() = default;
  explicit ModelResources(std::shared_ptr<Environment> env);

  ~ModelResources() = default;

  // Registers a CompiledModel instance under a string key.
  absl::Status AddCompiledModel(absl::string_view key,
                                std::shared_ptr<CompiledModel> model);

  // Retrieves a CompiledModel instance by string key.
  absl::StatusOr<std::shared_ptr<CompiledModel>> GetCompiledModel(
      absl::string_view key) const;

  // Checks if a CompiledModel exists under the given string key.
  bool HasCompiledModel(absl::string_view key) const;

  // Sets the shared LiteRT Environment.
  void SetEnvironment(std::shared_ptr<Environment> env);

  // Returns the shared LiteRT Environment.
  std::shared_ptr<Environment> environment() const;

 private:
  std::shared_ptr<Environment> env_;
  absl::flat_hash_map<std::string, std::shared_ptr<CompiledModel>> models_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_RESOURCES_H_
