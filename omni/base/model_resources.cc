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

#include "omni/base/model_resources.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert

namespace litert::omni {

ModelResources::ModelResources(std::shared_ptr<Environment> env)
    : env_(std::move(env)) {}

absl::Status ModelResources::AddCompiledModel(
    absl::string_view key, std::shared_ptr<CompiledModel> model) {
  if (model == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot add null CompiledModel for key: ", key));
  }
  models_.insert_or_assign(std::string(key), std::move(model));
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<CompiledModel>> ModelResources::GetCompiledModel(
    absl::string_view key) const {
  auto it = models_.find(key);
  if (it == models_.end()) {
    return absl::NotFoundError(
        absl::StrCat("CompiledModel not found for key: ", key));
  }
  return it->second;
}

bool ModelResources::HasCompiledModel(absl::string_view key) const {
  return models_.contains(key);
}

void ModelResources::SetEnvironment(std::shared_ptr<Environment> env) {
  env_ = std::move(env);
}

std::shared_ptr<Environment> ModelResources::environment() const {
  return env_;
}

}  // namespace litert::omni
