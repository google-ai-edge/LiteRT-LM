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

#include "runtime/executor/executor_backend_registry.h"

#include <optional>
#include <utility>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

ExecutorBackendRegistry& ExecutorBackendRegistry::Instance() {
  static absl::NoDestructor<ExecutorBackendRegistry> instance;
  return *instance;
}

Backend ExecutorBackendRegistry::Register(absl::string_view backend_name,
                                          CreatorFunc creator,
                                          BackendTraits traits) {
  const Backend backend = RegisterCustomBackend(backend_name);
  Register(backend, std::move(creator), traits);
  return backend;
}

void ExecutorBackendRegistry::Register(Backend backend, CreatorFunc creator,
                                       BackendTraits traits) {
  absl::MutexLock lock(mutex_);
  entries_[backend] = Entry{
      .creator = std::move(creator),
      .traits = traits,
  };
}

bool ExecutorBackendRegistry::IsRegistered(Backend backend) const {
  absl::MutexLock lock(mutex_);
  return entries_.contains(backend);
}

bool ExecutorBackendRegistry::IsRegistered(
    absl::string_view backend_name) const {
  auto backend_or = GetBackendFromString(backend_name);
  if (!backend_or.ok()) {
    return false;
  }
  return IsRegistered(*backend_or);
}

std::optional<BackendTraits> ExecutorBackendRegistry::GetTraits(
    Backend backend) const {
  absl::MutexLock lock(mutex_);
  auto it = entries_.find(backend);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  return it->second.traits;
}

absl::StatusOr<BackendInstance> ExecutorBackendRegistry::Create(
    Backend backend, const EngineSettings& engine_settings) const {
  CreatorFunc creator;
  {
    absl::MutexLock lock(mutex_);
    auto it = entries_.find(backend);
    if (it == entries_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "No executor backend registered for: ", GetBackendString(backend)));
    }
    creator = it->second.creator;
  }
  return creator(engine_settings);
}

}  // namespace litert::lm
