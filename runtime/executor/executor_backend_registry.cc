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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {
namespace {

std::string NormalizeExtension(absl::string_view extension) {
  std::string lower = absl::AsciiStrToLower(extension);
  if (!lower.empty() && lower.front() != '.') {
    return absl::StrCat(".", lower);
  }
  return lower;
}

}  // namespace

ModelResourcesRegistry& ModelResourcesRegistry::Instance() {
  static absl::NoDestructor<ModelResourcesRegistry> instance;
  return *instance;
}

void ModelResourcesRegistry::Register(absl::string_view extension,
                                      CreatorFunc creator,
                                      absl::string_view magic_bytes) {
  const std::string norm_ext = NormalizeExtension(extension);
  absl::MutexLock lock(mutex_);
  for (auto& entry : entries_) {
    if (entry.extension == norm_ext) {
      entry.magic_bytes = std::string(magic_bytes);
      entry.creator = std::move(creator);
      return;
    }
  }
  entries_.push_back(Entry{
      .extension = norm_ext,
      .magic_bytes = std::string(magic_bytes),
      .creator = std::move(creator),
  });
}

absl::Status ModelResourcesRegistry::TryRegister(
    absl::string_view extension, CreatorFunc creator,
    absl::string_view magic_bytes) {
  if (extension.empty()) {
    return absl::InvalidArgumentError(
        "File format extension must not be empty.");
  }
  if (!creator) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ModelResources creator for extension ", extension, " is empty."));
  }
  const std::string norm_ext = NormalizeExtension(extension);
  absl::MutexLock lock(mutex_);
  for (const auto& entry : entries_) {
    if (entry.extension == norm_ext) {
      return absl::AlreadyExistsError(absl::StrCat(
          "ModelResources loader already registered for extension: ",
          norm_ext));
    }
  }
  entries_.push_back(Entry{
      .extension = norm_ext,
      .magic_bytes = std::string(magic_bytes),
      .creator = std::move(creator),
  });
  return absl::OkStatus();
}

absl::Status ModelResourcesRegistry::Unregister(absl::string_view extension) {
  const std::string norm_ext = NormalizeExtension(extension);
  absl::MutexLock lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    if (it->extension == norm_ext) {
      entries_.erase(it);
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat(
      "No ModelResources loader registered for extension: ", norm_ext));
}

bool ModelResourcesRegistry::IsRegistered(absl::string_view extension) const {
  if (extension.empty()) {
    return false;
  }
  const std::string norm_ext = NormalizeExtension(extension);
  absl::MutexLock lock(mutex_);
  for (const auto& entry : entries_) {
    if (entry.extension == norm_ext) {
      return true;
    }
  }
  return false;
}

std::optional<ModelResourcesRegistry::Entry>
ModelResourcesRegistry::FindMatchingEntry(
    const ModelAssets& model_assets) const {
  std::vector<Entry> snapshot;
  {
    absl::MutexLock lock(mutex_);
    if (entries_.empty()) {
      return std::nullopt;
    }
    snapshot = entries_;
  }

  const absl::string_view model_path = model_assets.GetPath().value_or("");
  if (!model_path.empty()) {
    const std::string lower_path = absl::AsciiStrToLower(model_path);
    for (const auto& entry : snapshot) {
      if (!entry.extension.empty() &&
          absl::EndsWith(lower_path, entry.extension)) {
        return entry;
      }
    }
  }

  size_t max_magic_len = 0;
  for (const auto& entry : snapshot) {
    max_magic_len = std::max(max_magic_len, entry.magic_bytes.size());
  }
  if (max_magic_len == 0) {
    return std::nullopt;
  }

  if (model_assets.HasMemoryMappedFile()) {
    auto mmfile_or = model_assets.GetMemoryMappedFile();
    if (!mmfile_or.ok() || *mmfile_or == nullptr) {
      return std::nullopt;
    }
    const auto& mmfile = *mmfile_or;
    const size_t header_len = static_cast<size_t>(
        std::min<uint64_t>(mmfile->length(), max_magic_len));
    absl::string_view header(reinterpret_cast<const char*>(mmfile->data()),
                             header_len);
    for (const auto& entry : snapshot) {
      if (!entry.magic_bytes.empty() &&
          absl::StartsWith(header, entry.magic_bytes)) {
        return entry;
      }
    }
    return std::nullopt;
  }

  auto scoped_file = model_assets.GetScopedFile().value_or(nullptr);
  if (scoped_file != nullptr) {
    auto file_size_or = ScopedFile::GetSize(scoped_file->file());
    if (!file_size_or.ok() || *file_size_or == 0) {
      return std::nullopt;
    }
    const uint64_t bytes_to_map =
        std::min<uint64_t>(*file_size_or, max_magic_len);
    auto mapped_file_or = MemoryMappedFile::Create(
        scoped_file->file(), /*offset=*/0, /*size=*/bytes_to_map);
    if (!mapped_file_or.ok()) {
      return std::nullopt;
    }
    absl::string_view header(
        reinterpret_cast<const char*>((*mapped_file_or)->data()),
        static_cast<size_t>((*mapped_file_or)->length()));
    for (const auto& entry : snapshot) {
      if (!entry.magic_bytes.empty() &&
          absl::StartsWith(header, entry.magic_bytes)) {
        return entry;
      }
    }
  }
  return std::nullopt;
}

bool ModelResourcesRegistry::CanLoad(const ModelAssets& model_assets) const {
  return FindMatchingEntry(model_assets).has_value();
}

absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesRegistry::Create(
    const ModelAssets& model_assets) const {
  std::optional<Entry> entry = FindMatchingEntry(model_assets);
  if (!entry.has_value()) {
    return absl::NotFoundError(
        "No registered ModelResources loader matches the given ModelAssets.");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ModelResources> resources,
                        entry->creator(model_assets));
  if (resources == nullptr) {
    return absl::InternalError(
        absl::StrCat("Registered ModelResources factory for '",
                     entry->extension, "' returned a null ModelResources."));
  }
  return resources;
}

absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesRegistry::Create(
    absl::string_view extension, const ModelAssets& model_assets) const {
  const std::string norm_ext = NormalizeExtension(extension);
  CreatorFunc creator;
  {
    absl::MutexLock lock(mutex_);
    for (const auto& entry : entries_) {
      if (entry.extension == norm_ext) {
        creator = entry.creator;
        break;
      }
    }
  }
  if (!creator) {
    return absl::NotFoundError(absl::StrCat(
        "No ModelResources loader registered for extension: ", norm_ext));
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ModelResources> resources,
                        creator(model_assets));
  if (resources == nullptr) {
    return absl::InternalError(
        absl::StrCat("Registered ModelResources factory for '", norm_ext,
                     "' returned a null ModelResources."));
  }
  return resources;
}

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

absl::StatusOr<Backend> ExecutorBackendRegistry::TryRegister(
    absl::string_view backend_name, CreatorFunc creator, BackendTraits traits) {
  if (backend_name.empty()) {
    return absl::InvalidArgumentError("Backend name must not be empty.");
  }
  const Backend backend = RegisterCustomBackend(backend_name);
  ABSL_RETURN_IF_ERROR(TryRegister(backend, std::move(creator), traits));
  return backend;
}

absl::Status ExecutorBackendRegistry::TryRegister(Backend backend,
                                                  CreatorFunc creator,
                                                  BackendTraits traits) {
  if (!creator) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Creator for backend ", GetBackendString(backend), " is empty."));
  }
  absl::MutexLock lock(mutex_);
  auto [it, inserted] =
      entries_.try_emplace(backend, Entry{
                                        .creator = std::move(creator),
                                        .traits = traits,
                                    });
  if (!inserted) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Executor backend already registered: ", GetBackendString(backend)));
  }
  return absl::OkStatus();
}

absl::Status ExecutorBackendRegistry::Unregister(Backend backend) {
  absl::MutexLock lock(mutex_);
  if (entries_.erase(backend) == 0) {
    return absl::NotFoundError(absl::StrCat(
        "No executor backend registered for: ", GetBackendString(backend)));
  }
  return absl::OkStatus();
}

absl::Status ExecutorBackendRegistry::Unregister(
    absl::string_view backend_name) {
  ABSL_ASSIGN_OR_RETURN(const Backend backend,
                        GetBackendFromString(backend_name));
  return Unregister(backend);
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

absl::StatusOr<std::unique_ptr<LlmExecutor>> ExecutorBackendRegistry::Create(
    Backend backend, const LlmExecutorSettings& executor_settings,
    ModelResources& model_resources) const {
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
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<LlmExecutor> executor,
                        creator(executor_settings, model_resources));
  if (executor == nullptr) {
    return absl::InternalError(absl::StrCat("Registered backend factory for '",
                                            GetBackendString(backend),
                                            "' returned a null llm_executor."));
  }
  return executor;
}

}  // namespace litert::lm
