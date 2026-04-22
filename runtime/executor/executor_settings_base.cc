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

#include "runtime/executor/executor_settings_base.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/data_stream.h"
#include "runtime/util/file_util.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

std::string GetBackendString(Backend backend) {
  switch (backend) {
    case Backend::CPU_ARTISAN:
      return "CPU_ARTISAN";
    case Backend::GPU_ARTISAN:
      return "GPU_ARTISAN";
    case Backend::GPU:
      return "GPU";
    case Backend::CPU:
      return "CPU";
    case Backend::GOOGLE_TENSOR_ARTISAN:
      return "GOOGLE_TENSOR_ARTISAN";
    case Backend::NPU:
      return "NPU";
    default:
      return "UNSPECIFIED";
  }
}

std::ostream& operator<<(std::ostream& os, const Backend& backend) {
  return os << GetBackendString(backend);
}

absl::StatusOr<Backend> GetBackendFromString(absl::string_view backend_str) {
  if (absl::EqualsIgnoreCase(backend_str, "cpu")) {
    return Backend::CPU;
  } else if (absl::EqualsIgnoreCase(backend_str, "gpu")) {
    return Backend::GPU;
  } else if (absl::EqualsIgnoreCase(backend_str, "npu")) {
    return Backend::NPU;
  } else if (absl::EqualsIgnoreCase(backend_str, "gpu_artisan")) {
    return Backend::GPU_ARTISAN;
  } else if (absl::EqualsIgnoreCase(backend_str, "cpu_artisan")) {
    return Backend::CPU_ARTISAN;
  } else if (absl::EqualsIgnoreCase(backend_str, "google_tensor_artisan")) {
    return Backend::GOOGLE_TENSOR_ARTISAN;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend: ", backend_str,
                     ". Supported backends are: [CPU, GPU, NPU, GPU_ARTISAN, "
                     "CPU_ARTISAN, GOOGLE_TENSOR_ARTISAN]"));
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ActivationDataType& activation) {
  switch (activation) {
    case ActivationDataType::FLOAT32:
      return os << "FLOAT32";
    case ActivationDataType::FLOAT16:
      return os << "FLOAT16";
    case ActivationDataType::INT16:
      return os << "INT16";
    case ActivationDataType::INT8:
      return os << "INT8";
    default:
      return os << "UNKNOWN";
  }
}

absl::StatusOr<ActivationDataType> GetActivationDataTypeFromString(
    const std::string& activation_data_type) {
  if (absl::EqualsIgnoreCase(activation_data_type, "float32")) {
    return ActivationDataType::FLOAT32;
  } else if (absl::EqualsIgnoreCase(activation_data_type, "float16")) {
    return ActivationDataType::FLOAT16;
  } else if (absl::EqualsIgnoreCase(activation_data_type, "int16")) {
    return ActivationDataType::INT16;
  } else if (absl::EqualsIgnoreCase(activation_data_type, "int8")) {
    return ActivationDataType::INT8;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported activation data type: ", activation_data_type,
                     ". Supported activation data types are: [FLOAT32, "
                     "FLOAT16, INT16, INT8]"));
  }
}

std::ostream& operator<<(std::ostream& os,
                         const FakeWeightsMode& fake_weights_mode) {
  switch (fake_weights_mode) {
    case FakeWeightsMode::FAKE_WEIGHTS_NONE:
      return os << "FAKE_WEIGHTS_NONE";
    case FakeWeightsMode::FAKE_WEIGHTS_8BITS_ALL_LAYERS:
      return os << "FAKE_WEIGHTS_8BITS_ALL_LAYERS";
    case FakeWeightsMode::FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4:
      return os << "FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4";
    default:
      return os << "FAKE_WEIGHTS_NONE";
  }
}

std::ostream& operator<<(std::ostream& os, const FileFormat& file_format) {
  switch (file_format) {
    case FileFormat::TFLITE:
      return os << "TFLITE";
    case FileFormat::TASK:
      return os << "TASK";
    case FileFormat::LITERT_LM:
      return os << "LITERT_LM";
  }
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(absl::string_view model_path) {
  return ModelAssets(model_path);
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::ScopedFile> model_file) {
  return ModelAssets(std::move(model_file), /*model_path=*/"");
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::MemoryMappedFile> model_file) {
  return ModelAssets(std::move(model_file));
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::DataStream> data_stream) {
  return ModelAssets(std::move(data_stream));
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::ScopedFile> model_file,
    absl::string_view model_path) {
  return ModelAssets(std::move(model_file), model_path);
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::MemoryMappedFile> model_file,
    absl::string_view model_path) {
  return ModelAssets(std::move(model_file), model_path);
}

ModelAssets::ModelAssets(std::shared_ptr<litert::lm::ScopedFile> model_file,
                         absl::string_view model_path)
    : path_(model_path), scoped_file_(std::move(model_file)) {}

ModelAssets::ModelAssets(absl::string_view model_path)
    : path_(std::string(model_path)) {}

ModelAssets::ModelAssets(
    std::shared_ptr<litert::lm::MemoryMappedFile> model_file)
    : memory_mapped_file_(std::move(model_file)) {}

ModelAssets::ModelAssets(
    std::shared_ptr<litert::lm::MemoryMappedFile> model_file,
    absl::string_view model_path)
    : path_(model_path), memory_mapped_file_(std::move(model_file)) {}

ModelAssets::ModelAssets(std::shared_ptr<litert::lm::DataStream> data_stream)
    : data_stream_(std::move(data_stream)) {}

absl::StatusOr<absl::string_view> ModelAssets::GetPath() const {
  if (!path_.empty()) {
    return path_;
  }
  return absl::InvalidArgumentError("Assets were not created with a path.");
}

absl::StatusOr<std::shared_ptr<ScopedFile>> ModelAssets::GetScopedFile() const {
  if (!HasScopedFile()) {
    return absl::InvalidArgumentError(
        "Assets were not created with a scoped file.");
  }
  return scoped_file_;
}

absl::StatusOr<std::shared_ptr<MemoryMappedFile>>
ModelAssets::GetMemoryMappedFile() const {
  if (!HasMemoryMappedFile()) {
    return absl::InvalidArgumentError(
        "Assets were not created with a memory mapped file.");
  }
  return memory_mapped_file_;
}

absl::StatusOr<std::shared_ptr<DataStream>> ModelAssets::GetDataStream() const {
  if (!HasDataStream()) {
    return absl::InvalidArgumentError(
        "Assets were not created with a data stream.");
  }
  return data_stream_;
}

absl::StatusOr<std::shared_ptr<ScopedFile>> ModelAssets::GetOrCreateScopedFile()
    const {
  if (HasScopedFile()) {
    return scoped_file_;
  }
  if (HasMemoryMappedFile()) {
    return absl::InvalidArgumentError(
        "Cannot create ScopedFile from MemoryMappedFile.");
  }

  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(path_));
  return std::make_shared<ScopedFile>(std::move(scoped_file));
}

std::ostream& operator<<(std::ostream& os, const ModelAssets& model_assets) {
  if (model_assets.HasScopedFile()) {
    os << "model_file file descriptor ID: "
       << model_assets.GetScopedFile().value()->file() << "\n";
  } else if (model_assets.HasMemoryMappedFile()) {
    os << "model_file memory mapped file: "
       << model_assets.GetMemoryMappedFile().value()->data() << "\n";
  } else if (model_assets.HasDataStream()) {
    os << "model_file is loading from a data stream\n";
  } else {
    os << "model_path: " << model_assets.GetPath().value() << "\n";
  }
  os << "fake_weights_mode: " << model_assets.fake_weights_mode() << "\n";
  return os;
}

absl::StatusOr<ExecutorSettingsBase::CacheSuffix>
ExecutorSettingsBase::GetCacheSuffix(Backend backend,
                                     absl::string_view model_path,
                                     absl::string_view module_name) {
  if (backend != Backend::CPU && backend != Backend::GPU) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported backend for cache naming: ", GetBackendString(backend)));
  }

  static const auto* kValidModuleNames =
      new absl::flat_hash_set<absl::string_view>{"",
                                                 "vision_encoder",
                                                 "vision_adapter",
                                                 "streaming_audio_encoder",
                                                 "audio_adapter",
                                                 "static_audio_encoder"};
  if (!kValidModuleNames->contains(module_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid module name for cache naming: ", module_name));
  }

  CacheSuffix naming;
  std::string module_part =
      module_name.empty() ? "" : absl::StrCat(".", module_name);

  if (backend == Backend::CPU) {
    naming.weight_suffix = absl::StrCat(module_part, kXnnpackCacheSuffix);
  } else if (backend == Backend::GPU) {
    naming.program_suffix =
        module_name.empty()
            ? kMlDriftCacheSuffix
            : absl::StrCat(".mldrift_program_cache.", module_name, ".bin");
    naming.gpu_weight_cache_suffix =
        module_name.empty() ? std::string(Basename(model_path))
                            : absl::StrCat(Basename(model_path), module_part);
  }
  return naming;
}

absl::StatusOr<
    std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
ExecutorSettingsBase::GetWeightCacheFile(absl::string_view suffix,
                                         bool check_and_clean) const {
  // Cache is explicitly disabled.
  if (GetCacheDir() == ":nocache") {
    return absl::InvalidArgumentError("Cache is explicitly disabled.");
  }

  // Prefer to use the scoped cache file if it's set.
  if (GetScopedCacheFile()) {
    return GetScopedCacheFile();
  }

  ASSIGN_OR_RETURN(auto model_path, GetModelAssets().GetPath());

  // Get unique identifier based on the model file's content and metadata.
  std::string metadata_id = "";
  if (auto id_or = GetFileCacheIdentifier(model_path); id_or.ok()) {
    metadata_id = absl::StrCat("_", *id_or);
  } else {
    ABSL_LOG(WARNING) << "GetFileCacheIdentifier failed for " << model_path
                      << ": " << id_or.status();
  }

  std::string path;
  if (GetCacheDir().empty()) {
    path = absl::StrCat(model_path, suffix, metadata_id);
  } else {
    ASSIGN_OR_RETURN(
        path, JoinPath(GetCacheDir(), absl::StrCat(Basename(model_path), suffix,
                                                   metadata_id)));
  }

  // Try to delete stale caches if the current cache file doesn't exist.
  if (check_and_clean) {
    if (!FileExists(path)) {
      std::string dir_to_clean = GetCacheDir().empty()
                                     ? std::string(Dirname(model_path))
                                     : GetCacheDir();
      auto num_deleted_or =
          DeleteStaleCaches(dir_to_clean, Basename(model_path), suffix);
      if (!num_deleted_or.ok()) {
        ABSL_LOG(WARNING) << "Failed to clean stale caches: "
                          << num_deleted_or.status();
      } else {
        ABSL_LOG(INFO) << "Deleted " << *num_deleted_or
                       << " stale cache files.";
      }
    }
  }

  return path;
}

absl::StatusOr<
    std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
ExecutorSettingsBase::GetProgramCacheFile(absl::string_view suffix,
                                          bool check_and_clean) const {
  // Cache is explicitly disabled.
  if (GetCacheDir() == ":nocache") {
    return absl::InvalidArgumentError("Cache is explicitly disabled.");
  }

  // Prefer to use the scoped cache file if it's set.
  if (GetScopedProgramCacheFile()) {
    return GetScopedProgramCacheFile();
  }

  ASSIGN_OR_RETURN(auto model_path, GetModelAssets().GetPath());

  // Get unique identifier based on the model file's content and metadata.
  std::string metadata_id = "";
  if (auto id_or = GetFileCacheIdentifier(model_path); id_or.ok()) {
    metadata_id = absl::StrCat("_", *id_or);
  } else {
    ABSL_LOG(WARNING) << "GetFileCacheIdentifier failed for " << model_path
                      << ": " << id_or.status();
  }

  std::string path;
  if (GetCacheDir().empty()) {
    path = absl::StrCat(model_path, metadata_id, suffix);
  } else {
    ASSIGN_OR_RETURN(
        path, JoinPath(GetCacheDir(), absl::StrCat(Basename(model_path),
                                                   metadata_id, suffix)));
  }

  // Try to delete stale caches if the current cache file doesn't exist.
  if (check_and_clean) {
    if (!FileExists(path)) {
      ABSL_LOG(INFO) << "File does not exist: " << path
                     << " Cleaning stale caches.";
      std::string dir_to_clean = GetCacheDir().empty()
                                     ? std::string(Dirname(model_path))
                                     : GetCacheDir();
      auto num_deleted_or =
          DeleteStaleCaches(dir_to_clean, Basename(model_path), suffix);
      if (!num_deleted_or.ok()) {
        ABSL_LOG(WARNING) << "Failed to clean stale caches: "
                          << num_deleted_or.status();
      } else {
        ABSL_LOG(INFO) << "Deleted " << *num_deleted_or
                       << " stale cache files.";
      }
    }
  }

  return path;
}

}  // namespace litert::lm
