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

#include "omni/base/model_cache.h"

#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <system_error>  // NOLINT
#include <utility>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/util/file_util.h"

namespace litert::omni {
namespace {

constexpr absl::string_view kXnnpackCacheSuffix = ".xnnpack_cache";
// File name suffixes appended to the model cache key by the GPU delegate.
constexpr absl::string_view kGpuProgramCacheSuffix =
    "_mldrift_program_cache.bin";
constexpr absl::string_view kGpuWeightCacheSuffix = "_mldrift_weight_cache.bin";

// Returns true if `s` is `<digits>_<digits>`, the format of
// `lm::GetFileCacheIdentifier`.
bool IsFileCacheIdentifier(absl::string_view s) {
  const size_t sep = s.find('_');
  if (sep == absl::string_view::npos || sep == 0 || sep + 1 == s.size()) {
    return false;
  }
  const auto all_digits = [](absl::string_view part) {
    return absl::c_all_of(part, [](char c) { return absl::ascii_isdigit(c); });
  };
  return all_digits(s.substr(0, sep)) && all_digits(s.substr(sep + 1));
}

// Returns `<cache.dir>/<cache.key><suffix>` after deleting stale caches of the
// same model, see `ConfigureGpuModelCache`.
std::string GetCachePathAndCleanStale(absl::string_view model_name,
                                      const ModelCacheConfig& cache,
                                      absl::string_view suffix) {
  const std::string current = absl::StrCat(cache.key, suffix);
  const std::string prefix = absl::StrCat(model_name, "_");
  // Iterate with explicit error codes: the implicit `operator++` of a
  // range-based loop throws on I/O errors.
  std::error_code ec;
  for (std::filesystem::directory_iterator it(cache.dir, ec), end;
       !ec && it != end; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    absl::string_view id = name;
    if (name == current || !absl::ConsumePrefix(&id, prefix) ||
        !absl::ConsumeSuffix(&id, suffix) || !IsFileCacheIdentifier(id)) {
      continue;
    }
    std::error_code remove_ec;
    std::filesystem::remove(it->path(), remove_ec);
  }
  return (std::filesystem::path(cache.dir) / current).string();
}

}  // namespace

absl::StatusOr<ModelCacheConfig> ResolveModelCache(
    absl::string_view model_name, absl::string_view model_path,
    absl::string_view cache_dir) {
  if (cache_dir == kNoCacheDir) {
    return ModelCacheConfig{};
  }
  absl::StatusOr<std::string> model_id = lm::GetFileCacheIdentifier(model_path);
  if (!model_id.ok()) {
    return ModelCacheConfig{};
  }
  const bool explicit_dir = !cache_dir.empty();
  // A bare file name has an empty parent: the model lives in the current
  // directory.
  const std::filesystem::path model_dir = [&] {
    std::filesystem::path parent =
        std::filesystem::path(std::string(model_path)).parent_path();
    return parent.empty() ? std::filesystem::path(".") : parent;
  }();
  std::string dir = explicit_dir ? std::string(cache_dir) : model_dir.string();
  if (explicit_dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
  }
  if (!lm::IsDirectoryWritable(dir)) {
    if (!explicit_dir) {
      return ModelCacheConfig{};
    }
    // `cache_dir` also holds the model file, e.g. a read-only directory of
    // pre-installed models (the Kotlin API has no separate model path), so
    // load without caches instead of failing.
    std::error_code ec;
    if (std::filesystem::equivalent(model_dir, dir, ec)) {
      ABSL_LOG(WARNING) << "cache_dir " << dir
                        << " is not writable; compiled-model caches disabled.";
      return ModelCacheConfig{};
    }
    return absl::PermissionDeniedError(absl::StrCat(
        "cache_dir is not a writable directory: ", dir,
        ". Set cache_dir to \":nocache\" to disable compiled-model caches."));
  }
  return ModelCacheConfig{
      .dir = std::move(dir),
      .key = absl::StrCat(model_name, "_", *model_id),
  };
}

absl::Status ConfigureCpuModelCache(absl::string_view model_name,
                                    const ModelCacheConfig& cache,
                                    ::litert::CpuOptions& cpu_options) {
  if (cache.key.empty()) {
    return absl::OkStatus();
  }
  // Persist XNNPack-packed weights in a file-backed cache: they are then
  // mmap'ed (clean, reclaimable pages) instead of living in the native heap,
  // and later loads skip re-packing.
  const std::string cache_path =
      GetCachePathAndCleanStale(model_name, cache, kXnnpackCacheSuffix);
  LITERT_RETURN_IF_ERROR(
      cpu_options.SetXNNPackWeightCachePath(cache_path.c_str()));
  return absl::OkStatus();
}

absl::Status ConfigureGpuModelCache(absl::string_view model_name,
                                    const ModelCacheConfig& cache,
                                    ::litert::GpuOptions& gpu_options) {
  if (cache.key.empty()) {
    return absl::OkStatus();
  }
  // Serialize compiled GPU programs and GPU-layout weights so later loads skip
  // shader compilation and weight conversion.
  for (absl::string_view suffix :
       {kGpuProgramCacheSuffix, kGpuWeightCacheSuffix}) {
    GetCachePathAndCleanStale(model_name, cache, suffix);
  }
  LITERT_RETURN_IF_ERROR(gpu_options.SetSerializationDir(cache.dir.c_str()));
  LITERT_RETURN_IF_ERROR(gpu_options.SetModelCacheKey(cache.key.c_str()));
  LITERT_RETURN_IF_ERROR(gpu_options.SetSerializeProgramCache(true));
  LITERT_RETURN_IF_ERROR(gpu_options.SetSerializeExternalTensors(true));
  return absl::OkStatus();
}

}  // namespace litert::omni
