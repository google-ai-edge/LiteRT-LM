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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_CACHE_H_

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert

namespace litert::omni {

// Sentinel `cache_dir` value that disables compiled-model caches.
inline constexpr absl::string_view kNoCacheDir = ":nocache";

// On-disk caches for compiled models:
//   CPU: `<dir>/<key>.xnnpack_cache` (XNNPack packed weights).
//   GPU: `<dir>/<key>_mldrift_{program,weight}_cache.bin` (compiled GPU
//        programs and GPU-layout weights).
// `key` is `<model_name>_<model file id>`, where the id is `<mtime>_<size>` of
// the model file (see `lm::GetFileCacheIdentifier`), so a replaced model never
// reuses caches built from an older one. Empty `dir` / `key` mean caching is
// disabled.
struct ModelCacheConfig {
  std::string dir;
  std::string key;
};

// Resolves the cache location for `model_name` whose weights come from
// `model_path`:
// - `cache_dir` is `kNoCacheDir` (":nocache"): caching is disabled.
// - `cache_dir` is empty: caches go next to the model file (LiteRT-LM
//   convention); skipped silently if that directory is not writable.
// - Otherwise caches go to `cache_dir`, which is created if missing. If it is
//   not writable, caching is skipped when it is the model file's directory
//   (read-only model directories are common); otherwise this is a
//   `PermissionDeniedError`.
// Caching is also skipped when `model_path` does not exist.
absl::StatusOr<ModelCacheConfig> ResolveModelCache(absl::string_view model_name,
                                                   absl::string_view model_path,
                                                   absl::string_view cache_dir);

// Points the XNNPack weight cache at `<cache.dir>/<cache.key>.xnnpack_cache`
// and deletes stale caches of the same model (see below). No-op when caching
// is disabled.
absl::Status ConfigureCpuModelCache(absl::string_view model_name,
                                    const ModelCacheConfig& cache,
                                    ::litert::CpuOptions& cpu_options);

// Enables GPU program and weight serialization under `cache.dir` with
// `cache.key` and deletes stale caches of the same model. No-op when caching
// is disabled.
//
// Stale caches are files named `<model_name>_<mtime>_<size><suffix>` whose id
// differs from the current one, i.e. built from an older version of the model
// file. Requiring a well-formed `<digits>_<digits>` id keeps caches of other
// models whose names merely start with `<model_name>_` untouched.
absl::Status ConfigureGpuModelCache(absl::string_view model_name,
                                    const ModelCacheConfig& cache,
                                    ::litert::GpuOptions& gpu_options);

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_CACHE_H_
