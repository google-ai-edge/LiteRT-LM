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

#include "runtime/util/streamed_weights_manager.h"

#ifdef __EMSCRIPTEN__
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include <webgpu/webgpu_cpp.h>
#include "weight_loader/external_weight_loader_litert.h"  // from @litert
#endif  // __EMSCRIPTEN__

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/util/data_stream.h"

namespace litert::lm {
namespace {

// Normalizes a model type wire string or enum name (e.g. "TF_LITE_EMBEDDER") to
// the canonical lowercase wire string used as the registry key, matching
// TfLiteModelTypeToWireString() and the other by-name ModelResources APIs.
std::string NormalizeModelTypeKey(absl::string_view model_type_str) {
  return absl::AsciiStrToLower(model_type_str);
}

// Process-wide registry of the DataStreams holding the external weights of the
// submodels which are being loaded by streaming, keyed by model type wire
// string, plus the model type of the submodel which the calling thread is
// currently compiling.
//
// Engines may be created concurrently on different threads, so the registry is
// guarded by a mutex. The currently compiling model, however, is a per-thread
// property: a thread sets it right before calling CompiledModel::Create(),
// during which the WebGPU delegate synchronously invokes the weight upload
// callback (UploadStoredWeightsOnWeb()) on that same thread, which reads it
// back. Keeping it thread-local lets concurrent compilations on different
// threads not clobber each other's model type, and needs no locking.
class WeightsStreams {
 public:
  // Returns the singleton instance.
  static WeightsStreams& Get() {
    static absl::NoDestructor<WeightsStreams> instance;
    return *instance;
  }

  void SetCurrentModel(absl::string_view model_type_str) {
    CurrentModel() = NormalizeModelTypeKey(model_type_str);
  }

  std::string GetCurrentModel() const { return CurrentModel(); }

  void Store(absl::string_view model_type_str,
             std::shared_ptr<DataStream> stream) {
    absl::MutexLock lock(mutex_);
    streams_[NormalizeModelTypeKey(model_type_str)] = std::move(stream);
  }

  absl::Status Read(absl::string_view model_type_str, uint64_t offset,
                    uint64_t size, void* buffer) {
    const std::string key = NormalizeModelTypeKey(model_type_str);
    absl::MutexLock lock(mutex_);
    auto it = streams_.find(key);
    if (it == streams_.end() || it->second == nullptr) {
      return absl::NotFoundError(absl::StrCat(
          "Stored weights stream not found for model type: ", model_type_str));
    }
    return it->second->ReadAndDiscard(buffer, offset, size);
  }

  absl::Status Clear(absl::string_view model_type_str) {
    const std::string key = NormalizeModelTypeKey(model_type_str);
    absl::MutexLock lock(mutex_);
    auto it = streams_.find(key);
    if (it != streams_.end()) {
      if (it->second) {
        (void)it->second->Discard(0, UINT64_MAX);
      }
      streams_.erase(it);
    }
    return absl::OkStatus();
  }

  absl::Status ClearAll() {
    absl::MutexLock lock(mutex_);
    for (auto& [model_type, stream] : streams_) {
      if (stream) {
        (void)stream->Discard(0, UINT64_MAX);
      }
    }
    streams_.clear();
    return absl::OkStatus();
  }

 private:
  friend class absl::NoDestructor<WeightsStreams>;
  WeightsStreams() = default;

  // Returns the model type wire string of the submodel currently being compiled
  // by the calling thread, or empty if none.
  static std::string& CurrentModel() {
    thread_local std::string current_model;
    return current_model;
  }

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, std::shared_ptr<DataStream>> streams_
      ABSL_GUARDED_BY(mutex_);
};

#ifdef __EMSCRIPTEN__
absl::Status UploadStoredWeightsOnWeb(
    const wgpu::Queue& queue,
    absl::Span<const weight_loader::WebWeightUploadRequest> requests) {
  std::vector<weight_loader::WebWeightUploadRequest> sorted_requests(
      requests.begin(), requests.end());
  // Requests should be sorted so we can efficiently discard chunks.
  // Otherwise, DataStream caches any weight we skip over while reading until
  // we eventually come back and read it.
  std::sort(sorted_requests.begin(), sorted_requests.end(),
            [](const auto& a, const auto& b) { return a.offset < b.offset; });

  constexpr size_t kChunkSize = 4 * 1024 * 1024;  // 4 MB
  std::vector<uint8_t> chunk_buf(kChunkSize);
  const std::string model_type_str = GetCurrentlyCompilingModel();

  for (const auto& req : sorted_requests) {
    uint64_t bytes_uploaded = 0;
    while (bytes_uploaded < req.length) {
      const size_t chunk_size =
          std::min<uint64_t>(kChunkSize, req.length - bytes_uploaded);
      absl::Status status =
          ReadStoredWeights(model_type_str, req.offset + bytes_uploaded,
                            chunk_size, chunk_buf.data());
      if (!status.ok()) {
        return status;
      }

      // WebGPU writeBuffer requires data size to be a multiple of 4 bytes.
      const size_t aligned_size = (chunk_size + 3) & ~static_cast<size_t>(3);
      if (aligned_size > chunk_size) {
        std::memset(chunk_buf.data() + chunk_size, 0,
                    aligned_size - chunk_size);
      }
      queue.WriteBuffer(req.buffer, bytes_uploaded, chunk_buf.data(),
                        aligned_size);
      bytes_uploaded += chunk_size;
    }
  }
  return absl::OkStatus();
}
#endif  // __EMSCRIPTEN__

}  // namespace

// The free functions below are the public API: besides regular callers, the
// executors and the web bindings bind to them as weak symbols (e.g.
// `extern void SetCurrentlyCompilingModel(absl::string_view)
// __attribute__((weak))` in litert_compiled_model_executor_utils.cc), so their
// names and signatures must not change.

void SetCurrentlyCompilingModel(absl::string_view model_type_str) {
  WeightsStreams::Get().SetCurrentModel(model_type_str);
}

std::string GetCurrentlyCompilingModel() {
  return WeightsStreams::Get().GetCurrentModel();
}

void StoreWeightsStream(absl::string_view model_type_str,
                        std::shared_ptr<DataStream> stream) {
#ifdef __EMSCRIPTEN__
  weight_loader::RegisterWebWeightUploadCallback(&UploadStoredWeightsOnWeb);
#endif  // __EMSCRIPTEN__
  WeightsStreams::Get().Store(model_type_str, std::move(stream));
}

absl::Status ReadStoredWeights(absl::string_view model_type_str,
                               uint64_t offset, uint64_t size, void* buffer) {
  return WeightsStreams::Get().Read(model_type_str, offset, size, buffer);
}

absl::Status ClearStoredWeightsStream(absl::string_view model_type_str) {
  return WeightsStreams::Get().Clear(model_type_str);
}

absl::Status ClearStoredWeightsStreams() {
  return WeightsStreams::Get().ClearAll();
}

}  // namespace litert::lm
