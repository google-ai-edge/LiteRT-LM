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
#include <vector>

#include <webgpu/webgpu_cpp.h>
#include "weight_loader/external_weight_loader_litert.h"  // from @litert
#endif  // __EMSCRIPTEN__

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/util/data_stream.h"

namespace litert::lm {
namespace {

ModelType g_currently_compiling_model = ModelType::kUnknown;

// Global map of stored weights streams, mapped by ModelType.
std::unordered_map<ModelType, std::shared_ptr<DataStream>>&
GetStoredWeightsStreams() {
  static auto* const m =
      new std::unordered_map<ModelType, std::shared_ptr<DataStream>>();
  return *m;
}

// Global map of stored in-memory weights sections, mapped by ModelType. Used
// for submodels whose weights were loaded into host memory rather than
// left on the stream. The spans are not owned; see `StoreWeightsBuffer()`.
std::unordered_map<ModelType, absl::Span<const std::byte>>&
GetStoredWeightsBuffers() {
  static auto* const m =
      new std::unordered_map<ModelType, absl::Span<const std::byte>>();
  return *m;
}

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
  const int model_type_int = static_cast<int>(GetCurrentlyCompilingModel());

  for (const auto& req : sorted_requests) {
    uint64_t bytes_uploaded = 0;
    while (bytes_uploaded < req.length) {
      const size_t chunk_size =
          std::min<uint64_t>(kChunkSize, req.length - bytes_uploaded);
      absl::Status status =
          ReadStoredWeights(model_type_int, req.offset + bytes_uploaded,
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

void SetCurrentlyCompilingModel(ModelType model_type) {
  g_currently_compiling_model = model_type;
}

ModelType GetCurrentlyCompilingModel() { return g_currently_compiling_model; }

void StoreWeightsStream(ModelType model_type,
                        std::shared_ptr<DataStream> stream) {
#ifdef __EMSCRIPTEN__
  weight_loader::RegisterWebWeightUploadCallback(&UploadStoredWeightsOnWeb);
#endif  // __EMSCRIPTEN__
  GetStoredWeightsStreams()[model_type] = std::move(stream);
}

void StoreWeightsBuffer(ModelType model_type,
                        absl::Span<const std::byte> weights) {
#ifdef __EMSCRIPTEN__
  weight_loader::RegisterWebWeightUploadCallback(&UploadStoredWeightsOnWeb);
#endif  // __EMSCRIPTEN__
  GetStoredWeightsBuffers()[model_type] = weights;
}

absl::Status ReadStoredWeights(int model_type_int, uint64_t offset,
                               uint64_t size, void* buffer) {
  ModelType model_type = static_cast<ModelType>(model_type_int);
  auto& streams = GetStoredWeightsStreams();
  auto it = streams.find(model_type);
  if (it != streams.end() && it->second != nullptr) {
    return it->second->ReadAndDiscard(buffer, offset, size);
  }

  auto& buffers = GetStoredWeightsBuffers();
  auto buffer_it = buffers.find(model_type);
  if (buffer_it == buffers.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Stored weights stream not found for model type: ", model_type_int));
  }
  const absl::Span<const std::byte>& weights = buffer_it->second;
  if (offset > weights.size() || size > weights.size() - offset) {
    return absl::OutOfRangeError(absl::StrCat(
        "Read of ", size, " bytes at offset ", offset,
        " is out of range for the stored weights buffer of model type ",
        model_type_int, ", which holds ", weights.size(), " bytes"));
  }
  std::memcpy(buffer, weights.data() + offset, size);
  return absl::OkStatus();
}

absl::Status ClearStoredWeightsStream(ModelType model_type) {
  auto& streams = GetStoredWeightsStreams();
  auto it = streams.find(model_type);
  if (it != streams.end()) {
    if (it->second) {
      (void)it->second->Discard(0, UINT64_MAX);
    }
    streams.erase(it);
  }
  GetStoredWeightsBuffers().erase(model_type);
  return absl::OkStatus();
}

absl::Status ClearStoredWeightsStreams() {
  auto& streams = GetStoredWeightsStreams();
  for (auto& [model_type, stream] : streams) {
    if (stream) {
      (void)stream->Discard(0, UINT64_MAX);
    }
  }
  streams.clear();
  GetStoredWeightsBuffers().clear();
  return absl::OkStatus();
}

}  // namespace litert::lm
