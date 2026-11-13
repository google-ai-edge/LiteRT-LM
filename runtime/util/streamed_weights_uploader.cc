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

#include "runtime/util/streamed_weights_uploader.h"

#ifdef __EMSCRIPTEN__
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include <webgpu/webgpu_cpp.h>
#include "weight_loader/external_weight_loader_litert.h"  // from @litert
#endif  // __EMSCRIPTEN__

#include <memory>
#include <utility>

#include "litert/cc/litert_options.h"  // from @litert
#include "runtime/util/data_stream.h"

namespace litert::lm {

#ifdef __EMSCRIPTEN__
// Uploads the requested weights by reading them from a DataStream.
class DataStreamWebWeightUploader : public weight_loader::WebWeightUploader {
 public:
  explicit DataStreamWebWeightUploader(std::shared_ptr<DataStream> stream)
      : stream_(std::move(stream)) {}

  // Drops the stream. Later uploads fail.
  void ReleaseStream() { stream_ = nullptr; }

  absl::Status Upload(
      const wgpu::Queue& queue,
      absl::Span<const weight_loader::WebWeightUploadRequest> requests)
      override {
    if (stream_ == nullptr) {
      return absl::FailedPreconditionError(
          "The streamed weights of the model have already been uploaded; they "
          "can't be uploaded again.");
    }
    std::vector<weight_loader::WebWeightUploadRequest> sorted_requests(
        requests.begin(), requests.end());
    // Requests should be sorted so we can efficiently discard chunks.
    // Otherwise, DataStream caches any weight we skip over while reading until
    // we eventually come back and read it.
    std::sort(sorted_requests.begin(), sorted_requests.end(),
              [](const auto& a, const auto& b) { return a.offset < b.offset; });

    constexpr size_t kChunkSize = 4 * 1024 * 1024;  // 4 MB
    std::vector<uint8_t> chunk_buf(kChunkSize);

    for (const auto& req : sorted_requests) {
      uint64_t bytes_uploaded = 0;
      while (bytes_uploaded < req.length) {
        const size_t chunk_size =
            std::min<uint64_t>(kChunkSize, req.length - bytes_uploaded);
        absl::Status status = stream_->ReadAndDiscard(
            chunk_buf.data(), req.offset + bytes_uploaded, chunk_size);
        if (!status.ok()) {
          return status;
        }

        // WebGPU writeBuffer requires data size to be a multiple of 4 bytes.
        const size_t aligned_size =
            (chunk_size + 3) & ~static_cast<size_t>(3);
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

 private:
  std::shared_ptr<DataStream> stream_;
};

StreamedWeightsUploader::StreamedWeightsUploader(
    std::shared_ptr<DataStream> stream)
    : web_uploader_(
          std::make_shared<DataStreamWebWeightUploader>(std::move(stream))) {}

StreamedWeightsUploader::~StreamedWeightsUploader() {
  web_uploader_->ReleaseStream();
}

void StreamedWeightsUploader::SetOnOptions(litert::Options& options) {
  options.SetWebWeightUploader(web_uploader_);
}
#else
StreamedWeightsUploader::StreamedWeightsUploader(
    std::shared_ptr<DataStream> stream)
    : stream_(std::move(stream)) {}

StreamedWeightsUploader::~StreamedWeightsUploader() = default;

void StreamedWeightsUploader::SetOnOptions(litert::Options& options) {}
#endif  // __EMSCRIPTEN__

}  // namespace litert::lm
