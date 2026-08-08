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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_DATA_STREAM_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_DATA_STREAM_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {

// A simple data stream for reading data in chunks.
// This is intentionally not thread-safe to keep it compatible with
// single-threaded Emscripten, and loading from a serial stream is a
// fundamentally single-threaded operation anyway.
// A `SubStream` is a view into a parent `DataStream`. It holds a weak pointer
// to its parent, so it does not keep the parent alive. The user must ensure
// that the parent `DataStream` outlives all of its `SubStream`s.
class DataStream : public std::enable_shared_from_this<DataStream> {
 public:
  virtual ~DataStream() = default;

  // Reads `size` bytes starting at `offset` into `buffer`. The implementation
  // may discard the data after reading to save memory.
  virtual absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                                      uint64_t size) = 0;

  // Reads `size` bytes starting at `offset` into `buffer`. The implementation
  // should preserve the data after reading if possible, for future reads.
  virtual absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                                       uint64_t size) = 0;

  // Discards `size` bytes starting at `offset`. The implementation may use this
  // as a hint to release memory. Discarding a region that has already been
  // discarded must be supported.
  virtual absl::Status Discard(uint64_t offset, uint64_t size) = 0;

  // Opens a view of this data stream restricted to the range [`offset`,
  // `offset` + `size`).
  // Note that substreams cannot overlap regions in the parent stream, even if
  // the first substream is destroyed.
  virtual absl::StatusOr<std::shared_ptr<DataStream>> OpenSubStream(
      uint64_t offset, uint64_t size);

 protected:
  std::vector<std::pair<uint64_t, uint64_t>> locked_regions_;
};

// This is not thread-safe.
// A `SubStream` is a view into a parent `DataStream`. It holds a weak pointer
// to its parent, so it does not keep the parent alive. The user must ensure
// that the parent `DataStream` outlives all of its `SubStream`s.
class SubStream : public DataStream {
 public:
  SubStream(std::shared_ptr<DataStream> parent, uint64_t offset, uint64_t size)
      : parent_(parent), offset_(offset), size_(size) {}

  ~SubStream() override;

  absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                              uint64_t size) override;

  absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                               uint64_t size) override;

  absl::Status Discard(uint64_t offset, uint64_t size) override;

  absl::StatusOr<std::shared_ptr<DataStream>> OpenSubStream(
      uint64_t offset, uint64_t size) override;

 private:
  std::shared_ptr<DataStream> parent_;
  uint64_t offset_;
  uint64_t size_;

  absl::Status CheckBounds(uint64_t offset, uint64_t size) const;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_DATA_STREAM_H_
