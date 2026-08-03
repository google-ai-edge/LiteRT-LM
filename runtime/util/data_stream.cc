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

#include "runtime/util/data_stream.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"

namespace litert::lm {

SubStream::~SubStream() {
  if (parent_) {
    // Note: We ignore errors here as we are in a destructor.
    (void)parent_->Discard(offset_, size_);
  }
}

absl::Status SubStream::ReadAndDiscard(void* buffer, uint64_t offset,
                                       uint64_t size) {
  RETURN_IF_ERROR(CheckBounds(offset, size));
  if (parent_) {
    return parent_->ReadAndDiscard(buffer, offset_ + offset, size);
  }
  return absl::FailedPreconditionError("Parent stream is null");
}

absl::Status SubStream::ReadAndPreserve(void* buffer, uint64_t offset,
                                        uint64_t size) {
  RETURN_IF_ERROR(CheckBounds(offset, size));
  if (parent_) {
    return parent_->ReadAndPreserve(buffer, offset_ + offset, size);
  }
  return absl::FailedPreconditionError("Parent stream is null");
}

absl::Status SubStream::Discard(uint64_t offset, uint64_t size) {
  RETURN_IF_ERROR(CheckBounds(offset, size));
  if (parent_) {
    return parent_->Discard(offset_ + offset, size);
  }
  return absl::FailedPreconditionError("Parent stream is null");
}

absl::Status SubStream::CheckBounds(uint64_t offset, uint64_t size) const {
  // Equivalent to `offset + size > size_`
  if (size > size_ || offset > size_ - size) {
    return absl::OutOfRangeError(
        absl::StrCat("Exceeded bounds of substream. offset: ", offset,
                     ", size: ", size, ", Max size: ", size_));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<DataStream>> SubStream::OpenSubStream(
    uint64_t offset, uint64_t size) {
  // Check if the requested substream fits within this SubStream's bounds.
  // Note that the parent DataStream::OpenSubStream method doesn't do this for
  // us since DataStream doesn't know its own size.
  RETURN_IF_ERROR(CheckBounds(offset, size));
  // Call the base class implementation to check this SubStream's
  // locked_regions_ and create a new SubStream child of this one.
  return DataStream::OpenSubStream(offset, size);
}

absl::StatusOr<std::unique_ptr<DataStream>> DataStream::OpenSubStream(
    uint64_t offset, uint64_t size) {
  for (const auto& region : locked_regions_) {
    // Check for overlap: Is [offset, offset + size) overlapping with
    // [region.first, region.first + region.second)? Overlap exists if
    // offset < region_end AND region_start < offset + size
    if (offset < region.first + region.second && region.first < offset + size) {
      return absl::AlreadyExistsError(absl::StrCat(
          "Failed to open substream: requested region [", offset, ", ",
          offset + size, ") overlaps with an existing locked region [",
          region.first, ", ", region.first + region.second, ")"));
    }
  }
  locked_regions_.emplace_back(offset, size);
  return std::make_unique<SubStream>(this, offset, size);
}

}  // namespace litert::lm
