// Copyright 2024 The ODML Authors.
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

#include <fcntl.h>
#include <sys/mman.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

class MemoryMappedFilePosix : public MemoryMappedFile {
 public:
  MemoryMappedFilePosix(uint64_t length, void* data)
      : length_(length), data_(data) {}
  ~MemoryMappedFilePosix() override {
    if (data_) {
      munmap(data_, length_);
    }
  }

  // Move constructor
  MemoryMappedFilePosix(MemoryMappedFilePosix&& other) noexcept
      : length_(other.length_), data_(other.data_) {
    // After transferring ownership of the data pointer and length,
    // we must reset the other object so its destructor doesn't free
    // the memory we just took ownership of.
    other.length_ = 0;
    other.data_ = nullptr;
  }

  // Move assignment
  MemoryMappedFilePosix& operator=(MemoryMappedFilePosix&& other) noexcept {
    if (this != &other) {
      // Free existing resource before taking ownership of the new one
      if (data_ != nullptr) {
        munmap(data_, length_);
      }

      // Transfer ownership from the other object
      length_ = other.length_;
      data_ = other.data_;

      // Reset the other object
      other.length_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  // Disable copy operations.
  MemoryMappedFilePosix(const MemoryMappedFilePosix&) = delete;
  MemoryMappedFilePosix& operator=(const MemoryMappedFilePosix&) = delete;

  uint64_t length() override { return length_; }

  void* data() override { return data_; }

 private:
  uint64_t length_;
  void* data_;
};

}  // namespace

// static
size_t MemoryMappedFile::GetOffsetAlignment() { return getpagesize(); }

// static
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MemoryMappedFile::Create(
    absl::string_view path) {
  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(path));
  return Create(scoped_file.file());
}

// static
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MemoryMappedFile::Create(
    int file, uint64_t offset, uint64_t length, absl::string_view key) {
  RET_CHECK_EQ(offset % GetOffsetAlignment(), 0)
      << "Offset must be a multiple of page size : " << offset << ", "
      << GetOffsetAlignment();

  ASSIGN_OR_RETURN(size_t file_size, ScopedFile::GetSize(file));
  RET_CHECK_GE(file_size, length + offset) << "Length and offset too large.";
  if (length == 0) {
    length = file_size - offset;
  }

  void* data =
      mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, file, offset);
  RET_CHECK_NE(data, MAP_FAILED) << "Failed to map, error: " << strerror(errno);
  RET_CHECK_NE(data, nullptr) << "Failed to map.";
#ifdef __APPLE__
  // Mark it not needed to avoid unnecessary page loading on MacOS or iOS.
  if (madvise(data, length, MADV_DONTNEED) != 0) {
    ABSL_LOG(WARNING) << "madvise failed: " << strerror(errno);
  }
#else
  if (madvise(data, length, MADV_WILLNEED) != 0) {
    ABSL_LOG(WARNING) << "madvise failed: " << strerror(errno);
  }
#endif

  return std::make_unique<MemoryMappedFilePosix>(length, data);
}

absl::StatusOr<std::unique_ptr<MemoryMappedFile>>
MemoryMappedFile::CreateMutable(absl::string_view path) {
  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::OpenWritable(path));
  return CreateMutable(scoped_file.file());
}

absl::StatusOr<std::unique_ptr<MemoryMappedFile>>
MemoryMappedFile::CreateMutable(int file, uint64_t offset, uint64_t length,
                                absl::string_view key) {
  RET_CHECK_EQ(offset % GetOffsetAlignment(), 0)
      << "Offset must be a multiple of page size : " << offset << ", "
      << GetOffsetAlignment();

  ASSIGN_OR_RETURN(size_t file_size, ScopedFile::GetSize(file));
  RET_CHECK_GE(file_size, length + offset) << "Length and offset too large.";
  if (length == 0) {
    length = file_size - offset;
  }
  if (length == 0) {
    return absl::InvalidArgumentError("Cannot mmap empty file.");
  }

  void* data =
      mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, file, offset);
  RET_CHECK_NE(data, MAP_FAILED) << "Failed to map, error: " << strerror(errno);
  RET_CHECK_NE(data, nullptr) << "Failed to map.";

  return std::make_unique<MemoryMappedFilePosix>(length, data);
}

}  // namespace litert::lm
