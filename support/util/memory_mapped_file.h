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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_UTIL_MEMORY_MAPPED_FILE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_UTIL_MEMORY_MAPPED_FILE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/util/scoped_file.h"

namespace litert::support {

// Represents a memory mapped file. All memory will be accessible while this
// object exists and will be cleaned up when it is destroyed.
class MemoryMappedFile {
 public:
  enum class Access {
    kRead = 0,
    kCopy = 1,
    kWrite = 2,
  };

  // Move constructor and move assignment operator
  MemoryMappedFile(MemoryMappedFile&&) = default;
  MemoryMappedFile& operator=(MemoryMappedFile&&) = default;

  // Delete copy operations
  MemoryMappedFile(const MemoryMappedFile&) = delete;
  MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

  // Gets the required alignment for a file offset passed to Create().
  static size_t GetOffsetAlignment();

  // Creates a read-only MemoryMappedFile object.
  static absl::StatusOr<std::unique_ptr<MemoryMappedFile>> Create(
      absl::string_view path, Access access = Access::kCopy);
  // Creates a MemoryMappedFile object from the platform file handle. This does
  // not take ownership of the passed handle. The `key` passed here is an
  // optimization when mapping the same file with different offsets.
  static absl::StatusOr<std::unique_ptr<MemoryMappedFile>> Create(
      ScopedFile::PlatformFile file, uint64_t offset = 0u, uint64_t length = 0u,
      absl::string_view key = "", Access access = Access::kCopy);

  // Creates a mutable MemoryMappedFile object, any modification through data()
  // pointer will be carried over to the underlying path.
  static absl::StatusOr<std::unique_ptr<MemoryMappedFile>> CreateMutable(
      absl::string_view path);
  // Creates a MemoryMappedFile object from the platform file handle. This does
  // not take ownership of the passed handle. The `key` passed here is an
  // optimization when mapping the same file with different offsets.
  static absl::StatusOr<std::unique_ptr<MemoryMappedFile>> CreateMutable(
      ScopedFile::PlatformFile file, uint64_t offset = 0u, uint64_t length = 0u,
      absl::string_view key = "");

  virtual ~MemoryMappedFile() = default;

  // Returns the file size in bytes.
  virtual uint64_t length() = 0;

  // Returns a pointer to the file data.
  virtual void* data() = 0;

 protected:
  // Protected default constructor to prevent direct instantiation
  MemoryMappedFile() = default;
};

// Represents an in-memory file from a memory data. This is useful for
// in-memory data that are not backed by a file. For example, the unit test data
// and the image or audio data passed as raw bytes.
class InMemoryFile : public MemoryMappedFile {
 public:
  // Creates a InMemoryFile object from a memory data.
  static absl::StatusOr<std::unique_ptr<InMemoryFile>> Create(
      std::string blob) {
    return absl::WrapUnique(new InMemoryFile(std::move(blob)));
  }

  // Returns the size of the memory data.
  uint64_t length() override { return blob_.size(); }

  // Returns a pointer to the memory data.
  void* data() override { return blob_.data(); }

 private:
  explicit InMemoryFile(std::string blob) : blob_(std::move(blob)) {}

  std::string blob_;
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_UTIL_MEMORY_MAPPED_FILE_H_
