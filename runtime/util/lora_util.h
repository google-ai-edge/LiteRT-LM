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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LORA_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LORA_UTIL_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// A wrapper class for MemoryMappedFile with extra offset and size to
// automatically align when accessing the file. It is used since
// MemoryMappedFile does not accept an offset and size that is not aligned.
// Alignment is calculated based on MemoryMappedFile::GetOffsetAlignment().
class MemoryMappedFileWithAutoAlignment {
 public:
  // Creates a MemoryMappedFileWithAutoAlignment instance.
  //
  // @param file The platform-specific file handle. This is the file source.
  // @param offset The starting offset within the file. Can be not aligned.
  // @param size The size of the memory-mapped region. If 0, it maps to the end
  // of the file.
  // @param key An optional key for optimizing multiple mmaps.
  // @return A unique pointer to the created instance or an error status.
  static absl::StatusOr<std::unique_ptr<MemoryMappedFileWithAutoAlignment>>
  Create(ScopedFile::PlatformFile file, uint64_t offset = 0, uint64_t size = 0,
         absl::string_view key = "");

  ~MemoryMappedFileWithAutoAlignment() = default;

  // Returns a pointer to the data with the internal offset.
  void* data() const { return static_cast<uint8_t*>(file_->data()) + offset_; }

  // Returns the length of the data in bytes.
  uint64_t length() const { return size_; }

 private:
  MemoryMappedFileWithAutoAlignment(std::unique_ptr<MemoryMappedFile> file,
                                    uint64_t offset, uint64_t size)
      : file_(std::move(file)), offset_(offset), size_(size) {}

  std::unique_ptr<MemoryMappedFile> file_;
  // Internal offset and size when accessing the file_.
  uint64_t offset_;
  uint64_t size_;
};

// A BufferRef that owns the memory mapped file.
template <typename ByteT = uint8_t>
class MmapBufferRef : public litert::BufferRef<ByteT> {
 public:
  explicit MmapBufferRef(
      std::unique_ptr<MemoryMappedFileWithAutoAlignment> mapped_file)
      : litert::BufferRef<ByteT>(mapped_file->data(), mapped_file->length()),
        mapped_file_(std::move(mapped_file)) {}

  ~MmapBufferRef() override = default;

 private:
  std::unique_ptr<MemoryMappedFileWithAutoAlignment> mapped_file_;
};

// Returns true if the given name is a LoRA input name for the model.
// The LoRA name is in the format of
// "(query|key|value|post)_w_prime_(left|right)_[0-num_layers)",
// "lora_atten_(q|k|v|o)_(a|b)_prime_weight_[0-num_layers)", or
// "lora_audio_attn_(q|k|v|o)_(a|b)_weight_<n>", or
// "lora_audio_(module)_(a|b)_weight_<n>".
bool IsLoRAInputName(absl::string_view name);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LORA_UTIL_H_
