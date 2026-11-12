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
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
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

// Input of GPU models exported with runtime-BMM LoRA. It holds the channel
// range of the LoRA BMM ops (see ml_drift::LlmRuntimeParams) and is not a LoRA
// weight.
inline constexpr absl::string_view kLoRAParamTensorName = "lora_param_tensor";

// Returns true if the given name is a LoRA input name for the model.
// The LoRA name is in the format of
// "(query|key|value|post)_w_prime_(left|right)_[0-num_layers)" or
// "lora_atten_(q|k|v|o)_(a|b)_prime_weight_[0-num_layers)".
bool IsLoRAInputName(absl::string_view name);

// Returns true if the buffer for the model input `name` is provided by a LoRA:
// a LoRA weight input (see IsLoRAInputName()) or "lora_param_tensor". Such
// buffers are owned by the LoRA and shared with every caller that binds them,
// so they must not be cleared or overwritten when resetting model inputs.
bool IsLoRABufferInputName(absl::string_view name);

// Computes the end-channel value for `lora_param_tensor` from the dimensions of
// all LoRA weight inputs of a signature. Returns the largest non-rank extent
// across `lora_input_dims`.
absl::StatusOr<int32_t> ComputeLoRAParamTensorEnd(
    absl::Span<const std::vector<int32_t>> lora_input_dims);

// Fills `params` (the int32 elements of `lora_param_tensor`) using the channel
// range computed from `lora_input_dims`: element 0 is set to 0 (start index)
// and the remaining elements are set to ComputeLoRAParamTensorEnd().
absl::Status FillLoRAParamTensorValues(
    absl::Span<const std::vector<int32_t>> lora_input_dims,
    absl::Span<int32_t> params);

// Populates `param_buffer` (`kLoRAParamTensorName`) with the LoRA channel
// range computed from the LoRA weight inputs of `signature_name` in
// `compiled_model`.
absl::Status PopulateLoRAParamTensor(
    const litert::CompiledModel& compiled_model,
    absl::string_view signature_name, litert::TensorBuffer& param_buffer);

// Populates `lora_param_tensor` (`kLoRAParamTensorName`) in `input_buffers_map`
// using PopulateLoRAParamTensor() when that entry is present; otherwise no-op.
absl::Status InitLoRAParamTensorIfPresent(
    const litert::CompiledModel& compiled_model,
    absl::string_view signature_name,
    absl::flat_hash_map<absl::string_view, litert::TensorBuffer>&
        input_buffers_map);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LORA_UTIL_H_
