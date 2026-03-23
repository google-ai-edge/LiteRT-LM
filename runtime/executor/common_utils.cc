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

#include "runtime/executor/common_utils.h"

#include <cstdint>
#include <cstring>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/status_macros.h"

namespace litert::lm::executor::utils {

absl::Status ExpandBuffer(const uint8_t* src_data,
                          absl::Span<const int> src_shape, uint8_t* dst_data,
                          absl::Span<const int> dst_shape,
                          size_t element_size) {
  RET_CHECK_EQ(src_shape.size(), dst_shape.size());
  int expansion_axis = -1;
  for (int i = 0; i < src_shape.size(); ++i) {
    if (src_shape[i] != dst_shape[i]) {
      if (expansion_axis != -1) {
        return absl::InvalidArgumentError(
            "Tensors differ in more than one dimension.");
      }
      if (dst_shape[i] < src_shape[i]) {
        return absl::InvalidArgumentError(
            "Destination tensor dimension is smaller than source along an "
            "axis.");
      }
      expansion_axis = i;
    }
  }
  if (expansion_axis == -1) {
    return absl::InvalidArgumentError("No expansion axis found.");
  }

  int64_t dest_total_elements = 1;
  for (int dim : dst_shape) {
    dest_total_elements *= dim;
  }
  memset(dst_data, 0, dest_total_elements * element_size);

  int64_t inner_block_size_in_elements = 1;
  for (int i = expansion_axis + 1; i < src_shape.size(); ++i) {
    inner_block_size_in_elements *= src_shape[i];
  }
  const size_t inner_block_size_in_bytes =
      inner_block_size_in_elements * element_size;

  int64_t outer_block_count = 1;
  for (int i = 0; i < expansion_axis; ++i) {
    outer_block_count *= src_shape[i];
  }

  int64_t src_outer_block_stride_in_elements =
      src_shape[expansion_axis] * inner_block_size_in_elements;
  int64_t dest_outer_block_stride_in_elements =
      dst_shape[expansion_axis] * inner_block_size_in_elements;

  for (int64_t i = 0; i < outer_block_count; ++i) {
    // Calculate the starting pointer for this outer block
    const uint8_t* src_outer_block_start =
        src_data + i * src_outer_block_stride_in_elements * element_size;
    uint8_t* dest_outer_block_start =
        dst_data + i * dest_outer_block_stride_in_elements * element_size;

    // Copy each inner block from source to destination
    for (int j = 0; j < src_shape[expansion_axis]; ++j) {
      const uint8_t* src_inner_block =
          src_outer_block_start + j * inner_block_size_in_bytes;
      uint8_t* dest_inner_block =
          dest_outer_block_start + j * inner_block_size_in_bytes;
      memcpy(dest_inner_block, src_inner_block, inner_block_size_in_bytes);
    }
  }

  return absl::OkStatus();
}

absl::Status CopyBuffer(const TensorBuffer& src_buffer,
                        TensorBuffer& dst_buffer, size_t src_offset,
                        size_t dst_offset, int64_t size) {
  LITERT_ASSIGN_OR_RETURN(auto src_buffer_size, src_buffer.PackedSize());
  LITERT_ASSIGN_OR_RETURN(auto dst_buffer_size, dst_buffer.PackedSize());
  if (size == -1) {
    size = src_buffer_size - src_offset;
  }
  LITERT_RETURN_IF_ERROR(src_offset + size <= src_buffer_size);
  LITERT_RETURN_IF_ERROR(dst_offset + size <= dst_buffer_size);

  LITERT_ASSIGN_OR_RETURN(auto src_read_lock,
                          TensorBufferScopedLock::Create(
                              src_buffer, TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto dst_write_lock,
                          TensorBufferScopedLock::Create(
                              dst_buffer, TensorBuffer::LockMode::kWrite));

  memcpy(static_cast<char*>(dst_write_lock.second) + dst_offset,
         static_cast<const char*>(src_read_lock.second) + src_offset, size);
  return absl::OkStatus();
}

}  // namespace litert::lm::executor::utils
