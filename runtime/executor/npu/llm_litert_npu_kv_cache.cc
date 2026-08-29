// Copyright 2026 Google LLC.
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

#include "runtime/executor/npu/llm_litert_npu_kv_cache.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(__ANDROID__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

constexpr int kSliceOuterRank = 2;

// Copies KV cache slice to main cache, transposing if necessary.
// Loop order ensures sequential writes to dst (enables write-coalescing on
// uncached memory).
template <typename T, bool cache_transposed>
void TransposeCopy(uint8_t* dst, const uint8_t* src, int64_t real_len,
                   int64_t real_start, int64_t wp, int64_t cache_seq,
                   int64_t hidden_dim, int64_t slice_seq) {
  T* d = reinterpret_cast<T*>(dst);
  const T* s = reinterpret_cast<const T*>(src);
  if constexpr (cache_transposed) {
    // Outer loop over hidden_dim ensures sequential writes to d[h * cache_seq +
    // wrapped_pos]
    for (int64_t h = 0; h < hidden_dim; ++h) {
      for (int64_t s_idx = 0; s_idx < real_len; ++s_idx) {
        int64_t wrapped_pos = (wp + s_idx) % cache_seq;
        int64_t slice_s = real_start + s_idx;
        d[h * cache_seq + wrapped_pos] = s[slice_s * hidden_dim + h];
      }
    }
  } else {
    // Outer loop over s_idx ensures sequential writes to d[wrapped_pos *
    // hidden_dim + h]
    for (int64_t s_idx = 0; s_idx < real_len; ++s_idx) {
      int64_t wrapped_pos = (wp + s_idx) % cache_seq;
      int64_t slice_s = real_start + s_idx;
      for (int64_t h = 0; h < hidden_dim; ++h) {
        d[wrapped_pos * hidden_dim + h] = s[h * slice_seq + slice_s];
      }
    }
  }
}

}  // namespace

int64_t GetKvCacheInitValue(ModelResources& resources) {
  int64_t kv_cache_init_value = 0;
  if (auto metadata_status = resources.GetLlmMetadata(); metadata_status.ok()) {
    const proto::LlmMetadata* metadata = *metadata_status;
    if (metadata && metadata->has_kv_cache_init_value()) {
      kv_cache_init_value = metadata->kv_cache_init_value();
    }
  }
  return kv_cache_init_value;
}

absl::Status FillKVCacheBuffer(TensorBuffer& buffer, int64_t init_value) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto size, buffer.PackedSize());
  LITERT_ASSIGN_OR_RETURN(
      auto lock, ::litert::TensorBufferScopedLock::Create(
                     buffer, ::litert::TensorBuffer::LockMode::kWrite));

  auto element_type = tensor_type.ElementType();
  if (element_type == ::litert::ElementType::Int16) {
    auto* ptr = static_cast<int16_t*>(lock.second);
    std::fill(ptr, ptr + size / sizeof(int16_t),
              static_cast<int16_t>(init_value));
  } else if (element_type == ::litert::ElementType::UInt16) {
    auto* ptr = static_cast<uint16_t*>(lock.second);
    std::fill(ptr, ptr + size / sizeof(uint16_t),
              static_cast<uint16_t>(init_value));
  } else {
    std::memset(lock.second, 0, size);
  }
  return absl::OkStatus();
}

absl::Status ClearKVCacheBuffers(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers,
    int64_t init_value) {
  for (auto& [buffer_name, buffer] : buffers) {
    if (buffer_name.starts_with(kKvCacheKRootName) ||
        buffer_name.starts_with(kKvCacheVRootName) ||
        buffer_name.starts_with(kKvCacheCRootName)) {
      LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(buffer, init_value));
    }
  }
  return absl::OkStatus();
}

absl::Status HWKVCacheUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& out_buffers,
    const absl::flat_hash_map<absl::string_view, HWQuantParams>& quant_params,
    bool enable_swa) {
  static constexpr absl::string_view kInputPos = "input_pos";
  if (!in_buffers.contains(kInputPos)) {
    return absl::InvalidArgumentError("Missing input_pos buffer");
  }
  auto& input_pos_buffer = in_buffers.at(kInputPos);

  LITERT_ASSIGN_OR_RETURN(auto pos_type, input_pos_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t pos_num_elements,
                          pos_type.Layout().NumElements());
  if (pos_num_elements == 0) {
    return absl::InvalidArgumentError("input_pos buffer is empty");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto pos_lock,
      ::litert::TensorBufferScopedLock::Create(
          input_pos_buffer, ::litert::TensorBuffer::LockMode::kRead));
  if (pos_lock.second == nullptr) {
    return absl::InternalError("Failed to lock input_pos buffer");
  }

  const int32_t* input_pos_ptr = static_cast<const int32_t*>(pos_lock.second);

  // Extract the valid mask if present, which is used to filter out padding
  // tokens and to only update the cache with valid tokens.
  static constexpr absl::string_view kValidMask = "valid_mask";
  const bool* valid_mask = nullptr;
  int64_t valid_mask_size = 0;
  std::optional<::litert::TensorBufferScopedLock> valid_mask_lock;
  if (in_buffers.contains(kValidMask)) {
    auto& buf = in_buffers.at(kValidMask);
    LITERT_ASSIGN_OR_RETURN(auto valid_mask_type, buf.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto num_elements,
                            valid_mask_type.Layout().NumElements());
    valid_mask_size = num_elements;
    if (valid_mask_type.ElementType() != ::litert::ElementType::Bool) {
      return absl::InvalidArgumentError("valid_mask must be Bool type");
    }
    LITERT_ASSIGN_OR_RETURN(auto lock,
                            ::litert::TensorBufferScopedLock::Create(
                                buf, ::litert::TensorBuffer::LockMode::kRead));
    valid_mask = static_cast<const bool*>(lock.second);
    valid_mask_lock.emplace(std::move(lock.first));
  }

  auto perform_update = [&](::litert::TensorBuffer& cache,
                            const ::litert::RankedTensorType& slice_type,
                            const void* slice_ptr, size_t slice_bytes,
                            absl::string_view cache_name,
                            absl::string_view slice_name) -> absl::Status {
    LITERT_ASSIGN_OR_RETURN(auto cache_type, cache.TensorType());

    int cache_rank = cache_type.Layout().Rank();
    int slice_rank = slice_type.Layout().Rank();
    if (cache_rank < 2 || slice_rank < 2) {
      return absl::InvalidArgumentError("Cache and slice must have rank >= 2");
    }

    auto cache_dims = cache_type.Layout().Dimensions();
    auto slice_dims = slice_type.Layout().Dimensions();

    LITERT_ASSIGN_OR_RETURN(size_t cache_bytes, cache.Size());

    if (cache_type.ElementType() != slice_type.ElementType()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cache and slice element types do not match: ",
                       (int)cache_type.ElementType(), " vs ",
                       (int)slice_type.ElementType()));
    }

    auto byte_width = ::litert::GetByteWidth(cache_type.ElementType());
    if (!byte_width.has_value()) {
      return absl::InvalidArgumentError("Unsupported cache element type");
    }
    size_t element_size = byte_width->NumBytes();

    LITERT_ASSIGN_OR_RETURN(size_t cache_num_elements,
                            cache_type.Layout().NumElements());
    if (cache_num_elements == 0) {
      return absl::InvalidArgumentError("Cache layout has 0 elements");
    }

    // Assume hidden_dim is the smaller of the last two dimensions of cache.
    int cache_last_dim = cache_dims[cache_rank - 1];
    int cache_second_last_dim = cache_dims[cache_rank - 2];
    int64_t hidden_dim = std::min(cache_last_dim, cache_second_last_dim);
    int64_t cache_seq = std::max(cache_last_dim, cache_second_last_dim);

    int cache_seq_dim = (cache_dims[cache_rank - 1] == cache_seq)
                            ? cache_rank - 1
                            : cache_rank - 2;

    int slice_seq_dim = -1;
    int slice_hidden_dim = -1;
    int64_t slice_seq = -1;

    // Find dimensions in slice
    if (slice_dims[slice_rank - 1] == hidden_dim) {
      slice_hidden_dim = slice_rank - 1;
      slice_seq_dim = slice_rank - 2;
      slice_seq = slice_dims[slice_seq_dim];
    } else if (slice_dims[slice_rank - 2] == hidden_dim) {
      slice_hidden_dim = slice_rank - 2;
      slice_seq_dim = slice_rank - 1;
      slice_seq = slice_dims[slice_seq_dim];
    }

    if (slice_hidden_dim == -1) {
      return absl::InternalError(
          "Failed to identify hidden dimension in slice");
    }

    if (slice_seq > cache_seq) {
      return absl::InvalidArgumentError(
          absl::StrCat("Slice sequence length (", slice_seq,
                       ") exceeds cache capacity (", cache_seq, ")"));
    }

    int64_t real_start = 0;
    int64_t real_len = slice_seq;
    int64_t start_pos = input_pos_ptr[0];

    if (valid_mask != nullptr) {
      int64_t mask_offset = pos_num_elements - slice_seq;
      if (mask_offset < 0) {
        return absl::InternalError(
            "slice_seq cannot be larger than pos_num_elements");
      }
      const bool* sub_mask = valid_mask + mask_offset;
      int64_t first_true = -1;
      int64_t true_count = 0;
      for (int64_t i = 0; i < slice_seq; ++i) {
        if (sub_mask[i]) {
          if (first_true == -1) {
            first_true = i;
          }
          ++true_count;
        }
      }
      if (first_true != -1) {
        real_start = first_true;
        real_len = true_count;
        // NOTE: The current implementation assumes that valid tokens in
        // `valid_mask` form a contiguous range from `first_true` to
        // `first_true + true_count - 1`. If `valid_mask` contains
        // non-contiguous valid entries (e.g., [T, F, T]), the memcpy operations
        // below will incorrectly copy a contiguous block. This is acceptable
        // for typical prefill inputs where valid tokens are always contiguous.
        start_pos = input_pos_ptr[mask_offset + real_start];
      } else {
        real_len = 0;
      }
    } else {
      if (slice_seq < pos_num_elements) {
        start_pos = input_pos_ptr[pos_num_elements - slice_seq];
      }
    }

    if (real_len == 0) {
      return absl::OkStatus();
    }

    if (start_pos < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("input_pos must be non-negative: ", start_pos));
    }

    int64_t wp = enable_swa ? (start_pos % cache_seq) : start_pos;

    if (wp + real_len > cache_seq) {
      if (!enable_swa) {
        return absl::OutOfRangeError("KV-cache update out of range");
      }
    }

    int64_t cache_outer_size = 1;
    for (int i = 0; i < cache_rank - kSliceOuterRank; ++i) {
      cache_outer_size *= cache_dims[i];
    }
    int64_t slice_outer_size = 1;
    for (int i = 0; i < slice_rank - kSliceOuterRank; ++i) {
      slice_outer_size *= slice_dims[i];
    }
    if (cache_outer_size != slice_outer_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cache and slice outer sizes do not match: ", cache_outer_size,
          " vs ", slice_outer_size));
    }

    size_t expected_cache_size =
        cache_outer_size * cache_seq * hidden_dim * element_size;
    if (cache_bytes < expected_cache_size) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cache buffer size is too small: ", cache_bytes,
                       " vs expected ", expected_cache_size));
    }
    size_t expected_slice_size =
        slice_outer_size * slice_seq * hidden_dim * element_size;
    if (slice_bytes < expected_slice_size) {
      return absl::InvalidArgumentError(
          absl::StrCat("Slice buffer size is too small: ", slice_bytes,
                       " vs expected ", expected_slice_size));
    }

    LITERT_ASSIGN_OR_RETURN(
        auto cache_lock, ::litert::TensorBufferScopedLock::Create(
                             cache, ::litert::TensorBuffer::LockMode::kWrite));

    if (cache_lock.second == nullptr || slice_ptr == nullptr) {
      return absl::InternalError(
          "Failed to lock cache or slice pointer is null");
    }

    uint8_t* cache_ptr = static_cast<uint8_t*>(cache_lock.second);
    const uint8_t* s_ptr_base = static_cast<const uint8_t*>(slice_ptr);

    bool cache_is_transposed = (cache_seq_dim == cache_rank - 1);
    bool slice_is_transposed = (slice_seq_dim == slice_rank - 1);

    for (int64_t o = 0; o < cache_outer_size; ++o) {
      uint8_t* c_ptr = cache_ptr + o * (cache_seq * hidden_dim * element_size);
      const uint8_t* s_ptr =
          s_ptr_base + o * (slice_seq * hidden_dim * element_size);

      if (!cache_is_transposed) {
        if (!slice_is_transposed || slice_seq == 1) {
          // Cache is [..., seq, hidden], Slice is [..., seq, hidden] (or seq=1)
          int64_t chunk1_seq = std::min(real_len, cache_seq - wp);
          int64_t chunk2_seq = real_len - chunk1_seq;
          const uint8_t* s_ptr_offset =
              s_ptr + (real_start * hidden_dim * element_size);
          std::memcpy(c_ptr + (wp * hidden_dim * element_size), s_ptr_offset,
                      chunk1_seq * hidden_dim * element_size);
          if (chunk2_seq > 0) {
            std::memcpy(c_ptr,
                        s_ptr_offset + (chunk1_seq * hidden_dim * element_size),
                        chunk2_seq * hidden_dim * element_size);
          }
        } else {
          // Cache is [..., seq, hidden], Slice is [..., hidden, seq]
          if (element_size == 1) {
            TransposeCopy<uint8_t, false>(c_ptr, s_ptr, real_len, real_start,
                                          wp, cache_seq, hidden_dim, slice_seq);
          } else if (element_size == 2) {
            TransposeCopy<uint16_t, false>(c_ptr, s_ptr, real_len, real_start,
                                           wp, cache_seq, hidden_dim,
                                           slice_seq);
          } else if (element_size == 4) {
            TransposeCopy<uint32_t, false>(c_ptr, s_ptr, real_len, real_start,
                                           wp, cache_seq, hidden_dim,
                                           slice_seq);
          } else {
            // Slow fallback
            for (int64_t s = 0; s < real_len; ++s) {
              int64_t wrapped_pos = (wp + s) % cache_seq;
              int64_t slice_s = real_start + s;
              for (int64_t h = 0; h < hidden_dim; ++h) {
                std::memcpy(
                    c_ptr + (wrapped_pos * hidden_dim + h) * element_size,
                    s_ptr + (h * slice_seq + slice_s) * element_size,
                    element_size);
              }
            }
          }
        }
      } else {
        // Cache is [..., hidden, seq]
        if ((!slice_is_transposed && real_len == 1) || slice_seq == 1) {
          const uint8_t* s_ptr_offset =
              s_ptr + (real_start * hidden_dim * element_size);
#if defined(__ANDROID__) && defined(__ARM_NEON) && defined(__aarch64__)
          if (element_size == 1) {
            int64_t h = 0;
            for (; h <= hidden_dim - 16; h += 16) {
              uint8x16_t v = vld1q_u8(s_ptr_offset + h);
              c_ptr[(h + 0) * cache_seq + wp] = vgetq_lane_u8(v, 0);
              c_ptr[(h + 1) * cache_seq + wp] = vgetq_lane_u8(v, 1);
              c_ptr[(h + 2) * cache_seq + wp] = vgetq_lane_u8(v, 2);
              c_ptr[(h + 3) * cache_seq + wp] = vgetq_lane_u8(v, 3);
              c_ptr[(h + 4) * cache_seq + wp] = vgetq_lane_u8(v, 4);
              c_ptr[(h + 5) * cache_seq + wp] = vgetq_lane_u8(v, 5);
              c_ptr[(h + 6) * cache_seq + wp] = vgetq_lane_u8(v, 6);
              c_ptr[(h + 7) * cache_seq + wp] = vgetq_lane_u8(v, 7);
              c_ptr[(h + 8) * cache_seq + wp] = vgetq_lane_u8(v, 8);
              c_ptr[(h + 9) * cache_seq + wp] = vgetq_lane_u8(v, 9);
              c_ptr[(h + 10) * cache_seq + wp] = vgetq_lane_u8(v, 10);
              c_ptr[(h + 11) * cache_seq + wp] = vgetq_lane_u8(v, 11);
              c_ptr[(h + 12) * cache_seq + wp] = vgetq_lane_u8(v, 12);
              c_ptr[(h + 13) * cache_seq + wp] = vgetq_lane_u8(v, 13);
              c_ptr[(h + 14) * cache_seq + wp] = vgetq_lane_u8(v, 14);
              c_ptr[(h + 15) * cache_seq + wp] = vgetq_lane_u8(v, 15);
            }
            for (; h < hidden_dim; ++h) {
              c_ptr[h * cache_seq + wp] = s_ptr_offset[h];
            }
          } else if (element_size == 2) {
            int64_t h = 0;
            const uint16_t* s_ptr16 =
                reinterpret_cast<const uint16_t*>(s_ptr_offset);
            uint16_t* c_ptr16 = reinterpret_cast<uint16_t*>(c_ptr);
            for (; h <= hidden_dim - 8; h += 8) {
              uint16x8_t v = vld1q_u16(s_ptr16 + h);
              c_ptr16[(h + 0) * cache_seq + wp] = vgetq_lane_u16(v, 0);
              c_ptr16[(h + 1) * cache_seq + wp] = vgetq_lane_u16(v, 1);
              c_ptr16[(h + 2) * cache_seq + wp] = vgetq_lane_u16(v, 2);
              c_ptr16[(h + 3) * cache_seq + wp] = vgetq_lane_u16(v, 3);
              c_ptr16[(h + 4) * cache_seq + wp] = vgetq_lane_u16(v, 4);
              c_ptr16[(h + 5) * cache_seq + wp] = vgetq_lane_u16(v, 5);
              c_ptr16[(h + 6) * cache_seq + wp] = vgetq_lane_u16(v, 6);
              c_ptr16[(h + 7) * cache_seq + wp] = vgetq_lane_u16(v, 7);
            }
            for (; h < hidden_dim; ++h) {
              c_ptr16[h * cache_seq + wp] = s_ptr16[h];
            }
          } else {
#endif
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(c_ptr + (h * cache_seq + wp) * element_size,
                          s_ptr_offset + h * element_size, element_size);
            }
#if defined(__ANDROID__) && defined(__ARM_NEON) && defined(__aarch64__)
          }
#endif
        } else if (slice_is_transposed) {
          // Cache is [..., hidden, seq], Slice is [..., hidden, seq]
          int64_t chunk1_seq = std::min(real_len, cache_seq - wp);
          int64_t chunk2_seq = real_len - chunk1_seq;
          for (int64_t h = 0; h < hidden_dim; ++h) {
            std::memcpy(c_ptr + (h * cache_seq + wp) * element_size,
                        s_ptr + (h * slice_seq + real_start) * element_size,
                        chunk1_seq * element_size);
            if (chunk2_seq > 0) {
              std::memcpy(c_ptr + (h * cache_seq) * element_size,
                          s_ptr + (h * slice_seq + real_start + chunk1_seq) *
                                      element_size,
                          chunk2_seq * element_size);
            }
          }
        } else {
          // Cache is [..., hidden, seq], Slice is [..., seq, hidden]
          if (element_size == 1) {
            TransposeCopy<uint8_t, true>(c_ptr, s_ptr, real_len, real_start, wp,
                                         cache_seq, hidden_dim, slice_seq);
          } else if (element_size == 2) {
            TransposeCopy<uint16_t, true>(c_ptr, s_ptr, real_len, real_start,
                                          wp, cache_seq, hidden_dim, slice_seq);
          } else if (element_size == 4) {
            TransposeCopy<uint32_t, true>(c_ptr, s_ptr, real_len, real_start,
                                          wp, cache_seq, hidden_dim, slice_seq);
          } else {
            // Slow fallback
            for (int64_t s = 0; s < real_len; ++s) {
              int64_t wrapped_pos = (wp + s) % cache_seq;
              int64_t slice_s = real_start + s;
              for (int64_t h = 0; h < hidden_dim; ++h) {
                std::memcpy(
                    c_ptr + (h * cache_seq + wrapped_pos) * element_size,
                    s_ptr + (slice_s * hidden_dim + h) * element_size,
                    element_size);
              }
            }
          }
        }
      }
    }
    return absl::OkStatus();
  };

  std::vector<float> dequantized_slice_scratch;
  auto run_single_update =
      [&](::litert::TensorBuffer& cache, const ::litert::TensorBuffer& slice,
          const RankedTensorType& cache_type,
          const RankedTensorType& slice_type, absl::string_view cache_name,
          absl::string_view slice_name) -> absl::Status {
    LITERT_ASSIGN_OR_RETURN(auto slice_lock,
                            ::litert::TensorBufferScopedLock::Create(
                                const_cast<::litert::TensorBuffer&>(slice),
                                ::litert::TensorBuffer::LockMode::kRead));
    if (slice_lock.second == nullptr) {
      return absl::InternalError(
          absl::StrCat("Failed to lock slice buffer for ", slice_name));
    }

    LITERT_ASSIGN_OR_RETURN(size_t slice_bytes, slice.Size());

    if (cache_type.ElementType() != slice_type.ElementType()) {
      if (cache_type.ElementType() == ::litert::ElementType::Float32 &&
          slice_type.ElementType() == ::litert::ElementType::Int16) {
        // Dequantize Int16 to Float32
        LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                                slice_type.Layout().NumElements());
        dequantized_slice_scratch.resize(num_elements);

        float scale = 1.0f;
        int64_t zero_point = 0;
        std::string s_name = std::string(slice_name);
        if (quant_params.contains(s_name)) {
          scale = quant_params.at(s_name).scale;
          zero_point = quant_params.at(s_name).zero_point;
        }

        const int16_t* src = static_cast<const int16_t*>(slice_lock.second);
        for (size_t i = 0; i < num_elements; ++i) {
          dequantized_slice_scratch[i] =
              (static_cast<float>(src[i]) - zero_point) * scale;
        }

        RankedTensorType dequantized_slice_type(
            ::litert::ElementType::Float32,
            ::litert::Layout(
                static_cast<const LiteRtLayout&>(slice_type.Layout())));
        size_t dequantized_slice_bytes = num_elements * sizeof(float);

        auto status = perform_update(
            cache, dequantized_slice_type, dequantized_slice_scratch.data(),
            dequantized_slice_bytes, cache_name, slice_name);
        if (!status.ok()) {
          return absl::Status(
              status.code(),
              absl::StrCat("Failed updating ", cache_name, " with ", slice_name,
                           " (dequantized): ", status.message()));
        }
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported type mismatch for ", cache_name, " vs ",
                         slice_name, ": ", (int)cache_type.ElementType(),
                         " vs ", (int)slice_type.ElementType()));
      }
    } else {
      // Direct update
      auto status = perform_update(cache, slice_type, slice_lock.second,
                                   slice_bytes, cache_name, slice_name);
      if (!status.ok()) {
        return absl::Status(
            status.code(),
            absl::StrCat("Failed updating ", cache_name, " with ", slice_name,
                         ": ", status.message()));
      }
    }
    return absl::OkStatus();
  };

  auto perform_copy = [](::litert::TensorBuffer& dest,
                         const ::litert::TensorBuffer& src) -> absl::Status {
    LITERT_ASSIGN_OR_RETURN(size_t dest_bytes, dest.Size());
    LITERT_ASSIGN_OR_RETURN(size_t src_bytes, src.Size());
    if (dest_bytes != src_bytes) {
      return absl::InvalidArgumentError("Buffer size mismatch for copy");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto dest_lock, ::litert::TensorBufferScopedLock::Create(
                            dest, ::litert::TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto src_lock,
                            ::litert::TensorBufferScopedLock::Create(
                                src, ::litert::TensorBuffer::LockMode::kRead));
    std::memcpy(dest_lock.second, src_lock.second, dest_bytes);
    return absl::OkStatus();
  };

  for (const auto& [name, buffer] : in_buffers) {
    if (name.starts_with("kv_cache_k_")) {
      int layer_id = std::stoi(std::string(name).substr(11));
      char v_cache_name[32];
      snprintf(v_cache_name, sizeof(v_cache_name), "kv_cache_v_%d", layer_id);
      char k_slice_name[32];
      snprintf(k_slice_name, sizeof(k_slice_name), "kv_slice_k_%d", layer_id);
      char v_slice_name[32];
      snprintf(v_slice_name, sizeof(v_slice_name), "kv_slice_v_%d", layer_id);

      if (!in_buffers.contains(v_cache_name) ||
          !in_buffers.contains(k_slice_name) ||
          !in_buffers.contains(v_slice_name)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Missing matching K/V cache/slice buffers for layer ", layer_id));
      }

      auto& in_k_cache = in_buffers.at(name);
      auto& in_v_cache = in_buffers.at(v_cache_name);
      const auto& k_slice = in_buffers.at(k_slice_name);
      const auto& v_slice = in_buffers.at(v_slice_name);

      LITERT_ASSIGN_OR_RETURN(auto k_cache_type, in_k_cache.TensorType());
      LITERT_ASSIGN_OR_RETURN(auto v_cache_type, in_v_cache.TensorType());
      LITERT_ASSIGN_OR_RETURN(auto k_slice_type, k_slice.TensorType());
      LITERT_ASSIGN_OR_RETURN(auto v_slice_type, v_slice.TensorType());

      LITERT_RETURN_IF_ERROR(run_single_update(
          in_k_cache, k_slice, k_cache_type, k_slice_type, name, k_slice_name));
      LITERT_RETURN_IF_ERROR(run_single_update(in_v_cache, v_slice,
                                               v_cache_type, v_slice_type,
                                               v_cache_name, v_slice_name));

      if (out_buffers.contains(name)) {
        auto& out_k_cache = out_buffers.at(name);
        if (in_k_cache.Get() != out_k_cache.Get()) {
          LITERT_RETURN_IF_ERROR(run_single_update(out_k_cache, k_slice,
                                                   k_cache_type, k_slice_type,
                                                   name, k_slice_name));
        }
      }
      if (out_buffers.contains(v_cache_name)) {
        auto& out_v_cache = out_buffers.at(v_cache_name);
        if (in_v_cache.Get() != out_v_cache.Get()) {
          LITERT_RETURN_IF_ERROR(run_single_update(out_v_cache, v_slice,
                                                   v_cache_type, v_slice_type,
                                                   v_cache_name, v_slice_name));
        }
      }
    }
  }

  // Update C caches if exists.
  for (const auto& [name, buffer] : in_buffers) {
    if (name.starts_with("kv_cache_c_")) {
      int layer_id = std::stoi(std::string(name).substr(11));
      char c_slice_name[32];
      snprintf(c_slice_name, sizeof(c_slice_name), "kv_slice_c_%d", layer_id);

      if (!in_buffers.contains(c_slice_name)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Missing matching C slice buffer for layer ", layer_id));
      }

      auto& in_c_cache = in_buffers.at(name);
      const auto& c_slice = in_buffers.at(c_slice_name);

      LITERT_RETURN_IF_ERROR(perform_copy(in_c_cache, c_slice));

      if (out_buffers.contains(name)) {
        auto& out_c_cache = out_buffers.at(name);
        if (in_c_cache.Get() != out_c_cache.Get()) {
          LITERT_RETURN_IF_ERROR(perform_copy(out_c_cache, c_slice));
        }
      }
    }
  }

  return absl::OkStatus();
}

// -----------------------------------------------------------------------------
// NpuKVCache Implementation
// -----------------------------------------------------------------------------

absl::StatusOr<NpuKVCache> NpuKVCache::CreateForTest(
    KVCacheUpdateMethod method, const ::litert::CompiledModel* compiled_model,
    InferenceContext cache_update_context,
    absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params,
    bool has_sliding_window_attention, int64_t kv_cache_init_value) {
  if (method == KVCacheUpdateMethod::kModel && compiled_model == nullptr) {
    return absl::InvalidArgumentError(
        "Compiled model is required when using kModel cache update method.");
  }
  return NpuKVCache(method, compiled_model, std::move(cache_update_context),
                    std::move(kv_quant_params), has_sliding_window_attention,
                    kv_cache_init_value);
}

absl::StatusOr<NpuKVCache> NpuKVCache::Create(
    KVCacheUpdateMethod method,
    const ::litert::CompiledModel* npu_auxiliary_compiled_model,
    absl::string_view prefill_signature, absl::string_view decode_signature,
    absl::string_view verify_signature,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        prefill_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        decode_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        verify_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params,
    bool has_sliding_window_attention, int64_t kv_cache_init_value) {
  RET_CHECK(npu_auxiliary_compiled_model != nullptr)
      << "Auxiliary compiled model cannot be null for NpuKVCache";

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

  for (const auto& [key, value] : input_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    LITERT_ASSIGN_OR_RETURN(prefill_output_buffers[key], value.Duplicate());
    LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
    LITERT_ASSIGN_OR_RETURN(decode_output_buffers[key], value.Duplicate());
  }

  for (const auto& [key, value] : prefill_output_kv_cache_slice_buffers) {
    LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
  }
  for (const auto& [key, value] : decode_output_kv_cache_slice_buffers) {
    LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
  }

  if (!prefill_signature.empty() &&
      npu_auxiliary_compiled_model->FindSignature(prefill_signature)) {
    LITERT_ASSIGN_OR_RETURN(
        prefill_input_buffers[CacheUpdateSignatures::kInputPos],
        npu_auxiliary_compiled_model->CreateInputBuffer(
            prefill_signature, CacheUpdateSignatures::kInputPos));
    prefill_input_buffers[CacheUpdateSignatures::kInputPos].Clear();

    LITERT_ASSIGN_OR_RETURN(
        auto prefill_cache_input_names,
        npu_auxiliary_compiled_model->GetSignatureInputNames(
            prefill_signature));
    if (absl::c_find(prefill_cache_input_names,
                     CacheUpdateSignatures::kInputValidMask) !=
        prefill_cache_input_names.end()) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_input_buffers[CacheUpdateSignatures::kInputValidMask],
          npu_auxiliary_compiled_model->CreateInputBuffer(
              prefill_signature, CacheUpdateSignatures::kInputValidMask));
      prefill_input_buffers[CacheUpdateSignatures::kInputValidMask].Clear();
    }
  }

  if (!decode_signature.empty() &&
      npu_auxiliary_compiled_model->FindSignature(decode_signature)) {
    LITERT_ASSIGN_OR_RETURN(
        decode_input_buffers[CacheUpdateSignatures::kInputPos],
        npu_auxiliary_compiled_model->CreateInputBuffer(
            decode_signature, CacheUpdateSignatures::kInputPos));
    decode_input_buffers[CacheUpdateSignatures::kInputPos].Clear();

    LITERT_ASSIGN_OR_RETURN(
        auto decode_cache_input_names,
        npu_auxiliary_compiled_model->GetSignatureInputNames(decode_signature));
    if (absl::c_find(decode_cache_input_names,
                     CacheUpdateSignatures::kInputValidMask) !=
        decode_cache_input_names.end()) {
      LITERT_ASSIGN_OR_RETURN(
          decode_input_buffers[CacheUpdateSignatures::kInputValidMask],
          npu_auxiliary_compiled_model->CreateInputBuffer(
              decode_signature, CacheUpdateSignatures::kInputValidMask));
      decode_input_buffers[CacheUpdateSignatures::kInputValidMask].Clear();
    }
  }

  if (!verify_signature.empty() &&
      npu_auxiliary_compiled_model->FindSignature(verify_signature)) {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
      LITERT_ASSIGN_OR_RETURN(verify_output_buffers[key], value.Duplicate());
    }
    for (const auto& [key, value] : verify_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
    }
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[CacheUpdateSignatures::kInputPos],
        npu_auxiliary_compiled_model->CreateInputBuffer(
            verify_signature, CacheUpdateSignatures::kInputPos));
    verify_input_buffers[CacheUpdateSignatures::kInputPos].Clear();

    LITERT_ASSIGN_OR_RETURN(
        auto verify_cache_input_names,
        npu_auxiliary_compiled_model->GetSignatureInputNames(verify_signature));
    if (absl::c_find(verify_cache_input_names,
                     CacheUpdateSignatures::kInputValidMask) !=
        verify_cache_input_names.end()) {
      LITERT_ASSIGN_OR_RETURN(
          verify_input_buffers[CacheUpdateSignatures::kInputValidMask],
          npu_auxiliary_compiled_model->CreateInputBuffer(
              verify_signature, CacheUpdateSignatures::kInputValidMask));
      verify_input_buffers[CacheUpdateSignatures::kInputValidMask].Clear();
    }
  }

  InferenceContext cache_update_context(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
  return NpuKVCache(method, npu_auxiliary_compiled_model,
                    std::move(cache_update_context), std::move(kv_quant_params),
                    has_sliding_window_attention, kv_cache_init_value);
}

absl::Status NpuKVCache::SetPrefillPositions(
    absl::Span<const int32_t> seq_positions) {
  if (cache_update_context_.prefill_input_buffers.contains(
          CacheUpdateSignatures::kInputPos)) {
    auto& pos_buf =
        cache_update_context_
            .prefill_input_buffers[CacheUpdateSignatures::kInputPos];
    LITERT_ASSIGN_OR_RETURN(
        auto pos_lock, ::litert::TensorBufferScopedLock::Create(
                           pos_buf, ::litert::TensorBuffer::LockMode::kWrite));
    auto* pos_ptr = static_cast<int32_t*>(pos_lock.second);
    LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, pos_buf.TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                            tensor_type.Layout().NumElements());
    const size_t copy_size = std::min(num_elements, seq_positions.size());
    std::memcpy(pos_ptr, seq_positions.data(), copy_size * sizeof(int32_t));
    if (num_elements > copy_size) {
      std::memset(pos_ptr + copy_size, 0,
                  (num_elements - copy_size) * sizeof(int32_t));
    }
  }

  if (cache_update_context_.prefill_input_buffers.contains(
          CacheUpdateSignatures::kInputValidMask)) {
    auto& mask_buf =
        cache_update_context_
            .prefill_input_buffers[CacheUpdateSignatures::kInputValidMask];
    LITERT_ASSIGN_OR_RETURN(
        auto mask_lock,
        ::litert::TensorBufferScopedLock::Create(
            mask_buf, ::litert::TensorBuffer::LockMode::kWrite));
    auto* mask_ptr = static_cast<bool*>(mask_lock.second);
    LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                            mask_buf.TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                            tensor_type.Layout().NumElements());
    for (size_t i = 0; i < num_elements; ++i) {
      mask_ptr[i] = (i < seq_positions.size());
    }
  }
  return absl::OkStatus();
}

absl::Status NpuKVCache::SetDecodePosition(int32_t step) {
  if (cache_update_context_.decode_input_buffers.contains(
          CacheUpdateSignatures::kInputPos)) {
    LITERT_RETURN_IF_ERROR(SetFirstElement(
        cache_update_context_
            .decode_input_buffers[CacheUpdateSignatures::kInputPos],
        step));
  }
  if (cache_update_context_.decode_input_buffers.contains(
          CacheUpdateSignatures::kInputValidMask)) {
    LITERT_RETURN_IF_ERROR(SetFirstElement(
        cache_update_context_
            .decode_input_buffers[CacheUpdateSignatures::kInputValidMask],
        true));
  }
  return absl::OkStatus();
}

absl::Status NpuKVCache::RunPrefill(absl::string_view signature) {
  if (method_ == KVCacheUpdateMethod::kWH) {
    return HWKVCacheUpdate(cache_update_context_.prefill_input_buffers,
                           cache_update_context_.prefill_output_buffers,
                           kv_quant_params_, has_sliding_window_attention_);
  }
  absl::string_view sig =
      signature.empty() ? kPrefillCacheUpdateBase : signature;
  auto res =
      compiled_model_->Run(sig, cache_update_context_.prefill_input_buffers,
                           cache_update_context_.prefill_output_buffers);
  RET_CHECK(res) << "Failed to run cache update model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuKVCache::RunDecode(absl::string_view signature) {
  if (method_ == KVCacheUpdateMethod::kWH) {
    return HWKVCacheUpdate(cache_update_context_.decode_input_buffers,
                           cache_update_context_.decode_output_buffers,
                           kv_quant_params_, has_sliding_window_attention_);
  }
  absl::string_view sig =
      signature.empty() ? CacheUpdateSignatures::kDecodeCacheUpdate : signature;
  auto res =
      compiled_model_->Run(sig, cache_update_context_.decode_input_buffers,
                           cache_update_context_.decode_output_buffers);
  RET_CHECK(res) << "Failed to run cache update model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuKVCache::SetVerifyPos(int start_step) {
  if (cache_update_context_.verify_input_buffers.contains(
          CacheUpdateSignatures::kInputPos)) {
    auto& pos_buf = cache_update_context_
                        .verify_input_buffers[CacheUpdateSignatures::kInputPos];
    LITERT_ASSIGN_OR_RETURN(
        auto pos_lock, ::litert::TensorBufferScopedLock::Create(
                           pos_buf, ::litert::TensorBuffer::LockMode::kWrite));
    auto* pos_ptr = static_cast<int32_t*>(pos_lock.second);
    LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, pos_buf.TensorType());
    int tensor_size = tensor_type.Layout().Dimensions()[0];
    for (int i = 0; i < tensor_size; ++i) {
      pos_ptr[i] = start_step + i;
    }
  }
  return absl::OkStatus();
}

absl::Status NpuKVCache::CommitVerifiedKVCache(int start_step,
                                               absl::string_view signature) {
  LITERT_RETURN_IF_ERROR(SetVerifyPos(start_step));
  if (method_ == KVCacheUpdateMethod::kWH) {
    return HWKVCacheUpdate(cache_update_context_.verify_input_buffers,
                           cache_update_context_.verify_output_buffers,
                           kv_quant_params_, has_sliding_window_attention_);
  }
  absl::string_view sig =
      signature.empty() ? CacheUpdateSignatures::kVerifyCacheUpdate : signature;
  LITERT_RETURN_IF_ERROR(
      compiled_model_->Run(sig, cache_update_context_.verify_input_buffers,
                           cache_update_context_.verify_output_buffers));
  return absl::OkStatus();
}

absl::Status NpuKVCache::CopySingleKVCacheBuffer(
    const ::litert::TensorBuffer& src, ::litert::TensorBuffer& dst,
    int active_seq_len, int64_t kv_cache_init_value) {
  if (active_seq_len <= 0) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(auto src_type, src.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto dst_type, dst.TensorType());

  auto src_dims = src_type.Layout().Dimensions();
  auto dst_dims = dst_type.Layout().Dimensions();

  int seq_dim_idx = -1;
  for (int i = 0; i < src_dims.size(); ++i) {
    if (src_dims[i] != dst_dims[i]) {
      seq_dim_idx = i;
      break;
    }
  }

  if (seq_dim_idx == -1) {
    return absl::OkStatus();
  }

  for (int i = 0; i < src_dims.size(); ++i) {
    if (i != seq_dim_idx && src_dims[i] != dst_dims[i]) {
      return absl::InternalError(
          absl::StrCat("KV cache buffers differ in non-sequence dimension ", i,
                       ": src=", src_dims[i], ", dst=", dst_dims[i]));
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto src_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              const_cast<::litert::TensorBuffer&>(src),
                              ::litert::TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto dst_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              dst, ::litert::TensorBuffer::LockMode::kWrite));

  auto byte_width_opt = ::litert::GetByteWidth(src_type.ElementType());
  if (!byte_width_opt.has_value()) {
    return absl::InternalError("Unsupported element type in KV cache.");
  }
  const auto& byte_width = *byte_width_opt;

  const char* src_ptr = static_cast<const char*>(src_lock.second);
  char* dst_ptr = static_cast<char*>(dst_lock.second);

  auto clear_range = [&](size_t element_offset, size_t num_elements) {
    size_t byte_offset = byte_width.NumBytes(element_offset);
    size_t byte_size = byte_width.NumBytes(num_elements);
    void* target_ptr = dst_ptr + byte_offset;
    auto element_type = dst_type.ElementType();
    if (element_type == ::litert::ElementType::Int16) {
      int16_t* ptr = static_cast<int16_t*>(target_ptr);
      std::fill(ptr, ptr + num_elements,
                static_cast<int16_t>(kv_cache_init_value));
    } else if (element_type == ::litert::ElementType::UInt16) {
      uint16_t* ptr = static_cast<uint16_t*>(target_ptr);
      std::fill(ptr, ptr + num_elements,
                static_cast<uint16_t>(kv_cache_init_value));
    } else if (element_type == ::litert::ElementType::Int8) {
      int8_t* ptr = static_cast<int8_t*>(target_ptr);
      std::fill(ptr, ptr + num_elements,
                static_cast<int8_t>(kv_cache_init_value));
    } else if (element_type == ::litert::ElementType::UInt8) {
      uint8_t* ptr = static_cast<uint8_t*>(target_ptr);
      std::fill(ptr, ptr + num_elements,
                static_cast<uint8_t>(kv_cache_init_value));
    } else {
      std::memset(target_ptr, 0, byte_size);
    }
  };

  size_t outer_count = 1;
  for (int i = 0; i < seq_dim_idx; ++i) {
    outer_count *= src_dims[i];
  }
  size_t inner_count = 1;
  for (size_t i = seq_dim_idx + 1; i < src_dims.size(); ++i) {
    inner_count *= src_dims[i];
  }

  size_t S_src = src_dims[seq_dim_idx];
  size_t S_dst = dst_dims[seq_dim_idx];
  size_t valid_seq_len = std::min(static_cast<size_t>(active_seq_len), S_src);
  size_t copy_elements_per_seq = valid_seq_len * inner_count;
  size_t bytes_to_copy = byte_width.NumBytes(copy_elements_per_seq);

  for (int64_t outer = static_cast<int64_t>(outer_count) - 1; outer >= 0;
       --outer) {
    size_t src_element_offset = outer * S_src * inner_count;
    size_t dst_element_offset = outer * S_dst * inner_count;
    size_t src_byte_offset = byte_width.NumBytes(src_element_offset);
    size_t dst_byte_offset = byte_width.NumBytes(dst_element_offset);
    std::memmove(dst_ptr + dst_byte_offset, src_ptr + src_byte_offset,
                 bytes_to_copy);
  }

  for (size_t outer = 0; outer < outer_count; ++outer) {
    size_t padding_element_offset =
        (outer * S_dst + valid_seq_len) * inner_count;
    size_t padding_num_elements = (S_dst - valid_seq_len) * inner_count;
    clear_range(padding_element_offset, padding_num_elements);
  }

  return absl::OkStatus();
}

absl::Status NpuKVCache::CopyKVCache(
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        src_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& dst_buffers,
    int active_seq_len) {
  if (active_seq_len <= 0) {
    return absl::OkStatus();
  }

  for (const auto& [name, src_buf] : src_buffers) {
    if (name.starts_with(kKvCacheKRootName) ||
        name.starts_with(kKvCacheVRootName) ||
        name.starts_with(kKvCacheCRootName)) {
      if (dst_buffers.contains(name)) {
        LITERT_RETURN_IF_ERROR(CopySingleKVCacheBuffer(
            src_buf, dst_buffers[name], active_seq_len, kv_cache_init_value_));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status NpuKVCache::UpdateKVCacheBuffers(
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        prefill_output_kv_cache_slice_buffers,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        decode_output_kv_cache_slice_buffers,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        verify_output_kv_cache_slice_buffers) {
  for (const auto& [name, buf] : input_kv_cache_buffers) {
    if (cache_update_context_.prefill_input_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(cache_update_context_.prefill_input_buffers[name],
                              buf.Duplicate());
    }
    if (cache_update_context_.prefill_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(
          cache_update_context_.prefill_output_buffers[name], buf.Duplicate());
    }
    if (cache_update_context_.decode_input_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(cache_update_context_.decode_input_buffers[name],
                              buf.Duplicate());
    }
    if (cache_update_context_.decode_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(cache_update_context_.decode_output_buffers[name],
                              buf.Duplicate());
    }
    if (cache_update_context_.verify_input_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(cache_update_context_.verify_input_buffers[name],
                              buf.Duplicate());
    }
    if (cache_update_context_.verify_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(cache_update_context_.verify_output_buffers[name],
                              buf.Duplicate());
    }
  }
  for (const auto& [name, buf] : prefill_output_kv_cache_slice_buffers) {
    if (cache_update_context_.prefill_input_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(cache_update_context_.prefill_input_buffers[name],
                              buf.Duplicate());
    }
  }
  for (const auto& [name, buf] : decode_output_kv_cache_slice_buffers) {
    if (cache_update_context_.decode_input_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(cache_update_context_.decode_input_buffers[name],
                              buf.Duplicate());
    }
  }
  for (const auto& [name, buf] : verify_output_kv_cache_slice_buffers) {
    if (cache_update_context_.verify_input_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(cache_update_context_.verify_input_buffers[name],
                              buf.Duplicate());
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
