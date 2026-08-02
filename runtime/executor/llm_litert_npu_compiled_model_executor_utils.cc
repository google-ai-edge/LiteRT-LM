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

#include "runtime/executor/llm_litert_npu_compiled_model_executor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/prefetch.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/util/status_macros.h"  // IWYU pragma: keep NOLINT
#include "tflite/types/half.h"  // from @litert

#if defined(__ANDROID__) && defined(__ARM_NEON)
#include <arm_neon.h>

#include <limits>  // IWYU pragma: keep
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>  // SSE2 NOLINT

#include <limits>  // IWYU pragma: keep NOLINT
#endif

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

static constexpr int kSliceOuterRank = 2;
#if defined(__ANDROID__) && defined(__ARM_NEON)
int FindMaxIndexFloatNeon(const float* data, int size) {
  if (size <= 0) return 0;

  float32x4_t max_v4 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  uint32x4_t max_i4 = vdupq_n_u32(0);
  uint32x4_t curr_i4 = {0, 1, 2, 3};
  uint32x4_t step4 = vdupq_n_u32(4);

  int i = 0;
  for (; i <= size - 4; i += 4) {
    float32x4_t v4 = vld1q_f32(data + i);
    uint32x4_t mask = vcgtq_f32(v4, max_v4);
    // Update max values and their corresponding indices in one pass.
    max_v4 = vmaxq_f32(v4, max_v4);
    max_i4 = vbslq_u32(mask, curr_i4, max_i4);
    curr_i4 = vaddq_u32(curr_i4, step4);
  }

  // Reduce the 4-lane registers to a single max value and index.
  float max_vals[4];
  uint32_t max_idxs[4];
  vst1q_f32(max_vals, max_v4);
  vst1q_u32(max_idxs, max_i4);

  float max_v = max_vals[0];
  int max_idx = max_idxs[0];
  for (int j = 1; j < 4; ++j) {
    if (max_vals[j] > max_v) {
      max_v = max_vals[j];
      max_idx = max_idxs[j];
    } else if (max_vals[j] == max_v) {
      max_idx = std::min(max_idx, (int)max_idxs[j]);
    }
  }

  // Handle remaining elements.
  for (; i < size; ++i) {
    if (data[i] > max_v) {
      max_v = data[i];
      max_idx = i;
    }
  }
  return max_idx;
}

int FindMaxIndexInt16Neon(const int16_t* data, int size) {
  if (size <= 0) return 0;
  int16x8_t max_v8 = vdupq_n_s16(std::numeric_limits<int16_t>::lowest());
  int i = 0;
  for (; i <= size - 8; i += 8) {
    max_v8 = vmaxq_s16(max_v8, vld1q_s16(data + i));
  }
  int16_t max_vals_arr[8];
  vst1q_s16(max_vals_arr, max_v8);
  int16_t max_v = max_vals_arr[0];
  for (int j = 1; j < 8; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  int16x8_t target = vdupq_n_s16(max_v);
  for (i = 0; i <= size - 8; i += 8) {
    uint16x8_t cmp = vceqq_s16(vld1q_s16(data + i), target);
    uint16_t mask[8];
    vst1q_u16(mask, cmp);
    if (mask[0] || mask[1] || mask[2] || mask[3] || mask[4] || mask[5] ||
        mask[6] || mask[7]) {
      for (int j = 0; j < 8; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}

int FindMaxIndexInt8Neon(const int8_t* data, int size) {
  if (size <= 0) return 0;
  int8x16_t max_v16 = vdupq_n_s8(std::numeric_limits<int8_t>::lowest());
  int i = 0;
  for (; i <= size - 16; i += 16) {
    max_v16 = vmaxq_s8(max_v16, vld1q_s8(data + i));
  }
  int8_t max_vals_arr[16];
  vst1q_s8(max_vals_arr, max_v16);
  int8_t max_v = max_vals_arr[0];
  for (int j = 1; j < 16; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  int8x16_t target = vdupq_n_s8(max_v);
  for (i = 0; i <= size - 16; i += 16) {
    uint8x16_t cmp = vceqq_s8(vld1q_s8(data + i), target);
    uint8_t mask[16];
    vst1q_u8(mask, cmp);
    bool match = false;
    for (int j = 0; j < 16; ++j) {
      if (mask[j]) {
        match = true;
        break;
      }
    }
    if (match) {
      for (int j = 0; j < 16; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
int FindMaxIndexSse2Float(const float* data, int size) {
  if (size <= 0) return 0;
  __m128 max_v4 = _mm_set1_ps(-std::numeric_limits<float>::infinity());
  int i = 0;
  for (; i <= size - 4; i += 4) {
    max_v4 = _mm_max_ps(max_v4, _mm_loadu_ps(data + i));
  }
  // Horizontal max reduction.
  __m128 shuf = _mm_shuffle_ps(max_v4, max_v4, _MM_SHUFFLE(2, 3, 0, 1));
  max_v4 = _mm_max_ps(max_v4, shuf);
  shuf = _mm_shuffle_ps(max_v4, max_v4, _MM_SHUFFLE(1, 0, 3, 2));
  max_v4 = _mm_max_ps(max_v4, shuf);
  float max_v;
  _mm_store_ss(&max_v, max_v4);
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  // Second pass: find first index matching max_v.
  __m128 target = _mm_set1_ps(max_v);
  for (i = 0; i <= size - 4; i += 4) {
    int mask = _mm_movemask_ps(_mm_cmpeq_ps(_mm_loadu_ps(data + i), target));
    if (mask) {
      for (int j = 0; j < 4; ++j) {
        if (mask & (1 << j)) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}

int FindMaxIndexSse2Int16(const int16_t* data, int size) {
  // NOLINTBEGIN
  if (size <= 0) return 0;
  __m128i max_v8 = _mm_set1_epi16(std::numeric_limits<int16_t>::lowest());
  int i = 0;
  for (; i <= size - 8; i += 8) {
    max_v8 = _mm_max_epi16(
        max_v8, _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i)));
  }
  // Horizontal max reduction.
  __m128i shuf =
      _mm_shufflehi_epi16(_mm_shufflelo_epi16(max_v8, _MM_SHUFFLE(1, 0, 3, 2)),
                          _MM_SHUFFLE(1, 0, 3, 2));
  max_v8 = _mm_max_epi16(max_v8, shuf);
  shuf = _mm_shuffle_epi32(max_v8, _MM_SHUFFLE(1, 0, 3, 2));
  max_v8 = _mm_max_epi16(max_v8, shuf);
  shuf = _mm_shufflelo_epi16(max_v8, _MM_SHUFFLE(0, 1, 2, 3));
  max_v8 = _mm_max_epi16(max_v8, shuf);
  int16_t max_v = static_cast<int16_t>(_mm_extract_epi16(max_v8, 0));
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  // Second pass: find first index matching max_v.
  __m128i target = _mm_set1_epi16(max_v);
  for (i = 0; i <= size - 8; i += 8) {
    __m128i cmp = _mm_cmpeq_epi16(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i)), target);
    int mask = _mm_movemask_epi8(cmp);
    if (mask) {
      // Each int16 produces 2 bits in the mask; check every other bit.
      for (int j = 0; j < 8; ++j) {
        if (mask & (1 << (j * 2))) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}

int FindMaxIndexSse2Int8(const int8_t* data, int size) {
  if (size <= 0) return 0;
  // SSE2 only has _mm_max_epu8 (unsigned). XOR with 0x80 to convert signed
  // comparison to unsigned: signed_max(a,b) == unsigned_max(a^0x80, b^0x80).
  __m128i bias = _mm_set1_epi8(static_cast<char>(0x80));
  __m128i max_v16 = _mm_set1_epi8(0);  // lowest unsigned after bias
  int i = 0;
  for (; i <= size - 16; i += 16) {
    __m128i vals = _mm_xor_si128(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i)), bias);
    max_v16 = _mm_max_epu8(max_v16, vals);
  }
  // Horizontal max reduction (16 bytes → 8 → 4 → 2 → 1).
  __m128i shuf = _mm_shuffle_epi32(max_v16, _MM_SHUFFLE(1, 0, 3, 2));
  max_v16 = _mm_max_epu8(max_v16, shuf);
  shuf = _mm_shuffle_epi32(max_v16, _MM_SHUFFLE(0, 0, 0, 1));
  max_v16 = _mm_max_epu8(max_v16, shuf);
  shuf = _mm_shufflelo_epi16(max_v16, _MM_SHUFFLE(0, 0, 0, 1));
  max_v16 = _mm_max_epu8(max_v16, shuf);
  shuf = _mm_srli_epi16(max_v16, 8);
  max_v16 = _mm_max_epu8(max_v16, shuf);
  // Extract lowest byte and convert back to signed.
  uint8_t max_unsigned =
      static_cast<uint8_t>(_mm_extract_epi16(max_v16, 0) & 0xFF);
  int8_t max_v = static_cast<int8_t>(max_unsigned ^ 0x80);
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  // Second pass: find first index matching max_v.
  __m128i target = _mm_set1_epi8(max_v);
  for (i = 0; i <= size - 16; i += 16) {
    __m128i cmp = _mm_cmpeq_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i)), target);
    int mask = _mm_movemask_epi8(cmp);
    if (mask) {
      for (int j = 0; j < 16; ++j) {
        if (mask & (1 << j)) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
  // NOLINTEND
}
#endif

absl::StatusOr<int> ApplyGreedySampling(::litert::TensorBuffer& decoded_logits,
                                        bool use_neon_sampling) {
  LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType logits_tensor_type,
                          decoded_logits.TensorType());
  if (logits_tensor_type.ElementType() == ::litert::ElementType::Float32) {
    return FindMaxIndex<float>(decoded_logits, use_neon_sampling);
  } else if (logits_tensor_type.ElementType() == ::litert::ElementType::Int16) {
    return FindMaxIndex<int16_t>(decoded_logits, use_neon_sampling);
  } else if (logits_tensor_type.ElementType() == ::litert::ElementType::Int8) {
    return FindMaxIndex<int8_t>(decoded_logits, use_neon_sampling);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported tensor element type for greedy sampling: ",
                     logits_tensor_type.ElementType()));
  }
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

    // static constexpr int kSliceOuterRank = 2;
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
          for (int64_t s = 0; s < real_len; ++s) {
            int64_t wrapped_pos = (wp + s) % cache_seq;
            int64_t slice_s = real_start + s;
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(c_ptr + (wrapped_pos * hidden_dim + h) * element_size,
                          s_ptr + (h * slice_seq + slice_s) * element_size,
                          element_size);
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
          for (int64_t s = 0; s < real_len; ++s) {
            int64_t wrapped_pos = (wp + s) % cache_seq;
            int64_t slice_s = real_start + s;
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(c_ptr + (h * cache_seq + wrapped_pos) * element_size,
                          s_ptr + (slice_s * hidden_dim + h) * element_size,
                          element_size);
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

namespace {

// Internal template for filling masks.
// T: element type (int8_t or int16_t).
// valid_val: value for "unmasked" (127 for i8, 0 for i16).
// masked_val: value for "masked" (-128 for i8, -32767 for i16).
template <typename T>
void FillMasksInternal(T* mask_local, T* mask_global, int64_t seq_q,
                       int64_t seq_k, int32_t time_step,
                       const int32_t* input_tokens, int64_t input_tokens_size,
                       const bool* valid_mask, int64_t valid_mask_size,
                       T valid_val, T masked_val) {
  // Detection logic for capacity and batch_size.
  int64_t kv_cache_capacity = seq_k;
  bool has_batch_suffix = false;
  if (seq_k > seq_q) {
    int64_t candidate_cap = seq_k - seq_q;
    // We assume capacity is a multiple of 64 for all models.
    // If it's not a multiple of 64, it might still be a regular mask
    // if seq_q is large (prefill) or if it's very close to a multiple of 64.
    if (candidate_cap % 64 == 0 || seq_q > 8) {
      kv_cache_capacity = candidate_cap;
      has_batch_suffix = true;
    } else {
      // Fallback: if seq_k is just slightly above a multiple of 64,
      // it's likely capacity + batch (e.g. 4097 for decode).
      int64_t nearest_64 = (seq_k / 64) * 64;
      if (nearest_64 > 0 && nearest_64 < seq_k && (seq_k - nearest_64) <= 8) {
        kv_cache_capacity = nearest_64;
        has_batch_suffix = true;
      }
    }
  }
  const int64_t batch_size = has_batch_suffix ? seq_q : 0;

  // Initialize with masked value (performance: memset if i8).
  if (sizeof(T) == 1) {
    if (mask_local) std::memset(mask_local, (int)masked_val, seq_q * seq_k);
    if (mask_global) std::memset(mask_global, (int)masked_val, seq_q * seq_k);
  } else {
    for (int64_t i = 0; i < seq_q * seq_k; ++i) {
      if (mask_local) mask_local[i] = masked_val;
      if (mask_global) mask_global[i] = masked_val;
    }
  }

  // Fill valid regions.
  for (int64_t q = 0; q < seq_q; ++q) {
    // effective_pos is the position of the query token in the sequence.
    const int64_t effective_pos = time_step + q;
    T* local_row = mask_local ? mask_local + (q * seq_k) : nullptr;
    T* global_row = mask_global ? mask_global + (q * seq_k) : nullptr;

    // KV Cache Part (indices 0 to capacity-1)
    // For Regular: valid if k < time_step.
    // For MTP: valid if k <= time_step.
    const int64_t kv_valid_limit =
        has_batch_suffix ? time_step : (time_step + 1);
    for (int64_t k = 0; k < std::min(kv_valid_limit, kv_cache_capacity); ++k) {
      if (global_row) global_row[k] = valid_val;
      // Sliding window (512 tokens).
      if (local_row && k >= effective_pos - 511) {
        local_row[k] = valid_val;
      }
    }

    // Batch/Draft Part (indices capacity to seq_k-1)
    if (has_batch_suffix) {
      for (int64_t k_rel = 0; k_rel < batch_size; ++k_rel) {
        int64_t k = kv_cache_capacity + k_rel;
        if (k >= seq_k) break;
        // Causal + Validity check (for verify_mask).
        bool is_valid = true;
        if (valid_mask != nullptr) {
          if (k_rel < valid_mask_size) {
            is_valid = valid_mask[k_rel];
          } else {
            is_valid = true;
          }
        } else if (input_tokens != nullptr) {
          if (seq_q > 1) {
            if (k_rel < input_tokens_size) {
              is_valid = (input_tokens[k_rel] != -1);
            }
          } else {
            is_valid = true;
          }
        }
        if (k_rel <= q && is_valid) {
          if (global_row) global_row[k] = valid_val;
          if (local_row)
            local_row[k] = valid_val;  // Current batch is always in window.
        }
      }
    }
  }
}

}  // namespace

absl::Status HWMaskUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers) {
  static constexpr absl::string_view kMaskLocal = "mask_local";
  static constexpr absl::string_view kMaskGlobal = "mask_global";
  static constexpr absl::string_view kInputTimeStep = "time_step";
  static constexpr absl::string_view kInputTokens = "input_tokens";
  static constexpr absl::string_view kValidMask = "valid_mask";

  LITERT_ASSIGN_OR_RETURN(auto time_step_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              in_buffers.at(kInputTimeStep),
                              ::litert::TensorBuffer::LockMode::kRead));
  int32_t time_step = static_cast<const int32_t*>(time_step_lock.second)[0];

  int64_t input_tokens_size = 0;
  std::optional<::litert::TensorBufferScopedLock> input_tokens_lock;
  const int32_t* input_tokens = nullptr;
  if (in_buffers.contains(kInputTokens)) {
    auto& buf = in_buffers.at(kInputTokens);
    LITERT_ASSIGN_OR_RETURN(auto type, buf.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto num_elements, type.Layout().NumElements());
    input_tokens_size = num_elements;
    LITERT_ASSIGN_OR_RETURN(auto lock,
                            ::litert::TensorBufferScopedLock::Create(
                                buf, ::litert::TensorBuffer::LockMode::kRead));
    input_tokens = static_cast<const int32_t*>(lock.second);
    input_tokens_lock.emplace(std::move(lock.first));
  }

  // Get Outputs and Shapes
  ::litert::TensorBuffer* mask_local_buf =
      in_buffers.contains(kMaskLocal) ? &in_buffers.at(kMaskLocal) : nullptr;
  ::litert::TensorBuffer* mask_global_buf =
      in_buffers.contains(kMaskGlobal) ? &in_buffers.at(kMaskGlobal) : nullptr;

  // Fallback to out_buffers if not in in_buffers (legacy behavior or output
  // only)
  if (!mask_local_buf && out_buffers.contains(kMaskLocal))
    mask_local_buf = &out_buffers.at(kMaskLocal);
  if (!mask_global_buf && out_buffers.contains(kMaskGlobal))
    mask_global_buf = &out_buffers.at(kMaskGlobal);

  if (!mask_local_buf && !mask_global_buf) {
    return absl::InvalidArgumentError(
        "No mask buffer found in in_buffers or out_buffers");
  }

  // Assume they have the same type and dimensions if both present.
  ::litert::TensorBuffer* reference_buf =
      mask_local_buf ? mask_local_buf : mask_global_buf;

  LITERT_ASSIGN_OR_RETURN(auto mask_type, reference_buf->TensorType());
  auto mask_dims = mask_type.Layout().Dimensions();
  int rank = mask_type.Layout().Rank();
  int64_t seq_q = mask_dims[rank - 2];
  int64_t seq_k = mask_dims[rank - 1];

  void* local_ptr = nullptr;
  void* global_ptr = nullptr;

  std::optional<::litert::TensorBufferScopedLock> mask_local_lock;
  std::optional<::litert::TensorBufferScopedLock> mask_global_lock;

  if (mask_local_buf) {
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        ::litert::TensorBufferScopedLock::Create(
            *mask_local_buf, ::litert::TensorBuffer::LockMode::kWrite));
    mask_local_lock.emplace(std::move(lock.first));
    local_ptr = lock.second;
  }

  if (mask_global_buf) {
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        ::litert::TensorBufferScopedLock::Create(
            *mask_global_buf, ::litert::TensorBuffer::LockMode::kWrite));
    mask_global_lock.emplace(std::move(lock.first));
    global_ptr = lock.second;
  }

  int64_t valid_mask_size = 0;
  std::optional<::litert::TensorBufferScopedLock> valid_mask_lock;
  const bool* valid_mask = nullptr;
  auto it = in_buffers.find(kValidMask);
  if (it != in_buffers.end()) {
    auto& buf = it->second;
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

  // Dispatch by Dtype
  if (mask_type.ElementType() == ::litert::ElementType::Int8) {
    FillMasksInternal<int8_t>(static_cast<int8_t*>(local_ptr),
                              static_cast<int8_t*>(global_ptr), seq_q, seq_k,
                              time_step, input_tokens, input_tokens_size,
                              valid_mask, valid_mask_size,
                              /*valid_val=*/127, /*masked_val=*/-128);
  } else if (mask_type.ElementType() == ::litert::ElementType::Int16) {
    FillMasksInternal<int16_t>(static_cast<int16_t*>(local_ptr),
                               static_cast<int16_t*>(global_ptr), seq_q, seq_k,
                               time_step, input_tokens, input_tokens_size,
                               valid_mask, valid_mask_size,
                               /*valid_val=*/0, /*masked_val=*/-32767);
  } else if (mask_type.ElementType() == ::litert::ElementType::Float32) {
    FillMasksInternal<float>(static_cast<float*>(local_ptr),
                             static_cast<float*>(global_ptr), seq_q, seq_k,
                             time_step, input_tokens, input_tokens_size,
                             valid_mask, valid_mask_size,
                             /*valid_val=*/0.0f, /*masked_val=*/-1e9f);
  } else if (mask_type.ElementType() == ::litert::ElementType::Float16) {
    // Opaque uint16_t representation of IEEE 754 Float16.
    // valid_val is 0.0f (0x0000) and masked_val is -infinity (0xFC00).
    FillMasksInternal<uint16_t>(static_cast<uint16_t*>(local_ptr),
                                static_cast<uint16_t*>(global_ptr), seq_q,
                                seq_k, time_step, input_tokens,
                                input_tokens_size, valid_mask, valid_mask_size,
                                /*valid_val=*/0x0000, /*masked_val=*/0xFC00);
  } else if (mask_type.ElementType() == ::litert::ElementType::BFloat16) {
    // Opaque uint16_t representation of Brain Float16.
    // valid_val is 0.0f (0x0000) and masked_val is -infinity (0xFF80).
    FillMasksInternal<uint16_t>(static_cast<uint16_t*>(local_ptr),
                                static_cast<uint16_t*>(global_ptr), seq_q,
                                seq_k, time_step, input_tokens,
                                input_tokens_size, valid_mask, valid_mask_size,
                                /*valid_val=*/0x0000, /*masked_val=*/0xFF80);
  } else {
    return absl::InvalidArgumentError("Unsupported mask element type");
  }

  return absl::OkStatus();
}
namespace {
#if defined(__ANDROID__) && defined(__ARM_NEON) && defined(__aarch64__)
void UnpackInt4Row(const uint8_t* packed_data, float scale, int col_size,
                   float* output) {
  float scale0 = scale / 16.0f;
  int j = 0;
  int idx = 0;

  int8x16_t mask_f0 = vdupq_n_s8(0xF0);

  for (; j <= col_size - 32; j += 32, idx += 16) {
    int8x16_t val =
        vld1q_s8(reinterpret_cast<const int8_t*>(packed_data + idx));

    int8x16_t low_16 = vshlq_n_s8(val, 4);
    int8x16_t high_16 = vandq_s8(val, mask_f0);

    // Low nibbles
    int16x8_t l_i16_low = vmovl_s8(vget_low_s8(low_16));
    int16x8_t l_i16_high = vmovl_high_s8(low_16);

    int32x4_t l_i32_0 = vmovl_s16(vget_low_s16(l_i16_low));
    int32x4_t l_i32_1 = vmovl_high_s16(l_i16_low);
    int32x4_t l_i32_2 = vmovl_s16(vget_low_s16(l_i16_high));
    int32x4_t l_i32_3 = vmovl_high_s16(l_i16_high);

    float32x4_t l_f32_0 = vcvtq_f32_s32(l_i32_0);
    float32x4_t l_f32_1 = vcvtq_f32_s32(l_i32_1);
    float32x4_t l_f32_2 = vcvtq_f32_s32(l_i32_2);
    float32x4_t l_f32_3 = vcvtq_f32_s32(l_i32_3);

    // High nibbles
    int16x8_t h_i16_low = vmovl_s8(vget_low_s8(high_16));
    int16x8_t h_i16_high = vmovl_high_s8(high_16);

    int32x4_t h_i32_0 = vmovl_s16(vget_low_s16(h_i16_low));
    int32x4_t h_i32_1 = vmovl_high_s16(h_i16_low);
    int32x4_t h_i32_2 = vmovl_s16(vget_low_s16(h_i16_high));
    int32x4_t h_i32_3 = vmovl_high_s16(h_i16_high);

    float32x4_t h_f32_0 = vcvtq_f32_s32(h_i32_0);
    float32x4_t h_f32_1 = vcvtq_f32_s32(h_i32_1);
    float32x4_t h_f32_2 = vcvtq_f32_s32(h_i32_2);
    float32x4_t h_f32_3 = vcvtq_f32_s32(h_i32_3);

    float32x4x2_t z0 = vzipq_f32(l_f32_0, h_f32_0);
    float32x4x2_t z1 = vzipq_f32(l_f32_1, h_f32_1);
    float32x4x2_t z2 = vzipq_f32(l_f32_2, h_f32_2);
    float32x4x2_t z3 = vzipq_f32(l_f32_3, h_f32_3);

    vst1q_f32(output + j + 0, vmulq_n_f32(z0.val[0], scale0));
    vst1q_f32(output + j + 4, vmulq_n_f32(z0.val[1], scale0));
    vst1q_f32(output + j + 8, vmulq_n_f32(z1.val[0], scale0));
    vst1q_f32(output + j + 12, vmulq_n_f32(z1.val[1], scale0));
    vst1q_f32(output + j + 16, vmulq_n_f32(z2.val[0], scale0));
    vst1q_f32(output + j + 20, vmulq_n_f32(z2.val[1], scale0));
    vst1q_f32(output + j + 24, vmulq_n_f32(z3.val[0], scale0));
    vst1q_f32(output + j + 28, vmulq_n_f32(z3.val[1], scale0));
  }

  for (; j < col_size - 1; j += 2, ++idx) {
    uint8_t packed_val = packed_data[idx];
    int8_t i8_val0 = static_cast<int8_t>(packed_val << 4);
    int8_t i8_val1 = static_cast<int8_t>(packed_val & 0xF0);

    output[j] = static_cast<float>(i8_val0) * scale0;
    output[j + 1] = static_cast<float>(i8_val1) * scale0;
  }
  if (col_size & 1) {
    uint8_t packed_val = packed_data[idx];
    int8_t i8_val0 = static_cast<int8_t>(packed_val << 4);
    output[j] = static_cast<float>(i8_val0) * scale0;
  }
}
#else
void UnpackInt4Row(const uint8_t* packed_data, float scale, int col_size,
                   float* output) {
  float scale0 = scale / 16.0f;
  int j = 0;
  int idx = 0;
  for (; j < col_size - 1; j += 2, ++idx) {
    uint8_t packed_val = packed_data[idx];
    int8_t i8_val0 = static_cast<int8_t>(packed_val << 4);
    int8_t i8_val1 = static_cast<int8_t>(packed_val & 0xF0);

    output[j] = static_cast<float>(i8_val0) * scale0;
    output[j + 1] = static_cast<float>(i8_val1) * scale0;
  }
  if (col_size & 1) {
    uint8_t packed_val = packed_data[idx];
    int8_t i8_val0 = static_cast<int8_t>(packed_val << 4);
    output[j] = static_cast<float>(i8_val0) * scale0;
  }
}
#endif

void DequantizeInt8Row(const int8_t* packed_data, float scale, int col_size,
                       float* output) {
  for (int i = 0; i < col_size; ++i) {
    output[i] = static_cast<float>(packed_data[i]) * scale;
  }
}
}  // namespace

absl::Status HWPerLayerEmbeddingLookup(
    const int* token_ids, int num_tokens, const uint8_t* const* table_ptrs,
    const HWQuantizationParams* quant_params, int num_tables,
    int ple_embedding_dim, void* output_buffer, litert::ElementType output_type,
    litert::ElementType ple_table_element_type, float mul_scale,
    float output_scale, int32_t final_zero_point) {
  constexpr int kVocabSize = 262144;
  std::vector<float> row_float;
  if (output_type == litert::ElementType::Int16) {
    row_float.resize(ple_embedding_dim);
  }

  int row_size_bytes = 0;
  if (ple_table_element_type == litert::ElementType::Int4) {
    row_size_bytes = ple_embedding_dim / 2;
  } else if (ple_table_element_type == litert::ElementType::Int8) {
    row_size_bytes = ple_embedding_dim;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported table element type: ",
                     static_cast<int>(ple_table_element_type)));
  }

  for (int t = 0; t < num_tokens; ++t) {
    int id = token_ids[t];
    if (id < 0 || id >= kVocabSize) {
      id = 0;  // Default to 0 as in model
    }

    size_t row_offset = id * row_size_bytes;

    for (int table_idx = 0; table_idx < num_tables; ++table_idx) {
      const uint8_t* table = table_ptrs[table_idx];
      const HWQuantizationParams& qp = quant_params[table_idx];
      const uint8_t* row_data = table + row_offset;

      if (table_idx + 1 < num_tables) {
        absl::PrefetchToLocalCache(table_ptrs[table_idx + 1] + row_offset);
      }

      float scale = 1.0f;
      if (qp.scales) {
        scale = qp.is_per_channel ? qp.scales[id] : qp.scales[0];
      }
      scale *= mul_scale;

      if (output_type == litert::ElementType::Int16) {
        if (ple_table_element_type == litert::ElementType::Int4) {
          UnpackInt4Row(row_data, scale, ple_embedding_dim, row_float.data());
        } else if (ple_table_element_type == litert::ElementType::Int8) {
          DequantizeInt8Row(reinterpret_cast<const int8_t*>(row_data), scale,
                            ple_embedding_dim, row_float.data());
        }
        int16_t* int16_output = static_cast<int16_t*>(output_buffer) +
                                t * num_tables * ple_embedding_dim +
                                table_idx * ple_embedding_dim;
        for (int i = 0; i < ple_embedding_dim; ++i) {
          float fval = row_float[i];
          int32_t qval = std::round(fval / output_scale) + final_zero_point;
          qval = std::clamp<int32_t>(qval, std::numeric_limits<int16_t>::min(),
                                     std::numeric_limits<int16_t>::max());
          int16_output[i] = static_cast<int16_t>(qval);
        }
      } else if (output_type == litert::ElementType::Float32) {
        float* float_output = static_cast<float*>(output_buffer) +
                              t * num_tables * ple_embedding_dim +
                              table_idx * ple_embedding_dim;
        if (ple_table_element_type == litert::ElementType::Int4) {
          UnpackInt4Row(row_data, scale, ple_embedding_dim, float_output);
        } else if (ple_table_element_type == litert::ElementType::Int8) {
          DequantizeInt8Row(reinterpret_cast<const int8_t*>(row_data), scale,
                            ple_embedding_dim, float_output);
        }
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported output type: ", static_cast<int>(output_type)));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status DequantizeLogits(const ::litert::TensorBuffer& src,
                              ::litert::TensorBuffer& dst, float scale,
                              int32_t zero_point, bool should_dump) {
  LITERT_ASSIGN_OR_RETURN(auto src_type, src.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto dst_type, dst.TensorType());
  RET_CHECK_EQ((int)dst_type.ElementType(),
               (int)::litert::ElementType::Float32);

  LITERT_ASSIGN_OR_RETURN(size_t num_elements, src_type.Layout().NumElements());

  const auto src_elem_type = src_type.ElementType();

  LITERT_ASSIGN_OR_RETURN(auto src_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              const_cast<::litert::TensorBuffer&>(src),
                              ::litert::TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto dst_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              dst, ::litert::TensorBuffer::LockMode::kWrite));

  float* dst_ptr = static_cast<float*>(dst_lock.second);
  const void* src_raw_ptr = src_lock.second;

  if (src_elem_type == ::litert::ElementType::Int16) {
    const int16_t* src_ptr = static_cast<const int16_t*>(src_raw_ptr);
    for (size_t i = 0; i < num_elements; ++i) {
      dst_ptr[i] = scale * (static_cast<float>(src_ptr[i]) -
                            static_cast<float>(zero_point));
    }
  } else if (src_elem_type == ::litert::ElementType::Int8) {
    const int8_t* src_ptr = static_cast<const int8_t*>(src_raw_ptr);
    for (size_t i = 0; i < num_elements; ++i) {
      dst_ptr[i] = scale * (static_cast<float>(src_ptr[i]) -
                            static_cast<float>(zero_point));
    }
  } else if (src_elem_type == ::litert::ElementType::Float32) {
    // This is for dealing with unquantized float 32 logits.
    const float* src_ptr = static_cast<const float*>(src_raw_ptr);
    for (size_t i = 0; i < num_elements; ++i) {
      dst_ptr[i] = src_ptr[i];
    }
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported source type for dequantization: ", (int)src_elem_type));
  }

  return absl::OkStatus();
}

absl::Status WritePleEmbeddingsToPtr(void* dest_ptr,
                                     absl::Span<const float> ple_embeddings,
                                     litert::ElementType output_type,
                                     float final_scale,
                                     int32_t final_zero_point) {
  if (output_type == litert::ElementType::Int16) {
    int16_t* int16_ptr = static_cast<int16_t*>(dest_ptr);
    for (size_t i = 0; i < ple_embeddings.size(); ++i) {
      int16_ptr[i] =
          Quantize<int16_t>(ple_embeddings[i], final_scale, final_zero_point);
    }
  } else if (output_type == litert::ElementType::Float16) {
    tflite::half* fp16_ptr = static_cast<tflite::half*>(dest_ptr);
    for (size_t i = 0; i < ple_embeddings.size(); ++i) {
      fp16_ptr[i] = tflite::half(ple_embeddings[i]);
    }
  } else if (output_type == litert::ElementType::Float32 ||
             output_type == litert::ElementType::None) {
    float* float_ptr = static_cast<float*>(dest_ptr);
    std::memcpy(float_ptr, ple_embeddings.data(),
                ple_embeddings.size() * sizeof(float));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported PLE output type: ", output_type));
  }
  return absl::OkStatus();
}

absl::Status WritePleEmbeddings(::litert::TensorBuffer& buffer,
                                absl::Span<const float> ple_embeddings,
                                litert::ElementType output_type,
                                float final_scale, int32_t final_zero_point) {
  LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer.PackedSize());
  size_t element_size = 0;
  if (output_type == litert::ElementType::Int16) {
    element_size = sizeof(int16_t);
  } else if (output_type == litert::ElementType::Float16) {
    element_size = sizeof(tflite::half);
  } else if (output_type == litert::ElementType::Float32 ||
             output_type == litert::ElementType::None) {
    element_size = sizeof(float);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported PLE output type: ", output_type));
  }
  RET_CHECK_GE(buffer_size, ple_embeddings.size() * element_size);

  LITERT_ASSIGN_OR_RETURN(
      auto lock, ::litert::TensorBufferScopedLock::Create(
                     buffer, ::litert::TensorBuffer::LockMode::kWrite));
  return WritePleEmbeddingsToPtr(lock.second, ple_embeddings, output_type,
                                 final_scale, final_zero_point);
}

absl::Status WriteAndPadPleEmbeddings(::litert::TensorBuffer& buffer,
                                      absl::Span<const float> ple_embeddings,
                                      size_t ple_dim, size_t seq_pos_size,
                                      const std::vector<float>& default_ple_emb,
                                      litert::ElementType output_type,
                                      float final_scale,
                                      int32_t final_zero_point) {
  RET_CHECK_EQ(ple_embeddings.size(), seq_pos_size * ple_dim);
  LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer.PackedSize());
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          buffer, ::litert::TensorBuffer::LockMode::kWrite));

  size_t num_tokens_to_fill =
      buffer_size /
      (ple_dim * (output_type == litert::ElementType::Int16 ? sizeof(int16_t)
                  : output_type == litert::ElementType::Float16
                      ? sizeof(tflite::half)
                      : sizeof(float)));
  RET_CHECK_LE(seq_pos_size, num_tokens_to_fill);

  LITERT_RETURN_IF_ERROR(
      WritePleEmbeddingsToPtr(lock_and_addr.second, ple_embeddings, output_type,
                              final_scale, final_zero_point));

  if (output_type == litert::ElementType::Int16) {
    int16_t* int16_ptr = static_cast<int16_t*>(lock_and_addr.second);

    // Quantize default PLE embedding. If not provided or size doesn't match
    // ple_dim, pad with quantized zeros.
    std::vector<int16_t> quantized_default_ple_emb(
        ple_dim, Quantize<int16_t>(0.0f, final_scale, final_zero_point));
    if (default_ple_emb.size() == ple_dim) {
      for (size_t i = 0; i < ple_dim; ++i) {
        quantized_default_ple_emb[i] = Quantize<int16_t>(
            default_ple_emb[i], final_scale, final_zero_point);
      }
    }

    // Pad the rest
    int16_t* padding_ptr = int16_ptr + seq_pos_size * ple_dim;
    for (size_t i = seq_pos_size; i < num_tokens_to_fill; ++i) {
      std::memcpy(padding_ptr, quantized_default_ple_emb.data(),
                  ple_dim * sizeof(int16_t));
      padding_ptr += ple_dim;
    }
  } else if (output_type == litert::ElementType::Float16) {
    tflite::half* fp16_ptr = static_cast<tflite::half*>(lock_and_addr.second);
    std::vector<tflite::half> fp16_default_ple_emb(ple_dim, tflite::half(0.0f));
    if (default_ple_emb.size() == ple_dim) {
      for (size_t i = 0; i < ple_dim; ++i) {
        fp16_default_ple_emb[i] = tflite::half(default_ple_emb[i]);
      }
    }
    tflite::half* padding_ptr = fp16_ptr + seq_pos_size * ple_dim;
    for (size_t i = seq_pos_size; i < num_tokens_to_fill; ++i) {
      std::memcpy(padding_ptr, fp16_default_ple_emb.data(),
                  ple_dim * sizeof(tflite::half));
      padding_ptr += ple_dim;
    }
  } else if (output_type == litert::ElementType::Float32 ||
             output_type == litert::ElementType::None) {
    // Float32 path
    float* float_ptr = static_cast<float*>(lock_and_addr.second);

    float* padding_ptr = float_ptr + seq_pos_size * ple_dim;
    if (default_ple_emb.size() == ple_dim) {
      for (size_t i = seq_pos_size; i < num_tokens_to_fill; ++i) {
        std::memcpy(padding_ptr, default_ple_emb.data(),
                    ple_dim * sizeof(float));
        padding_ptr += ple_dim;
      }
    } else {
      std::memset(
          padding_ptr, 0,
          (num_tokens_to_fill - seq_pos_size) * ple_dim * sizeof(float));
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
