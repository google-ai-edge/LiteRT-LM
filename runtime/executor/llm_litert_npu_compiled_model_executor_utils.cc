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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/base/prefetch.h"  // from @com_google_absl

#if defined(__ANDROID__) && defined(__ARM_NEON)
#include <arm_neon.h>

#include <limits>  // IWYU pragma: keep
#endif

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

#if defined(__ANDROID__) && defined(__ARM_NEON)
int FindMaxIndexFloatNeon(const float* data, int size) {
  if (size <= 0) return 0;
  float32x4_t max_v4 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  int i = 0;
  for (; i <= size - 4; i += 4) {
    max_v4 = vmaxq_f32(max_v4, vld1q_f32(data + i));
  }
  float max_vals_arr[4];
  vst1q_f32(max_vals_arr, max_v4);
  float max_v = max_vals_arr[0];
  for (int j = 1; j < 4; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  // Second pass: find first index with max_v
  float32x4_t target = vdupq_n_f32(max_v);
  for (i = 0; i <= size - 4; i += 4) {
    uint32x4_t cmp = vceqq_f32(vld1q_f32(data + i), target);
    uint32_t mask[4];
    vst1q_u32(mask, cmp);
    if (mask[0] || mask[1] || mask[2] || mask[3]) {
      for (int j = 0; j < 4; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
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
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers) {
  static constexpr absl::string_view kInputPos = "input_pos";
  auto& input_pos_buffer = in_buffers.at(kInputPos);

  LITERT_ASSIGN_OR_RETURN(
      auto pos_lock,
      ::litert::TensorBufferScopedLock::Create(
          input_pos_buffer, ::litert::TensorBuffer::LockMode::kRead));
  int start_pos = static_cast<const int32_t*>(pos_lock.second)[0];

  auto perform_update =
      [&](::litert::TensorBuffer& cache,
          const ::litert::TensorBuffer& slice) -> absl::Status {
    LITERT_ASSIGN_OR_RETURN(auto cache_type, cache.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto slice_type, slice.TensorType());
    auto cache_dims = cache_type.Layout().Dimensions();
    auto slice_dims = slice_type.Layout().Dimensions();
    int cache_rank = cache_type.Layout().Rank();
    int slice_rank = slice_type.Layout().Rank();

    LITERT_ASSIGN_OR_RETURN(size_t cache_bytes, cache.Size());
    LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                            cache_type.Layout().NumElements());
    size_t element_size = cache_bytes / num_elements;

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

    if (start_pos + slice_seq > cache_seq) {
      return absl::OutOfRangeError("KV-cache update out of range");
    }

    LITERT_ASSIGN_OR_RETURN(
        auto cache_lock, ::litert::TensorBufferScopedLock::Create(
                             cache, ::litert::TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(
        auto slice_lock, ::litert::TensorBufferScopedLock::Create(
                             slice, ::litert::TensorBuffer::LockMode::kRead));

    uint8_t* cache_ptr = static_cast<uint8_t*>(cache_lock.second);
    const uint8_t* slice_ptr = static_cast<const uint8_t*>(slice_lock.second);

    bool cache_is_transposed = (cache_seq_dim == cache_rank - 1);
    bool slice_is_transposed = (slice_seq_dim == slice_rank - 1);

    int64_t outer_size = 1;
    for (int i = 0; i < cache_rank - 2; ++i) {
      outer_size *= cache_dims[i];
    }

    for (int64_t o = 0; o < outer_size; ++o) {
      uint8_t* c_ptr = cache_ptr + o * (cache_seq * hidden_dim * element_size);
      const uint8_t* s_ptr =
          slice_ptr + o * (slice_seq * hidden_dim * element_size);

      if (!cache_is_transposed) {
        if (!slice_is_transposed || slice_seq == 1) {
          // Cache is [..., seq, hidden], Slice is [..., seq, hidden] (or seq=1)
          std::memcpy(c_ptr + (start_pos * hidden_dim * element_size), s_ptr,
                      slice_seq * hidden_dim * element_size);
        } else {
          // Cache is [..., seq, hidden], Slice is [..., hidden, seq]
          for (int64_t s = 0; s < slice_seq; ++s) {
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(
                  c_ptr + ((start_pos + s) * hidden_dim + h) * element_size,
                  s_ptr + (h * slice_seq + s) * element_size, element_size);
            }
          }
        }
      } else {
        // Cache is [..., hidden, seq]
        if (slice_seq == 1) {
#if defined(__ANDROID__) && defined(__ARM_NEON) && defined(__aarch64__)
          if (element_size == 1) {
            int64_t h = 0;
            for (; h <= hidden_dim - 16; h += 16) {
              uint8x16_t v = vld1q_u8(s_ptr + h);
              c_ptr[(h + 0) * cache_seq + start_pos] = vgetq_lane_u8(v, 0);
              c_ptr[(h + 1) * cache_seq + start_pos] = vgetq_lane_u8(v, 1);
              c_ptr[(h + 2) * cache_seq + start_pos] = vgetq_lane_u8(v, 2);
              c_ptr[(h + 3) * cache_seq + start_pos] = vgetq_lane_u8(v, 3);
              c_ptr[(h + 4) * cache_seq + start_pos] = vgetq_lane_u8(v, 4);
              c_ptr[(h + 5) * cache_seq + start_pos] = vgetq_lane_u8(v, 5);
              c_ptr[(h + 6) * cache_seq + start_pos] = vgetq_lane_u8(v, 6);
              c_ptr[(h + 7) * cache_seq + start_pos] = vgetq_lane_u8(v, 7);
              c_ptr[(h + 8) * cache_seq + start_pos] = vgetq_lane_u8(v, 8);
              c_ptr[(h + 9) * cache_seq + start_pos] = vgetq_lane_u8(v, 9);
              c_ptr[(h + 10) * cache_seq + start_pos] = vgetq_lane_u8(v, 10);
              c_ptr[(h + 11) * cache_seq + start_pos] = vgetq_lane_u8(v, 11);
              c_ptr[(h + 12) * cache_seq + start_pos] = vgetq_lane_u8(v, 12);
              c_ptr[(h + 13) * cache_seq + start_pos] = vgetq_lane_u8(v, 13);
              c_ptr[(h + 14) * cache_seq + start_pos] = vgetq_lane_u8(v, 14);
              c_ptr[(h + 15) * cache_seq + start_pos] = vgetq_lane_u8(v, 15);
            }
            for (; h < hidden_dim; ++h) {
              c_ptr[h * cache_seq + start_pos] = s_ptr[h];
            }
          } else if (element_size == 2) {
            int64_t h = 0;
            const uint16_t* s_ptr16 = reinterpret_cast<const uint16_t*>(s_ptr);
            uint16_t* c_ptr16 = reinterpret_cast<uint16_t*>(c_ptr);
            for (; h <= hidden_dim - 8; h += 8) {
              uint16x8_t v = vld1q_u16(s_ptr16 + h);
              c_ptr16[(h + 0) * cache_seq + start_pos] = vgetq_lane_u16(v, 0);
              c_ptr16[(h + 1) * cache_seq + start_pos] = vgetq_lane_u16(v, 1);
              c_ptr16[(h + 2) * cache_seq + start_pos] = vgetq_lane_u16(v, 2);
              c_ptr16[(h + 3) * cache_seq + start_pos] = vgetq_lane_u16(v, 3);
              c_ptr16[(h + 4) * cache_seq + start_pos] = vgetq_lane_u16(v, 4);
              c_ptr16[(h + 5) * cache_seq + start_pos] = vgetq_lane_u16(v, 5);
              c_ptr16[(h + 6) * cache_seq + start_pos] = vgetq_lane_u16(v, 6);
              c_ptr16[(h + 7) * cache_seq + start_pos] = vgetq_lane_u16(v, 7);
            }
            for (; h < hidden_dim; ++h) {
              c_ptr16[h * cache_seq + start_pos] = s_ptr16[h];
            }
          } else
#endif
          {
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(c_ptr + (h * cache_seq + start_pos) * element_size,
                          s_ptr + h * element_size, element_size);
            }
          }
        } else if (slice_is_transposed) {
          // Cache is [..., hidden, seq], Slice is [..., hidden, seq]
          for (int64_t h = 0; h < hidden_dim; ++h) {
            std::memcpy(c_ptr + (h * cache_seq + start_pos) * element_size,
                        s_ptr + (h * slice_seq) * element_size,
                        slice_seq * element_size);
          }
        } else {
          // Cache is [..., hidden, seq], Slice is [..., seq, hidden]
          for (int64_t s = 0; s < slice_seq; ++s) {
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(
                  c_ptr + (h * cache_seq + start_pos + s) * element_size,
                  s_ptr + (s * hidden_dim + h) * element_size, element_size);
            }
          }
        }
      }
    }
    return absl::OkStatus();
  };

  for (int layer_id = 0;; ++layer_id) {
    char k_cache_name[32];
    snprintf(k_cache_name, sizeof(k_cache_name), "kv_cache_k_%d", layer_id);
    if (!in_buffers.contains(k_cache_name)) break;

    char v_cache_name[32];
    snprintf(v_cache_name, sizeof(v_cache_name), "kv_cache_v_%d", layer_id);
    char k_slice_name[32];
    snprintf(k_slice_name, sizeof(k_slice_name), "kv_slice_k_%d", layer_id);
    char v_slice_name[32];
    snprintf(v_slice_name, sizeof(v_slice_name), "kv_slice_v_%d", layer_id);

    auto& in_k_cache = in_buffers.at(k_cache_name);
    auto& in_v_cache = in_buffers.at(v_cache_name);
    const auto& k_slice = in_buffers.at(k_slice_name);
    const auto& v_slice = in_buffers.at(v_slice_name);

    LITERT_RETURN_IF_ERROR(perform_update(in_k_cache, k_slice));
    LITERT_RETURN_IF_ERROR(perform_update(in_v_cache, v_slice));

    if (out_buffers.contains(k_cache_name)) {
      auto& out_k_cache = out_buffers.at(k_cache_name);
      if (in_k_cache.Get() != out_k_cache.Get()) {
        LITERT_RETURN_IF_ERROR(perform_update(out_k_cache, k_slice));
      }
    }
    if (out_buffers.contains(v_cache_name)) {
      auto& out_v_cache = out_buffers.at(v_cache_name);
      if (in_v_cache.Get() != out_v_cache.Get()) {
        LITERT_RETURN_IF_ERROR(perform_update(out_v_cache, v_slice));
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
                       const int32_t* input_tokens, T valid_val, T masked_val) {
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
        if (k_rel <= q &&
            (input_tokens == nullptr || input_tokens[k_rel] != -1)) {
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

  LITERT_ASSIGN_OR_RETURN(auto time_step_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              in_buffers.at(kInputTimeStep),
                              ::litert::TensorBuffer::LockMode::kRead));
  int32_t time_step = static_cast<const int32_t*>(time_step_lock.second)[0];

  const int32_t* input_tokens = nullptr;
  if (in_buffers.contains(kInputTokens)) {
    LITERT_ASSIGN_OR_RETURN(auto input_tokens_lock,
                            ::litert::TensorBufferScopedLock::Create(
                                in_buffers.at(kInputTokens),
                                ::litert::TensorBuffer::LockMode::kRead));
    input_tokens = static_cast<const int32_t*>(input_tokens_lock.second);
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

  // Dispatch by Dtype
  if (mask_type.ElementType() == ::litert::ElementType::Int8) {
    FillMasksInternal<int8_t>(static_cast<int8_t*>(local_ptr),
                              static_cast<int8_t*>(global_ptr), seq_q, seq_k,
                              time_step, input_tokens,
                              /*valid_val=*/127, /*masked_val=*/-128);
  } else if (mask_type.ElementType() == ::litert::ElementType::Int16) {
    FillMasksInternal<int16_t>(static_cast<int16_t*>(local_ptr),
                               static_cast<int16_t*>(global_ptr), seq_q, seq_k,
                               time_step, input_tokens,
                               /*valid_val=*/0, /*masked_val=*/-32767);
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
}  // namespace

absl::Status HWPerLayerEmbeddingLookup(
    const int* token_ids, int num_tokens, const uint8_t* const* table_ptrs,
    const HWQuantizationParams* quant_params, int num_tables, int col_size,
    void* output_buffer, litert::ElementType output_type, float final_scale,
    int32_t final_zero_point) {
  constexpr int kVocabSize = 262144;
  std::vector<float> row_float;
  if (output_type == litert::ElementType::Int16) {
    row_float.resize(col_size);
  }

  for (int t = 0; t < num_tokens; ++t) {
    int id = token_ids[t];
    if (id < 0 || id >= kVocabSize) {
      id = 0;  // Default to 0 as in model
    }

    size_t row_offset = id * (col_size / 2);

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

      if (output_type == litert::ElementType::Int16) {
        UnpackInt4Row(row_data, scale, col_size, row_float.data());
        int16_t* int16_output = static_cast<int16_t*>(output_buffer) +
                                t * num_tables * col_size +
                                table_idx * col_size;
        for (int i = 0; i < col_size; ++i) {
          float fval = row_float[i];
          int32_t qval = std::round(fval / final_scale) + final_zero_point;
          qval = std::max(-32768, std::min(32767, qval));
          int16_output[i] = static_cast<int16_t>(qval);
        }
      } else {
        float* float_output = static_cast<float*>(output_buffer) +
                              t * num_tables * col_size + table_idx * col_size;
        UnpackInt4Row(row_data, scale, col_size, float_output);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
