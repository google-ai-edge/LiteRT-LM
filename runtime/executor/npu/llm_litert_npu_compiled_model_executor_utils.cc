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

#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/prefetch.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_layout.h"  // from @litert
#include "litert/c/litert_model_types.h"  // from @litert
#include "litert/c/litert_op_code.h"  // from @litert
#include "litert/cc/internal/litert_extended_model.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
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

namespace litert::lm {

namespace {
class CompiledModelWrapper : public ::litert::CompiledModel {
 public:
  static ::litert::Expected<::litert::CompiledModel> Create(
      ::litert::Environment& env, const LiteRtModel litert_model,
      ::litert::Options& compilation_options) {
    return ::litert::CompiledModel::Create(env, litert_model,
                                           compilation_options);
  }
  static ::litert::Expected<::litert::CompiledModel> Create(
      ::litert::Environment& env, const LiteRtModel litert_model,
      litert::HwAccelerators accelerators) {
    return ::litert::CompiledModel::Create(env, litert_model, accelerators);
  }
};
}  // namespace

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

// -----------------------------------------------------------------------------
// CPU Options & Embedder Context Creation
// -----------------------------------------------------------------------------

litert::Expected<litert::Options> CreateLiteRtCpuOptions(
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  return options;
}

// Creates LiteRT options for NPU accelerator.
litert::Expected<litert::Options> CreateLiteRtNpuOptions(
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kNpu |
                                  litert::HwAccelerators::kCpu);
  // TODO: saliltambe - Bug: 498622107
#if defined(__ANDROID__)
  LITERT_ASSIGN_OR_RETURN(::litert::qualcomm::QualcommOptions & qnn_opts,
                          options.GetQualcommOptions());
  qnn_opts.SetLogLevel(::litert::qualcomm::QualcommOptions::LogLevel::kOff);
  qnn_opts.SetHtpPerformanceMode(
      ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
  LITERT_ASSIGN_OR_RETURN(auto& google_tensor_opts,
                          options.GetGoogleTensorOptions());
  google_tensor_opts.SetPerformanceMode(
      ::litert::google_tensor::GoogleTensorOptions::PerformanceMode::kBurst);
#endif
  return options;
}

std::ostream& operator<<(std::ostream& os, const LatencyStats& stats) {
  auto safe_tokens_per_sec = [](uint32_t num_tokens,
                                uint64_t latency_us) -> float {
    if (latency_us == 0) return 0.0f;
    return (static_cast<float>(num_tokens) * 1000000.0f) /
           static_cast<float>(latency_us);
  };
  auto safe_percentage = [](uint64_t part_us, uint64_t total_us) -> float {
    if (total_us == 0) return 0.0f;
    return (static_cast<float>(part_us) * 100.0f) /
           static_cast<float>(total_us);
  };

  os << "\n" << "====== PREFILL STATS ======";
  os << "\n" << "Total prefill latency [us]: " << stats.prefill_e2e_latency_us;
  os << "\n" << "(e2e) Prefill num tokens: " << stats.prefill_num_tokens;
  os << "\n"
     << "(e2e) Prefill tokens per second: "
     << safe_tokens_per_sec(stats.prefill_num_tokens,
                            stats.prefill_e2e_latency_us);
  os << "\n"
     << "(TransformerStackOnly) Prefill tokens per second: "
     << safe_tokens_per_sec(stats.prefill_num_tokens,
                            stats.prefill_llm_inference_latency_us);

  os << "\n" << "------ Prefill breakdown ------";
  os << "\n"
     << "Total prefill prepare input tensors latency [us]: "
     << stats.prefill_prepare_input_latency_us << " ("
     << safe_percentage(stats.prefill_prepare_input_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill embedder inference latency [us]: "
     << stats.prefill_embedder_inference_latency_us << " ("
     << safe_percentage(stats.prefill_embedder_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  if (stats.prefill_embedder_per_layer_inference_latency_us > 0) {
    os << "\n"
       << "Total prefill embedder per layer inference latency [us]: "
       << stats.prefill_embedder_per_layer_inference_latency_us << " ("
       << safe_percentage(stats.prefill_embedder_per_layer_inference_latency_us,
                          stats.prefill_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total prefill rope inference latency [us]: "
     << stats.prefill_rope_inference_latency_us << " ("
     << safe_percentage(stats.prefill_rope_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill mask inference latency [us]: "
     << stats.prefill_mask_inference_latency_us << " ("
     << safe_percentage(stats.prefill_mask_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill llm inference latency [us]: "
     << stats.prefill_llm_inference_latency_us << " ("
     << safe_percentage(stats.prefill_llm_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill cache update inference latency [us]: "
     << stats.prefill_cache_update_inference_latency_us << " ("
     << safe_percentage(stats.prefill_cache_update_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";

  os << "\n\n" << "====== DECODE STATS ======";
  os << "\n" << "Total decode latency [us]: " << stats.decode_e2e_latency_us;
  os << "\n" << "(e2e) Decode num tokens: " << stats.decode_num_tokens;
  os << "\n"
     << "(e2e) Decode tokens per second (avg): "
     << safe_tokens_per_sec(stats.decode_num_tokens,
                            stats.decode_e2e_latency_us);
  if (stats.mtp_num_draft_tokens > 0) {
    os << "\n"
       << "Speculative decoding acceptance rate [%]: "
       << (float)stats.mtp_num_accepted_tokens / stats.mtp_num_draft_tokens *
              100;
  }
  os << "\n"
     << "(TransformerStackOnly) Decode tokens per second: "
     << safe_tokens_per_sec(stats.decode_num_tokens,
                            stats.decode_llm_inference_latency_us);

  os << "\n" << "------ Decode breakdown ------";
  os << "\n"
     << "Total decode prepare input tensors latency [us]: "
     << stats.decode_prepare_input_latency_us << " ("
     << safe_percentage(stats.decode_prepare_input_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode embedder inference latency [us]: "
     << stats.decode_embedder_inference_latency_us << " ("
     << safe_percentage(stats.decode_embedder_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  if (stats.decode_embedder_per_layer_inference_latency_us > 0) {
    os << "\n"
       << "Total decode embedder per layer inference latency [us]: "
       << stats.decode_embedder_per_layer_inference_latency_us << " ("
       << safe_percentage(stats.decode_embedder_per_layer_inference_latency_us,
                          stats.decode_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total decode rope inference latency [us]: "
     << stats.decode_rope_inference_latency_us << " ("
     << safe_percentage(stats.decode_rope_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode mask inference latency [us]: "
     << stats.decode_mask_inference_latency_us << " ("
     << safe_percentage(stats.decode_mask_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode llm inference latency [us]: "
     << stats.decode_llm_inference_latency_us << " ("
     << safe_percentage(stats.decode_llm_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode cache update inference latency [us]: "
     << stats.decode_cache_update_inference_latency_us << " ("
     << safe_percentage(stats.decode_cache_update_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode sampling latency [us]: "
     << stats.decode_sampling_latency_us << " ("
     << safe_percentage(stats.decode_sampling_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  if (stats.decode_mtp_rejection_sampling_latency_us > 0) {
    os << "\n"
       << "Total decode MTP rejection sampling latency [us]: "
       << stats.decode_mtp_rejection_sampling_latency_us << " ("
       << safe_percentage(stats.decode_mtp_rejection_sampling_latency_us,
                          stats.decode_e2e_latency_us)
       << "%)";
  }
  if (stats.decode_mtp_activation_copy_latency_us > 0) {
    os << "\n"
       << "Total decode MTP activation copy latency [us]: "
       << stats.decode_mtp_activation_copy_latency_us << " ("
       << safe_percentage(stats.decode_mtp_activation_copy_latency_us,
                          stats.decode_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total decode token queue latency [us]: "
     << stats.decode_token_queue_latency_us << " ("
     << safe_percentage(stats.decode_token_queue_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";

  return os;
}

namespace {
template <typename ContextT>
absl::StatusOr<ContextT> CreateEmbedderContextHelper(
    ::litert::Environment& env, const litert::Model& embedder_model,
    absl::string_view prefill_signature, absl::string_view decode_signature,
    absl::string_view verify_signature, absl::string_view input_name,
    absl::string_view output_name, absl::string_view decoder_output_name,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers,
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtCpuOptions(settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel embedder_compiled_model,
      CompiledModelWrapper::Create(env, embedder_model.Get(), options));

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

  LITERT_ASSIGN_OR_RETURN(
      prefill_input_buffers[input_name],
      embedder_compiled_model.CreateInputBuffer(prefill_signature, input_name));
  prefill_input_buffers[input_name].Clear();

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[output_name],
      text_decoder_prefill_input_buffers[decoder_output_name].Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[input_name],
      embedder_compiled_model.CreateInputBuffer(decode_signature, input_name));
  decode_input_buffers[input_name].Clear();

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[output_name],
      text_decoder_decode_input_buffers[decoder_output_name].Duplicate());

  if (embedder_compiled_model.FindSignature(verify_signature)) {
    LITERT_ASSIGN_OR_RETURN(verify_input_buffers[input_name],
                            embedder_compiled_model.CreateInputBuffer(
                                verify_signature, input_name));
    verify_input_buffers[input_name].Clear();

    if (text_decoder_verify_input_buffers.contains(decoder_output_name)) {
      LITERT_ASSIGN_OR_RETURN(
          verify_output_buffers[output_name],
          text_decoder_verify_input_buffers[decoder_output_name].Duplicate());
    } else {
      LITERT_ASSIGN_OR_RETURN(verify_output_buffers[output_name],
                              embedder_compiled_model.CreateOutputBuffer(
                                  verify_signature, output_name));
    }
  }

  return ContextT(
      std::move(embedder_compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(verify_input_buffers),
      std::move(verify_output_buffers));
}
}  // namespace

absl::StatusOr<EmbedderContext> NpuEmbedder::CreateEmbedderContext(
    ::litert::Environment& env, const litert::Model& embedder_model,
    const ResolvedPrefillSignatures& prefill_signatures,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers,
    const LlmExecutorSettings& settings) {
  return CreateEmbedderContextHelper<EmbedderContext>(
      env, embedder_model, prefill_signatures.embedder,
      EmbedderSignatures::kDecodeEmbedder, EmbedderSignatures::kVerifyEmbedder,
      EmbedderSignatures::kEmbedderInput, EmbedderSignatures::kEmbedderOutput,
      TextDecoderSignatures::kInputEmbeddings,
      text_decoder_prefill_input_buffers, text_decoder_decode_input_buffers,
      text_decoder_verify_input_buffers, settings);
}

absl::StatusOr<EmbedderPerLayerContext>
NpuEmbedder::CreateEmbedderPerLayerContext(
    ::litert::Environment& env, const litert::Model& embedder_model,
    const ResolvedPrefillSignatures& prefill_signatures,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers,
    const LlmExecutorSettings& settings) {
  return CreateEmbedderContextHelper<EmbedderPerLayerContext>(
      env, embedder_model, prefill_signatures.embedder_per_layer,
      EmbedderPerLayerSignatures::kDecodeEmbedderPerLayer,
      EmbedderPerLayerSignatures::kVerifyEmbedderPerLayer,
      EmbedderPerLayerSignatures::kEmbedderInput,
      EmbedderPerLayerSignatures::kEmbedderOutput, kPerLayerEmbedderTensor,
      text_decoder_prefill_input_buffers, text_decoder_decode_input_buffers,
      text_decoder_verify_input_buffers, settings);
}

// -----------------------------------------------------------------------------
// NpuEmbedder Implementation
// -----------------------------------------------------------------------------

absl::StatusOr<NpuEmbedder> NpuEmbedder::Create(
    ::litert::Environment& env, ModelResources& resources,
    const LlmExecutorSettings& executor_settings,
    const ResolvedPrefillSignatures& prefill_signatures,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers,
    bool has_per_layer_embeddings) {
  LITERT_ASSIGN_OR_RETURN(auto embedder_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteEmbedder));
  LITERT_ASSIGN_OR_RETURN(
      auto embedder_context,
      CreateEmbedderContext(
          env, *embedder_lrt_model, prefill_signatures,
          text_decoder_prefill_input_buffers, text_decoder_decode_input_buffers,
          text_decoder_verify_input_buffers, executor_settings));

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager = nullptr;
  const bool has_vision_or_audio =
      resources.GetTFLiteModel(ModelType::kTfLiteVisionEncoder).ok() ||
      resources.GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw).ok();

  if (has_per_layer_embeddings || has_vision_or_audio) {
    absl::flat_hash_map<int, const Model*> end_of_multi_modal_embedding_models;
    auto add_multi_modal_end_model = [&](ModelType type, int token) {
      auto model_buffer = resources.GetTFLiteModelBuffer(type);
      if (model_buffer.ok() && !model_buffer->empty()) {
        auto model = resources.GetTFLiteModel(type);
        if (model.ok()) {
          end_of_multi_modal_embedding_models[token] = *model;
        }
      }
    };

    add_multi_modal_end_model(ModelType::kTfLiteEndOfAudio,
                              litert::lm::ExecutorAudioData::kEndToken);
    add_multi_modal_end_model(ModelType::kTfLiteEndOfVision,
                              litert::lm::ExecutorVisionData::kEndToken);

    LITERT_ASSIGN_OR_RETURN(
        embedding_lookup_manager,
        EmbeddingLookupManager::Create(env, embedder_lrt_model,
                                       end_of_multi_modal_embedding_models,
                                       true, "decode_embedder"));
  }

  std::optional<EmbedderPerLayerContext> embedder_per_layer_context =
      std::nullopt;
  HWPleParams ple_params;
  const litert::Model* embedder_per_layer_model = nullptr;

  if (has_per_layer_embeddings) {
    LITERT_ASSIGN_OR_RETURN(
        embedder_per_layer_model,
        resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder));

    bool use_hw_ple_for_npu = false;
    auto npu_config_status = executor_settings.GetBackendConfig<NpuConfig>();
    if (npu_config_status.ok()) {
      use_hw_ple_for_npu = npu_config_status->use_hw_ple_for_npu;
    }

    if (use_hw_ple_for_npu) {
      ple_params.use_hw_ple = true;
      auto extended_model = ExtendedModel::CreateFromNonOwnedHandle(
          embedder_per_layer_model->Get());
      LITERT_ASSIGN_OR_RETURN(auto subgraph, extended_model.MainSubgraph());
      auto ops = subgraph.Ops();
      for (const auto& op : ops) {
        if (op.Code() == kLiteRtOpCodeTflEmbeddingLookup) {
          LITERT_ASSIGN_OR_RETURN(auto table_tensor, op.Input(1));
          LITERT_ASSIGN_OR_RETURN(auto table_type_info,
                                  table_tensor.RankedTensorType());
          auto table_dims = table_type_info.Layout().Dimensions();
          int col_size = table_dims[1];

          if (ple_params.num_tables == 0) {
            ple_params.ple_table_element_type = table_tensor.ElementType();
            ple_params.ple_embedding_dim = col_size;
          } else {
            RET_CHECK_EQ((int)ple_params.ple_table_element_type,
                         (int)table_tensor.ElementType())
                << "All embedding tables must have the same element type";
            RET_CHECK_EQ(ple_params.ple_embedding_dim, col_size)
                << "All embedding tables must have the same embedding "
                   "dimension.";
          }

          auto weights = table_tensor.Weights();
          ple_params.ple_table_ptrs.push_back(weights.Bytes().data());

          HWQuantizationParams qp;
          qp.scales = nullptr;
          qp.is_per_channel = false;

          if (table_tensor.HasQuantization()) {
            auto q_type = table_tensor.QTypeId();
            if (q_type == kLiteRtQuantizationPerTensor) {
              auto q_params = table_tensor.PerTensorQuantization();
              ple_params.ple_per_tensor_scales.push_back(q_params.scale);
              qp.scales = &ple_params.ple_per_tensor_scales.back();
            } else if (q_type == kLiteRtQuantizationPerChannel) {
              auto q_params = table_tensor.PerChannelQuantization();
              qp.scales = q_params.scales;
              qp.is_per_channel = true;
            }
          }
          ple_params.ple_quant_params.push_back(qp);
          ple_params.num_tables++;
        }
        if (op.Code() == kLiteRtOpCodeTflMul) {
          auto inputs = op.Inputs();
          for (const auto& input : inputs) {
            if (input.HasWeights()) {
              auto type_info = input.RankedTensorType();
              if (type_info.HasValue() && type_info.Value().ElementType() ==
                                              litert::ElementType::Float32) {
                auto weights = input.Weights();
                const float* vals =
                    reinterpret_cast<const float*>(weights.Bytes().data());
                ple_params.mul_scale = vals[0];
              }
            }
          }
        }
      }

      auto outputs = subgraph.Outputs();
      RET_CHECK(!outputs.empty()) << "No outputs in subgraph";
      auto output_tensor = outputs[0];
      ple_params.output_type = output_tensor.ElementType();

      if (ple_params.output_type == litert::ElementType::Int16) {
        RET_CHECK(output_tensor.HasQuantization());
        auto q_params = output_tensor.PerTensorQuantization();
        ple_params.output_scale = q_params.scale;
        ple_params.final_zero_point = q_params.zero_point;
      }
    } else {
      LITERT_ASSIGN_OR_RETURN(
          embedder_per_layer_context,
          CreateEmbedderPerLayerContext(
              env, *embedder_per_layer_model, prefill_signatures,
              text_decoder_prefill_input_buffers,
              text_decoder_decode_input_buffers,
              text_decoder_verify_input_buffers, executor_settings));
    }
  }

  std::optional<::litert::TensorBuffer> prefill_embeddings_buffer;
  if (text_decoder_prefill_input_buffers.contains(
          TextDecoderSignatures::kInputEmbeddings)) {
    LITERT_ASSIGN_OR_RETURN(prefill_embeddings_buffer,
                            text_decoder_prefill_input_buffers
                                [TextDecoderSignatures::kInputEmbeddings]
                                    .Duplicate());
  }

  std::optional<::litert::TensorBuffer> decode_embeddings_buffer;
  if (text_decoder_decode_input_buffers.contains(
          TextDecoderSignatures::kInputEmbeddings)) {
    LITERT_ASSIGN_OR_RETURN(decode_embeddings_buffer,
                            text_decoder_decode_input_buffers
                                [TextDecoderSignatures::kInputEmbeddings]
                                    .Duplicate());
  }

  std::optional<::litert::TensorBuffer> verify_embeddings_buffer;
  if (text_decoder_verify_input_buffers.contains(
          TextDecoderSignatures::kInputEmbeddings)) {
    LITERT_ASSIGN_OR_RETURN(verify_embeddings_buffer,
                            text_decoder_verify_input_buffers
                                [TextDecoderSignatures::kInputEmbeddings]
                                    .Duplicate());
  }

  std::optional<::litert::TensorBuffer> prefill_ple_buffer;
  if (text_decoder_prefill_input_buffers.contains(kPerLayerEmbedderTensor)) {
    LITERT_ASSIGN_OR_RETURN(
        prefill_ple_buffer,
        text_decoder_prefill_input_buffers[kPerLayerEmbedderTensor]
            .Duplicate());
  }

  std::optional<::litert::TensorBuffer> decode_ple_buffer;
  if (text_decoder_decode_input_buffers.contains(kPerLayerEmbedderTensor)) {
    LITERT_ASSIGN_OR_RETURN(
        decode_ple_buffer,
        text_decoder_decode_input_buffers[kPerLayerEmbedderTensor].Duplicate());
  }

  std::optional<::litert::TensorBuffer> verify_ple_buffer;
  if (text_decoder_verify_input_buffers.contains(kPerLayerEmbedderTensor)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_ple_buffer,
        text_decoder_verify_input_buffers[kPerLayerEmbedderTensor].Duplicate());
  }

  return Create(
      std::move(embedding_lookup_manager), std::move(embedder_context),
      std::move(embedder_per_layer_context),
      /*per_layer_embedding_lookup_manager=*/nullptr, embedder_per_layer_model,
      std::move(ple_params), std::move(prefill_embeddings_buffer),
      std::move(decode_embeddings_buffer), std::move(verify_embeddings_buffer),
      std::move(prefill_ple_buffer), std::move(decode_ple_buffer),
      std::move(verify_ple_buffer));
}

absl::StatusOr<NpuEmbedder> NpuEmbedder::Create(
    std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager,
    std::optional<EmbedderContext> embedder_context,
    std::optional<EmbedderPerLayerContext> embedder_per_layer_context,
    std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup_manager,
    const litert::Model* embedder_per_layer_model, HWPleParams ple_params,
    std::optional<::litert::TensorBuffer> prefill_embeddings_buffer,
    std::optional<::litert::TensorBuffer> decode_embeddings_buffer,
    std::optional<::litert::TensorBuffer> verify_embeddings_buffer,
    std::optional<::litert::TensorBuffer> prefill_ple_buffer,
    std::optional<::litert::TensorBuffer> decode_ple_buffer,
    std::optional<::litert::TensorBuffer> verify_ple_buffer) {
  NpuEmbedder embedder;
  embedder.embedding_lookup_manager_ = std::move(embedding_lookup_manager);
  embedder.embedder_context_ = std::move(embedder_context);
  embedder.embedder_per_layer_context_ = std::move(embedder_per_layer_context);
  embedder.per_layer_embedding_lookup_manager_ =
      std::move(per_layer_embedding_lookup_manager);
  embedder.embedder_per_layer_model_ = embedder_per_layer_model;
  embedder.ple_params_ = std::move(ple_params);
  embedder.prefill_embeddings_buffer_ = std::move(prefill_embeddings_buffer);
  embedder.decode_embeddings_buffer_ = std::move(decode_embeddings_buffer);
  embedder.verify_embeddings_buffer_ = std::move(verify_embeddings_buffer);
  embedder.prefill_ple_buffer_ = std::move(prefill_ple_buffer);
  embedder.decode_ple_buffer_ = std::move(decode_ple_buffer);
  embedder.verify_ple_buffer_ = std::move(verify_ple_buffer);
  return embedder;
}

absl::Status NpuEmbedder::UpdateMultiModalEmbeddings(
    const ExecutorInputs& inputs) {
  if (embedding_lookup_manager_ != nullptr) {
    return embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs);
  }
  return absl::OkStatus();
}

absl::Status NpuEmbedder::CleanupMultiModalEmbeddings() {
  if (embedding_lookup_manager_ != nullptr) {
    return embedding_lookup_manager_->CleanupMultiModalEmbeddings();
  }
  return absl::OkStatus();
}

std::vector<float> NpuEmbedder::GetDefaultEmbeddingVector() const {
  if (embedding_lookup_manager_ != nullptr) {
    auto* text_lookup = embedding_lookup_manager_->GetTextEmbeddingLookup();
    if (text_lookup != nullptr) {
      return text_lookup->GetDefaultEmbeddingVector();
    }
  }
  return {};
}

absl::Status NpuEmbedder::RunPrefill(
    absl::string_view embedder_signature, const TokenData* pending_token,
    absl::Span<const int> processed_input_tokens, TokenData* last_input_token) {
  if (embedding_lookup_manager_ != nullptr) {
    RET_CHECK(prefill_embeddings_buffer_.has_value())
        << "Prefill embeddings buffer not available.";
    // Step 1: If a pending token from a previous prefill/turn is present, its
    // embedding was already looked up in the previous step. Copy it to slot 0.
    size_t offset = 0;
    if (pending_token != nullptr && !pending_token->embedding().empty()) {
      LITERT_RETURN_IF_ERROR(CopyEmbeddingToBuffer(
          pending_token->embedding(), *prefill_embeddings_buffer_));
      offset = 1;
    }

    // Step 2: Look up embeddings for the N processed input tokens directly into
    // the prefill embedding tensor buffer, starting at the calculated offset.
    LITERT_RETURN_IF_ERROR(embedding_lookup_manager_->LookupPrefill(
        processed_input_tokens, &*prefill_embeddings_buffer_, offset));

    // Step 3: Immediately look up and cache the embedding for the new holdback
    // / pending token so that it is available for subsequent prefill chunks or
    // the first decode step. Performing this lookup immediately maintains the
    // sequential order of the prompt inside EmbeddingLookupManager.
    if (last_input_token != nullptr) {
      LITERT_RETURN_IF_ERROR(embedding_lookup_manager_->LookupPrefill(
          last_input_token->id(), last_input_token->mutable_embedding()));
    }
  } else {
    RET_CHECK(embedder_context_.has_value())
        << "Embedder context not available for prefill embedder model.";
    {
      auto& in_buf =
          embedder_context_->inference_context
              .prefill_input_buffers[EmbedderSignatures::kEmbedderInput];
      LITERT_ASSIGN_OR_RETURN(auto prefill_input_size, in_buf.Size());
      LITERT_ASSIGN_OR_RETURN(
          auto in_lock, ::litert::TensorBufferScopedLock::Create(
                            in_buf, ::litert::TensorBuffer::LockMode::kWrite));
      auto* prefill_input_ptr = static_cast<int32_t*>(in_lock.second);
      std::memset(prefill_input_ptr, 0, prefill_input_size);
      const size_t max_tokens = prefill_input_size / sizeof(int32_t);
      size_t idx = 0;
      if (pending_token != nullptr && pending_token->id() != kInvalidTokenId &&
          idx < max_tokens) {
        prefill_input_ptr[idx++] = pending_token->id();
      }
      for (size_t i = 0; i < processed_input_tokens.size() && idx < max_tokens;
           ++i) {
        prefill_input_ptr[idx++] = processed_input_tokens[i];
      }
    }
    return RunPrefillEmbedder(embedder_signature);
  }
  return absl::OkStatus();
}

absl::Status NpuEmbedder::RunDecode(const TokenData& token) {
  // When using EmbeddingLookupManager (or for empty/invalid token IDs), the
  // token embedding was already retrieved via LookupDecode at the end of the
  // previous decode step or during prefill holdback. Copy it into the hardware
  // decode embeddings buffer.
  if (embedding_lookup_manager_ != nullptr || token.id() == kInvalidTokenId) {
    RET_CHECK(decode_embeddings_buffer_.has_value())
        << "Decode embeddings buffer not available.";
    return CopyEmbeddingToBuffer(token.embedding(), *decode_embeddings_buffer_);
  }
  // For compiled embedder models, set the token ID and invoke the compiled
  // model.
  RET_CHECK(embedder_context_.has_value())
      << "Embedder context not available for decode embedder model.";
  LITERT_RETURN_IF_ERROR(SetFirstElement(
      embedder_context_->inference_context
          .decode_input_buffers[EmbedderSignatures::kEmbedderInput],
      token.id()));
  return RunDecodeEmbedder();
}

absl::Status NpuEmbedder::RunPrefillEmbedder(absl::string_view signature) {
  RET_CHECK(embedder_context_.has_value())
      << "Embedder context not available for prefill embedder model.";
  auto res = embedder_context_->embedder_compiled_model.Run(
      signature, embedder_context_->inference_context.prefill_input_buffers,
      embedder_context_->inference_context.prefill_output_buffers);
  RET_CHECK(res) << "Failed to run embedder model: " << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuEmbedder::RunDecodeEmbedder() {
  RET_CHECK(embedder_context_.has_value())
      << "Embedder context not available for decode embedder model.";
  auto res = embedder_context_->embedder_compiled_model.Run(
      EmbedderSignatures::kDecodeEmbedder,
      embedder_context_->inference_context.decode_input_buffers,
      embedder_context_->inference_context.decode_output_buffers);
  RET_CHECK(res) << "Failed to run embedder model: " << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuEmbedder::LookupDecode(int32_t token_id,
                                       std::vector<float>& out_embedding) {
  if (embedding_lookup_manager_ == nullptr) {
    return absl::OkStatus();
  }
  if (!out_embedding.empty()) {
    return absl::OkStatus();
  }
  auto* text_lookup = embedding_lookup_manager_->GetTextEmbeddingLookup();
  RET_CHECK(text_lookup != nullptr)
      << "Text embedding lookup not available for decode.";
  out_embedding.resize(text_lookup->GetFloatsPerToken());
  return embedding_lookup_manager_->LookupDecode(token_id, out_embedding);
}

absl::Status NpuEmbedder::LookupDecode(TokenData* token) {
  if (token == nullptr || embedding_lookup_manager_ == nullptr) {
    return absl::OkStatus();
  }
  return LookupDecode(token->id(), token->mutable_embedding());
}

absl::Status NpuEmbedder::RunVerify(absl::Span<const int> verify_ids) {
  if (embedding_lookup_manager_ != nullptr) {
    RET_CHECK(verify_embeddings_buffer_.has_value())
        << "Verify embeddings buffer not available.";
    return embedding_lookup_manager_->LookupPrefill(
        verify_ids, &*verify_embeddings_buffer_, /*offset=*/0);
  }
  return RunVerifyEmbedder(verify_ids);
}

absl::Status NpuEmbedder::CopyEmbeddingToBuffer(
    absl::Span<const float> embedding,
    ::litert::TensorBuffer& destination_buffer) {
  LITERT_ASSIGN_OR_RETURN(
      auto lock,
      ::litert::TensorBufferScopedLock::Create(
          destination_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  float* ptr = static_cast<float*>(lock.second);
  RET_CHECK(!embedding.empty()) << "Token embedding is empty.";
  std::memcpy(ptr, embedding.data(), embedding.size() * sizeof(float));
  return absl::OkStatus();
}

absl::Status NpuEmbedder::RunVerifyEmbedder(absl::Span<const int> verify_ids) {
  RET_CHECK(embedder_context_.has_value())
      << "Embedder context not available for verify embedder model.";
  {
    auto& in_buf =
        embedder_context_->inference_context
            .verify_input_buffers[EmbedderSignatures::kEmbedderInput];
    LITERT_ASSIGN_OR_RETURN(
        auto lock, ::litert::TensorBufferScopedLock::Create(
                       in_buf, ::litert::TensorBuffer::LockMode::kWrite));
    auto* in_ptr = static_cast<int32_t*>(lock.second);
    for (size_t i = 0; i < verify_ids.size(); ++i) {
      in_ptr[i] = verify_ids[i];
    }
  }
  auto res = embedder_context_->embedder_compiled_model.Run(
      EmbedderSignatures::kVerifyEmbedder,
      embedder_context_->inference_context.verify_input_buffers,
      embedder_context_->inference_context.verify_output_buffers);
  RET_CHECK(res) << "Failed to run verify embedder model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuEmbedder::LookupHwPle(const int* token_ids, int num_tokens,
                                      void* output_buffer) const {
  return HWPerLayerEmbeddingLookup(
      token_ids, num_tokens, ple_params_.ple_table_ptrs.data(),
      ple_params_.ple_quant_params.data(), ple_params_.num_tables,
      ple_params_.ple_embedding_dim, output_buffer, ple_params_.output_type,
      ple_params_.ple_table_element_type, ple_params_.mul_scale,
      ple_params_.output_scale, ple_params_.final_zero_point);
}

absl::Status NpuEmbedder::RunPrefillPerLayer(
    absl::string_view signature, absl::Span<const int> tokens_to_embed) {
  if (ple_params_.use_hw_ple && !ple_params_.ple_table_ptrs.empty()) {
    RET_CHECK(prefill_ple_buffer_.has_value())
        << "Prefill PLE buffer not available.";
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        ::litert::TensorBufferScopedLock::Create(
            *prefill_ple_buffer_, ::litert::TensorBuffer::LockMode::kWrite));
    return LookupHwPle(tokens_to_embed.data(), tokens_to_embed.size(),
                       lock.second);
  } else if (embedder_per_layer_context_.has_value()) {
    {
      LITERT_ASSIGN_OR_RETURN(
          auto ple_input_lock,
          ::litert::TensorBufferScopedLock::Create(
              embedder_per_layer_context_->inference_context
                  .prefill_input_buffers
                      [EmbedderPerLayerSignatures::kEmbedderInput],
              ::litert::TensorBuffer::LockMode::kWrite));
      auto* input_ptr = static_cast<int32_t*>(ple_input_lock.second);
      for (size_t i = 0; i < tokens_to_embed.size(); ++i) {
        input_ptr[i] = tokens_to_embed[i] < 0 ? 0 : tokens_to_embed[i];
      }
    }
    auto res =
        embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
            signature,
            embedder_per_layer_context_->inference_context
                .prefill_input_buffers,
            embedder_per_layer_context_->inference_context
                .prefill_output_buffers);
    RET_CHECK(res) << "Failed to run embedder per layer model: "
                   << res.Error().Message();
  }
  return absl::OkStatus();
}

absl::Status NpuEmbedder::RunDecodePerLayer(int32_t token_id) {
  if (ple_params_.use_hw_ple && !ple_params_.ple_table_ptrs.empty()) {
    RET_CHECK(decode_ple_buffer_.has_value())
        << "Decode PLE buffer not available.";
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        ::litert::TensorBufferScopedLock::Create(
            *decode_ple_buffer_, ::litert::TensorBuffer::LockMode::kWrite));
    return LookupHwPle(&token_id, 1, lock.second);
  } else if (embedder_per_layer_context_.has_value()) {
    LITERT_RETURN_IF_ERROR(SetFirstElement(
        embedder_per_layer_context_->inference_context
            .decode_input_buffers[EmbedderPerLayerSignatures::kEmbedderInput],
        token_id));
    auto res =
        embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
            EmbedderPerLayerSignatures::kDecodeEmbedderPerLayer,
            embedder_per_layer_context_->inference_context.decode_input_buffers,
            embedder_per_layer_context_->inference_context
                .decode_output_buffers);
    RET_CHECK(res) << "Failed to run embedder per layer model: "
                   << res.Error().Message();
  }
  return absl::OkStatus();
}

absl::Status NpuEmbedder::RunVerifyPerLayer(absl::Span<const int> verify_ids) {
  if (ple_params_.use_hw_ple && !ple_params_.ple_table_ptrs.empty()) {
    RET_CHECK(verify_ple_buffer_.has_value())
        << "Verify PLE buffer not available.";
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        ::litert::TensorBufferScopedLock::Create(
            *verify_ple_buffer_, ::litert::TensorBuffer::LockMode::kWrite));
    return LookupHwPle(verify_ids.data(), verify_ids.size(), lock.second);
  } else if (embedder_per_layer_context_.has_value()) {
    {
      LITERT_ASSIGN_OR_RETURN(
          auto verify_ple_input_lock,
          ::litert::TensorBufferScopedLock::Create(
              embedder_per_layer_context_->inference_context
                  .verify_input_buffers
                      [EmbedderPerLayerSignatures::kEmbedderInput],
              ::litert::TensorBuffer::LockMode::kWrite));
      auto* input_ptr = static_cast<int32_t*>(verify_ple_input_lock.second);
      for (size_t i = 0; i < verify_ids.size(); ++i) {
        input_ptr[i] = verify_ids[i];
      }
    }
    auto res =
        embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
            EmbedderPerLayerSignatures::kVerifyEmbedderPerLayer,
            embedder_per_layer_context_->inference_context.verify_input_buffers,
            embedder_per_layer_context_->inference_context
                .verify_output_buffers);
    RET_CHECK(res) << "Failed to run embedder per layer model: "
                   << res.Error().Message();
  }
  return absl::OkStatus();
}

absl::Status NpuEmbedder::WriteDecodePleEmbeddings(
    absl::Span<const float> ple_embeddings) {
  RET_CHECK(decode_ple_buffer_.has_value())
      << "Decode PLE buffer not available.";
  return WritePleEmbeddings(*decode_ple_buffer_, ple_embeddings,
                            ple_params_.output_type, ple_params_.output_scale,
                            ple_params_.final_zero_point);
}

absl::Status NpuEmbedder::WriteAndPadPleEmbeddings(
    ::litert::Environment& env, absl::Span<const float> ple_embeddings) {
  RET_CHECK(prefill_ple_buffer_.has_value())
      << "Prefill PLE buffer not available.";
  return WriteAndPadPleEmbeddings(env, *prefill_ple_buffer_, ple_embeddings);
}

absl::Status NpuEmbedder::WriteAndPadPleEmbeddings(
    ::litert::Environment& env, ::litert::TensorBuffer& buffer,
    absl::Span<const float> ple_embeddings) {
  std::vector<float> default_ple_emb;
  if (per_layer_embedding_lookup_manager_ == nullptr &&
      embedder_per_layer_model_ != nullptr) {
    LITERT_ASSIGN_OR_RETURN(
        per_layer_embedding_lookup_manager_,
        EmbeddingLookupManager::Create(env, embedder_per_layer_model_, false));
  }
  if (per_layer_embedding_lookup_manager_ != nullptr) {
    auto* ple_lookup =
        per_layer_embedding_lookup_manager_->GetTextEmbeddingLookup();
    if (ple_lookup != nullptr) {
      default_ple_emb = ple_lookup->GetDefaultEmbeddingVector();
    }
  }

  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, buffer.TensorType());
  const auto& dims = tensor_type.Layout().Dimensions();
  if (dims.size() < 3) {
    return absl::InternalError(
        "Prefill per-layer embeddings tensor has unexpected shape.");
  }
  const size_t ple_dim =
      default_ple_emb.empty() ? dims[2] : default_ple_emb.size();
  size_t starting_token = ple_embeddings.size() / ple_dim;

  return ::litert::lm::WriteAndPadPleEmbeddings(
      buffer, ple_embeddings, ple_dim, starting_token, default_ple_emb,
      ple_params_.output_type, ple_params_.output_scale,
      ple_params_.final_zero_point);
}

// -----------------------------------------------------------------------------
// NpuAuxiliaryContext Implementation
// -----------------------------------------------------------------------------

absl::StatusOr<NpuAuxiliaryContext> NpuAuxiliaryContext::Create(
    ::litert::Environment& env, const litert::Model& npu_auxiliary_model,
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtNpuOptions(settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel npu_auxiliary_compiled_model,
      CompiledModelWrapper::Create(env, npu_auxiliary_model.Get(), options));
  return NpuAuxiliaryContext(std::move(npu_auxiliary_compiled_model));
}

absl::StatusOr<NpuAuxiliaryContext> CreateNpuAuxiliaryContext(
    ::litert::Environment& env, const litert::Model& npu_auxiliary_model,
    const LlmExecutorSettings& settings) {
  return NpuAuxiliaryContext::Create(env, npu_auxiliary_model, settings);
}

absl::StatusOr<DrafterContext> DrafterContext::Create(
    ::litert::Environment& env, const litert::Model& mtp_drafter_model,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        drafter_input_kv_cache_buffers,
    ::litert::TensorBuffer& output_activations_buffers) {
  LITERT_ASSIGN_OR_RETURN(
      auto mtp_compiled_model,
      CompiledModelWrapper::Create(env, mtp_drafter_model.Get(),
                                   litert::HwAccelerators::kCpu));
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mtp_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mtp_output_buffers;

  // Create input and output buffers for the MTP drafter.
  auto mtp_signature =
      mtp_compiled_model.FindSignature(MtpSignatures::kMtpDrafter);
  for (const auto& input_name : mtp_signature->InputNames()) {
    // Reuse kv cache buffers from main model
    if (absl::StartsWith(input_name, kKvCacheKRootName) ||
        absl::StartsWith(input_name, kKvCacheVRootName)) {
      LITERT_ASSIGN_OR_RETURN(
          mtp_input_buffers[input_name],
          drafter_input_kv_cache_buffers[input_name].Duplicate());
    } else {
      LITERT_ASSIGN_OR_RETURN(mtp_input_buffers[input_name],
                              mtp_compiled_model.CreateInputBuffer(
                                  MtpSignatures::kMtpDrafter, input_name));
      mtp_input_buffers[input_name].Clear();
    }
  }
  for (const auto& output_name : mtp_signature->OutputNames()) {
    {
      LITERT_ASSIGN_OR_RETURN(mtp_output_buffers[output_name],
                              mtp_compiled_model.CreateOutputBuffer(
                                  MtpSignatures::kMtpDrafter, output_name));
    }
  }
  return DrafterContext(std::move(mtp_compiled_model),
                        std::move(mtp_input_buffers),
                        std::move(mtp_output_buffers));
}

absl::Status DrafterContext::SetInputPos(int32_t pos) {
  auto it = mtp_input_buffers.find(MtpSignatures::kInputPos);
  if (it == mtp_input_buffers.end()) {
    return absl::NotFoundError("Drafter input pos buffer not found.");
  }
  LITERT_ASSIGN_OR_RETURN(
      auto lock, ::litert::TensorBufferScopedLock::Create(
                     it->second, ::litert::TensorBuffer::LockMode::kWrite));
  static_cast<int32_t*>(lock.second)[0] = pos;
  return absl::OkStatus();
}

bool IsNpuSyncWorkaroundEnabled() {
  static bool enabled = []() {
    const char* env = std::getenv("LITERT_LM_SYNC_EXECUTION");
    if (env == nullptr) {
      // Default is async (false)
      return false;
    }
    std::string env_str(env);
    return env_str == "1" || env_str == "true" || env_str == "sync";
  }();
  return enabled;
}

absl::Status Fill(TensorBuffer& tensor_buffer, uint16_t value) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_buffer_type,
                          tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          tensor_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          tensor_buffer_type.Layout().NumElements());
  if (tensor_buffer_type.ElementType() == ::litert::ElementType::Float32) {
    float* ptr = static_cast<float*>(lock_and_addr.second);
    float float_value = static_cast<float>(value);
    for (int i = 0; i < num_elements; ++i) {
      ptr[i] = float_value;
    }
  } else if (tensor_buffer_type.ElementType() == ::litert::ElementType::Int16) {
    int16_t* ptr = static_cast<int16_t*>(lock_and_addr.second);
    int16_t int16_value = static_cast<int16_t>(value);
    for (int i = 0; i < num_elements; ++i) {
      ptr[i] = int16_value;
    }
  } else if (tensor_buffer_type.ElementType() ==
             ::litert::ElementType::UInt16) {
    uint16_t* ptr = static_cast<uint16_t*>(lock_and_addr.second);
    for (int i = 0; i < num_elements; ++i) {
      ptr[i] = value;
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported tensor element type for Fill: ",
                     tensor_buffer_type.ElementType()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<uint8_t>> CopyRawBytesFromTensorBuffer(
    const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_bytes, buffer.Size());
  if (tensor_type.ElementType() == ::litert::ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto buf, CopyFromTensorBuffer<float>(buffer));
    std::vector<uint8_t> res(num_bytes);
    std::memcpy(res.data(), buf.data(), num_bytes);
    return res;
  } else if (tensor_type.ElementType() == ::litert::ElementType::Int16) {
    LITERT_ASSIGN_OR_RETURN(auto buf, CopyFromTensorBuffer<int16_t>(buffer));
    std::vector<uint8_t> res(num_bytes);
    std::memcpy(res.data(), buf.data(), num_bytes);
    return res;
  } else if (tensor_type.ElementType() == ::litert::ElementType::Int8) {
    LITERT_ASSIGN_OR_RETURN(auto buf, CopyFromTensorBuffer<int8_t>(buffer));
    std::vector<uint8_t> res(num_bytes);
    std::memcpy(res.data(), buf.data(), num_bytes);
    return res;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported tensor element type for copying: ",
                     tensor_type.ElementType()));
  }
}

bool DetectIsSwa(const absl::flat_hash_map<absl::string_view, TensorBuffer>&
                     input_kv_cache_buffers) {
  std::set<int64_t> cache_seqs;
  for (const auto& [name, buffer] : input_kv_cache_buffers) {
    if (name.starts_with(kKvCacheKRootName) ||
        name.starts_with(kKvCacheVRootName) ||
        name.starts_with(kKvCacheCRootName)) {
      auto tensor_type_expected = buffer.TensorType();
      if (tensor_type_expected.HasValue()) {
        auto dims = tensor_type_expected->Layout().Dimensions();
        int rank = dims.size();
        if (rank >= 2) {
          int last_dim = dims[rank - 1];
          int second_last_dim = dims[rank - 2];
          int64_t cache_seq = std::max(last_dim, second_last_dim);
          cache_seqs.insert(cache_seq);
        }
      }
    }
  }
  return cache_seqs.size() > 1;
}

std::string PrefillSig(absl::string_view base, int prefill_size) {
  return absl::StrCat(base, "_", prefill_size);
}

absl::StatusOr<int> DetectPrefillSize(const litert::Model& transformer_model) {
  const std::string prefix = absl::StrCat(kPrefillSignatureBase, "_");
  auto signatures = transformer_model.GetSignatures();
  if (signatures) {
    for (const auto& signature : *signatures) {
      absl::string_view key = signature.Key();
      if (!absl::StartsWith(key, prefix)) {
        continue;
      }
      // Only the bare LLM prefill signature has a purely numeric suffix; the
      // auxiliary signatures (prefill_mask_..., prefill_rope_..., etc.) carry a
      // non-numeric segment after the "prefill_" prefix.
      absl::string_view suffix = key.substr(prefix.size());
      int prefill_size = 0;
      if (absl::SimpleAtoi(suffix, &prefill_size) && prefill_size > 0) {
        return prefill_size;
      }
    }
  }
  // Fallback: probe a list of common prefill sizes.
  for (int candidate : {256, 128, 512, 1024, 64}) {
    if (transformer_model
            .FindSignature(PrefillSig(kPrefillSignatureBase, candidate))
            .HasValue()) {
      return candidate;
    }
  }
  return absl::NotFoundError(
      "Could not detect a prefill signature (e.g. \"prefill_128\") in the "
      "transformer model.");
}

ResolvedPrefillSignatures BuildResolvedPrefillSignatures(int prefill_size) {
  return ResolvedPrefillSignatures{
      .size = prefill_size,
      .prefill = PrefillSig(kPrefillSignatureBase, prefill_size),
      .embedder = PrefillSig(kPrefillEmbedderBase, prefill_size),
      .embedder_per_layer =
          PrefillSig(kPrefillEmbedderPerLayerBase, prefill_size),
      .mask = PrefillSig(kPrefillMaskBase, prefill_size),
      .rope = PrefillSig(kPrefillRopeBase, prefill_size),
      .cache_update = PrefillSig(kPrefillCacheUpdateBase, prefill_size)};
}

litert::Expected<bool> HasPerLayerEmbedder(
    const litert::Model& transformer_model,
    absl::string_view prefill_signature) {
  LITERT_ASSIGN_OR_RETURN(
      auto input_names,
      transformer_model.GetSignatureInputNames(prefill_signature));
  for (auto input_name : input_names) {
    if (kPerLayerEmbedderTensor == input_name) {
      return true;
    }
  }
  return false;
}


}  // namespace litert::lm
