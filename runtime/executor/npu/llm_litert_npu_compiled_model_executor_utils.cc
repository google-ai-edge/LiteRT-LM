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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
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
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep NOLINT

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

absl::Status DrafterContext::UpdateKVCacheBuffers(
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        new_input_kv_cache_buffers) {
  for (const auto& [name, buf] : new_input_kv_cache_buffers) {
    if (mtp_input_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(mtp_input_buffers[name], buf.Duplicate());
    }
  }
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
      // Only the bare LLM prefill signature has a purely numeric suffix or
      // numeric_cache_<N> suffix (excluding auxiliary signatures).
      if (absl::StartsWith(key, kPrefillMaskBase) ||
          absl::StartsWith(key, kPrefillRopeBase) ||
          absl::StartsWith(key, kPrefillCacheUpdateBase) ||
          absl::StartsWith(key, kPrefillEmbedderBase) ||
          absl::StartsWith(key, kPrefillEmbedderPerLayerBase)) {
        continue;
      }
      absl::string_view suffix = key.substr(prefix.size());
      size_t cache_pos = suffix.find("_cache_");
      if (cache_pos != absl::string_view::npos) {
        suffix = suffix.substr(0, cache_pos);
      }
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

absl::StatusOr<std::vector<int>> DetectSupportedContextSizes(
    const litert::Model& transformer_model) {
  std::set<int> context_sizes_set;
  auto signatures = transformer_model.GetSignatures();
  if (signatures) {
    for (const auto& signature : *signatures) {
      absl::string_view key = signature.Key();
      size_t cache_pos = key.find("_cache_");
      if (cache_pos != absl::string_view::npos) {
        absl::string_view suffix = key.substr(cache_pos + 7);
        // Suffix should be pure integer (e.g. "640")
        int ctx_size = 0;
        if (absl::SimpleAtoi(suffix, &ctx_size) && ctx_size > 0) {
          context_sizes_set.insert(ctx_size);
        }
      }
    }
  }
  return std::vector<int>(context_sizes_set.begin(), context_sizes_set.end());
}

ResolvedPrefillSignatures BuildResolvedPrefillSignatures(int prefill_size,
                                                         int context_size) {
  if (context_size <= 0) {
    return ResolvedPrefillSignatures{
        .size = prefill_size,
        .context_size = 0,
        .prefill = PrefillSig(kPrefillSignatureBase, prefill_size),
        .embedder = PrefillSig(kPrefillEmbedderBase, prefill_size),
        .embedder_per_layer =
            PrefillSig(kPrefillEmbedderPerLayerBase, prefill_size),
        .mask = PrefillSig(kPrefillMaskBase, prefill_size),
        .rope = PrefillSig(kPrefillRopeBase, prefill_size),
        .cache_update = PrefillSig(kPrefillCacheUpdateBase, prefill_size)};
  }
  return ResolvedPrefillSignatures{
      .size = prefill_size,
      .context_size = context_size,
      .prefill = absl::StrCat(kPrefillSignatureBase, "_", prefill_size,
                              "_cache_", context_size),
      .embedder = PrefillSig(kPrefillEmbedderBase, prefill_size),
      .embedder_per_layer =
          PrefillSig(kPrefillEmbedderPerLayerBase, prefill_size),
      .mask = absl::StrCat(kPrefillMaskBase, "_", prefill_size, "_cache_",
                           context_size),
      .rope = PrefillSig(kPrefillRopeBase, prefill_size),
      .cache_update = absl::StrCat(kPrefillCacheUpdateBase, "_", prefill_size,
                                   "_cache_", context_size)};
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

ResolvedAuxiliarySignatures BuildResolvedDecodeAuxiliarySignatures(
    const ::litert::CompiledModel& aux_model, int context_size) {
  ResolvedAuxiliarySignatures sigs;

  // 1. Mask signature
  std::string mask_cand =
      absl::StrCat(kDecodeMaskBase, "_cache_", context_size);
  if (context_size > 0 && aux_model.FindSignature(mask_cand)) {
    sigs.mask = mask_cand;
  } else {
    sigs.mask = std::string(kDecodeMaskBase);
  }

  // 2. RoPE signature (single signature, does not vary by context size)
  if (aux_model.FindSignature(kDecodeRopeBase)) {
    sigs.rope = std::string(kDecodeRopeBase);
  } else {
    sigs.rope = "rope";
  }

  // 3. Cache Update signature
  std::string cache_cand =
      absl::StrCat(kDecodeCacheUpdateBase, "_cache_", context_size);
  if (context_size > 0 && aux_model.FindSignature(cache_cand)) {
    sigs.cache_update = cache_cand;
  } else if (aux_model.FindSignature(kDecodeCacheUpdateBase)) {
    sigs.cache_update = std::string(kDecodeCacheUpdateBase);
  } else {
    sigs.cache_update = "cache_update";
  }

  return sigs;
}

ResolvedAuxiliarySignatures BuildResolvedVerifyAuxiliarySignatures(
    const ::litert::CompiledModel& aux_model, int context_size) {
  ResolvedAuxiliarySignatures sigs;

  std::string mask_cand =
      absl::StrCat(kVerifyMaskBase, "_cache_", context_size);
  sigs.mask = (context_size > 0 && aux_model.FindSignature(mask_cand))
                  ? mask_cand
                  : std::string(kVerifyMaskBase);

  // RoPE signature (single signature, does not vary by context size)
  sigs.rope = std::string(kVerifyRopeBase);

  std::string cache_cand =
      absl::StrCat(kVerifyCacheUpdateBase, "_cache_", context_size);
  sigs.cache_update = (context_size > 0 && aux_model.FindSignature(cache_cand))
                          ? cache_cand
                          : std::string(kVerifyCacheUpdateBase);

  return sigs;
}

absl::StatusOr<::litert::TensorBuffer> CreateAliasBuffer(
    const ::litert::Environment& env,
    const ::litert::TensorBuffer& source_buffer,
    const ::litert::RankedTensorType& target_type) {
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, source_buffer.BufferType());
  LITERT_ASSIGN_OR_RETURN(size_t source_bytes, source_buffer.Size());
  LITERT_ASSIGN_OR_RETURN(size_t target_bytes, target_type.Bytes());

  if (target_bytes > source_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot create alias buffer: target bytes (", target_bytes,
                     ") exceeds source bytes (", source_bytes, ")"));
  }

  if (buffer_type == ::litert::TensorBufferType::kHostMemory) {
    auto env_holder = env.GetHolder();
    void* host_mem_addr = nullptr;
    if (env_holder.runtime->GetTensorBufferHostMemory(
            source_buffer.Get(), &host_mem_addr) == kLiteRtStatusOk) {
      LITERT_ASSIGN_OR_RETURN(
          auto buf, ::litert::TensorBuffer::CreateFromHostMemory(
                        env, target_type, host_mem_addr, target_bytes));
      return std::move(buf);
    }
  }

#if LITERT_HAS_AHWB_SUPPORT
  if (buffer_type == ::litert::TensorBufferType::kAhwb) {
    auto ahwb_res = source_buffer.GetAhwb();
    if (ahwb_res.HasValue()) {
      LITERT_ASSIGN_OR_RETURN(
          auto buf, ::litert::TensorBuffer::CreateFromAhwb(
                        env, target_type, ahwb_res.Value(), /*ahwb_offset=*/0));
      return std::move(buf);
    }
  }
#endif

#if LITERT_HAS_DMABUF_SUPPORT
  if (buffer_type == ::litert::TensorBufferType::kDmaBuf) {
    auto dma_res = source_buffer.GetDmaBuf();
    if (dma_res.HasValue()) {
      LITERT_ASSIGN_OR_RETURN(auto buf,
                              ::litert::TensorBuffer::CreateFromDmaBufBuffer(
                                  env, target_type, dma_res->addr, dma_res->fd,
                                  source_bytes, /*dmabuf_buffer_offset=*/0));
      return std::move(buf);
    }
  }
#endif

#if LITERT_HAS_ION_SUPPORT
  if (buffer_type == ::litert::TensorBufferType::kIon) {
    auto ion_res = source_buffer.GetIonBuf();
    if (ion_res.HasValue()) {
      LITERT_ASSIGN_OR_RETURN(auto buf,
                              ::litert::TensorBuffer::CreateFromIonBuffer(
                                  env, target_type, ion_res->addr, ion_res->fd,
                                  source_bytes, /*ion_buffer_offset=*/0));
      return std::move(buf);
    }
  }
#endif

#if LITERT_HAS_FASTRPC_SUPPORT
  if (buffer_type == ::litert::TensorBufferType::kFastRpc) {
    auto fastrpc_res = source_buffer.GetFastRpcBuf();
    if (fastrpc_res.HasValue()) {
      LITERT_ASSIGN_OR_RETURN(
          auto buf, ::litert::TensorBuffer::CreateFromFastRpcBuffer(
                        env, target_type, fastrpc_res->addr, fastrpc_res->fd,
                        source_bytes, /*fastrpc_buffer_offset=*/0));
      return std::move(buf);
    }
  }
#endif

  return absl::InternalError(absl::StrCat(
      "Unsupported tensor buffer type for cross-platform aliasing: ",
      static_cast<int>(buffer_type)));
}

}  // namespace litert::lm
