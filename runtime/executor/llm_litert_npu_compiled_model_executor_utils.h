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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Formats the first N elements of a span as a comma-separated list inside
// brackets, e.g., "[1, 2, 3, ...]".
template <typename T>
std::string FormatFirstN(absl::Span<const T> span, size_t n = 10) {
  return absl::StrFormat("[%s%s]", absl::StrJoin(span.subspan(0, n), ", "),
                         span.size() > n ? ", ..." : "");
}

// Quantizes a float value to a quantized type T.
template <typename T>
T Quantize(float value, float scale, int32_t zero_point) {
  static_assert(std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t>,
                "Unsupported quantization type.");
  int32_t qval = std::round(value / scale) + zero_point;
  return static_cast<T>(
      std::clamp(qval, static_cast<int32_t>(std::numeric_limits<T>::min()),
                 static_cast<int32_t>(std::numeric_limits<T>::max())));
}
#if defined(__ANDROID__) && defined(__ARM_NEON)
int FindMaxIndexFloatNeon(const float* data, int size);
int FindMaxIndexInt16Neon(const int16_t* data, int size);
int FindMaxIndexInt8Neon(const int8_t* data, int size);
#endif
#if defined(__x86_64__) || defined(_M_X64)
int FindMaxIndexSse2Float(const float* data, int size);
int FindMaxIndexSse2Int16(const int16_t* data, int size);
int FindMaxIndexSse2Int8(const int8_t* data, int size);
#endif

// Generic function to find the index of the maximum value in a TensorBuffer.
// Uses NEON optimizations if available.
template <typename T>
absl::StatusOr<int> FindMaxIndex(::litert::TensorBuffer& decoded_logits,
                                 bool use_neon_sampling) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, decoded_logits.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          tensor_type.Layout().NumElements());
  if (num_elements == 0) {
    return absl::InvalidArgumentError("Logits buffer is empty.");
  }
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          const_cast<::litert::TensorBuffer&>(decoded_logits),
          ::litert::TensorBuffer::LockMode::kRead));
  const T* data = static_cast<const T*>(lock_and_addr.second);

  LITERT_ASSIGN_OR_RETURN(size_t size, tensor_type.Layout().NumElements());

  if (size == 0) {
    return absl::InvalidArgumentError("Logits buffer is empty.");
  }

#if defined(__ANDROID__) && defined(__ARM_NEON)
  if (use_neon_sampling) {
    if constexpr (std::is_same_v<T, float>) {
      return FindMaxIndexFloatNeon(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return FindMaxIndexInt16Neon(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return FindMaxIndexInt8Neon(data, static_cast<int>(size));
    }
  }
#endif
#if defined(__x86_64__) || defined(_M_X64)
  if (use_neon_sampling) {
    if constexpr (std::is_same_v<T, float>) {
      return FindMaxIndexSse2Float(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return FindMaxIndexSse2Int16(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return FindMaxIndexSse2Int8(data, static_cast<int>(size));
    }
  }
#endif

  int max_index = 0;
  T max_value = data[0];
  for (size_t i = 1; i < num_elements; ++i) {
    if (data[i] > max_value) {
      max_value = data[i];
      max_index = static_cast<int>(i);
    }
  }
  return max_index;
}

// Applies greedy sampling (argmax) to the decoded logits.
absl::StatusOr<int> ApplyGreedySampling(::litert::TensorBuffer& decoded_logits,
                                        bool use_neon_sampling);

struct HWQuantParams {
  float scale = 1.0f;
  int64_t zero_point = 0;
};

// Performs manual KV cache update.
absl::Status HWKVCacheUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& out_buffers,
    const absl::flat_hash_map<absl::string_view, HWQuantParams>& quant_params =
        {},
    bool enable_swa = false);

// Performs manual attention mask update.
absl::Status HWMaskUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers);

struct HWQuantizationParams {
  const float* scales;
  bool is_per_channel;
};

// Performs manual per-layer embedding lookup.
absl::Status HWPerLayerEmbeddingLookup(
    const int* token_ids, int num_tokens, const uint8_t* const* table_ptrs,
    const HWQuantizationParams* quant_params, int num_tables,
    int ple_embedding_dim, void* output_buffer, litert::ElementType output_type,
    litert::ElementType ple_table_element_type, float mul_scale = 1.0f,
    float output_scale = 1.0f, int32_t final_zero_point = 0);

// Dequantize logits to float32.
absl::Status DequantizeLogits(const ::litert::TensorBuffer& src,
                              ::litert::TensorBuffer& dst, float scale,
                              int32_t zero_point, bool should_dump);

// Writes PLE embeddings to the buffer, quantizing them to Int16 if needed.
absl::Status WritePleEmbeddings(::litert::TensorBuffer& buffer,
                                absl::Span<const float> ple_embeddings,
                                litert::ElementType output_type,
                                float final_scale, int32_t final_zero_point);

// Writes PLE embeddings to the buffer, quantizing them to Int16 if needed,
// and pads the remaining buffer space with a default embedding.
absl::Status WriteAndPadPleEmbeddings(::litert::TensorBuffer& buffer,
                                      absl::Span<const float> ple_embeddings,
                                      size_t ple_dim, size_t seq_pos_size,
                                      const std::vector<float>& default_ple_emb,
                                      litert::ElementType output_type,
                                      float final_scale,
                                      int32_t final_zero_point);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_
