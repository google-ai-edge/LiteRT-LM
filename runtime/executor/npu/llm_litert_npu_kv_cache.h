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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_KV_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_KV_CACHE_H_

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"

namespace litert::lm {

// Extracts the KV cache init value from ModelResources metadata if available.
int64_t GetKvCacheInitValue(ModelResources& resources);

// Fills a KV cache TensorBuffer with the specified initialization value.
absl::Status FillKVCacheBuffer(::litert::TensorBuffer& buffer,
                               int64_t init_value);

// Clears all KV cache buffers in the map with the specified initialization
// value.
absl::Status ClearKVCacheBuffers(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers,
    int64_t init_value = 0);

// Performs manual KV cache update (CPU fallback).
absl::Status HWKVCacheUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& out_buffers,
    const absl::flat_hash_map<absl::string_view, HWQuantParams>& quant_params =
        {},
    bool enable_swa = false);

enum class KVCacheUpdateMethod {
  kModel,
  kWH,
};

struct CacheUpdateSignatures {
  static constexpr absl::string_view kDecodeCacheUpdate = "decode_cache_update";
  static constexpr absl::string_view kVerifyCacheUpdate = "verify_cache_update";
  static constexpr absl::string_view kInputPos = "input_pos";
  static constexpr absl::string_view kInputValidMask = "valid_mask";
};

// =============================================================================
// NpuKVCache Usage Guide:
//
// 1. Regular Prefill:
//    cache_.SetPrefillPositions(seq_positions);
//    cache_.RunPrefill(prefill_signature);
//
// 2. Regular Single-Token Decode:
//    cache_.SetDecodePosition(current_step);
//    cache_.RunDecode();
//
// 3. MTP Speculative Decoding - Verification:
//    cache_.CommitVerifiedKVCache(start_step);
//
// 4. Dynamic Context Migration:
//    When switching from a smaller context size to a larger context size:
//    a) Migrate existing active tokens to the larger stride in-place:
//       cache_.CopyKVCache(old_group.input_kv_cache_buffers,
//                          new_group.input_kv_cache_buffers,
//                          current_step);
//    b) Rebind cache update input/output buffers to the new context group:
//       cache_.UpdateKVCacheBuffers(
//           new_group.input_kv_cache_buffers,
//           new_group.text_decoder_inference_context.prefill_output_buffers,
//           new_group.text_decoder_inference_context.decode_output_buffers,
//           new_group.text_decoder_inference_context.verify_output_buffers);
// =============================================================================
class NpuKVCache {
 public:
  NpuKVCache() = default;
  NpuKVCache(const NpuKVCache&) = delete;
  NpuKVCache& operator=(const NpuKVCache&) = delete;
  NpuKVCache(NpuKVCache&&) = default;
  NpuKVCache& operator=(NpuKVCache&&) = default;

  // --- Lifecycle & Creation ---
  static absl::StatusOr<NpuKVCache> Create(
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
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params =
          {},
      bool has_sliding_window_attention = false,
      int64_t kv_cache_init_value = 0);

  static absl::StatusOr<NpuKVCache> CreateForTest(
      KVCacheUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext cache_update_context,
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params =
          {},
      bool has_sliding_window_attention = false,
      int64_t kv_cache_init_value = 0);

  void SetCompiledModel(const ::litert::CompiledModel* compiled_model) {
    compiled_model_ = compiled_model;
  }

  // --- Stage 1: Prefill ---
  absl::Status SetPrefillPositions(absl::Span<const int32_t> seq_positions);
  absl::Status RunPrefill(absl::string_view signature = "");

  // --- Stage 2: Decode ---
  absl::Status SetDecodePosition(int32_t step);
  absl::Status RunDecode(absl::string_view signature = "");

  // --- Stage 3: Speculative Decoding (Verify Commit) ---
  absl::Status SetVerifyPos(int start_step);
  absl::Status CommitVerifiedKVCache(int start_step,
                                     absl::string_view signature = "");

  // --- Stage 4: Dynamic Context Migration ---
  // Copies the first `active_seq_len` tokens from `src_buffers` into
  // `dst_buffers` across all KV cache tensors (K, V, and C).
  //
  // Contract & Invariants:
  // - Copies active range [0, active_seq_len) along the sequence dimension.
  // - Supports aliased memory buffers where `src` and `dst` share the exact
  //   same underlying physical memory allocation. In-place expansion is
  //   guaranteed safe by copying outer slices in reverse order (back-to-front).
  // - Requirement: The caller must ensure `dst` capacity >= `src` capacity.
  // - Padding: Newly exposed strided slots [active_seq_len, dst_capacity) are
  //   cleanly reset to `kv_cache_init_value_` (e.g. quantization zero-point) to
  //   overwrite stale leftover data from adjacent slices.
  absl::Status CopyKVCache(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          src_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          dst_buffers,
      int active_seq_len);

  // Updates the internal cache update runner's input and output tensor buffers
  // to point to the newly active context group's KV cache buffers when
  // switching from one context length to the next.
  absl::Status UpdateKVCacheBuffers(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers = {},
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers = {},
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          verify_output_kv_cache_slice_buffers = {});

  static absl::Status CopySingleKVCacheBuffer(const ::litert::TensorBuffer& src,
                                              ::litert::TensorBuffer& dst,
                                              int active_seq_len,
                                              int64_t kv_cache_init_value = 0);

  // --- Accessors ---
  KVCacheUpdateMethod GetMethod() const { return method_; }
  const InferenceContext& Context() const { return cache_update_context_; }
  InferenceContext ReleaseContext() { return std::move(cache_update_context_); }

 private:
  explicit NpuKVCache(
      KVCacheUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext cache_update_context,
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params,
      bool has_sliding_window_attention, int64_t kv_cache_init_value = 0)
      : method_(method),
        compiled_model_(compiled_model),
        cache_update_context_(std::move(cache_update_context)),
        kv_quant_params_(std::move(kv_quant_params)),
        has_sliding_window_attention_(has_sliding_window_attention),
        kv_cache_init_value_(kv_cache_init_value) {}

  KVCacheUpdateMethod method_ = KVCacheUpdateMethod::kModel;
  const ::litert::CompiledModel* compiled_model_ = nullptr;
  InferenceContext cache_update_context_;
  absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params_;
  bool has_sliding_window_attention_ = false;
  int64_t kv_cache_init_value_ = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_KV_CACHE_H_
