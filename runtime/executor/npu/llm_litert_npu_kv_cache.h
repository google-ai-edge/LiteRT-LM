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
#include "runtime/proto/executor_metadata.pb.h"

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
    const NpuModelGeometry* geometry,
    const absl::flat_hash_map<absl::string_view, HWQuantParams>& quant_params =
        {});

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
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params,
      const NpuModelGeometry* geometry, int64_t kv_cache_init_value = 0);

  static absl::StatusOr<NpuKVCache> CreateForTest(
      KVCacheUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext cache_update_context,
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params,
      const NpuModelGeometry* geometry, int64_t kv_cache_init_value = 0);

  void SetCompiledModel(const ::litert::CompiledModel* compiled_model) {
    compiled_model_ = compiled_model;
  }

  void SetGeometry(const NpuModelGeometry* geometry) { geometry_ = geometry; }
  const NpuModelGeometry* GetGeometry() const { return geometry_; }

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
  //   guaranteed safe (copies in reverse order).
  // - Copies all buffers present in `src_buffers` that start with "kv_cache_k",
  //   "kv_cache_v", or "kv_cache_c".
  // - Requirement: The caller must ensure `dst` capacity >= `src` capacity.
  // - Padding: Newly exposed strided slots [active_seq_len, dst_capacity) are
  //   cleanly reset to `kv_cache_init_value_` (e.g. quantization zero-point) to
  //   overwrite stale leftover data from adjacent slices.
  // - The sequence axis of each buffer is taken from
  //   `NpuModelGeometry::kv_buffer_info`. Buffers the geometry does not
  //   describe fall back to inferring the axis from the `src`/`dst` shape
  //   difference (see `CopySingleKVCacheBuffer`).
  absl::Status CopyKVCache(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          src_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          dst_buffers,
      int active_seq_len);

  // Updates the internal cache update context's buffer mappings to point to
  // the new context group's KV cache and slice buffers.
  absl::Status UpdateKVCacheBuffers(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          verify_output_kv_cache_slice_buffers);

  // Migrates a single KV cache buffer from `src` to `dst`.
  //
  // `sequence_axis` is the axis to migrate along and should come from
  // `NpuModelGeometry::kv_buffer_info`. Pass -1 when it is unknown, in which
  // case it is inferred from the `src`/`dst` shape difference; that inference
  // only works if the buffer was actually resized, so identically shaped
  // buffers are then assumed to alias and are skipped.
  //
  // With a known axis the behavior is:
  // - `dst` larger than `src`: copy [0, active_seq_len) and reset the newly
  //   exposed tail to `kv_cache_init_value`.
  // - Same extent, same allocation (aliased context groups): no-op.
  // - Same extent, distinct allocations (e.g. a fixed-capacity layer during a
  //   dynamic KV cache grow): copy the buffer verbatim, so that content which
  //   does not live in a [0, active_seq_len) prefix, such as wrapped
  //   ring-buffer entries, survives the migration.
  static absl::Status CopySingleKVCacheBuffer(const ::litert::TensorBuffer& src,
                                              ::litert::TensorBuffer& dst,
                                              int active_seq_len,
                                              int64_t kv_cache_init_value = 0,
                                              int sequence_axis = -1);

  // --- Accessors ---
  KVCacheUpdateMethod GetMethod() const { return method_; }
  const InferenceContext& Context() const { return cache_update_context_; }
  InferenceContext ReleaseContext() { return std::move(cache_update_context_); }

 private:
  explicit NpuKVCache(
      KVCacheUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext cache_update_context,
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params,
      const NpuModelGeometry* geometry, int64_t kv_cache_init_value = 0)
      : method_(method),
        compiled_model_(compiled_model),
        cache_update_context_(std::move(cache_update_context)),
        kv_quant_params_(std::move(kv_quant_params)),
        geometry_(geometry),
        kv_cache_init_value_(kv_cache_init_value) {}

  KVCacheUpdateMethod method_ = KVCacheUpdateMethod::kModel;
  const ::litert::CompiledModel* compiled_model_ = nullptr;
  InferenceContext cache_update_context_;
  absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params_;
  const NpuModelGeometry* geometry_ = nullptr;
  int64_t kv_cache_init_value_ = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_KV_CACHE_H_
