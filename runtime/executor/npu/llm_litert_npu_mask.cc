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

#include "runtime/executor/npu/llm_litert_npu_mask.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

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

// Fills a single attention mask (either local or global) for a given
// time step.
//
// Args:
//   mask: Pointer to the mask buffer to be filled (size seq_q * seq_k).
//   seq_q: Query sequence length (number of queries/current batch size).
//   seq_k: Total key sequence length (capacity + batch_size).
//   time_step: Current logical time step in the generation.
//   input_tokens: Optional token IDs of the current batch (used to check
//     validity).
//   input_tokens_size: Size of the input_tokens array.
//   valid_mask: Optional boolean mask indicating valid tokens in the batch.
//   valid_mask_size: Size of the valid_mask array.
//   valid_val: The value representing "allow attention" (e.g., 0 or 127).
//   masked_val: The value representing "mask out attention" (e.g., -1e9 or
//     -128).
//   capacity: The physical capacity of the historical KV cache (excluding the
//     current batch).
//   uses_ringbuffer: If true, applies sliding window attention with ring buffer
//     logic.
//   window_size: The attention window size (only used if uses_ringbuffer is
//     true).
template <typename T>
void FillMaskSingle(T* mask, int64_t seq_q, int64_t seq_k, int32_t time_step,
                    const int32_t* input_tokens, int64_t input_tokens_size,
                    const bool* valid_mask, int64_t valid_mask_size,
                    T valid_val, T masked_val, int64_t capacity,
                    bool uses_ringbuffer, int64_t window_size) {
  // Number of tokens processed in this step (e.g., 1 for decode, chunk size for
  // prefill, or draft length for speculative verification).
  const int64_t batch_size = seq_q;

  // Start out by attending to no tokens.
  if (sizeof(T) == 1) {
    std::memset(mask, (int)masked_val, seq_q * seq_k);
  } else {
    for (int64_t i = 0; i < seq_q * seq_k; ++i) {
      mask[i] = masked_val;
    }
  }

  for (int64_t q = 0; q < seq_q; ++q) {
    const int64_t logical_pos = time_step + q;
    T* row = mask + (q * seq_k);

    // Fill in the mask for historical tokens stored in the KV cache.
    if (uses_ringbuffer) {
      for (int64_t k = 0; k < capacity; ++k) {
        // t_k is the logical token index (the token's absolute position in the
        // sequence, e.g., token 1050), whereas k is the physical slot in the
        // circular cache buffer (bounded by capacity, e.g., 0-511). We need
        // the logical index to check if the token falls within the attention
        // window.
        //
        // Before the cache is full, tokens are written sequentially, so
        // physical index k maps directly to logical token k.
        int64_t t_k = k;
        if (time_step >= capacity) {
          // 'age' is how many steps ago the token in slot k was written.
          // age = 0 means it was written at time_step - 1 (the most recent
          // token).
          // age = capacity - 1 means it is the oldest token still in the cache.
          int64_t age = (time_step - 1 - k + capacity) % capacity;
          t_k = time_step - 1 - age;
        }
        // Ensure the token is a valid past token (causality) and falls
        // within the sliding window constraint.
        if (t_k >= 0 && t_k < time_step &&
            t_k >= logical_pos - window_size + 1) {
          row[k] = valid_val;
        }
      }
    } else {
      for (int64_t k = 0; k < std::min<int64_t>(time_step, capacity); ++k) {
        row[k] = valid_val;
      }
    }

    // Fill in the mask for the tokens in the current batch.
    for (int64_t k_rel = 0; k_rel < batch_size; ++k_rel) {
      // The current batch's tokens are appended after the history cache slots
      // [0, capacity-1] in the KV cache layout, so we offset by capacity.
      int64_t k = capacity + k_rel;
      if (k >= seq_k) break;

      bool is_valid = true;
      if (valid_mask != nullptr) {
        if (k_rel < valid_mask_size) {
          is_valid = valid_mask[k_rel];
        }
      } else if (input_tokens != nullptr) {
        if (seq_q > 1) {
          if (k_rel < input_tokens_size) {
            is_valid = (input_tokens[k_rel] != -1);
          }
        }
      }

      if (k_rel <= q && is_valid) {
        row[k] = valid_val;
      }
    }
  }
}

absl::Status UpdateInterleavedSWAMasks(
    void* local_ptr, void* global_ptr, ::litert::ElementType element_type,
    int64_t seq_q, int64_t seq_k_local, int64_t seq_k_global, int32_t time_step,
    const int32_t* input_tokens, int64_t input_tokens_size,
    const bool* valid_mask, int64_t valid_mask_size) {
  // The physical capacity of the local KV cache buffer (excluding current
  // batch/draft).
  int64_t local_capacity = seq_k_local - seq_q;
  // The physical capacity of the global KV cache buffer (excluding current
  // batch/draft).
  int64_t global_capacity = seq_k_global - seq_q;
  // The attention window size (how far back a token can attend).
  // In practice, this is optimized to match the local cache capacity to save
  // memory, but we keep them conceptually separate for flexibility and
  // clarity in FillMaskSingle.
  int64_t local_window_size = local_capacity;

  if (element_type == ::litert::ElementType::Int8) {
    if (local_ptr) {
      FillMaskSingle<int8_t>(
          static_cast<int8_t*>(local_ptr), seq_q, seq_k_local, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 127,
          -128, local_capacity, /*uses_ringbuffer=*/true, local_window_size);
    }
    if (global_ptr) {
      FillMaskSingle<int8_t>(
          static_cast<int8_t*>(global_ptr), seq_q, seq_k_global, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 127,
          -128, global_capacity, /*uses_ringbuffer=*/false, 0);
    }
  } else if (element_type == ::litert::ElementType::Int16) {
    if (local_ptr) {
      FillMaskSingle<int16_t>(
          static_cast<int16_t*>(local_ptr), seq_q, seq_k_local, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 0,
          -32767, local_capacity, /*uses_ringbuffer=*/true, local_window_size);
    }
    if (global_ptr) {
      FillMaskSingle<int16_t>(
          static_cast<int16_t*>(global_ptr), seq_q, seq_k_global, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 0,
          -32767, global_capacity, /*uses_ringbuffer=*/false, 0);
    }
  } else if (element_type == ::litert::ElementType::Float32) {
    if (local_ptr) {
      FillMaskSingle<float>(
          static_cast<float*>(local_ptr), seq_q, seq_k_local, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 0.0f,
          -1e9f, local_capacity, /*uses_ringbuffer=*/true, local_window_size);
    }
    if (global_ptr) {
      FillMaskSingle<float>(
          static_cast<float*>(global_ptr), seq_q, seq_k_global, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 0.0f,
          -1e9f, global_capacity, /*uses_ringbuffer=*/false, 0);
    }
  } else if (element_type == ::litert::ElementType::Float16) {
    if (local_ptr) {
      FillMaskSingle<uint16_t>(
          static_cast<uint16_t*>(local_ptr), seq_q, seq_k_local, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 0x0000,
          0xFC00, local_capacity, /*uses_ringbuffer=*/true, local_window_size);
    }
    if (global_ptr) {
      FillMaskSingle<uint16_t>(
          static_cast<uint16_t*>(global_ptr), seq_q, seq_k_global, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 0x0000,
          0xFC00, global_capacity, /*uses_ringbuffer=*/false, 0);
    }
  } else if (element_type == ::litert::ElementType::BFloat16) {
    if (local_ptr) {
      FillMaskSingle<uint16_t>(
          static_cast<uint16_t*>(local_ptr), seq_q, seq_k_local, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 0x0000,
          0xFF80, local_capacity, /*uses_ringbuffer=*/true, local_window_size);
    }
    if (global_ptr) {
      FillMaskSingle<uint16_t>(
          static_cast<uint16_t*>(global_ptr), seq_q, seq_k_global, time_step,
          input_tokens, input_tokens_size, valid_mask, valid_mask_size, 0x0000,
          0xFF80, global_capacity, /*uses_ringbuffer=*/false, 0);
    }
  } else {
    return absl::InvalidArgumentError("Unsupported mask element type");
  }

  return absl::OkStatus();
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

  int64_t seq_q_local = 0;  // query sequence length, local mask
  int64_t seq_k_local = 0;  // key sequence length, local mask
  if (mask_local_buf) {
    LITERT_ASSIGN_OR_RETURN(auto type, mask_local_buf->TensorType());
    auto dims = type.Layout().Dimensions();
    int rank = type.Layout().Rank();
    seq_q_local = dims[rank - 2];
    seq_k_local = dims[rank - 1];
  }

  int64_t seq_q_global = 0;  // query sequence length, global mask
  int64_t seq_k_global = 0;  // key sequence length, global mask
  if (mask_global_buf) {
    LITERT_ASSIGN_OR_RETURN(auto type, mask_global_buf->TensorType());
    auto dims = type.Layout().Dimensions();
    int rank = type.Layout().Rank();
    seq_q_global = dims[rank - 2];
    seq_k_global = dims[rank - 1];
  }

  // Detect if local and global masks use different KV cache sizes. If so,
  // we assume the local mask uses a ring buffer (wrap-around logic). Once
  // the 'Executor Metadata' design is implemented, we can remove this
  // heuristic because the metadata will inform the execution behavior on
  // regular vs ring buffer attention mask.
  bool is_interleaved_swa = false;
  if (mask_local_buf && mask_global_buf && seq_k_local != seq_k_global) {
    is_interleaved_swa = true;
  }

  ::litert::TensorBuffer* reference_buf =
      mask_local_buf ? mask_local_buf : mask_global_buf;
  LITERT_ASSIGN_OR_RETURN(auto mask_type, reference_buf->TensorType());

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

  if (is_interleaved_swa) {
    return UpdateInterleavedSWAMasks(
        local_ptr, global_ptr, mask_type.ElementType(), seq_q_local,
        seq_k_local, seq_k_global, time_step, input_tokens, input_tokens_size,
        valid_mask, valid_mask_size);
  }

  // If we made it here, all layers use the same KV cache size.
  int64_t seq_q = seq_q_global ? seq_q_global : seq_q_local;
  int64_t seq_k = seq_k_global ? seq_k_global : seq_k_local;

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

absl::StatusOr<NpuMask> NpuMask::CreateForTest(
    MaskUpdateMethod method, const ::litert::CompiledModel* compiled_model,
    InferenceContext mask_context) {
  if (method == MaskUpdateMethod::kModel && compiled_model == nullptr) {
    return absl::InvalidArgumentError(
        "Compiled model must be provided when MaskUpdateMethod is kModel.");
  }
  return NpuMask(method, compiled_model, std::move(mask_context));
}

absl::StatusOr<NpuMask> NpuMask::Create(
    MaskUpdateMethod method,
    const ::litert::CompiledModel* npu_auxiliary_compiled_model,
    absl::string_view prefill_signature, absl::string_view decode_signature,
    absl::string_view verify_signature,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers) {
  RET_CHECK(npu_auxiliary_compiled_model != nullptr)
      << "Auxiliary compiled model cannot be null for NpuMask";
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers, prefill_output_buffers, decode_input_buffers,
      decode_output_buffers, verify_input_buffers, verify_output_buffers;

  auto setup_mask_stage =
      [&](absl::string_view signature,
          const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
              decoder_inputs,
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
              in_buffers,
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
              out_buffers) -> absl::Status {
    LITERT_ASSIGN_OR_RETURN(in_buffers[MaskSignatures::kMaskInputTimeStep],
                            npu_auxiliary_compiled_model->CreateInputBuffer(
                                signature, MaskSignatures::kMaskInputTimeStep));
    in_buffers[MaskSignatures::kMaskInputTimeStep].Clear();

    LITERT_ASSIGN_OR_RETURN(in_buffers[MaskSignatures::kMaskInputTokens],
                            npu_auxiliary_compiled_model->CreateInputBuffer(
                                signature, MaskSignatures::kMaskInputTokens));
    in_buffers[MaskSignatures::kMaskInputTokens].Clear();

    LITERT_ASSIGN_OR_RETURN(
        auto input_names,
        npu_auxiliary_compiled_model->GetSignatureInputNames(signature));
    if (absl::c_find(input_names, MaskSignatures::kMaskInputValidMask) !=
        input_names.end()) {
      LITERT_ASSIGN_OR_RETURN(
          in_buffers[MaskSignatures::kMaskInputValidMask],
          npu_auxiliary_compiled_model->CreateInputBuffer(
              signature, MaskSignatures::kMaskInputValidMask));
      in_buffers[MaskSignatures::kMaskInputValidMask].Clear();
    }

    LITERT_ASSIGN_OR_RETURN(
        auto output_names,
        npu_auxiliary_compiled_model->GetSignatureOutputNames(signature));
    for (const auto& name : output_names) {
      if (decoder_inputs.contains(name)) {
        LITERT_ASSIGN_OR_RETURN(out_buffers[name],
                                decoder_inputs.at(name).Duplicate());
      } else {
        LITERT_ASSIGN_OR_RETURN(
            out_buffers[name],
            npu_auxiliary_compiled_model->CreateOutputBuffer(signature, name));
      }
    }
    return absl::OkStatus();
  };

  LITERT_RETURN_IF_ERROR(
      setup_mask_stage(prefill_signature, text_decoder_prefill_input_buffers,
                       prefill_input_buffers, prefill_output_buffers));

  LITERT_RETURN_IF_ERROR(
      setup_mask_stage(decode_signature, text_decoder_decode_input_buffers,
                       decode_input_buffers, decode_output_buffers));

  if (!verify_signature.empty() &&
      npu_auxiliary_compiled_model->FindSignature(verify_signature)) {
    LITERT_RETURN_IF_ERROR(
        setup_mask_stage(verify_signature, text_decoder_verify_input_buffers,
                         verify_input_buffers, verify_output_buffers));
  }

  InferenceContext mask_context(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
  return NpuMask(method, npu_auxiliary_compiled_model, std::move(mask_context));
}

absl::StatusOr<NpuMask> NpuMask::CreateForDrafter(
    MaskUpdateMethod method, const ::litert::CompiledModel* compiled_model,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        mask_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        mask_output_buffers) {
  if (method == MaskUpdateMethod::kModel && compiled_model == nullptr) {
    return absl::InvalidArgumentError(
        "Compiled model must be provided when MaskUpdateMethod is kModel.");
  }
  InferenceContext ctx;
  ctx.decode_input_buffers = std::move(mask_input_buffers);
  ctx.decode_output_buffers = std::move(mask_output_buffers);
  return NpuMask(method, compiled_model, std::move(ctx));
}

absl::Status NpuMask::UpdateOutputBuffers(
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers) {
  for (const auto& [name, buf] : text_decoder_prefill_input_buffers) {
    if (mask_context_.prefill_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(mask_context_.prefill_output_buffers[name],
                              buf.Duplicate());
    }
  }
  for (const auto& [name, buf] : text_decoder_decode_input_buffers) {
    if (mask_context_.decode_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(mask_context_.decode_output_buffers[name],
                              buf.Duplicate());
    }
  }
  for (const auto& [name, buf] : text_decoder_verify_input_buffers) {
    if (mask_context_.verify_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(mask_context_.verify_output_buffers[name],
                              buf.Duplicate());
    }
  }
  return absl::OkStatus();
}

absl::Status NpuMask::RunPrefill(absl::string_view signature) const {
  if (method_ == MaskUpdateMethod::kWH) {
    return HWMaskUpdate(
        const_cast<
            absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
            mask_context_.prefill_input_buffers),
        const_cast<
            absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
            mask_context_.prefill_output_buffers));
  }
  RET_CHECK(compiled_model_ != nullptr)
      << "Compiled model must be provided for kModel mask update.";
  absl::string_view sig = signature.empty() ? kPrefillMaskBase : signature;
  auto res = compiled_model_->Run(
      sig,
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          mask_context_.prefill_input_buffers),
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          mask_context_.prefill_output_buffers));
  RET_CHECK(res) << "Failed to run prefill mask model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuMask::RunDecode(absl::string_view signature) const {
  if (method_ == MaskUpdateMethod::kWH) {
    return HWMaskUpdate(
        const_cast<
            absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
            mask_context_.decode_input_buffers),
        const_cast<
            absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
            mask_context_.decode_output_buffers));
  }
  RET_CHECK(compiled_model_ != nullptr)
      << "Compiled model must be provided for kModel mask update.";
  absl::string_view sig =
      signature.empty() ? MaskSignatures::kDecodeMask : signature;
  auto res = compiled_model_->Run(
      sig,
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          mask_context_.decode_input_buffers),
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          mask_context_.decode_output_buffers));
  RET_CHECK(res) << "Failed to run decode mask model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuMask::RunVerify(absl::string_view signature) const {
  if (method_ == MaskUpdateMethod::kWH) {
    return HWMaskUpdate(
        const_cast<
            absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
            mask_context_.verify_input_buffers),
        const_cast<
            absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
            mask_context_.verify_output_buffers));
  }
  RET_CHECK(compiled_model_ != nullptr)
      << "Compiled model must be provided for kModel mask update.";
  absl::string_view sig =
      signature.empty() ? MaskSignatures::kVerifyMask : signature;
  auto res = compiled_model_->Run(
      sig,
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          mask_context_.verify_input_buffers),
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          mask_context_.verify_output_buffers));
  RET_CHECK(res) << "Failed to run verify mask model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuMask::RunDrafter() const {
  if (method_ == MaskUpdateMethod::kWH) {
    return HWMaskUpdate(
        const_cast<
            absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
            mask_context_.decode_input_buffers),
        const_cast<
            absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
            mask_context_.decode_output_buffers));
  }
  RET_CHECK(compiled_model_ != nullptr)
      << "Compiled model must be provided for kModel mask update.";
  auto res = compiled_model_->Run(
      MaskSignatures::kMtpMask,
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          mask_context_.decode_input_buffers),
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          mask_context_.decode_output_buffers));
  RET_CHECK(res) << "Failed to run drafter mask model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuMask::SetDecodeInput(int32_t step, int32_t token_id) {
  LITERT_RETURN_IF_ERROR(SetFirstElement(
      mask_context_.decode_input_buffers[MaskSignatures::kMaskInputTimeStep],
      step));
  if (mask_context_.decode_input_buffers.contains(
          MaskSignatures::kMaskInputTokens)) {
    LITERT_RETURN_IF_ERROR(SetFirstElement(
        mask_context_.decode_input_buffers[MaskSignatures::kMaskInputTokens],
        token_id));
  }
  if (mask_context_.decode_input_buffers.contains(
          MaskSignatures::kMaskInputValidMask)) {
    auto& buf =
        mask_context_.decode_input_buffers[MaskSignatures::kMaskInputValidMask];
    LITERT_ASSIGN_OR_RETURN(auto lock,
                            ::litert::TensorBufferScopedLock::Create(
                                buf, ::litert::TensorBuffer::LockMode::kWrite));
    static_cast<bool*>(lock.second)[0] = true;
  }
  return absl::OkStatus();
}

absl::Status NpuMask::SetVerifyInput(int32_t start_step,
                                     absl::Span<const int> verify_ids) {
  LITERT_RETURN_IF_ERROR(SetFirstElement(
      mask_context_.verify_input_buffers[MaskSignatures::kMaskInputTimeStep],
      start_step));
  if (mask_context_.verify_input_buffers.contains(
          MaskSignatures::kMaskInputTokens)) {
    LITERT_ASSIGN_OR_RETURN(
        auto mask_tokens_lock,
        ::litert::TensorBufferScopedLock::Create(
            mask_context_
                .verify_input_buffers[MaskSignatures::kMaskInputTokens],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* mask_tokens_ptr = static_cast<int32_t*>(mask_tokens_lock.second);
    for (size_t i = 0; i < verify_ids.size(); ++i) {
      mask_tokens_ptr[i] = verify_ids[i];
    }
  }
  return absl::OkStatus();
}

absl::Status NpuMask::SetPrefillInput(int32_t start_step,
                                      absl::Span<const int> token_ids) {
  LITERT_RETURN_IF_ERROR(SetFirstElement(
      mask_context_.prefill_input_buffers[MaskSignatures::kMaskInputTimeStep],
      start_step));

  if (mask_context_.prefill_input_buffers.contains(
          MaskSignatures::kMaskInputTokens)) {
    auto& buf =
        mask_context_.prefill_input_buffers[MaskSignatures::kMaskInputTokens];
    LITERT_ASSIGN_OR_RETURN(auto lock,
                            ::litert::TensorBufferScopedLock::Create(
                                buf, ::litert::TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto type, buf.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto num_elements, type.Layout().NumElements());
    auto* ptr = static_cast<int32_t*>(lock.second);
    for (size_t i = 0; i < num_elements; ++i) {
      if (i < token_ids.size()) {
        ptr[i] = token_ids[i] < 0 ? 0 : token_ids[i];
      } else {
        ptr[i] = 0;
      }
    }
  }

  if (mask_context_.prefill_input_buffers.contains(
          MaskSignatures::kMaskInputValidMask)) {
    auto& buf = mask_context_
                    .prefill_input_buffers[MaskSignatures::kMaskInputValidMask];
    LITERT_ASSIGN_OR_RETURN(auto lock,
                            ::litert::TensorBufferScopedLock::Create(
                                buf, ::litert::TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto type, buf.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto num_elements, type.Layout().NumElements());
    auto* ptr = static_cast<bool*>(lock.second);
    for (size_t i = 0; i < num_elements; ++i) {
      ptr[i] = (i < token_ids.size());
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::lm
