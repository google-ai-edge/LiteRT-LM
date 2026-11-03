// Copyright 2026 RMAA Holdings & The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_DELEGATES_RMAA_FABRIC_H_
#define THIRD_PARTY_ODML_LITERT_LM_DELEGATES_RMAA_FABRIC_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "c/api_export.h"
#include "c/error_reporter.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// RMAA High-Performance Execution & Storage Fabric for LiteRT-LM
// Fuses:
// 1. TorchJIT: SMTF v3 2MiB Transparent Huge Page mmap & O_DIRECT DMA
// 2. NJIT: ~64µs KV-Cache zero-prefill memory grafting via /dev/shm
// 3. LocalJIT: 64-bit FNV-1a L1 cache & JIT execution dispatch
// 4. UPAAS: QUIC Datagram / RaptorQ FEC distributed activation pipeline
// ============================================================================

/// Opaque handle to an RMAA Fabric Runtime Instance
typedef struct LiteRtLmRmaaFabricInstance* LiteRtLmRmaaFabricHandle;

/// Configuration for RMAA Fabric Delegate
typedef struct LiteRtLmRmaaFabricConfig {
  bool enable_smtf_v3_hugepages;      // Enable 2 MiB THP + O_DIRECT DMA mapping
  bool enable_njit_memory_grafting;  // Enable 0x54494A4E direct KV grafting
  bool enable_localjit_dispatch;     // Enable 64-bit FNV-1a JIT micro-op cache
  bool enable_upaas_quic_transport;  // Enable distributed QUIC/RaptorQ activation ring
  const char* shm_namespace;         // Default: "/litert_njit_graft"
  uint32_t quic_mtu_bytes;           // Default: 1200 bytes
  uint32_t mtp_speculative_gamma;    // Default: 2 (Multi-Token Prediction)
} LiteRtLmRmaaFabricConfig;

/// Return default production configuration for RMAA Fabric
LITERT_LM_C_API_EXPORT LiteRtLmRmaaFabricConfig
litert_lm_rmaa_fabric_config_default(void);

/// Create an RMAA Fabric delegate instance
LITERT_LM_C_API_EXPORT LiteRtLmStatusCode
litert_lm_rmaa_fabric_create(const LiteRtLmRmaaFabricConfig* config,
                             LiteRtLmRmaaFabricHandle* out_handle);

/// Destroy an RMAA Fabric delegate instance
LITERT_LM_C_API_EXPORT LiteRtLmStatusCode
litert_lm_rmaa_fabric_destroy(LiteRtLmRmaaFabricHandle handle);

/// Map SMTF v3 model weights with 2 MiB THP inductive alignment (TorchJIT)
LITERT_LM_C_API_EXPORT LiteRtLmStatusCode
litert_lm_rmaa_fabric_map_smtf_v3(LiteRtLmRmaaFabricHandle handle,
                                  const char* file_path,
                                  void** out_mapped_base,
                                  size_t* out_mapped_size);

/// Zero-Prefill KV Cache Memory Grafting (NJIT)
/// Validates 36-byte header ('NJIT'), checks CID, and grafts attention KV pages
LITERT_LM_C_API_EXPORT LiteRtLmStatusCode
litert_lm_rmaa_fabric_graft_kv_cache(LiteRtLmRmaaFabricHandle handle,
                                     uint64_t content_id,
                                     void* engine_kv_cache_ptr,
                                     size_t max_cache_bytes,
                                     size_t* out_grafted_sequence_pos);

/// Query LocalJIT 64-bit FNV-1a dispatch cache for pre-compiled micro-ops
LITERT_LM_C_API_EXPORT LiteRtLmStatusCode
litert_lm_rmaa_fabric_dispatch_kernel(LiteRtLmRmaaFabricHandle handle,
                                      uint64_t kernel_signature_hash,
                                      void* execution_context,
                                      bool* out_cache_hit);

/// Broadcast token activations over UPAAS QUIC Datagram / RaptorQ ring
LITERT_LM_C_API_EXPORT LiteRtLmStatusCode
litert_lm_rmaa_fabric_stream_activations(LiteRtLmRmaaFabricHandle handle,
                                         uint32_t layer_idx,
                                         const void* bf16_activations,
                                         size_t tensor_bytes);

/// Position-indexed deterministic inverse-CDF token verification for MTP
LITERT_LM_C_API_EXPORT LiteRtLmStatusCode
litert_lm_rmaa_fabric_verify_mtp_draft(LiteRtLmRmaaFabricHandle handle,
                                       uint64_t seed,
                                       uint64_t seq_id,
                                       uint32_t position,
                                       const int32_t* draft_tokens,
                                       size_t gamma,
                                       size_t* out_accepted_count);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_DELEGATES_RMAA_FABRIC_H_
