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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_EXPERIMENTAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_EXPERIMENTAL_H_

#include <stdbool.h>
#include <stdint.h>

#if defined(__APPLE__)
#include "api_export.h"      // NOLINT
#include "conversation.h"    // NOLINT
#include "engine.h"          // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/api_export.h"
#include "c/conversation.h"
#include "c/engine.h"
#include "c/error_reporter.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// WARNING: The APIs declared in this header are EXPERIMENTAL and subject to
// change or removal without notice. API stability and backward compatibility
// are not guaranteed.
// =============================================================================

// Updates whether to enable Metal residency set on GPU for the given engine at
// runtime.
//
// To configure this setting during initialization, use
// `litert_lm_engine_settings_set_gpu_enable_metal_residency_set` instead.
//
// When enabled on Apple platforms (macOS and iOS with Metal GPU backend), this
// uses Apple's MTLResidencySet API to ensure model weights and allocations
// remain resident in GPU memory, preventing memory swapping and reducing
// allocation overhead.
//
// This setting is only supported on Apple platforms (macOS / iOS) with the GPU
// backend. On other platforms (e.g. Linux, Android, Windows) or non-GPU
// backends, this setting has no effect and is safely ignored.
//
// @param engine The engine to update.
// @param enable_metal_residency_set Whether to enable Metal residency set.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `engine`
//   is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns LiteRtLmStatusCode and failures return a
// canonical LiteRtLmStatusCode instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_engine_update_gpu_enable_metal_residency_set(
    LiteRtLmEngine* engine, bool enable_metal_residency_set);

// Sets whether to enable the AI Edge Debugger. Defaults to false.
//
// When enabled, the engine captures per-token traces (JSONL) and, on CPU/GPU
// backends, intermediate activations (Safetensors) under
// `<cache_dir>/litert_lm_debugger/<session_id>/`. A cache directory must be set
// with `litert_lm_engine_settings_set_cache_dir` (not ":memory" or
// ":nocache") and must be a writable directory; otherwise engine creation
// fails. When weights are streamed, the setting is ignored with a warning and
// nothing is captured. Use
// `litert_lm_experimental_session_get_debug_info` to find the captures.
//
// Captures can grow large over long sessions.
//
// WARNING: The captures contain user-derived data (decoded generated text and
// activations computed from the prompt), stored unencrypted and never deleted
// by the engine. Enable only for development or diagnostics, not for
// production end-user traffic; the caller must delete the capture directory.
//
// @param settings The engine settings.
// @param enable_debugger Whether to enable the AI Edge Debugger.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_engine_settings_set_enable_debugger(
    LiteRtLmEngineSettings* settings, bool enable_debugger);

// Sets whether the GPU backend uses the IrModel initialization path.
//
// When unset, the engine uses its default. On non-GPU backends, this setting
// has no effect and is safely ignored.
//
// @param settings The engine settings.
// @param use_ir_model Whether to use the IrModel path on the GPU backend.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_engine_settings_set_gpu_use_ir_model(
    LiteRtLmEngineSettings* settings, bool use_ir_model);

// Sets the target size in MiB for splitting multi-token GQA prefill attention
// scores across query-head chunks to reduce peak intermediate score buffer
// size.
//
// Negative values (e.g. -1, the default) disable head-chunked attention, 0
// enables it with the default target size (768 MiB), and positive values enable
// it with the specified per-chunk target size in MiB.
//
// Note: This setting is currently only supported by the GPU Artisan backend.
// On other backends, this call has no effect and returns kLiteRtLmStatusOk.
//
// @param settings The engine settings.
// @param head_chunked_attention_mib Target chunk size in MiB (< 0 disables,
//   0 uses the 768 MiB default, > 0 sets the target in MiB).
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If the main backend is not the Artisan GPU backend,
//   this call has no effect and returns kLiteRtLmStatusOk.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_engine_settings_set_head_chunked_attention_mib(
    LiteRtLmEngineSettings* settings, int32_t head_chunked_attention_mib);

// KV cache quantization type for inference.
//
// Added in version 1.1.0.
typedef enum {
  // Uses the model/backend default KV cache representation.
  kLiteRtLmKvCacheTypeDefault = 0,
  // Uses dynamic INT8 KV cache quantization on global attention layers for
  // supported Artisan GPU models when static range quantization is not used.
  kLiteRtLmKvCacheTypeInt8 = 1,
} LiteRtLmKvCacheType;

// Sets the KV cache quantization type for the engine.
//
// `kLiteRtLmKvCacheTypeInt8` enables dynamic INT8 KV cache quantization on
// global attention layers for supported Artisan GPU models when static range
// quantization is not used. If `kLiteRtLmKvCacheTypeInt8` is selected for a
// non-Artisan-GPU model or backend, a warning is logged and execution continues
// as normal with the default KV cache.
//
// @param settings The engine settings.
// @param kv_cache_type The KV cache type (`kLiteRtLmKvCacheTypeDefault` or
//   `kLiteRtLmKvCacheTypeInt8`).
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL or `kv_cache_type` is not a valid `LiteRtLmKvCacheType`.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_engine_settings_set_kv_cache_type(
    LiteRtLmEngineSettings* settings, LiteRtLmKvCacheType kv_cache_type);

// Opaque pointer for session debug info.
// Use `litert_lm_experimental_session_debug_info_delete` to free memory.
//
// Added in version 0.2.0.
typedef struct LiteRtLmSessionDebugInfo LiteRtLmSessionDebugInfo;

// Gets debug info for the session. The caller is responsible for deleting the
// returned object using `litert_lm_experimental_session_debug_info_delete`.
//
// If debugging is unsupported or disabled for the session, this succeeds and
// `*out_debug_info` is set to NULL.
//
// @param session The session to query.
// @param out_debug_info On success, receives the session debug info, or NULL
//   if debugging is unsupported or disabled; set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `session` or `out_debug_info` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the debug info is returned
// through out_debug_info.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_session_get_debug_info(
    LiteRtLmSession* session, LiteRtLmSessionDebugInfo** out_debug_info);

// Gets session debug info for the conversation's underlying session. The
// caller is responsible for deleting the returned object using
// `litert_lm_experimental_session_debug_info_delete`.
//
// If debugging is unsupported or disabled for the session, this succeeds and
// `*out_debug_info` is set to NULL.
//
// @param conversation The conversation to query.
// @param out_debug_info On success, receives the session debug info, or NULL
//   if debugging is unsupported or disabled; set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `conversation` or `out_debug_info` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the debug info is returned
// through out_debug_info.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_conversation_get_session_debug_info(
    LiteRtLmConversation* conversation,
    LiteRtLmSessionDebugInfo** out_debug_info);

// Destroys a LiteRtLmSessionDebugInfo object.
//
// @param debug_info The session debug info object to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_experimental_session_debug_info_delete(
    LiteRtLmSessionDebugInfo* debug_info);

// Gets the relative debug capture directory from a LiteRtLmSessionDebugInfo
// object (e.g. "litert_lm_debugger/0"), containing intermediate activation
// Safetensors dumps and token generation trace logs. The returned string is
// owned by the `debug_info` object and is valid only for its lifetime.
//
// @param debug_info The session debug info object.
// @param out_capture_dir On success, receives the relative capture directory
//   string; set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `debug_info` or `out_capture_dir` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the directory is returned
// through out_capture_dir.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_session_debug_info_get_capture_dir(
    const LiteRtLmSessionDebugInfo* debug_info, const char** out_capture_dir);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_EXPERIMENTAL_H_
