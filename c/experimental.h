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

#if defined(__APPLE__)
#include "api_export.h"      // NOLINT
#include "engine.h"          // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/api_export.h"
#include "c/engine.h"
#include "c/error_reporter.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LiteRtLmConversation LiteRtLmConversation;

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

// Opaque pointer for session debug info.
// Use `litert_lm_experimental_session_debug_info_delete` to free memory.
//
// Added in version 0.2.0.
typedef struct LiteRtLmSessionDebugInfo LiteRtLmSessionDebugInfo;

// Checks whether the LiteRT-LM runtime binary was built with the debugger
// tracing backend enabled (LITERT_LM_DEBUGGER_ENABLED=1).
//
// @param out_enabled On success, receives true if the debugger is enabled at
//   compile-time, false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_enabled` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_enabled.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_is_debugger_enabled(
    bool* out_enabled);

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
