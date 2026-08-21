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

#if defined(__APPLE__)
#include "engine.h"  // NOLINT
#else
#include "c/engine.h"
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

// Opaque pointer for session debug info.
// Use `litert_lm_experimental_session_debug_info_delete` to free memory.
//
// Added in version 0.2.0.
typedef struct LiteRtLmSessionDebugInfo LiteRtLmSessionDebugInfo;

// Checks whether the LiteRT-LM runtime binary was built with the debugger
// tracing backend enabled (LITERT_LM_DEBUGGER_ENABLED=1).
//
// @return 1 if debugger is enabled at compile-time, 0 otherwise.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
int litert_lm_experimental_is_debugger_enabled(void);

// Returns debug info for the session, or NULL on failure or if
// debugging is unsupported/disabled. The caller is responsible for deleting
// the returned object using `litert_lm_experimental_session_debug_info_delete`.
//
// @param session The session to query.
// @return A pointer to the session debug info result, or NULL on failure
//   or if debugging is unsupported.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmSessionDebugInfo* litert_lm_experimental_session_get_debug_info(
    LiteRtLmSession* session);

// Returns session debug info for the conversation's underlying session, or NULL
// on failure or if debugging is unsupported/disabled. The caller is
// responsible for deleting the returned object using
// `litert_lm_experimental_session_debug_info_delete`.
//
// @param conversation The conversation to query.
// @return A pointer to the session debug info result, or NULL on failure
//   or if debugging is unsupported.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmSessionDebugInfo*
litert_lm_experimental_conversation_get_session_debug_info(
    LiteRtLmConversation* conversation);

// Destroys a LiteRtLmSessionDebugInfo object.
//
// @param debug_info The session debug info object to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_experimental_session_debug_info_delete(
    LiteRtLmSessionDebugInfo* debug_info);

// Returns the relative debug capture directory from a LiteRtLmSessionDebugInfo
// object (e.g. "litert_lm_debugger/0"), containing intermediate activation
// Safetensors dumps and token generation trace logs. The returned string is
// owned by the `debug_info` object and is valid only for its lifetime.
//
// @param debug_info The session debug info object.
// @return The relative capture directory string.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_experimental_session_debug_info_get_capture_dir(
    const LiteRtLmSessionDebugInfo* debug_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_EXPERIMENTAL_H_
