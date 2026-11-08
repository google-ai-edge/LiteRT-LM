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

// Opaque pointer for a process memory monitor that samples the host and
// accelerator memory usage of the current process on a background thread and
// tracks the peaks, as `litert_lm_advanced_main --report_peak_memory_footprint`
// does. Use `litert_lm_experimental_memory_monitor_delete` to free memory.
typedef struct LiteRtLmMemoryMonitor LiteRtLmMemoryMonitor;

// Creates a memory monitor. Sampling does not begin until
// `litert_lm_experimental_memory_monitor_start` is called.
//
// @param sampling_interval_ms The sampling interval in milliseconds; must be
//   positive.
// @param out_monitor On success, receives the monitor, owned by the caller;
//   release with `litert_lm_experimental_memory_monitor_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_monitor` is NULL or `sampling_interval_ms` is not positive.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_memory_monitor_create(
    int sampling_interval_ms, LiteRtLmMemoryMonitor** out_monitor);

// Destroys a memory monitor, stopping it first if it is running.
//
// @param monitor The monitor to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_experimental_memory_monitor_delete(
    LiteRtLmMemoryMonitor* monitor);

// Starts sampling. A no-op if the monitor is already running.
//
// @param monitor The monitor to start.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `monitor` is NULL.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_memory_monitor_start(
    LiteRtLmMemoryMonitor* monitor);

// Stops sampling. The peaks remain available after stopping. A no-op if the
// monitor is not running.
//
// @param monitor The monitor to stop.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `monitor` is NULL.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_memory_monitor_stop(
    LiteRtLmMemoryMonitor* monitor);

// Gets the peak memory resident in RAM of the process observed by the
// monitor, in MB: the resident set size (RSS) or working set, plus accelerator
// memory that is resident but not in it (e.g. dedicated GPU memory).
//
// @param monitor The monitor to query.
// @param out_mb On success, receives the peak in MB; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `monitor` or `out_mb` is NULL. kLiteRtLmStatusNotFound if host memory
//   tracking is unsupported on the platform or no sample was taken.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_memory_monitor_get_peak_memory_mb(
    const LiteRtLmMemoryMonitor* monitor, float* out_mb);

// Gets the peak memory the OS attributes to the process, as observed by the
// monitor, in MB, plus accelerator memory the OS keeps no charge for. The
// definition differs by OS, so do not compare it across them: the commit charge
// on Windows, RSS plus swap on Linux and Android, and the physical footprint on
// Apple platforms.
//
// @param monitor The monitor to query.
// @param out_mb On success, receives the peak in MB; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `monitor` or `out_mb` is NULL. kLiteRtLmStatusNotFound if host memory
//   tracking is unsupported on the platform or no sample was taken.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_memory_monitor_get_peak_private_memory_mb(
    const LiteRtLmMemoryMonitor* monitor, float* out_mb);

// Gets the peak accelerator (e.g. GPU) memory of the process observed by the
// monitor, in MB. The two peaks above already include accelerator memory, so
// do not add this to them.
//
// @param monitor The monitor to query.
// @param out_mb On success, receives the peak in MB; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `monitor` or `out_mb` is NULL. kLiteRtLmStatusNotFound if accelerator
//   memory tracking is unsupported on the platform or no sample was taken.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_memory_monitor_get_peak_accelerator_memory_mb(
    const LiteRtLmMemoryMonitor* monitor, float* out_mb);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_EXPERIMENTAL_H_
