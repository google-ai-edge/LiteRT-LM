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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_MODEL_SYNC_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_MODEL_SYNC_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__)
#include "api_export.h"      // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/api_export.h"
#include "c/error_reporter.h"  // IWYU pragma: export
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// LiteRT-LM ModelSync C API
// =============================================================================

// Opaque handle representing synchronization configuration options.
//
// Added in version 1.1.0.
typedef struct LiteRtLmModelSyncOptions LiteRtLmModelSyncOptions;

// Opaque handle representing a computed synchronization plan.
//
// Added in version 1.1.0.
typedef struct LiteRtLmModelSyncPlan LiteRtLmModelSyncPlan;

// Opaque writer passed to `LiteRtLmModelSyncRangeFetcherCallback` to stream
// fetched bytes into the sync engine and optionally record an error message.
// Valid only for the duration of the callback invocation.
//
// Added in version 1.1.0.
typedef struct LiteRtLmModelSyncRangeWriter LiteRtLmModelSyncRangeWriter;

// Opaque handle representing the outcome of `litert_lm_model_sync_cache_sync`.
//
// Added in version 1.1.0.
typedef struct LiteRtLmModelSyncResult LiteRtLmModelSyncResult;

// Streams a chunk of fetched bytes into `writer` from inside a
// `LiteRtLmModelSyncRangeFetcherCallback`.
//
// @param writer The writer handle passed to the fetcher callback.
// @param data Byte buffer to write (may be NULL only when `size` is 0).
// @param size Number of bytes to write from `data`.
// @return kLiteRtLmStatusOk on success, kLiteRtLmStatusInvalidArgument if
//   `writer` is NULL (or `data` is NULL when `size > 0`), or
//   kLiteRtLmStatusDataLoss if writing into the underlying sink fails.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_range_writer_write(
    LiteRtLmModelSyncRangeWriter* writer, const void* data, size_t size);

// Records an optional diagnostic error message on `writer` from inside a
// `LiteRtLmModelSyncRangeFetcherCallback`.
//
// Setting an error message does not fail the fetch on its own; the callback
// must still return a non-OK `LiteRtLmStatusCode`. Passing NULL for `message`
// clears any previously recorded message on `writer`.
//
// @param writer The writer handle passed to the fetcher callback.
// @param message NUL-terminated UTF-8 error message, or NULL to clear.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `writer` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_range_writer_set_error(
    LiteRtLmModelSyncRangeWriter* writer, const char* message);

// Callback invoked by ModelSync to fetch a byte range starting at `offset` of
// `length` bytes (`[offset, offset + length)`).
//
// Implementations must pass received byte chunks via
// `litert_lm_model_sync_range_writer_write(writer, data, size)` and return
// `kLiteRtLmStatusOk` on success or a non-OK `LiteRtLmStatusCode` on failure
// (optionally calling `litert_lm_model_sync_range_writer_set_error` first).
//
// @param offset Byte offset of the start of the range to fetch.
// @param length Number of bytes to fetch starting from `offset`.
// @param writer Opaque writer handle valid for the duration of the callback.
// @param user_data Caller-supplied context pointer.
// @return kLiteRtLmStatusOk on success or a non-OK `LiteRtLmStatusCode`.
//
// Added in version 1.1.0.
typedef LiteRtLmStatusCode (*LiteRtLmModelSyncRangeFetcherCallback)(
    uint64_t offset, uint64_t length, LiteRtLmModelSyncRangeWriter* writer,
    void* user_data);

// =============================================================================
// Options
// =============================================================================

// Creates ModelSync options initialized to the C++ engine defaults (the same
// defaults used when NULL is passed for `options`).
//
// @param out_options On success, receives the newly created options handle,
//   owned by the caller; release with `litert_lm_model_sync_options_delete`.
//   Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `out_options` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_options_create(
    LiteRtLmModelSyncOptions** out_options);

// Deletes a ModelSync options handle. NULL-safe.
//
// @param options The options handle to delete, or NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_model_sync_options_delete(LiteRtLmModelSyncOptions* options);

// Sets the change-ratio threshold (0.0 to 1.0) above which ModelSync falls back
// to a full download instead of differential slice assembly (default: 0.80).
// Passing 0.0 disables threshold-based full-download fallback.
//
// @param options The options handle to modify.
// @param threshold Ratio in `[0.0, 1.0]`.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL or `threshold` is outside `[0.0, 1.0]` or NaN.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_options_set_full_download_threshold(
    LiteRtLmModelSyncOptions* options, double threshold);

// Sets whether to delete the local model upfront before downloading when disk
// space is constrained (default: false).
//
// @param options The options handle to modify.
// @param clean True to enable in-place cleanup on constrained storage.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_options_set_clean(
    LiteRtLmModelSyncOptions* options, bool clean);

// Sets whether to verify available disk space before staging downloads
// (default: true).
//
// @param options The options handle to modify.
// @param check_disk_space True to perform pre-flight disk space checks.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_options_set_check_disk_space(
    LiteRtLmModelSyncOptions* options, bool check_disk_space);

// Sets whether `litert_lm_model_sync_cache_sync` falls back to an existing
// cached model when remote fetches fail with `kLiteRtLmStatusUnavailable` or
// `kLiteRtLmStatusDeadlineExceeded` (default: true).
//
// @param options The options handle to modify.
// @param allow_offline_fallback True to allow offline cached fallback.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_options_set_allow_offline_fallback(
    LiteRtLmModelSyncOptions* options, bool allow_offline_fallback);

// =============================================================================
// Sync Plan
// =============================================================================

// Computes a sync plan comparing a local file against a remote fetcher.
//
// A local model that is missing, unreadable, or has an invalid header is
// ignored rather than reported as an error: every section is then planned for
// download.
//
// @param local_model_path Path to the local .litertlm file (can be NULL or
//   empty).
// @param remote_fetcher Remote RangeFetcher callback.
// @param user_data User context pointer passed to remote_fetcher.
// @param options Optional configuration (or NULL for defaults).
// @param out_plan On success, receives the newly allocated plan, owned by the
//   caller; release with `litert_lm_model_sync_plan_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `remote_fetcher` or `out_plan` is NULL; otherwise the code of the error
//   that prevented the plan from being computed (e.g. a failed range fetch or
//   an invalid model header).
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_plan_create_from_file(
    const char* local_model_path,
    LiteRtLmModelSyncRangeFetcherCallback remote_fetcher, void* user_data,
    const LiteRtLmModelSyncOptions* options, LiteRtLmModelSyncPlan** out_plan);

// Releases a sync plan handle. NULL-safe.
//
// @param plan The sync plan to delete, or NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_model_sync_plan_delete(LiteRtLmModelSyncPlan* plan);

// Gets whether the local model is already completely up-to-date with remote.
//
// @param plan The sync plan to inspect.
// @param out_is_up_to_date On success, receives true if nothing needs to be
//   downloaded, false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `plan` or `out_is_up_to_date` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_plan_is_up_to_date(
    const LiteRtLmModelSyncPlan* plan, bool* out_is_up_to_date);

// Gets the total number of bytes to download over the network.
//
// @param plan The sync plan to inspect.
// @param out_bytes On success, receives the number of bytes; not written on
//   failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `plan` or `out_bytes` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_plan_get_bytes_to_download(
    const LiteRtLmModelSyncPlan* plan, uint64_t* out_bytes);

// Gets the total number of bytes to reuse from the local model.
//
// @param plan The sync plan to inspect.
// @param out_bytes On success, receives the number of bytes; not written on
//   failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `plan` or `out_bytes` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_plan_get_bytes_to_reuse(
    const LiteRtLmModelSyncPlan* plan, uint64_t* out_bytes);

// Gets the total size of the remote model in bytes.
//
// @param plan The sync plan to inspect.
// @param out_bytes On success, receives the size in bytes; not written on
//   failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `plan` or `out_bytes` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_plan_get_total_remote_size(
    const LiteRtLmModelSyncPlan* plan, uint64_t* out_bytes);

// Gets a human-readable summary of the sync plan.
//
// @param plan The sync plan to format.
// @param out_formatted_string On success, receives the NUL-terminated summary.
//   The string is owned by `plan` and valid until `plan` is deleted. Set to
//   NULL on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `plan` or `out_formatted_string` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_plan_format(
    const LiteRtLmModelSyncPlan* plan, const char** out_formatted_string);

// =============================================================================
// Execution & Cache Management
// =============================================================================

// Synchronizes `local_model_path` against `remote_fetcher`, writing the result
// to `output_model_path`. If `local_model_path == output_model_path`, updates
// in-place atomically.
//
// @param local_model_path Path to the local .litertlm file to reuse unchanged
//   sections from (can be NULL or empty).
// @param output_model_path Path that the synchronized model is written to;
//   must not be NULL or empty.
// @param remote_fetcher Remote RangeFetcher callback.
// @param user_data User context pointer passed to remote_fetcher.
// @param options Optional configuration (or NULL for defaults).
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `output_model_path` or `remote_fetcher` is NULL, or `output_model_path`
//   is empty; otherwise the code of the error that stopped the update.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_update(
    const char* local_model_path, const char* output_model_path,
    LiteRtLmModelSyncRangeFetcherCallback remote_fetcher, void* user_data,
    const LiteRtLmModelSyncOptions* options);

// High-level cache manager: checks `cache_dir` for `model_id`, differentially
// updates it, and returns an opaque `LiteRtLmModelSyncResult` containing the
// cached model path.
//
// If `allow_offline_fallback` is enabled (the default) and `remote_fetcher`
// fails with `kLiteRtLmStatusUnavailable` or `kLiteRtLmStatusDeadlineExceeded`,
// returns `kLiteRtLmStatusOk` with a result pointing to the existing cached
// model path if a cached file with a valid LiteRT-LM header already exists on
// disk.
//
// @param cache_dir Directory that holds the cached models; created if it does
//   not exist. Must not be NULL or empty.
// @param model_id Identifier of the model within `cache_dir`; must not be NULL
//   or empty.
// @param remote_fetcher Remote RangeFetcher callback.
// @param user_data User context pointer passed to remote_fetcher.
// @param options Optional configuration (or NULL for defaults).
// @param out_result On success, receives a newly allocated result handle,
//   owned by the caller; release with `litert_lm_model_sync_result_delete`.
//   Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `cache_dir`, `model_id`, `remote_fetcher`, or `out_result` is NULL, or if
//   `cache_dir` or `model_id` is empty; kLiteRtLmStatusResourceExhausted if
//   disk space is insufficient (when `check_disk_space` is enabled); otherwise
//   the code of the error that stopped the sync.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_cache_sync(
    const char* cache_dir, const char* model_id,
    LiteRtLmModelSyncRangeFetcherCallback remote_fetcher, void* user_data,
    const LiteRtLmModelSyncOptions* options,
    LiteRtLmModelSyncResult** out_result);

// Releases a sync result handle. NULL-safe.
//
// @param result The sync result to delete, or NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_model_sync_result_delete(LiteRtLmModelSyncResult* result);

// Gets the filesystem path of the synchronized cached model file.
//
// @param result The sync result to inspect.
// @param out_cached_path On success, receives a NUL-terminated UTF-8 path
//   owned by `result` and valid until `result` is deleted. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `result` or `out_cached_path` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_model_sync_result_get_cached_path(
    const LiteRtLmModelSyncResult* result, const char** out_cached_path);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_MODEL_SYNC_H_
