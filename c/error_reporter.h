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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_H_

#include <stdint.h>

#if defined(__APPLE__)
#include "api_export.h"  // NOLINT
#else
#include "c/api_export.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Error Handling Model and Conventions
// =============================================================================
//
// LiteRT LM C API uses thread-local error reporting modeled after standard
// POSIX errno conventions:
//
// 1. Thread Locality:
//    Errors are stored in thread-local storage. Each thread maintains its own
//    independent error code and message. Concurrent calls across different
//    threads do not overwrite or interfere with each other's error state.
//
// 2. Error Setting on Failure Only:
//    The error state is updated ONLY when an API function fails. Functions
//    that succeed DO NOT clear, reset, or modify the existing error state.
//
// 3. User Expectations & Error Extraction Logic:
//    - Callers MUST check the return value of an API function first (such as
//      verifying if a returned pointer is NULL or an operation returns a
//      failure indicator) to determine whether an operation actually failed.
//    - Callers MUST NOT rely on `litert_lm_get_last_error_code() != 0` or
//      `litert_lm_get_last_error_message() != NULL` to infer failure, because
//      a successful call following a failed one will leave the prior error
//      intact in thread-local storage.
//    - Error retrieval must be performed on the SAME thread that called the
//      failing API function before that thread encounters another failure or
//      clears the error.
//
// 4. Memory Ownership and Lifetime:
//    - The error message string returned by `litert_lm_get_last_error_message`
//      is owned by the library and resides in thread-local storage.
//    - The caller MUST NOT attempt to free, delete, or modify this pointer.
//    - The pointer remains valid only until the next failing LiteRT LM C API
//      call on the same thread, until `litert_lm_clear_last_error` is called,
//      or until the calling thread terminates.
//    - If the error message needs to be retained across subsequent API calls or
//      transferred to another thread, callers MUST create a copy (e.g.,
//      via strdup()).
//
// 5. Error Clearing Logic (Optional / Defensive):
//    - Calling `litert_lm_clear_last_error` explicitly resets the calling
//      thread's error state (code = 0 / kOk, message = NULL).
//    - Calling this function is NOT required during normal error handling:
//      callers determine failure from API return values, and subsequent
//      errors will automatically overwrite previous error state.
//    - This function is provided as an optional utility for defensive
//      programming (e.g., resetting state before a call sequence or between
//      test cases, analogous to setting `errno = 0` in POSIX) or to release
//      thread-local error message memory on long-lived threads.
// =============================================================================

// =============================================================================
// Status Codes
// =============================================================================
//
// Canonical status codes returned by `litert_lm_get_last_error_code()`.
//
// These values are 1:1 identical to Google canonical error codes
// (`absl::StatusCode` / `google.rpc.Code`).
//
// ABI stability:
//   - These numeric values are frozen. They MUST NOT be renumbered or
//     reordered.
//   - This is a closed set mirroring the canonical error codes; no new
//     enumerators are expected. Callers that switch on the value SHOULD
//     nevertheless provide a `default:` branch and treat any unrecognized value
//     as `kLiteRtLmStatusUnknown`, so that a consumer built against this header
//     stays forward compatible with a newer shared library.
//   - Values always cross the ABI as a non-negative `int`. Do not test for
//     negative values: a C enum whose enumerators are all non-negative may be
//     given an unsigned underlying type, which makes `value < 0` always false.
//
// Naming: unlike the other enums in this API (for example
// `LiteRtLmSamplerType` / `kLiteRtLmSamplerTypeTopK`), the enumerators here are
// deliberately prefixed with `kLiteRtLmStatus` rather than the full type name
// `kLiteRtLmStatusCode`, to keep the very frequently used constants readable.
// This is an intentional exception; do not "fix" it, as renaming would be a
// source-breaking change for all consumers.
//
// Added in version 0.2.0.
typedef enum LiteRtLmStatusCode {
  kLiteRtLmStatusOk = 0,
  kLiteRtLmStatusCancelled = 1,
  kLiteRtLmStatusUnknown = 2,
  kLiteRtLmStatusInvalidArgument = 3,
  kLiteRtLmStatusDeadlineExceeded = 4,
  kLiteRtLmStatusNotFound = 5,
  kLiteRtLmStatusAlreadyExists = 6,
  kLiteRtLmStatusPermissionDenied = 7,
  kLiteRtLmStatusResourceExhausted = 8,
  kLiteRtLmStatusFailedPrecondition = 9,
  kLiteRtLmStatusAborted = 10,
  kLiteRtLmStatusOutOfRange = 11,
  kLiteRtLmStatusUnimplemented = 12,
  kLiteRtLmStatusInternal = 13,
  kLiteRtLmStatusUnavailable = 14,
  kLiteRtLmStatusDataLoss = 15,
  kLiteRtLmStatusUnauthenticated = 16,
} LiteRtLmStatusCode;

// Returns the last error message recorded on the calling thread.
//
// Extraction Precondition & Expectations:
// Callers should only call this function AFTER an API function has signaled
// failure via its return value (such as returning NULL or an error status).
// Following standard C conventions, functions that succeed DO NOT clear or
// modify the error state; inspecting this function without verifying a return
// failure may return stale error messages from an earlier failure.
//
// Thread Locality:
// The error message is stored in thread-local storage and must be retrieved
// from the same thread that executed the failed API function.
//
// Memory Ownership & Lifetime:
// The returned pointer is owned by the library (pointing to internal
// thread-local storage). The caller MUST NOT free, modify, or deallocate it.
// The pointer remains valid until the next LiteRT LM C API error occurs on the
// same thread, until `litert_lm_clear_last_error` is called on that thread, or
// until the thread terminates. If the caller needs the error message beyond
// this lifetime or across threads, the caller must make a copy (e.g. strdup).
//
// Returns:
// A null-terminated UTF-8 string containing the error description, or NULL if
// no error has occurred on the calling thread or if the error state has been
// cleared.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_get_last_error_message(void);

// Returns the last error code recorded on the calling thread.
//
// Extraction Precondition & Expectations:
// Callers should only call this function AFTER an API function has signaled
// failure via its return value (such as returning NULL or an error status).
// Following standard C conventions, functions that succeed DO NOT clear or
// modify the error state; inspecting this function without verifying a return
// failure may return a stale error code from an earlier failure.
//
// Thread Locality:
// The error code is stored in thread-local storage and must be retrieved
// from the same thread that executed the failed API function.
//
// Returns:
// The status code for the last failure on the calling thread, or
// `kLiteRtLmStatusOk` (0) if no error has occurred on this thread or if the
// error state has been cleared. The returned value is always one of the
// `LiteRtLmStatusCode` enumerators.
//
// The return type is `int` rather than `LiteRtLmStatusCode` so that the ABI
// does not depend on the compiler's choice of underlying type for the enum,
// which keeps this function easy to bind from other languages.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
int litert_lm_get_last_error_code(void);

// Clears the last error recorded on the calling thread, resetting the error
// message to NULL and the error code to `kLiteRtLmStatusOk` (0).
//
// Clearing Logic & Expectations:
// Calling this function is optional during normal API usage since return
// values indicate failure and any subsequent error automatically overwrites the
// thread-local error state. It is primarily intended as a defensive measure
// (e.g. establishing a clean baseline before an API call sequence or in test
// fixtures) or to promptly free thread-local memory held by error messages
// on long-lived worker threads.
//
// Thread Locality:
// Clears only the error state for the calling thread. Error states on other
// threads remain unaffected.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_clear_last_error(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_H_
