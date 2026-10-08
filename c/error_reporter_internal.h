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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_INTERNAL_H_

#include <utility>  // IWYU pragma: keep, used by LITERT_LM_C_ASSIGN_OR_RETURN

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/error_reporter.h"

namespace litert::lm::c {

// Maps a canonical `absl::StatusCode` to the corresponding
// `LiteRtLmStatusCode`. Any value outside the canonical range is mapped to
// `kLiteRtLmStatusUnknown`.
//
// This is the single conversion point between the C++ and C status domains.
// Prefer it over an open-coded `static_cast`, so that C API entry points return
// a value that is statically known to be a valid `LiteRtLmStatusCode`.
LiteRtLmStatusCode ToLiteRtLmStatusCode(absl::StatusCode code);

// Sets the thread-local last error from an absl::Status.
// If status is OK, clears the thread-local error.
void SetLastError(const absl::Status& status);

// Sets the thread-local last error with a specific StatusCode and message.
void SetLastError(absl::StatusCode code, absl::string_view message);

// Converts `status` into the `LiteRtLmStatusCode` that a C API entry point
// returns.
//
// A non-OK status is recorded as the calling thread's last error and its code
// is returned. An OK status returns `kLiteRtLmStatusOk` and leaves the
// thread-local error state untouched, honoring the "success does not clear the
// last error" contract documented in error_reporter.h.
LiteRtLmStatusCode ToCStatus(const absl::Status& status);

// Records an error with `code` and `message` as the calling thread's last error
// and returns `code` as the `LiteRtLmStatusCode` that a C API entry point
// returns.
LiteRtLmStatusCode ReturnError(absl::StatusCode code,
                               absl::string_view message);

}  // namespace litert::lm::c

// The macros below are for use inside `LiteRtLmStatusCode`-returning C API
// entry points only.

// Returns `kLiteRtLmStatusInvalidArgument` from the enclosing function, after
// recording an error naming `arg`, if `arg` is NULL.
#define LITERT_LM_C_RETURN_IF_NULL(arg)                                     \
  do {                                                                      \
    if ((arg) == nullptr) {                                                 \
      return ::litert::lm::c::ReturnError(                                  \
          ::absl::StatusCode::kInvalidArgument, #arg " must not be NULL."); \
    }                                                                       \
  } while (false)

// Evaluates `expr`, which must yield an `absl::Status`. If it is not OK,
// records it as the last error and returns its code from the enclosing
// function.
#define LITERT_LM_C_RETURN_IF_ERROR(expr)                     \
  do {                                                        \
    const ::absl::Status litert_lm_c_status_ = (expr);        \
    if (!litert_lm_c_status_.ok()) {                          \
      return ::litert::lm::c::ToCStatus(litert_lm_c_status_); \
    }                                                         \
  } while (false)

#define LITERT_LM_C_INTERNAL_CONCAT_IMPL(a, b) a##b
#define LITERT_LM_C_INTERNAL_CONCAT(a, b) LITERT_LM_C_INTERNAL_CONCAT_IMPL(a, b)

// Evaluates `rexpr`, which must yield an `absl::StatusOr<T>`. On error, records
// it as the last error and returns its code from the enclosing function.
// Otherwise moves the value into `lhs`, which may be a declaration.
#define LITERT_LM_C_ASSIGN_OR_RETURN(lhs, rexpr)                         \
  LITERT_LM_C_INTERNAL_ASSIGN_OR_RETURN_IMPL(                            \
      LITERT_LM_C_INTERNAL_CONCAT(litert_lm_c_statusor_, __LINE__), lhs, \
      rexpr)

#define LITERT_LM_C_INTERNAL_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                               \
  if (!statusor.ok()) {                                                  \
    return ::litert::lm::c::ToCStatus(statusor.status());                \
  }                                                                      \
  lhs = *std::move(statusor)

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_INTERNAL_H_
