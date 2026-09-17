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

#include "c/error_reporter.h"

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/error_reporter_internal.h"

namespace {

struct ThreadLocalError {
  LiteRtLmStatusCode code = kLiteRtLmStatusOk;
  std::string message;
};

ThreadLocalError& GetThreadLocalError() {
  static thread_local ThreadLocalError error;
  return error;
}

// Releases the buffer rather than just resetting the size, so that a thread
// that once reported a large error message does not hold onto that allocation
// for its entire lifetime. `litert_lm_clear_last_error` documents this.
void ResetThreadLocalError(ThreadLocalError& error) {
  error.code = kLiteRtLmStatusOk;
  std::string().swap(error.message);
}

}  // namespace

namespace litert::lm::c {

// The canonical error codes and the LiteRtLmStatusCode enumerators are required
// to be numerically identical, which is what makes ToLiteRtLmStatusCode below a
// plain cast rather than a switch. These assertions are the enforcement.
static_assert(kLiteRtLmStatusOk == static_cast<int>(absl::StatusCode::kOk));
static_assert(kLiteRtLmStatusCancelled ==
              static_cast<int>(absl::StatusCode::kCancelled));
static_assert(kLiteRtLmStatusUnknown ==
              static_cast<int>(absl::StatusCode::kUnknown));
static_assert(kLiteRtLmStatusInvalidArgument ==
              static_cast<int>(absl::StatusCode::kInvalidArgument));
static_assert(kLiteRtLmStatusDeadlineExceeded ==
              static_cast<int>(absl::StatusCode::kDeadlineExceeded));
static_assert(kLiteRtLmStatusNotFound ==
              static_cast<int>(absl::StatusCode::kNotFound));
static_assert(kLiteRtLmStatusAlreadyExists ==
              static_cast<int>(absl::StatusCode::kAlreadyExists));
static_assert(kLiteRtLmStatusPermissionDenied ==
              static_cast<int>(absl::StatusCode::kPermissionDenied));
static_assert(kLiteRtLmStatusResourceExhausted ==
              static_cast<int>(absl::StatusCode::kResourceExhausted));
static_assert(kLiteRtLmStatusFailedPrecondition ==
              static_cast<int>(absl::StatusCode::kFailedPrecondition));
static_assert(kLiteRtLmStatusAborted ==
              static_cast<int>(absl::StatusCode::kAborted));
static_assert(kLiteRtLmStatusOutOfRange ==
              static_cast<int>(absl::StatusCode::kOutOfRange));
static_assert(kLiteRtLmStatusUnimplemented ==
              static_cast<int>(absl::StatusCode::kUnimplemented));
static_assert(kLiteRtLmStatusInternal ==
              static_cast<int>(absl::StatusCode::kInternal));
static_assert(kLiteRtLmStatusUnavailable ==
              static_cast<int>(absl::StatusCode::kUnavailable));
static_assert(kLiteRtLmStatusDataLoss ==
              static_cast<int>(absl::StatusCode::kDataLoss));
static_assert(kLiteRtLmStatusUnauthenticated ==
              static_cast<int>(absl::StatusCode::kUnauthenticated));

LiteRtLmStatusCode ToLiteRtLmStatusCode(absl::StatusCode code) {
  // absl::StatusCode reserves a value for future expansion, so a value outside
  // the canonical range is possible in principle. Fold it to
  // kLiteRtLmStatusUnknown to honor the closed-set contract documented in
  // error_reporter.h.
  const int value = static_cast<int>(code);
  if (value < static_cast<int>(kLiteRtLmStatusOk) ||
      value > static_cast<int>(kLiteRtLmStatusUnauthenticated)) {
    return kLiteRtLmStatusUnknown;
  }
  return static_cast<LiteRtLmStatusCode>(value);
}

void SetLastError(const absl::Status& status) {
  auto& error = GetThreadLocalError();
  if (status.ok()) {
    ResetThreadLocalError(error);
    return;
  }
  error.code = ToLiteRtLmStatusCode(status.code());
  error.message = status.ToString();
}

void SetLastError(absl::StatusCode code, absl::string_view message) {
  auto& error = GetThreadLocalError();
  error.code = ToLiteRtLmStatusCode(code);
  error.message = std::string(message);
}

}  // namespace litert::lm::c

extern "C" {

const char* litert_lm_get_last_error_message(void) {
  const auto& error = GetThreadLocalError();
  if (error.message.empty()) {
    return nullptr;
  }
  return error.message.c_str();
}

int litert_lm_get_last_error_code(void) { return GetThreadLocalError().code; }

void litert_lm_clear_last_error(void) {
  ResetThreadLocalError(GetThreadLocalError());
}

}  // extern "C"
