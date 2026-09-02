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

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/error_reporter_internal.h"

namespace {

struct ThreadLocalError {
  int code = 0;
  std::string message;
};

ThreadLocalError& GetThreadLocalError() {
  static thread_local absl::NoDestructor<ThreadLocalError> error;
  return *error;
}

}  // namespace

namespace litert::lm::c {

void SetLastError(const absl::Status& status) {
  auto& error = GetThreadLocalError();
  if (status.ok()) {
    error.code = 0;
    error.message.clear();
    return;
  }
  error.code = static_cast<int>(status.code());
  error.message = status.ToString();
}

void SetLastError(absl::StatusCode code, absl::string_view message) {
  SetLastError(static_cast<int>(code), message);
}

void SetLastError(int code, absl::string_view message) {
  auto& error = GetThreadLocalError();
  error.code = code;
  error.message = std::string(message);
}

}  // namespace litert::lm::c

extern "C" {

const char* litert_lm_get_last_error_message(void) {
  const auto& error = GetThreadLocalError();
  if (error.code == 0 && error.message.empty()) {
    return nullptr;
  }
  return error.message.c_str();
}

int litert_lm_get_last_error_code(void) { return GetThreadLocalError().code; }

void litert_lm_clear_last_error(void) {
  auto& error = GetThreadLocalError();
  error.code = 0;
  error.message.clear();
}

}  // extern "C"
