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

#ifndef THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_UNWRAP_STATUSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_UNWRAP_STATUSOR_H_

// This package depends on global_error_reporter, which is only defined for web.
#ifdef __EMSCRIPTEN__

#include <functional>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/js/packages/core/src/cpp/global_error_reporter.h"  // from @litert

namespace litertlm_web {

// Unwrap a StatusOr<T> and report any errors.
template <typename T>
T UnwrapStatusOr(absl::StatusOr<T> status_or) {
  if (!status_or.ok()) {
    // This throws an error in the JS side, and the C++ stack empties. The
    // function does not return, and control flow returns to JS.
    litert_web::GetGlobalErrorReporter()->ReportAndThrowError(
        status_or.status().message());
  }
  return std::move(status_or).value();
}

// Throw an error if the status is not ok.
void UnwrapStatus(absl::Status status) {
  if (!status.ok()) {
    // This throws an error in JS, and the stack empties immediately.
    // The function does not return.
    litert_web::GetGlobalErrorReporter()->ReportAndThrowError(status.message());
  }
}

}  // namespace litertlm_web

#endif  // __EMSCRIPTEN__
#endif  // THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_UNWRAP_STATUSOR_H_
