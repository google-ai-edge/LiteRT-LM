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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm::c {

// Sets the thread-local last error from an absl::Status.
// If status is OK, clears the thread-local error.
void SetLastError(const absl::Status& status);

// Sets the thread-local last error with a specific StatusCode and message.
void SetLastError(absl::StatusCode code, absl::string_view message);

// Sets the thread-local last error with an integer error code and message.
void SetLastError(int code, absl::string_view message);

}  // namespace litert::lm::c

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_INTERNAL_H_
