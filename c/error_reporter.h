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

#ifdef __cplusplus
extern "C" {
#endif

// For Windows, __declspec( dllexport ) is required to export function in .dll.
#if defined(_WIN32)
#define LITERT_LM_C_API_EXPORT __declspec(dllexport)
#else
#define LITERT_LM_C_API_EXPORT __attribute__((visibility("default")))
#endif

// Returns the last error message recorded on the calling thread.
//
// The returned pointer is owned by the library, points to thread-local storage,
// and remains valid until the next LiteRT LM C API call on the same thread
// or until `litert_lm_clear_last_error` is called.
// Returns NULL if no error has occurred on this thread or if the error has been
// cleared.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_get_last_error_message(void);

// Returns the last error code (corresponding to absl::StatusCode integer value)
// recorded on the calling thread.
// Returns 0 (kOk) if no error has occurred on this thread or if the error has
// been cleared.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
int litert_lm_get_last_error_code(void);

// Clears the last error recorded on the calling thread, resetting the error
// message to NULL and the error code to 0.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_clear_last_error(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_H_
