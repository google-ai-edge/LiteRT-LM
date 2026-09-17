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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_API_EXPORT_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_API_EXPORT_H_

// Marks a declaration as part of the public LiteRT LM C API, ensuring it is
// exported from the shared library.
//
// For Windows, __declspec( dllexport ) is required to export function in .dll.
// https://learn.microsoft.com/en-us/cpp/cpp/using-dllimport-and-dllexport-in-cpp-classes?view=msvc-170
//
// _WIN32 is defined as 1 when the compilation target is 32-bit ARM, 64-bit ARM,
// x86, x64, or ARM64EC. Otherwise, undefined.
// https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros
//
// Clients may define `LITERT_LM_C_API_EXPORT` before including any LiteRT LM
// header to override the default, e.g. with `__declspec(dllimport)` when
// consuming the Windows .dll, or with an empty definition when linking against
// the API statically.
#ifndef LITERT_LM_C_API_EXPORT
#if defined(_WIN32)
#define LITERT_LM_C_API_EXPORT __declspec(dllexport)
#else
// Ensure symbols are exported when building the shared library with
// -fvisibility=hidden.
#define LITERT_LM_C_API_EXPORT __attribute__((visibility("default")))
#endif  // defined(_WIN32)
#endif  // LITERT_LM_C_API_EXPORT

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_API_EXPORT_H_
