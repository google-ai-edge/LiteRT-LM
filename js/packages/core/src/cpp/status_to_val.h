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

#ifndef THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_STATUS_TO_VAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_STATUS_TO_VAL_H_

// Helpers for returning absl::Status errors to JS as values.
//
// Bindings must not throw JS exceptions from C++. The Wasm build has no C++
// exception support, so throwing skips the destructors of every C++ frame
// between the throw and JS (leaking heap memory) and never restores the Wasm
// stack pointer (leaking Wasm stack until it overflows). Instead, bindings
// that can fail return an `emscripten::val` holding either the result or a JS
// `Error`, and the TS wrapper throws the error (see `unwrapWasmResult` in
// wasm_utils.ts).

#ifdef __EMSCRIPTEN__

#include <emscripten/val.h>
#include <emscripten/wire.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litertlm_web {

namespace internal {

template <typename T>
struct IsUniquePtr : std::false_type {};
template <typename T, typename D>
struct IsUniquePtr<std::unique_ptr<T, D>> : std::true_type {};

// Converts `value` to a JS value, transferring ownership of C++ objects to JS.
template <typename T>
emscripten::val ToVal(T value) {
  if constexpr (IsUniquePtr<T>::value) {
    return emscripten::val(std::move(value));
  } else if constexpr (std::is_base_of_v<
                           emscripten::internal::GenericBindingType<T>,
                           emscripten::internal::BindingType<T>>) {
    // `T` is a class bound with embind (e.g. `class_` or `value_object`).
    // Constructing a val from `T` directly would copy it, so move it into a
    // unique_ptr and hand that to JS instead.
    return emscripten::val(std::make_unique<T>(std::move(value)));
  } else {
    return emscripten::val(std::move(value));
  }
}

}  // namespace internal

// Returns a JS `Error` describing `status`, which must not be OK. The error's
// `code` property holds the name of the status code, e.g. "INVALID_ARGUMENT".
inline emscripten::val StatusToJsError(const absl::Status& status) {
  emscripten::val error =
      emscripten::val::global("Error").new_(std::string(status.message()));
  error.set("code", absl::StatusCodeToString(status.code()));
  return error;
}

// Returns `undefined` if `status` is OK, or a JS `Error` otherwise.
inline emscripten::val StatusToVal(const absl::Status& status) {
  if (!status.ok()) {
    return StatusToJsError(status);
  }
  return emscripten::val::undefined();
}

// Returns the value held by `status_or` as a JS value, or a JS `Error` if it
// holds an error. Ownership of C++ objects is transferred to JS, which must
// delete them.
template <typename T>
emscripten::val StatusOrToVal(absl::StatusOr<T> status_or) {
  if (!status_or.ok()) {
    return StatusToJsError(status_or.status());
  }
  return internal::ToVal<T>(*std::move(status_or));
}

}  // namespace litertlm_web

#endif  // __EMSCRIPTEN__
#endif  // THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_STATUS_TO_VAL_H_
