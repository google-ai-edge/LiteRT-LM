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

#include "js/packages/core/src/cpp/readable_stream_data_stream.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/val.h>

#include <algorithm>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

using emscripten::val;

namespace {

absl::Status ParseJsError(const val& result) {
  val error = result["error"];

  if (!error.isUndefined() && !error.isNull()) {
    val::global("console").call<void>("warn", error);
    std::string msg = error["message"].as<std::string>();
    std::string name = error["name"].as<std::string>();

    return absl::InternalError(
        absl::StrCat("JS Stream Error [", name, "]: ", msg));
  }
  return absl::OkStatus();
}

}  // namespace

namespace litert::lm {

absl::Status ReadableStreamDataStream::ReadAndDiscard(void* buffer,
                                                      uint64_t offset,
                                                      uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }

  uintptr_t dest_address = reinterpret_cast<uintptr_t>(buffer);
  val result =
      js_stream_wrapper_.call<val>("readAndDiscard", dest_address, offset, size)
          .await();

  RETURN_IF_ERROR(ParseJsError(result));
  return absl::OkStatus();
}

absl::Status ReadableStreamDataStream::ReadAndPreserve(void* buffer,
                                                       uint64_t offset,
                                                       uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }

  uintptr_t dest_address = reinterpret_cast<uintptr_t>(buffer);
  val result = js_stream_wrapper_
                   .call<val>("readAndPreserve", dest_address, offset, size)
                   .await();

  RETURN_IF_ERROR(ParseJsError(result));
  return absl::OkStatus();
}

absl::Status ReadableStreamDataStream::Discard(uint64_t offset, uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }
  val result = js_stream_wrapper_.call<val>("discard", offset, size).await();

  RETURN_IF_ERROR(ParseJsError(result));
  return absl::OkStatus();
}
}  // namespace litert::lm

#endif  // __EMSCRIPTEN__
