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

#ifndef THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_READABLE_STREAM_DATA_STREAM_H_
#define THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_READABLE_STREAM_DATA_STREAM_H_

#include <cstdint>
#include <memory>

#include "runtime/util/data_stream.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/val.h>

namespace litert::lm {

class ReadableStreamDataStream : public DataStream {
 public:
  ReadableStreamDataStream(emscripten::val js_stream_wrapper)
      : js_stream_wrapper_(js_stream_wrapper) {}

  static std::shared_ptr<ReadableStreamDataStream> Create(
      emscripten::val js_stream_wrapper) {
    return std::make_shared<ReadableStreamDataStream>(js_stream_wrapper);
  }

  absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                              uint64_t size) override;
  absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                               uint64_t size) override;
  absl::Status Discard(uint64_t offset, uint64_t size) override;

 private:
  emscripten::val js_stream_wrapper_;
};

}  // namespace litert::lm

#endif  // __EMSCRIPTEN__
#endif  // THIRD_PARTY_ODML_LITERT_LM_JS_PACKAGES_CORE_SRC_CPP_READABLE_STREAM_DATA_STREAM_H_
