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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STREAMED_WEIGHTS_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STREAMED_WEIGHTS_MANAGER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/util/data_stream.h"

namespace litert::lm {

// Global registry and state manager for streaming external weights of
// GPU/WebGPU submodels.
//
// During streamed WebGPU model loading, weight sections from the .litertlm
// DataStream are not buffered in WASM host memory ahead of time. Instead, C++
// registers the active submodel's DataStream via `StoreWeightsStream()` and
// sets the active model type wire string via `SetCurrentlyCompilingModel()`
// prior to invoking `litert::CompiledModel::Create()`.
//
// When LiteRT's WebGPU delegate compiles the model, it invokes the registered
// C++ WebGPU weight upload callback (`UploadStoredWeightsOnWeb`), which queries
// `GetCurrentlyCompilingModel()`, calls `ReadStoredWeights()` in chunks, and
// writes directly to `wgpu::Queue::WriteBuffer`.

// Sets the model type wire string (e.g. "tf_lite_embedder") of the submodel
// currently being compiled so the WebGPU weight streaming callback knows which
// stored stream to read from. An empty string means no submodel is being
// compiled.
//
// args:
//   model_type_str: The model type wire string, or empty to reset.
void SetCurrentlyCompilingModel(absl::string_view model_type_str);

// Sets the capability-scoped Protobuf model type of the submodel currently
// being compiled. See SetCurrentlyCompilingModel(absl::string_view).
//
// args:
//   model_type: A capability-scoped Protobuf TfLiteModelType enum value.
template <typename TfLiteModelTypeT,
          typename = std::enable_if_t<kIsTfLiteModelType<TfLiteModelTypeT>>>
void SetCurrentlyCompilingModel(TfLiteModelTypeT model_type) {
  SetCurrentlyCompilingModel(TfLiteModelTypeToWireString(model_type));
}

// Returns the model type wire string of the submodel currently being compiled,
// or an empty string if unknown.
std::string GetCurrentlyCompilingModel();

// Registers a DataStream containing external weights for `model_type_str` so
// that it can be read on demand during WebGPU model compilation.
//
// args:
//   model_type_str: The model type wire string (e.g. "tf_lite_embedder").
//   stream: The DataStream to read the external weights from.
void StoreWeightsStream(absl::string_view model_type_str,
                        std::shared_ptr<DataStream> stream);

// Registers a DataStream for a capability-scoped Protobuf model type. See
// StoreWeightsStream(absl::string_view, std::shared_ptr<DataStream>).
template <typename TfLiteModelTypeT,
          typename = std::enable_if_t<kIsTfLiteModelType<TfLiteModelTypeT>>>
void StoreWeightsStream(TfLiteModelTypeT model_type,
                        std::shared_ptr<DataStream> stream) {
  StoreWeightsStream(TfLiteModelTypeToWireString(model_type),
                     std::move(stream));
}

// Reads `size` bytes at `offset` from the stored weights stream for
// `model_type_str` into `buffer`, discarding the read bytes from the stream.
//
// args:
//   model_type_str: The model type wire string (e.g. "tf_lite_embedder").
//   offset: The byte offset to read from.
//   size: The number of bytes to read.
//   buffer: The destination buffer, at least `size` bytes large.
// returns:
//   OkStatus on success, NotFoundError if nothing is stored for the model
//   type, or OutOfRangeError if the read exceeds the stored stream.
absl::Status ReadStoredWeights(absl::string_view model_type_str,
                               uint64_t offset, uint64_t size, void* buffer);

// Discards any remaining unread bytes and removes the stored weights stream
// for `model_type_str`.
//
// args:
//   model_type_str: The model type wire string (e.g. "tf_lite_embedder").
// returns:
//   OkStatus.
absl::Status ClearStoredWeightsStream(absl::string_view model_type_str);

// Discards any remaining unread bytes and removes the stored weights stream
// for a capability-scoped Protobuf model type. See
// ClearStoredWeightsStream(absl::string_view).
template <typename TfLiteModelTypeT,
          typename = std::enable_if_t<kIsTfLiteModelType<TfLiteModelTypeT>>>
absl::Status ClearStoredWeightsStream(TfLiteModelTypeT model_type) {
  return ClearStoredWeightsStream(TfLiteModelTypeToWireString(model_type));
}

// Discards any remaining unread bytes and clears all stored weight streams.
absl::Status ClearStoredWeightsStreams();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STREAMED_WEIGHTS_MANAGER_H_
