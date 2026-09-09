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

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/util/data_stream.h"

namespace litert::lm {

// Global registry and state manager for streaming external weights of
// GPU/WebGPU submodels.
//
// During streamed WebGPU model loading, weight sections from the .litertlm
// DataStream are not buffered in WASM host memory ahead of time. Instead, C++
// registers the active submodel's DataStream via `StoreWeightsStream()` and
// sets the active `ModelType` via `SetCurrentlyCompilingModel()` prior to
// invoking `litert::CompiledModel::Create()`.
//
// When LiteRT's WebGPU delegate compiles the model, it invokes a JavaScript
// callback (`setupStreamWeightsCallback` in `stream_utils.ts`), which queries
// `GetCurrentlyCompilingModel()` and calls `ReadStoredWeights()` across the
// WASM boundary to stream weight chunks directly into WebGPU buffers.

// Sets the ModelType of the submodel currently being compiled so the JavaScript
// WebGPU weight streaming callback knows which stored stream to read from.
void SetCurrentlyCompilingModel(ModelType model_type);

// Returns the ModelType of the submodel currently being compiled. Called by
// JavaScript during WebGPU model compilation.
ModelType GetCurrentlyCompilingModel();

// Registers a DataStream containing external weights for `model_type` so that
// it can be read on demand during WebGPU model compilation.
void StoreWeightsStream(ModelType model_type,
                        std::shared_ptr<DataStream> stream);

// Reads `size` bytes at `offset` from the stored weights stream for
// `model_type_int` into `buffer`, discarding the read bytes from the stream.
// Called by JavaScript (`wasm.readStoredWeights`) as part of streamed WebGPU
// model loading to upload weights directly into GPUBuffers.
absl::Status ReadStoredWeights(int model_type_int, uint64_t offset,
                               uint64_t size, void* buffer);

// Discards any remaining unread bytes and removes the stored weights stream
// for `model_type`.
absl::Status ClearStoredWeightsStream(ModelType model_type);

// Discards any remaining unread bytes and clears all stored weight streams.
absl::Status ClearStoredWeightsStreams();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STREAMED_WEIGHTS_MANAGER_H_
