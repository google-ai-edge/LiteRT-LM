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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STREAMED_WEIGHTS_UPLOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STREAMED_WEIGHTS_UPLOADER_H_

#include <memory>

#include "litert/cc/litert_options.h"  // from @litert
#include "runtime/util/data_stream.h"

namespace litert::lm {

// Implements LiteRT's WebWeightUploader by reading from a DataStream. Only
// defined on the Web.
class DataStreamWebWeightUploader;

// Uploads the external weights of one model to WebGPU buffers on the Web by
// reading them from the DataStream of the model's TFLiteWeights section, so
// that the weights never need to be held in host memory.
//
// Create one uploader per model being compiled, set it on the model's
// compilation options, and keep it alive until the model has been compiled:
//
//   StreamedWeightsUploader uploader(std::move(weights_stream));
//   uploader.SetOnOptions(compilation_options);
//   auto compiled_model =
//       CompiledModel::Create(env, model_buffer, compilation_options);
//
// LiteRT's WebGPU delegate calls the uploader while the model is being
// compiled. Since each uploader only reads from its own stream, models can be
// compiled concurrently. Off the Web, models are not compiled from streamed
// weights, and `SetOnOptions()` does nothing.
//
// The compiled model keeps LiteRT's side of the uploader alive, and LiteRT may
// re-apply the delegate after the model has been created (e.g. when an input
// is resized). Since a stream can only be read once, destroying the
// StreamedWeightsUploader drops the stream, and any later upload fails instead
// of reading from a stream that is gone.
class StreamedWeightsUploader {
 public:
  // Creates an uploader which reads the weights from `stream`. The weights are
  // read in increasing order of offset, and discarded from the stream once
  // read.
  explicit StreamedWeightsUploader(std::shared_ptr<DataStream> stream);
  // Drops the stream, which discards whatever was not uploaded.
  ~StreamedWeightsUploader();

  StreamedWeightsUploader(const StreamedWeightsUploader&) = delete;
  StreamedWeightsUploader& operator=(const StreamedWeightsUploader&) = delete;

  // Sets this uploader as the uploader of the external weights of the model
  // which is compiled with `options`.
  void SetOnOptions(litert::Options& options);

 private:
#ifdef __EMSCRIPTEN__
  // Holds the stream. Shared with the compiled model's weight loader.
  std::shared_ptr<DataStreamWebWeightUploader> web_uploader_;
#else
  std::shared_ptr<DataStream> stream_;
#endif  // __EMSCRIPTEN__
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STREAMED_WEIGHTS_UPLOADER_H_
