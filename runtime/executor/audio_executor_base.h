// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_BASE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_BASE_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

class AudioExecutorBase {
 public:
  virtual ~AudioExecutorBase() = default;

  // ------------Encode APIs------------:
  // Basic API to trigger the "encode" process.
  // Input is audio spectrogram tensor with shape `[batch, height, width,
  // channels]`. Output is audio data which contains main embeddings with shape
  // `[batch, 1, num_audio_tokens, model_dimension]`.
  virtual absl::StatusOr<::litert::lm::ExecutorAudioData> Encode(
      const litert::TensorBuffer& spectrogram_tensor) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_BASE_H_
