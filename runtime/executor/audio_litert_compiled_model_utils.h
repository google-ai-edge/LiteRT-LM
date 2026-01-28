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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_LITERT_COMPILED_MODEL_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_LITERT_COMPILED_MODEL_UTILS_H_

#include <ostream>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"

namespace litert::lm {

struct AudioExecutorProperties {
  // Whether the audio model is a streaming model.
  bool is_streaming_model = false;

  // The size of each streaming chunk.
  int streaming_chunk_size = 0;

  // The overlap size of each streaming chunk.
  int streaming_chunk_overlap_size = 0;
};

std::ostream& operator<<(std::ostream& os,
                         const AudioExecutorProperties& properties);

// Utility function to get the properties of the audio executor from LiteRT
// Model.
absl::StatusOr<AudioExecutorProperties>
GetAudioExecutorPropertiesFromModelResources(ModelResources& model_resources);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_LITERT_COMPILED_MODEL_UTILS_H_
