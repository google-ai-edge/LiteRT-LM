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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_UTILS_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Utility function to get the properties of the audio executor from LiteRT
// Model. This function fetches the properties from the audio model by
// inspecting the model signature, input tensor names and tensor shapes.
//
// It assumes there is a `prev_mask` input tensor for the streaming audio
// encoder model. The number of elements in `prev_mask` tensor is the overlap
// size. The `segment_mask` tensor is used to infer the chunk size.
//
// This function throws an error if the model is not a valid audio encoder
// model.
//
// Args:
//   - model_resources: The model resources to inspect.
// Returns:
//   A AudioExecutorProperties object containing the properties of the audio
//   executor.
absl::StatusOr<AudioExecutorProperties>
GetAudioExecutorPropertiesFromModelResources(ModelResources& model_resources);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_UTILS_H_
