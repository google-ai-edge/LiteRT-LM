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

#include "third_party/odml/litert_lm/runtime/executor/llm_executor.h"

#include <iostream>

#include "third_party/odml/litert_lm/runtime/util/logging_tensor_buffer.h"

namespace litert::lm {

std::ostream& operator<<(std::ostream& os, const Inputs& inputs) {
  os << "token_ids: " << inputs.text_input << "\n";
  if (inputs.vision_input.has_value()) {
    os << "vision_embeddings: " << inputs.vision_input.value() << "\n";
  }
  if (inputs.audio_input.has_value()) {
    os << "audio_embeddings: " << inputs.audio_input.value() << "\n";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const TextInput& text_input) {
  os << "token_ids: " << text_input.token_ids << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const VisionInput& vision_input) {
  if (vision_input.embeddings.has_value()) {
    os << "embeddings: " << *vision_input.embeddings << "\n";
  }
  if (vision_input.per_layer_embeddings.has_value()) {
    os << "per_layer_embeddings: " << *vision_input.per_layer_embeddings
       << "\n";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const AudioInput& audio_input) {
  if (audio_input.embeddings.has_value()) {
    os << "embeddings: " << *audio_input.embeddings << "\n";
  }
  if (audio_input.per_layer_embeddings.has_value()) {
    os << "per_layer_embeddings: " << *audio_input.per_layer_embeddings << "\n";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const PrefillQueryParams& params) {
  os << "current_step: " << params.current_step
     << "\nwait_for_completion: " << params.wait_for_completion;
  if (params.cancel != nullptr) {
    os << "\ncancel: " << *params.cancel << "\n";
  }
  return os;
}

}  // namespace litert::lm
