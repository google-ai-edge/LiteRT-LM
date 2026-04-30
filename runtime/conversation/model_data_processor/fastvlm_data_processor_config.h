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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FASTVLM_DATA_PROCESSOR_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FASTVLM_DATA_PROCESSOR_CONFIG_H_

#include <string>

namespace litert::lm {

// Config for FastVlmDataProcessor.
struct FastVlmDataProcessorConfig {
  // The string for beginning of image token.
  std::string boi_token = " <start_of_image>";
  // The string for end of image token.
  std::string eoi_token = "<end_of_image>";

  int image_tensor_height = 1024;
  int image_tensor_width = 1024;
};

// Arguments for FastVlmDataProcessor.
struct FastVlmDataProcessorArguments {};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FASTVLM_DATA_PROCESSOR_CONFIG_H_
