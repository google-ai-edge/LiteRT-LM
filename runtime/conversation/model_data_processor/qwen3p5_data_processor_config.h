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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN3P5_DATA_PROCESSOR_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN3P5_DATA_PROCESSOR_CONFIG_H_

#include <string>

namespace litert::lm {

struct Qwen3p5DataProcessorConfig {
  std::string code_fence_start = "<tool_call>";
  std::string code_fence_end = "</tool_call>";
  bool escape_fence_strings = true;
};

struct Qwen3p5DataProcessorArguments {};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN3P5_DATA_PROCESSOR_CONFIG_H_
