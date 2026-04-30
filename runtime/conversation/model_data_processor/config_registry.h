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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_DATA_PROCESSOR_CONFIG_REGISTRY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_DATA_PROCESSOR_CONFIG_REGISTRY_H_

#include <variant>

#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#include "runtime/conversation/model_data_processor/function_gemma_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"

namespace litert::lm {

// DataProcessorConfig is a registry of all the model-specific data processor
// configs. The DataProcessorConfig is used to initialize the
// ModelDataProcessor.
using DataProcessorConfig =
    std::variant<Gemma3DataProcessorConfig, GenericDataProcessorConfig,
                 Qwen3DataProcessorConfig, FunctionGemmaDataProcessorConfig,
                 Gemma4DataProcessorConfig,
                 FastVlmDataProcessorConfig
                 >;

// DataProcessorArguments is a registry of all the model-specific data processor
// arguments. The DataProcessorArguments is used to pass arguments of single
// turn to the ModelDataProcessor during the conversation.
using DataProcessorArguments =
    std::variant<std::monostate, GenericDataProcessorArguments,
                 Gemma3DataProcessorArguments, Qwen3DataProcessorArguments,
                 FunctionGemmaDataProcessorArguments,
                 Gemma4DataProcessorArguments,
                 FastVlmDataProcessorArguments
                 >;

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_DATA_PROCESSOR_CONFIG_REGISTRY_H_
