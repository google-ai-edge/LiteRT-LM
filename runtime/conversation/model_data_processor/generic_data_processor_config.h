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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GENERIC_DATA_PROCESSOR_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GENERIC_DATA_PROCESSOR_CONFIG_H_

#include <optional>
#include <string>

#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/image_preprocessor.h"

namespace litert::lm {

using ImagePreprocessParameter = ::litert::support::ImagePreprocessParameter;
using AudioPreprocessorConfig = ::litert::support::AudioPreprocessorConfig;

// Configuration for multimodal processing in the generic data processor.
struct MultimodalConfig {
  // Image modality related config.
  // Whether to enable image modality.
  bool image_enabled = false;
  // Parameters for image preprocessing.
  ImagePreprocessParameter image_preprocess_parameter;

  // Audio modality related config.
  // Whether to enable audio modality.
  bool audio_enabled = false;
  // Configuration for audio preprocessing.
  AudioPreprocessorConfig audio_preprocessor_config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();

  // Configuration for prompt processing.
  MultimodalPromptProcessingConfig processing_config;
};

struct GenericDataProcessorConfig {
  std::string model_role = "assistant";
  // If true, force the content from the model to be a string instead of an
  // array. Some legacy templates only support string content.
  bool force_string_content = false;

  // When set, the generic processor operates in multimodal mode
  std::optional<MultimodalConfig> multimodal;
};

struct GenericDataProcessorArguments {
  std::optional<int> visual_token_budget;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GENERIC_DATA_PROCESSOR_CONFIG_H_
