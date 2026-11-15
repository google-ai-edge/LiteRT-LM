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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_ARGUMENTS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_ARGUMENTS_H_

#include <string>

namespace litert::lm {

// Config for Gemma3DataProcessor.
struct Gemma3DataProcessorConfig {
  // The string for beginning of image token.
  std::string boi_token = "<start_of_image>";
  // The string for end of image token.
  std::string eoi_token = "<end_of_image>";

  int image_tensor_height = 768;
  int image_tensor_width = 768;

  // The string for beginning of audio token.
  std::string boa_token = "<start_of_audio>";
  // The string for end of audio token.
  std::string eoa_token = "<end_of_audio>";

  // Tool call parsing configuration.
  std::string code_fence_start = "```tool_code\n";
  std::string code_fence_end = "\n```";
  std::string syntax_type = "python";
  bool escape_fence_strings = true;
  std::string tool_code_regex =
      R"regex(print\((?:default_api\.)?(.+\(.*\))\))regex";
};

// Arguments for Gemma3DataProcessor.
struct Gemma3DataProcessorArguments {
  // Whether to only dry run the media preprocessing, to count the tokens the
  // input consumes. See ModelDataProcessor::WithDryRun().
  bool dry_run = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_ARGUMENTS_H_
