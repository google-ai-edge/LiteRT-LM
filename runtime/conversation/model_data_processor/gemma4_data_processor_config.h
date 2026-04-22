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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA4_DATA_PROCESSOR_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA4_DATA_PROCESSOR_CONFIG_H_

#include <string>

namespace litert::lm {

// Config for Gemma4DataProcessor.
struct Gemma4DataProcessorConfig {
  // The constraint mode when constrained decoding is enabled.
  enum class ConstraintMode {
    kUnspecified,
    // Both function call and text output are allowed.
    kTextAndOr,
    // Only function call is allowed.
    kFunctionCallOnly,
  };
  // The string for beginning of image token.
  std::string boi_token = "<|image>";
  // The string for end of image token.
  std::string eoi_token = "<image|>";

  // The patch width that the image preprocessor should patchify the image to.
  int patch_width = 16;
  // The patch height that the image preprocessor should patchify the image to.
  int patch_height = 16;

  // The maximum number of patches that the image preprocessor should patchify
  // the image to. If the number of patches that the patchified input image
  // would produce is more than this, the input image will be downsampled to the
  // max size that can fit the max number of patches, with respect to the aspect
  // ratio, before patchifying.
  int max_num_patches = 2520;

  // The pooling kernel size that the image preprocessor should use for
  // patchifying the image.
  int pooling_kernel_size = 3;

  // The string for beginning of audio token.
  std::string boa_token = "<|audio>";
  // The string for end of audio token.
  std::string eoa_token = "<audio|>";

  // Signifies the beginning of a tool call.
  std::string code_fence_start = "<|tool_call>";
  // Signifies the end of tool call.
  std::string code_fence_end = "<tool_call|>";
  std::string open_quote = "<|\"|>";
  std::string close_quote = "<|\"|>";
  std::string function_response_start = "<|tool_response>";
  // The syntax type of the tool call.
  std::string syntax_type = "fc";
  // Whether to escape the fence strings for regex matching.
  bool escape_fence_strings = true;
  // An optional regex to match each line of the tool code block.
  std::string tool_code_regex = "";
  // Whether to use the chat template for applying FC format.
  bool use_template_for_fc_format = true;
  // The constraint mode when constrained decoding is enabled. Default is
  // kTextAndOr.
  ConstraintMode constraint_mode = ConstraintMode::kTextAndOr;
};

// Arguments for Gemma4DataProcessor.
struct Gemma4DataProcessorArguments {};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA4_DATA_PROCESSOR_CONFIG_H_
