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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MULTIMODAL_PROCESSOR_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MULTIMODAL_PROCESSOR_HELPER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/engine/io_types.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/image_preprocessor.h"

namespace litert::lm {

using ImagePreprocessor = ::litert::support::ImagePreprocessor;
using AudioPreprocessor = ::litert::support::AudioPreprocessor;
using ImagePreprocessParameter = ::litert::support::ImagePreprocessParameter;

// Configuration defining how the prompt is parsed and formatted
struct MultimodalPromptProcessingConfig {
  // Regex for matching the delimiters between different modalities.
  std::string delimiter_regex;
  // Regex for matching the image token placeholders.
  std::string image_token_regex;
  // Regex for matching the audio token placeholders.
  std::string audio_token_regex;

  // Image formatting
  // The token that indicates the beginning of an image.
  std::string boi_token;
  // The token that indicates the end of an image.
  std::string eoi_token;
  // The prefix to add before the image token.
  std::string image_prefix;
  // The suffix to add after the image token.
  std::string image_suffix;
  // Whether to add the end of image token after the image content.
  bool add_image_end = false;

  // Audio formatting
  // The token that indicates the beginning of an audio.
  std::string boa_token;
  // The token that indicates the end of an audio.
  std::string eoa_token;
  // The prefix to add before the audio token.
  std::string audio_prefix;
  // The suffix to add after the audio token.
  std::string audio_suffix;
  // Whether to add the end of audio token after the audio content.
  bool add_audio_end = true;
};

// The core generic preprocessing loop
//
// Args:
//   rendered_template_prompt: The rendered template prompt with placeholders
//     for images and audio.
//   messages: The messages to be processed.
//   image_preprocessor: The image preprocessor to use for preprocessing images.
//   audio_preprocessor: The audio preprocessor to use for preprocessing audio.
//   config: The configuration for multimodal prompt processing.
//   image_params: The parameters for image preprocessing.
//   visual_token_budget: Optional budget for maximum visual tokens.
//
// Returns:
//   A vector of InputData objects representing the preprocessed prompt.
absl::StatusOr<std::vector<InputData>> ProcessMultimodalPrompt(
    absl::string_view rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    ImagePreprocessor* image_preprocessor,
    AudioPreprocessor* audio_preprocessor,
    const MultimodalPromptProcessingConfig& config,
    std::optional<ImagePreprocessParameter> image_params = std::nullopt,
    std::optional<int> visual_token_budget = std::nullopt);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MULTIMODAL_PROCESSOR_HELPER_H_
