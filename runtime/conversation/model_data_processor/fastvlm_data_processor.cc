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

#include "runtime/conversation/model_data_processor/fastvlm_data_processor.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

namespace {

using ::nlohmann::ordered_json;

}  // namespace

absl::StatusOr<std::unique_ptr<FastVlmDataProcessor>>
FastVlmDataProcessor::Create(FastVlmDataProcessorConfig config,
                             const PromptTemplateCapabilities& capabilities) {
  return absl::WrapUnique(new FastVlmDataProcessor(
      config, capabilities, std::make_unique<StbImagePreprocessor>()));
}

absl::StatusOr<ordered_json> FastVlmDataProcessor::FormatTools(
    const ordered_json& tools) const {
  return tools;
}

absl::StatusOr<std::vector<InputData>>
FastVlmDataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt, const ordered_json& messages,
    const FastVlmDataProcessorArguments& args) const {
  MultimodalPromptProcessingConfig multi_config{
      .delimiter_regex = "(<image_soft_token>)",
      .image_token_regex = "(<image_soft_token>)",
      .audio_token_regex = "",
      .boi_token = "",
      .eoi_token = "",
      .image_prefix = "",
      .image_suffix = "",
      .add_image_end = false,
      .boa_token = "",
      .eoa_token = "",
      .audio_prefix = "",
      .audio_suffix = "",
      .add_audio_end = false,
  };
  ImagePreprocessParameter image_preprocess_parameter;
  image_preprocess_parameter.SetTargetDimensions(Dimensions(
      {1, config_.image_tensor_height, config_.image_tensor_width, 3}));
  return ProcessMultimodalPrompt(
      rendered_template_prompt, messages, image_preprocessor_.get(),
      /*audio_preprocessor=*/nullptr, multi_config, image_preprocess_parameter);
}

absl::StatusOr<Message> FastVlmDataProcessor::ToMessageImpl(
    const Responses& responses,
    const FastVlmDataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  ordered_json content = ordered_json::array(
      {{{"type", "text"}, {"text", std::string(response_text)}}});
  return ordered_json::object({{"role", "assistant"}, {"content", content}});
}

absl::Status FastVlmDataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<FastVlmDataProcessorConfig,
                                     FastVlmDataProcessorArguments>& other) {
  return absl::OkStatus();
}

}  // namespace litert::lm
