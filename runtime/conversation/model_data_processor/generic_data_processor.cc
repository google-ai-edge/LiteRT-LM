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

#include "runtime/conversation/model_data_processor/generic_data_processor.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ModelDataProcessor>>
GenericDataProcessor::Create(GenericDataProcessorConfig config,
                             const PromptTemplateCapabilities& capabilities) {
  std::unique_ptr<ImagePreprocessor> image_preprocessor = nullptr;
  std::unique_ptr<AudioPreprocessor> audio_preprocessor = nullptr;

  if (config.multimodal.has_value()) {
    if (config.multimodal->image_enabled) {
      image_preprocessor = ImagePreprocessor::Create();
    }
    if (config.multimodal->audio_enabled) {
      ABSL_ASSIGN_OR_RETURN(audio_preprocessor,
                            AudioPreprocessorMiniAudio::Create(
                                config.multimodal->audio_preprocessor_config));
    }
  }

  return absl::WrapUnique(new GenericDataProcessor(
      config, capabilities, std::move(image_preprocessor),
      std::move(audio_preprocessor)));
}

absl::StatusOr<std::vector<InputData>>
GenericDataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const GenericDataProcessorArguments& args) const {
  if (!config_.multimodal.has_value()) {
    std::vector<InputData> input_data;
    input_data.emplace_back(InputText(rendered_template_prompt));
    return input_data;
  }

  return ProcessMultimodalPrompt(
      rendered_template_prompt, messages, image_preprocessor_.get(),
      audio_preprocessor_.get(), config_.multimodal->processing_config,
      config_.multimodal->image_preprocess_parameter, args.visual_token_budget);
}

absl::StatusOr<Message> GenericDataProcessor::ToMessageImpl(
    const Responses& responses,
    const GenericDataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  nlohmann::ordered_json content;
  if (GetConfig().force_string_content) {
    content = response_text;
  } else {
    content = nlohmann::ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return nlohmann::ordered_json::object(
      {{"role", GetConfig().model_role}, {"content", content}});
}

absl::StatusOr<nlohmann::ordered_json>
GenericDataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  if (message["content"].is_string() && capabilities_.requires_typed_content) {
    // If the content is a string and the template requires typed content,
    // convert the content to a typed content.
    return nlohmann::ordered_json::object(
        {{"role", message["role"]},
         {"content", nlohmann::ordered_json::array(
                         {{{"type", "text"}, {"text", message["content"]}}})}});
  } else if (message["content"].is_array() && message["content"].size() == 1 &&
             message["content"][0]["type"] == "text" &&
             !capabilities_.requires_typed_content) {
    // If the content is a typed content and the template does not require
    // typed content, always convert the content to a string.
    return nlohmann::ordered_json::object(
        {{"role", message["role"]},
         {"content", message["content"][0]["text"]}});
  } else {
    return message;
  }
}

absl::StatusOr<ModelDataProcessor::SingleTurnTemplateRenderResult>
GenericDataProcessor::RenderSingleTurnTemplate(
    std::vector<Message>& history, const Preface& preface,
    const Message& message, const PromptTemplate& prompt_template,
    bool current_is_appending_message, bool append_message,
    std::optional<nlohmann::ordered_json> extra_context) const {
  return RenderSingleTurnTemplateCommon(
      *this, history, preface, message, prompt_template,
      current_is_appending_message, append_message, extra_context,
      /*push_dummy_user_message_to_preface=*/false);
}

absl::Status GenericDataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<GenericDataProcessorConfig,
                                     GenericDataProcessorArguments>& other) {
  const auto& generic_other = static_cast<const GenericDataProcessor&>(other);
  if (generic_other.audio_preprocessor_ != nullptr) {
    if (audio_preprocessor_ == nullptr) {
      const auto& multi_config = *generic_other.config_.multimodal;
      ABSL_ASSIGN_OR_RETURN(audio_preprocessor_,
                            AudioPreprocessorMiniAudio::Create(
                                multi_config.audio_preprocessor_config));
    }
    *static_cast<AudioPreprocessorMiniAudio*>(audio_preprocessor_.get()) =
        *static_cast<AudioPreprocessorMiniAudio*>(
            generic_other.audio_preprocessor_.get());
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
