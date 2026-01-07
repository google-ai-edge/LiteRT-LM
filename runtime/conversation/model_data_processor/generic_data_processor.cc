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
#include <string>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ModelDataProcessor>>
GenericDataProcessor::Create(GenericDataProcessorConfig config,
                             const PromptTemplateCapabilities& capabilities) {
  return absl::WrapUnique(new GenericDataProcessor(config, capabilities));
}

absl::StatusOr<std::vector<InputData>>
GenericDataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const GenericDataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  input_data.emplace_back(InputText(rendered_template_prompt));
  return input_data;
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

}  // namespace litert::lm
