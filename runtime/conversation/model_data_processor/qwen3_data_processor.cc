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

#include "runtime/conversation/model_data_processor/qwen3_data_processor.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ModelDataProcessor>> Qwen3DataProcessor::Create(
    Qwen3DataProcessorConfig config, std::optional<Preface> preface) {
  return absl::WrapUnique(
      new Qwen3DataProcessor(std::move(config), std::move(preface)));
}

absl::StatusOr<nlohmann::ordered_json>
Qwen3DataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  if (message["content"].is_array()) {
    const auto& content = message["content"];
    if (content.size() == 1 && content[0].contains("text")) {
      auto result = nlohmann::ordered_json::object(
          {{"role", message["role"]}, {"content", content[0]["text"]}});
      return result;
    }
  }
  return message;
}

absl::StatusOr<std::vector<InputData>>
Qwen3DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const Qwen3DataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  input_data.emplace_back(InputText(rendered_template_prompt));
  return input_data;
}

absl::StatusOr<Message> Qwen3DataProcessor::ToMessageImpl(
    const Responses& responses, const Qwen3DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  Message message = {{"role", "assistant"}};
  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ASSIGN_OR_RETURN(
        nlohmann::ordered_json content_and_tool_calls,
        ParseTextAndToolCalls(response_text, config_.code_fence_start,
                              config_.code_fence_end, SyntaxType::kJson,
                              config_.escape_fence_strings,
                              config_.tool_code_regex));
    if (content_and_tool_calls.contains("content")) {
      message["content"] = content_and_tool_calls["content"];
    }
    if (content_and_tool_calls.contains("tool_calls")) {
      message["tool_calls"] = content_and_tool_calls["tool_calls"];
    }
  } else {
    message["content"] = nlohmann::ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return message;
}

absl::string_view Qwen3DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Qwen3DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
