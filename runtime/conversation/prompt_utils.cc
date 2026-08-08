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

#include "runtime/conversation/prompt_utils.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

void StripBlobs(nlohmann::ordered_json& json) {
  if (json.is_array()) {
    for (auto& item : json) {
      StripBlobs(item);
    }
  } else if (json.is_object()) {
    if (json.contains("type") && json["type"].is_string()) {
      std::string type = json["type"].get<std::string>();
      if (type == "image" || type == "audio") {
        json.erase("blob");
      }
    }
    for (auto& [key, value] : json.items()) {
      StripBlobs(value);
    }
  }
}

}  // namespace

absl::Status FillPrefaceForPromptTemplateInput(
    const Preface& preface, const ModelDataProcessor* model_data_processor,
    PromptTemplateInput& tmpl_input) {
  if (std::holds_alternative<JsonPreface>(preface)) {
    auto json_preface = std::get<JsonPreface>(preface);

    if (json_preface.messages.is_array()) {
      for (auto& message : json_preface.messages) {
        ABSL_ASSIGN_OR_RETURN(
            nlohmann::ordered_json message_tmpl_input,
            model_data_processor->MessageToTemplateInput(message));
        tmpl_input.messages.push_back(message_tmpl_input);
      }
    }

    if (json_preface.tools.is_null()) {
      tmpl_input.tools = nullptr;
    } else {
      ABSL_ASSIGN_OR_RETURN(tmpl_input.tools, model_data_processor->FormatTools(
                                                  json_preface.tools));
    }
    tmpl_input.extra_context = json_preface.extra_context;
  } else {
    return absl::UnimplementedError("Preface type is not supported yet");
  }
  return absl::OkStatus();
}

absl::StatusOr<ModelDataProcessor::SingleTurnTemplateRenderResult>
RenderSingleTurnTemplateCommon(
    const ModelDataProcessor& processor, std::vector<Message>& history,
    const Preface& preface, const Message& message,
    const PromptTemplate& prompt_template, bool current_is_appending_message,
    bool append_message, std::optional<nlohmann::ordered_json> extra_context,
    bool push_dummy_user_message_to_preface) {
  const auto& json_preface = std::get<JsonPreface>(preface);
  std::string prefill_text = "";
  bool is_first_part = false;
  bool is_last_part = false;

  if (!current_is_appending_message) {
    is_first_part = true;
  }
  if (!append_message) {
    is_last_part = true;
  }

  bool new_is_appending_message = current_is_appending_message;
  if (is_first_part) {
    new_is_appending_message = true;
  }
  if (is_last_part) {
    new_is_appending_message = false;
  }

  bool is_role_changed = false;
  if (!history.empty()) {
    const auto& last_message = history.back();
    // If the last message is in appending state and the current message is
    // different role, then we need to add a closing message to the prefill.
    if (current_is_appending_message &&
        (last_message["role"] != message["role"] &&
         last_message["role"] != "system")) {
      is_role_changed = true;
      PromptTemplateInput closing_tmpl_input;
      nlohmann::ordered_json closing_message = {
          {"role", last_message["role"]},
          {"content", ""},
      };
      ABSL_ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                            processor.MessageToTemplateInput(closing_message));
      closing_tmpl_input.extra_context["message"] = message_tmpl_input;
      closing_tmpl_input.extra_context["is_appending_to_prefill"] = true;
      closing_tmpl_input.extra_context["is_first_part"] = false;
      closing_tmpl_input.extra_context["is_last_part"] = true;
      closing_tmpl_input.add_generation_prompt = false;
      StripBlobsFromTemplateInput(closing_tmpl_input);
      ABSL_ASSIGN_OR_RETURN(std::string closing_text,
                            prompt_template.Apply(closing_tmpl_input));
      prefill_text += closing_text;
    }
  } else {
    PromptTemplateInput preface_tmpl_input;
    ABSL_RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
        json_preface, &processor, preface_tmpl_input));
    if (!json_preface.messages.empty() || !json_preface.tools.empty() ||
        !json_preface.extra_context.is_null()) {
      if (push_dummy_user_message_to_preface) {
        preface_tmpl_input.messages.push_back(
            Message{{"role", "user"}, {"content", ""}});
      }
      preface_tmpl_input.add_generation_prompt = false;

      if (extra_context.has_value()) {
        for (const auto& [key, value] : extra_context.value().items()) {
          preface_tmpl_input.extra_context[key] = value;
        }
      }

      StripBlobsFromTemplateInput(preface_tmpl_input);
      ABSL_ASSIGN_OR_RETURN(std::string preface_text,
                            prompt_template.Apply(preface_tmpl_input));
      prefill_text += preface_text;
    }
  }
  if (message.is_object()) {
    PromptTemplateInput tmpl_input;
    ABSL_ASSIGN_OR_RETURN(tmpl_input.extra_context["message"],
                          processor.MessageToTemplateInput(message));
    tmpl_input.extra_context["is_appending_to_prefill"] = true;
    tmpl_input.extra_context["is_first_part"] =
        is_first_part || is_role_changed;
    tmpl_input.extra_context["is_last_part"] = is_last_part;
    tmpl_input.add_generation_prompt = !new_is_appending_message;

    if (extra_context.has_value()) {
      for (const auto& [key, value] : extra_context.value().items()) {
        tmpl_input.extra_context[key] = value;
      }
    }

    StripBlobsFromTemplateInput(tmpl_input);
    ABSL_ASSIGN_OR_RETURN(std::string new_text,
                          prompt_template.Apply(tmpl_input));
    prefill_text += new_text;
  }
  return ModelDataProcessor::SingleTurnTemplateRenderResult{
      prefill_text, new_is_appending_message};
}

void StripBlobsFromTemplateInput(PromptTemplateInput& input) {
  StripBlobs(input.messages);
  if (input.extra_context.contains("message")) {
    StripBlobs(input.extra_context["message"]);
  }
}

}  // namespace litert::lm
