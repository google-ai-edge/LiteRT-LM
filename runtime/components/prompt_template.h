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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PROMPT_TEMPLATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PROMPT_TEMPLATE_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/rust/minijinja_template.rs.h"

namespace litert::lm {

// The input to the prompt template.
// Note the json objects structure should follow the jinja template expect. See
// more details how the messages and tools are used in the template at
// https://huggingface.co/docs/transformers/v4.48.2/en/chat_templating
// https://huggingface.co/blog/unified-tool-use
//
// Example usage:
//
// const json user_message{
//     {"role", "user"},
//     {"content", "I need help"},
// };
// const json assistant_message{
//     {"role", "assistant"},
//     {"content", "Hi, what can I do for you?"},
// };
// const json tools = json::parse(R"({
//   "type": "function",
//   "function": {
//     "name": "GetWeather",
//     "description": "Get the weather of the location.",
//     "parameters": {
//       "type": "object",
//       "properties": {
//         "location": {
//           "type": "string",
//           "description": "The location to get the weather for."
//         }
//       },
//       "required": ["location"]
//     }
//   }
// })");
//
// PromptTemplateInput input;
// input.messages = json::array({user_message,
//                                message_assistant_text});
// input.tools = tools;
// input.add_generation_prompt = false;
// input.extra_context = json::object({{"extra_key", "extra_value"}});
// input.now = absl::Now();
//
// const std::string template_content = ReadTemplateFromFile(template_file);
// PromptTemplate template(template_content);
// ASSIGN_OR_RETURN(std::string rendered_prompt, template.Apply(input));
struct PromptTemplateInput {
  // The messages in the conversation.
  nlohmann::ordered_json messages;

  // The tools available to the model.
  nlohmann::ordered_json tools;

  // Whether to add the generation prompt to the end of the template, to hint
  // the model to generate the response. Default to true.
  bool add_generation_prompt = true;

  // The extra context to the model. This leaves the template to be able to
  // use extra variables, e.g. enabling thinking mode, or extra settings.
  nlohmann::ordered_json extra_context;

  // The time when the prompt template is applied. This is used to support
  // things time-aware knowledge cut models, e.g. SmolLM-V3,
  absl::Time now = absl::Now();

  // The beginning of sequence token.
  nlohmann::ordered_json bos_token = "";

  // The end of sequence token.
  nlohmann::ordered_json eos_token = "";
};

// The capabilities of the prompt template.
struct PromptTemplateCapabilities {
  // Whether the template supports tools.
  bool supports_tools = false;
  // Whether the template supports tool calls.
  bool supports_tool_calls = false;
  // Whether the template supports system role.
  bool supports_system_role = false;
  // Whether the template supports parallel tool calls.
  bool supports_parallel_tool_calls = false;
  // Whether the template supports tool call id.
  bool supports_tool_call_id = false;

  // Whether the template requires typed content. {"type": "text"}, {"type":
  // "image"}, {"type": "audio"}, {"type": "video"} etc.
  bool requires_typed_content = false;

  // Whether the template supports single turn, i.e. appending to the prefill
  // without repeating the whole history.
  bool supports_single_turn = false;
};

// PromptTemplate to render the jinja prompt template.
class PromptTemplate {
 public:
  // Creates a PromptTemplate from the string content.
  // template_content: the jinja template string.
  // edit_template_for_minijinja: whether to edit the template to be compatible
  // with Mini Jinja.
  explicit PromptTemplate(absl::string_view template_content,
                          bool edit_template_for_minijinja = true);

  // Copying constructor.
  PromptTemplate(const PromptTemplate&);

  // Copying assignment operator.
  PromptTemplate& operator=(const PromptTemplate&);

  // Moving constructor.
  PromptTemplate(PromptTemplate&&);

  // Moving assignment operator.
  PromptTemplate& operator=(PromptTemplate&&);

  // Applies the prompt template to the input.
  absl::StatusOr<std::string> Apply(const PromptTemplateInput& input) const;

  // Returns the template source string.
  absl::string_view GetTemplateSource() const;

  // Returns the capabilities that the prompt template supports.
  const PromptTemplateCapabilities& GetCapabilities() const {
    return capabilities_;
  }

 private:
  rust::Box<MinijinjaTemplate> minijinja_template_;

  // The capabilities of the prompt template. Auto inferred from the template
  // source string.
  PromptTemplateCapabilities capabilities_;
};
}  // namespace litert::lm
#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PROMPT_TEMPLATE_H_
