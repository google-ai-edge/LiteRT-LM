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

#include "runtime/util/model_type_utils.h"

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr std::array<int, 1> kStartTurnTokenIdsToCheck = {
    105,  // Gemma family.
};

bool IsGemma3nModel(const std::string& start_turn_text,
                    const std::vector<int>& audio_token_ids) {
  return audio_token_ids.size() == 1 && audio_token_ids[0] == 256000 &&
         start_turn_text == "<start_of_turn>";
}

bool IsGemma3Model(const std::string& start_turn_text,
                   const std::vector<int>& audio_token_ids) {
  return (audio_token_ids.size() != 1 || (audio_token_ids[0] != 256000)) &&
         start_turn_text == "<start_of_turn>";
}

void PopulateDefaultGemma3N(proto::Gemma3N& gemma3n) {
  gemma3n.mutable_start_of_image_token()->set_token_str("<start_of_image>");
  gemma3n.mutable_end_of_image_token()->set_token_str("<end_of_image>");
  gemma3n.set_image_tensor_height(768);
  gemma3n.set_image_tensor_width(768);
  gemma3n.mutable_start_of_audio_token()->set_token_str("<start_of_audio>");
  gemma3n.mutable_end_of_audio_token()->set_token_str("<end_of_audio>");
}

absl::StatusOr<proto::LlmModelType> CreateModelType(
    const std::string& start_turn_text, Tokenizer* tokenizer) {
  if (tokenizer == nullptr) {
    proto::LlmModelType model_type;
    model_type.mutable_generic_model();
    return model_type;
  }
  proto::LlmModelType model_type;
  ASSIGN_OR_RETURN(auto audio_token_ids,
                   tokenizer->TextToTokenIds("<start_of_audio>"));
  if (IsGemma3nModel(start_turn_text, audio_token_ids)) {
    PopulateDefaultGemma3N(*model_type.mutable_gemma3n());
    return model_type;
  } else if (IsGemma3Model(start_turn_text, audio_token_ids)) {
    model_type.mutable_gemma3();
    return model_type;
  } else {
    model_type.mutable_generic_model();
  }
  return model_type;
}

}  // namespace

absl::StatusOr<proto::LlmModelType> InferLlmModelType(
    const proto::LlmMetadata& metadata, Tokenizer* tokenizer) {
  if (metadata.has_llm_model_type()) {
    return metadata.llm_model_type();
  }

  if (tokenizer == nullptr) {
    proto::LlmModelType model_type;
    model_type.mutable_generic_model();
    return model_type;
  }

  proto::LlmModelType model_type;
  model_type.mutable_generic_model();

  for (int token_id : kStartTurnTokenIdsToCheck) {
    auto start_turn_text = tokenizer->TokenIdsToText({token_id});
    if (!start_turn_text.ok()) {
      if (start_turn_text.status().code() == absl::StatusCode::kDataLoss) {
        // If the error is DataLoss, it means the start turn token id coincides
        // with the middle of an incomplete BPE sequence by chance used by
        // HungingFace tokenizer. We should keep searching for the next start
        // turn token id.
        continue;
      } else if (start_turn_text.status().code() ==
                 absl::StatusCode::kNotFound) {
        // If the error is NotFound, it means the start turn token id is out of
        // range, indicating the model is a fake one that runs in unittest.
        // Return default model type.
        return model_type;
      } else {
        return start_turn_text.status();
      }
    }
    ASSIGN_OR_RETURN(model_type, CreateModelType(*start_turn_text, tokenizer));
    // If the model type is not generic, we can stop checking.
    if (model_type.model_type_case() != proto::LlmModelType::kGenericModel) {
      break;
    }
  }
  return model_type;
}

absl::StatusOr<std::string> GetDefaultJinjaPromptTemplate(
    const proto::PromptTemplates& prompt_templates,
    const proto::LlmModelType& llm_model_type) {
  switch (llm_model_type.model_type_case()) {
    case proto::LlmModelType::kFunctionGemma:
      return R"tmpl({{ bos_token }}
{%- set ns = namespace(prev_message_type=None) -%}
{#- Tool Declarations -#}
{%- set loop_messages = messages -%}
{%- if tools or messages[0]['role'] == 'system' -%}
    {{- '<start_of_turn>developer\n' -}}
    {%- if messages[0]['role'] == 'system' -%}
        {%- if messages[0]['content'] is string -%}
            {{- messages[0]['content'] | trim -}}
        {%- else -%}
            {%- for item in messages[0]['content'] -%}
                {%- if item['type'] == 'text' -%}
                    {{- item['text'] | trim -}}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
        {%- set loop_messages = messages[1:] -%}
        {%- if tools -%}
            {{- '\n\n' -}}
        {%- endif -%}
    {%- endif -%}
    {%- for tool in tools %}
        {{- '<start_function_declaration>' -}}
        {{- tool | trim }}
        {{- '<end_function_declaration>' -}}
    {%- endfor %}
    {{- '<end_of_turn>\n'}}
{%- endif %}
{#- Loop through messages. -#}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'assistant') -%}
        {#- Rename "assistant" to "model". -#}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {%- if role != 'tool' -%}
        {%- if ns.prev_message_type == 'tool_call' -%}
            {{ raise_exception("Tool call must be followed by a tool response.") }}
        {%- endif -%}
        {%- if ns.prev_message_type != 'tool_response' -%}
            {{- '<start_of_turn>' + role + '\n'}}
        {%- endif -%}
        {%- set ns.prev_message_type = None -%}
        {%- if 'content' in message -%}
            {%- if message['content'] is string -%}
                {{ message['content'] | trim }}
            {%- elif message['content'] is iterable -%}
                {%- for item in message['content'] -%}
                    {%- if item['type'] == 'text' -%}
                        {{ item['text'] | trim }}
                    {%- endif -%}
                {%- endfor -%}
            {%- else -%}
                {{ raise_exception("Invalid content type") }}
            {%- endif -%}
            {%- set ns.prev_message_type = 'content' -%}
        {%- endif -%}
        {%- if 'tool_calls' in message and message['tool_calls'] and message['tool_calls'] is iterable -%}
            {#- Tool Calls -#}
            {%- for tool_call in message['tool_calls'] -%}
                {%- if 'function' in tool_call -%}
                    {%- set tool_call = tool_call['function'] -%}
                {%- endif -%}
                {{-  '<start_function_call>call:' + tool_call['name'] + '{' -}}
                {%- if 'arguments' in tool_call -%}
                    {%- for key in tool_call['arguments'] -%}
                        {{- key + ':' + tool_call['arguments'][key] -}}
                        {% if not loop.last %}
                            {{- ',' -}}
                        {% endif %}
                    {%- endfor %}
                {%- endif -%}
                {{- '}' + '<end_function_call>' -}}
            {%- endfor -%}
            {%- if loop.last -%}
                {{ '<start_function_response>' }}
            {%- endif -%}
            {%- set ns.prev_message_type = 'tool_call' -%}
        {%- endif -%}
    {%- else -%}
        {#- Tool Responses -#}
        {%- if 'content' in message -%}
            {%- if message['content'] is string -%}
                {{- '<start_function_response>response:' -}}
                {{ message['content'] | trim }}
                {{- '<end_function_response>' -}}
            {%- elif message['content'] is iterable -%}
                {%- for item in message['content'] -%}
                    {%- if item['type'] == 'text' -%}
                        {{ '<start_function_response>response:' + item['text'] + '<end_function_response>' }}
                    {%- else -%}
                        {{ raise_exception("Invalid content type for tool response.") }}
                    {%- endif -%}
                {%- endfor -%}
            {%- else -%}
                {{ raise_exception("Invalid content type") }}
            {%- endif -%}
        {%- endif -%}
        {%- set ns.prev_message_type = 'tool_response' -%}
    {%- endif -%}
    {%- if ns.prev_message_type not in ['tool_call', 'tool_response'] -%}
        {{ '<end_of_turn>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type == 'tool_call' -%}
        {{- '<start_function_response>' -}}
    {%- elif ns.prev_message_type != 'tool_response' -%}
        {{- '<start_of_turn>model\n' -}}
    {%- endif -%}
{%- endif -%})tmpl";
    case proto::LlmModelType::kGemma3:
      return R"tmpl({{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%})tmpl";
    case proto::LlmModelType::kGemma3N:
      return R"tmpl({{ bos_token }}
{%- if tools %}
    {{- '<start_of_turn>system\n' }}
    {%- for tool in tools %}
        {{- tool | trim }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- '<end_of_turn>\n'}}
{%- endif %}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- elif (message['role'] == 'tool') -%}
        {%- set is_tool = True -%}
        {%- set role = "user" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if is_tool -%}
        {{ '```tool_outputs\n' }}
    {%- endif -%}
    {%- if 'content' in message -%}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'audio' -%}
                    {{ '<audio_soft_token>' }}
                {%- elif item['type'] == 'image' -%}
                    {{ '<image_soft_token>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
                {%- if is_tool -%}
                    {{ '\n' }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
    {%- endif -%}
    {%- if is_tool -%}
        {{ '```' }}
        {%- set is_tool = False -%}
    {%- endif -%}
    {%- if 'tool_calls' in message -%}
        {{- '```tool_code\n' -}}
        {%- for tool_call in message['tool_calls'] -%}
            {%- if 'function' in tool_call -%}
                {%- set tool_call = tool_call['function'] -%}
            {%- endif -%}
            {{-  tool_call['name'] + '(' -}}
            {%- if 'arguments' in tool_call -%}
                {%- for key in tool_call['arguments'] -%}
                    {{- key + '=' + tool_call['arguments'][key] -}}
                    {% if not loop.last %}
                        {{- ', ' -}}
                    {% endif %}
                {%- endfor %}
            {{- ')\n' -}}
            {%- endif -%}
        {%- endfor -%}
        {{- '```' -}}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%})tmpl";
    case proto::LlmModelType::kQwen3:
    case proto::LlmModelType::kQwen2P5:
    case proto::LlmModelType::kGenericModel:
    case proto::LlmModelType::kGemma4:
      // absl::Substitute takes up to 10 arguments, so we have to split the
      // template into two parts.
      return absl::StrCat(
          absl::Substitute("{%- for message in messages -%}"
                           "{%- if message.content is string -%}"
                           "{%- if message.role == 'user' %}"
                           "$0{{ message.content }}$1"
                           "{% endif -%}"
                           "{%- if message.role == 'model' %}"
                           "$2{{ message.content }}$3"
                           "{% endif -%}"
                           "{%- if message.role == 'system' %}"
                           "$4{{ message.content }}$5"
                           "{% endif -%}"
                           "{%- else -%}",
                           prompt_templates.user().prefix(),
                           prompt_templates.user().suffix(),
                           prompt_templates.model().prefix(),
                           prompt_templates.model().suffix(),
                           prompt_templates.system().prefix(),
                           prompt_templates.system().suffix()),
          absl::Substitute("{%- if message.role == 'user' %}"
                           "$0"
                           "{% elif message.role == 'model' %}"
                           "$1"
                           "{% elif message.role == 'system' %}"
                           "$2"
                           "{% endif -%}"
                           "{%- for item in message.content %}"
                           "{%- if item.type == 'text' %}"
                           "{{ item.text }}"
                           "{% elif item.type == 'image' -%}"
                           "{{ '<start_of_image>' }}"
                           "{%- elif item.type == 'audio' -%}"
                           "{{ '<start_of_audio>' }}"
                           "{%- endif -%}"
                           "{%- endfor -%}"
                           "{%- if message.role == 'user' %}"
                           "$3"
                           "{% elif message.role == 'model' %}"
                           "$4"
                           "{% elif message.role == 'system' %}"
                           "$5"
                           "{% endif -%}"
                           "{%- endif -%}"
                           "{%- endfor -%}"
                           "{%- if add_generation_prompt %}"
                           "$6"
                           "{% endif -%}",
                           prompt_templates.user().prefix(),
                           prompt_templates.model().prefix(),
                           prompt_templates.system().prefix(),
                           prompt_templates.user().suffix(),
                           prompt_templates.model().suffix(),
                           prompt_templates.system().suffix(),
                           prompt_templates.model().prefix()));
    case proto::LlmModelType::MODEL_TYPE_NOT_SET:
      return absl::InvalidArgumentError("LlmModelType is not set.");
  }
}

}  // namespace litert::lm
