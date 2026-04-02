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

#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#endif
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/components/preprocessor/audio_preprocessor_miniaudio.h"
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/components/prompt_template.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/components/tool_use/python_tool_format_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2
#include "sentencepiece_model.pb.h"  // from @sentencepiece

namespace litert::lm {
namespace {

using ::nlohmann::ordered_json;

bool IsImage(absl::string_view part) {
  return part == "<start_of_image>" || part == "<image_soft_token>";
}

bool IsAudio(absl::string_view part) {
  return part == "<start_of_audio>" || part == "<audio_soft_token>";
}

bool HasToolCalls(const ordered_json& message) {
  return message.contains("tool_calls") && message["tool_calls"].is_array();
}

bool IsToolMessage(const ordered_json& message) {
  return message.contains("role") && message["role"] == "tool";
}

// Formats a tool response in Python format.
//
// The fields of the tool response may be under the key "tool_response",
// "response", or at the top-level.
//
// Example:
//
// Input:
//
// ```json
// {
//   "tool_response": {
//     "key1": "bar",
//     "key2": true
//   }
// }
// ```
//
// Output:
//
// ```
// {"key1": "bar", "key2": True}
// ```

absl::StatusOr<std::string> FormatToolResponse(
    const ordered_json& tool_response) {
  absl::string_view tool_response_key;
  if (tool_response.contains("tool_response")) {
    tool_response_key = "tool_response";
  } else if (tool_response.contains("response")) {
    tool_response_key = "response";
  } else {
    return FormatValueAsPython(tool_response);
  }

  return FormatValueAsPython(tool_response[tool_response_key]);
}

}  // namespace

absl::StatusOr<std::unique_ptr<Gemma3DataProcessor>>
Gemma3DataProcessor::Create(Gemma3DataProcessorConfig config,
                            std::optional<Preface> preface,
                            const Tokenizer* tokenizer,
                            const std::vector<std::vector<int>>& stop_token_ids,
                            bool enable_constrained_decoding) {
#if defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
  if (enable_constrained_decoding) {
    return absl::FailedPreconditionError(
        "Constrained decoding was disabled at build time.");
  }
  ASSIGN_OR_RETURN(auto audio_preprocessor,
                   AudioPreprocessorMiniAudio::Create(
                       AudioPreprocessorConfig::CreateDefaultUsmConfig()));
  return absl::WrapUnique(new Gemma3DataProcessor(
      config, preface, std::make_unique<StbImagePreprocessor>(),
      std::move(audio_preprocessor)));
#else
  std::unique_ptr<LiteRtLmGemmaModelConstraintProvider,
                  decltype(&LiteRtLmGemmaModelConstraintProvider_Destroy)>
      constraint_provider(nullptr,
                          &LiteRtLmGemmaModelConstraintProvider_Destroy);
  if (enable_constrained_decoding) {
    std::vector<const int*> stop_token_ids_ptrs;
    std::vector<size_t> stop_token_lengths;
    stop_token_ids_ptrs.reserve(stop_token_ids.size());
    stop_token_lengths.reserve(stop_token_ids.size());
    for (const auto& stop_tokens : stop_token_ids) {
      stop_token_ids_ptrs.push_back(stop_tokens.data());
      stop_token_lengths.push_back(stop_tokens.size());
    }
    if (tokenizer->GetTokenizerType() != TokenizerType::kSentencePiece) {
      return absl::InvalidArgumentError(
          "Constrained decoding is only supported for SentencePiece "
          "tokenizer.");
    }
    auto sp_tokenizer =
        reinterpret_cast<const SentencePieceTokenizer*>(tokenizer);
    auto serialized_model_proto =
        sp_tokenizer->GetProcessor().model_proto().SerializeAsString();
    LiteRtLmGemmaModelConstraintProvider* provider =
        LiteRtLmGemmaModelConstraintProvider_Create(
            serialized_model_proto.data(), serialized_model_proto.size(),
            stop_token_ids_ptrs.data(), stop_token_lengths.data(),
            stop_token_ids.size());
    if (provider == nullptr) {
      return absl::InternalError(
          "Failed to create GemmaModelConstraintProvider.");
    }
    constraint_provider.reset(provider);
  }
  ASSIGN_OR_RETURN(auto audio_preprocessor,
                   AudioPreprocessorMiniAudio::Create(
                       AudioPreprocessorConfig::CreateDefaultUsmConfig()));
  return absl::WrapUnique(new Gemma3DataProcessor(
      std::move(constraint_provider), config, preface,
      std::make_unique<StbImagePreprocessor>(), std::move(audio_preprocessor)));
#endif
}

absl::StatusOr<ordered_json> Gemma3DataProcessor::MessageToTemplateInput(
    const ordered_json& message) const {
  // If the message doesn't contain any tool calls and isn't a tool message,
  // then the template input is the same as the message.
  if (!HasToolCalls(message) && !IsToolMessage(message)) {
    return message;
  }

  ordered_json template_input = ordered_json::object();
  if (message.contains("role")) {
    template_input["role"] = message["role"];
  }

  // Process content.
  if (message.contains("content")) {
    // If the role is "tool", convert the tool responses to Python format.
    if (IsToolMessage(message)) {
      if (message["content"].is_array()) {
        // If the content is an array, treat each item as a tool response.
        template_input["content"] = ordered_json::array();
        for (const auto& item : message["content"]) {
          ASSIGN_OR_RETURN(std::string formatted_tool_response,
                           FormatToolResponse(item));
          template_input["content"].push_back(
              {{"type", "text"}, {"text", formatted_tool_response}});
        }
      } else if (message["content"].is_object()) {
        // If the content is an object, treat it as a single tool response.
        ASSIGN_OR_RETURN(std::string formatted_tool_response,
                         FormatToolResponse(message["content"]));
        template_input["content"] = formatted_tool_response;
      } else {
        // If the content is neither an array nor an object, pass it through
        // unchanged.
        template_input["content"] = message["content"];
      }
    } else {
      // If the role is not "tool", then pass through content unchanged.
      template_input["content"] = message["content"];
    }
  }

  // If the message contains tool calls, then convert them to Python and
  // add them to the template input.
  if (message.contains("tool_calls")) {
    template_input["tool_calls"] = ordered_json::array();
    for (const auto& tool_call : message["tool_calls"]) {
      if (!tool_call.contains("function")) {
        continue;
      }
      const nlohmann::ordered_json& function = tool_call["function"];
      ordered_json tool_call_input = ordered_json::object();
      tool_call_input["type"] = "function";
      tool_call_input["function"]["name"] = function["name"];

      if (function.contains("arguments")) {
        if (function["arguments"].is_object()) {
          for (const auto& [key, value] : function["arguments"].items()) {
            ASSIGN_OR_RETURN(std::string formatted_value,
                             FormatValueAsPython(value));
            tool_call_input["function"]["arguments"][key] = formatted_value;
          }
        } else {
          tool_call_input["function"]["arguments"] = function["arguments"];
        }
      }

      template_input["tool_calls"].push_back(tool_call_input);
    }
  }

  return template_input;
}

absl::StatusOr<std::vector<InputData>>
Gemma3DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt, const ordered_json& messages,
    const Gemma3DataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  std::deque<std::unique_ptr<MemoryMappedFile>> image_files;
  std::deque<std::unique_ptr<MemoryMappedFile>> audio_files;
  // Find all images and audio contained in the messages.
  for (const auto& message : messages) {
    if (message.contains("content") && message["content"].is_array()) {
      for (const auto& item : message["content"]) {
        if (item.is_string()) {
          continue;
        }
        ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                         LoadItemData(item));
        if (item["type"] == "image") {
          image_files.push_back(std::move(mmap_file));
        } else if (item["type"] == "audio") {
          audio_files.push_back(std::move(mmap_file));
        }
      }
    }
  }

  RE2 re_delimiter(
      "(<start_of_image>|<image_soft_token>|<start_of_audio>|<audio_soft_token>"
      ")");
  absl::string_view prompt_view(rendered_template_prompt);
  const char* start = prompt_view.data();
  std::string part;
  ImagePreprocessParameter image_params;
  image_params.SetTargetDimensions(Dimensions(
      {1, config_.image_tensor_height, config_.image_tensor_width, 3}));
  // Replace the placeholders with the actual data. Note for Gemma3N the
  // placeholders in the prompt are <image_soft_token> and <audio_soft_token>,
  // while for Gemma3 the placeholders in the prompt are <start_of_image> and
  // <start_of_audio>.
  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_part(start, prompt_view.data() - part.size());
    start = prompt_view.data();
    if (IsImage(part)) {
      input_data.emplace_back(
          InputText(absl::StrCat(text_part, "\n\n", config_.boi_token)));
      if (image_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less images than expected in the prompt.");
      }
      auto image_file = std::move(image_files.front());
      image_files.pop_front();
      ASSIGN_OR_RETURN(auto preprocessed_image,
                       image_preprocessor_->Preprocess(
                           InputImage(std::string(
                               static_cast<const char*>(image_file->data()),
                               image_file->length())),
                           image_params));
      input_data.emplace_back(InputImage(std::move(preprocessed_image)));
      input_data.emplace_back(InputText("\n\n"));
    } else if (IsAudio(part)) {
      input_data.emplace_back(
          InputText(absl::StrCat(text_part, "\n\n", config_.boa_token)));
      if (audio_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less audio than expected in the prompt.");
      }
      auto audio_file = std::move(audio_files.front());
      audio_files.pop_front();
      ASSIGN_OR_RETURN(auto preprocessed_audio,
                       audio_preprocessor_->Preprocess(InputAudio(std::string(
                           static_cast<const char*>(audio_file->data()),
                           audio_file->length()))));
      audio_preprocessor_->Reset();
      input_data.emplace_back(InputAudio(std::move(preprocessed_audio)));
      input_data.emplace_back(InputAudioEnd());
      input_data.emplace_back(InputText("\n\n"));
    }
  }
  if (!image_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more images than expected in the prompt.");
  }
  if (!audio_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more audio than expected in the prompt.");
  }
  // Add the remaining text in the prompt.
  if (!prompt_view.empty()) {
    input_data.push_back(InputText(std::string(prompt_view)));
  }
  return input_data;
}

absl::StatusOr<ModelDataProcessor::SingleTurnTemplateRenderResult>
Gemma3DataProcessor::RenderSingleTurnTemplate(
    std::vector<Message>& history, const Preface& preface,
    const Message& message, const PromptTemplate& prompt_template,
    bool current_is_appending_message, bool append_message,
    std::optional<nlohmann::ordered_json> extra_context) const {
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
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       MessageToTemplateInput(closing_message));
      closing_tmpl_input.extra_context["message"] = message_tmpl_input;
      closing_tmpl_input.extra_context["is_appending_to_prefill"] = true;
      closing_tmpl_input.extra_context["is_first_part"] = false;
      closing_tmpl_input.extra_context["is_last_part"] = true;
      closing_tmpl_input.add_generation_prompt = false;
      ASSIGN_OR_RETURN(std::string closing_text,
                       prompt_template.Apply(closing_tmpl_input));
      prefill_text += closing_text;
    }
  } else {
    PromptTemplateInput preface_tmpl_input;
    RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(json_preface, this,
                                                      preface_tmpl_input));
    if (!json_preface.messages.empty() || !json_preface.tools.empty() ||
        !json_preface.extra_context.is_null()) {
      preface_tmpl_input.messages.push_back(
          Message{{"role", "user"}, {"content", ""}});
      preface_tmpl_input.add_generation_prompt = false;

      if (extra_context.has_value()) {
        for (const auto& [key, value] : extra_context.value().items()) {
          preface_tmpl_input.extra_context[key] = value;
        }
      }

      ASSIGN_OR_RETURN(std::string preface_text,
                       prompt_template.Apply(preface_tmpl_input));
      prefill_text += preface_text;
    }
  }
  if (message.is_object()) {
    PromptTemplateInput tmpl_input;
    ASSIGN_OR_RETURN(tmpl_input.extra_context["message"],
                     MessageToTemplateInput(message));
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

    ASSIGN_OR_RETURN(std::string new_text, prompt_template.Apply(tmpl_input));
    prefill_text += new_text;
  }
  return SingleTurnTemplateRenderResult{prefill_text, new_is_appending_message};
}

absl::StatusOr<Message> Gemma3DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Gemma3DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  Message message = {{"role", "assistant"}};
  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ASSIGN_OR_RETURN(
        ordered_json content_and_tool_calls,
        ParseTextAndToolCalls(
            response_text, config_.code_fence_start, config_.code_fence_end,
            GetSyntaxType(config_.syntax_type), config_.escape_fence_strings,
            config_.tool_code_regex));
    if (content_and_tool_calls.contains("content")) {
      message["content"] = content_and_tool_calls["content"];
    }
    if (content_and_tool_calls.contains("tool_calls")) {
      message["tool_calls"] = content_and_tool_calls["tool_calls"];
    }
  } else {
    message["content"] = ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return message;
}

absl::StatusOr<ordered_json> Gemma3DataProcessor::FormatTools(
    const ordered_json& tools) const {
  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  ordered_json formatted_tools = ordered_json::array();
  for (const auto& tool : tools) {
    ASSIGN_OR_RETURN(std::string formatted_tool, FormatToolAsPython(tool));
    formatted_tools.push_back(formatted_tool);
  }
  return formatted_tools;
}

absl::StatusOr<std::unique_ptr<Constraint>>
Gemma3DataProcessor::CreateConstraint(
    const nlohmann::ordered_json& tools) const {
#if defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
  return absl::FailedPreconditionError(
      "Constrained decoding is disabled at build time, but it was requested "
      "for inference.");
#else
  if (constraint_provider_c_ == nullptr) {
    return nullptr;
  }
  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  nlohmann::ordered_json functions = nlohmann::ordered_json::array();
  for (const auto& tool : tools) {
    if (tool.contains("function")) {
      functions.push_back(tool["function"]);
    } else {
      functions.push_back(tool);
    }
  }
  LiteRtLmGemmaModelConstraintOptions gemma_options = {
      .funcall_format = kLiteRtLmGemmaFuncallFormatPythonStyle,
      .code_fence_start = config_.code_fence_start.c_str(),
      .code_fence_end = config_.code_fence_end.c_str(),
      .open_quote = nullptr,
      .close_quote = nullptr,
      .function_response_start = nullptr};
  std::string functions_str = functions.dump();
  LiteRtLmConstraint* constraint =
      LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools(
          constraint_provider_c_.get(), functions_str.c_str(), &gemma_options);
  if (constraint == nullptr) {
    return absl::InternalError("Failed to create constraint with tools.");
  }
  return absl::WrapUnique(reinterpret_cast<Constraint*>(constraint));
#endif
}

absl::string_view Gemma3DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Gemma3DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

absl::Status Gemma3DataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<Gemma3DataProcessorConfig,
                                     Gemma3DataProcessorArguments>& other) {
  const Gemma3DataProcessor& other_gemma3_data_processor =
      static_cast<const Gemma3DataProcessor&>(other);
  if (other_gemma3_data_processor.audio_preprocessor_ != nullptr) {
    if (audio_preprocessor_ == nullptr) {
      ASSIGN_OR_RETURN(audio_preprocessor_,
                       AudioPreprocessorMiniAudio::Create(
                           AudioPreprocessorConfig::CreateDefaultUsmConfig()));
    }
    *static_cast<AudioPreprocessorMiniAudio*>(audio_preprocessor_.get()) =
        *static_cast<AudioPreprocessorMiniAudio*>(
            other_gemma3_data_processor.audio_preprocessor_.get());
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
