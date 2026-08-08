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
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constraint.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
#include "runtime/components/logits_processor/constrained_decoding/gemma_model_constraint_provider.h"
#endif
#include "runtime/components/prompt_template.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/components/tool_use/python_tool_format_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"
#include "sentencepiece_model.pb.h"  // from @sentencepiece

namespace litert::lm {
namespace {

using ::nlohmann::ordered_json;

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
  ABSL_ASSIGN_OR_RETURN(auto audio_preprocessor,
                        AudioPreprocessorMiniAudio::Create(
                            AudioPreprocessorConfig::CreateDefaultUsmConfig()));
  return absl::WrapUnique(
      new Gemma3DataProcessor(config, preface, ImagePreprocessor::Create(),
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
  ABSL_ASSIGN_OR_RETURN(auto audio_preprocessor,
                        AudioPreprocessorMiniAudio::Create(
                            AudioPreprocessorConfig::CreateDefaultUsmConfig()));
  return absl::WrapUnique(new Gemma3DataProcessor(
      std::move(constraint_provider), config, preface,
      ImagePreprocessor::Create(), std::move(audio_preprocessor)));
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
          ABSL_ASSIGN_OR_RETURN(std::string formatted_tool_response,
                                FormatToolResponse(item));
          template_input["content"].push_back(
              {{"type", "text"}, {"text", formatted_tool_response}});
        }
      } else if (message["content"].is_object()) {
        // If the content is an object, treat it as a single tool response.
        ABSL_ASSIGN_OR_RETURN(std::string formatted_tool_response,
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
            ABSL_ASSIGN_OR_RETURN(std::string formatted_value,
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
  MultimodalPromptProcessingConfig multi_config{
      .delimiter_regex =
          "(<start_of_image>|<image_soft_token>|<start_of_audio>|<audio_soft_"
          "token>)",
      .image_token_regex = "(<start_of_image>|<image_soft_token>)",
      .audio_token_regex = "(<start_of_audio>|<audio_soft_token>)",
      .boi_token = config_.boi_token,
      .eoi_token = config_.eoi_token,
      .image_prefix = "\n\n",
      .image_suffix = "\n\n",
      .add_image_end = false,
      .boa_token = config_.boa_token,
      .eoa_token = config_.eoa_token,
      .audio_prefix = "\n\n",
      .audio_suffix = "\n\n",
      .add_audio_end = true,
  };
  ImagePreprocessParameter image_preprocess_parameter;
  image_preprocess_parameter.SetTargetDimensions(Dimensions(
      {1, config_.image_tensor_height, config_.image_tensor_width, 3}));
  return ProcessMultimodalPrompt(
      rendered_template_prompt, messages, image_preprocessor_.get(),
      audio_preprocessor_.get(), multi_config, image_preprocess_parameter);
}

absl::StatusOr<ModelDataProcessor::SingleTurnTemplateRenderResult>
Gemma3DataProcessor::RenderSingleTurnTemplate(
    std::vector<Message>& history, const Preface& preface,
    const Message& message, const PromptTemplate& prompt_template,
    bool current_is_appending_message, bool append_message,
    std::optional<nlohmann::ordered_json> extra_context) const {
  return RenderSingleTurnTemplateCommon(
      *this, history, preface, message, prompt_template,
      current_is_appending_message, append_message, extra_context,
      /*push_dummy_user_message_to_preface=*/true);
}

absl::StatusOr<Message> Gemma3DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Gemma3DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  ordered_json message = {{"role", "assistant"}};
  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        ordered_json content_and_tool_calls,
        ParseTextAndToolCalls(
            response_text, config_.code_fence_start, config_.code_fence_end,
            GetSyntaxType(config_.syntax_type),
            {.escape_fence_strings = config_.escape_fence_strings,
             .tool_code_regex = config_.tool_code_regex,
             .return_error_on_parse_failure = ReturnErrorOnParseFailure()}));
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
    ABSL_ASSIGN_OR_RETURN(std::string formatted_tool, FormatToolAsPython(tool));
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
      ABSL_ASSIGN_OR_RETURN(
          audio_preprocessor_,
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
