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

#include "runtime/conversation/model_data_processor/gemma4_data_processor.h"

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
#include "runtime/components/constrained_decoding/constraint.h"
#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#endif
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/components/preprocessor/audio_preprocessor_miniaudio.h"
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/components/tool_use/fc_tool_format_utils.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2
#include "sentencepiece_model.pb.h"  // from @sentencepiece

namespace litert::lm {
namespace {

// Formats a tool response in FC format.
//
// The expected structure of a tool response is:
//
// ```json
//   {
//     "name": "foo",
//     "response": {
//       "key1": "bar",
//       "key2": true
//     }
//   }
// ```
//
// (fields are inside an object named "response")
//
// or
//
// ```json
// {
//   "name": "foo",
//   "key1": "bar",
//   "key2": true
// }
// ```
//
// (fields are at the top-level)
//
// The key-value pairs of the response can be set under the key "response" or
// "value".
//
// The tool's name can be set under the key "name" or "tool_name".
//
// The formatted tool response string will look like:
//
// ```
// foo{key1:<escape>bar<escape>,key2:true}
// ```
//
// If the tool response does not have a tool name, the tool response will be
// formatted as an FC object *without* a tool name before it. This is not a
// valid tool response in FC format, but it's the best we can do without a tool
// name.
absl::StatusOr<std::string> FormatToolResponse(
    const nlohmann::ordered_json& tool_response, absl::string_view escape_tag) {
  std::optional<std::string> tool_name;
  if (tool_response.contains("name") && tool_response["name"].is_string()) {
    tool_name = tool_response["name"].get<std::string>();
  } else if (tool_response.contains("tool_name") &&
             tool_response["tool_name"].is_string()) {
    tool_name = tool_response["tool_name"].get<std::string>();
  }

  if (!tool_name.has_value()) {
    return FormatValueAsFc(tool_response, escape_tag);
  }

  // If the contents of the tool response are contained inside a "response" or
  // "value" field, format it as an FC object.
  nlohmann::ordered_json response;
  if (tool_response.contains("response") &&
      tool_response["response"].is_object()) {
    response = tool_response["response"];
  } else if (tool_response.contains("value") &&
             tool_response["value"].is_object()) {
    response = tool_response["value"];
  }

  if (!response.is_null()) {
    ASSIGN_OR_RETURN(std::string value, FormatValueAsFc(response, escape_tag));
    return absl::StrCat(*tool_name, value);
  }

  // If the contents of the tool response are at the top-level, format them as
  // an FC object (without the tool name).
  nlohmann::ordered_json fields = tool_response;
  fields.erase("tool_name");
  fields.erase("name");
  ASSIGN_OR_RETURN(std::string value, FormatValueAsFc(fields, escape_tag));
  return absl::StrCat(*tool_name, value);
}

// Formats "content" as a tool response in FC format.
//
// Case 1: If "content" is an object, formats "content" directly as a tool
// response in FC format as a string.
//
// Case 2: If "content" is an array, formats each tool response item in the
// array in FC format and returns an array of *text* items. A tool response
// item is an object with "name" and "response" fields or an object with a
// "tool_response" field.
//
// Case 3: If "content" is neither an object nor an array, returns it unchanged.
absl::StatusOr<nlohmann::ordered_json> FormatToolResponses(
    const nlohmann::ordered_json& content, absl::string_view escape_tag) {
  if (content.is_object()) {
    return FormatToolResponse(content, escape_tag);
  }

  if (content.is_array()) {
    nlohmann::ordered_json tool_content = nlohmann::ordered_json::array();
    for (const auto& item : content) {
      nlohmann::ordered_json tool_response;
      if (item.contains("tool_response")) {
        tool_response = item["tool_response"];
      } else {
        tool_response = item;
      }

      // Format each tool response in FC format and add it as a text item.
      ASSIGN_OR_RETURN(std::string formatted_tool_response,
                       FormatToolResponse(tool_response, escape_tag));
      tool_content.push_back(
          {{"type", "text"}, {"text", formatted_tool_response}});
    }

    return tool_content;
  }

  // If the content of the message is not an array or object, pass it through
  // unchanged.
  return content;
}

// A message is a tool response if its role is "tool".
bool IsToolMessage(const nlohmann::ordered_json& template_input,
                   const nlohmann::ordered_json& message) {
  return template_input.contains("role") && template_input["role"] == "tool";
}

bool IsImage(absl::string_view part) {
  return part == "<start_of_image>" || part == "<image_soft_token>" ||
         part == "<|image|>";
}

bool IsAudio(absl::string_view part) {
  return part == "<start_of_audio>" || part == "<audio_soft_token>" ||
         part == "<|audio|>";
}

}  // namespace

absl::StatusOr<std::unique_ptr<Gemma4DataProcessor>>
Gemma4DataProcessor::Create(Gemma4DataProcessorConfig config,
                            std::optional<Preface> preface,
                            const Tokenizer* tokenizer,
                            const std::vector<std::vector<int>>& stop_token_ids,
                            bool enable_constrained_decoding) {
  ASSIGN_OR_RETURN(
      auto audio_preprocessor,
      AudioPreprocessorMiniAudio::Create(AudioPreprocessorConfig::Create(
          /* sample_rate_hz= */ 16000,
          /* num_channels= */ 1,
          /* frame_length= */ 320,
          /* hop_length= */ 160,
          /* fft_length = */ 512,
          /* input_scale = */ 1.0,
          /* pre_emphasis_factor = */ 0.0,
          /* num_mel_bins= */ 128,
          /* mel_low_hz= */ 0.0,
          /* mel_high_hz= */ 8000.0,
          /* mel_floor= */ 1e-3,
          /* normalize_mel= */ false,
          /* add_floor_to_mel_before_log= */ true,
          /* semicausal_padding= */ true,
          /* non_zero_hanning= */ false,
          /* periodic_hanning= */ true,
          /* fft_padding_type= */
          AudioPreprocessorConfig::FftPaddingType::kCenter)));
#if defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
  if (enable_constrained_decoding) {
    return absl::FailedPreconditionError(
        "Constrained decoding was disabled at build time.");
  }
  return absl::WrapUnique(new Gemma4DataProcessor(
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
  return absl::WrapUnique(new Gemma4DataProcessor(
      std::move(constraint_provider), config, preface,
      std::make_unique<StbImagePreprocessor>(), std::move(audio_preprocessor)));
#endif
}

absl::StatusOr<nlohmann::ordered_json>
Gemma4DataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  if (config_.use_template_for_fc_format) {
    return message;
  }

  // If the message doesn't contain any tool calls and isn't a tool message,
  // then the template input is the same as the message.
  if (!message.contains("tool_calls") && message["role"] != "tool") {
    return message;
  }

  nlohmann::ordered_json template_input = nlohmann::ordered_json::object();
  if (message.contains("role")) {
    template_input["role"] = message["role"];
  }

  if (message.contains("content")) {
    if (IsToolMessage(template_input, message)) {
      // Convert tool responses to FC format.
      ASSIGN_OR_RETURN(
          template_input["content"],
          FormatToolResponses(message["content"], config_.open_quote));
    } else {
      // If the role is not "tool" or "content" is a string, pass through the
      // content unchanged.
      template_input["content"] = message["content"];
    }
  }

  // If the message contains tool calls, convert them to FC format and add them
  // to the template input.
  if (message.contains("tool_calls")) {
    template_input["tool_calls"] = nlohmann::ordered_json::array();
    for (const auto& tool_call : message["tool_calls"]) {
      if (!tool_call.contains("function")) {
        continue;
      }
      const nlohmann::ordered_json& function = tool_call["function"];
      nlohmann::ordered_json tool_call_input = nlohmann::ordered_json::object();
      tool_call_input["type"] = "function";
      tool_call_input["function"]["name"] = function["name"];

      if (function.contains("arguments")) {
        if (function["arguments"].is_object()) {
          // If `arguments` is an object, format the values in FC format.
          for (const auto& [key, value] : function["arguments"].items()) {
            ASSIGN_OR_RETURN(std::string formatted_value,
                             FormatValueAsFc(value, config_.open_quote));
            tool_call_input["function"]["arguments"][key] = formatted_value;
          }
        } else {
          // Otherwise, pass through `arguments` unchanged.
          tool_call_input["function"]["arguments"] = function["arguments"];
        }
      }

      template_input["tool_calls"].push_back(tool_call_input);
    }
  }

  return template_input;
}

absl::StatusOr<std::vector<InputData>>
Gemma4DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const Gemma4DataProcessorArguments& args) const {
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
      R"regex((<start_of_image>|<\|image\|>|<start_of_audio>|<\|audio\|>))regex");
  absl::string_view prompt_view(rendered_template_prompt);
  const char* start = prompt_view.data();
  std::string part;
  ImagePreprocessParameter image_params;
  image_params.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = config_.patch_width,
      .patch_height = config_.patch_height,
      .max_num_patches = config_.max_num_patches,
      .pooling_kernel_size = config_.pooling_kernel_size});
  // Replace the placeholders with the actual data.
  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_part(start, prompt_view.data() - part.size());
    start = prompt_view.data();
    if (IsImage(part)) {
      // Vision modality needs double newlines before image soft token.
      // while audio modality does not. See b/476152260.
      input_data.emplace_back(
          InputText(absl::StrCat(text_part, config_.boi_token)));
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
      input_data.emplace_back(InputImageEnd());
      input_data.emplace_back(InputText("\n\n"));
    } else if (IsAudio(part)) {
      input_data.emplace_back(
          InputText(absl::StrCat(text_part, config_.boa_token)));
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

absl::StatusOr<Message> Gemma4DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Gemma4DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  nlohmann::ordered_json message = {{"role", "assistant"}};
  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ASSIGN_OR_RETURN(
        nlohmann::ordered_json content_and_tool_calls,
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
    message["content"] = nlohmann::ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return message;
}

absl::StatusOr<nlohmann::ordered_json> Gemma4DataProcessor::FormatTools(
    const nlohmann::ordered_json& tools) const {
  if (config_.use_template_for_fc_format) {
    return tools;
  }

  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  nlohmann::ordered_json formatted_tools = nlohmann::ordered_json::array();
  for (const auto& tool : tools) {
    ASSIGN_OR_RETURN(std::string formatted_tool,
                     FormatToolAsFc(tool, config_.open_quote));
    formatted_tools.push_back(formatted_tool);
  }
  return formatted_tools;
}

absl::StatusOr<std::unique_ptr<Constraint>>
Gemma4DataProcessor::CreateConstraint(
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
      .funcall_format = kLiteRtLmGemmaFuncallFormatFcStyle,
      .code_fence_start = config_.code_fence_start.c_str(),
      .code_fence_end = config_.code_fence_end.c_str(),
      .open_quote = config_.open_quote.c_str(),
      .close_quote = config_.close_quote.c_str(),
      .function_response_start = config_.function_response_start.c_str()};
  switch (config_.constraint_mode) {
    case Gemma4DataProcessorConfig::ConstraintMode::kFunctionCallOnly:
      gemma_options.constraint_mode =
          kLiteRtLmGemmaConstraintModeFunctionCallOnly;
      break;
    case Gemma4DataProcessorConfig::ConstraintMode::kTextAndOr:
    default:
      gemma_options.constraint_mode = kLiteRtLmGemmaConstraintModeTextAndOr;
      break;
  }
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

absl::string_view Gemma4DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Gemma4DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
