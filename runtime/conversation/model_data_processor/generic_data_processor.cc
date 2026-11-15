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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {
namespace {

absl::Status ValidateAndCompileToolCallConfig(
    const proto::ToolCallConfig& tool_call_config,
    std::unique_ptr<RE2>* tool_call_re = nullptr,
    std::unique_ptr<RE2>* argument_re = nullptr) {
  if (tool_call_config.fence_start().empty() ||
      tool_call_config.fence_end().empty()) {
    return absl::InvalidArgumentError(
        "ToolCallConfig must specify non-empty fence_start and fence_end.");
  }
  switch (tool_call_config.syntax_config_case()) {
    case proto::ToolCallConfig::kRegexSyntax: {
      const auto& regex_syntax = tool_call_config.regex_syntax();
      if (regex_syntax.tool_call_regex().empty()) {
        return absl::InvalidArgumentError(
            "tool_call_regex must not be empty for RegexSyntax.");
      }
      auto compiled_tool_call_re =
          std::make_unique<RE2>(regex_syntax.tool_call_regex());
      if (!compiled_tool_call_re->ok() ||
          compiled_tool_call_re->NumberOfCapturingGroups() < 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid tool_call_regex: ", regex_syntax.tool_call_regex()));
      }
      std::unique_ptr<RE2> compiled_argument_re;
      if (!regex_syntax.argument_regex().empty()) {
        compiled_argument_re =
            std::make_unique<RE2>(regex_syntax.argument_regex());
        if (!compiled_argument_re->ok() ||
            compiled_argument_re->NumberOfCapturingGroups() < 2) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Invalid argument_regex: ", regex_syntax.argument_regex()));
        }
      }
      if (tool_call_re != nullptr) {
        *tool_call_re = std::move(compiled_tool_call_re);
      }
      if (argument_re != nullptr) {
        *argument_re = std::move(compiled_argument_re);
      }
      return absl::OkStatus();
    }
    case proto::ToolCallConfig::SYNTAX_CONFIG_NOT_SET:
      return absl::InvalidArgumentError(
          "ToolCallConfig must specify a syntax_config.");
  }
  return absl::InvalidArgumentError(
      "Unsupported ToolCallConfig syntax_config.");
}

}  // namespace

absl::Status GenericDataProcessor::ValidateToolCallConfig(
    const proto::ToolCallConfig& tool_call_config) {
  return ValidateAndCompileToolCallConfig(tool_call_config);
}

absl::StatusOr<std::unique_ptr<ModelDataProcessor>>
GenericDataProcessor::Create(GenericDataProcessorConfig config,
                             std::optional<Preface> preface) {
  std::unique_ptr<ImagePreprocessor> image_preprocessor = nullptr;
  std::unique_ptr<AudioPreprocessor> audio_preprocessor = nullptr;
  std::unique_ptr<RE2> tool_call_re = nullptr;
  std::unique_ptr<RE2> argument_re = nullptr;

  if (config.tool_call_config.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateAndCompileToolCallConfig(
        *config.tool_call_config, &tool_call_re, &argument_re));
  }

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
      std::move(config), std::move(preface), std::move(image_preprocessor),
      std::move(audio_preprocessor), std::move(tool_call_re),
      std::move(argument_re)));
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
      config_.multimodal->image_preprocess_parameter, args.visual_token_budget,
      args.dry_run);
}

absl::StatusOr<Message> GenericDataProcessor::ToMessageImpl(
    const Responses& responses,
    const GenericDataProcessorArguments& args) const {
  if (responses.GetTexts().empty()) {
    return absl::InternalError("No response text found.");
  }
  absl::string_view response_text = responses.GetTexts()[0];
  if (!config_.tool_call_config.has_value()) {
    return nlohmann::ordered_json::object(
        {{"role", "assistant"},
         {"content",
          nlohmann::ordered_json::array(
              {{{"type", "text"}, {"text", std::string(response_text)}}})}});
  }

  const auto& tool_call_config = *config_.tool_call_config;
  SyntaxType syntax_type = SyntaxType::kUnknown;
  ParserOptions options;
  options.return_error_on_parse_failure = ReturnErrorOnParseFailure();

  switch (tool_call_config.syntax_config_case()) {
    case proto::ToolCallConfig::kRegexSyntax:
      syntax_type = SyntaxType::kRegex;
      options.regex_options = RegexParserOptions{
          .tool_call_regex = tool_call_re_.get(),
          .argument_regex = argument_re_.get(),
      };
      break;
    case proto::ToolCallConfig::SYNTAX_CONFIG_NOT_SET:
      break;
  }

  return ResponseTextToMessage(
      response_text, preface_, tool_call_config.fence_start(),
      tool_call_config.fence_end(), syntax_type, options);
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
