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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#endif
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/image_preprocessor.h"
#include "sentencepiece_model.pb.h"  // from @sentencepiece

namespace litert::lm {

absl::StatusOr<std::unique_ptr<Gemma4DataProcessor>>
Gemma4DataProcessor::Create(Gemma4DataProcessorConfig config,
                            std::optional<Preface> preface,
                            const Tokenizer* tokenizer,
                            const std::vector<std::vector<int>>& stop_token_ids,
                            bool enable_constrained_decoding) {
  // For Gemma4 models without mel-spectrogram extraction, it use 640 PCM
  // samples as one audio token.
  ABSL_ASSIGN_OR_RETURN(
      auto audio_preprocessor,
      AudioPreprocessorMiniAudio::Create(
          litert::support::AudioPreprocessorConfig::CreateDefaultGemma4Config(
              config.skip_mel_spectrogram_extraction)));
#if defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
  if (enable_constrained_decoding) {
    return absl::FailedPreconditionError(
        "Constrained decoding was disabled at build time.");
  }
  return absl::WrapUnique(
      new Gemma4DataProcessor(config, preface, ImagePreprocessor::Create(),
                              std::move(audio_preprocessor)));
#else
  std::unique_ptr<LiteRtLmGemmaModelConstraintProvider,
                  decltype(&LiteRtLmGemmaModelConstraintProvider_Destroy)>
      constraint_provider(nullptr,
                          &LiteRtLmGemmaModelConstraintProvider_Destroy);
  if (enable_constrained_decoding) {
    if (tokenizer->GetTokenizerType() != TokenizerType::kSentencePiece) {
      ABSL_LOG(WARNING)
          << "Constrained decoding is only supported for SentencePiece "
             "tokenizer.";
    } else {
      std::vector<const int*> stop_token_ids_ptrs;
      std::vector<size_t> stop_token_lengths;
      stop_token_ids_ptrs.reserve(stop_token_ids.size());
      stop_token_lengths.reserve(stop_token_ids.size());
      for (const auto& stop_tokens : stop_token_ids) {
        stop_token_ids_ptrs.push_back(stop_tokens.data());
        stop_token_lengths.push_back(stop_tokens.size());
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
  }
  return absl::WrapUnique(new Gemma4DataProcessor(
      std::move(constraint_provider), config, preface,
      ImagePreprocessor::Create(), std::move(audio_preprocessor)));
#endif
}

absl::StatusOr<nlohmann::ordered_json>
Gemma4DataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  return message;
}

absl::StatusOr<std::vector<InputData>>
Gemma4DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const Gemma4DataProcessorArguments& args) const {
  MultimodalPromptProcessingConfig multi_config{
      .delimiter_regex =
          R"regex((<start_of_image>|<\|image\|>|<start_of_audio>|<\|audio\|>))regex",
      .image_token_regex = R"regex((<start_of_image>|<\|image\|>))regex",
      .audio_token_regex = R"regex((<start_of_audio>|<\|audio\|>))regex",
      .boi_token = config_.boi_token,
      .eoi_token = config_.eoi_token,
      .image_prefix = "",
      .image_suffix = "",
      .add_image_end = true,
      .boa_token = config_.boa_token,
      .eoa_token = config_.eoa_token,
      .audio_prefix = "",
      .audio_suffix = "",
      .add_audio_end = true,
  };
  ImagePreprocessParameter image_preprocess_parameter;
  image_preprocess_parameter.SetPatchifyConfig(
      ImagePreprocessParameter::PatchifyConfig{
          .patch_width = config_.patch_width,
          .patch_height = config_.patch_height,
          .max_num_patches = config_.max_num_patches,
          .pooling_kernel_size = config_.pooling_kernel_size,
          .merge_patches = config_.merge_patches,
      });
  return ProcessMultimodalPrompt(
      rendered_template_prompt, messages, image_preprocessor_.get(),
      audio_preprocessor_.get(), multi_config, image_preprocess_parameter,
      args.visual_token_budget);
}

absl::StatusOr<Message> Gemma4DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Gemma4DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  nlohmann::ordered_json message = {{"role", "assistant"}};
  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        nlohmann::ordered_json content_and_tool_calls,
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
    message["content"] = nlohmann::ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return message;
}

absl::StatusOr<ModelDataProcessor::SingleTurnTemplateRenderResult>
Gemma4DataProcessor::RenderSingleTurnTemplate(
    std::vector<Message>& history, const Preface& preface,
    const Message& message, const PromptTemplate& prompt_template,
    bool current_is_appending_message, bool append_message,
    std::optional<nlohmann::ordered_json> extra_context) const {
  return RenderSingleTurnTemplateCommon(
      *this, history, preface, message, prompt_template,
      current_is_appending_message, append_message, extra_context,
      /*push_dummy_user_message_to_preface=*/false);
}

absl::StatusOr<nlohmann::ordered_json> Gemma4DataProcessor::FormatTools(
    const nlohmann::ordered_json& tools) const {
  return tools;
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
