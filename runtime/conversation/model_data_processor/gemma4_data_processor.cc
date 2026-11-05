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
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#if defined(LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS)
#include <variant>

#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#include "sentencepiece_model.pb.h"  // from @sentencepiece
#elif !defined(LITERT_LM_TOOL_CALL_CONSTRAINTS_DISABLED)
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/constrained_decoding/llg_fc_tool_calls.h"
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"
#endif  // LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS
#include "runtime/components/constrained_decoding/tool_choice.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/image_preprocessor.h"

namespace litert::lm {

#if !defined(LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS) && \
    !defined(LITERT_LM_TOOL_CALL_CONSTRAINTS_DISABLED)
namespace {

// Returns whether Lark grammars can refer to `token` as a special token.
// LLGuidance only accepts "<...>" without "<", ">" or whitespace inside.
bool IsLarkSpecialToken(absl::string_view token) {
  if (token.size() < 3 || token.front() != '<' || token.back() != '>') {
    return false;
  }
  for (const char c : token.substr(1, token.size() - 2)) {
    if (c == '<' || c == '>' || absl::ascii_isspace(c)) {
      return false;
    }
  }
  return true;
}

// Returns the first token of `text` if Lark grammars can refer to it as a
// special token, e.g. "<|channel>" for "<|channel>thought\n", but not "\n" for
// "\n</think>".
absl::StatusOr<std::string> GetFirstSpecialToken(const Tokenizer& tokenizer,
                                                 absl::string_view text) {
  Tokenizer& mutable_tokenizer = const_cast<Tokenizer&>(tokenizer);
  ABSL_ASSIGN_OR_RETURN(std::vector<int> ids,
                        mutable_tokenizer.TextToTokenIds(text));
  if (ids.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("\"", text, "\" has no tokens."));
  }
  for (size_t length = 1; length <= text.size(); ++length) {
    const absl::string_view token = text.substr(0, length);
    absl::StatusOr<int> id = mutable_tokenizer.TokenToId(token);
    if (id.ok() && *id == ids.front()) {
      if (!IsLarkSpecialToken(token)) {
        return absl::InvalidArgumentError(
            absl::StrCat("The first token of \"", absl::CEscape(text),
                         "\" isn't a special token like \"<|channel>\"."));
      }
      return std::string(token);
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("The first token of \"", text, "\" is not a prefix of it."));
}

}  // namespace
#endif  // !LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS &&
        // !LITERT_LM_TOOL_CALL_CONSTRAINTS_DISABLED

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

  std::string channel_start_token;
  std::string channel_end_token;
#if defined(LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS)
  ConstraintProviderPtr constraint_provider(
      nullptr, &LiteRtLmGemmaModelConstraintProvider_Destroy);
  if (enable_constrained_decoding) {
    if (tokenizer == nullptr) {
      return absl::InvalidArgumentError(
          "Tokenizer must not be null when constrained decoding is enabled.");
    }
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
#elif defined(LITERT_LM_TOOL_CALL_CONSTRAINTS_DISABLED)
  if (enable_constrained_decoding) {
    return absl::FailedPreconditionError(
        "Constrained decoding was disabled at build time.");
  }
  ConstraintProviderPtr constraint_provider;
#else
  std::unique_ptr<ConstraintProvider> constraint_provider = nullptr;
  if (enable_constrained_decoding) {
    if (tokenizer == nullptr) {
      return absl::InvalidArgumentError(
          "Tokenizer must not be null when constrained decoding is enabled.");
    }
    LlGuidanceConfig llg_config;
    AddStopTokensAsEosIds(stop_token_ids, llg_config);
    if (llg_config.eos_ids.empty()) {
      auto eos_id = const_cast<Tokenizer*>(tokenizer)->TokenToId("<eos>");
      if (eos_id.ok()) {
        llg_config.eos_ids.push_back(*eos_id);
      }
    }
    llg_config.special_tokens = {
        config.code_fence_start,
        config.code_fence_end,
        config.function_response_start,
        config.open_quote,
        config.close_quote,
    };
    // As special tokens, the channel markup can't be matched as text, only as
    // the channels the constraints allow at the start of the turn. So the model
    // can't hide its whole answer in a channel.
    if (!config.channel_start.empty() && !config.channel_end.empty()) {
      absl::StatusOr<std::string> start_token =
          GetFirstSpecialToken(*tokenizer, config.channel_start);
      absl::StatusOr<std::string> end_token =
          GetFirstSpecialToken(*tokenizer, config.channel_end);
      if (start_token.ok() && end_token.ok()) {
        channel_start_token = *std::move(start_token);
        channel_end_token = *std::move(end_token);
        llg_config.special_tokens.push_back(channel_start_token);
        llg_config.special_tokens.push_back(channel_end_token);
      } else {
        ABSL_LOG(WARNING) << "The tool call constraints won't accept channels: "
                          << (start_token.ok() ? end_token.status()
                                               : start_token.status());
      }
    }
    auto provider_or =
        LlgConstraintProvider::Create(*tokenizer, std::move(llg_config));
    if (provider_or.ok()) {
      constraint_provider = std::move(*provider_or);
    } else {
      ABSL_LOG(WARNING) << "Failed to create LlgConstraintProvider: "
                        << provider_or.status();
    }
  }
#endif  // LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS

  return absl::WrapUnique(new Gemma4DataProcessor(
      std::move(constraint_provider), config, preface,
      ImagePreprocessor::Create(), std::move(audio_preprocessor),
      std::move(channel_start_token), std::move(channel_end_token)));
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
      args.visual_token_budget, args.dry_run);
}

absl::StatusOr<Message> Gemma4DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Gemma4DataProcessorArguments& args) const {
  return ResponseTextToMessage(
      responses.GetTexts()[0], preface_, config_.code_fence_start,
      config_.code_fence_end, GetSyntaxType(config_.syntax_type),
      {.escape_fence_strings = config_.escape_fence_strings,
       .tool_code_regex = config_.tool_code_regex,
       .return_error_on_parse_failure = ReturnErrorOnParseFailure()});
}

absl::StatusOr<std::unique_ptr<Constraint>>
Gemma4DataProcessor::CreateConstraint(
    const nlohmann::ordered_json& tools,
    std::optional<ToolChoice> tool_choice) const {
#if defined(LITERT_LM_TOOL_CALL_CONSTRAINTS_DISABLED)
  return absl::FailedPreconditionError(
      "Constrained decoding is disabled at build time, but it was requested "
      "for inference.");
#else
  if (constraint_provider_ == nullptr) {
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

#if defined(LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS)
  if (!config_.allowed_function_names.empty()) {
    return absl::FailedPreconditionError(
        "allowed_function_names requires LLGuidance tool call constraints. "
        "Build with --define=LITERT_LM_FST_CONSTRAINTS_DISABLED=1 to enable "
        "them.");
  }
  // The FST path only supports the default "auto" tool choice.
  const bool is_auto_tool_choice =
      tool_choice.has_value() &&
      std::holds_alternative<ToolChoice::Auto>(tool_choice->choice);
  if (tool_choice.has_value() && !is_auto_tool_choice) {
    return absl::FailedPreconditionError(
        "tool_choice other than \"auto\" requires LLGuidance tool call "
        "constraints. Build with "
        "--define=LITERT_LM_FST_CONSTRAINTS_DISABLED=1 to enable them.");
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
  if (is_auto_tool_choice) {
    gemma_options.constraint_mode = kLiteRtLmGemmaConstraintModeTextAndOr;
  }
  std::string functions_str = functions.dump();
  LiteRtLmConstraint* constraint =
      LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools(
          constraint_provider_.get(), functions_str.c_str(), &gemma_options);
  if (constraint == nullptr) {
    return absl::InternalError("Failed to create constraint with tools.");
  }
  return absl::WrapUnique(reinterpret_cast<Constraint*>(constraint));
#else
  LlgConstraintsOptions options;
  options.funcall_format = FuncallFormat::kFc;
  options.code_fence_start = config_.code_fence_start;
  options.code_fence_end = config_.code_fence_end;
  options.open_quote = config_.open_quote;
  options.close_quote = config_.close_quote;
  options.function_response_start = config_.function_response_start;
  options.channel_start = channel_start_token_;
  options.channel_end = channel_end_token_;
  options.allowed_function_names = config_.allowed_function_names;

  switch (config_.constraint_mode) {
    case Gemma4DataProcessorConfig::ConstraintMode::kFunctionCallOnly:
      options.constraint_mode = LlgConstraintMode::kFunctionCallsOnly;
      break;
    case Gemma4DataProcessorConfig::ConstraintMode::kTextAndOr:
    default:
      options.constraint_mode = LlgConstraintMode::kTextAndOrFunctionCalls;
      break;
  }

  if (tool_choice.has_value()) {
    ABSL_RETURN_IF_ERROR(ApplyToolChoiceToOptions(*tool_choice, options));
  }

  ABSL_ASSIGN_OR_RETURN(std::string lark_grammar,
                        CreateLarkGrammarForFcToolCalls(functions, options));
  return constraint_provider_->CreateConstraint(
      LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kLark,
                              .constraint_string = std::move(lark_grammar)});
#endif  // LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS
#endif  // LITERT_LM_TOOL_CALL_CONSTRAINTS_DISABLED
}

absl::string_view Gemma4DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Gemma4DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
