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

#include "c/conversation.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "c/conversation_internal.h"
#include "c/engine.h"
#include "c/engine_internal.h"  // IWYU pragma: keep
#include "c/error_reporter.h"
#include "c/error_reporter_internal.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/thinking_config.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_executor_settings.h"

namespace {

absl::AnyInvocable<void(absl::StatusOr<litert::lm::Message>)>
CreateConversationCallback(LiteRtLmStreamCallback callback, void* user_data) {
  return [callback, user_data](absl::StatusOr<litert::lm::Message> message) {
    if (!message.ok()) {
      std::string error_str = message.status().ToString();
      LiteRtLmStreamChunk chunk;
      chunk.text = nullptr;
      chunk.is_final = true;
      chunk.error_msg = error_str.c_str();
      callback(user_data, &chunk);
      return;
    }
    if (message->empty()) {  // End of stream marker
      LiteRtLmStreamChunk chunk;
      chunk.text = nullptr;
      chunk.is_final = true;
      chunk.error_msg = nullptr;
      callback(user_data, &chunk);
    } else {
      std::string json_str = message->dump();
      LiteRtLmStreamChunk chunk;
      chunk.text = json_str.c_str();
      chunk.is_final = false;
      chunk.error_msg = nullptr;
      callback(user_data, &chunk);
    }
  };
}

std::optional<litert::lm::DataProcessorArguments> GetDataProcessorArguments(
    const litert::lm::Conversation* conversation,
    const int visual_token_budget) {
  bool is_gemma4 = conversation->GetConfig()
                       .GetSessionConfig()
                       .GetLlmModelType()
                       .has_gemma4();
  if (is_gemma4) {
    return litert::lm::Gemma4DataProcessorArguments{.visual_token_budget =
                                                        visual_token_budget};
  }
  return std::nullopt;
}

litert::lm::OptionalArgs CreateOptionalArgs(
    const litert::lm::Conversation* conversation, const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args) {
  litert::lm::OptionalArgs litert_lm_optional_args;
  if (extra_context) {
    auto extra_context_json =
        nlohmann::ordered_json::parse(extra_context, nullptr, false);
    if (!extra_context_json.is_null() && !extra_context_json.empty()) {
      litert_lm_optional_args.extra_context = extra_context_json;
    }
  }
  if (optional_args) {
    if (optional_args->repetition_penalty_config.has_value()) {
      litert_lm_optional_args.repetition_penalty_config =
          optional_args->repetition_penalty_config;
    }
    if (optional_args->no_repeat_ngram_config.has_value()) {
      litert_lm_optional_args.no_repeat_ngram_config =
          optional_args->no_repeat_ngram_config;
    }
    if (optional_args->suppress_tokens_config.has_value()) {
      litert_lm_optional_args.suppress_tokens_config =
          optional_args->suppress_tokens_config;
    }
    if (optional_args->visual_token_budget.has_value()) {
      litert_lm_optional_args.args = GetDataProcessorArguments(
          conversation, *optional_args->visual_token_budget);
    }
    if (optional_args->max_output_tokens.has_value()) {
      litert_lm_optional_args.max_output_tokens =
          optional_args->max_output_tokens;
    }
    if (optional_args->constraint_type != kLiteRtLmConstraintTypeNone) {
      litert::lm::LlGuidanceConstraintArg constraint_arg;
      if (optional_args->constraint_type == kLiteRtLmConstraintTypeRegex) {
        constraint_arg.constraint_type = litert::lm::LlgConstraintType::kRegex;
      } else if (optional_args->constraint_type ==
                 kLiteRtLmConstraintTypeJsonSchema) {
        constraint_arg.constraint_type =
            litert::lm::LlgConstraintType::kJsonSchema;
      } else {
        ABSL_LOG(ERROR) << "Unknown constraint type: "
                        << optional_args->constraint_type;
      }
      constraint_arg.constraint_string = optional_args->constraint_string;
      litert_lm_optional_args.decoding_constraint = constraint_arg;
    }
    if (optional_args->thinking_config.has_value()) {
      litert_lm_optional_args.thinking_config = *optional_args->thinking_config;
    }
  }
  return litert_lm_optional_args;
}

bool IsValidConstraintType(LiteRtLmConstraintType type) {
  switch (type) {
    case kLiteRtLmConstraintTypeNone:
    case kLiteRtLmConstraintTypeRegex:
    case kLiteRtLmConstraintTypeJsonSchema:
      return true;
  }
  return false;
}

bool IsValidConstraintProviderType(LiteRtLmConstraintProviderType type) {
  switch (type) {
    case kLiteRtLmConstraintProviderTypeLlGuidance:
      return true;
  }
  return false;
}

}  // namespace

using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::OptionalArgs;
using ::litert::lm::SessionConfig;

extern "C" {

LiteRtLmStatusCode litert_lm_conversation_config_create(
    LiteRtLmConversationConfig** out_config) {
  LITERT_LM_C_RETURN_IF_NULL(out_config);
  *out_config = new LiteRtLmConversationConfig;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_session_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmSessionConfig* session_config) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  if (!session_config || !session_config->config) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session config.");
  }
  config->session_config = *session_config->config;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_system_message(
    LiteRtLmConversationConfig* config, const char* system_message_json) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  LITERT_LM_C_RETURN_IF_NULL(system_message_json);
  config->system_message_json = system_message_json;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_tools(
    LiteRtLmConversationConfig* config, const char* tools_json) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  LITERT_LM_C_RETURN_IF_NULL(tools_json);
  config->tools_json = tools_json;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_messages(
    LiteRtLmConversationConfig* config, const char* messages_json) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  LITERT_LM_C_RETURN_IF_NULL(messages_json);
  config->messages_json = messages_json;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_extra_context(
    LiteRtLmConversationConfig* config, const char* extra_context_json) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  LITERT_LM_C_RETURN_IF_NULL(extra_context_json);
  config->extra_context_json = extra_context_json;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_prompt_template(
    LiteRtLmConversationConfig* config, const char* prompt_template) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  LITERT_LM_C_RETURN_IF_NULL(prompt_template);
  config->prompt_template = prompt_template;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_conversation_config_set_enable_constrained_decoding(
    LiteRtLmConversationConfig* config, bool enable_constrained_decoding) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->enable_constrained_decoding = enable_constrained_decoding;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_constraint_provider(
    LiteRtLmConversationConfig* config,
    const LiteRtLmConstraintProviderType* provider_type) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  if (provider_type == nullptr) {
    config->constraint_provider_type = std::nullopt;
    return kLiteRtLmStatusOk;
  }
  if (!IsValidConstraintProviderType(*provider_type)) {
    return litert::lm::c::ReturnError(
        absl::StatusCode::kInvalidArgument,
        "Unknown LiteRtLmConstraintProviderType.");
  }
  config->constraint_provider_type = *provider_type;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
    LiteRtLmConversationConfig* config,
    bool filter_channel_content_from_kv_cache) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->filter_channel_content_from_kv_cache =
      filter_channel_content_from_kv_cache;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_stream_tool_calls(
    LiteRtLmConversationConfig* config, bool stream_tool_calls,
    const char* channel_name) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->stream_tool_calls = stream_tool_calls;
  if (channel_name != nullptr) {
    config->stream_tool_calls_channel_name = channel_name;
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_thinking_config_create(
    LiteRtLmThinkingConfig** out_config) {
  LITERT_LM_C_RETURN_IF_NULL(out_config);
  *out_config =
      new LiteRtLmThinkingConfig{litert::lm::ThinkingConfig(true, -1)};
  return kLiteRtLmStatusOk;
}

void litert_lm_thinking_config_delete(LiteRtLmThinkingConfig* config) {
  delete config;
}

LiteRtLmStatusCode litert_lm_thinking_config_set_enable_thinking(
    LiteRtLmThinkingConfig* config, bool enable_thinking) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->thinking_config = litert::lm::ThinkingConfig(
      enable_thinking, config->thinking_config.thinking_token_budget());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_thinking_config_set_thinking_token_budget(
    LiteRtLmThinkingConfig* config, int thinking_token_budget) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->thinking_config = litert::lm::ThinkingConfig(
      config->thinking_config.enable_thinking(), thinking_token_budget);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_config_set_thinking_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmThinkingConfig* thinking_config) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  if (thinking_config) {
    config->thinking_config = thinking_config->thinking_config;
  } else {
    config->thinking_config = std::nullopt;
  }
  return kLiteRtLmStatusOk;
}

void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config) {
  delete config;
}

LiteRtLmStatusCode litert_lm_conversation_optional_args_create(
    LiteRtLmConversationOptionalArgs** out_optional_args) {
  LITERT_LM_C_RETURN_IF_NULL(out_optional_args);
  *out_optional_args = new LiteRtLmConversationOptionalArgs;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_repetition_penalty_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmRepetitionPenaltyConfig* repetition_penalty_config) {
  LITERT_LM_C_RETURN_IF_NULL(optional_args);
  if (!repetition_penalty_config ||
      !repetition_penalty_config->repetition_penalty_config.enabled()) {
    optional_args->repetition_penalty_config = std::nullopt;
    return kLiteRtLmStatusOk;
  }

  optional_args->repetition_penalty_config =
      repetition_penalty_config->repetition_penalty_config;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmNoRepeatNgramConfig* no_repeat_ngram_config) {
  LITERT_LM_C_RETURN_IF_NULL(optional_args);
  if (!no_repeat_ngram_config ||
      !no_repeat_ngram_config->no_repeat_ngram_config.enabled()) {
    optional_args->no_repeat_ngram_config = std::nullopt;
    return kLiteRtLmStatusOk;
  }

  optional_args->no_repeat_ngram_config =
      no_repeat_ngram_config->no_repeat_ngram_config;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_suppress_tokens_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmSuppressTokensConfig* suppress_tokens_config) {
  LITERT_LM_C_RETURN_IF_NULL(optional_args);
  if (!suppress_tokens_config ||
      !suppress_tokens_config->suppress_tokens_config.enabled()) {
    optional_args->suppress_tokens_config = std::nullopt;
    return kLiteRtLmStatusOk;
  }

  optional_args->suppress_tokens_config =
      suppress_tokens_config->suppress_tokens_config;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_optional_args_set_visual_token_budget(
    LiteRtLmConversationOptionalArgs* optional_args, int visual_token_budget) {
  LITERT_LM_C_RETURN_IF_NULL(optional_args);
  optional_args->visual_token_budget = visual_token_budget;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_optional_args_set_max_output_tokens(
    LiteRtLmConversationOptionalArgs* optional_args, int max_output_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(optional_args);
  optional_args->max_output_tokens = max_output_tokens;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_optional_args_set_thinking_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmThinkingConfig* thinking_config) {
  LITERT_LM_C_RETURN_IF_NULL(optional_args);
  if (thinking_config) {
    optional_args->thinking_config = thinking_config->thinking_config;
  } else {
    optional_args->thinking_config = std::nullopt;
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_optional_args_set_constraint(
    LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmConstraintType constraint_type, const char* constraint_string) {
  LITERT_LM_C_RETURN_IF_NULL(optional_args);
  if (!IsValidConstraintType(constraint_type)) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Unknown LiteRtLmConstraintType.");
  }
  optional_args->constraint_type = constraint_type;
  if (constraint_string) {
    optional_args->constraint_string = constraint_string;
  } else {
    optional_args->constraint_string.clear();
  }
  return kLiteRtLmStatusOk;
}

void litert_lm_conversation_optional_args_delete(
    LiteRtLmConversationOptionalArgs* args) {
  delete args;
}

LiteRtLmStatusCode litert_lm_conversation_create(
    LiteRtLmEngine* engine, LiteRtLmConversationConfig* c_config,
    LiteRtLmConversation** out_conversation) {
  LITERT_LM_C_RETURN_IF_NULL(out_conversation);
  *out_conversation = nullptr;
  if (!engine || !engine->engine) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid engine.");
  }

  absl::StatusOr<std::unique_ptr<Conversation>> conversation;
  if (c_config) {
    litert::lm::JsonPreface json_preface;
    if (!c_config->system_message_json.empty()) {
      nlohmann::ordered_json system_message;
      system_message["role"] = "system";
      auto content = nlohmann::ordered_json::parse(
          c_config->system_message_json, nullptr, false);
      if (content.is_discarded()) {
        system_message["content"] = c_config->system_message_json;
      } else {
        system_message["content"] = content;
      }
      json_preface.messages = nlohmann::ordered_json::array({system_message});
    }

    if (!c_config->messages_json.empty()) {
      auto messages = nlohmann::ordered_json::parse(c_config->messages_json,
                                                    nullptr, false);
      if (messages.is_discarded()) {
        ABSL_LOG(ERROR) << "Failed to parse messages JSON.";
      } else if (!messages.is_array()) {
        ABSL_LOG(ERROR) << "Messages JSON is not an array.";
      } else {
        if (json_preface.messages.is_array()) {
          json_preface.messages.insert(json_preface.messages.end(),
                                       messages.begin(), messages.end());
        } else {
          json_preface.messages = std::move(messages);
        }
      }
    }

    if (!c_config->tools_json.empty()) {
      auto tool_json_parsed =
          nlohmann::ordered_json::parse(c_config->tools_json, nullptr, false);
      if (!tool_json_parsed.is_discarded() && tool_json_parsed.is_array()) {
        json_preface.tools = tool_json_parsed;
      } else {
        ABSL_LOG(ERROR) << "Failed to parse tools JSON or not an array: "
                        << c_config->tools_json;
      }
    }

    if (!c_config->extra_context_json.empty()) {
      auto extra_context_parsed = nlohmann::ordered_json::parse(
          c_config->extra_context_json, nullptr, false);
      if (!extra_context_parsed.is_discarded() &&
          extra_context_parsed.is_object()) {
        json_preface.extra_context = std::move(extra_context_parsed);
      } else {
        ABSL_LOG(ERROR)
            << "Failed to parse extra context JSON or not an object: "
            << c_config->extra_context_json;
      }
    }

    auto builder = litert::lm::ConversationConfig::Builder();
    SessionConfig session_config = c_config->session_config
                                       ? *c_config->session_config
                                       : SessionConfig::CreateDefault();
    if (engine->engine->GetEngineSettings()
            .GetAudioExecutorSettings()
            .has_value()) {
      session_config.SetAudioModalityEnabled(true);
    }
    if (engine->engine->GetEngineSettings()
            .GetVisionExecutorSettings()
            .has_value()) {
      session_config.SetVisionModalityEnabled(true);
    }
    builder.SetSessionConfig(session_config);

    builder.SetPreface(json_preface);
    builder.SetEnableConstrainedDecoding(c_config->enable_constrained_decoding);

    if (c_config->constraint_provider_type.has_value() &&
        *c_config->constraint_provider_type ==
            kLiteRtLmConstraintProviderTypeLlGuidance) {
      builder.SetConstraintProviderConfig(litert::lm::LlGuidanceConfig());
    }

    if (c_config->filter_channel_content_from_kv_cache.has_value()) {
      builder.SetFilterChannelContentFromKvCache(
          *c_config->filter_channel_content_from_kv_cache);
    }
    builder.SetStreamToolCalls(c_config->stream_tool_calls,
                               c_config->stream_tool_calls_channel_name);
    if (!c_config->prompt_template.empty()) {
      builder.SetOverwritePromptTemplate(
          litert::lm::PromptTemplate(c_config->prompt_template));
    }
    if (c_config->thinking_config.has_value()) {
      builder.SetThinkingConfig(*c_config->thinking_config);
    }
    // For disabling rewinding, we don't use a config, but instead force the
    // option if and only if GPU artisan ringbuffers are being used in the
    // engine.
    auto& main_settings =
        engine->engine->GetEngineSettings().GetMainExecutorSettings();
    auto gpu_artisan_config =
        main_settings.GetBackendConfig<litert::lm::GpuArtisanConfig>();
    if (gpu_artisan_config.ok() &&
        gpu_artisan_config->use_autosized_ringbuffers) {
      builder.SetEnableRewinding(false);
    }
    auto config = builder.Build(*engine->engine);

    if (!config.ok()) {
      ABSL_LOG(ERROR) << "Failed to create conversation config: "
                      << config.status();
      return litert::lm::c::ToCStatus(config.status());
    }
    conversation = Conversation::Create(*engine->engine, *config);
  } else {
    auto default_conversation_config =
        ConversationConfig::CreateDefault(*engine->engine);
    if (!default_conversation_config.ok()) {
      ABSL_LOG(ERROR) << "Failed to create default conversation config: "
                      << default_conversation_config.status();
      return litert::lm::c::ToCStatus(default_conversation_config.status());
    }
    conversation =
        Conversation::Create(*engine->engine, *default_conversation_config);
  }

  if (!conversation.ok()) {
    ABSL_LOG(ERROR) << "Failed to create conversation: "
                    << conversation.status();
    return litert::lm::c::ToCStatus(conversation.status());
  }
  auto c_conversation = std::make_unique<LiteRtLmConversation>();
  c_conversation->conversation = *std::move(conversation);
  *out_conversation = c_conversation.release();
  return kLiteRtLmStatusOk;
}

void litert_lm_conversation_delete(LiteRtLmConversation* conversation) {
  delete conversation;
}

LiteRtLmStatusCode litert_lm_conversation_clone(
    LiteRtLmConversation* conversation,
    LiteRtLmConversation** out_conversation) {
  LITERT_LM_C_RETURN_IF_NULL(out_conversation);
  *out_conversation = nullptr;
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  auto cloned = conversation->conversation->Clone();
  if (!cloned.ok()) {
    ABSL_LOG(ERROR) << "Failed to clone conversation: " << cloned.status();
    return litert::lm::c::ToCStatus(cloned.status());
  }
  auto c_conversation = std::make_unique<LiteRtLmConversation>();
  c_conversation->conversation = std::move(*cloned);
  *out_conversation = c_conversation.release();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmJsonResponse** out_response) {
  LITERT_LM_C_RETURN_IF_NULL(out_response);
  *out_response = nullptr;
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  LITERT_LM_C_RETURN_IF_NULL(message_json);
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Failed to parse message JSON.");
  }

  OptionalArgs litert_lm_optional_args = CreateOptionalArgs(
      conversation->conversation.get(), extra_context, optional_args);

  auto response = conversation->conversation->SendMessage(
      json_message, std::move(litert_lm_optional_args));
  if (!response.ok()) {
    ABSL_LOG(ERROR) << "Failed to send message: " << response.status();
    return litert::lm::c::ToCStatus(response.status());
  }
  auto c_response = std::make_unique<LiteRtLmJsonResponse>();
  c_response->json_string = response->dump();
  *out_response = c_response.release();
  return kLiteRtLmStatusOk;
}

void litert_lm_json_response_delete(LiteRtLmJsonResponse* response) {
  delete response;
}

LiteRtLmStatusCode litert_lm_json_response_get_string(
    const LiteRtLmJsonResponse* response, const char** out_json) {
  LITERT_LM_C_RETURN_IF_NULL(out_json);
  *out_json = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(response);
  *out_json = response->json_string.c_str();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmStreamCallback callback, void* callback_data) {
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  LITERT_LM_C_RETURN_IF_NULL(message_json);
  LITERT_LM_C_RETURN_IF_NULL(callback);
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Failed to parse message JSON.");
  }

  litert::lm::OptionalArgs litert_lm_optional_args = CreateOptionalArgs(
      conversation->conversation.get(), extra_context, optional_args);

  absl::Status status = conversation->conversation->SendMessageAsync(
      json_message, CreateConversationCallback(callback, callback_data),
      std::move(litert_lm_optional_args));

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start message stream: " << status;
    return litert::lm::c::ToCStatus(status);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_render_message_to_string(
    LiteRtLmConversation* conversation, const char* message_json,
    const char** out_text) {
  LITERT_LM_C_RETURN_IF_NULL(out_text);
  *out_text = nullptr;
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  LITERT_LM_C_RETURN_IF_NULL(message_json);
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Failed to parse message JSON.");
  }

  auto rendered = conversation->conversation->RenderMessageIntoString(
      json_message, litert::lm::OptionalArgs());
  if (!rendered.ok()) {
    ABSL_LOG(ERROR) << "Failed to render message: " << rendered.status();
    return litert::lm::c::ToCStatus(rendered.status());
  }
  conversation->last_rendered_message = std::move(*rendered);
  *out_text = conversation->last_rendered_message.c_str();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_render_preface_to_string(
    LiteRtLmConversation* conversation, const char** out_text) {
  LITERT_LM_C_RETURN_IF_NULL(out_text);
  *out_text = nullptr;
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  auto rendered = conversation->conversation->RenderPrefaceIntoString(
      litert::lm::OptionalArgs());
  if (!rendered.ok()) {
    ABSL_LOG(ERROR) << "Failed to render preface: " << rendered.status();
    return litert::lm::c::ToCStatus(rendered.status());
  }
  conversation->last_rendered_preface = std::move(*rendered);
  *out_text = conversation->last_rendered_preface.c_str();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_cancel_process(
    LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  conversation->conversation->CancelProcess();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_wait_until_done(
    LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  absl::Status status = conversation->conversation->WaitUntilDone();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to wait until done: " << status;
    return litert::lm::c::ToCStatus(status);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_get_benchmark_info(
    LiteRtLmConversation* conversation,
    LiteRtLmBenchmarkInfo** out_benchmark_info) {
  LITERT_LM_C_RETURN_IF_NULL(out_benchmark_info);
  *out_benchmark_info = nullptr;
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  auto benchmark_info = conversation->conversation->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ABSL_LOG(ERROR) << "Failed to get benchmark info: "
                    << benchmark_info.status();
    return litert::lm::c::ToCStatus(benchmark_info.status());
  }
  *out_benchmark_info = new LiteRtLmBenchmarkInfo{std::move(*benchmark_info)};
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_conversation_get_token_count(
    LiteRtLmConversation* conversation, int* out_count) {
  LITERT_LM_C_RETURN_IF_NULL(out_count);
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
  }
  absl::StatusOr<int> token_count = conversation->conversation->GetTokenCount();
  if (!token_count.ok()) {
    ABSL_LOG(ERROR) << "Failed to get token count: " << token_count.status();
    return litert::lm::c::ToCStatus(token_count.status());
  }
  *out_count = *token_count;
  return kLiteRtLmStatusOk;
}

}  // extern "C"
