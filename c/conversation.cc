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

}  // namespace

using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::OptionalArgs;
using ::litert::lm::SessionConfig;

extern "C" {

LiteRtLmConversationConfig* litert_lm_conversation_config_create() {
  return new LiteRtLmConversationConfig;
}

void litert_lm_conversation_config_set_session_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmSessionConfig* session_config) {
  if (config && session_config && session_config->config) {
    config->session_config = *session_config->config;
  }
}

void litert_lm_conversation_config_set_system_message(
    LiteRtLmConversationConfig* config, const char* system_message_json) {
  if (config && system_message_json) {
    config->system_message_json = system_message_json;
  }
}

void litert_lm_conversation_config_set_tools(LiteRtLmConversationConfig* config,
                                             const char* tools_json) {
  if (config && tools_json) {
    config->tools_json = tools_json;
  }
}

void litert_lm_conversation_config_set_messages(
    LiteRtLmConversationConfig* config, const char* messages_json) {
  if (config && messages_json) {
    config->messages_json = messages_json;
  }
}

void litert_lm_conversation_config_set_extra_context(
    LiteRtLmConversationConfig* config, const char* extra_context_json) {
  if (config && extra_context_json) {
    config->extra_context_json = extra_context_json;
  }
}

void litert_lm_conversation_config_set_prompt_template(
    LiteRtLmConversationConfig* config, const char* prompt_template) {
  if (config && prompt_template) {
    config->prompt_template = prompt_template;
  }
}

void litert_lm_conversation_config_set_enable_constrained_decoding(
    LiteRtLmConversationConfig* config, bool enable_constrained_decoding) {
  if (config) {
    config->enable_constrained_decoding = enable_constrained_decoding;
  }
}

void litert_lm_conversation_config_set_constraint_provider(
    LiteRtLmConversationConfig* config,
    const LiteRtLmConstraintProviderType* provider_type) {
  if (config) {
    if (provider_type != nullptr) {
      config->constraint_provider_type = *provider_type;
    } else {
      config->constraint_provider_type = std::nullopt;
    }
  }
}

void litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
    LiteRtLmConversationConfig* config,
    bool filter_channel_content_from_kv_cache) {
  if (config) {
    config->filter_channel_content_from_kv_cache =
        filter_channel_content_from_kv_cache;
  }
}

void litert_lm_conversation_config_set_stream_tool_calls(
    LiteRtLmConversationConfig* config, bool stream_tool_calls,
    const char* channel_name) {
  if (config) {
    config->stream_tool_calls = stream_tool_calls;
    if (channel_name != nullptr) {
      config->stream_tool_calls_channel_name = channel_name;
    }
  }
}

LiteRtLmThinkingConfig* litert_lm_thinking_config_create() {
  return new LiteRtLmThinkingConfig{litert::lm::ThinkingConfig(true, -1)};
}

void litert_lm_thinking_config_delete(LiteRtLmThinkingConfig* config) {
  delete config;
}

void litert_lm_thinking_config_set_enable_thinking(
    LiteRtLmThinkingConfig* config, bool enable_thinking) {
  if (config) {
    config->thinking_config = litert::lm::ThinkingConfig(
        enable_thinking, config->thinking_config.thinking_token_budget());
  }
}

void litert_lm_thinking_config_set_thinking_token_budget(
    LiteRtLmThinkingConfig* config, int thinking_token_budget) {
  if (config) {
    config->thinking_config = litert::lm::ThinkingConfig(
        config->thinking_config.enable_thinking(), thinking_token_budget);
  }
}

void litert_lm_conversation_config_set_thinking_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmThinkingConfig* thinking_config) {
  if (config) {
    if (thinking_config) {
      config->thinking_config = thinking_config->thinking_config;
    } else {
      config->thinking_config = std::nullopt;
    }
  }
}

void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config) {
  delete config;
}

LiteRtLmConversationOptionalArgs*
litert_lm_conversation_optional_args_create() {
  return new LiteRtLmConversationOptionalArgs;
}

void litert_lm_conversation_optional_args_set_repetition_penalty_config(
    LiteRtLmConversationOptionalArgs* args,
    const LiteRtLmRepetitionPenaltyConfig* repetition_penalty_config) {
  if (!args) {
    return;
  }

  if (!repetition_penalty_config ||
      !repetition_penalty_config->repetition_penalty_config.enabled()) {
    args->repetition_penalty_config = std::nullopt;
    return;
  }

  args->repetition_penalty_config =
      repetition_penalty_config->repetition_penalty_config;
}

void litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
    LiteRtLmConversationOptionalArgs* args,
    const LiteRtLmNoRepeatNgramConfig* no_repeat_ngram_config) {
  if (!args) {
    return;
  }

  if (!no_repeat_ngram_config ||
      !no_repeat_ngram_config->no_repeat_ngram_config.enabled()) {
    args->no_repeat_ngram_config = std::nullopt;
    return;
  }

  args->no_repeat_ngram_config = no_repeat_ngram_config->no_repeat_ngram_config;
}

void litert_lm_conversation_optional_args_set_suppress_tokens_config(
    LiteRtLmConversationOptionalArgs* args,
    const LiteRtLmSuppressTokensConfig* suppress_tokens_config) {
  if (!args) {
    return;
  }

  if (!suppress_tokens_config ||
      !suppress_tokens_config->suppress_tokens_config.enabled()) {
    args->suppress_tokens_config = std::nullopt;
    return;
  }

  args->suppress_tokens_config = suppress_tokens_config->suppress_tokens_config;
}

void litert_lm_conversation_optional_args_set_visual_token_budget(
    LiteRtLmConversationOptionalArgs* args, int visual_token_budget) {
  if (args) {
    args->visual_token_budget = visual_token_budget;
  }
}

void litert_lm_conversation_optional_args_set_max_output_tokens(
    LiteRtLmConversationOptionalArgs* args, int max_output_tokens) {
  if (args) {
    args->max_output_tokens = max_output_tokens;
  }
}

void litert_lm_conversation_optional_args_set_thinking_config(
    LiteRtLmConversationOptionalArgs* args,
    const LiteRtLmThinkingConfig* thinking_config) {
  if (args) {
    if (thinking_config) {
      args->thinking_config = thinking_config->thinking_config;
    } else {
      args->thinking_config = std::nullopt;
    }
  }
}

void litert_lm_conversation_optional_args_set_constraint(
    LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmConstraintType constraint_type, const char* constraint_string) {
  if (optional_args) {
    optional_args->constraint_type = constraint_type;
    if (constraint_string) {
      optional_args->constraint_string = constraint_string;
    } else {
      optional_args->constraint_string.clear();
    }
  }
}

void litert_lm_conversation_optional_args_delete(
    LiteRtLmConversationOptionalArgs* args) {
  delete args;
}

LiteRtLmConversation* litert_lm_conversation_create(
    LiteRtLmEngine* engine, LiteRtLmConversationConfig* c_config) {
  if (!engine || !engine->engine) {
    return nullptr;
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
      return nullptr;
    }
    conversation = Conversation::Create(*engine->engine, *config);
  } else {
    auto default_conversation_config =
        ConversationConfig::CreateDefault(*engine->engine);
    if (!default_conversation_config.ok()) {
      ABSL_LOG(ERROR) << "Failed to create default conversation config: "
                      << default_conversation_config.status();
      return nullptr;
    }
    conversation =
        Conversation::Create(*engine->engine, *default_conversation_config);
  }

  if (!conversation.ok()) {
    ABSL_LOG(ERROR) << "Failed to create conversation: "
                    << conversation.status();
    return nullptr;
  }
  auto* c_conversation = new LiteRtLmConversation;
  c_conversation->conversation = *std::move(conversation);
  return c_conversation;
}

void litert_lm_conversation_delete(LiteRtLmConversation* conversation) {
  delete conversation;
}

LiteRtLmConversation* litert_lm_conversation_clone(
    LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return nullptr;
  }
  auto cloned = conversation->conversation->Clone();
  if (!cloned.ok()) {
    ABSL_LOG(ERROR) << "Failed to clone conversation: " << cloned.status();
    return nullptr;
  }
  auto c_conversation = std::make_unique<LiteRtLmConversation>();
  c_conversation->conversation = std::move(*cloned);
  return c_conversation.release();
}

LiteRtLmJsonResponse* litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args) {
  if (!conversation || !conversation->conversation) {
    return nullptr;
  }
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return nullptr;
  }

  OptionalArgs litert_lm_optional_args = CreateOptionalArgs(
      conversation->conversation.get(), extra_context, optional_args);

  auto response = conversation->conversation->SendMessage(
      json_message, std::move(litert_lm_optional_args));
  if (!response.ok()) {
    ABSL_LOG(ERROR) << "Failed to send message: " << response.status();
    return nullptr;
  }
  auto* c_response = new LiteRtLmJsonResponse;
  c_response->json_string = response->dump();
  return c_response;
}

void litert_lm_json_response_delete(LiteRtLmJsonResponse* response) {
  delete response;
}

const char* litert_lm_json_response_get_string(
    const LiteRtLmJsonResponse* response) {
  if (!response) {
    return nullptr;
  }
  return response->json_string.c_str();
}

int litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmStreamCallback callback, void* callback_data) {
  if (!conversation || !conversation->conversation) {
    return -1;
  }
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return -1;
  }

  litert::lm::OptionalArgs litert_lm_optional_args = CreateOptionalArgs(
      conversation->conversation.get(), extra_context, optional_args);

  absl::Status status = conversation->conversation->SendMessageAsync(
      json_message, CreateConversationCallback(callback, callback_data),
      std::move(litert_lm_optional_args));

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start message stream: " << status;
    return static_cast<int>(status.code());
  }
  return 0;
}

const char* litert_lm_conversation_render_message_to_string(
    LiteRtLmConversation* conversation, const char* message_json) {
  if (!conversation || !conversation->conversation || !message_json) {
    return nullptr;
  }
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return nullptr;
  }

  auto rendered = conversation->conversation->RenderMessageIntoString(
      json_message, litert::lm::OptionalArgs());
  if (!rendered.ok()) {
    ABSL_LOG(ERROR) << "Failed to render message: " << rendered.status();
    return nullptr;
  }
  conversation->last_rendered_message = std::move(*rendered);
  return conversation->last_rendered_message.c_str();
}

const char* litert_lm_conversation_render_preface_to_string(
    LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return nullptr;
  }
  auto rendered = conversation->conversation->RenderPrefaceIntoString(
      litert::lm::OptionalArgs());
  if (!rendered.ok()) {
    ABSL_LOG(ERROR) << "Failed to render preface: " << rendered.status();
    return nullptr;
  }
  conversation->last_rendered_preface = std::move(*rendered);
  return conversation->last_rendered_preface.c_str();
}

void litert_lm_conversation_cancel_process(LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return;
  }
  conversation->conversation->CancelProcess();
}

LiteRtLmBenchmarkInfo* litert_lm_conversation_get_benchmark_info(
    LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return nullptr;
  }
  auto benchmark_info = conversation->conversation->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ABSL_LOG(ERROR) << "Failed to get benchmark info: "
                    << benchmark_info.status();
    return nullptr;
  }
  return new LiteRtLmBenchmarkInfo{std::move(*benchmark_info)};
}

int litert_lm_conversation_get_token_count(LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return -1;
  }
  absl::StatusOr<int> token_count = conversation->conversation->GetTokenCount();
  if (!token_count.ok()) {
    ABSL_LOG(ERROR) << "Failed to get token count: " << token_count.status();
    return -1;
  }
  return *token_count;
}

}  // extern "C"
