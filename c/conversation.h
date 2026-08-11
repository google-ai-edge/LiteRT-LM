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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__)
#include "engine.h"  // NOLINT
#else
#include "c/engine.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer for the LiteRT LM Conversation.
//
// Added in version 0.1.0.
typedef struct LiteRtLmConversation LiteRtLmConversation;

// Opaque pointer for the LiteRT LM Conversation Optional Args.
//
// Added in version 0.1.0.
typedef struct LiteRtLmConversationOptionalArgs
    LiteRtLmConversationOptionalArgs;

// Opaque pointer for a JSON response.
//
// Added in version 0.1.0.
typedef struct LiteRtLmJsonResponse LiteRtLmJsonResponse;

// Opaque pointer for LiteRT LM Conversation Config.
//
// Added in version 0.1.0.
typedef struct LiteRtLmConversationConfig LiteRtLmConversationConfig;

// Opaque pointer for the LiteRT LM Thinking Config.
//
// Added in version 0.1.0.
typedef struct LiteRtLmThinkingConfig LiteRtLmThinkingConfig;

// Represents the type of constraint for constrained decoding.
//
// Added in version 0.1.0.
typedef enum {
  kLiteRtLmConstraintTypeNone = 0,
  kLiteRtLmConstraintTypeRegex = 1,
  kLiteRtLmConstraintTypeJsonSchema = 2,
} LiteRtLmConstraintType;

// Represents the type of constraint provider.
//
// Added in version 0.1.0.
typedef enum {
  kLiteRtLmConstraintProviderTypeLlGuidance = 1,
} LiteRtLmConstraintProviderType;

// Creates a LiteRT LM Conversation Config.
// The caller is responsible for destroying the config using
// `litert_lm_conversation_config_delete`.
// @return A pointer to the created config, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmConversationConfig* litert_lm_conversation_config_create();

// Sets the session config for this conversation config.
// @param config The config to modify.
// @param session_config The session config to use.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_session_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmSessionConfig* session_config);

// Sets the system message for this conversation config.
// @param config The config to modify.
// @param system_message_json The system message in JSON format.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_system_message(
    LiteRtLmConversationConfig* config, const char* system_message_json);

// Sets the tools for this conversation config.
// @param config The config to modify.
// @param tools_json The tools description in JSON array format.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_tools(LiteRtLmConversationConfig* config,
                                             const char* tools_json);

// Sets the initial messages for this conversation config.
// @param config The config to modify.
// @param messages_json The initial messages in JSON array format.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_messages(
    LiteRtLmConversationConfig* config, const char* messages_json);

// Sets the extra context for the conversation preface.
// @param config The config to modify.
// @param extra_context_json A JSON string representing the extra context
// object.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_extra_context(
    LiteRtLmConversationConfig* config, const char* extra_context_json);

// Sets the prompt template for this conversation config.
// @param config The config to modify.
// @param prompt_template The prompt template string (e.g. Jinja template). If
// not set, use the default provided by the model or the engine.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_prompt_template(
    LiteRtLmConversationConfig* config, const char* prompt_template);

// Sets whether to enable constrained decoding for this conversation config.
// @param config The config to modify.
// @param enable_constrained_decoding Whether to enable constrained decoding.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_enable_constrained_decoding(
    LiteRtLmConversationConfig* config, bool enable_constrained_decoding);

// Sets the constraint provider type for this conversation config.
// @param config The config to modify.
// @param provider_type The constraint provider type to use, or NULL to unset.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_constraint_provider(
    LiteRtLmConversationConfig* config,
    const LiteRtLmConstraintProviderType* provider_type);

// Sets whether to filter channel content from the KV cache.
// @param config The config to modify.
// @param filter_channel_content_from_kv_cache Whether to filter channel
// content.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
    LiteRtLmConversationConfig* config,
    bool filter_channel_content_from_kv_cache);

// Sets whether to stream tool call tokens.
// @param config The config to modify.
// @param stream_tool_calls Whether to stream tool call tokens.
// @param channel_name The channel name to use for tool call tokens.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_stream_tool_calls(
    LiteRtLmConversationConfig* config, bool stream_tool_calls,
    const char* channel_name);

// Creates a default LiteRT LM Thinking Config (enabled with infinite budget
// -1). The caller is responsible for destroying the config using
// `litert_lm_thinking_config_delete`.
// @return A pointer to the created config, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmThinkingConfig* litert_lm_thinking_config_create();

// Destroys a LiteRT LM Thinking Config.
// @param config The config to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_thinking_config_delete(LiteRtLmThinkingConfig* config);

// Sets whether thinking/reasoning generation is enabled.
// @param config The config to modify.
// @param enable_thinking Whether thinking is enabled.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_thinking_config_set_enable_thinking(
    LiteRtLmThinkingConfig* config, bool enable_thinking);

// Sets the thinking token budget.
// @param config The config to modify.
// @param thinking_token_budget Budget for token-by-token reasoning generation
// (-1 for infinite).
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_thinking_config_set_thinking_token_budget(
    LiteRtLmThinkingConfig* config, int thinking_token_budget);

// Sets the thinking config for this conversation config.
// @param config The config to modify.
// @param thinking_config The thinking config to set. If NULL, clears any
// previously set thinking config.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_thinking_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmThinkingConfig* thinking_config);

// Destroys a LiteRT LM Conversation Config.
// @param config The config to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config);

// Creates a LiteRT LM Conversation Optional Args. The caller is responsible
// for destroying the optional args using
// `litert_lm_conversation_optional_args_delete`.
// @return A pointer to the created optional args, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmConversationOptionalArgs* litert_lm_conversation_optional_args_create();

// Destroys a LiteRT LM Conversation Optional Args.
// @param optional_args The optional args to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_delete(
    LiteRtLmConversationOptionalArgs* optional_args);

// Sets the repetition penalty configuration for the per-turn conversation
// optional arguments (`OptionalArgs`).
//
// The configured penalties (`repetition_penalty`, `presence_penalty`,
// `frequency_penalty`, `window_size`) apply exclusively to the output sequence
// generated during the current `send_message` or `send_message_async` call.
//
// @param optional_args The optional arguments structure (`OptionalArgs`) to
// modify.
// @param repetition_penalty_config The repetition penalty configuration struct
// (`LiteRtLmRepetitionPenaltyConfig`) created via
// `litert_lm_repetition_penalty_config_create`. The contents are deep-copied
// when set. If NULL, clears any previously set repetition penalty config so no
// penalties apply.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_set_repetition_penalty_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmRepetitionPenaltyConfig* repetition_penalty_config);

// Sets the no repeat ngram configuration for the per-turn conversation
// optional arguments (`OptionalArgs`).
//
// The configured parameters (`no_repeat_ngram_size`, `window_size`) apply
// exclusively to the output sequence generated during the current
// `send_message` or `send_message_async` call.
//
// @param optional_args The optional arguments structure (`OptionalArgs`) to
// modify.
// @param no_repeat_ngram_config The no repeat ngram configuration struct
// (`LiteRtLmNoRepeatNgramConfig`) created via
// `litert_lm_no_repeat_ngram_config_create`. The contents are deep-copied when
// set. If NULL, clears any previously set no repeat ngram config so no
// repeat ngram banning applies.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmNoRepeatNgramConfig* no_repeat_ngram_config);

// Sets the suppress tokens configuration for the per-turn conversation
// optional arguments (`OptionalArgs`).
//
// The configured list of suppressed tokens applies exclusively to the output
// sequence generated during the current `send_message` or
// `send_message_async` call.
//
// @param optional_args The optional arguments structure (`OptionalArgs`) to
// modify.
// @param suppress_tokens_config The suppress tokens configuration struct
// (`LiteRtLmSuppressTokensConfig`) created via
// `litert_lm_suppress_tokens_config_create`. The contents are deep-copied when
// set. If NULL or if the inner token set is disabled/empty, clears any
// previously set suppress tokens config so no token suppression applies.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_set_suppress_tokens_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmSuppressTokensConfig* suppress_tokens_config);

// Sets the visual token budget for the conversation optional args.
// @param optional_args The optional args to modify.
// @param visual_token_budget The visual token budget.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_set_visual_token_budget(
    LiteRtLmConversationOptionalArgs* optional_args, int visual_token_budget);

// Sets the maximum number of output tokens for the conversation optional args.
// For thinking models, both thinking (reasoning) tokens and the final response
// tokens count towards this limit.
// @param optional_args The optional args to modify.
// @param max_output_tokens The maximum number of tokens to generate (including
// thinking tokens).
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_set_max_output_tokens(
    LiteRtLmConversationOptionalArgs* optional_args, int max_output_tokens);

// Sets the thinking config for the conversation optional args.
// @param optional_args The optional args to modify.
// @param thinking_config The thinking config to set. If NULL, clears any
// previously set thinking config.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_set_thinking_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmThinkingConfig* thinking_config);

// Sets the constraint for the conversation optional args.
// @param optional_args The optional args to modify.
// @param constraint_type The type of constraint.
// @param constraint_string The constraint pattern/schema/grammar string.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_set_constraint(
    LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmConstraintType constraint_type, const char* constraint_string);

// Creates a LiteRT LM Conversation. The caller is responsible for destroying
// the conversation using `litert_lm_conversation_delete`.
//
// @param engine The engine to create the conversation from.
// @param config The conversation config to use. If NULL, the default config
//   will be used.
// @return A pointer to the created conversation, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmConversation* litert_lm_conversation_create(
    LiteRtLmEngine* engine, LiteRtLmConversationConfig* config);

// Destroys a LiteRT LM Conversation.
//
// @param conversation The conversation to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_delete(LiteRtLmConversation* conversation);

// Clones a LiteRT LM Conversation, duplicating its prefilled state.
// The caller is responsible for destroying the cloned conversation using
// `litert_lm_conversation_delete`.
//
// @param conversation The conversation to clone.
// @return A pointer to the cloned conversation, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmConversation* litert_lm_conversation_clone(
    LiteRtLmConversation* conversation);

// Sends a message to the conversation and returns the response.
// This is a blocking call.
//
// @param conversation The conversation to use.
// @param message_json A JSON string representing the message to send.
// @param extra_context A JSON string representing the extra context to use.
// @param optional_args A pointer to the optional arguments to use.
// @return A pointer to the JSON response, or NULL on failure. The caller is
//   responsible for deleting the response using
//   `litert_lm_json_response_delete`.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmJsonResponse* litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args);

// Destroys a LiteRT LM Json Response object.
//
// @param response The response to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_json_response_delete(LiteRtLmJsonResponse* response);

// Returns the JSON response string from a response object.
//
// @param response The response object.
// @return The response JSON string. The returned string is owned by the
//   `response` object and is valid only for its lifetime. Returns NULL if
//   response is NULL.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_json_response_get_string(
    const LiteRtLmJsonResponse* response);

// Sends a message to the conversation and streams the response via a
// callback. This is a non-blocking call that will invoke the callback from a
// background thread for each chunk.
//
// @param conversation The conversation to use.
// @param message_json A JSON string representing the message to send.
// @param extra_context A JSON string representing the extra context to use.
// @param optional_args A pointer to the optional arguments to use.
// @param callback The callback function to receive response chunks.
// @param callback_data A pointer to user data that will be passed to the
// callback.
// @return 0 on success, non-zero on failure to start the stream.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmStreamCallback callback, void* callback_data);

// Renders the message into a string according to the template.
//
// This function does not need to be called for actual message sending, as the
// `litert_lm_conversation_send_message` and
// `litert_lm_conversation_send_message_stream` functions will handle rendering
// internally.
//
// @param conversation The conversation instance.
// @param message_json A JSON string representing the message to render.
// @return A pointer to the rendered string, or NULL on failure. The returned
//   string is owned by the `conversation` object and is valid until the next
//   call to this function or until the conversation is deleted.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_conversation_render_message_to_string(
    LiteRtLmConversation* conversation, const char* message_json);

// Renders the preface into a string according to the template.
//
// @param conversation The conversation instance.
// @return A pointer to the rendered string, or NULL on failure. The returned
//   string is owned by the `conversation` object and is valid until the next
//   call to this function or until the conversation is deleted.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_conversation_render_preface_to_string(
    LiteRtLmConversation* conversation);

// Cancels the ongoing inference process, for asynchronous inference.
//
// @param conversation The conversation to cancel the inference for.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_cancel_process(LiteRtLmConversation* conversation);

// Retrieves the benchmark information from the conversation. The caller is
// responsible for destroying the benchmark info using
// `litert_lm_benchmark_info_delete`.
//
// @param conversation The conversation to get the benchmark info from.
// @return A pointer to the benchmark info, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmBenchmarkInfo* litert_lm_conversation_get_benchmark_info(
    LiteRtLmConversation* conversation);

// Gets the number of tokens in the conversation KV Cache (prefill + decode).
// Returns the number of tokens, or a negative value on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_conversation_get_token_count(LiteRtLmConversation* conversation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_H_
