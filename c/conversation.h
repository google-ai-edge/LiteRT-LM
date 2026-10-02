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
#include "engine.h"          // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/engine.h"
#include "c/error_reporter.h"
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
//
// @param out_config On success, receives the created config, owned by the
//   caller; release with `litert_lm_conversation_config_delete`. Set to NULL
//   on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the config is returned
// through out_config.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_create(
    LiteRtLmConversationConfig** out_config);

// Sets the session config for this conversation config.
// @param config The config to modify.
// @param session_config The session config to use.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` or `session_config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_session_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmSessionConfig* session_config);

// Sets the system message for this conversation config.
// @param config The config to modify.
// @param system_message_json The system message in JSON format.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` or `system_message_json` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_system_message(
    LiteRtLmConversationConfig* config, const char* system_message_json);

// Sets the tools for this conversation config.
// @param config The config to modify.
// @param tools_json The tools description in JSON array format.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` or `tools_json` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_tools(
    LiteRtLmConversationConfig* config, const char* tools_json);

// Sets the initial messages for this conversation config.
// @param config The config to modify.
// @param messages_json The initial messages in JSON array format.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` or `messages_json` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_messages(
    LiteRtLmConversationConfig* config, const char* messages_json);

// Sets the extra context for the conversation preface.
// @param config The config to modify.
// @param extra_context_json A JSON string representing the extra context
// object.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` or `extra_context_json` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_extra_context(
    LiteRtLmConversationConfig* config, const char* extra_context_json);

// Sets the prompt template for this conversation config.
// @param config The config to modify.
// @param prompt_template The prompt template string (e.g. Jinja template). If
// not set, use the default provided by the model or the engine.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` or `prompt_template` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_prompt_template(
    LiteRtLmConversationConfig* config, const char* prompt_template);

// Sets whether to enable constrained decoding for this conversation config.
// @param config The config to modify.
// @param enable_constrained_decoding Whether to enable constrained decoding.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_config_set_enable_constrained_decoding(
    LiteRtLmConversationConfig* config, bool enable_constrained_decoding);

// Sets the constraint provider type for this conversation config.
// @param config The config to modify.
// @param provider_type The constraint provider type to use, or NULL to unset.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL or `*provider_type` is not a declared
//   LiteRtLmConstraintProviderType value.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_constraint_provider(
    LiteRtLmConversationConfig* config,
    const LiteRtLmConstraintProviderType* provider_type);

// Sets whether to filter channel content from the KV cache.
// @param config The config to modify.
// @param filter_channel_content_from_kv_cache Whether to filter channel
// content.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
    LiteRtLmConversationConfig* config,
    bool filter_channel_content_from_kv_cache);

// Sets whether to stream tool call tokens.
// @param config The config to modify.
// @param stream_tool_calls Whether to stream tool call tokens.
// @param channel_name The channel name to use for tool call tokens.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_stream_tool_calls(
    LiteRtLmConversationConfig* config, bool stream_tool_calls,
    const char* channel_name);

// Creates a default LiteRT LM Thinking Config (enabled with infinite budget
// -1).
//
// @param out_config On success, receives the created config, owned by the
//   caller; release with `litert_lm_thinking_config_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the config is returned
// through out_config.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_thinking_config_create(
    LiteRtLmThinkingConfig** out_config);

// Destroys a LiteRT LM Thinking Config.
// @param config The config to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_thinking_config_delete(LiteRtLmThinkingConfig* config);

// Sets whether thinking/reasoning generation is enabled.
// @param config The config to modify.
// @param enable_thinking Whether thinking is enabled.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_thinking_config_set_enable_thinking(
    LiteRtLmThinkingConfig* config, bool enable_thinking);

// Sets the thinking token budget.
// @param config The config to modify.
// @param thinking_token_budget Budget for token-by-token reasoning generation
// (-1 for infinite).
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_thinking_config_set_thinking_token_budget(
    LiteRtLmThinkingConfig* config, int thinking_token_budget);

// Sets the thinking config for this conversation config.
// @param config The config to modify.
// @param thinking_config The thinking config to set. If NULL, clears any
// previously set thinking config.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_thinking_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmThinkingConfig* thinking_config);

// Destroys a LiteRT LM Conversation Config.
// @param config The config to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config);

// Creates a LiteRT LM Conversation Optional Args.
//
// @param out_optional_args On success, receives the created optional args,
//   owned by the caller; release with
//   `litert_lm_conversation_optional_args_delete`. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_optional_args` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the optional args are
// returned through out_optional_args.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_create(
    LiteRtLmConversationOptionalArgs** out_optional_args);

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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `optional_args` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_repetition_penalty_config(
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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `optional_args` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `optional_args` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_suppress_tokens_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmSuppressTokensConfig* suppress_tokens_config);

// Sets the visual token budget for the conversation optional args.
// @param optional_args The optional args to modify.
// @param visual_token_budget The visual token budget.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `optional_args` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_set_visual_token_budget(
    LiteRtLmConversationOptionalArgs* optional_args, int visual_token_budget);

// Sets the maximum number of output tokens for the conversation optional args.
// For thinking models, both thinking (reasoning) tokens and the final response
// tokens count towards this limit.
// @param optional_args The optional args to modify.
// @param max_output_tokens The maximum number of tokens to generate (including
// thinking tokens).
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `optional_args` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_set_max_output_tokens(
    LiteRtLmConversationOptionalArgs* optional_args, int max_output_tokens);

// Sets the thinking config for the conversation optional args.
// @param optional_args The optional args to modify.
// @param thinking_config The thinking config to set. If NULL, clears any
// previously set thinking config.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `optional_args` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_set_thinking_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmThinkingConfig* thinking_config);

// Sets the constraint for the conversation optional args.
// @param optional_args The optional args to modify.
// @param constraint_type The type of constraint.
// @param constraint_string The constraint pattern/schema/grammar string.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `optional_args` is NULL or `constraint_type` is not a declared
//   LiteRtLmConstraintType value.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_set_constraint(
    LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmConstraintType constraint_type, const char* constraint_string);

// Creates a LiteRT LM Conversation.
//
// @param engine The engine to create the conversation from.
// @param config The conversation config to use. If NULL, the default config
//   will be used.
// @param out_conversation On success, receives the created conversation, owned
//   by the caller; release with `litert_lm_conversation_delete`. Set to NULL
//   on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `engine` or `out_conversation` is NULL; otherwise the code of the error
//   that prevented the conversation from being created.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the conversation is
// returned through out_conversation.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_create(
    LiteRtLmEngine* engine, LiteRtLmConversationConfig* config,
    LiteRtLmConversation** out_conversation);

// Destroys a LiteRT LM Conversation.
//
// @param conversation The conversation to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_delete(LiteRtLmConversation* conversation);

// Clones a LiteRT LM Conversation, duplicating its prefilled state.
//
// @param conversation The conversation to clone.
// @param out_conversation On success, receives the cloned conversation, owned
//   by the caller; release with `litert_lm_conversation_delete`. Set to NULL
//   on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `conversation` or `out_conversation` is NULL; otherwise the code of the
//   error that prevented the clone.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the clone is returned
// through out_conversation.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_clone(
    LiteRtLmConversation* conversation,
    LiteRtLmConversation** out_conversation);

// Sends a message to the conversation and returns the response.
// This is a blocking call.
//
// @param conversation The conversation to use.
// @param message_json A JSON string representing the message to send.
// @param extra_context A JSON string representing the extra context to use.
// @param optional_args A pointer to the optional arguments to use.
// @param out_response On success, receives the JSON response, owned by the
//   caller; release with `litert_lm_json_response_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `conversation`, `message_json` or `out_response` is NULL or
//   `message_json` is not valid JSON; otherwise the code of the generation
//   error.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the response is returned
// through out_response.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmJsonResponse** out_response);

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
// @param out_json On success, receives the response JSON string. The string
//   is owned by the `response` object and is valid only for its lifetime. Set
//   to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `response` or `out_json` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the string is returned
// through out_json.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_json_response_get_string(
    const LiteRtLmJsonResponse* response, const char** out_json);

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
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `conversation`, `message_json`, or `callback` is NULL, or `message_json`
//   is not valid JSON; otherwise the code of the error that prevented the
//   stream from starting. Errors that occur after the stream has started are
//   reported through the callback.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns LiteRtLmStatusCode and failures return a
// canonical LiteRtLmStatusCode instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_send_message_stream(
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
// @param out_text On success, receives the rendered string. The string is
//   owned by the `conversation` object and is valid until the next call to
//   this function or until the conversation is deleted. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `conversation`, `message_json` or `out_text` is NULL or `message_json` is
//   not valid JSON; otherwise the code of the rendering error.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the rendered string is
// returned through out_text.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_render_message_to_string(
    LiteRtLmConversation* conversation, const char* message_json,
    const char** out_text);

// Renders the preface into a string according to the template.
//
// @param conversation The conversation instance.
// @param out_text On success, receives the rendered string. The string is
//   owned by the `conversation` object and is valid until the next call to
//   this function or until the conversation is deleted. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `conversation` or `out_text` is NULL; otherwise the code of the rendering
//   error.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the rendered string is
// returned through out_text.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_render_preface_to_string(
    LiteRtLmConversation* conversation, const char** out_text);

// Cancels the ongoing inference process, for asynchronous inference.
//
// @param conversation The conversation to cancel the inference for.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `conversation` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_cancel_process(
    LiteRtLmConversation* conversation);

// Triggers execution of, and waits for, all pending tasks in the conversation
// session to complete.
//
// This must be called after `litert_lm_conversation_send_message_stream` when
// the engine was created with single threaded execution enabled (see
// `litert_lm_engine_settings_set_single_threaded_execution`): in that mode the
// queued work only makes progress while a caller drives it, so the streaming
// callback is invoked from within this call.
//
// @param conversation The conversation to wait for.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns LiteRtLmStatusCode instead of int.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_wait_until_done(
    LiteRtLmConversation* conversation);

// Retrieves the benchmark information from the conversation.
//
// @param conversation The conversation to get the benchmark info from.
// @param out_benchmark_info On success, receives the benchmark info, owned by
//   the caller; release with `litert_lm_benchmark_info_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `conversation` or `out_benchmark_info` is NULL; otherwise the code of the
//   runtime error (e.g. if benchmarking is not enabled).
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the benchmark info is
// returned through out_benchmark_info.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_get_benchmark_info(
    LiteRtLmConversation* conversation,
    LiteRtLmBenchmarkInfo** out_benchmark_info);

// Gets the number of tokens in the conversation KV Cache (prefill + decode).
//
// @param conversation The conversation instance.
// @param out_count On success, receives the number of tokens; not written on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `conversation` or `out_count` is NULL; otherwise the code of the runtime
//   error.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of a negative value
// on failure; the count is returned through out_count.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_get_token_count(
    LiteRtLmConversation* conversation, int* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_H_
