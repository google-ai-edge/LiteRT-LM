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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// For Windows, __declspec( dllexport ) is required to export function in .dll.
// https://learn.microsoft.com/en-us/cpp/cpp/using-dllimport-and-dllexport-in-cpp-classes?view=msvc-170
//
// _WIN32 is defined as 1 when the compilation target is 32-bit ARM, 64-bit ARM,
// x86, x64, or ARM64EC. Otherwise, undefined.
// https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros
#if defined(_WIN32)
#define LITERT_LM_C_API_EXPORT __declspec(dllexport)
#else
#define LITERT_LM_C_API_EXPORT
#endif

// Opaque pointer for the LiteRT LM Engine.
typedef struct LiteRtLmEngine LiteRtLmEngine;

// Opaque pointer for the LiteRT LM Session.
typedef struct LiteRtLmSession LiteRtLmSession;

// Opaque pointer for the LiteRT LM Responses.
typedef struct LiteRtLmResponses LiteRtLmResponses;

// Opaque pointer for the LiteRT LM Engine Settings.
typedef struct LiteRtLmEngineSettings LiteRtLmEngineSettings;

// Opaque pointer for the LiteRT LM Benchmark Info.
typedef struct LiteRtLmBenchmarkInfo LiteRtLmBenchmarkInfo;

// Opaque pointer for the LiteRT LM Conversation.
typedef struct LiteRtLmConversation LiteRtLmConversation;

// Opaque pointer for a JSON response.
typedef struct LiteRtLmJsonResponse LiteRtLmJsonResponse;

// Opaque pointer for LiteRT LM Session Config.
typedef struct LiteRtLmSessionConfig LiteRtLmSessionConfig;

// Opaque pointer for LiteRT LM Conversation Config.
typedef struct LiteRtLmConversationConfig LiteRtLmConversationConfig;

// Represents the type of sampler.
typedef enum {
  kTypeUnspecified = 0,
  // Probabilistically pick among the top k tokens.
  kTopK = 1,
  // Probabilistically pick among the tokens such that the sum is greater
  // than or equal to p tokens after first performing top-k sampling.
  kTopP = 2,
  // Pick the token with maximum logit (i.e., argmax).
  kGreedy = 3,
} Type;

// Parameters for the sampler.
typedef struct {
  Type type;
  int32_t top_k;
  float top_p;
  float temperature;
  int32_t seed;
} LiteRtLmSamplerParams;

// Creates a LiteRT LM Session Config.
// The caller is responsible for destroying the config using
// `litert_lm_session_config_delete`.
// @return A pointer to the created config, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmSessionConfig* litert_lm_session_config_create();

// Sets the maximum number of output tokens per decode step for this session.
// @param config The config to modify.
// @param max_output_tokens The maximum number of output tokens.
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_set_max_output_tokens(
    LiteRtLmSessionConfig* config, int max_output_tokens);

// Sets the sampler parameters for this session config.
// @param config The config to modify.
// @param sampler_params The sampler parameters to use.
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_set_sampler_params(
    LiteRtLmSessionConfig* config, const LiteRtLmSamplerParams* sampler_params);

// Destroys a LiteRT LM Session Config.
// @param config The config to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_delete(LiteRtLmSessionConfig* config);

// Creates a LiteRT LM Conversation Config.
// The caller is responsible for destroying the config using
// `litert_lm_conversation_config_delete`.
// @param engine The engine to use.
// @param session_config The session config to use. If NULL, default
// session config will be used.
// @param system_message_json The system message in JSON format.
// @param tools_json The tools description in JSON array format.
// @param enable_constrained_decoding Whether to enable constrained decoding.
// @return A pointer to the created config, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmConversationConfig* litert_lm_conversation_config_create(
    LiteRtLmEngine* engine, const LiteRtLmSessionConfig* session_config,
    const char* system_message_json, const char* tools_json,
    bool enable_constrained_decoding);

// Destroys a LiteRT LM Conversation Config.
// @param config The config to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config);

// Sets the minimum log level for the LiteRT LM library.
// Log levels are: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL.
LITERT_LM_C_API_EXPORT
void litert_lm_set_min_log_level(int level);

// Represents the type of input data.
typedef enum {
  kInputText,
  kInputImage,
  kInputAudio,
  kInputAudioEnd,
} InputDataType;

// Represents a single piece of input data.
typedef struct {
  InputDataType type;
  // The data pointer. The interpretation depends on the `type`.
  // For kInputText, it's a UTF-8 string.
  // For kInputImage and kInputAudio, it's a pointer to the raw bytes.
  const void* data;
  // The size of the data in bytes.
  size_t size;
} InputData;

// Creates LiteRT LM Engine Settings. The caller is responsible for destroying
// the settings using `litert_lm_engine_settings_delete`.
//
// @param model_path The path to the model file.
// @param backend_str The backend to use (e.g., "cpu", "gpu").
// @param vision_backend_str The vision backend to use, or NULL if not set.
// @param audio_backend_str The audio backend to use, or NULL if not set.
// @return A pointer to the created settings, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmEngineSettings* litert_lm_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str);

// Destroys LiteRT LM Engine Settings.
//
// @param settings The settings to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_delete(LiteRtLmEngineSettings* settings);

// Sets the maximum number of tokens for the engine.
//
// @param settings The engine settings.
// @param max_num_tokens The maximum number of tokens.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_max_num_tokens(
    LiteRtLmEngineSettings* settings, int max_num_tokens);

// Sets the cache directory for the engine.
//
// @param settings The engine settings.
// @param cache_dir The cache directory.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_cache_dir(LiteRtLmEngineSettings* settings,
                                             const char* cache_dir);

// Sets the activation data type.
//
// @param settings The engine settings.
// @param activation_data_type_int The activation data type. See
// `ActivationDataType` in executor_settings_base.h for the possible values
// (e.g., 0 for F32, 1 for F16, 2 for I16, 3 for I8).
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_activation_data_type(
    LiteRtLmEngineSettings* settings, int activation_data_type_int);

// Enables benchmarking for the engine.
//
// @param settings The engine settings.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings);

// Creates a LiteRT LM Engine from the given settings. The caller is responsible
// for destroying the engine using `litert_lm_engine_delete`.
//
// @param settings The engine settings.
// @return A pointer to the created engine, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmEngine* litert_lm_engine_create(const LiteRtLmEngineSettings* settings);

// Destroys a LiteRT LM Engine.
//
// @param engine The engine to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_delete(LiteRtLmEngine* engine);

// Creates a LiteRT LM Session. The caller is responsible for destroying the
// session using `litert_lm_session_delete`.
//
// @param engine The engine to create the session from.
// @param config The session config of the session. If NULL, use the default
// session config.
// @return A pointer to the created session, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmSession* litert_lm_engine_create_session(LiteRtLmEngine* engine,
                                                 LiteRtLmSessionConfig* config);

// Destroys a LiteRT LM Session.
//
// @param session The session to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_session_delete(LiteRtLmSession* session);

// Generates content from the input prompt.
//
// @param session The session to use for generation.
// @param inputs An array of InputData structs representing the multimodal
//   input.
// @param num_inputs The number of InputData structs in the array.
// @return A pointer to the responses, or NULL on failure. The caller is
//   responsible for deleting the responses using `litert_lm_responses_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmResponses* litert_lm_session_generate_content(LiteRtLmSession* session,
                                                      const InputData* inputs,
                                                      size_t num_inputs);
// Destroys a LiteRT LM Responses object.
//
// @param responses The responses to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_responses_delete(LiteRtLmResponses* responses);

// Returns the number of response candidates.
//
// @param responses The responses object.
// @return The number of candidates.
LITERT_LM_C_API_EXPORT
int litert_lm_responses_get_num_candidates(const LiteRtLmResponses* responses);

// Returns the response text at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return The response text. The returned string is owned by the `responses`
//   object and is valid only for its lifetime. Returns NULL if index is out of
//   bounds.
LITERT_LM_C_API_EXPORT
const char* litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index);

// Retrieves the benchmark information from the session. The caller is
// responsible for destroying the benchmark info using
// `litert_lm_benchmark_info_delete`.
//
// @param session The session to get the benchmark info from.
// @return A pointer to the benchmark info, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmBenchmarkInfo* litert_lm_session_get_benchmark_info(
    LiteRtLmSession* session);

// Destroys a LiteRT LM Benchmark Info object.
//
// @param benchmark_info The benchmark info to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_benchmark_info_delete(LiteRtLmBenchmarkInfo* benchmark_info);

// Returns the time to the first token in seconds.
//
// Note that the first time to token doesn't include the time for
// initialization. It is the sum of the prefill time for the first turn and
// the time spent for decoding the first token.
//
// @param benchmark_info The benchmark info object.
// @return The time to the first token in seconds.
LITERT_LM_C_API_EXPORT
double litert_lm_benchmark_info_get_time_to_first_token(
    const LiteRtLmBenchmarkInfo* benchmark_info);

// Returns the number of prefill turns.
//
// @param benchmark_info The benchmark info object.
// @return The number of prefill turns.
LITERT_LM_C_API_EXPORT
int litert_lm_benchmark_info_get_num_prefill_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info);

// Returns the number of decode turns.
//
// @param benchmark_info The benchmark info object.
// @return The number of decode turns.
LITERT_LM_C_API_EXPORT
int litert_lm_benchmark_info_get_num_decode_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info);

// Returns the prefill token count at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the prefill turn.
// @return The prefill token count.
LITERT_LM_C_API_EXPORT
int litert_lm_benchmark_info_get_prefill_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index);


// Returns the decode token count at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the decode turn.
// @return The decode token count.
LITERT_LM_C_API_EXPORT
int litert_lm_benchmark_info_get_decode_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index);


// Returns the prefill tokens per second at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the prefill turn.
// @return The prefill tokens per second.
LITERT_LM_C_API_EXPORT
double litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index);

// Returns the decode tokens per second at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the decode turn.
// @return The decode tokens per second.
LITERT_LM_C_API_EXPORT
double litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index);

// Callback for streaming responses.
// `callback_data` is a pointer to user-defined data passed to the stream
// function. `chunk` is the piece of text from the stream. It's only valid for
// the duration of the call. `is_final` is true if this is the last chunk in the
// stream. `error_msg` is a null-terminated string with an error message, or
// NULL on success.
typedef void (*LiteRtLmStreamCallback)(void* callback_data, const char* chunk,
                                       bool is_final, const char* error_msg);

// Generates content from the input prompt and streams the response via a
// callback. This is a non-blocking call that will invoke the callback from a
// background thread for each chunk.
//
// @param session The session to use for generation.
// @param inputs An array of InputData structs representing the multimodal
//   input.
// @param num_inputs The number of InputData structs in the array.
// @param callback The callback function to receive response chunks.
// @param callback_data A pointer to user data that will be passed to the
// callback.
// @return 0 on success, non-zero on failure to start the stream.
LITERT_LM_C_API_EXPORT
int litert_lm_session_generate_content_stream(LiteRtLmSession* session,
                                              const InputData* inputs,
                                              size_t num_inputs,
                                              LiteRtLmStreamCallback callback,
                                              void* callback_data);

// Creates a LiteRT LM Conversation. The caller is responsible for destroying
// the conversation using `litert_lm_conversation_delete`.
//
// @param engine The engine to create the conversation from.
// @param config The conversation config to use. If NULL, the default config
//   will be used.
// @return A pointer to the created conversation, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmConversation* litert_lm_conversation_create(
    LiteRtLmEngine* engine, LiteRtLmConversationConfig* config);

// Destroys a LiteRT LM Conversation.
//
// @param conversation The conversation to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_delete(LiteRtLmConversation* conversation);

// Sends a message to the conversation and returns the response.
// This is a blocking call.
//
// @param conversation The conversation to use.
// @param message_json A JSON string representing the message to send.
// @return A pointer to the JSON response, or NULL on failure. The caller is
//   responsible for deleting the response using
//   `litert_lm_json_response_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmJsonResponse* litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json);

// Destroys a LiteRT LM Json Response object.
//
// @param response The response to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_json_response_delete(LiteRtLmJsonResponse* response);

// Returns the JSON response string from a response object.
//
// @param response The response object.
// @return The response JSON string. The returned string is owned by the
//   `response` object and is valid only for its lifetime. Returns NULL if
//   response is NULL.
LITERT_LM_C_API_EXPORT
const char* litert_lm_json_response_get_string(
    const LiteRtLmJsonResponse* response);

// Sends a message to the conversation and streams the response via a
// callback. This is a non-blocking call that will invoke the callback from a
// background thread for each chunk.
//
// @param conversation The conversation to use.
// @param message_json A JSON string representing the message to send.
// @param callback The callback function to receive response chunks.
// @param callback_data A pointer to user data that will be passed to the
// callback.
// @return 0 on success, non-zero on failure to start the stream.
LITERT_LM_C_API_EXPORT
int litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation, const char* message_json,
    LiteRtLmStreamCallback callback, void* callback_data);

// Cancels the ongoing inference process, for asynchronous inference.
//
// @param conversation The conversation to cancel the inference for.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_cancel_process(LiteRtLmConversation* conversation);

// Retrieves the benchmark information from the conversation. The caller is
// responsible for destroying the benchmark info using
// `litert_lm_benchmark_info_delete`.
//
// @param conversation The conversation to get the benchmark info from.
// @return A pointer to the benchmark info, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmBenchmarkInfo* litert_lm_conversation_get_benchmark_info(
    LiteRtLmConversation* conversation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_H_
