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
// Ensure symbols are exported when building the shared library with
// -fvisibility=hidden.
#define LITERT_LM_C_API_EXPORT __attribute__((visibility("default")))
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

// Opaque pointer for a detokenize result.
// Use `litert_lm_detokenize_result_delete` to free memory.
typedef struct LiteRtLmDetokenizeResult LiteRtLmDetokenizeResult;

// Opaque pointer for a tokenize result.
// Use `litert_lm_tokenize_result_delete` to free memory.
typedef struct LiteRtLmTokenizeResult LiteRtLmTokenizeResult;

// Represents the type of a TokenUnion.
typedef enum {
  kLiteRtLmTokenUnionTypeString = 0,
  kLiteRtLmTokenUnionTypeIds = 1,
} LiteRtLmTokenUnionType;

// Opaque pointer for LiteRT LM Token Union.
// Represents a single start or stop token, which could be either a string or a
// sequence of token ids.
// Use `litert_lm_token_union_delete` to free memory.
typedef struct LiteRtLmTokenUnion LiteRtLmTokenUnion;

// Opaque pointer for LiteRT LM Token Unions.
// Represents a collection of TokenUnion, typically used for model stop
// conditions.
// Use `litert_lm_token_unions_delete` to free memory.
typedef struct LiteRtLmTokenUnions LiteRtLmTokenUnions;

// Opaque pointer for LiteRT LM Session Config.
typedef struct LiteRtLmSessionConfig LiteRtLmSessionConfig;

// Opaque pointer for LiteRT LM Conversation Config.
typedef struct LiteRtLmConversationConfig LiteRtLmConversationConfig;

// Represents the type of sampler.
typedef enum {
  kLiteRtLmSamplerTypeUnspecified = 0,
  // Probabilistically pick among the top k tokens.
  kLiteRtLmSamplerTypeTopK = 1,
  // Probabilistically pick among the tokens such that the sum is greater
  // than or equal to p tokens after first performing top-k sampling.
  kLiteRtLmSamplerTypeTopP = 2,
  // Pick the token with maximum logit (i.e., argmax).
  kLiteRtLmSamplerTypeGreedy = 3,
} LiteRtLmSamplerType;

// Parameters for the sampler.
typedef struct {
  LiteRtLmSamplerType type;
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

// Sets whether to apply prompt template for this session.
// @param config The config to modify.
// @param apply_prompt_template Whether to apply prompt template.
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_set_apply_prompt_template(
    LiteRtLmSessionConfig* config, bool apply_prompt_template);

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
// @return A pointer to the created config, or NULL on failure.
LITERT_LM_C_API_EXPORT
LiteRtLmConversationConfig* litert_lm_conversation_config_create();

// Sets the session config for this conversation config.
// @param config The config to modify.
// @param session_config The session config to use.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_session_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmSessionConfig* session_config);

// Sets the system message for this conversation config.
// @param config The config to modify.
// @param system_message_json The system message in JSON format.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_system_message(
    LiteRtLmConversationConfig* config, const char* system_message_json);

// Sets the tools for this conversation config.
// @param config The config to modify.
// @param tools_json The tools description in JSON array format.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_tools(LiteRtLmConversationConfig* config,
                                             const char* tools_json);

// Sets the initial messages for this conversation config.
// @param config The config to modify.
// @param messages_json The initial messages in JSON array format.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_messages(
    LiteRtLmConversationConfig* config, const char* messages_json);

// Sets the extra context for the conversation preface.
// @param config The config to modify.
// @param extra_context_json A JSON string representing the extra context
// object.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_extra_context(
    LiteRtLmConversationConfig* config, const char* extra_context_json);

// Sets whether to enable constrained decoding for this conversation config.
// @param config The config to modify.
// @param enable_constrained_decoding Whether to enable constrained decoding.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_enable_constrained_decoding(
    LiteRtLmConversationConfig* config, bool enable_constrained_decoding);

// Sets whether to filter channel content from the KV cache.
// @param config The config to modify.
// @param filter_channel_content_from_kv_cache Whether to filter channel
// content.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
    LiteRtLmConversationConfig* config,
    bool filter_channel_content_from_kv_cache);

// Destroys a LiteRT LM Conversation Config.
// @param config The config to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config);

// Sets the minimum log level for the LiteRT LM library.
// Log levels are: 0=VERBOSE, 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR, 5=FATAL,
// 1000=SILENT.
LITERT_LM_C_API_EXPORT
void litert_lm_set_min_log_level(int level);

// Represents the type of input data.
typedef enum {
  kLiteRtLmInputDataTypeText,
  kLiteRtLmInputDataTypeImage,
  kLiteRtLmInputDataTypeImageEnd,
  kLiteRtLmInputDataTypeAudio,
  kLiteRtLmInputDataTypeAudioEnd,
} LiteRtLmInputDataType;

// Represents a single piece of input data.
typedef struct {
  LiteRtLmInputDataType type;
  // The data pointer. The interpretation depends on the `type`.
  // For kInputText, it's a UTF-8 string.
  // For kInputImage and kInputAudio, it's a pointer to the raw bytes.
  const void* data;
  // The size of the data in bytes.
  size_t size;
} LiteRtLmInputData;

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

// Sets whether the engine should load different sections of the litertlm file
// in parallel. Defaults to true.
//
// @param settings The engine settings.
// @param parallel_file_section_loading Whether to load in parallel.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_parallel_file_section_loading(
    LiteRtLmEngineSettings* settings, bool parallel_file_section_loading);

// Sets the maximum number of images for the engine.
//
// This is only used for the legacy implementation of the engine.
//
// @param settings The engine settings.
// @param max_num_images The maximum number of images.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_max_num_images(
    LiteRtLmEngineSettings* settings, int max_num_images);

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

// Sets the prefill chunk size for the engine. Only applicable for CPU backend
// with dynamic models.
//
// @param settings The engine settings.
// @param prefill_chunk_size The prefill chunk size.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_prefill_chunk_size(
    LiteRtLmEngineSettings* settings, int prefill_chunk_size);

// Enables benchmarking for the engine.
//
// @param settings The engine settings.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings);

// Sets the number of prefill tokens for benchmarking.
//
// @param settings The engine settings.
// @param num_prefill_tokens The number of prefill tokens.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_num_prefill_tokens(
    LiteRtLmEngineSettings* settings, int num_prefill_tokens);

// Sets the number of decode tokens for benchmarking.
//
// @param settings The engine settings.
// @param num_decode_tokens The number of decode tokens.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_num_decode_tokens(
    LiteRtLmEngineSettings* settings, int num_decode_tokens);

// Sets whether to enable speculative decoding.
//
// @param settings The engine settings.
// @param enable_speculative_decoding Whether to enable speculative decoding.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_enable_speculative_decoding(
    LiteRtLmEngineSettings* settings, bool enable_speculative_decoding);

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

// Cancels the current processing in the session.
//
// @param session The session to cancel processing on.
LITERT_LM_C_API_EXPORT
void litert_lm_session_cancel_process(LiteRtLmSession* session);

// Adds the input prompt/query to the model for starting the prefilling
// process. This is a blocking call and the function will return when the
// prefill process is done.
//
// @param session The session to use.
// @param inputs An array of InputData structs representing the multimodal
//   input.
// @param num_inputs The number of InputData structs in the array.
// @return 0 on success, non-zero on failure.
LITERT_LM_C_API_EXPORT
int litert_lm_session_run_prefill(LiteRtLmSession* session,
                                  const LiteRtLmInputData* inputs,
                                  size_t num_inputs);

// Starts the decoding process for the model to predict the response based
// on the input prompt/query added after using litert_lm_session_run_prefill.
// This is a blocking call and the function will return when the decoding
// process is done.
//
// @param session The session to use.
// @return A pointer to the responses, or NULL on failure. The caller is
//   responsible for deleting the responses using `litert_lm_responses_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmResponses* litert_lm_session_run_decode(LiteRtLmSession* session);

// Scores the target text after the prefill process is done.
//
// @param session The session to use.
// @param target_text An array of target text strings to score.
// @param num_targets The number of strings in the target_text array.
// @param store_token_lengths Whether to store the token lengths of the target
//   texts in the responses.
// @return A pointer to the responses, or NULL on failure. The caller is
//   responsible for deleting the responses using `litert_lm_responses_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmResponses* litert_lm_session_run_text_scoring(LiteRtLmSession* session,
                                                      const char** target_text,
                                                      size_t num_targets,
                                                      bool store_token_lengths);

// Generates content from the input prompt.
//
// @param session The session to use for generation.
// @param inputs An array of LiteRtLmInputData structs representing the
// multimodal
//   input.
// @param num_inputs The number of LiteRtLmInputData structs in the array.
// @return A pointer to the responses, or NULL on failure. The caller is
//   responsible for deleting the responses using `litert_lm_responses_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmResponses* litert_lm_session_generate_content(
    LiteRtLmSession* session, const LiteRtLmInputData* inputs,
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

// Returns whether the response contains a score at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return true if the score is available at the given index, false otherwise.
LITERT_LM_C_API_EXPORT
bool litert_lm_responses_has_score_at(const LiteRtLmResponses* responses,
                                      int index);

// Returns the score at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return The score. Returns 0.0f if index is out of bounds or no score is
//   present.
LITERT_LM_C_API_EXPORT
float litert_lm_responses_get_score_at(const LiteRtLmResponses* responses,
                                       int index);

// Returns whether the response contains a token length at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return true if the token length is available at the given index, false
//   otherwise.
LITERT_LM_C_API_EXPORT
bool litert_lm_responses_has_token_length_at(const LiteRtLmResponses* responses,
                                             int index);

// Returns the token length at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return The token length. Returns 0 if index is out of bounds or no token
//   length is present.
LITERT_LM_C_API_EXPORT
int litert_lm_responses_get_token_length_at(const LiteRtLmResponses* responses,
                                            int index);

// Returns whether the response contains token scores at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return true if token scores are available at the given index, false
// otherwise.
LITERT_LM_C_API_EXPORT
bool litert_lm_responses_has_token_scores_at(const LiteRtLmResponses* responses,
                                             int index);

// Returns the number of tokens for which scores are present at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return The number of token scores. Returns 0 if index is out of bounds or no
//   token scores are present.
LITERT_LM_C_API_EXPORT
int litert_lm_responses_get_num_token_scores_at(
    const LiteRtLmResponses* responses, int index);

// Returns the token scores at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return A pointer to the internal array of token scores. Returns NULL if
// index
//   is out of bounds or no token scores are present.
LITERT_LM_C_API_EXPORT
const float* litert_lm_responses_get_token_scores_at(
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

// Returns the total initialization time in seconds.
//
// @param benchmark_info The benchmark info object.
// @return The total initialization time in seconds.
LITERT_LM_C_API_EXPORT
double litert_lm_benchmark_info_get_total_init_time_in_second(
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

// Starts the decoding process for the model to predict the response based
// on the input prompt/query added after using litert_lm_session_run_prefill.
// This is a non-blocking call that will stream responses via a callback.
//
// @param session The session to use.
// @param callback The callback function to receive response chunks.
// @param callback_data A pointer to user data that will be passed to the
// callback.
// @return 0 on success, non-zero on failure.
LITERT_LM_C_API_EXPORT
int litert_lm_session_run_decode_async(LiteRtLmSession* session,
                                       LiteRtLmStreamCallback callback,
                                       void* callback_data);

// Generates content from the input prompt and streams the response via a
// callback. This is a non-blocking call that will invoke the callback from a
// background thread for each chunk.
//
// @param session The session to use for generation.
// @param inputs An array of LiteRtLmInputData structs representing the
// multimodal
//   input.
// @param num_inputs The number of LiteRtLmInputData structs in the array.
// @param callback The callback function to receive response chunks.
// @param callback_data A pointer to user data that will be passed to the
// callback.
// @return 0 on success, non-zero on failure to start the stream.
LITERT_LM_C_API_EXPORT
int litert_lm_session_generate_content_stream(LiteRtLmSession* session,
                                              const LiteRtLmInputData* inputs,
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
// @param extra_context A JSON string representing the extra context to use.
// @return A pointer to the JSON response, or NULL on failure. The caller is
//   responsible for deleting the response using
//   `litert_lm_json_response_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmJsonResponse* litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context);

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
// @param extra_context A JSON string representing the extra context to use.
// @param callback The callback function to receive response chunks.
// @param callback_data A pointer to user data that will be passed to the
// callback.
// @return 0 on success, non-zero on failure to start the stream.
LITERT_LM_C_API_EXPORT
int litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context, LiteRtLmStreamCallback callback,
    void* callback_data);

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
LITERT_LM_C_API_EXPORT
const char* litert_lm_conversation_render_message_to_string(
    LiteRtLmConversation* conversation, const char* message_json);

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

// Tokenizes text using the engine's tokenizer.
//
// @param engine The engine instance.
// @param text The UTF-8 string to tokenize.
// @return A pointer to the tokenize result, or NULL on failure.
//   The caller is responsible for deleting the result using
//   `litert_lm_tokenize_result_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenizeResult* litert_lm_engine_tokenize(LiteRtLmEngine* engine,
                                                  const char* text);

// Destroys a LiteRT LM Tokenize Result.
//
// @param result The tokenize result to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_tokenize_result_delete(LiteRtLmTokenizeResult* result);

// Returns the token ids from a tokenize result.
//
// @param result The tokenize result.
// @return A pointer to the internal array of token ids. The returned pointer
//   is valid only for the lifetime of the `result` object.
LITERT_LM_C_API_EXPORT
const int* litert_lm_tokenize_result_get_tokens(
    const LiteRtLmTokenizeResult* result);

// Returns the number of token ids from a tokenize result.
//
// @param result The tokenize result.
// @return The number of token ids.
LITERT_LM_C_API_EXPORT
size_t litert_lm_tokenize_result_get_num_tokens(
    const LiteRtLmTokenizeResult* result);

// Detokenizes token ids using the engine's tokenizer.
//
// @param engine The engine instance.
// @param tokens An array of token ids to detokenize.
// @param num_tokens The number of token ids in the array.
// @return A pointer to the detokenize result, or NULL on failure.
//   The caller is responsible for deleting the result using
//   `litert_lm_detokenize_result_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmDetokenizeResult* litert_lm_engine_detokenize(LiteRtLmEngine* engine,
                                                      const int* tokens,
                                                      size_t num_tokens);

// Destroys a LiteRT LM Detokenize Result.
//
// @param result The detokenize result to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_detokenize_result_delete(LiteRtLmDetokenizeResult* result);

// Returns the string from a detokenize result.
//
// @param result The detokenize result.
// @return The detokenized UTF-8 string. The returned string is owned by the
//   `result` object and is valid only for its lifetime.
LITERT_LM_C_API_EXPORT
const char* litert_lm_detokenize_result_get_string(
    const LiteRtLmDetokenizeResult* result);

// Destroys a LiteRT LM Token Union.
//
// @param token_union The token union to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_token_union_delete(LiteRtLmTokenUnion* token_union);

// Returns the type of the token union.
//
// @param token_union The token union.
// @return The type of the token union.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenUnionType litert_lm_token_union_get_type(
    const LiteRtLmTokenUnion* token_union);

// Returns the string value from a token union.
//
// @param token_union The token union.
// @return The string value, or NULL if the type is not
//   kLiteRtLmTokenUnionTypeString. The returned string is owned by the
//   `token_union` object and is valid only for its lifetime.
LITERT_LM_C_API_EXPORT
const char* litert_lm_token_union_get_string(
    const LiteRtLmTokenUnion* token_union);

// Returns the token ids from a token union.
//
// @param token_union The token union.
// @param out_tokens A pointer to receive the internal array of token ids.
//   The received pointer is valid only for the lifetime of the `token_union`
//   object.
// @param out_num_tokens A pointer to receive the number of token ids.
// @return 0 on success, non-zero if the type is not kLiteRtLmTokenUnionTypeIds.
LITERT_LM_C_API_EXPORT
int litert_lm_token_union_get_ids(const LiteRtLmTokenUnion* token_union,
                                  const int** out_tokens,
                                  size_t* out_num_tokens);

// Destroys a LiteRT LM Token Unions object.
//
// @param tokens The token unions object to destroy.
LITERT_LM_C_API_EXPORT
void litert_lm_token_unions_delete(LiteRtLmTokenUnions* tokens);

// Returns the number of token unions in the collection.
//
// @param tokens The token unions object.
// @return The number of token unions.
LITERT_LM_C_API_EXPORT
size_t litert_lm_token_unions_get_num_tokens(const LiteRtLmTokenUnions* tokens);

// Returns the token union at a given index from a collection.
//
// @param tokens The token unions collection.
// @param index The index of the token union.
// @return A pointer to the token union at the given index, or NULL if the index
//   is out of bounds. The caller is responsible for deleting the result using
//   `litert_lm_token_union_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenUnion* litert_lm_token_unions_get_token_at(
    const LiteRtLmTokenUnions* tokens, size_t index);

// Returns the configured start token (BOS), if any.
//
// @param engine The engine instance.
// @return A pointer to the start token, or NULL if none configured. The caller
//   is responsible for deleting the result using
//   `litert_lm_token_union_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenUnion* litert_lm_engine_get_start_token(LiteRtLmEngine* engine);

// Returns the configured stop tokens (EOS).
//
// @param engine The engine instance.
// @return A pointer to the stop tokens collection, or NULL if none configured.
//   The caller is responsible for deleting the result using
//   `litert_lm_token_unions_delete`.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenUnions* litert_lm_engine_get_stop_tokens(LiteRtLmEngine* engine);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_H_
