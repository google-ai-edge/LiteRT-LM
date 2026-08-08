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
//
// Added in version 0.1.0.
typedef struct LiteRtLmEngine LiteRtLmEngine;

// Opaque pointer for the LiteRT LM Session.
//
// Added in version 0.1.0.
typedef struct LiteRtLmSession LiteRtLmSession;

// Opaque pointer for the LiteRT LM Responses.
//
// Added in version 0.1.0.
typedef struct LiteRtLmResponses LiteRtLmResponses;

// Opaque pointer for the LiteRT LM Engine Settings.
//
// Added in version 0.1.0.
typedef struct LiteRtLmEngineSettings LiteRtLmEngineSettings;

// Opaque pointer for the LiteRT LM Benchmark Info.
//
// Added in version 0.1.0.
typedef struct LiteRtLmBenchmarkInfo LiteRtLmBenchmarkInfo;

// Opaque pointer for the LiteRT LM Repetition Penalty Config.
//
// Added in version 0.1.0.
typedef struct LiteRtLmRepetitionPenaltyConfig LiteRtLmRepetitionPenaltyConfig;

// Opaque pointer for the LiteRT LM No Repeat Ngram Config.
//
// Added in version 0.1.0.
typedef struct LiteRtLmNoRepeatNgramConfig LiteRtLmNoRepeatNgramConfig;

// Opaque pointer for the LiteRT LM Suppress Tokens Config.
//
// Added in version 0.1.0.
typedef struct LiteRtLmSuppressTokensConfig LiteRtLmSuppressTokensConfig;

// Opaque pointer for a detokenize result.
// Use `litert_lm_detokenize_result_delete` to free memory.
//
// Added in version 0.1.0.
typedef struct LiteRtLmDetokenizeResult LiteRtLmDetokenizeResult;

// Opaque pointer for a tokenize result.
// Use `litert_lm_tokenize_result_delete` to free memory.
//
// Added in version 0.1.0.
typedef struct LiteRtLmTokenizeResult LiteRtLmTokenizeResult;

// Represents the type of a TokenUnion.
//
// Added in version 0.1.0.
typedef enum {
  kLiteRtLmTokenUnionTypeString = 0,
  kLiteRtLmTokenUnionTypeIds = 1,
} LiteRtLmTokenUnionType;

// Opaque pointer for LiteRT LM Token Union.
// Represents a single start or stop token, which could be either a string or a
// sequence of token ids.
// Use `litert_lm_token_union_delete` to free memory.
//
// Added in version 0.1.0.
typedef struct LiteRtLmTokenUnion LiteRtLmTokenUnion;

// Opaque pointer for LiteRT LM Token Unions.
// Represents a collection of TokenUnion, typically used for model stop
// conditions.
// Use `litert_lm_token_unions_delete` to free memory.
//
// Added in version 0.1.0.
typedef struct LiteRtLmTokenUnions LiteRtLmTokenUnions;

// Opaque pointer for LiteRT LM Input Data.
// Use `litert_lm_input_data_delete` to free memory.
//
// Added in version 0.1.0.
typedef struct LiteRtLmInputData LiteRtLmInputData;

// Opaque pointer for LiteRT LM Session Config.
//
// Added in version 0.1.0.
typedef struct LiteRtLmSessionConfig LiteRtLmSessionConfig;

// Represents the type of sampler.
//
// Added in version 0.1.0.
typedef enum {
  // Probabilistically pick among the top k tokens.
  kLiteRtLmSamplerTypeTopK = 1,
  // Probabilistically pick among the tokens such that the sum is greater
  // than or equal to p tokens after first performing top-k sampling.
  kLiteRtLmSamplerTypeTopP = 2,
  // Pick the token with maximum logit (i.e., argmax).
  kLiteRtLmSamplerTypeGreedy = 3,
} LiteRtLmSamplerType;

// Opaque pointer for LiteRT LM Sampler Parameters.
// Use `litert_lm_sampler_params_delete` to free memory.
//
// Added in version 0.1.0.
typedef struct LiteRtLmSamplerParams LiteRtLmSamplerParams;

// Creates LiteRT LM Sampler Parameters with a specific sampler type.
// The caller is responsible for destroying the parameters using
// `litert_lm_sampler_params_delete`.
//
// @param type The sampler type to use.
// @return A pointer to the created parameters, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmSamplerParams* litert_lm_sampler_params_create(
    LiteRtLmSamplerType type);

// Destroys LiteRT LM Sampler Parameters.
//
// @param params The parameters to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_sampler_params_delete(LiteRtLmSamplerParams* params);

// Sets the top-k value.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_sampler_params_set_top_k(LiteRtLmSamplerParams* params,
                                        int32_t top_k);

// Sets the top-p value.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_sampler_params_set_top_p(LiteRtLmSamplerParams* params,
                                        float top_p);

// Sets the temperature.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_sampler_params_set_temperature(LiteRtLmSamplerParams* params,
                                              float temperature);

// Sets the seed.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_sampler_params_set_seed(LiteRtLmSamplerParams* params,
                                       int32_t seed);

// Creates a LiteRT LM Session Config.
// The caller is responsible for destroying the config using
// `litert_lm_session_config_delete`.
// @return A pointer to the created config, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmSessionConfig* litert_lm_session_config_create();

// Sets the maximum number of output tokens per decode step for this session.
// @param config The config to modify.
// @param max_output_tokens The maximum number of output tokens.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_set_max_output_tokens(
    LiteRtLmSessionConfig* config, int max_output_tokens);

// Sets whether to apply prompt template for this session.
// @param config The config to modify.
// @param apply_prompt_template Whether to apply prompt template.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_set_apply_prompt_template(
    LiteRtLmSessionConfig* config, bool apply_prompt_template);

// Sets the sampler parameters for this session config.
// @param config The config to modify.
// @param sampler_params The sampler parameters to use.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_set_sampler_params(
    LiteRtLmSessionConfig* config, const LiteRtLmSamplerParams* sampler_params);

// Destroys a LiteRT LM Session Config.
// @param config The config to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_delete(LiteRtLmSessionConfig* config);

// Sets the path to the LoRA weights file.
// @param config The config to modify.
// @param lora_path The path to the text LoRA weights file.
// @return 0 on success, non-zero on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_session_config_set_lora_path(LiteRtLmSessionConfig* config,
                                           const char* lora_path);

// Sets the path to the Audio LoRA weights file.
// @param config The config to modify.
// @param audio_lora_path The path to the audio LoRA weights file.
// @return 0 on success, non-zero on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_session_config_set_audio_lora_path(LiteRtLmSessionConfig* config,
                                                 const char* audio_lora_path);

// Creates a LiteRT LM Repetition Penalty Config with default values
// (`repetition_penalty` = 1.0f, `presence_penalty` = 0.0f,
// `frequency_penalty` = 0.0f, `window_size` = 0, which means all history with
// no penalties active).
//
// When multiple penalties are configured and active, the order of application
// to output logits during decoding is:
// 1. Multiplicative penalty (`repetition_penalty`)
// 2. Subtractive penalties (`presence_penalty` and `frequency_penalty`)
//
// The caller is responsible for destroying the config using
// `litert_lm_repetition_penalty_config_delete`.
// @return A pointer to the created config, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmRepetitionPenaltyConfig* litert_lm_repetition_penalty_config_create();

// Destroys a LiteRT LM Repetition Penalty Config.
// @param config The config to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_repetition_penalty_config_delete(
    LiteRtLmRepetitionPenaltyConfig* config);

// Sets the multiplicative repetition penalty for the repetition penalty config.
// @param config The config to modify.
// @param repetition_penalty A multiplicative penalty applied to a token's logit
// if that token has appeared at least once inside the generated window history
// (e.g., 1.0 = no penalty, 1.2 = moderate penalty). Positive logits are divided
// by this parameter, and negative logits are multiplied (HuggingFace style).
// The parameter must be >= 1.0f; values less than 1.0f are automatically
// clamped to 1.0f during execution.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_repetition_penalty_config_set_repetition_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float repetition_penalty);

// Sets the subtractive presence penalty for the repetition penalty config.
// @param config The config to modify.
// @param presence_penalty A scalar subtracted from a token's logit if that
// token has appeared at least once inside the generated window history.
// Positive values discourage repetition, while negative values reward repeating
// tokens (OpenAI style). Defaults to 0.0f.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_repetition_penalty_config_set_presence_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float presence_penalty);

// Sets the subtractive frequency penalty for the repetition penalty config.
// @param config The config to modify.
// @param frequency_penalty A scalar subtracted from a token's logit, scaled
// linearly by the number of times that token has previously appeared inside the
// generated window history. Positive values discourage repetition, while
// negative values reward repeating tokens (OpenAI style). Defaults to 0.0f.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_repetition_penalty_config_set_frequency_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float frequency_penalty);

// Sets the window size for the repetition penalty config.
// @param config The config to modify.
// @param window_size The maximum number of recent tokens in generation history
// to consider when computing penalization. Tokens generated prior to this
// window are forgotten. A value of 0 means tracking all infinite generation
// history. Must be >= 0; negative values are clamped to 0 during execution.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_repetition_penalty_config_set_window_size(
    LiteRtLmRepetitionPenaltyConfig* config, int window_size);

// Creates a LiteRT LM No Repeat Ngram Config with default values
// (`no_repeat_ngram_size` = 0, `window_size` = 0, which means no repeat ngram
// banning is disabled).
//
// When `no_repeat_ngram_size` is set greater than 0, any sequence of tokens (an
// ngram of that exact length) generated during decoding or present inside the
// window history can only occur at most once. If generating a candidate token
// would complete a repeating ngram, that candidate token's logit is set to
// -inf.
//
// The caller is responsible for destroying the config using
// `litert_lm_no_repeat_ngram_config_delete`.
// @return A pointer to the created config, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmNoRepeatNgramConfig* litert_lm_no_repeat_ngram_config_create();

// Destroys a LiteRT LM No Repeat Ngram Config.
// @param config The config to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_no_repeat_ngram_config_delete(
    LiteRtLmNoRepeatNgramConfig* config);

// Sets the no repeat ngram size for the no repeat ngram config.
// @param config The config to modify.
// @param no_repeat_ngram_size The size of ngrams (consecutive token sequences)
// that are banned from repeating within the generation history window. If set
// > 0, when generating the next token would complete an already observed
// `no_repeat_ngram_size` sequence, the logit of the candidate token is set to
// -inf. If set <= 0, no repeat ngram banning is disabled. Negative values are
// automatically clamped to 0 during execution.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
    LiteRtLmNoRepeatNgramConfig* config, int no_repeat_ngram_size);

// Sets the window size for the no repeat ngram config.
// @param config The config to modify.
// @param window_size The maximum number of recent tokens in generation history
// to consider when checking for repeating ngrams. Tokens generated prior to
// this window are forgotten. A value of 0 means tracking all infinite
// generation history. Must be >= 0; negative values are clamped to 0. If
// `window_size` is greater than 0 but less than `no_repeat_ngram_size`, it is
// automatically clamped to `no_repeat_ngram_size` so that the ngrams can fit
// and be tracked.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_no_repeat_ngram_config_set_window_size(
    LiteRtLmNoRepeatNgramConfig* config, int window_size);

// Creates a LiteRT LM Suppress Tokens Config with default values (an empty set
// of suppressed tokens, which means token suppression is disabled).
//
// When `suppress_tokens` is configured with one or more token IDs, the logits
// corresponding to those exact token IDs will be set directly to -inf during
// generation. This guarantees that those tokens can never be sampled by the
// model.
//
// The caller is responsible for destroying the config using
// `litert_lm_suppress_tokens_config_delete`.
// @return A pointer to the created config, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmSuppressTokensConfig* litert_lm_suppress_tokens_config_create();

// Destroys a LiteRT LM Suppress Tokens Config.
// @param config The config to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_suppress_tokens_config_delete(
    LiteRtLmSuppressTokensConfig* config);

// Sets the list of token IDs to suppress for the suppress tokens config.
// @param config The config to modify.
// @param suppress_tokens An array of integer token IDs that should be banned
// from generation. During every decode step, each listed token ID's candidate
// logit will be forced to -inf. If `suppress_tokens` is NULL or `num_tokens` is
// 0, any previously set suppressed tokens are cleared and token suppression is
// disabled.
// @param num_tokens The number of token IDs in the `suppress_tokens` array.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_suppress_tokens_config_set_suppress_tokens(
    LiteRtLmSuppressTokensConfig* config, const int* suppress_tokens,
    size_t num_tokens);

// Represents the log severity / level.
//
// Added in version 0.1.0.
typedef enum {
  kLiteRtLmLogSeverityVerbose = 0,
  kLiteRtLmLogSeverityDebug = 1,
  kLiteRtLmLogSeverityInfo = 2,
  kLiteRtLmLogSeverityWarning = 3,
  kLiteRtLmLogSeverityError = 4,
  kLiteRtLmLogSeverityFatal = 5,
  kLiteRtLmLogSeveritySilent = 1000,
} LiteRtLmLogSeverity;
// Sets the minimum log level for the LiteRT LM library.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_set_min_log_level(LiteRtLmLogSeverity level);

// Represents the type of input data.
//
// Added in version 0.1.0.
typedef enum {
  kLiteRtLmInputDataTypeText,
  kLiteRtLmInputDataTypeImage,
  kLiteRtLmInputDataTypeImageEnd,
  kLiteRtLmInputDataTypeAudio,
  kLiteRtLmInputDataTypeAudioEnd,
} LiteRtLmInputDataType;

// Creates a LiteRT LM Input Data. The caller is responsible for destroying
// the input data using `litert_lm_input_data_delete`.
//
// @param type The type of the input data.
// @param data The data pointer. For kLiteRtLmInputDataTypeText, it's a UTF-8
// string.
//             For image/audio types, it's a pointer to the raw bytes.
//             The data is copied internally.
// @param size The size of the data in bytes.
// @return A pointer to the created input data, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmInputData* litert_lm_input_data_create(LiteRtLmInputDataType type,
                                               const void* data, size_t size);

// Destroys a LiteRT LM Input Data.
//
// @param input_data The input data to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_input_data_delete(LiteRtLmInputData* input_data);

// Creates LiteRT LM Engine Settings. The caller is responsible for destroying
// the settings using `litert_lm_engine_settings_delete`.
//
// @param model_path The path to the model file.
// @param backend_str The backend to use (e.g., "cpu", "gpu").
// @param vision_backend_str The vision backend to use, or NULL if not set.
// @param audio_backend_str The audio backend to use, or NULL if not set.
// @return A pointer to the created settings, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmEngineSettings* litert_lm_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str);

// Creates LiteRT LM Engine Settings from a raw file descriptor. The engine
// takes ownership of the file descriptor and will close it when done.
// The caller is responsible for destroying the settings using
// `litert_lm_engine_settings_delete`.
//
// @param fd The file descriptor of the model.
// @param backend_str The backend to use (e.g., "cpu", "gpu").
// @param vision_backend_str The vision backend to use, or NULL if not set.
// @param audio_backend_str The audio backend to use, or NULL if not set.
// @return A pointer to the created settings, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmEngineSettings*
litert_lm_engine_settings_create_from_raw_file_descriptor(
    int fd, const char* backend_str, const char* vision_backend_str,
    const char* audio_backend_str);

// Destroys LiteRT LM Engine Settings.
//
// @param settings The settings to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_delete(LiteRtLmEngineSettings* settings);

// Sets the maximum number of tokens for the engine.
//
// @param settings The engine settings.
// @param max_num_tokens The maximum number of tokens.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_max_num_tokens(
    LiteRtLmEngineSettings* settings, int max_num_tokens);

// Sets the number of threads for the CPU backend.
//
// @param settings The engine settings.
// @param num_threads The number of threads.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_num_threads(LiteRtLmEngineSettings* settings,
                                               int num_threads);

// Sets the number of threads for the audio CPU backend.
//
// @param settings The engine settings.
// @param num_threads The number of threads.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_audio_num_threads(
    LiteRtLmEngineSettings* settings, int num_threads);

// Sets whether the engine should load different sections of the litertlm file
// in parallel. Defaults to true.
//
// @param settings The engine settings.
// @param parallel_file_section_loading Whether to load in parallel.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_parallel_file_section_loading(
    LiteRtLmEngineSettings* settings, bool parallel_file_section_loading);

// Sets the maximum number of images for the engine.
//
// This is only used for the legacy implementation of the engine.
//
// @param settings The engine settings.
// @param max_num_images The maximum number of images.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_max_num_images(
    LiteRtLmEngineSettings* settings, int max_num_images);

// Sets the cache directory for the engine.
//
// @param settings The engine settings.
// @param cache_dir The cache directory.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_cache_dir(LiteRtLmEngineSettings* settings,
                                             const char* cache_dir);

// Sets the LiteRT dispatch library directory for NPU backend.
//
// @param settings The engine settings.
// @param lib_dir The dispatch library directory.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEngineSettings* settings, const char* lib_dir);

// Represents the activation data type.
//
// Added in version 0.1.0.
typedef enum {
  kLiteRtLmActivationDataTypeFloat32 = 0,
  kLiteRtLmActivationDataTypeFloat16 = 1,
  kLiteRtLmActivationDataTypeInt16 = 2,
  kLiteRtLmActivationDataTypeInt8 = 3,
} LiteRtLmActivationDataType;

// Sets the activation data type.
//
// @param settings The engine settings.
// @param activation_data_type The activation data type.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_activation_data_type(
    LiteRtLmEngineSettings* settings,
    LiteRtLmActivationDataType activation_data_type);

// Sets the prefill chunk size for the engine. Only applicable for CPU backend
// with dynamic models.
//
// @param settings The engine settings.
// @param prefill_chunk_size The prefill chunk size.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_prefill_chunk_size(
    LiteRtLmEngineSettings* settings, int prefill_chunk_size);

// Sets whether YNNPACK should delegate supported operations before XNNPACK.
//
// @param settings The engine settings.
// @param enable_ynnpack Whether to enable YNNPACK.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_enable_ynnpack(
    LiteRtLmEngineSettings* settings, bool enable_ynnpack);

// Enables benchmarking for the engine.
//
// @param settings The engine settings.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings);

// Sets the number of prefill tokens for benchmarking.
//
// @param settings The engine settings.
// @param num_prefill_tokens The number of prefill tokens.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_num_prefill_tokens(
    LiteRtLmEngineSettings* settings, int num_prefill_tokens);

// Sets the number of decode tokens for benchmarking.
//
// @param settings The engine settings.
// @param num_decode_tokens The number of decode tokens.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_num_decode_tokens(
    LiteRtLmEngineSettings* settings, int num_decode_tokens);

// Sets whether to enable speculative decoding.
//
// @param settings The engine settings.
// @param enable_speculative_decoding Whether to enable speculative decoding.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_enable_speculative_decoding(
    LiteRtLmEngineSettings* settings, bool enable_speculative_decoding);

// Sets the number of decode steps per sync for the GPU backend.
// Note: This setting is currently only supported for the Artisan GPU
// backend (Artisan).
//
// @param settings The engine settings.
// @param num_decode_steps_per_sync The number of decode steps per sync.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_gpu_decode_steps_per_sync(
    LiteRtLmEngineSettings* settings, int num_decode_steps_per_sync);

// Sets whether to wait for weight uploads for the GPU backend.
// Note: This setting is currently only supported for the Artisan GPU backend.
//
// @param settings The engine settings.
// @param wait_for_weight_uploads Whether to wait for weight uploads.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_gpu_wait_for_weight_uploads(
    LiteRtLmEngineSettings* settings, bool wait_for_weight_uploads);

// Sets whether to use ringbuffers for local attention KV cache.
//
// When enabled for supported models, a ringbuffer stores only necessary KV
// cache memory for local attention layers, minimizing memory usage. When
// disabled, memory is allocated for the full context length, enabling instant
// rewinding at the cost of higher memory usage.
//
// Note: This feature is backend-agnostic in interface design, but currently
// only supported by the GPU Artisan backend. Enabling it on unsupported models
// or backends will be ignored with a warning.
//
// @param settings The engine settings.
// @param use_ringbuffers_local_attention Whether to use ringbuffers for local
// attention.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_use_ringbuffers_local_attention(
    LiteRtLmEngineSettings* settings, bool use_ringbuffers_local_attention);

// Sets the LoRA rank for the engine.
//
// @param settings The engine settings.
// @param lora_rank The LoRA rank.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_lora_rank(LiteRtLmEngineSettings* settings,
                                             int lora_rank);

// Sets the supported LoRA ranks for the engine.
//
// @param settings The engine settings.
// @param lora_ranks An array of supported LoRA ranks.
// @param num_ranks The number of ranks in the array.
// @return 0 on success, non-zero on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_engine_settings_set_supported_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks);

// Sets the Audio LoRA rank for the engine.
//
// @param settings The engine settings.
// @param lora_rank The Audio LoRA rank.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_audio_lora_rank(
    LiteRtLmEngineSettings* settings, int lora_rank);

// Sets the supported Audio LoRA ranks for the engine.
//
// @param settings The engine settings.
// @param lora_ranks An array of supported Audio LoRA ranks.
// @param num_ranks The number of ranks in the array.
// @return 0 on success, non-zero on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_engine_settings_set_supported_audio_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks);

// Creates a LiteRT LM Engine from the given settings. The caller is responsible
// for destroying the engine using `litert_lm_engine_delete`.
//
// @param settings The engine settings.
// @return A pointer to the created engine, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmEngine* litert_lm_engine_create(const LiteRtLmEngineSettings* settings);

// Destroys a LiteRT LM Engine.
//
// @param engine The engine to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_delete(LiteRtLmEngine* engine);

// Creates a LiteRT LM Session. The caller is responsible for destroying the
// session using `litert_lm_session_delete`.
//
// @param engine The engine to create the session from.
// @param config The session config of the session. If NULL, use the default
// session config.
// @return A pointer to the created session, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmSession* litert_lm_engine_create_session(LiteRtLmEngine* engine,
                                                 LiteRtLmSessionConfig* config);

// Destroys a LiteRT LM Session.
//
// @param session The session to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_session_delete(LiteRtLmSession* session);

// Cancels the current processing in the session.
//
// @param session The session to cancel processing on.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_session_cancel_process(LiteRtLmSession* session);

// Saves the current state of the session to a checkpoint with the given label.
//
// @param session The session to save checkpoint for.
// @param label Label for the checkpoint.
// @return 0 on success, non-zero on failure.
LITERT_LM_C_API_EXPORT
int litert_lm_session_save_checkpoint(LiteRtLmSession* session,
                                      const char* label);

// Rewinds the session to the given checkpoint label.
//
// @param session The session to rewind.
// @param label Label of the checkpoint to rewind to.
// @return 0 on success, non-zero on failure.
LITERT_LM_C_API_EXPORT
int litert_lm_session_rewind_to_checkpoint(LiteRtLmSession* session,
                                           const char* label);

// Rewinds the session to a specific step number.
//
// @param session The session to rewind.
// @param step The step number to rewind to.
// @return 0 on success, non-zero on failure.
LITERT_LM_C_API_EXPORT
int litert_lm_session_rewind_to_step(LiteRtLmSession* session, int step);

// Adds the input prompt/query to the model for starting the prefilling
// process. This is a blocking call and the function will return when the
// prefill process is done.
//
// @param session The session to use.
// @param inputs An array of InputData structs representing the multimodal
//   input.
// @param num_inputs The number of InputData structs in the array.
// @return 0 on success, non-zero on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_session_run_prefill(LiteRtLmSession* session,
                                  const LiteRtLmInputData* const* inputs,
                                  size_t num_inputs);

// Starts the decoding process for the model to predict the response based
// on the input prompt/query added after using litert_lm_session_run_prefill.
// This is a blocking call and the function will return when the decoding
// process is done.
//
// @param session The session to use.
// @return A pointer to the responses, or NULL on failure. The caller is
//   responsible for deleting the responses using `litert_lm_responses_delete`.
//
// Added in version 0.1.0.
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
//
// Added in version 0.1.0.
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
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmResponses* litert_lm_session_generate_content(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs);
// Destroys a LiteRT LM Responses object.
//
// @param responses The responses to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_responses_delete(LiteRtLmResponses* responses);

// Returns the number of response candidates.
//
// @param responses The responses object.
// @return The number of candidates.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_responses_get_num_candidates(const LiteRtLmResponses* responses);

// Returns the response text at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return The response text. The returned string is owned by the `responses`
//   object and is valid only for its lifetime. Returns NULL if index is out of
//   bounds.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index);

// Returns whether the response contains a score at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return true if the score is available at the given index, false otherwise.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_responses_has_score_at(const LiteRtLmResponses* responses,
                                      int index);

// Returns the score at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return The score. Returns 0.0f if index is out of bounds or no score is
//   present.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
float litert_lm_responses_get_score_at(const LiteRtLmResponses* responses,
                                       int index);

// Returns whether the response contains a token length at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return true if the token length is available at the given index, false
//   otherwise.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_responses_has_token_length_at(const LiteRtLmResponses* responses,
                                             int index);

// Returns the token length at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return The token length. Returns 0 if index is out of bounds or no token
//   length is present.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_responses_get_token_length_at(const LiteRtLmResponses* responses,
                                            int index);

// Returns whether the response contains token scores at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return true if token scores are available at the given index, false
// otherwise.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_responses_has_token_scores_at(const LiteRtLmResponses* responses,
                                             int index);

// Returns the number of tokens for which scores are present at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @return The number of token scores. Returns 0 if index is out of bounds or no
//   token scores are present.
//
// Added in version 0.1.0.
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
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const float* litert_lm_responses_get_token_scores_at(
    const LiteRtLmResponses* responses, int index);

// Retrieves the benchmark information from the session. The caller is
// responsible for destroying the benchmark info using
// `litert_lm_benchmark_info_delete`.
//
// @param session The session to get the benchmark info from.
// @return A pointer to the benchmark info, or NULL on failure.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmBenchmarkInfo* litert_lm_session_get_benchmark_info(
    LiteRtLmSession* session);

// Destroys a LiteRT LM Benchmark Info object.
//
// @param benchmark_info The benchmark info to destroy.
//
// Added in version 0.1.0.
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
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
double litert_lm_benchmark_info_get_time_to_first_token(
    const LiteRtLmBenchmarkInfo* benchmark_info);

// Returns the total initialization time in seconds.
//
// @param benchmark_info The benchmark info object.
// @return The total initialization time in seconds.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
double litert_lm_benchmark_info_get_total_init_time_in_second(
    const LiteRtLmBenchmarkInfo* benchmark_info);

// Returns the number of prefill turns.
//
// @param benchmark_info The benchmark info object.
// @return The number of prefill turns.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_benchmark_info_get_num_prefill_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info);

// Returns the number of decode turns.
//
// @param benchmark_info The benchmark info object.
// @return The number of decode turns.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_benchmark_info_get_num_decode_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info);

// Returns the prefill token count at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the prefill turn.
// @return The prefill token count.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_benchmark_info_get_prefill_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index);

// Returns the decode token count at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the decode turn.
// @return The decode token count.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_benchmark_info_get_decode_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index);

// Returns the prefill tokens per second at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the prefill turn.
// @return The prefill tokens per second.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
double litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index);

// Returns the decode tokens per second at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the decode turn.
// @return The decode tokens per second.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
double litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index);

// Opaque pointer for LiteRT LM Stream Chunk.
// This object represents a single chunk of data returned during streaming.
// It is owned by the library and is only valid for the duration of the
// callback.
//
// Added in version 0.1.0.
typedef struct LiteRtLmStreamChunk LiteRtLmStreamChunk;

// Gets the text content of the chunk.
// The returned string is owned by the chunk and is only valid as long as the
// chunk is valid. Returns NULL if there is no text content in this chunk (e.g.
// if it is an error or metadata-only chunk).
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_stream_chunk_get_text(const LiteRtLmStreamChunk* chunk);

// Returns true if this is the final chunk of the stream.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_stream_chunk_is_final(const LiteRtLmStreamChunk* chunk);

// Gets the error message associated with this chunk, if any.
// Returns NULL if there is no error.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_stream_chunk_get_error(const LiteRtLmStreamChunk* chunk);

// Callback for streaming responses.
// `callback_data` is a pointer to user-defined data passed to the stream
// function. `chunk` is a pointer to the stream chunk object. It's only valid
// for the duration of the call.
//
// Added in version 0.1.0.
typedef void (*LiteRtLmStreamCallback)(void* callback_data,
                                       const LiteRtLmStreamChunk* chunk);

// Starts the decoding process for the model to predict the response based
// on the input prompt/query added after using litert_lm_session_run_prefill.
// This is a non-blocking call that will stream responses via a callback.
//
// @param session The session to use.
// @param callback The callback function to receive response chunks.
// @param callback_data A pointer to user data that will be passed to the
// callback.
// @return 0 on success, non-zero on failure.
//
// Added in version 0.1.0.
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
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_session_generate_content_stream(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, LiteRtLmStreamCallback callback, void* callback_data);

// Tokenizes text using the engine's tokenizer.
//
// @param engine The engine instance.
// @param text The UTF-8 string to tokenize.
// @return A pointer to the tokenize result, or NULL on failure.
//   The caller is responsible for deleting the result using
//   `litert_lm_tokenize_result_delete`.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenizeResult* litert_lm_engine_tokenize(LiteRtLmEngine* engine,
                                                  const char* text);

// Destroys a LiteRT LM Tokenize Result.
//
// @param result The tokenize result to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_tokenize_result_delete(LiteRtLmTokenizeResult* result);

// Returns the token ids from a tokenize result.
//
// @param result The tokenize result.
// @return A pointer to the internal array of token ids. The returned pointer
//   is valid only for the lifetime of the `result` object.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const int* litert_lm_tokenize_result_get_tokens(
    const LiteRtLmTokenizeResult* result);

// Returns the number of token ids from a tokenize result.
//
// @param result The tokenize result.
// @return The number of token ids.
//
// Added in version 0.1.0.
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
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmDetokenizeResult* litert_lm_engine_detokenize(LiteRtLmEngine* engine,
                                                      const int* tokens,
                                                      size_t num_tokens);

// Destroys a LiteRT LM Detokenize Result.
//
// @param result The detokenize result to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_detokenize_result_delete(LiteRtLmDetokenizeResult* result);

// Returns the string from a detokenize result.
//
// @param result The detokenize result.
// @return The detokenized UTF-8 string. The returned string is owned by the
//   `result` object and is valid only for its lifetime.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_detokenize_result_get_string(
    const LiteRtLmDetokenizeResult* result);

// Destroys a LiteRT LM Token Union.
//
// @param token_union The token union to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_token_union_delete(LiteRtLmTokenUnion* token_union);

// Returns the type of the token union.
//
// @param token_union The token union.
// @return The type of the token union.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenUnionType litert_lm_token_union_get_type(
    const LiteRtLmTokenUnion* token_union);

// Returns the string value from a token union.
//
// @param token_union The token union.
// @return The string value, or NULL if the type is not
//   kLiteRtLmTokenUnionTypeString. The returned string is owned by the
//   `token_union` object and is valid only for its lifetime.
//
// Added in version 0.1.0.
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
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
int litert_lm_token_union_get_ids(const LiteRtLmTokenUnion* token_union,
                                  const int** out_tokens,
                                  size_t* out_num_tokens);

// Destroys a LiteRT LM Token Unions object.
//
// @param tokens The token unions object to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_token_unions_delete(LiteRtLmTokenUnions* tokens);

// Returns the number of token unions in the collection.
//
// @param tokens The token unions object.
// @return The number of token unions.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
size_t litert_lm_token_unions_get_num_tokens(const LiteRtLmTokenUnions* tokens);

// Returns the token union at a given index from a collection.
//
// @param tokens The token unions collection.
// @param index The index of the token union.
// @return A pointer to the token union at the given index, or NULL if the index
//   is out of bounds. The caller is responsible for deleting the result using
//   `litert_lm_token_union_delete`.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenUnion* litert_lm_token_unions_get_token_at(
    const LiteRtLmTokenUnions* tokens, size_t index);

// Returns the configured start token (BOS), if any.
//
// @param engine The engine instance.
// @return A pointer to the start token, or NULL if none configured. The caller
//   is responsible for deleting the result using
//   `litert_lm_token_union_delete`.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenUnion* litert_lm_engine_get_start_token(LiteRtLmEngine* engine);

// Returns the configured stop tokens (EOS).
//
// @param engine The engine instance.
// @return A pointer to the stop tokens collection, or NULL if none configured.
//   The caller is responsible for deleting the result using
//   `litert_lm_token_unions_delete`.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmTokenUnions* litert_lm_engine_get_stop_tokens(LiteRtLmEngine* engine);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_H_
