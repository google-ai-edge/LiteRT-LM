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

#if defined(__APPLE__)
#include "api_export.h"      // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/api_export.h"
#include "c/error_reporter.h"
#endif

#ifdef __cplusplus
extern "C" {
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
  // Default fallback/unspecified.
  kLiteRtLmSamplerTypeUnspecified = 0,
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
//
// @param type The sampler type to use.
// @param out_params On success, receives the created parameters, owned by the
//   caller; release with `litert_lm_sampler_params_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_params` is NULL or `type` is not a recognized LiteRtLmSamplerType.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the parameters are returned
// through out_params.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_create(
    LiteRtLmSamplerType type, LiteRtLmSamplerParams** out_params);

// Destroys LiteRT LM Sampler Parameters.
//
// @param params The parameters to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_sampler_params_delete(LiteRtLmSamplerParams* params);

// Sets the top-k value.
//
// @param params The sampler parameters to modify.
// @param top_k The top-k value.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `params` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_set_top_k(
    LiteRtLmSamplerParams* params, int32_t top_k);

// Sets the top-p value.
//
// @param params The sampler parameters to modify.
// @param top_p The top-p value.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `params` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_set_top_p(
    LiteRtLmSamplerParams* params, float top_p);

// Sets the temperature.
//
// @param params The sampler parameters to modify.
// @param temperature The sampling temperature.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `params` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_set_temperature(
    LiteRtLmSamplerParams* params, float temperature);

// Sets the seed.
//
// @param params The sampler parameters to modify.
// @param seed The random seed.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `params` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_set_seed(
    LiteRtLmSamplerParams* params, int32_t seed);

// Creates a LiteRT LM Session Config.
//
// @param out_config On success, receives the created config, owned by the
//   caller; release with `litert_lm_session_config_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the config is returned
// through out_config.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_create(
    LiteRtLmSessionConfig** out_config);

// Sets the maximum number of output tokens per decode step for this session.
// For thinking models, both thinking (reasoning) tokens and the final response
// tokens count towards this limit.
// @param config The config to modify.
// @param max_output_tokens The maximum number of tokens to generate (including
// thinking tokens).
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_max_output_tokens(
    LiteRtLmSessionConfig* config, int max_output_tokens);

// Sets whether to apply prompt template for this session.
// @param config The config to modify.
// @param apply_prompt_template Whether to apply prompt template.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_apply_prompt_template(
    LiteRtLmSessionConfig* config, bool apply_prompt_template);

// Sets whether to enable speculative decoding for this session.
// @param config The config to modify.
// @param enable_speculative_decoding Whether to enable speculative decoding.
// If set to true, speculative decoding is enabled for this session. If the
// engine was not initialized with speculative decoding enabled, setting this
// flag to true causes the executor to perform lazy loading of the
// MTP drafter on the first session request. If set to false, speculative
// decoding is explicitly disabled for this session even if the engine was
// initialized with speculative decoding enabled. If this function is not called
// on the config, the session inherits the engine's speculative decoding setting
// by default.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_enable_speculative_decoding(
    LiteRtLmSessionConfig* config, bool enable_speculative_decoding);

// Sets the sampler parameters for this session config.
// @param config The config to modify.
// @param sampler_params The sampler parameters to use.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` or `sampler_params` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_sampler_params(
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
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `config`
//   or `lora_path` is NULL or `lora_path` is empty; otherwise the code of the
//   error encountered while opening the file.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns LiteRtLmStatusCode and failures return a
// canonical LiteRtLmStatusCode instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_lora_path(
    LiteRtLmSessionConfig* config, const char* lora_path);

// Sets the path to the Audio LoRA weights file.
// @param config The config to modify.
// @param audio_lora_path The path to the audio LoRA weights file.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `config`
//   or `audio_lora_path` is NULL or `audio_lora_path` is empty; otherwise the
//   code of the error encountered while opening the file.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns LiteRtLmStatusCode and failures return a
// canonical LiteRtLmStatusCode instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_audio_lora_path(
    LiteRtLmSessionConfig* config, const char* audio_lora_path);

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
// @param out_config On success, receives the created config, owned by the
//   caller; release with `litert_lm_repetition_penalty_config_delete`. Set to
//   NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the config is returned
// through out_config.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_create(
    LiteRtLmRepetitionPenaltyConfig** out_config);

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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_repetition_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float repetition_penalty);

// Sets the subtractive presence penalty for the repetition penalty config.
// @param config The config to modify.
// @param presence_penalty A scalar subtracted from a token's logit if that
// token has appeared at least once inside the generated window history.
// Positive values discourage repetition, while negative values reward repeating
// tokens (OpenAI style). Defaults to 0.0f.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_presence_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float presence_penalty);

// Sets the subtractive frequency penalty for the repetition penalty config.
// @param config The config to modify.
// @param frequency_penalty A scalar subtracted from a token's logit, scaled
// linearly by the number of times that token has previously appeared inside the
// generated window history. Positive values discourage repetition, while
// negative values reward repeating tokens (OpenAI style). Defaults to 0.0f.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_frequency_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float frequency_penalty);

// Sets the window size for the repetition penalty config.
// @param config The config to modify.
// @param window_size The maximum number of recent tokens in generation history
// to consider when computing penalization. Tokens generated prior to this
// window are forgotten. A value of 0 means tracking all infinite generation
// history. Must be >= 0; negative values are clamped to 0 during execution.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_window_size(
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
// @param out_config On success, receives the created config, owned by the
//   caller; release with `litert_lm_no_repeat_ngram_config_delete`. Set to
//   NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the config is returned
// through out_config.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_create(
    LiteRtLmNoRepeatNgramConfig** out_config);

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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_set_window_size(
    LiteRtLmNoRepeatNgramConfig* config, int window_size);

// Creates a LiteRT LM Suppress Tokens Config with default values (an empty set
// of suppressed tokens, which means token suppression is disabled).
//
// When `suppress_tokens` is configured with one or more token IDs, the logits
// corresponding to those exact token IDs will be set directly to -inf during
// generation. This guarantees that those tokens can never be sampled by the
// model.
//
// @param out_config On success, receives the created config, owned by the
//   caller; release with `litert_lm_suppress_tokens_config_delete`. Set to
//   NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_config` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the config is returned
// through out_config.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_suppress_tokens_config_create(
    LiteRtLmSuppressTokensConfig** out_config);

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
// logit will be forced to -inf. If `num_tokens` is 0, any previously set
// suppressed tokens are cleared and token suppression is disabled; in that case
// `suppress_tokens` may be NULL.
// @param num_tokens The number of token IDs in the `suppress_tokens` array.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `config` is NULL, or if `suppress_tokens` is NULL while `num_tokens` is
//   non-zero (the config is left unchanged).
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_suppress_tokens_config_set_suppress_tokens(
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
// @param level The minimum severity to log.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `level` is not a declared LiteRtLmLogSeverity value.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_set_min_log_level(LiteRtLmLogSeverity level);

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

// Creates a LiteRT LM Input Data.
//
// @param type The type of the input data.
// @param data The data pointer. For kLiteRtLmInputDataTypeText, it's a UTF-8
// string.
//             For image/audio types, it's a pointer to the raw bytes.
//             The data is copied internally.
// @param size The size of the data in bytes.
// @param out_input_data On success, receives the created input data, owned by
//   the caller; release with `litert_lm_input_data_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_input_data` is NULL, `data` is NULL while `size` is non-zero, or
//   `type` is not a declared LiteRtLmInputDataType value.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the input data is returned
// through out_input_data.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_input_data_create(
    LiteRtLmInputDataType type, const void* data, size_t size,
    LiteRtLmInputData** out_input_data);

// Destroys a LiteRT LM Input Data.
//
// @param input_data The input data to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_input_data_delete(LiteRtLmInputData* input_data);

// Creates LiteRT LM Engine Settings.
//
// @param model_path The path to the model file.
// @param backend_str The backend to use (e.g., "cpu", "gpu").
// @param vision_backend_str The vision backend to use, or NULL if not set.
// @param audio_backend_str The audio backend to use, or NULL if not set.
// @param out_settings On success, receives the created settings, owned by the
//   caller; release with `litert_lm_engine_settings_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_settings` or `model_path` is NULL or a backend string is not
//   recognized; otherwise the code of the error encountered while loading the
//   model assets or building the settings.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the settings are returned
// through out_settings.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str,
    LiteRtLmEngineSettings** out_settings);

// Creates LiteRT LM Engine Settings from a raw file descriptor. The engine
// takes ownership of the file descriptor and will close it when done. If
// `out_settings` is NULL or `fd` is negative, ownership is not taken.
//
// @param fd The file descriptor of the model.
// @param backend_str The backend to use (e.g., "cpu", "gpu").
// @param vision_backend_str The vision backend to use, or NULL if not set.
// @param audio_backend_str The audio backend to use, or NULL if not set.
// @param out_settings On success, receives the created settings, owned by the
//   caller; release with `litert_lm_engine_settings_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_settings` is NULL, `fd` is negative or a backend string is not
//   recognized; otherwise the code of the error encountered while loading the
//   model assets or building the settings.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the settings are returned
// through out_settings.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_create_from_raw_file_descriptor(
    int fd, const char* backend_str, const char* vision_backend_str,
    const char* audio_backend_str, LiteRtLmEngineSettings** out_settings);

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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_max_num_tokens(
    LiteRtLmEngineSettings* settings, int max_num_tokens);

// Sets the number of threads for the CPU backend.
//
// @param settings The engine settings.
// @param num_threads The number of threads.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If the main backend is not CPU, this call has no effect
//   and returns kLiteRtLmStatusOk.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_num_threads(
    LiteRtLmEngineSettings* settings, int num_threads);

// Sets the number of threads for the audio CPU backend.
//
// @param settings The engine settings.
// @param num_threads The number of threads.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If no audio backend is configured, this call has no
//   effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_audio_num_threads(
    LiteRtLmEngineSettings* settings, int num_threads);

// Sets whether the engine should load different sections of the litertlm file
// in parallel. Defaults to true.
//
// @param settings The engine settings.
// @param parallel_file_section_loading Whether to load in parallel.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_parallel_file_section_loading(
    LiteRtLmEngineSettings* settings, bool parallel_file_section_loading);

// Sets whether to enable single threaded execution.
//
// @param settings The engine settings.
// @param single_threaded_execution Whether to enable single threaded
// execution.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_single_threaded_execution(
    LiteRtLmEngineSettings* settings, bool single_threaded_execution);

// Sets the maximum number of images for the engine.
//
// This is only used for the legacy implementation of the engine.
//
// @param settings The engine settings.
// @param max_num_images The maximum number of images.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_max_num_images(
    LiteRtLmEngineSettings* settings, int max_num_images);

// Sets the maximum vision tokens generated per image for the engine.
//
// When set, the engine automatically selects vision encoder and adapter
// signatures with capacity up to this limit and configures vision patch
// metadata.
//
// @param settings The engine settings.
// @param max_vision_tokens_per_image The maximum vision tokens per image.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_max_vision_tokens_per_image(
    LiteRtLmEngineSettings* settings, int max_vision_tokens_per_image);

// Sets the cache directory for the engine.
//
// @param settings The engine settings.
// @param cache_dir The cache directory.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` or `cache_dir` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_cache_dir(
    LiteRtLmEngineSettings* settings, const char* cache_dir);

// Sets the LiteRT dispatch library directory for NPU backend.
//
// @param settings The engine settings.
// @param lib_dir The dispatch library directory.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` or `lib_dir` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_litert_dispatch_lib_dir(
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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL or `activation_data_type` is not a declared
//   LiteRtLmActivationDataType value.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_activation_data_type(
    LiteRtLmEngineSettings* settings,
    LiteRtLmActivationDataType activation_data_type);

// Sets the prefill chunk size for the engine. Only applicable for CPU backend
// with dynamic models.
//
// @param settings The engine settings.
// @param prefill_chunk_size The prefill chunk size.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If the main backend is not CPU, this call has no effect
//   and returns kLiteRtLmStatusOk.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_prefill_chunk_size(
    LiteRtLmEngineSettings* settings, int prefill_chunk_size);

// Sets whether YNNPACK should delegate supported operations before XNNPACK.
//
// @param settings The engine settings.
// @param enable_ynnpack Whether to enable YNNPACK.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If the main backend is not CPU, this call has no effect
//   and returns kLiteRtLmStatusOk.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_enable_ynnpack(
    LiteRtLmEngineSettings* settings, bool enable_ynnpack);

// Enables benchmarking for the engine.
//
// @param settings The engine settings.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings);

// Sets the number of prefill tokens for benchmarking.
//
// @param settings The engine settings.
// @param num_prefill_tokens The number of prefill tokens.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_num_prefill_tokens(
    LiteRtLmEngineSettings* settings, int num_prefill_tokens);

// Sets the number of decode tokens for benchmarking.
//
// @param settings The engine settings.
// @param num_decode_tokens The number of decode tokens.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_num_decode_tokens(
    LiteRtLmEngineSettings* settings, int num_decode_tokens);

// Sets whether to enable speculative decoding.
//
// @param settings The engine settings.
// @param enable_speculative_decoding Whether to enable speculative decoding.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_enable_speculative_decoding(
    LiteRtLmEngineSettings* settings, bool enable_speculative_decoding);

// Sets the number of decode steps per sync for the GPU backend.
// Note: This setting is currently only supported for the Artisan GPU
// backend (Artisan).
//
// @param settings The engine settings.
// @param num_decode_steps_per_sync The number of decode steps per sync.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If the main backend is not the Artisan GPU backend,
//   this call has no effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_decode_steps_per_sync(
    LiteRtLmEngineSettings* settings, int num_decode_steps_per_sync);

// Sets whether to wait for weight uploads for the GPU backend.
// Note: This setting is currently only supported for the Artisan GPU backend.
//
// @param settings The engine settings.
// @param wait_for_weight_uploads Whether to wait for weight uploads.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If the main backend is not the Artisan GPU backend,
//   this call has no effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_wait_for_weight_uploads(
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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If the main backend is not the Artisan GPU backend,
//   this call has no effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_engine_settings_set_use_ringbuffers_local_attention(
    LiteRtLmEngineSettings* settings, bool use_ringbuffers_local_attention);

// Sets the LoRA rank for the engine.
//
// @param settings The engine settings.
// @param lora_rank The LoRA rank.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_lora_rank(
    LiteRtLmEngineSettings* settings, int lora_rank);

// Sets the supported LoRA ranks for the engine.
//
// @param settings The engine settings.
// @param lora_ranks An array of supported LoRA ranks.
// @param num_ranks The number of ranks in the array.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `settings` or `lora_ranks` is NULL or `num_ranks` is 0.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_supported_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks);

// Sets the Audio LoRA rank for the engine.
//
// @param settings The engine settings.
// @param lora_rank The Audio LoRA rank.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If no audio backend is configured, this call has no
//   effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_audio_lora_rank(
    LiteRtLmEngineSettings* settings, int lora_rank);

// Sets the supported Audio LoRA ranks for the engine.
//
// @param settings The engine settings.
// @param lora_ranks An array of supported Audio LoRA ranks.
// @param num_ranks The number of ranks in the array.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `settings` or `lora_ranks` is NULL or `num_ranks` is 0;
//   kLiteRtLmStatusFailedPrecondition if no audio executor is configured.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_supported_audio_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks);

// Sets whether to enable Metal residency set on GPU.
//
// When enabled on Apple platforms (macOS and iOS with Metal GPU backend), this
// uses Apple's MTLResidencySet API to ensure model weights and allocations
// remain resident in GPU memory, preventing memory swapping and reducing
// allocation overhead.
//
// This setting is only supported on Apple platforms (macOS / iOS) with the GPU
// backend. On other platforms (e.g. Linux, Android, Windows) or non-GPU
// backends, this setting has no effect and is safely ignored.
//
// @param settings The engine settings.
// @param enable_metal_residency_set Whether to enable Metal residency set.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. The setting is still recorded (and kLiteRtLmStatusOk
//   returned) on platforms or backends where it has no effect.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_enable_metal_residency_set(
    LiteRtLmEngineSettings* settings, bool enable_metal_residency_set);

// Creates a LiteRT LM Engine from the given settings.
//
// @param settings The engine settings.
// @param out_engine On success, receives the created engine, owned by the
//   caller; release with `litert_lm_engine_delete`. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `settings` or `out_engine` is NULL; otherwise the code of the error
//   encountered while creating the engine (e.g. the model cannot be loaded).
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the engine is returned
// through out_engine.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_create(
    const LiteRtLmEngineSettings* settings, LiteRtLmEngine** out_engine);

// Destroys a LiteRT LM Engine.
//
// @param engine The engine to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_delete(LiteRtLmEngine* engine);

// Creates a LiteRT LM Session.
//
// @param engine The engine to create the session from.
// @param config The session config of the session. If NULL, use the default
// session config.
// @param out_session On success, receives the created session, owned by the
//   caller; release with `litert_lm_session_delete`. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `engine`
//   or `out_session` is NULL; otherwise the code of the error encountered
//   while creating the session.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the session is returned
// through out_session.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_create_session(
    LiteRtLmEngine* engine, LiteRtLmSessionConfig* config,
    LiteRtLmSession** out_session);

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
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `session` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_cancel_process(LiteRtLmSession* session);

// Saves the current state of the session to a checkpoint with the given label.
//
// @param session The session to save checkpoint for.
// @param label Label for the checkpoint.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `session`
//   or `label` is NULL.
//
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_save_checkpoint(LiteRtLmSession* session,
                                                     const char* label);

// Rewinds the session to the given checkpoint label.
//
// @param session The session to rewind.
// @param label Label of the checkpoint to rewind to.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `session`
//   or `label` is NULL; otherwise the runtime error (e.g. if no checkpoint with
//   `label` exists).
//
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_rewind_to_checkpoint(
    LiteRtLmSession* session, const char* label);

// Rewinds the session to a specific step number.
//
// @param session The session to rewind.
// @param step The step number to rewind to.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `session`
//   is NULL; otherwise the runtime error (e.g. if `step` is invalid).
//
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_rewind_to_step(LiteRtLmSession* session,
                                                    int step);

// Adds the input prompt/query to the model for starting the prefilling
// process. This is a blocking call and the function will return when the
// prefill process is done.
//
// @param session The session to use.
// @param inputs An array of InputData structs representing the multimodal
//   input.
// @param num_inputs The number of InputData structs in the array.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `session`
//   or `inputs` is NULL or `num_inputs` is 0.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_run_prefill(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs);

// Starts the decoding process for the model to predict the response based
// on the input prompt/query added after using litert_lm_session_run_prefill.
// This is a blocking call and the function will return when the decoding
// process is done.
//
// @param session The session to use.
// @param out_responses On success, receives the responses, owned by the
//   caller; release with `litert_lm_responses_delete`. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `session` or `out_responses` is NULL; otherwise the runtime error.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_responses.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_run_decode(
    LiteRtLmSession* session, LiteRtLmResponses** out_responses);

// Scores the target text after the prefill process is done.
//
// @param session The session to use.
// @param target_text An array of target text strings to score.
// @param num_targets The number of strings in the target_text array.
// @param store_token_lengths Whether to store the token lengths of the target
//   texts in the responses.
// @param out_responses On success, receives the responses, owned by the
//   caller; release with `litert_lm_responses_delete`. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `session`, `target_text` or `out_responses` is NULL or `num_targets` is 0;
//   otherwise the runtime error.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_responses.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_run_text_scoring(
    LiteRtLmSession* session, const char** target_text, size_t num_targets,
    bool store_token_lengths, LiteRtLmResponses** out_responses);

// Generates content from the input prompt.
//
// @param session The session to use for generation.
// @param inputs An array of LiteRtLmInputData structs representing the
// multimodal
//   input.
// @param num_inputs The number of LiteRtLmInputData structs in the array.
// @param out_responses On success, receives the responses, owned by the
//   caller; release with `litert_lm_responses_delete`. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `session` or `out_responses` is NULL; otherwise the code of the error
//   encountered while copying the inputs or generating content.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_responses.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_generate_content(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, LiteRtLmResponses** out_responses);

// Destroys a LiteRT LM Responses object.
//
// @param responses The responses to destroy.
//
// Added in version 0.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_responses_delete(LiteRtLmResponses* responses);

// Returns the number of response candidates.
//
// The number of candidates is the number of response texts or, if there are
// none (e.g. for text scoring), the number of scores or token lengths. Valid
// candidate indices for the `litert_lm_responses_*_at` functions are
// `[0, *out_num_candidates)`, although individual values may still be absent
// at a valid index (see the `has_*_at` predicates).
//
// @param responses The responses object.
// @param out_num_candidates On success, receives the number of candidates; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_num_candidates` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_num_candidates.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_num_candidates(
    const LiteRtLmResponses* responses, int* out_num_candidates);

// Returns the response text at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @param out_text On success, receives the response text. The string is owned
//   by the `responses` object and is valid only for its lifetime. Set to NULL
//   on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_text` is NULL; kLiteRtLmStatusOutOfRange if `index` is
//   out of bounds of the candidates; kLiteRtLmStatusNotFound if there is no
//   response text at `index`.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the text is returned
// through out_text.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index, const char** out_text);

// Returns whether the response contains a score at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @param out_has_score On success, receives true if a score is available at
//   `index`, false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_has_score` is NULL; kLiteRtLmStatusOutOfRange if
//   `index` is out of bounds of the candidates.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_has_score.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_has_score_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_score);

// Returns the score at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @param out_score On success, receives the score; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_score` is NULL; kLiteRtLmStatusOutOfRange if `index`
//   is out of bounds of the candidates; kLiteRtLmStatusNotFound if no score is
//   available at `index`.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the score is returned
// through out_score.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_score_at(
    const LiteRtLmResponses* responses, int index, float* out_score);

// Returns whether the response contains a token length at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @param out_has_token_length On success, receives true if a token length is
//   available at `index`, false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_has_token_length` is NULL; kLiteRtLmStatusOutOfRange
//   if `index` is out of bounds of the candidates.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_has_token_length.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_has_token_length_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_token_length);

// Returns the token length at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @param out_token_length On success, receives the token length; not written
//   on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_token_length` is NULL; kLiteRtLmStatusOutOfRange if
//   `index` is out of bounds of the candidates; kLiteRtLmStatusNotFound if no
//   token length is available at `index`.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the token length is
// returned through out_token_length.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_token_length_at(
    const LiteRtLmResponses* responses, int index, int* out_token_length);

// Returns whether the response contains token scores at the given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @param out_has_token_scores On success, receives true if token scores are
//   available at `index`, false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_has_token_scores` is NULL; kLiteRtLmStatusOutOfRange
//   if `index` is out of bounds of the candidates.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_has_token_scores.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_has_token_scores_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_token_scores);

// Returns the number of tokens for which scores are present at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @param out_num_token_scores On success, receives the number of token scores;
//   not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_num_token_scores` is NULL; kLiteRtLmStatusOutOfRange
//   if `index` is out of bounds of the candidates; kLiteRtLmStatusNotFound if
//   no token scores are available at `index`.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_num_token_scores.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_num_token_scores_at(
    const LiteRtLmResponses* responses, int index, int* out_num_token_scores);

// Returns the token scores at a given index.
//
// @param responses The responses object.
// @param index The index of the response.
// @param out_token_scores On success, receives a pointer to the internal array
//   of token scores, whose length is given by
//   `litert_lm_responses_get_num_token_scores_at`. The array is owned by the
//   `responses` object and is valid only for its lifetime. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_token_scores` is NULL; kLiteRtLmStatusOutOfRange if
//   `index` is out of bounds of the candidates; kLiteRtLmStatusNotFound if no
//   token scores are available at `index`.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the array is returned
// through out_token_scores.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_token_scores_at(
    const LiteRtLmResponses* responses, int index,
    const float** out_token_scores);

// Retrieves the benchmark information from the session.
//
// @param session The session to get the benchmark info from.
// @param out_benchmark_info On success, receives the benchmark info, owned by
//   the caller; release with `litert_lm_benchmark_info_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `session` or `out_benchmark_info` is NULL; otherwise the runtime error
//   (e.g. if benchmarking is not enabled).
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_benchmark_info.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_get_benchmark_info(
    LiteRtLmSession* session, LiteRtLmBenchmarkInfo** out_benchmark_info);

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
// @param out_seconds On success, receives the time to the first token in
//   seconds; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `benchmark_info` or `out_seconds` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_seconds.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_time_to_first_token(
    const LiteRtLmBenchmarkInfo* benchmark_info, double* out_seconds);

// Returns the total initialization time in seconds.
//
// @param benchmark_info The benchmark info object.
// @param out_seconds On success, receives the total initialization time in
//   seconds; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `benchmark_info` or `out_seconds` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_seconds.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_total_init_time_in_second(
    const LiteRtLmBenchmarkInfo* benchmark_info, double* out_seconds);

// Returns the number of prefill turns.
//
// @param benchmark_info The benchmark info object.
// @param out_num_turns On success, receives the number of prefill turns; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `benchmark_info` or `out_num_turns` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_num_turns.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_num_prefill_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info, int* out_num_turns);

// Returns the number of decode turns.
//
// @param benchmark_info The benchmark info object.
// @param out_num_turns On success, receives the number of decode turns; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `benchmark_info` or `out_num_turns` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_num_turns.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_num_decode_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info, int* out_num_turns);

// Returns the prefill token count at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the prefill turn.
// @param out_token_count On success, receives the prefill token count; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `benchmark_info` or `out_token_count` is NULL; kLiteRtLmStatusOutOfRange
//   if `index` is out of bounds of the prefill turns.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_token_count.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_prefill_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    int* out_token_count);

// Returns the decode token count at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the decode turn.
// @param out_token_count On success, receives the decode token count; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `benchmark_info` or `out_token_count` is NULL; kLiteRtLmStatusOutOfRange
//   if `index` is out of bounds of the decode turns.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_token_count.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_decode_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    int* out_token_count);

// Returns the prefill tokens per second at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the prefill turn.
// @param out_tokens_per_sec On success, receives the prefill tokens per
//   second (0 if the turn has no measurable duration); not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `benchmark_info` or `out_tokens_per_sec` is NULL;
//   kLiteRtLmStatusOutOfRange if `index` is out of bounds of the prefill turns.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_tokens_per_sec.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    double* out_tokens_per_sec);

// Returns the decode tokens per second at a given turn index.
//
// @param benchmark_info The benchmark info object.
// @param index The index of the decode turn.
// @param out_tokens_per_sec On success, receives the decode tokens per second
//   (0 if the turn has no measurable duration); not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `benchmark_info` or `out_tokens_per_sec` is NULL;
//   kLiteRtLmStatusOutOfRange if `index` is out of bounds of the decode turns.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_tokens_per_sec.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    double* out_tokens_per_sec);

// Opaque pointer for LiteRT LM Stream Chunk.
// This object represents a single chunk of data returned during streaming.
// It is owned by the library and is only valid for the duration of the
// callback.
//
// Added in version 0.1.0.
typedef struct LiteRtLmStreamChunk LiteRtLmStreamChunk;

// Gets the text content of the chunk.
//
// @param chunk The stream chunk.
// @param out_text On success, receives the text content of the chunk, or NULL
//   if the chunk has no text content (e.g. if it is an error or final
//   metadata-only chunk). The string is owned by the chunk and is only valid
//   as long as the chunk is valid. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success (including when the chunk has no text),
//   or another LiteRtLmStatusCode on failure (see error_reporter.h).
//   kLiteRtLmStatusInvalidArgument if `chunk` or `out_text` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the text is returned
// through out_text.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_stream_chunk_get_text(
    const LiteRtLmStreamChunk* chunk, const char** out_text);

// Returns whether this is the final chunk of the stream.
//
// @param chunk The stream chunk.
// @param out_is_final On success, receives true if this is the final chunk of
//   the stream, false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `chunk`
//   or `out_is_final` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_is_final.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_stream_chunk_is_final(
    const LiteRtLmStreamChunk* chunk, bool* out_is_final);

// Gets the error message associated with this chunk, if any.
//
// @param chunk The stream chunk.
// @param out_error On success, receives the error message of the chunk, or
//   NULL if the chunk carries no error. The string is owned by the chunk and
//   is only valid as long as the chunk is valid. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success (including when the chunk carries no
//   error), or another LiteRtLmStatusCode on failure (see error_reporter.h).
//   kLiteRtLmStatusInvalidArgument if `chunk` or `out_error` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the message is returned
// through out_error.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_stream_chunk_get_error(
    const LiteRtLmStreamChunk* chunk, const char** out_error);

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
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `session`
//   or `callback` is NULL; otherwise the code of the error that prevented the
//   stream from starting. Errors that occur after the stream has started are
//   reported through the callback.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_run_decode_async(
    LiteRtLmSession* session, LiteRtLmStreamCallback callback,
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
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `session`
//   or `callback` is NULL; otherwise the code of the error that prevented the
//   stream from starting. Errors that occur after the stream has started are
//   reported through the callback.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_generate_content_stream(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, LiteRtLmStreamCallback callback, void* callback_data);

// Tokenizes text using the engine's tokenizer.
//
// @param engine The engine instance.
// @param text The UTF-8 string to tokenize.
// @param out_result On success, receives the tokenize result, owned by the
//   caller; release with `litert_lm_tokenize_result_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `engine`, `text` or `out_result` is NULL; otherwise the code of the
//   tokenizer error.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_result.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_tokenize(
    LiteRtLmEngine* engine, const char* text,
    LiteRtLmTokenizeResult** out_result);

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
// @param out_tokens On success, receives a pointer to the internal array of
//   token ids, whose length is given by
//   `litert_lm_tokenize_result_get_num_tokens`. The pointer is valid only for
//   the lifetime of the `result` object, and may be NULL if the result holds
//   no tokens. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `result`
//   or `out_tokens` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the array is returned
// through out_tokens.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_tokenize_result_get_tokens(
    const LiteRtLmTokenizeResult* result, const int** out_tokens);

// Returns the number of token ids from a tokenize result.
//
// @param result The tokenize result.
// @param out_num_tokens On success, receives the number of token ids; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `result`
//   or `out_num_tokens` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_num_tokens.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_tokenize_result_get_num_tokens(
    const LiteRtLmTokenizeResult* result, size_t* out_num_tokens);

// Detokenizes token ids using the engine's tokenizer.
//
// @param engine The engine instance.
// @param tokens An array of token ids to detokenize.
// @param num_tokens The number of token ids in the array.
// @param out_result On success, receives the detokenize result, owned by the
//   caller; release with `litert_lm_detokenize_result_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `engine`, `tokens` or `out_result` is NULL; otherwise the code of the
//   tokenizer error.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_result.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_detokenize(
    LiteRtLmEngine* engine, const int* tokens, size_t num_tokens,
    LiteRtLmDetokenizeResult** out_result);

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
// @param out_text On success, receives the detokenized UTF-8 string. The
//   string is owned by the `result` object and is valid only for its lifetime.
//   Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `result`
//   or `out_text` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the string is returned
// through out_text.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_detokenize_result_get_string(
    const LiteRtLmDetokenizeResult* result, const char** out_text);

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
// @param out_type On success, receives the type of the token union; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `token_union` or `out_type` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the type is returned
// through out_type.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_union_get_type(
    const LiteRtLmTokenUnion* token_union, LiteRtLmTokenUnionType* out_type);

// Returns the string value from a token union.
//
// @param token_union The token union.
// @param out_string On success, receives the string value. The string is owned
//   by the `token_union` object and is valid only for its lifetime. Set to
//   NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `token_union` or `out_string` is NULL, or if the type of `token_union` is
//   not kLiteRtLmTokenUnionTypeString.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the string is returned
// through out_string.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_union_get_string(
    const LiteRtLmTokenUnion* token_union, const char** out_string);

// Returns the token ids from a token union.
//
// @param token_union The token union.
// @param out_tokens On success, receives a pointer to the internal array of
//   token ids. The received pointer is valid only for the lifetime of the
//   `token_union` object. Set to NULL on failure.
// @param out_num_tokens On success, receives the number of token ids; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if any
//   argument is NULL or the type of `token_union` is not
//   kLiteRtLmTokenUnionTypeIds.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: failures return a canonical LiteRtLmStatusCode
// instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_union_get_ids(
    const LiteRtLmTokenUnion* token_union, const int** out_tokens,
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
// @param out_num_tokens On success, receives the number of token unions; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `tokens`
//   or `out_num_tokens` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_num_tokens.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_unions_get_num_tokens(
    const LiteRtLmTokenUnions* tokens, size_t* out_num_tokens);

// Returns a copy of the token union at a given index from a collection.
//
// @param tokens The token unions collection.
// @param index The index of the token union.
// @param out_token On success, receives a new copy of the token union at
//   `index`, owned by the caller; release with `litert_lm_token_union_delete`.
//   Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if `tokens`
//   or `out_token` is NULL; kLiteRtLmStatusOutOfRange if `index` is out of
//   bounds.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_token.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_unions_get_token_at(
    const LiteRtLmTokenUnions* tokens, size_t index,
    LiteRtLmTokenUnion** out_token);

// Returns the configured start token (BOS), if any.
//
// @param engine The engine instance.
// @param out_token On success, receives the start token, owned by the caller;
//   release with `litert_lm_token_union_delete`. Receives NULL on success if
//   the engine has no start token configured. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success (including when no start token is
//   configured), or another LiteRtLmStatusCode on failure (see
//   error_reporter.h). kLiteRtLmStatusInvalidArgument if `engine` or
//   `out_token` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_token.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_get_start_token(
    LiteRtLmEngine* engine, LiteRtLmTokenUnion** out_token);

// Returns the configured stop tokens (EOS).
//
// @param engine The engine instance.
// @param out_tokens On success, receives the stop tokens collection, owned by
//   the caller; release with `litert_lm_token_unions_delete`. Receives NULL on
//   success if the engine has no stop tokens configured. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success (including when no stop tokens are
//   configured), or another LiteRtLmStatusCode on failure (see
//   error_reporter.h). kLiteRtLmStatusInvalidArgument if `engine` or
//   `out_tokens` is NULL.
//
// Added in version 0.1.0.
// Changed in version 1.0.0: returns a status code; the result is returned
// through out_tokens.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_get_stop_tokens(
    LiteRtLmEngine* engine, LiteRtLmTokenUnions** out_tokens);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_H_
