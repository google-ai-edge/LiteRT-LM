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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_EMBEDDING_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_EMBEDDING_ENGINE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__)
#include "api_export.h"      // NOLINT
#include "engine.h"          // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/api_export.h"
#include "c/engine.h"
#include "c/error_reporter.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer for the LiteRT LM Embedding Engine.
//
// Added in version 0.2.0.
typedef struct LiteRtLmEmbeddingEngine LiteRtLmEmbeddingEngine;

// Opaque pointer for the LiteRT LM Embedding Engine Settings.
//
// Added in version 0.2.0.
typedef struct LiteRtLmEmbeddingEngineSettings LiteRtLmEmbeddingEngineSettings;

// Strategy for handling inputs longer than the maximum supported signature
// length.
//
// Added in version 0.2.0.
typedef enum {
  // Chunks the input into sub-sequences, embeds each chunk, and returns the
  // mean embedding across all chunks.
  kLiteRtLmInputOverflowStrategyChunkAndAverage = 0,
  // Truncates the input to the longest signature length.
  kLiteRtLmInputOverflowStrategyTruncate = 1,
  // Returns an error status if the input exceeds the longest signature length.
  kLiteRtLmInputOverflowStrategyError = 2,
} LiteRtLmInputOverflowStrategy;

// Opaque pointer for the LiteRT LM Embedding Options.
//
// Added in version 0.2.0.
typedef struct LiteRtLmEmbeddingOptions LiteRtLmEmbeddingOptions;

// Opaque pointer for a single LiteRT LM Embedding Response.
//
// Added in version 0.2.0.
typedef struct LiteRtLmEmbeddingResponse LiteRtLmEmbeddingResponse;

// Opaque pointer for a collection of LiteRT LM Embedding Responses (batch).
//
// Added in version 0.2.0.
typedef struct LiteRtLmEmbeddingResponses LiteRtLmEmbeddingResponses;

// Creates LiteRT LM Embedding Engine Settings.
//
// @param model_path The path to the model file.
// @param backend_str The backend to use (e.g., "cpu", "gpu", "npu").
// @param vision_backend_str The vision backend to use, or NULL if not set.
// @param audio_backend_str The audio backend to use, or NULL if not set.
// @param out_settings On success, receives the created settings, owned by the
//   caller; release with `litert_lm_embedding_engine_settings_delete`. Set to
//   NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `model_path`, `backend_str` or `out_settings` is NULL, or a backend string
//   is not recognized; otherwise the code of the error that prevented the
//   settings from being created (e.g. the model file could not be opened).
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the settings are returned
// through out_settings.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str,
    LiteRtLmEmbeddingEngineSettings** out_settings);

// Destroys LiteRT LM Embedding Engine Settings.
//
// @param settings The settings to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_delete(
    LiteRtLmEmbeddingEngineSettings* settings);

// Sets the number of threads for the CPU backend in Embedding Engine Settings.
//
// @param settings The embedding engine settings.
// @param num_threads The number of threads.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If `num_threads` is not positive, this call has no
//   effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_num_threads(
    LiteRtLmEmbeddingEngineSettings* settings, int num_threads);

// Sets the number of threads for the audio CPU backend in Embedding Engine
// Settings.
//
// @param settings The embedding engine settings.
// @param num_threads The number of threads.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL. If `num_threads` is not positive or no audio backend is
//   configured, this call has no effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_audio_num_threads(
    LiteRtLmEmbeddingEngineSettings* settings, int num_threads);

// Sets the cache directory for the Embedding Engine.
//
// @param settings The embedding engine settings.
// @param cache_dir The cache directory.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` or `cache_dir` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_cache_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* cache_dir);

// Sets the LiteRT dispatch library directory for the main NPU backend.
//
// @param settings The embedding engine settings.
// @param lib_dir The dispatch library directory.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` or `lib_dir` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir);

// Sets the LiteRT dispatch library directory for the vision NPU backend.
//
// @param settings The embedding engine settings.
// @param lib_dir The dispatch library directory.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` or `lib_dir` is NULL. If no vision backend is configured, this
//   call has no effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir);

// Sets the LiteRT dispatch library directory for the audio NPU backend.
//
// @param settings The embedding engine settings.
// @param lib_dir The dispatch library directory.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` or `lib_dir` is NULL. If no audio backend is configured, this
//   call has no effect and returns kLiteRtLmStatusOk.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir);

// Sets the maximum sequence length (in tokens) for text encoder signatures in
// Embedding Engine Settings.
//
// @param settings The embedding engine settings.
// @param max_input_length The maximum input length. Passing a non-positive
//   value unsets the option.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_max_input_length(
    LiteRtLmEmbeddingEngineSettings* settings, int max_input_length);

// Sets the minimum sequence length (in tokens) for text encoder signatures in
// Embedding Engine Settings.
//
// @param settings The embedding engine settings.
// @param min_input_length The minimum input length. Passing a negative
//   value unsets the option.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_min_input_length(
    LiteRtLmEmbeddingEngineSettings* settings, int min_input_length);

// Sets the desired number of vision tokens generated per image in Embedding
// Engine Settings.
//
// @param settings The embedding engine settings.
// @param vision_tokens_per_image The vision tokens per image. Passing a
//   non-positive value unsets the option.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
    LiteRtLmEmbeddingEngineSettings* settings, int vision_tokens_per_image);

// Sets the activation data type for the embedding engine settings.
//
// @param settings The embedding engine settings.
// @param activation_data_type The activation data type (FLOAT32, FLOAT16,
// etc.).
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `settings` is NULL or `activation_data_type` is not a declared
//   LiteRtLmActivationDataType value.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_activation_data_type(
    LiteRtLmEmbeddingEngineSettings* settings,
    LiteRtLmActivationDataType activation_data_type);

// Creates LiteRT LM Embedding Options with default values (`normalize = true`,
// `insert_special_tokens = true`).
//
// @param out_options On success, receives the created options, owned by the
//   caller; release with `litert_lm_embedding_options_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_options` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the options are returned
// through out_options.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_create(
    LiteRtLmEmbeddingOptions** out_options);

// Destroys LiteRT LM Embedding Options.
//
// @param options The options to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_options_delete(LiteRtLmEmbeddingOptions* options);

// Sets whether the embedding should be L2 normalized.
//
// @param options The options to modify.
// @param normalize Whether to normalize.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_set_normalize(
    LiteRtLmEmbeddingOptions* options, bool normalize);

// Gets whether the embedding should be L2 normalized.
//
// @param options The options to inspect.
// @param out_normalize On success, receives true if normalization is enabled,
//   false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `options` or `out_normalize` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_normalize.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_get_normalize(
    const LiteRtLmEmbeddingOptions* options, bool* out_normalize);

// Sets whether special tokens (BOS, EOS, start/end of image, start/end of
// audio) should be automatically inserted.
//
// @param options The options to modify.
// @param insert_special_tokens Whether to insert special tokens.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_set_insert_special_tokens(
    LiteRtLmEmbeddingOptions* options, bool insert_special_tokens);

// Gets whether special tokens should be automatically inserted.
//
// @param options The options to inspect.
// @param out_insert_special_tokens On success, receives true if special tokens
//   insertion is enabled, false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `options` or `out_insert_special_tokens` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_insert_special_tokens.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_get_insert_special_tokens(
    const LiteRtLmEmbeddingOptions* options, bool* out_insert_special_tokens);

// Sets the input overflow strategy.
//
// @param options The options to modify.
// @param strategy The overflow strategy to use.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL or `strategy` is not a declared
//   LiteRtLmInputOverflowStrategy value.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_set_input_overflow_strategy(
    LiteRtLmEmbeddingOptions* options, LiteRtLmInputOverflowStrategy strategy);

// Gets the input overflow strategy.
//
// @param options The options to inspect.
// @param out_strategy On success, receives the overflow strategy configured in
//   options; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `options` or `out_strategy` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the strategy is returned
// through out_strategy.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_get_input_overflow_strategy(
    const LiteRtLmEmbeddingOptions* options,
    LiteRtLmInputOverflowStrategy* out_strategy);

// Sets the output embedding size to truncate the embedding to.
//
// @param options The options to modify.
// @param output_size The output embedding size to truncate to. Pass 0 or a
//   negative value (e.g., 0 or -1) to unset and use the default output
//   embedding size.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_set_output_size(
    LiteRtLmEmbeddingOptions* options, int output_size);

// Gets the output embedding size.
//
// @param options The options to inspect.
// @param out_output_size On success, receives the output embedding size; not
//   written on failure or if the size is not set.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `options` or `out_output_size` is NULL; kLiteRtLmStatusNotFound if the
//   output size is not set (the default output embedding size is used).
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the size is returned
// through out_output_size, and an unset size is reported as
// kLiteRtLmStatusNotFound instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_get_output_size(
    const LiteRtLmEmbeddingOptions* options, int* out_output_size);

// Sets the vision tokens per image.
//
// @param options The options to modify.
// @param vision_tokens_per_image The number of vision tokens per image. Passing
//   a non-positive value unsets the option.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `options` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code instead of void.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_set_vision_tokens_per_image(
    LiteRtLmEmbeddingOptions* options, int vision_tokens_per_image);

// Gets the vision tokens per image.
//
// @param options The options to inspect.
// @param out_vision_tokens_per_image On success, receives the vision tokens per
//   image configured in options; not written on failure or if the value is not
//   set.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `options` or `out_vision_tokens_per_image` is NULL;
//   kLiteRtLmStatusNotFound if the vision tokens per image is not set (the
//   engine or model default is used).
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_vision_tokens_per_image, and an unset value is reported as
// kLiteRtLmStatusNotFound instead of 0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_options_get_vision_tokens_per_image(
    const LiteRtLmEmbeddingOptions* options, int* out_vision_tokens_per_image);

// Destroys a LiteRT LM Embedding Response.
//
// @param response The response to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_response_delete(LiteRtLmEmbeddingResponse* response);

// Gets the dimension (number of float values) of the embedding response.
//
// @param response The response to inspect.
// @param out_size On success, receives the number of float elements; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `response` or `out_size` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the size is returned
// through out_size.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_response_get_size(
    const LiteRtLmEmbeddingResponse* response, size_t* out_size);

// Gets a pointer to the array of float embedding values.
//
// @param response The response to inspect.
// @param out_values On success, receives a pointer to the
//   `litert_lm_embedding_response_get_size` float values, or NULL if the
//   embedding is empty. The array is owned by `response` and valid until
//   `response` (or the `LiteRtLmEmbeddingResponses` that owns it) is deleted.
//   Set to NULL on failure.
// @return kLiteRtLmStatusOk on success (including an empty embedding), or
//   another LiteRtLmStatusCode on failure (see error_reporter.h).
//   kLiteRtLmStatusInvalidArgument if `response` or `out_values` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the values are returned
// through out_values.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_response_get_values(
    const LiteRtLmEmbeddingResponse* response, const float** out_values);

// Destroys a collection of LiteRT LM Embedding Responses.
//
// @param responses The responses collection to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_responses_delete(
    LiteRtLmEmbeddingResponses* responses);

// Gets the number of responses in the collection.
//
// @param responses The responses collection.
// @param out_size On success, receives the batch size; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_size` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the size is returned
// through out_size.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_responses_get_size(
    const LiteRtLmEmbeddingResponses* responses, size_t* out_size);

// Gets the embedding response at the given index in the batch.
//
// @param responses The responses collection.
// @param index The batch index.
// @param out_response On success, receives the embedding response at `index`.
//   It is owned by `responses` and valid until `responses` is deleted; do not
//   delete it. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `responses` or `out_response` is NULL; kLiteRtLmStatusOutOfRange if
//   `index` is not less than the batch size.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the response is returned
// through out_response, and an out-of-range index is reported as
// kLiteRtLmStatusOutOfRange.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_responses_get_at(
    const LiteRtLmEmbeddingResponses* responses, size_t index,
    const LiteRtLmEmbeddingResponse** out_response);

// Creates a LiteRT LM Embedding Engine from the given settings.
//
// @param settings The embedding engine settings.
// @param out_engine On success, receives the created engine, owned by the
//   caller; release with `litert_lm_embedding_engine_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `settings` or `out_engine` is NULL; otherwise the code of the error that
//   prevented the engine from being created.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the engine is returned
// through out_engine.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_create(
    const LiteRtLmEmbeddingEngineSettings* settings,
    LiteRtLmEmbeddingEngine** out_engine);

// Destroys a LiteRT LM Embedding Engine.
//
// @param engine The engine to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_delete(LiteRtLmEmbeddingEngine* engine);

// Computes embedding response for a single request.
//
// @param engine The embedding engine.
// @param inputs Array of LiteRtLmInputData pointers representing multimodal
//   input.
// @param num_inputs Number of inputs in the array.
// @param options Optional embedding options. If NULL, default options are
//   used.
// @param out_response On success, receives the embedding response, owned by
//   the caller; release with `litert_lm_embedding_response_delete`. Set to NULL
//   on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `engine` or `out_response` is NULL; otherwise the code of the error that
//   prevented the embedding from being computed.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the response is returned
// through out_response.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_compute_embedding(
    LiteRtLmEmbeddingEngine* engine, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, const LiteRtLmEmbeddingOptions* options,
    LiteRtLmEmbeddingResponse** out_response);

// Computes embedding responses for a batch of requests.
//
// @param engine The embedding engine.
// @param inputs_batch An array of arrays of LiteRtLmInputData pointers.
// @param num_inputs_per_batch An array specifying the number of inputs for each
//   request in the batch.
// @param batch_size The number of requests in the batch.
// @param options Optional embedding options. If NULL, default options are
//   used.
// @param out_responses On success, receives the batch responses, owned by the
//   caller; release with `litert_lm_embedding_responses_delete`. Set to NULL
//   on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `engine` or `out_responses` is NULL; otherwise the code of the error that
//   prevented the embeddings from being computed.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the responses are returned
// through out_responses.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_embedding_engine_compute_embedding_batch(
    LiteRtLmEmbeddingEngine* engine,
    const LiteRtLmInputData* const* const* inputs_batch,
    const size_t* num_inputs_per_batch, size_t batch_size,
    const LiteRtLmEmbeddingOptions* options,
    LiteRtLmEmbeddingResponses** out_responses);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_EMBEDDING_ENGINE_H_
