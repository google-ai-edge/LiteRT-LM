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
#include "engine.h"  // NOLINT
#else
#include "c/engine.h"
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

// Creates LiteRT LM Embedding Engine Settings. The caller is responsible for
// destroying the settings using `litert_lm_embedding_engine_settings_delete`.
//
// @param model_path The path to the model file.
// @param backend_str The backend to use (e.g., "cpu", "gpu", "npu").
// @param vision_backend_str The vision backend to use, or NULL if not set.
// @param audio_backend_str The audio backend to use, or NULL if not set.
// @return A pointer to the created settings, or NULL on failure.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmEmbeddingEngineSettings* litert_lm_embedding_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str);

// Destroys LiteRT LM Embedding Engine Settings.
//
// @param settings The settings to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_delete(
    LiteRtLmEmbeddingEngineSettings* settings);

// Sets the number of threads for the audio CPU backend in Embedding Engine
// Settings.
//
// @param settings The embedding engine settings.
// @param num_threads The number of threads.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_set_audio_num_threads(
    LiteRtLmEmbeddingEngineSettings* settings, int num_threads);

// Sets the cache directory for the Embedding Engine.
//
// @param settings The embedding engine settings.
// @param cache_dir The cache directory.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_set_cache_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* cache_dir);

// Sets the LiteRT dispatch library directory for the main NPU backend.
//
// @param settings The embedding engine settings.
// @param lib_dir The dispatch library directory.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir);

// Sets the LiteRT dispatch library directory for the vision NPU backend.
//
// @param settings The embedding engine settings.
// @param lib_dir The dispatch library directory.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir);

// Sets the LiteRT dispatch library directory for the audio NPU backend.
//
// @param settings The embedding engine settings.
// @param lib_dir The dispatch library directory.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir);

// Sets the maximum sequence length (in tokens) for text encoder signatures in
// Embedding Engine Settings.
//
// @param settings The embedding engine settings.
// @param max_input_length The maximum input length. Passing a non-positive
//   value unsets the option.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_set_max_input_length(
    LiteRtLmEmbeddingEngineSettings* settings, int max_input_length);

// Sets the desired number of vision tokens generated per image in Embedding
// Engine Settings.
//
// @param settings The embedding engine settings.
// @param vision_tokens_per_image The vision tokens per image. Passing a
//   non-positive value unsets the option.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
    LiteRtLmEmbeddingEngineSettings* settings, int vision_tokens_per_image);

// Creates LiteRT LM Embedding Options with default values (`normalize = true`,
// `insert_special_tokens = true`). The caller is responsible for destroying
// options using `litert_lm_embedding_options_delete`.
//
// @return A pointer to the created options, or NULL on failure.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmEmbeddingOptions* litert_lm_embedding_options_create(void);

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
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_options_set_normalize(
    LiteRtLmEmbeddingOptions* options, bool normalize);

// Gets whether the embedding should be L2 normalized.
//
// @param options The options to inspect.
// @return True if normalization is enabled, false otherwise.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_embedding_options_get_normalize(
    const LiteRtLmEmbeddingOptions* options);

// Sets whether special tokens (BOS, EOS, start/end of image, start/end of
// audio) should be automatically inserted.
//
// @param options The options to modify.
// @param insert_special_tokens Whether to insert special tokens.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_options_set_insert_special_tokens(
    LiteRtLmEmbeddingOptions* options, bool insert_special_tokens);

// Gets whether special tokens should be automatically inserted.
//
// @param options The options to inspect.
// @return True if special tokens insertion is enabled, false otherwise.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_embedding_options_get_insert_special_tokens(
    const LiteRtLmEmbeddingOptions* options);

// Sets the input overflow strategy.
//
// @param options The options to modify.
// @param strategy The overflow strategy to use.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_options_set_input_overflow_strategy(
    LiteRtLmEmbeddingOptions* options, LiteRtLmInputOverflowStrategy strategy);

// Gets the input overflow strategy.
//
// @param options The options to inspect.
// @return The overflow strategy configured in options.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmInputOverflowStrategy
litert_lm_embedding_options_get_input_overflow_strategy(
    const LiteRtLmEmbeddingOptions* options);

// Sets the output embedding size to truncate the embedding to.
//
// @param options The options to modify.
// @param output_size The output embedding size to truncate to. Pass 0 or a
//   negative value (e.g., 0 or -1) to unset and use the default output
//   embedding size.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_options_set_output_size(
    LiteRtLmEmbeddingOptions* options, int output_size);

// Gets the output embedding size.
//
// @param options The options to inspect.
// @return The output embedding size, or -1 if not set (using default size).
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
int litert_lm_embedding_options_get_output_size(
    const LiteRtLmEmbeddingOptions* options);

// Sets the vision tokens per image.
//
// @param options The options to modify.
// @param vision_tokens_per_image The number of vision tokens per image. Passing
//   a non-positive value unsets the option.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_options_set_vision_tokens_per_image(
    LiteRtLmEmbeddingOptions* options, int vision_tokens_per_image);

// Gets the vision tokens per image.
//
// @param options The options to inspect.
// @return The vision tokens per image configured in options, or 0 if not set.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
int litert_lm_embedding_options_get_vision_tokens_per_image(
    const LiteRtLmEmbeddingOptions* options);
// Destroys a LiteRT LM Embedding Response.
//
// @param response The response to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_response_delete(LiteRtLmEmbeddingResponse* response);

// Returns the dimension (number of float values) of the embedding response.
//
// @param response The response to inspect.
// @return Number of float elements.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
size_t litert_lm_embedding_response_get_size(
    const LiteRtLmEmbeddingResponse* response);

// Returns a pointer to the array of float embedding values.
// The returned pointer is owned by `response` and valid for its lifetime.
//
// @param response The response to inspect.
// @return Pointer to float array, or NULL if empty.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
const float* litert_lm_embedding_response_get_values(
    const LiteRtLmEmbeddingResponse* response);

// Destroys a collection of LiteRT LM Embedding Responses.
//
// @param responses The responses collection to destroy.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_embedding_responses_delete(
    LiteRtLmEmbeddingResponses* responses);

// Returns the number of responses in the collection.
//
// @param responses The responses collection.
// @return The batch size.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
size_t litert_lm_embedding_responses_get_size(
    const LiteRtLmEmbeddingResponses* responses);

// Returns the embedding response at the given index in the batch.
// The returned pointer is owned by `responses` and valid for its lifetime.
//
// @param responses The responses collection.
// @param index The batch index.
// @return Pointer to the embedding response, or NULL if out of bounds.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
const LiteRtLmEmbeddingResponse* litert_lm_embedding_responses_get_at(
    const LiteRtLmEmbeddingResponses* responses, size_t index);

// Creates a LiteRT LM Embedding Engine from the given settings. The caller is
// responsible for destroying the engine using
// `litert_lm_embedding_engine_delete`.
//
// @param settings The embedding engine settings.
// @return A pointer to the created engine, or NULL on failure.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmEmbeddingEngine* litert_lm_embedding_engine_create(
    const LiteRtLmEmbeddingEngineSettings* settings);

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
// @return A pointer to the embedding response, or NULL on failure. The caller
//   is responsible for deleting the response using
//   `litert_lm_embedding_response_delete`.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmEmbeddingResponse* litert_lm_embedding_engine_compute_embedding(
    LiteRtLmEmbeddingEngine* engine, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, const LiteRtLmEmbeddingOptions* options);

// Computes embedding responses for a batch of requests.
//
// @param engine The embedding engine.
// @param inputs_batch An array of arrays of LiteRtLmInputData pointers.
// @param num_inputs_per_batch An array specifying the number of inputs for each
//   request in the batch.
// @param batch_size The number of requests in the batch.
// @param options Optional embedding options. If NULL, default options are
//   used.
// @return A pointer to the batch responses, or NULL on failure. The caller is
//   responsible for deleting responses using
//   `litert_lm_embedding_responses_delete`.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmEmbeddingResponses* litert_lm_embedding_engine_compute_embedding_batch(
    LiteRtLmEmbeddingEngine* engine,
    const LiteRtLmInputData* const* const* inputs_batch,
    const size_t* num_inputs_per_batch, size_t batch_size,
    const LiteRtLmEmbeddingOptions* options);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_EMBEDDING_ENGINE_H_
