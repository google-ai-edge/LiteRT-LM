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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_CAPABILITIES_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_CAPABILITIES_H_

#include <stdbool.h>
#include <stdint.h>

#if defined(__APPLE__)
#include "engine.h"  // NOLINT
#else
#include "c/engine.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct representing a loaded LiteRT-LM file for capability checks.
//
// Added in version 0.2.0.
typedef struct LiteRtLmLoadedFile LiteRtLmLoadedFile;

// Input and output modalities supported by LiteRT-LM models.
//
// Added in version 0.2.0.
typedef enum LiteRtLmModality {
  kLiteRtLmModalityText = 0,
  kLiteRtLmModalityVision = 1,
  kLiteRtLmModalityAudio = 2,
  kLiteRtLmModalityVideo = 3,
} LiteRtLmModality;

// Loads a LiteRT-LM file from the given path for capability queries.
// Returns NULL if the file cannot be opened.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmLoadedFile* litert_lm_loaded_file_create(const char* litertlm_path);

// Deletes a loaded LiteRT-LM file.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file);

// Returns true if the loaded LiteRT-LM file supports speculative decoding.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_loaded_file_has_speculative_decoding_support(
    LiteRtLmLoadedFile* loaded_file);

// Returns true if the model supports thinking / reasoning steps.
// If the metadata is not explicitly set in the model, this returns false.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_loaded_file_supports_thinking(LiteRtLmLoadedFile* loaded_file);

// Returns true if the model supports function calling / tool use.
// If the metadata is not explicitly set in the model, this returns false.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_loaded_file_supports_function_calling(
    LiteRtLmLoadedFile* loaded_file);

// Returns the default sampler type for the model.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
LiteRtLmSamplerType litert_lm_loaded_file_sampler_type(
    LiteRtLmLoadedFile* loaded_file);

// Returns the default sampler temperature for the model.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
float litert_lm_loaded_file_sampler_temperature(
    LiteRtLmLoadedFile* loaded_file);

// Returns the default sampler top_k for the model.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
int32_t litert_lm_loaded_file_sampler_top_k(LiteRtLmLoadedFile* loaded_file);

// Returns the default sampler top_p for the model.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
float litert_lm_loaded_file_sampler_top_p(LiteRtLmLoadedFile* loaded_file);

// Returns true if the input modality is supported.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_loaded_file_supports_input_modality(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality);

// Returns the maximum vision token budget for the model.
// Returns -1 if the model does not support vision or if the budget is not
// defined.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
int32_t litert_lm_loaded_file_max_vision_token_budget(
    LiteRtLmLoadedFile* loaded_file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_CAPABILITIES_H_
