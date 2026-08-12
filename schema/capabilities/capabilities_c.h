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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_C_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_C_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct representing a loaded LiteRT-LM file for capability checks.
typedef struct LiteRtLmLoadedFile LiteRtLmLoadedFile;

// Loads a LiteRT-LM file from the given path for capability queries.
// Returns NULL if the file cannot be opened.
LiteRtLmLoadedFile* litert_lm_loaded_file_create(const char* litertlm_path);

// Deletes a loaded LiteRT-LM file.
void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file);

// Returns true if the loaded LiteRT-LM file supports speculative decoding.
bool litert_lm_loaded_file_has_speculative_decoding_support(
    LiteRtLmLoadedFile* loaded_file);

// Returns true if the loaded LiteRT-LM file supports vision input.
bool litert_lm_loaded_file_has_vision_support(LiteRtLmLoadedFile* loaded_file);

// Returns true if the loaded LiteRT-LM file supports audio input.
bool litert_lm_loaded_file_has_audio_support(LiteRtLmLoadedFile* loaded_file);

// Returns true if the loaded LiteRT-LM file supports function calling.
bool litert_lm_loaded_file_has_function_calling_support(
    LiteRtLmLoadedFile* loaded_file);

// Returns true if the loaded LiteRT-LM file supports thinking/reasoning.
bool litert_lm_loaded_file_has_thinking_support(
    LiteRtLmLoadedFile* loaded_file);

// Returns maximum context length for the model (-1 if not specified or error).
int litert_lm_loaded_file_get_max_context_length(
    LiteRtLmLoadedFile* loaded_file);

// Returns the model class (e.g. "IT", "PT"), or NULL if not specified.
const char* litert_lm_loaded_file_get_model_class(
    LiteRtLmLoadedFile* loaded_file);

// Returns the TF Hub model ID, or NULL if not specified.
const char* litert_lm_loaded_file_get_tf_hub_model_id(
    LiteRtLmLoadedFile* loaded_file);

// Returns the minimum required LiteRT-LM version, or NULL if not specified.
const char* litert_lm_loaded_file_get_min_litertlm_version(
    LiteRtLmLoadedFile* loaded_file);

// Returns the default temperature (e.g. 0.8f, or -1.0f if not set).
float litert_lm_loaded_file_get_default_temperature(
    LiteRtLmLoadedFile* loaded_file);

// Returns the default top_k (e.g. 40, or -1 if not set).
int litert_lm_loaded_file_get_default_top_k(LiteRtLmLoadedFile* loaded_file);

// Returns the default top_p (e.g. 0.95f, or -1.0f if not set).
float litert_lm_loaded_file_get_default_top_p(LiteRtLmLoadedFile* loaded_file);

// Returns the number of supported hardware backends.
int litert_lm_loaded_file_get_supported_backend_count(
    LiteRtLmLoadedFile* loaded_file);

// Returns the backend name at index (e.g. "cpu", "gpu"), or NULL if index out of range.
const char* litert_lm_loaded_file_get_supported_backend(
    LiteRtLmLoadedFile* loaded_file, int index);

// Returns the number of supported vision resolutions (e.g. 70, 140, 270, 560, 1120).
int litert_lm_loaded_file_get_supported_vision_resolution_count(
    LiteRtLmLoadedFile* loaded_file);

// Returns the vision resolution at index, or -1 if out of range.
int litert_lm_loaded_file_get_supported_vision_resolution(
    LiteRtLmLoadedFile* loaded_file, int index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_C_H_
