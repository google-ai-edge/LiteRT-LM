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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_MODEL_INFO_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_MODEL_INFO_H_

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

// Returns the maximum supported context tokens for the loaded LiteRT-LM file.
// - If the model is static (litert_lm_loaded_file_is_dynamic_context is
//   false), this is the fixed context size determined by the model graph.
// - If the model is dynamic (litert_lm_loaded_file_is_dynamic_context is
//   true), this is the largest context size that can be set.
// Returns 0 if not found or on error.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
uint32_t litert_lm_loaded_file_max_context_tokens(
    LiteRtLmLoadedFile* loaded_file);

// Returns whether the model has dynamic context.
// Dynamic context means the context size can be configured by the caller
// up to the maximum limit.
//
// @param loaded_file The loaded file handle.
// @return True if the model has dynamic context, false otherwise.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
bool litert_lm_loaded_file_is_dynamic_context(LiteRtLmLoadedFile* loaded_file);

// Returns the number of supported vision token lengths.
// Writes up to `max_size` lengths to the provided array.
// If lengths is NULL, only returns the count.
// Returns -1 if the model does not support vision.
//
// Added in version 0.3.0.
LITERT_LM_C_API_EXPORT
int32_t litert_lm_loaded_file_vision_signature_selection(
    LiteRtLmLoadedFile* loaded_file, int32_t* lengths, int32_t max_size);

// Hardware backend type.
//
// Added in version 0.3.0.
typedef enum LiteRtLmBackendType {
  kLiteRtLmBackendTypeCpu = 1,
  kLiteRtLmBackendTypeGpu = 2,
  kLiteRtLmBackendTypeNpu = 3,
} LiteRtLmBackendType;

// Returns the number of supported backends for a given modality, ordered by
// priority (first entry is the default/highest-priority backend).
// Writes up to `max_size` backends to the provided `backends` array.
// If `backends` is NULL, only returns the count of supported backends.
// Returns 0 if the modality is not supported.
//
// Added in version 0.3.0.
LITERT_LM_C_API_EXPORT
int32_t litert_lm_loaded_file_modality_supported_backends(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    LiteRtLmBackendType* backends, int32_t max_size);

// NPU brand options.
//
// Added in version 0.3.0.
typedef enum LiteRtLmNpuBrand {
  kLiteRtLmNpuBrandUnknown = 0,
  kLiteRtLmNpuBrandQualcomm = 1,
  kLiteRtLmNpuBrandGoogleTensor = 2,
  kLiteRtLmNpuBrandMediaTek = 3,
  kLiteRtLmNpuBrandIntel = 4,
  kLiteRtLmNpuBrandSamsung = 5,
} LiteRtLmNpuBrand;

// Returns the detected NPU brand of the model for a given modality, or
// kLiteRtLmNpuBrandUnknown if not NPU-compiled for this modality.
//
// Added in version 0.3.0.
LITERT_LM_C_API_EXPORT
LiteRtLmNpuBrand litert_lm_loaded_file_modality_npu_brand(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality);

// Returns the target SoC name for a given modality (e.g. "SM8750",
// "Tensor_G5"), or NULL if not specified or not NPU-compiled.
// The returned pointer is valid as long as the loaded_file is valid.
//
// Added in version 0.3.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_loaded_file_modality_soc_name(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality);

// Returns the minimum LiteRT-LM runtime version required to run this model.
// The returned pointer is valid as long as the loaded_file is valid.
// Returns NULL if the version requirement is not defined.
//
// Added in version 0.3.0.
LITERT_LM_C_API_EXPORT
const char* litert_lm_loaded_file_min_runtime_version(
    LiteRtLmLoadedFile* loaded_file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_MODEL_INFO_H_
