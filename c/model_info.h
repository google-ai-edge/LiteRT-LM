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
#include "engine.h"          // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/engine.h"
#include "c/error_reporter.h"  // IWYU pragma: export
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
//
// @param litertlm_path The path to the .litertlm file.
// @param out_loaded_file On success, receives the loaded file, owned by the
//   caller; release with `litert_lm_loaded_file_delete`. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `litertlm_path` or `out_loaded_file` is NULL; otherwise the code of the
//   error that prevented the file from being loaded (e.g. the file could not
//   be opened or parsed).
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the loaded file is returned
// through out_loaded_file.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_create(
    const char* litertlm_path, LiteRtLmLoadedFile** out_loaded_file);

// Deletes a loaded LiteRT-LM file.
//
// Added in version 0.2.0.
LITERT_LM_C_API_EXPORT
void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file);

// Gets whether the loaded LiteRT-LM file supports speculative decoding.
//
// @param loaded_file The loaded file to inspect.
// @param out_supported On success, receives true if the model supports
//   speculative decoding, false otherwise (including non-LLM models); not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_supported` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_supported.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_has_speculative_decoding_support(
    const LiteRtLmLoadedFile* loaded_file, bool* out_supported);

// Gets whether the model supports thinking / reasoning steps.
//
// @param loaded_file The loaded file to inspect.
// @param out_supported On success, receives true if the model supports
//   thinking, false otherwise (including when the metadata is not explicitly
//   set in the model, or the model is not an LLM); not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_supported` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_supported.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_supports_thinking(
    const LiteRtLmLoadedFile* loaded_file, bool* out_supported);

// Gets whether the model supports function calling / tool use.
//
// @param loaded_file The loaded file to inspect.
// @param out_supported On success, receives true if the model supports
//   function calling, false otherwise (including when the metadata is not
//   explicitly set in the model, or the model is not an LLM); not written on
//   failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_supported` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_supported.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_supports_function_calling(
    const LiteRtLmLoadedFile* loaded_file, bool* out_supported);

// Gets the default sampler type for the model.
//
// @param loaded_file The loaded file to inspect.
// @param out_sampler_type On success, receives the default sampler type. This
//   is kLiteRtLmSamplerTypeUnspecified if the LLM metadata does not specify a
//   sampler type. Not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_sampler_type` is NULL; kLiteRtLmStatusNotFound if
//   the model is not an LLM (it has no default sampler parameters).
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_sampler_type, and a model without LLM metadata is reported as
// kLiteRtLmStatusNotFound instead of kLiteRtLmSamplerTypeUnspecified.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_sampler_type(
    const LiteRtLmLoadedFile* loaded_file,
    LiteRtLmSamplerType* out_sampler_type);

// Gets the default sampler temperature for the model.
//
// @param loaded_file The loaded file to inspect.
// @param out_temperature On success, receives the default temperature as
//   stored in the LLM metadata (0 if the metadata does not specify it); not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_temperature` is NULL; kLiteRtLmStatusNotFound if the
//   model is not an LLM (it has no default sampler parameters).
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_temperature, and a model without LLM metadata is reported as
// kLiteRtLmStatusNotFound instead of 0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_sampler_temperature(
    const LiteRtLmLoadedFile* loaded_file, float* out_temperature);

// Gets the default sampler top_k for the model.
//
// @param loaded_file The loaded file to inspect.
// @param out_top_k On success, receives the default top_k as stored in the LLM
//   metadata (0 if the metadata does not specify it); not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_top_k` is NULL; kLiteRtLmStatusNotFound if the model
//   is not an LLM (it has no default sampler parameters).
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_top_k, and a model without LLM metadata is reported as
// kLiteRtLmStatusNotFound instead of 0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_sampler_top_k(
    const LiteRtLmLoadedFile* loaded_file, int32_t* out_top_k);

// Gets the default sampler top_p for the model.
//
// @param loaded_file The loaded file to inspect.
// @param out_top_p On success, receives the default top_p as stored in the LLM
//   metadata (0 if the metadata does not specify it); not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_top_p` is NULL; kLiteRtLmStatusNotFound if the model
//   is not an LLM (it has no default sampler parameters).
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_top_p, and a model without LLM metadata is reported as
// kLiteRtLmStatusNotFound instead of 0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_sampler_top_p(
    const LiteRtLmLoadedFile* loaded_file, float* out_top_p);

// Gets whether the input modality is supported.
//
// @param loaded_file The loaded file to inspect.
// @param modality The input modality to check.
// @param out_supported On success, receives true if `modality` is a supported
//   input modality, false otherwise; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_supported` is NULL, or `modality` is not a valid
//   LiteRtLmModality.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_supported.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_supports_input_modality(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    bool* out_supported);

// Gets the maximum vision token budget for the model.
//
// @param loaded_file The loaded file to inspect.
// @param out_budget On success, receives the maximum vision token budget; not
//   written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_budget` is NULL; kLiteRtLmStatusNotFound if the
//   model does not support vision or does not define the budget.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_budget, and an undefined budget is reported as
// kLiteRtLmStatusNotFound instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_max_vision_token_budget(
    const LiteRtLmLoadedFile* loaded_file, int32_t* out_budget);

// Gets the maximum supported context tokens for the loaded LiteRT-LM file.
// - If the model is static (litert_lm_loaded_file_is_dynamic_context reports
//   false), this is the fixed context size determined by the model graph.
// - If the model is dynamic (litert_lm_loaded_file_is_dynamic_context reports
//   true), this is the largest context size that can be set.
//
// @param loaded_file The loaded file to inspect.
// @param out_max_context_tokens On success, receives the maximum number of
//   context tokens (always positive); not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_max_context_tokens` is NULL; kLiteRtLmStatusNotFound
//   if the model does not define its maximum context size.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_max_context_tokens, and an undefined context size is reported as
// kLiteRtLmStatusNotFound instead of 0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_max_context_tokens(
    const LiteRtLmLoadedFile* loaded_file, uint32_t* out_max_context_tokens);

// Gets whether the model has dynamic context.
// Dynamic context means the context size can be configured by the caller
// up to the maximum limit.
//
// @param loaded_file The loaded file to inspect.
// @param out_is_dynamic On success, receives true if the model has dynamic
//   context, false otherwise (including when the model has neither LLM nor
//   embedding metadata); not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_is_dynamic` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_is_dynamic.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_is_dynamic_context(
    const LiteRtLmLoadedFile* loaded_file, bool* out_is_dynamic);

// Gets the supported vision token lengths (the discrete vision token budgets
// supported by each vision signature).
//
// Writes up to `max_size` lengths to `lengths`. To query only the count, pass
// `lengths` = NULL (and `max_size` = 0).
//
// @param loaded_file The loaded file to inspect.
// @param lengths Buffer that receives up to `max_size` lengths on success, or
//   NULL to query only the count. Not written on failure.
// @param max_size The capacity of `lengths`, in elements. Must not be
//   negative.
// @param out_count On success, receives the total number of supported lengths,
//   which may exceed `max_size`; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_count` is NULL, or `max_size` is negative;
//   kLiteRtLmStatusNotFound if the model does not support vision or does not
//   define vision signature lengths.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_count, and undefined lengths are reported as
// kLiteRtLmStatusNotFound instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_vision_signature_selection(
    const LiteRtLmLoadedFile* loaded_file, int32_t* lengths, int32_t max_size,
    int32_t* out_count);

// Hardware backend type.
//
// Added in version 0.2.0.
typedef enum LiteRtLmBackendType {
  kLiteRtLmBackendTypeCpu = 1,
  kLiteRtLmBackendTypeGpu = 2,
  kLiteRtLmBackendTypeNpu = 3,
} LiteRtLmBackendType;

// Gets the supported backends for a given modality, ordered by priority (first
// entry is the default/highest-priority backend).
//
// Writes up to `max_size` backends to `backends`. To query only the count, pass
// `backends` = NULL (and `max_size` = 0).
//
// @param loaded_file The loaded file to inspect.
// @param modality The modality to query.
// @param backends Buffer that receives up to `max_size` backends on success,
//   or NULL to query only the count. Not written on failure.
// @param max_size The capacity of `backends`, in elements. Must not be
//   negative.
// @param out_count On success, receives the total number of supported
//   backends, which may exceed `max_size`; 0 if the modality is not supported.
//   Not written on failure.
// @return kLiteRtLmStatusOk on success (including an unsupported modality), or
//   another LiteRtLmStatusCode on failure (see error_reporter.h).
//   kLiteRtLmStatusInvalidArgument if `loaded_file` or `out_count` is NULL,
//   `modality` is not a valid LiteRtLmModality, or `max_size` is negative.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_count.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_modality_supported_backends(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    LiteRtLmBackendType* backends, int32_t max_size, int32_t* out_count);

// NPU brand options.
//
// Added in version 0.2.0.
typedef enum LiteRtLmNpuBrand {
  kLiteRtLmNpuBrandUnknown = 0,
  kLiteRtLmNpuBrandQualcomm = 1,
  kLiteRtLmNpuBrandGoogleTensor = 2,
  kLiteRtLmNpuBrandMediaTek = 3,
  kLiteRtLmNpuBrandIntel = 4,
  kLiteRtLmNpuBrandSamsung = 5,
} LiteRtLmNpuBrand;

// Gets the detected NPU brand of the model for a given modality.
//
// @param loaded_file The loaded file to inspect.
// @param modality The modality to query.
// @param out_npu_brand On success, receives the detected NPU brand, or
//   kLiteRtLmNpuBrandUnknown if the model is not NPU-compiled for `modality`
//   or its NPU brand could not be detected. Not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_npu_brand` is NULL, or `modality` is not a valid
//   LiteRtLmModality.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_npu_brand.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_modality_npu_brand(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    LiteRtLmNpuBrand* out_npu_brand);

// Gets the target SoC name for a given modality (e.g. "SM8750", "Tensor_G5").
//
// @param loaded_file The loaded file to inspect.
// @param modality The modality to query.
// @param out_soc_name On success, receives the NUL-terminated SoC name, or
//   NULL if the model does not specify one for `modality` (e.g. it is not
//   NPU-compiled). The string is owned by `loaded_file` and valid until it is
//   deleted. Set to NULL on failure.
// @return kLiteRtLmStatusOk on success (including when no SoC name is
//   specified), or another LiteRtLmStatusCode on failure (see
//   error_reporter.h). kLiteRtLmStatusInvalidArgument if `loaded_file` or
//   `out_soc_name` is NULL, or `modality` is not a valid LiteRtLmModality.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the name is returned
// through out_soc_name.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_modality_soc_name(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    const char** out_soc_name);

// Gets the minimum LiteRT-LM runtime version required to run this model.
//
// @param loaded_file The loaded file to inspect.
// @param out_version On success, receives the NUL-terminated version string,
//   or NULL if the model does not define a version requirement. The string is
//   owned by `loaded_file` and valid until it is deleted. Set to NULL on
//   failure.
// @return kLiteRtLmStatusOk on success (including when no version requirement
//   is defined), or another LiteRtLmStatusCode on failure (see
//   error_reporter.h). kLiteRtLmStatusInvalidArgument if `loaded_file` or
//   `out_version` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the version is returned
// through out_version.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_min_runtime_version(
    const LiteRtLmLoadedFile* loaded_file, const char** out_version);

// Model type of the loaded LiteRT-LM file.
//
// Added in version 0.2.0.
typedef enum LiteRtLmModelType {
  kLiteRtLmModelTypeUnknown = 0,
  kLiteRtLmModelTypeLlm = 1,
  kLiteRtLmModelTypeEmbedding = 2,
} LiteRtLmModelType;

// Gets the model type of the loaded LiteRT-LM file.
//
// @param loaded_file The loaded file to inspect.
// @param out_model_type On success, receives the model type, or
//   kLiteRtLmModelTypeUnknown if the file has neither LLM nor embedding
//   metadata. Not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_model_type` is NULL.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_model_type.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_model_type(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModelType* out_model_type);

// Gets the output embedding dimension for the model.
//
// @param loaded_file The loaded file to inspect.
// @param out_dimension On success, receives the embedding dimension (always
//   positive); not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_dimension` is NULL; kLiteRtLmStatusNotFound if the
//   model is not an embedding model or does not define the dimension.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the value is returned
// through out_dimension, and an undefined dimension is reported as
// kLiteRtLmStatusNotFound instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_embedding_dimension(
    const LiteRtLmLoadedFile* loaded_file, int32_t* out_dimension);

// Gets the supported embedding signature sequence lengths.
//
// Writes up to `max_size` lengths to `lengths`. To query only the count, pass
// `lengths` = NULL (and `max_size` = 0).
//
// @param loaded_file The loaded file to inspect.
// @param lengths Buffer that receives up to `max_size` lengths on success, or
//   NULL to query only the count. Not written on failure.
// @param max_size The capacity of `lengths`, in elements. Must not be
//   negative.
// @param out_count On success, receives the total number of supported lengths,
//   which may exceed `max_size`; not written on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `loaded_file` or `out_count` is NULL, or `max_size` is negative;
//   kLiteRtLmStatusNotFound if the model is not an embedding model or does not
//   define signature lengths.
//
// Added in version 0.2.0.
// Changed in version 1.0.0: returns a status code; the count is returned
// through out_count, and undefined lengths are reported as
// kLiteRtLmStatusNotFound instead of -1.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_loaded_file_embedding_signature_selection(
    const LiteRtLmLoadedFile* loaded_file, int32_t* lengths, int32_t max_size,
    int32_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_MODEL_INFO_H_
