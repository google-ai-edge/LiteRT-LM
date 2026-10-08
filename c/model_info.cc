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

#include "c/model_info.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "c/engine.h"
#include "c/error_reporter_internal.h"
#include "schema/model_info/model_info.h"

// Definition of the internal C++ struct that implements the opaque
// LiteRtLmLoadedFile handle declared in the header.
struct LiteRtLmLoadedFile {
  // Contains the parsed model info and capabilities.
  litert::lm::schema::model_info::ModelInfo info;
};

namespace {
using ::litert::lm::c::ReturnError;
using ::litert::lm::schema::model_info::EmbeddingInferenceCapability;
using ::litert::lm::schema::model_info::LlmInferenceCapability;
using ::litert::lm::schema::model_info::SamplerParameters;
using ::litert::lm::schema::model_info::SupportedBackends;

// Helper functions to extract modality-specific SupportedBackends from the
// parsed LLM or Embedding capabilities struct. Returns nullptr if the modality
// is invalid or unhandled.
const SupportedBackends* GetModalityBackends(
    const LlmInferenceCapability& llm_cap, LiteRtLmModality modality) {
  switch (modality) {
    case kLiteRtLmModalityText:
      return &llm_cap.text_supported_backends;
    case kLiteRtLmModalityVision:
      return &llm_cap.vision_supported_backends;
    case kLiteRtLmModalityAudio:
      return &llm_cap.audio_supported_backends;
    case kLiteRtLmModalityVideo:
      return &llm_cap.video_supported_backends;
    default:
      return nullptr;
  }
}

const SupportedBackends* GetModalityBackends(
    const EmbeddingInferenceCapability& embed_cap, LiteRtLmModality modality) {
  switch (modality) {
    case kLiteRtLmModalityText:
      return &embed_cap.text_supported_backends;
    case kLiteRtLmModalityVision:
      return &embed_cap.vision_supported_backends;
    case kLiteRtLmModalityAudio:
      return &embed_cap.audio_supported_backends;
    case kLiteRtLmModalityVideo:
      return &embed_cap.video_supported_backends;
    default:
      return nullptr;
  }
}

const SupportedBackends* GetSupportedBackends(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality) {
  if (loaded_file == nullptr) return nullptr;
  if (loaded_file->info.llm_capability.has_value()) {
    const auto* backends =
        GetModalityBackends(*loaded_file->info.llm_capability, modality);
    if (backends != nullptr &&
        (!backends->preferred_backends.empty() || backends->cpu ||
         backends->gpu || backends->npu)) {
      return backends;
    }
  }
  if (loaded_file->info.embedding_capability.has_value()) {
    const auto* backends =
        GetModalityBackends(*loaded_file->info.embedding_capability, modality);
    if (backends != nullptr &&
        (!backends->preferred_backends.empty() || backends->cpu ||
         backends->gpu || backends->npu)) {
      return backends;
    }
  }
  return nullptr;
}

// Returns an InvalidArgument error unless `modality` is a known
// LiteRtLmModality.
absl::Status ValidateModality(LiteRtLmModality modality) {
  switch (modality) {
    case kLiteRtLmModalityText:
    case kLiteRtLmModalityVision:
    case kLiteRtLmModalityAudio:
    case kLiteRtLmModalityVideo:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown LiteRtLmModality: %d.", static_cast<int>(modality)));
}

// Returns an InvalidArgument error if `max_size` is negative.
absl::Status ValidateMaxSize(int32_t max_size) {
  if (max_size < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("max_size must not be negative, got %d.", max_size));
  }
  return absl::OkStatus();
}

// Points `*params` at the model's default sampler parameters. Returns a
// NotFound error if the model has no LLM capability.
absl::Status GetDefaultSamplerParams(const LiteRtLmLoadedFile* loaded_file,
                                     const SamplerParameters** params) {
  if (!loaded_file->info.llm_capability.has_value()) {
    return absl::NotFoundError(
        "The model is not an LLM and has no default sampler parameters.");
  }
  *params = &loaded_file->info.llm_capability->default_sampler_params;
  return absl::OkStatus();
}

// Copies up to `max_size` entries of `values` into `lengths` (if non-NULL) and
// returns the total number of entries.
int32_t CopyLengths(const std::vector<int>& values, int32_t* lengths,
                    int32_t max_size) {
  if (lengths != nullptr) {
    int32_t count = std::min(max_size, static_cast<int32_t>(values.size()));
    for (int32_t i = 0; i < count; ++i) {
      lengths[i] = values[i];
    }
  }
  return static_cast<int32_t>(values.size());
}

bool CheckModality(
    const litert::lm::schema::model_info::SupportedModalities& modalities,
    LiteRtLmModality modality) {
  auto target_modality =
      static_cast<litert::lm::schema::model_info::Modality>(modality);
  switch (target_modality) {
    case litert::lm::schema::model_info::Modality::kText:
      return modalities.text;
    case litert::lm::schema::model_info::Modality::kVision:
      return modalities.vision;
    case litert::lm::schema::model_info::Modality::kAudio:
      return modalities.audio;
    case litert::lm::schema::model_info::Modality::kVideo:
      return modalities.video;
  }
  return false;
}
}  // namespace

extern "C" {

LiteRtLmStatusCode litert_lm_loaded_file_create(
    const char* litertlm_path, LiteRtLmLoadedFile** out_loaded_file) {
  LITERT_LM_C_RETURN_IF_NULL(out_loaded_file);
  *out_loaded_file = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(litertlm_path);
  LITERT_LM_C_ASSIGN_OR_RETURN(
      auto info, litert::lm::schema::model_info::GetModelInfo(litertlm_path));
  auto* file = new LiteRtLmLoadedFile;
  file->info = std::move(info);
  *out_loaded_file = file;
  return kLiteRtLmStatusOk;
}

void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file) {
  delete loaded_file;
}

LiteRtLmStatusCode litert_lm_loaded_file_has_speculative_decoding_support(
    const LiteRtLmLoadedFile* loaded_file, bool* out_supported) {
  LITERT_LM_C_RETURN_IF_NULL(out_supported);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  const auto& llm_cap = loaded_file->info.llm_capability;
  *out_supported =
      llm_cap.has_value() && llm_cap->supports_speculative_decoding;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_supports_thinking(
    const LiteRtLmLoadedFile* loaded_file, bool* out_supported) {
  LITERT_LM_C_RETURN_IF_NULL(out_supported);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  const auto& llm_cap = loaded_file->info.llm_capability;
  *out_supported = llm_cap.has_value() && llm_cap->supports_thinking;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_supports_function_calling(
    const LiteRtLmLoadedFile* loaded_file, bool* out_supported) {
  LITERT_LM_C_RETURN_IF_NULL(out_supported);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  const auto& llm_cap = loaded_file->info.llm_capability;
  *out_supported = llm_cap.has_value() && llm_cap->supports_function_calling;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_sampler_type(
    const LiteRtLmLoadedFile* loaded_file,
    LiteRtLmSamplerType* out_sampler_type) {
  LITERT_LM_C_RETURN_IF_NULL(out_sampler_type);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  const SamplerParameters* params = nullptr;
  LITERT_LM_C_RETURN_IF_ERROR(GetDefaultSamplerParams(loaded_file, &params));
  *out_sampler_type = static_cast<LiteRtLmSamplerType>(params->type);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_sampler_temperature(
    const LiteRtLmLoadedFile* loaded_file, float* out_temperature) {
  LITERT_LM_C_RETURN_IF_NULL(out_temperature);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  const SamplerParameters* params = nullptr;
  LITERT_LM_C_RETURN_IF_ERROR(GetDefaultSamplerParams(loaded_file, &params));
  *out_temperature = params->temperature;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_sampler_top_k(
    const LiteRtLmLoadedFile* loaded_file, int32_t* out_top_k) {
  LITERT_LM_C_RETURN_IF_NULL(out_top_k);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  const SamplerParameters* params = nullptr;
  LITERT_LM_C_RETURN_IF_ERROR(GetDefaultSamplerParams(loaded_file, &params));
  *out_top_k = params->k;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_sampler_top_p(
    const LiteRtLmLoadedFile* loaded_file, float* out_top_p) {
  LITERT_LM_C_RETURN_IF_NULL(out_top_p);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  const SamplerParameters* params = nullptr;
  LITERT_LM_C_RETURN_IF_ERROR(GetDefaultSamplerParams(loaded_file, &params));
  *out_top_p = params->p;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_supports_input_modality(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    bool* out_supported) {
  LITERT_LM_C_RETURN_IF_NULL(out_supported);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  LITERT_LM_C_RETURN_IF_ERROR(ValidateModality(modality));
  bool supports = false;
  if (loaded_file->info.llm_capability.has_value()) {
    supports |= CheckModality(
        loaded_file->info.llm_capability->input_modalities, modality);
  }
  if (!supports && loaded_file->info.embedding_capability.has_value()) {
    supports |= CheckModality(
        loaded_file->info.embedding_capability->input_modalities, modality);
  }
  *out_supported = supports;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_max_vision_token_budget(
    const LiteRtLmLoadedFile* loaded_file, int32_t* out_budget) {
  LITERT_LM_C_RETURN_IF_NULL(out_budget);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  if (loaded_file->info.llm_capability.has_value() &&
      loaded_file->info.llm_capability->max_vision_token_budget >= 0) {
    *out_budget = loaded_file->info.llm_capability->max_vision_token_budget;
    return kLiteRtLmStatusOk;
  }
  if (loaded_file->info.embedding_capability.has_value() &&
      loaded_file->info.embedding_capability->max_vision_token_budget >= 0) {
    *out_budget =
        loaded_file->info.embedding_capability->max_vision_token_budget;
    return kLiteRtLmStatusOk;
  }
  return ReturnError(absl::StatusCode::kNotFound,
                     "The model does not define a max vision token budget.");
}

LiteRtLmStatusCode litert_lm_loaded_file_max_context_tokens(
    const LiteRtLmLoadedFile* loaded_file, uint32_t* out_max_context_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(out_max_context_tokens);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  uint32_t max_context_tokens = 0;
  if (loaded_file->info.llm_capability.has_value()) {
    max_context_tokens = loaded_file->info.llm_capability->max_context_tokens;
  } else if (loaded_file->info.embedding_capability.has_value()) {
    max_context_tokens =
        loaded_file->info.embedding_capability->max_context_tokens;
  }
  if (max_context_tokens == 0) {
    return ReturnError(absl::StatusCode::kNotFound,
                       "The model does not define max context tokens.");
  }
  *out_max_context_tokens = max_context_tokens;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_is_dynamic_context(
    const LiteRtLmLoadedFile* loaded_file, bool* out_is_dynamic) {
  LITERT_LM_C_RETURN_IF_NULL(out_is_dynamic);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  bool is_dynamic = false;
  if (loaded_file->info.llm_capability.has_value()) {
    is_dynamic = loaded_file->info.llm_capability->is_dynamic_context;
  } else if (loaded_file->info.embedding_capability.has_value()) {
    is_dynamic = loaded_file->info.embedding_capability->is_dynamic_context;
  }
  *out_is_dynamic = is_dynamic;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_vision_signature_selection(
    const LiteRtLmLoadedFile* loaded_file, int32_t* lengths, int32_t max_size,
    int32_t* out_count) {
  LITERT_LM_C_RETURN_IF_NULL(out_count);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  LITERT_LM_C_RETURN_IF_ERROR(ValidateMaxSize(max_size));
  const std::vector<int>* values = nullptr;
  if (loaded_file->info.llm_capability.has_value() &&
      loaded_file->info.llm_capability->vision_signature_selection
          .has_value()) {
    values = &*loaded_file->info.llm_capability->vision_signature_selection;
  } else if (loaded_file->info.embedding_capability.has_value() &&
             loaded_file->info.embedding_capability->vision_signature_selection
                 .has_value()) {
    values =
        &*loaded_file->info.embedding_capability->vision_signature_selection;
  }
  if (values == nullptr) {
    return ReturnError(
        absl::StatusCode::kNotFound,
        "The model does not define vision signature selection lengths.");
  }
  *out_count = CopyLengths(*values, lengths, max_size);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_modality_supported_backends(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    LiteRtLmBackendType* backends, int32_t max_size, int32_t* out_count) {
  LITERT_LM_C_RETURN_IF_NULL(out_count);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  LITERT_LM_C_RETURN_IF_ERROR(ValidateModality(modality));
  LITERT_LM_C_RETURN_IF_ERROR(ValidateMaxSize(max_size));
  const auto* modality_backends = GetSupportedBackends(loaded_file, modality);
  if (modality_backends == nullptr) {
    *out_count = 0;
    return kLiteRtLmStatusOk;
  }
  const auto& preferred = modality_backends->preferred_backends;
  if (backends != nullptr) {
    int32_t count = std::min(max_size, static_cast<int32_t>(preferred.size()));
    for (int32_t i = 0; i < count; ++i) {
      backends[i] = static_cast<LiteRtLmBackendType>(preferred[i]);
    }
  }
  *out_count = static_cast<int32_t>(preferred.size());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_modality_npu_brand(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    LiteRtLmNpuBrand* out_npu_brand) {
  LITERT_LM_C_RETURN_IF_NULL(out_npu_brand);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  LITERT_LM_C_RETURN_IF_ERROR(ValidateModality(modality));
  const auto* backends = GetSupportedBackends(loaded_file, modality);
  *out_npu_brand = backends == nullptr
                       ? kLiteRtLmNpuBrandUnknown
                       : static_cast<LiteRtLmNpuBrand>(backends->npu_brand);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_modality_soc_name(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    const char** out_soc_name) {
  LITERT_LM_C_RETURN_IF_NULL(out_soc_name);
  *out_soc_name = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  LITERT_LM_C_RETURN_IF_ERROR(ValidateModality(modality));
  const auto* backends = GetSupportedBackends(loaded_file, modality);
  if (backends != nullptr && !backends->soc_name.empty()) {
    *out_soc_name = backends->soc_name.c_str();
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_min_runtime_version(
    const LiteRtLmLoadedFile* loaded_file, const char** out_version) {
  LITERT_LM_C_RETURN_IF_NULL(out_version);
  *out_version = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  if (loaded_file->info.llm_capability.has_value() &&
      !loaded_file->info.llm_capability->min_runtime_version.empty()) {
    *out_version =
        loaded_file->info.llm_capability->min_runtime_version.c_str();
  } else if (loaded_file->info.embedding_capability.has_value() &&
             !loaded_file->info.embedding_capability->min_runtime_version
                  .empty()) {
    *out_version =
        loaded_file->info.embedding_capability->min_runtime_version.c_str();
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_model_type(
    const LiteRtLmLoadedFile* loaded_file, LiteRtLmModelType* out_model_type) {
  LITERT_LM_C_RETURN_IF_NULL(out_model_type);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  if (loaded_file->info.llm_capability.has_value()) {
    *out_model_type = kLiteRtLmModelTypeLlm;
  } else if (loaded_file->info.embedding_capability.has_value()) {
    *out_model_type = kLiteRtLmModelTypeEmbedding;
  } else {
    *out_model_type = kLiteRtLmModelTypeUnknown;
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_embedding_dimension(
    const LiteRtLmLoadedFile* loaded_file, int32_t* out_dimension) {
  LITERT_LM_C_RETURN_IF_NULL(out_dimension);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  if (!loaded_file->info.embedding_capability.has_value() ||
      loaded_file->info.embedding_capability->embedding_dimension <= 0) {
    return ReturnError(absl::StatusCode::kNotFound,
                       "The model does not define an embedding dimension.");
  }
  *out_dimension = loaded_file->info.embedding_capability->embedding_dimension;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_loaded_file_embedding_signature_selection(
    const LiteRtLmLoadedFile* loaded_file, int32_t* lengths, int32_t max_size,
    int32_t* out_count) {
  LITERT_LM_C_RETURN_IF_NULL(out_count);
  LITERT_LM_C_RETURN_IF_NULL(loaded_file);
  LITERT_LM_C_RETURN_IF_ERROR(ValidateMaxSize(max_size));
  if (!loaded_file->info.embedding_capability.has_value() ||
      !loaded_file->info.embedding_capability->supported_signature_lengths
           .has_value()) {
    return ReturnError(
        absl::StatusCode::kNotFound,
        "The model does not define embedding signature lengths.");
  }
  *out_count = CopyLengths(
      *loaded_file->info.embedding_capability->supported_signature_lengths,
      lengths, max_size);
  return kLiteRtLmStatusOk;
}

}  // extern "C"
