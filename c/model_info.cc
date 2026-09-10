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
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "c/error_reporter_internal.h"
#include "schema/model_info/model_info.h"

// Definition of the internal C++ struct that implements the opaque
// LiteRtLmLoadedFile handle declared in the header.
struct LiteRtLmLoadedFile {
  // Contains the parsed model info and capabilities.
  litert::lm::schema::model_info::ModelInfo info;
};

namespace {
using ::litert::lm::c::SetLastError;
using ::litert::lm::schema::model_info::EmbeddingInferenceCapability;
using ::litert::lm::schema::model_info::LlmInferenceCapability;
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
    if (backends != nullptr && (!backends->preferred_backends.empty() ||
                                backends->cpu || backends->gpu ||
                                backends->npu)) {
      return backends;
    }
  }
  if (loaded_file->info.embedding_capability.has_value()) {
    const auto* backends =
        GetModalityBackends(*loaded_file->info.embedding_capability, modality);
    if (backends != nullptr && (!backends->preferred_backends.empty() ||
                                backends->cpu || backends->gpu ||
                                backends->npu)) {
      return backends;
    }
  }
  return nullptr;
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

LiteRtLmLoadedFile* litert_lm_loaded_file_create(const char* litertlm_path) {
  if (litertlm_path == nullptr) {
    SetLastError(absl::StatusCode::kInvalidArgument,
                 "litertlm_path must not be null.");
    return nullptr;
  }
  auto info_or = litert::lm::schema::model_info::GetModelInfo(litertlm_path);
  if (!info_or.ok()) {
    SetLastError(info_or.status());
    return nullptr;
  }
  auto* file = new LiteRtLmLoadedFile;
  file->info = std::move(*info_or);
  return file;
}

void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file) {
  delete loaded_file;
}

bool litert_lm_loaded_file_has_speculative_decoding_support(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return false;
  }
  return loaded_file->info.llm_capability->supports_speculative_decoding;
}

bool litert_lm_loaded_file_supports_thinking(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return false;
  }
  return loaded_file->info.llm_capability->supports_thinking;
}

bool litert_lm_loaded_file_supports_function_calling(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return false;
  }
  return loaded_file->info.llm_capability->supports_function_calling;
}

LiteRtLmSamplerType litert_lm_loaded_file_sampler_type(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return kLiteRtLmSamplerTypeUnspecified;
  }
  return static_cast<LiteRtLmSamplerType>(
      loaded_file->info.llm_capability->default_sampler_params.type);
}

float litert_lm_loaded_file_sampler_temperature(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return 0.0f;
  }
  return loaded_file->info.llm_capability->default_sampler_params.temperature;
}

int32_t litert_lm_loaded_file_sampler_top_k(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return 0;
  }
  return loaded_file->info.llm_capability->default_sampler_params.k;
}

float litert_lm_loaded_file_sampler_top_p(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return 0.0f;
  }
  return loaded_file->info.llm_capability->default_sampler_params.p;
}

bool litert_lm_loaded_file_supports_input_modality(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality) {
  if (loaded_file == nullptr) return false;
  bool supports = false;
  if (loaded_file->info.llm_capability.has_value()) {
    supports |= CheckModality(
        loaded_file->info.llm_capability->input_modalities, modality);
  }
  if (!supports && loaded_file->info.embedding_capability.has_value()) {
    supports |= CheckModality(
        loaded_file->info.embedding_capability->input_modalities, modality);
  }
  return supports;
}

int32_t litert_lm_loaded_file_max_vision_token_budget(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) return -1;
  if (loaded_file->info.llm_capability.has_value()) {
    return loaded_file->info.llm_capability->max_vision_token_budget;
  }
  if (loaded_file->info.embedding_capability.has_value()) {
    return loaded_file->info.embedding_capability->max_vision_token_budget;
  }
  return -1;
}

uint32_t litert_lm_loaded_file_max_context_tokens(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) return 0;
  if (loaded_file->info.llm_capability.has_value()) {
    return loaded_file->info.llm_capability->max_context_tokens;
  }
  if (loaded_file->info.embedding_capability.has_value()) {
    return loaded_file->info.embedding_capability->max_context_tokens;
  }
  return 0;
}

bool litert_lm_loaded_file_is_dynamic_context(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) return false;
  if (loaded_file->info.llm_capability.has_value()) {
    return loaded_file->info.llm_capability->is_dynamic_context;
  }
  if (loaded_file->info.embedding_capability.has_value()) {
    return loaded_file->info.embedding_capability->is_dynamic_context;
  }
  return false;
}

int32_t litert_lm_loaded_file_vision_signature_selection(
    LiteRtLmLoadedFile* loaded_file, int32_t* lengths, int32_t max_size) {
  if (loaded_file == nullptr) {
    return -1;
  }
  const std::optional<std::vector<int>>* opt_lengths = nullptr;
  if (loaded_file->info.llm_capability.has_value() &&
      loaded_file->info.llm_capability->vision_signature_selection
          .has_value()) {
    opt_lengths = &loaded_file->info.llm_capability->vision_signature_selection;
  } else if (loaded_file->info.embedding_capability.has_value() &&
             loaded_file->info.embedding_capability->vision_signature_selection
                 .has_value()) {
    opt_lengths =
        &loaded_file->info.embedding_capability->vision_signature_selection;
  }
  if (opt_lengths == nullptr || !opt_lengths->has_value()) {
    return -1;
  }
  const auto& vec = **opt_lengths;
  if (lengths != nullptr) {
    int32_t count = std::min(max_size, static_cast<int32_t>(vec.size()));
    for (int32_t i = 0; i < count; ++i) {
      lengths[i] = vec[i];
    }
  }
  return static_cast<int32_t>(vec.size());
}

int32_t litert_lm_loaded_file_modality_supported_backends(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality,
    LiteRtLmBackendType* backends, int32_t max_size) {
  const auto* modality_backends = GetSupportedBackends(loaded_file, modality);
  if (modality_backends == nullptr) {
    return 0;
  }
  const auto& preferred = modality_backends->preferred_backends;
  if (backends != nullptr) {
    int32_t count =
        std::min(max_size, static_cast<int32_t>(preferred.size()));
    for (int32_t i = 0; i < count; ++i) {
      backends[i] = static_cast<LiteRtLmBackendType>(preferred[i]);
    }
  }
  return static_cast<int32_t>(preferred.size());
}

LiteRtLmNpuBrand litert_lm_loaded_file_modality_npu_brand(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality) {
  const auto* backends = GetSupportedBackends(loaded_file, modality);
  if (backends == nullptr) {
    return kLiteRtLmNpuBrandUnknown;
  }
  return static_cast<LiteRtLmNpuBrand>(backends->npu_brand);
}

const char* litert_lm_loaded_file_modality_soc_name(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality) {
  const auto* backends = GetSupportedBackends(loaded_file, modality);
  if (backends == nullptr || backends->soc_name.empty()) {
    return nullptr;
  }
  return backends->soc_name.c_str();
}

const char* litert_lm_loaded_file_min_runtime_version(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) return nullptr;
  if (loaded_file->info.llm_capability.has_value() &&
      !loaded_file->info.llm_capability->min_runtime_version.empty()) {
    return loaded_file->info.llm_capability->min_runtime_version.c_str();
  }
  if (loaded_file->info.embedding_capability.has_value() &&
      !loaded_file->info.embedding_capability->min_runtime_version.empty()) {
    return loaded_file->info.embedding_capability->min_runtime_version.c_str();
  }
  return nullptr;
}

bool litert_lm_loaded_file_is_embedding_model(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) return false;
  return loaded_file->info.embedding_capability.has_value();
}

bool litert_lm_loaded_file_is_llm_model(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) return false;
  return loaded_file->info.llm_capability.has_value();
}

int32_t litert_lm_loaded_file_embedding_dimension(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr ||
      !loaded_file->info.embedding_capability.has_value()) {
    return -1;
  }
  int dim = loaded_file->info.embedding_capability->embedding_dimension;
  return dim > 0 ? dim : -1;
}

int32_t litert_lm_loaded_file_embedding_signature_selection(
    LiteRtLmLoadedFile* loaded_file, int32_t* lengths, int32_t max_size) {
  if (loaded_file == nullptr ||
      !loaded_file->info.embedding_capability.has_value()) {
    return -1;
  }
  const auto& opt_lengths =
      loaded_file->info.embedding_capability->supported_signature_lengths;
  if (!opt_lengths.has_value()) {
    return -1;
  }
  if (lengths != nullptr) {
    int32_t count =
        std::min(max_size, static_cast<int32_t>(opt_lengths->size()));
    for (int32_t i = 0; i < count; ++i) {
      lengths[i] = (*opt_lengths)[i];
    }
  }
  return static_cast<int32_t>(opt_lengths->size());
}

}  // extern "C"
