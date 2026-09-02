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

#include "c/capabilities.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#if defined(__APPLE__)
#include "engine.h"  // NOLINT
#else
#include "c/engine.h"
#endif

#include "schema/capabilities/capabilities.h"

// Definition of the internal C++ struct that implements the opaque
// LiteRtLmLoadedFile handle declared in the header.
struct LiteRtLmLoadedFile {
  // Contains the parsed model capabilities.
  litert::lm::schema::capabilities::ModelCapabilities info;
};

namespace {
using ::litert::lm::schema::capabilities::LlmInferenceCapability;
using ::litert::lm::schema::capabilities::SupportedBackends;

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
}  // namespace

extern "C" {

LiteRtLmLoadedFile* litert_lm_loaded_file_create(const char* litertlm_path) {
  if (litertlm_path == nullptr) {
    return nullptr;
  }
  // Call the core capabilities parser to extract the metadata from the file.
  auto info_or = litert::lm::schema::capabilities::InspectModel(litertlm_path);
  if (!info_or.ok()) {
    return nullptr;
  }
  // Allocate the opaque handle on the heap and store the parsed
  // metadata inside it.
  auto* file = new LiteRtLmLoadedFile;
  file->info = std::move(*info_or);
  return file;
}

void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file) {
  delete loaded_file;
}

// Checks if speculative decoding is supported.
bool litert_lm_loaded_file_has_speculative_decoding_support(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return false;
  }
  return loaded_file->info.llm_capability->supports_speculative_decoding;
}

// Checks if thinking/reasoning budget generation is supported.
bool litert_lm_loaded_file_supports_thinking(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return false;
  }
  return loaded_file->info.llm_capability->supports_thinking;
}

// Checks if function calling/tool use is supported.
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
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return false;
  }
  const auto& modalities = loaded_file->info.llm_capability->input_modalities;
  auto target_modality =
      static_cast<litert::lm::schema::capabilities::Modality>(modality);
  switch (target_modality) {
    case litert::lm::schema::capabilities::Modality::kText:
      return modalities.text;
    case litert::lm::schema::capabilities::Modality::kVision:
      return modalities.vision;
    case litert::lm::schema::capabilities::Modality::kAudio:
      return modalities.audio;
    case litert::lm::schema::capabilities::Modality::kVideo:
      return modalities.video;
  }
  return false;
}

int32_t litert_lm_loaded_file_max_vision_token_budget(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return -1;
  }
  return loaded_file->info.llm_capability->max_vision_token_budget;
}

uint32_t litert_lm_loaded_file_max_context_tokens(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) return 0;
  if (!loaded_file->info.llm_capability.has_value()) return 0;
  return loaded_file->info.llm_capability->max_context_tokens;
}

bool litert_lm_loaded_file_is_dynamic_context(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) return false;
  if (!loaded_file->info.llm_capability.has_value()) return false;
  return loaded_file->info.llm_capability->is_dynamic_context;
}

int32_t litert_lm_loaded_file_vision_signature_selection(
    LiteRtLmLoadedFile* loaded_file, int32_t* lengths, int32_t max_size) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return -1;
  }
  const auto& opt_lengths =
      loaded_file->info.llm_capability->vision_signature_selection;
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

int32_t litert_lm_loaded_file_modality_supported_backends(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return 0;
  }
  const auto* backends =
      GetModalityBackends(*loaded_file->info.llm_capability, modality);
  if (backends == nullptr) {
    return 0;
  }
  int32_t mask = 0;
  if (backends->cpu) mask |= kLiteRtLmBackendCpu;
  if (backends->gpu) mask |= kLiteRtLmBackendGpu;
  if (backends->npu) mask |= kLiteRtLmBackendNpu;
  return mask;
}

LiteRtLmNpuBrand litert_lm_loaded_file_modality_npu_brand(
    LiteRtLmLoadedFile* loaded_file, LiteRtLmModality modality) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return kLiteRtLmNpuBrandUnknown;
  }
  const auto* backends =
      GetModalityBackends(*loaded_file->info.llm_capability, modality);
  if (backends == nullptr) {
    return kLiteRtLmNpuBrandUnknown;
  }
  return static_cast<LiteRtLmNpuBrand>(backends->npu_brand);
}

const char* litert_lm_loaded_file_min_runtime_version(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr || !loaded_file->info.llm_capability.has_value()) {
    return nullptr;
  }
  const auto& version = loaded_file->info.llm_capability->min_runtime_version;
  if (version.empty()) {
    return nullptr;
  }
  return version.c_str();
}

}  // extern "C"
