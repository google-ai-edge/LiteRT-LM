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

#include "schema/capabilities/capabilities_c.h"

#include <fstream>
#include <ios>
#include <optional>
#include <utility>

#include "schema/capabilities/capabilities.h"

using ::litert::lm::schema::capabilities::InspectModel;
using ::litert::lm::schema::capabilities::ModelMetadataInfo;

struct LiteRtLmLoadedFile {
  std::ifstream stream;
  mutable std::optional<ModelMetadataInfo> cached_info;

  const ModelMetadataInfo* GetInfo() const {
    if (!cached_info.has_value()) {
      auto* mutable_stream = const_cast<std::ifstream*>(&stream);
      auto result = InspectModel(*mutable_stream);
      if (result.ok()) {
        cached_info = std::move(*result);
      }
    }
    return cached_info.has_value() ? &*cached_info : nullptr;
  }
};

extern "C" {

LiteRtLmLoadedFile* litert_lm_loaded_file_create(const char* litertlm_path) {
  if (litertlm_path == nullptr) {
    return nullptr;
  }
  auto* file = new LiteRtLmLoadedFile;
  file->stream.open(litertlm_path, std::ios::binary);
  if (!file->stream.is_open()) {
    delete file;
    return nullptr;
  }
  return file;
}

void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file) {
  delete loaded_file;
}

bool litert_lm_loaded_file_has_speculative_decoding_support(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return false;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return false;
  }
  return info->llm_capability->supports_speculative_decoding;
}

bool litert_lm_loaded_file_has_vision_support(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return false;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return false;
  }
  for (const auto& modality : info->llm_capability->input_modalities) {
    if (modality == litert::lm::schema::capabilities::Modality::kVision) {
      return true;
    }
  }
  return false;
}

bool litert_lm_loaded_file_has_audio_support(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return false;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return false;
  }
  for (const auto& modality : info->llm_capability->input_modalities) {
    if (modality == litert::lm::schema::capabilities::Modality::kAudio) {
      return true;
    }
  }
  return false;
}

bool litert_lm_loaded_file_has_function_calling_support(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return false;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return false;
  }
  return info->llm_capability->supports_function_calling;
}

bool litert_lm_loaded_file_has_thinking_support(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return false;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return false;
  }
  return info->llm_capability->supports_thinking;
}

int litert_lm_loaded_file_get_max_context_length(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return -1;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return -1;
  }
  return info->llm_capability->max_context_length;
}

const char* litert_lm_loaded_file_get_model_class(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return nullptr;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || info->model_class.empty()) {
    return nullptr;
  }
  return info->model_class.c_str();
}

const char* litert_lm_loaded_file_get_tf_hub_model_id(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return nullptr;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || info->tf_hub_model_id.empty()) {
    return nullptr;
  }
  return info->tf_hub_model_id.c_str();
}

const char* litert_lm_loaded_file_get_min_litertlm_version(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return nullptr;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || info->min_litertlm_version.empty()) {
    return nullptr;
  }
  return info->min_litertlm_version.c_str();
}

float litert_lm_loaded_file_get_default_temperature(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return -1.0f;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value() ||
      !info->llm_capability->default_sampler_params.has_value()) {
    return -1.0f;
  }
  return info->llm_capability->default_sampler_params->temperature();
}

int litert_lm_loaded_file_get_default_top_k(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return -1;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value() ||
      !info->llm_capability->default_sampler_params.has_value()) {
    return -1;
  }
  return info->llm_capability->default_sampler_params->k();
}

float litert_lm_loaded_file_get_default_top_p(LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return -1.0f;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value() ||
      !info->llm_capability->default_sampler_params.has_value()) {
    return -1.0f;
  }
  return info->llm_capability->default_sampler_params->p();
}

int litert_lm_loaded_file_get_supported_backend_count(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return 0;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return 0;
  }
  return static_cast<int>(info->llm_capability->supported_backends.size());
}

const char* litert_lm_loaded_file_get_supported_backend(
    LiteRtLmLoadedFile* loaded_file, int index) {
  if (loaded_file == nullptr || index < 0) {
    return nullptr;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return nullptr;
  }
  if (index >=
      static_cast<int>(info->llm_capability->supported_backends.size())) {
    return nullptr;
  }
  return info->llm_capability->supported_backends[index].c_str();
}

int litert_lm_loaded_file_get_supported_vision_resolution_count(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return 0;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return 0;
  }
  return static_cast<int>(
      info->llm_capability->supported_vision_resolutions.size());
}

int litert_lm_loaded_file_get_supported_vision_resolution(
    LiteRtLmLoadedFile* loaded_file, int index) {
  if (loaded_file == nullptr || index < 0) {
    return -1;
  }
  const auto* info = loaded_file->GetInfo();
  if (info == nullptr || !info->llm_capability.has_value()) {
    return -1;
  }
  if (index >= static_cast<int>(
                   info->llm_capability->supported_vision_resolutions.size())) {
    return -1;
  }
  return info->llm_capability->supported_vision_resolutions[index];
}

}  // extern "C"
