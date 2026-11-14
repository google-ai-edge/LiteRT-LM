// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_UTILS_METADATA_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_UTILS_METADATA_UTILS_H_

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "tflite/model_builder.h"  // from @litert

namespace odml::infra::llm_utils {

constexpr absl::string_view kLlmModelTypeName = "odml.infra.LlmModelType";
constexpr absl::string_view kLlmBackendName = "backend";
constexpr absl::string_view kSpmVocabName = "spm_vocab_model";
constexpr absl::string_view kLoRARank = "lora_rank";
constexpr absl::string_view kImageEncoder = "image_encoder";
constexpr absl::string_view kImageAdapter = "image_adapter";

// Retrieve LlmModelType from tflite flatbuffer metadata.
absl::StatusOr<odml::infra::proto::LlmModelType> GetLlmModelType(
    const ::tflite::FlatBufferModel& fb_model);

inline bool RequireBytesToUnicodeMapping(proto::LlmModelType model_type) {
  return model_type == proto::LLM_MODEL_TYPE_STABLELM_4E1T_3B ||
         model_type == proto::LLM_MODEL_TYPE_FALCON_RW_1B ||
         model_type == proto::LLM_MODEL_TYPE_PHI_2;
}

inline bool RequireFp32Model(proto::LlmModelType model_type) {
  constexpr proto::LlmModelType kFp32Models[] = {
      proto::LLM_MODEL_TYPE_PHI_2,
      proto::LLM_MODEL_TYPE_FALCON_RW_1B,
  };
  return absl::c_linear_search(kFp32Models, model_type);
}

}  // namespace odml::infra::llm_utils

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_UTILS_METADATA_UTILS_H_
