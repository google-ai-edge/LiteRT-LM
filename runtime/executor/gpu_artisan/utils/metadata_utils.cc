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

#include "runtime/executor/gpu_artisan/utils/metadata_utils.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "tflite/model_builder.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert

namespace odml::infra::llm_utils {

absl::StatusOr<odml::infra::proto::LlmModelType> GetLlmModelType(
    const ::tflite::FlatBufferModel& fb_model) {
  const ::tflite::Model* tflite_model = fb_model.GetModel();
  if (tflite_model->metadata() != nullptr) {
    for (const auto& metadata : *tflite_model->metadata()) {
      if (kLlmModelTypeName == metadata->name()->c_str()) {
        int llm_model_type_index = metadata->buffer();
        odml::infra::proto::LlmModelType llm_model_type =
            static_cast<odml::infra::proto::LlmModelType>(llm_model_type_index);

        return llm_model_type;
      }
    }
  }
  return absl::NotFoundError(
      absl::StrCat("Failed to get LLM model type, missing ", kLlmModelTypeName,
                   " in tflite metadata"));
}

}  // namespace odml::infra::llm_utils
