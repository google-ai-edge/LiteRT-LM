// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/llm_executor_settings_utils.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/proto/llm_inference_engine.pb.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

using ::odml::infra::proto::SessionConfig;

absl::StatusOr<Backend> ConvertBackend(const SessionConfig::Backend& backend) {
  switch (backend) {
    case SessionConfig::XNNPACK:
      return Backend::CPU;
    case SessionConfig::ML_DRIFT:
      return Backend::GPU;
    case SessionConfig::GOOGLE_TENSOR:
      return Backend::GOOGLE_TENSOR_ARTISAN;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ", backend));
  };
}

absl::StatusOr<ActivationDataType> ConvertActivationDataType(
    const SessionConfig::ActivationDataType& activation_data_type) {
  switch (activation_data_type) {
    case SessionConfig::ACTIVATION_DATA_TYPE_F32:
      return ActivationDataType::FLOAT32;
    case SessionConfig::ACTIVATION_DATA_TYPE_F16:
      return ActivationDataType::FLOAT16;
    case SessionConfig::ACTIVATION_DATA_TYPE_I16:
      return ActivationDataType::INT16;
    case SessionConfig::ACTIVATION_DATA_TYPE_I8:
      return ActivationDataType::INT8;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported activation data type: ", activation_data_type));
  }
};

}  // namespace litert::lm
