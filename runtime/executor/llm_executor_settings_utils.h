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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_SETTINGS_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_SETTINGS_UTILS_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/proto/llm_inference_engine.pb.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

using ::odml::infra::proto::SessionConfig;

// Convert LLM Engine backend to LiteRT backend. If conversion fails, return
// the error.
absl::StatusOr<Backend> ConvertBackend(const SessionConfig::Backend& backend);

// Convert LLM Engine ActivationDataType to LiteRT ActivationDataType. If
// conversion fails, return the error.
absl::StatusOr<ActivationDataType> ConvertActivationDataType(
    const SessionConfig::ActivationDataType& activation_data_type);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_SETTINGS_UTILS_H_
