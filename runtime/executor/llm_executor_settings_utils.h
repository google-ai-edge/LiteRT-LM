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

#include <optional>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_options.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/llm_model_type.pb.h"

namespace litert::lm {

// Convert LLM Engine sampler backend to LiteRT backend. If conversion fails,
// return the error.
absl::StatusOr<Backend> GetSamplerBackend(
    const LlmExecutorSettings& executor_settings);

// Create LiteRT compilation options from LLM Engine executor settings. If
// creation fails, return the error.
absl::StatusOr<litert::Options> CreateCompilationOptions(
    const LlmExecutorSettings& executor_settings,
    const ActivationDataType& activation_data_type,
    std::optional<ModelSignatures*> signatures,
    std::optional<std::string> cache_suffix = std::nullopt,
    std::optional<proto::LlmModelType> model_type = std::nullopt);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_SETTINGS_UTILS_H_
