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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_FACTORY_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_FACTORY_COMMON_H_

#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"

namespace litert::lm {

absl::StatusOr<std::string> GetTokenString(
    const proto::TokenUnion& token_union);

absl::Status ValidateVisualTokenBudget(const DataProcessorArguments& args,
                                       int max_vision_tokens_per_image);

absl::StatusOr<DataProcessorConfig> CreateGemma3DataProcessorConfig(
    const proto::LlmModelType& model_type);

absl::StatusOr<DataProcessorConfig> CreateGemma4DataProcessorConfig(
    const proto::LlmModelType& model_type);

absl::StatusOr<DataProcessorConfig> CreateGenericDataProcessorConfig(
    const proto::LlmModelType& model_type,
    const std::optional<proto::ToolCallConfig>& tool_call_config =
        std::nullopt);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_FACTORY_COMMON_H_
