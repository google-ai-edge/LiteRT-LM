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

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory_common.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"

namespace litert::lm {
namespace {

void EnsureBuiltinProcessorsRegistered() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    auto& reg = ModelDataProcessorRegistry::Instance();
    reg.Register<GenericDataProcessor, GenericDataProcessorConfig>();
    reg.Register<Gemma3DataProcessor, Gemma3DataProcessorConfig>();
    reg.Register<Gemma4DataProcessor, Gemma4DataProcessorConfig>();
  });
}

}  // namespace

absl::StatusOr<DataProcessorConfig> CreateDataProcessorConfigFromLlmModelType(
    const proto::LlmModelType& model_type,
    const std::optional<proto::ToolCallConfig>& tool_call_config) {
  if (tool_call_config.has_value() && !model_type.has_generic_model()) {
    return absl::InvalidArgumentError(
        "tool_call_config is only supported for generic_model.");
  }
  switch (model_type.model_type_case()) {
    case proto::LlmModelType::kGemma3:
    case proto::LlmModelType::kGemma3N:
      return CreateGemma3DataProcessorConfig(model_type);
    case proto::LlmModelType::kGemma4:
      return CreateGemma4DataProcessorConfig(model_type);
    case proto::LlmModelType::kGenericModel:
      return CreateGenericDataProcessorConfig(model_type, tool_call_config);
    case proto::LlmModelType::kLfm2:
    case proto::LlmModelType::kMinicpm5:
    case proto::LlmModelType::kMinicpmv4:
    case proto::LlmModelType::kQwen3:
    case proto::LlmModelType::kQwen2P5:
    case proto::LlmModelType::kFastVlm:
    case proto::LlmModelType::kFunctionGemma:
      return absl::UnimplementedError(
          "Model family is not supported in "
          "model_data_processor_factory_lite.");
    default:
      return absl::InvalidArgumentError("Unsupported model type");
  }
}

absl::StatusOr<std::unique_ptr<ModelDataProcessor>> CreateModelDataProcessor(
    const DataProcessorConfig& config, std::optional<Preface> preface,
    const Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    bool enable_constrained_decoding) {
  EnsureBuiltinProcessorsRegistered();
  return ModelDataProcessorRegistry::Instance().Create(
      config, std::move(preface), tokenizer, stop_token_ids,
      enable_constrained_decoding);
}

}  // namespace litert::lm
