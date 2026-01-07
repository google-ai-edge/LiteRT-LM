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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GENERIC_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GENERIC_DATA_PROCESSOR_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// GenericDataProcessor is a ModelDataProcessor that is used for models that
// do not have any specific data processing requirements. It is a simple
// pass-through processor that does not perform any preprocessing or
// postprocessing on the data, and only support text modality.
class GenericDataProcessor
    : public TypeSafeModelDataProcessor<GenericDataProcessorConfig,
                                        GenericDataProcessorArguments> {
 public:
  static absl::StatusOr<std::unique_ptr<ModelDataProcessor>> Create(
      GenericDataProcessorConfig config = GenericDataProcessorConfig(),
      const PromptTemplateCapabilities& capabilities =
          PromptTemplateCapabilities());

  // Return the same tools as the input for generic models.
  absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const override {
    return tools;
  }

  // Return the same message as the template input for generic models.
  absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const override;

  // No-op for generic models.
  absl::string_view CodeFenceStart() const override { return ""; }

  // No-op for generic models.
  absl::string_view CodeFenceEnd() const override { return ""; }

  // Returns the config of the model data processor.
  const GenericDataProcessorConfig& GetConfig() const override {
    return config_;
  }

 private:
  explicit GenericDataProcessor(GenericDataProcessorConfig config,
                                const PromptTemplateCapabilities& capabilities)
      : config_(config), capabilities_(capabilities) {};

  absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const GenericDataProcessorArguments& args) const override;

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const GenericDataProcessorArguments& args) const override;

  GenericDataProcessorConfig config_;
  PromptTemplateCapabilities capabilities_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GENERIC_DATA_PROCESSOR_H_
