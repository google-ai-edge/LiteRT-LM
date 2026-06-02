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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FASTVLM_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FASTVLM_DATA_PROCESSOR_H_

#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// FastVlmDataProcessor is a model data processor for FastVLM models.
class FastVlmDataProcessor
    : public TypeSafeModelDataProcessor<FastVlmDataProcessorConfig,
                                        FastVlmDataProcessorArguments> {
 public:
  // Creates a FastVlmDataProcessor instance.
  static absl::StatusOr<std::unique_ptr<FastVlmDataProcessor>> Create(
      FastVlmDataProcessorConfig config,
      const PromptTemplateCapabilities& capabilities);

  // Returns the config of the FastVlmDataProcessor.
  const FastVlmDataProcessorConfig& GetConfig() const override {
    return config_;
  }

  // Converts a message into the template input for that message.
  absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const override;

  // Formats tool declarations.
  absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const override;

  // Returns the start of tool call blocks.
  absl::string_view CodeFenceStart() const override { return ""; }

  // Returns the end of tool call blocks.
  absl::string_view CodeFenceEnd() const override { return ""; }

 private:
  explicit FastVlmDataProcessor(
      FastVlmDataProcessorConfig config,
      const PromptTemplateCapabilities& capabilities,
      std::unique_ptr<ImagePreprocessor> image_preprocessor)
      : config_(config),
        capabilities_(capabilities),
        image_preprocessor_(std::move(image_preprocessor)) {}

  absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const FastVlmDataProcessorArguments& args) const override;

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const FastVlmDataProcessorArguments& args) const override;

  absl::Status CloneStateImpl(
      const TypeSafeModelDataProcessor<FastVlmDataProcessorConfig,
                                       FastVlmDataProcessorArguments>& other)
      override;

  FastVlmDataProcessorConfig config_;
  PromptTemplateCapabilities capabilities_;
  std::unique_ptr<ImagePreprocessor> image_preprocessor_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FASTVLM_DATA_PROCESSOR_H_
