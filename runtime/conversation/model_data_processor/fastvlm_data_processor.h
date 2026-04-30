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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/prompt_template.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// FastVlmDataProcessor is a thin wrapper around Gemma3DataProcessor that
// uses FastVlmDataProcessorConfig.
class FastVlmDataProcessor
    : public TypeSafeModelDataProcessor<FastVlmDataProcessorConfig,
                                        FastVlmDataProcessorArguments> {
 public:
  // Creates a FastVlmDataProcessor instance.
  static absl::StatusOr<std::unique_ptr<FastVlmDataProcessor>> Create(
      FastVlmDataProcessorConfig config, std::optional<Preface> preface,
      const Tokenizer* tokenizer,
      const std::vector<std::vector<int>>& stop_token_ids,
      bool enable_constrained_decoding);

  // Returns the config of the FastVlmDataProcessor.
  const FastVlmDataProcessorConfig& GetConfig() const override {
    return config_;
  }

  // Converts a message into the template input for that message.
  absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const override {
    return impl_->MessageToTemplateInput(message);
  }

  // Formats tool declarations.
  absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const override {
    return impl_->FormatTools(tools);
  }

  // Creates a constraint from the given tools.
  absl::StatusOr<std::unique_ptr<Constraint>> CreateConstraint(
      const nlohmann::ordered_json& tools) const override {
    return impl_->CreateConstraint(tools);
  }

  // Returns the start of tool call blocks.
  absl::string_view CodeFenceStart() const override {
    return impl_->CodeFenceStart();
  }

  // Returns the end of tool call blocks.
  absl::string_view CodeFenceEnd() const override {
    return impl_->CodeFenceEnd();
  }

  absl::StatusOr<SingleTurnTemplateRenderResult> RenderSingleTurnTemplate(
      std::vector<Message>& history, const Preface& preface,
      const Message& message, const PromptTemplate& prompt_template,
      bool current_is_appending_message, bool append_message,
      std::optional<nlohmann::ordered_json> extra_context) const override {
    return impl_->RenderSingleTurnTemplate(
        history, preface, message, prompt_template,
        current_is_appending_message, append_message, extra_context);
  }

 private:
  explicit FastVlmDataProcessor(FastVlmDataProcessorConfig config,
                                std::unique_ptr<Gemma3DataProcessor> impl)
      : config_(config), impl_(std::move(impl)) {}

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
  std::unique_ptr<Gemma3DataProcessor> impl_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FASTVLM_DATA_PROCESSOR_H_
