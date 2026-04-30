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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_DATA_PROCESSOR_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#endif
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/prompt_template.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Gemma3DataProcessor is a model data processor for Gemma3 and Gemma3N models.
class Gemma3DataProcessor
    : public TypeSafeModelDataProcessor<Gemma3DataProcessorConfig,
                                        Gemma3DataProcessorArguments> {
 public:
  // Creates a Gemma3DataProcessor instance.
  static absl::StatusOr<std::unique_ptr<Gemma3DataProcessor>> Create(
      Gemma3DataProcessorConfig config = Gemma3DataProcessorConfig(),
      std::optional<Preface> preface = std::nullopt,
      const Tokenizer* tokenizer = nullptr,
      const std::vector<std::vector<int>>& stop_token_ids = {},
      bool enable_constrained_decoding = false);

  // Returns the config of the Gemma3DataProcessor.
  const Gemma3DataProcessorConfig& GetConfig() const override {
    return config_;
  }

  // Converts a message into the template input for that message.
  absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const override;

  // Formats tool declarations.
  absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const override;

  absl::StatusOr<std::unique_ptr<Constraint>> CreateConstraint(
      const nlohmann::ordered_json& tools) const override;

  // Returns the start of tool call blocks.
  absl::string_view CodeFenceStart() const override;

  // Returns the end of tool call blocks.
  absl::string_view CodeFenceEnd() const override;

  absl::StatusOr<SingleTurnTemplateRenderResult> RenderSingleTurnTemplate(
      std::vector<Message>& history, const Preface& preface,
      const Message& message, const PromptTemplate& prompt_template,
      bool current_is_appending_message, bool append_message,
      std::optional<nlohmann::ordered_json> extra_context) const override;

 private:
#if defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
  explicit Gemma3DataProcessor(
      const Gemma3DataProcessorConfig& config = Gemma3DataProcessorConfig(),
      std::optional<Preface> preface = std::nullopt,
      std::unique_ptr<ImagePreprocessor> image_preprocessor = nullptr,
      std::unique_ptr<AudioPreprocessor> audio_preprocessor = nullptr)
      : config_(config),
        preface_(preface),
        image_preprocessor_(std::move(image_preprocessor)),
        audio_preprocessor_(std::move(audio_preprocessor)) {};
#else
  explicit Gemma3DataProcessor(
      std::unique_ptr<LiteRtLmGemmaModelConstraintProvider,
                      decltype(&LiteRtLmGemmaModelConstraintProvider_Destroy)>
          constraint_provider,
      const Gemma3DataProcessorConfig& config = Gemma3DataProcessorConfig(),
      std::optional<Preface> preface = std::nullopt,
      std::unique_ptr<ImagePreprocessor> image_preprocessor = nullptr,
      std::unique_ptr<AudioPreprocessor> audio_preprocessor = nullptr)
      : constraint_provider_c_(std::move(constraint_provider)),
        config_(config),
        preface_(preface),
        image_preprocessor_(std::move(image_preprocessor)),
        audio_preprocessor_(std::move(audio_preprocessor)) {};
#endif

  absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const Gemma3DataProcessorArguments& args) const override;

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const Gemma3DataProcessorArguments& args) const override;

  absl::Status CloneStateImpl(
      const TypeSafeModelDataProcessor<Gemma3DataProcessorConfig,
                                       Gemma3DataProcessorArguments>& other)
      override;

#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
  std::unique_ptr<LiteRtLmGemmaModelConstraintProvider,
                  decltype(&LiteRtLmGemmaModelConstraintProvider_Destroy)>
      constraint_provider_c_;
#endif
  Gemma3DataProcessorConfig config_;
  std::optional<Preface> preface_;
  std::unique_ptr<ImagePreprocessor> image_preprocessor_;
  std::unique_ptr<AudioPreprocessor> audio_preprocessor_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_DATA_PROCESSOR_H_
