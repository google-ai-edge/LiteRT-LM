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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_H_

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// ModelDataProcessor is a model-specific component that converts between the
// generic Json messages and the Litert LM InputData type.
class ModelDataProcessor {
 public:
  // The result of rendering a single turn template.
  struct SingleTurnTemplateRenderResult {
    // The rendered text.
    std::string text;
    // The new state of is_appending_message of Conversation should be updated
    // to.
    bool is_appending_message;
  };

  virtual ~ModelDataProcessor() = default;

  // Converts a rendered template prompt and a list of messages to a vector of
  // InputData, which is the input to the LLM Session.
  virtual absl::StatusOr<std::vector<InputData>> ToInputDataVector(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const DataProcessorArguments& args) const = 0;

  // Converts a list of responses from the LLM Session to a Message, which is
  // the output to the user.
  virtual absl::StatusOr<Message> ToMessage(
      const Responses& responses, const DataProcessorArguments& args) const = 0;

  // Converts a message into the Jinja template input for that message.
  //
  // Although the message is already a JSON object, some models require
  // additional processing to convert the message into the input needed by the
  // Jinja template.
  //
  // For example, messages represent tool calls as a list of JSON objects, but a
  // model's Jinja template may expect the tool calls to already be formatted
  // in a particular tool calling syntax.
  virtual absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const = 0;

  // Renders a single turn template for the given message and history. Only the
  // prompt template supporting single turn is valid for this method.
  //  - history: The history of the conversation.
  //  - preface: The preface of the conversation.
  //  - message: The current message to be rendered.
  //  - prompt_template: The prompt template to use for rendering.
  //  - current_is_appending_message: Whether the current conversation is in
  //  appending state.
  //  - append_message: Whether the current message is for appending.
  //  - extra_context: Optional context to merge into the PromptTemplateInput
  //  for prompt template rendering.
  //
  // Returns the rendered text and the new is_appending_message as a
  // SingleTurnTemplateRenderResult.
  virtual absl::StatusOr<SingleTurnTemplateRenderResult>
  RenderSingleTurnTemplate(
      std::vector<Message>& history, const Preface& preface,
      const Message& message, const PromptTemplate& prompt_template,
      bool current_is_appending_message, bool append_message,
      std::optional<nlohmann::ordered_json> extra_context) const {
    return absl::UnimplementedError(
        "RenderSingleTurnTemplate is not implemented.");
  }

  // Formats the provided tools to be inserted into the system/developer
  // instruction of the prompt.
  virtual absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const = 0;

  // Creates a constraint from the given tools. The constraint is used for
  // constrained decoding. It is created from the tools defined in the preface,
  // if any.
  virtual absl::StatusOr<std::unique_ptr<Constraint>> CreateConstraint(
      const nlohmann::ordered_json& tools) const {
    return absl::UnimplementedError("CreateConstraint is not implemented.");
  };

  // Returns the start of tool call blocks.
  virtual absl::string_view CodeFenceStart() const = 0;

  // Returns the end of tool call blocks.
  virtual absl::string_view CodeFenceEnd() const = 0;

  // Clones the state of the other model data processor.
  virtual absl::Status CloneState(const ModelDataProcessor& other) = 0;

  // Sets whether to return an error status when a tool call fails to parse.
  void SetReturnErrorOnParseFailure(bool return_error_on_parse_failure) {
    return_error_on_parse_failure_ = return_error_on_parse_failure;
  }

  // Returns whether to return an error status when a tool call fails to parse.
  bool ReturnErrorOnParseFailure() const {
    return return_error_on_parse_failure_;
  }

 private:
  bool return_error_on_parse_failure_ = true;
};

// TypeSafeModelDataProcessor is a ModelDataProcessor that expects a specific
// type of arguments. It guarantees that the model data processor will only be
// called with the expected arguments type.
//
// The model data processor should overwrite the ToInputDataVectorImpl and
// ToMessageImpl to handle the model-specific logic.
template <typename ExpectedConfigT, typename ExpectedArgsT>
class TypeSafeModelDataProcessor : public ModelDataProcessor {
 public:
  // Converts a rendered template prompt and a list of messages to a vector of
  // InputData, with arguments type validated.
  absl::StatusOr<std::vector<InputData>> ToInputDataVector(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const DataProcessorArguments& args) const final {
    if (std::holds_alternative<ExpectedArgsT>(args)) {
      return this->ToInputDataVectorImpl(rendered_template_prompt, messages,
                                         std::get<ExpectedArgsT>(args));
    } else if (std::holds_alternative<std::monostate>(args)) {
      return this->ToInputDataVectorImpl(rendered_template_prompt, messages,
                                         ExpectedArgsT{});
    }
    return absl::InvalidArgumentError(
        "DataProcessorArguments does not hold the expected type");
  }

  // Converts a list of responses from the LLM Session to a Message, with
  // arguments type validated.
  absl::StatusOr<Message> ToMessage(
      const Responses& responses,
      const DataProcessorArguments& args) const final {
    if (std::holds_alternative<ExpectedArgsT>(args)) {
      return this->ToMessageImpl(responses, std::get<ExpectedArgsT>(args));
    } else if (std::holds_alternative<std::monostate>(args)) {
      return this->ToMessageImpl(responses, ExpectedArgsT{});
    }
    return absl::InvalidArgumentError(
        "DataProcessorArguments does not hold the expected type");
  }

  // Returns the config of the model data processor.
  virtual const ExpectedConfigT& GetConfig() const = 0;

  // Clones the state of the other model data processor.
  absl::Status CloneState(const ModelDataProcessor& other) final {
    const auto* typed_other = dynamic_cast<
        const TypeSafeModelDataProcessor<ExpectedConfigT, ExpectedArgsT>*>(
        &other);
    if (typed_other == nullptr) {
      return absl::InvalidArgumentError(
          "The other ModelDataProcessor is not of the expected type.");
    }
    SetReturnErrorOnParseFailure(typed_other->ReturnErrorOnParseFailure());
    return this->CloneStateImpl(*typed_other);
  }

 private:
  virtual absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const ExpectedArgsT& typed_args) const = 0;

  virtual absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses, const ExpectedArgsT& typed_args) const = 0;

  virtual absl::Status CloneStateImpl(
      const TypeSafeModelDataProcessor<ExpectedConfigT, ExpectedArgsT>&
          other) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_H_
