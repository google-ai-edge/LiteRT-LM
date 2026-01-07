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

#include "runtime/conversation/conversation.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/internal_callback_util.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {
absl::Status FillPrefaceForPromptTemplateInput(
    const Preface& preface, ModelDataProcessor* model_data_processor,
    PromptTemplateInput& tmpl_input) {
  if (std::holds_alternative<JsonPreface>(preface)) {
    auto json_preface = std::get<JsonPreface>(preface);

    if (json_preface.messages.is_array()) {
      for (auto& message : json_preface.messages) {
        ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                         model_data_processor->MessageToTemplateInput(message));
        tmpl_input.messages.push_back(message_tmpl_input);
      }
    }

    if (json_preface.tools.is_null()) {
      tmpl_input.tools = nullptr;
    } else {
      ASSIGN_OR_RETURN(tmpl_input.tools,
                       model_data_processor->FormatTools(json_preface.tools));
    }
    tmpl_input.extra_context = json_preface.extra_context;
  } else {
    return absl::UnimplementedError("Preface type is not supported yet");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ConversationConfig> ConversationConfig::CreateDefault(
    const Engine& engine, std::optional<Preface> preface,
    std::optional<PromptTemplate> overwrite_prompt_template,
    std::optional<DataProcessorConfig> overwrite_processor_config,
    bool enable_constrained_decoding, bool prefill_preface_on_init) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  if (overwrite_prompt_template.has_value()) {
    session_config.GetMutableJinjaPromptTemplate() =
        overwrite_prompt_template->GetTemplateSource();
  } else {
    ABSL_LOG(INFO) << "Overwrite prompt template is not provided, using the "
                      "default template from the model metadata.";
  }
  return CreateFromSessionConfig(engine, session_config, preface,
                                 overwrite_processor_config,
                                 enable_constrained_decoding);
}

absl::StatusOr<ConversationConfig> ConversationConfig::CreateFromSessionConfig(
    const Engine& engine, const SessionConfig& session_config,
    std::optional<Preface> preface,
    std::optional<DataProcessorConfig> overwrite_processor_config,
    bool enable_constrained_decoding, bool prefill_preface_on_init) {
  if (preface.has_value() && !std::holds_alternative<JsonPreface>(*preface)) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }
  SessionConfig session_config_copy = session_config;
  // Disable the deprecated prompt templates in the session.
  // TODO - b/453312248: Remove this once the prompt template is removed from
  // Session
  session_config_copy.SetApplyPromptTemplateInSession(false);
  RETURN_IF_ERROR(
      session_config_copy.MaybeUpdateAndValidate(engine.GetEngineSettings()));
  if (session_config_copy.GetJinjaPromptTemplate().empty()) {
    return absl::InvalidArgumentError(
        "Jinja prompt template is empty. Either the prompt template is not "
        "read from model metadata or the prompt template is not set in "
        "SessionConfig.");
  }
  DataProcessorConfig processor_config;
  if (overwrite_processor_config.has_value()) {
    // Use the overwrite processor config if provided.
    processor_config = *overwrite_processor_config;
  } else {
    // Build the processor config from the model metadata.
    ASSIGN_OR_RETURN(processor_config,
                     CreateDataProcessorConfigFromLlmModelType(
                         session_config_copy.GetLlmModelType()));
  }
  return ConversationConfig(
      session_config_copy, preface.value_or(JsonPreface()),
      PromptTemplate(session_config_copy.GetJinjaPromptTemplate()),
      processor_config, enable_constrained_decoding, prefill_preface_on_init);
}

absl::StatusOr<std::string> Conversation::GetSingleTurnText(
    const Message& message) const {
  PromptTemplateInput old_tmpl_input;
  RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface_, model_data_processor_.get(), old_tmpl_input));

  absl::MutexLock lock(history_mutex_);  // NOLINT
  for (const auto& history_msg : history_) {
    if (std::holds_alternative<nlohmann::ordered_json>(history_msg)) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(
                           std::get<nlohmann::ordered_json>(history_msg)));
      old_tmpl_input.messages.push_back(message_tmpl_input);
    } else {
      return absl::UnimplementedError("Message type is not supported yet");
    }
  }

  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  nlohmann::ordered_json json_message =
      std::get<nlohmann::ordered_json>(message);
  nlohmann::ordered_json messages =
      json_message.is_array() ? json_message
                              : nlohmann::ordered_json::array({json_message});
  if (history_.empty() && !config_.prefill_preface_on_init()) {
    PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
    for (const auto& message : messages) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(message));
      new_tmpl_input.messages.push_back(message_tmpl_input);
    }
    new_tmpl_input.add_generation_prompt = true;
    return prompt_template_.Apply(new_tmpl_input);
  }

  old_tmpl_input.add_generation_prompt = false;
  ASSIGN_OR_RETURN(const std::string old_string,
                   prompt_template_.Apply(old_tmpl_input));

  PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
  for (const auto& message : messages) {
    ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                     model_data_processor_->MessageToTemplateInput(message));
    new_tmpl_input.messages.push_back(message_tmpl_input);
  }
  new_tmpl_input.add_generation_prompt = true;
  ASSIGN_OR_RETURN(const std::string& new_string,
                   prompt_template_.Apply(new_tmpl_input));
  if (new_string.substr(0, old_string.size()) != old_string) {
    return absl::InternalError(absl::StrCat(
        "The new rendered template string does not start with the previous "
        "rendered template string. \nold_string: ",
        old_string, "\nnew_string: ", new_string));
  }
  return {new_string.substr(old_string.size(),
                            new_string.size() - old_string.size())};
}

absl::StatusOr<DecodeConfig> Conversation::CreateDecodeConfig() {
  auto decode_config = DecodeConfig::CreateDefault();
  // Create a constraint from the tools defined in the preface, if any.
  if (config_.constrained_decoding_enabled() && constraint_ == nullptr &&
      std::holds_alternative<JsonPreface>(preface_)) {
    auto json_preface = std::get<JsonPreface>(preface_);
    if (!json_preface.tools.is_null()) {
      auto constraint =
          model_data_processor_->CreateConstraint(json_preface.tools);
      if (constraint.ok()) {
        constraint_ = std::move(constraint.value());
      } else if (!absl::IsUnimplemented(constraint.status())) {
        return constraint.status();
      }
    }
  }
  decode_config.SetConstraint(constraint_.get());
  return decode_config;
}

absl::StatusOr<std::unique_ptr<Conversation>> Conversation::Create(
    const Engine& engine, const ConversationConfig& config) {
  if (!std::holds_alternative<JsonPreface>(config.GetPreface())) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                   engine.CreateSession(config.GetSessionConfig()));
  ASSIGN_OR_RETURN(
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      CreateModelDataProcessor(config.GetProcessorConfig(), config.GetPreface(),
                               &session->GetTokenizer(),
                               session->GetSessionConfig().GetStopTokenIds(),
                               config.constrained_decoding_enabled(),
                               config.GetPromptTemplate().GetCapabilities()));
  auto conversation = absl::WrapUnique(new Conversation(
      std::move(session), std::move(model_data_processor), config.GetPreface(),
      config.GetPromptTemplate(), config));
  if (config.prefill_preface_on_init()) {
    PromptTemplateInput tmpl_input;
    RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
        config.GetPreface(), conversation->model_data_processor_.get(),
        tmpl_input));
    ASSIGN_OR_RETURN(const std::string single_turn_text,
                     conversation->prompt_template_.Apply(tmpl_input));
    ASSIGN_OR_RETURN(const auto session_inputs,
                     conversation->model_data_processor_->ToInputDataVector(
                         single_turn_text,
                         std::get<JsonPreface>(config.GetPreface()).messages,
                         std::monostate()));
    RETURN_IF_ERROR(conversation->session_->RunPrefill(session_inputs));
  }
  return conversation;
}

absl::StatusOr<Message> Conversation::SendMessage(
    const Message& message, std::optional<DataProcessorArguments> args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  auto json_message = std::get<nlohmann::ordered_json>(message);
  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message));
  absl::MutexLock lock(history_mutex_);  // NOLINT
  if (json_message.is_array()) {
    for (const auto& message : json_message) {
      history_.push_back(message);
    }
  } else {
    history_.push_back(json_message);
  }
  ASSIGN_OR_RETURN(
      const auto session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({json_message}),
          args.value_or(std::monostate())));
  RETURN_IF_ERROR(session_->RunPrefill(session_inputs));
  ASSIGN_OR_RETURN(auto decode_config, CreateDecodeConfig());
  ASSIGN_OR_RETURN(const Responses& responses,
                   session_->RunDecode(decode_config));
  ASSIGN_OR_RETURN(const Message assistant_message,
                   model_data_processor_->ToMessage(
                       responses, args.value_or(std::monostate())));
  history_.push_back(assistant_message);
  return assistant_message;
}

absl::Status Conversation::SendMessageAsync(
    const Message& message,
    absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
    std::optional<DataProcessorArguments> args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  auto json_message = std::get<nlohmann::ordered_json>(message);
  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message));
  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    if (json_message.is_array()) {
      for (const auto& message : json_message) {
        history_.push_back(message);
      }
    } else {
      history_.push_back(json_message);
    }
  }

  ASSIGN_OR_RETURN(
      const auto session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({json_message}),
          args.value_or(std::monostate())));

  absl::AnyInvocable<void(Message)> complete_message_callback =
      [this](const Message& complete_message) {
        absl::MutexLock lock(this->history_mutex_);  // NOLINT
        this->history_.push_back(complete_message);
      };

  absl::AnyInvocable<void()> cancel_callback = [this]() {
    absl::MutexLock lock(this->history_mutex_);  // NOLINT
    this->history_.pop_back();
  };

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> internal_callback =
      CreateInternalCallback(
          *model_data_processor_, args.value_or(std::monostate()),
          std::move(user_callback), std::move(cancel_callback),
          std::move(complete_message_callback));

  ASSIGN_OR_RETURN(auto decode_config, CreateDecodeConfig());
  ASSIGN_OR_RETURN(
      std::ignore,
      session_->RunPrefillAsync(
          session_inputs,
          [this, callback = std::move(internal_callback),
           decode_config](absl::StatusOr<Responses> responses) mutable {
            if (!responses.ok()) {
              callback(responses.status());
            } else if (responses->GetTaskState() == TaskState::kDone) {
              auto status =
                  session_->RunDecodeAsync(std::move(callback), decode_config);
            }
          }));

  return absl::OkStatus();
};

absl::StatusOr<BenchmarkInfo> Conversation::GetBenchmarkInfo() {
  return session_->GetBenchmarkInfo();
}

void Conversation::CancelProcess() { session_->CancelProcess(); }

}  // namespace litert::lm
