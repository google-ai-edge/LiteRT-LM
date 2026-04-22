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

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/constraint_provider_factory.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/channel_util.h"
#include "runtime/conversation/internal_callback_util.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/model_type_utils.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

constexpr absl::string_view kRoleKey = "role";
constexpr absl::string_view kUser = "user";
constexpr absl::string_view kChannelsKey = "channels";
constexpr absl::string_view kChannelContentCheckpoint =
    "channel_content_checkpoint";

bool IsEmptyInputError(const absl::Status& status) {
  return absl::IsInvalidArgument(status) &&
         absl::StrContains(status.message(), "Input is empty");
}

// Ignores the invalid argument error when Session Prefill is called with empty
// input.
absl::Status IgnoreEmptyInputError(const absl::Status& status) {
  return IsEmptyInputError(status) ? absl::OkStatus() : status;
}

bool IsEmptyPreface(const Preface& preface) {
  auto json_preface = std::get<JsonPreface>(preface);
  return (json_preface.messages.is_null() || json_preface.messages.empty()) &&
         (json_preface.tools.is_null() || json_preface.tools.empty()) &&
         (json_preface.extra_context.is_null() ||
          json_preface.extra_context.empty());
}

bool IsUserMessage(const nlohmann::ordered_json& json_msg) {
  return json_msg.contains(kRoleKey) && json_msg[kRoleKey].is_string() &&
         json_msg[kRoleKey].get<absl::string_view>() == kUser;
}

}  // namespace

absl::StatusOr<ConversationConfig> ConversationConfig::CreateDefault(
    const Engine& engine) {
  return ConversationConfig::Builder().Build(engine);
}

absl::StatusOr<ConversationConfig> ConversationConfig::CreateInternal(
    const Engine& engine, const SessionConfig& session_config,
    std::optional<Preface> preface,
    std::optional<PromptTemplate> overwrite_prompt_template,
    std::optional<DataProcessorConfig> overwrite_processor_config,
    bool enable_constrained_decoding, bool prefill_preface_on_init,
    std::optional<ConstraintProviderConfig> constraint_provider_config,
    std::optional<std::vector<Channel>> overwrite_channels,
    bool filter_channel_content_from_kv_cache) {
  if (preface.has_value() && !std::holds_alternative<JsonPreface>(*preface)) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }

  SessionConfig session_config_copy = session_config;
  session_config_copy.SetApplyPromptTemplateInSession(false);
  RETURN_IF_ERROR(
      session_config_copy.MaybeUpdateAndValidate(engine.GetEngineSettings()));

  auto metadata = engine.GetEngineSettings().GetLlmMetadata();
  PromptTemplate prompt_template("");
  if (overwrite_prompt_template.has_value()) {
    prompt_template = *overwrite_prompt_template;
  } else if (metadata.has_value()) {
    if (metadata->has_jinja_prompt_template()) {
      prompt_template = PromptTemplate(metadata->jinja_prompt_template());
    } else if (metadata->has_prompt_templates()) {
      ASSIGN_OR_RETURN(
          std::string jinja_source,
          GetDefaultJinjaPromptTemplate(metadata->prompt_templates(),
                                        metadata->llm_model_type()));
      prompt_template = PromptTemplate(jinja_source);
    } else {
      return absl::InvalidArgumentError(
          "Failed to select jinja prompt template from llm metadata.");
    }
  } else {
    return absl::InvalidArgumentError(
        "Failed to select jinja prompt template. No llm metadata provided.");
  }

  std::vector<Channel> channels;
  if (overwrite_channels.has_value()) {
    channels = *std::move(overwrite_channels);
  } else if (metadata.has_value()) {
    for (const auto& channel : metadata->channels()) {
      channels.push_back(
          litert::lm::Channel{.channel_name = channel.channel_name(),
                              .start = channel.start(),
                              .end = channel.end()});
    }
  }

  for (const auto& channel : channels) {
    if (channel.channel_name.empty()) {
      return absl::InvalidArgumentError(
          "Custom channel must have a non-empty channel_name.");
    }
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
      session_config_copy, preface.value_or(JsonPreface()), prompt_template,
      processor_config, enable_constrained_decoding, prefill_preface_on_init,
      std::move(constraint_provider_config), std::move(channels),
      filter_channel_content_from_kv_cache);
}

absl::StatusOr<std::string>
Conversation::GetSingleTurnTextFromSingleTurnTemplate(
    const Message& message, const OptionalArgs& optional_args) {
  absl::MutexLock lock(history_mutex_);  // NOLINT
  ASSIGN_OR_RETURN(
      auto result,
      model_data_processor_->RenderSingleTurnTemplate(
          history_,
          config_.prefill_preface_on_init() ? JsonPreface() : preface_, message,
          prompt_template_,
          /*current_is_appending_message=*/is_appending_message_,
          /*append_message=*/optional_args.has_pending_message,
          optional_args.extra_context));
  is_appending_message_ = result.is_appending_message;
  return result.text;
}

absl::StatusOr<std::string> Conversation::GetSingleTurnTextFromFullHistory(
    const Message& message, const OptionalArgs& optional_args) {
  PromptTemplateInput old_tmpl_input;
  RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface_, model_data_processor_.get(), old_tmpl_input));

  // Merge extra context for the message into the extra context provided in the
  // preface. Existing keys will be overwritten.
  if (optional_args.extra_context.has_value()) {
    for (const auto& [key, value] : optional_args.extra_context->items()) {
      old_tmpl_input.extra_context[key] = value;
    }
  }

  absl::MutexLock lock(history_mutex_);  // NOLINT
  for (const auto& history_msg : history_) {
    ASSIGN_OR_RETURN(
        nlohmann::ordered_json message_tmpl_input,
        model_data_processor_->MessageToTemplateInput(history_msg));
    old_tmpl_input.messages.push_back(message_tmpl_input);
  }
  nlohmann::ordered_json messages =
      message.is_array() ? message : nlohmann::ordered_json::array({message});
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

  std::string old_string;
  if (!IsEmptyPreface(preface_) || !history_.empty()) {
    old_tmpl_input.add_generation_prompt = false;
    ASSIGN_OR_RETURN(old_string, prompt_template_.Apply(old_tmpl_input));
  }

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

absl::StatusOr<std::string> Conversation::GetSingleTurnText(
    const Message& message, const OptionalArgs& optional_args) {
  if (!prompt_template_.GetCapabilities().supports_single_turn &&
      optional_args.has_pending_message) {
    return absl::InvalidArgumentError(
        "The prompt template does not support single turn template, but "
        "has_pending_message is true. `has_pending_message` is only valid for "
        "model templates and ModelDataProcessor that supports single turn "
        "prompt rendering.");
  }
  if (prompt_template_.GetCapabilities().supports_single_turn) {
    auto single_turn_text =
        GetSingleTurnTextFromSingleTurnTemplate(message, optional_args);
    if (!absl::IsUnimplemented(single_turn_text.status())) {
      return single_turn_text;
    }
  }
  return GetSingleTurnTextFromFullHistory(message, optional_args);
}

absl::StatusOr<DecodeConfig> Conversation::CreateDecodeConfig(
    std::optional<ConstraintArg> decoding_constraint,
    std::optional<int> max_output_tokens) {
  auto decode_config = DecodeConfig::CreateDefault();
  if (max_output_tokens.has_value()) {
    decode_config.SetMaxOutputTokens(max_output_tokens.value());
  }
  if (decoding_constraint.has_value() && constraint_provider_ != nullptr) {
    ASSIGN_OR_RETURN(constraint_, constraint_provider_->CreateConstraint(
                                      std::move(decoding_constraint).value()));
  } else if (config_.constrained_decoding_enabled() && constraint_ == nullptr &&
             std::holds_alternative<JsonPreface>(preface_)) {
    // Create a constraint from the tools defined in the preface, if any.
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
    Engine& engine, const ConversationConfig& config) {
  absl::Time start_time = absl::Now();
  if (!std::holds_alternative<JsonPreface>(config.GetPreface())) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                   engine.CreateSession(config.GetSessionConfig()));
  ASSIGN_OR_RETURN(
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      CreateModelDataProcessor(config.GetProcessorConfig(), config.GetPreface(),
                               &engine.GetTokenizer(),
                               session->GetSessionConfig().GetStopTokenIds(),
                               config.constrained_decoding_enabled(),
                               config.GetPromptTemplate().GetCapabilities()));
  std::unique_ptr<ConstraintProvider> constraint_provider;
  if (config.constraint_provider_config().has_value()) {
    ASSIGN_OR_RETURN(
        constraint_provider,
        CreateConstraintProvider(
            config.constraint_provider_config().value(), engine.GetTokenizer(),
            session->GetSessionConfig().GetStopTokenIds()));
  }
  auto conversation = absl::WrapUnique(new Conversation(
      engine, std::move(session), std::move(model_data_processor),
      config.GetPreface(), config.GetPromptTemplate(), config,
      std::move(constraint_provider)));
  if (config.prefill_preface_on_init() &&
      !IsEmptyPreface(config.GetPreface())) {
    std::string single_turn_text;
    std::vector<Message> tmp_history;
    bool fallback =
        !conversation->prompt_template_.GetCapabilities().supports_single_turn;
    const auto render_result =
        conversation->model_data_processor_->RenderSingleTurnTemplate(
            tmp_history, config.GetPreface(), Message(),
            config.GetPromptTemplate(),
            /*current_is_appending_message=*/false,
            /*append_message=*/false,
            /*extra_context=*/std::nullopt);
    if (fallback || absl::IsUnimplemented(render_result.status())) {
      // Fallback to the old way of prefilling the preface.
      PromptTemplateInput tmpl_input;
      RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
          config.GetPreface(), conversation->model_data_processor_.get(),
          tmpl_input));
      tmpl_input.add_generation_prompt = false;
      ASSIGN_OR_RETURN(single_turn_text,
                       conversation->prompt_template_.Apply(tmpl_input));
    } else if (render_result.ok()) {
      single_turn_text = render_result->text;
    } else {
      return render_result.status();
    }
    ASSIGN_OR_RETURN(const auto session_inputs,
                     conversation->model_data_processor_->ToInputDataVector(
                         single_turn_text,
                         std::get<JsonPreface>(config.GetPreface()).messages,
                         std::monostate()));
    if (!session_inputs.empty()) {
      RETURN_IF_ERROR(conversation->session_->RunPrefill(session_inputs));
    }
  }

  if (engine.GetEngineSettings().IsBenchmarkEnabled()) {
    ASSIGN_OR_RETURN(BenchmarkInfo * benchmark_info,
                     conversation->GetMutableBenchmarkInfo());
    RETURN_IF_ERROR(benchmark_info->InitPhaseRecord(
        BenchmarkInfo::InitPhase::kConversation, absl::Now() - start_time));
  }

  return conversation;
}

void Conversation::AddTaskController(
    const std::optional<std::string>& task_group_id,
    std::unique_ptr<Engine::Session::TaskController> task_controller) {
  if (task_group_id.has_value() && task_controller != nullptr) {
    absl::MutexLock lock(task_controllers_mutex_);
    task_controllers_[*task_group_id].emplace_back(std::move(task_controller));
  }
}

absl::StatusOr<Message> Conversation::SendMessage(const Message& message,
                                                  OptionalArgs optional_args) {
  absl::Notification done;
  absl::Status error_status;
  bool appending = is_appending_message_;

  absl::Status status = SendMessageAsync(
      message,
      [&](absl::StatusOr<Message> message) {
        if (!message.ok()) {
          // If the message is an error, set the error status and notify done.
          error_status = message.status();
          if (!done.HasBeenNotified()) {
            done.Notify();
          }
          return;
        }

        if (message->is_null()) {
          // Message is null when decode is done.
          if (!done.HasBeenNotified()) {
            done.Notify();
          }
        }
      },
      std::move(optional_args));

  if (!status.ok()) {
    return status;
  }

  done.WaitForNotification();

  if (!error_status.ok()) {
    return error_status;
  }

  if (appending) {
    return Message();
  }

  absl::MutexLock lock(history_mutex_);
  if (history_.empty()) {
    return absl::InternalError("History is empty after SendMessage");
  }
  return history_.back();
}

absl::Status Conversation::SendMessageAsync(
    const Message& message,
    absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
    OptionalArgs optional_args) {
  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message, optional_args));
  auto open_channel_name =
      GetOpenChannelName(single_turn_text, config_.GetChannels());

  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    if (message.is_array()) {
      for (const auto& message : message) {
        history_.push_back(message);
      }
    } else {
      history_.push_back(message);
    }
  }

  // If channel content (e.g. reasoning) needs to be filtered from the KV cache,
  // we rewind the session to the last user message and compute the inputs that
  // need to be "refilled" into the session.
  std::vector<InputData> refill_session_inputs;
  if (config_.filter_channel_content_from_kv_cache() &&
      IsUserMessage(message) && !is_appending_message_) {
    if (channel_content_since_last_user_message_) {
      ASSIGN_OR_RETURN(refill_session_inputs,
                       RewindAndGetInputDataVector(optional_args));
      channel_content_since_last_user_message_ = false;
    }

    if (refill_session_inputs.empty()) {
      // If there are no refill session inputs, save a session checkpoint here.
      RETURN_IF_ERROR(session_->SaveCheckpoint(kChannelContentCheckpoint));
    }

    absl::MutexLock lock(history_mutex_);
    checkpoint_message_index_ = history_.size() - 1;
  }

  ASSIGN_OR_RETURN(
      auto session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({message}),
          optional_args.args.value_or(std::monostate())));

  if (is_appending_message_) {
    ASSIGN_OR_RETURN(
        auto task_controller,
        session_->RunPrefillAsync(
            session_inputs, [callback = std::move(user_callback)](
                                absl::StatusOr<Responses> responses) mutable {
              auto status = IgnoreEmptyInputError(responses.status());
              if (!status.ok()) {
                callback(responses.status());
              } else {
                callback(Message());
              }
            }));
    AddTaskController(optional_args.task_group_id, std::move(task_controller));
    return absl::OkStatus();
  }

  absl::AnyInvocable<void(Message)> complete_message_callback =
      [this](const Message& complete_message) {
        absl::MutexLock lock(this->history_mutex_);
        this->history_.push_back(complete_message);

        // If the model's output message contains channel content, set a
        // variable indicating the session needs to be rewound to the last user
        // message.
        if (config_.filter_channel_content_from_kv_cache() &&
            complete_message.contains(kChannelsKey)) {
          channel_content_since_last_user_message_ = true;
        }
      };

  absl::AnyInvocable<void()> cancel_callback = [this]() {
    absl::MutexLock lock(this->history_mutex_);
    this->history_.pop_back();
  };

  auto internal_callback =
      std::make_shared<absl::AnyInvocable<void(absl::StatusOr<Responses>)>>(
          CreateInternalCallback(
              *model_data_processor_,
              optional_args.args.value_or(std::monostate()),
              config_.GetChannels(), std::move(user_callback),
              std::move(cancel_callback), std::move(complete_message_callback),
              open_channel_name));

  ASSIGN_OR_RETURN(
      auto decode_config,
      CreateDecodeConfig(std::move(optional_args.decoding_constraint),
                         optional_args.max_output_tokens));

  std::optional<std::string> task_group_id = optional_args.task_group_id;

  // This lambda contains the async calls to prefill and decode. It is called
  // immediately if refill_session_inputs is empty. If refill_session_inputs is
  // not empty, this lambda is called after refill_session_inputs is prefilled.
  auto run_prefill = [this, session_inputs = std::move(session_inputs),
                      internal_callback, decode_config,
                      optional_args =
                          std::move(optional_args)]() -> absl::Status {
    ASSIGN_OR_RETURN(
        auto prefill_task_controller,
        session_->RunPrefillAsync(
            session_inputs, [this, callback = internal_callback, decode_config,
                             task_group_id = optional_args.task_group_id](
                                absl::StatusOr<Responses> responses) mutable {
              // First, check if prefill returned an error. Ignore errors
              // caused by empty input, as this is a valid case for triggering
              // decode only.
              auto status = IgnoreEmptyInputError(responses.status());
              // Scenario 1: Prefill failed with an unexpected error.
              if (!status.ok()) {
                // If prefill failed, invoke the callback with the error
                // status and do not proceed to decode.
                (*callback)(responses.status());
              } else if (IsEmptyInputError(responses.status()) ||
                         responses->GetTaskState() == TaskState::kDone) {
                // Scenario 2: Prefill was skipped due to empty input, or
                // prefill completed successfully. In either case, we can now
                // start the decode process.

                // Run decode.
                auto decode_task_controller = session_->RunDecodeAsync(
                    [callback](absl::StatusOr<Responses> responses) {
                      (*callback)(responses);
                    },
                    decode_config);
                // If RunDecodeAsync returns a task controller, it means the
                // decode task was scheduled successfully. Add the controller
                // to our map if a task_group_id was provided, so it can be
                // cancelled later.
                if (decode_task_controller.ok()) {
                  AddTaskController(task_group_id,
                                    std::move(*decode_task_controller));
                } else {
                  // If !decode_task_controller.ok(), it means
                  // RunDecodeAsync failed to schedule. Invoke the callback
                  // with the error status.
                  (*callback)(decode_task_controller.status());
                }
              }
            }));
    AddTaskController(optional_args.task_group_id,
                      std::move(prefill_task_controller));

    return absl::OkStatus();
  };

  // If there are refill session inputs, run prefill for the refill session
  // inputs first, then save a checkpoint, and then run prefill for the
  // input message.
  if (!refill_session_inputs.empty()) {
    auto refill_callback = [this, run_prefill = std::move(run_prefill),
                            internal_callback](
                               absl::StatusOr<Responses> responses) mutable {
      if (!responses.ok()) {
        (*internal_callback)(responses.status());
        return;
      }

      if (!session_->SaveCheckpoint(kChannelContentCheckpoint).ok()) {
        (*internal_callback)(absl::InternalError(
            "Failed to save checkpoint for channel content."));
        return;
      }

      if (!run_prefill().ok()) {
        (*internal_callback)(absl::InternalError("Failed to start prefill."));
        return;
      }
    };
    ASSIGN_OR_RETURN(auto refill_task_controller,
                     session_->RunPrefillAsync(refill_session_inputs,
                                               std::move(refill_callback)));
    AddTaskController(task_group_id, std::move(refill_task_controller));
    return absl::OkStatus();
  }

  // Run prefill for the input message.
  return run_prefill();
};

absl::StatusOr<Responses> Conversation::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    OptionalArgs optional_args) {
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> cloned_session,
                   session_->Clone());
  return cloned_session->RunTextScoring(target_text,
                                        /*store_token_lengths=*/true);
}

absl::Status Conversation::RunTextScoringAsync(
    const std::vector<absl::string_view>& target_text,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    OptionalArgs optional_args) {
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> cloned_session,
                   session_->CloneAsync(nullptr));
  ASSIGN_OR_RETURN(auto task_controller, cloned_session->RunTextScoringAsync(
                                             target_text, std::move(callback),
                                             /*store_token_lengths=*/true));
  AddTaskController(optional_args.task_group_id, std::move(task_controller));
  return absl::OkStatus();
}

absl::StatusOr<BenchmarkInfo> Conversation::GetBenchmarkInfo() {
  return session_->GetBenchmarkInfo();
}

absl::StatusOr<BenchmarkInfo*> Conversation::GetMutableBenchmarkInfo() {
  return session_->GetMutableBenchmarkInfo();
}

void Conversation::CancelProcess() { session_->CancelProcess(); }

void Conversation::CancelGroup(absl::string_view task_group_id) {
  absl::MutexLock lock(task_controllers_mutex_);
  if (auto it = task_controllers_.find(task_group_id);
      it != task_controllers_.end()) {
    for (auto& task_controller : it->second) {
      if (task_controller != nullptr) {
        task_controller->Cancel().IgnoreError();
      }
    }
    task_controllers_.erase(it);
  }
}

absl::StatusOr<std::unique_ptr<Conversation>> Conversation::Clone() {
  ASSIGN_OR_RETURN(auto session, session_->Clone());
  ASSIGN_OR_RETURN(
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      CreateModelDataProcessor(config_.GetProcessorConfig(),
                               config_.GetPreface(), &engine_.GetTokenizer(),
                               session->GetSessionConfig().GetStopTokenIds(),
                               config_.constrained_decoding_enabled(),
                               config_.GetPromptTemplate().GetCapabilities()));
  auto status = model_data_processor->CloneState(*model_data_processor_);
  if (!status.ok() && !absl::IsUnimplemented(status)) {
    return status;
  }
  std::unique_ptr<ConstraintProvider> constraint_provider;
  if (config_.constraint_provider_config().has_value()) {
    ASSIGN_OR_RETURN(constraint_provider,
                     CreateConstraintProvider(
                         config_.constraint_provider_config().value(),
                         engine_.GetTokenizer(),
                         session->GetSessionConfig().GetStopTokenIds()));
  }
  auto new_conversation = absl::WrapUnique(new Conversation(
      engine_, std::move(session), std::move(model_data_processor),
      config_.GetPreface(), config_.GetPromptTemplate(), config_,
      std::move(constraint_provider)));
  new_conversation->is_appending_message_ = is_appending_message_;
  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    new_conversation->history_ = history_;
  }
  return new_conversation;
}

absl::StatusOr<std::string> Conversation::RenderMessageIntoString(
    const Message& message, OptionalArgs optional_args) {
  return GetSingleTurnText(message, optional_args);
}

absl::StatusOr<std::string> Conversation::GetPrefillTextForMessages(
    absl::Span<const Message> old_messages,
    absl::Span<const Message> new_messages, const OptionalArgs& optional_args,
    bool include_preface) {
  // Create the template context for the `old` string.
  PromptTemplateInput old_context;
  old_context.add_generation_prompt = false;

  // Fill the `old` template context with the preface.
  RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface_, model_data_processor_.get(), old_context));

  // Merge extra context for the message into the extra context provided in the
  // preface. Existing keys will be overwritten.
  if (optional_args.extra_context.has_value()) {
    for (const auto& [key, value] : optional_args.extra_context->items()) {
      old_context.extra_context[key] = value;
    }
  }

  // Add old messages to the `old` template context.
  for (const auto& message : old_messages) {
    ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                     model_data_processor_->MessageToTemplateInput(message));
    old_context.messages.push_back(message_tmpl_input);
  }

  // Render the `old` string.
  //
  // When `old_messages` is empty, the behavior depends on the value of
  // `include_preface`.
  // - If `include_preface` is true, `old_string` will be empty so that the
  // preface will be *included* in the returned text.
  // - If `include_preface` is false, `old_string` will contain the preface
  // text, so the preface text will be *subtracted* from the returned text.
  std::string old_string;
  if (!old_messages.empty() || !include_preface) {
    ASSIGN_OR_RETURN(old_string, prompt_template_.Apply(old_context));
  }

  // Copy the `old` template context to the `new` template context.
  PromptTemplateInput new_context = old_context;

  // Add new messages to the `new` template context.
  nlohmann::ordered_json prefill_messages = nlohmann::ordered_json::array();
  for (const auto& message : new_messages) {
    prefill_messages.push_back(message);
    ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                     model_data_processor_->MessageToTemplateInput(message));
    new_context.messages.push_back(message_tmpl_input);
  }

  // Render the `new` string.
  ASSIGN_OR_RETURN(std::string new_string, prompt_template_.Apply(new_context));

  if (old_string.length() > new_string.length()) {
    return absl::InternalError(
        absl::StrCat("The new rendered string is shorter than the previous "
                     "rendered string. \nold_string: ",
                     old_string, "\nnew_string: ", new_string));
  }

  if (new_string.substr(0, old_string.size()) != old_string) {
    return absl::InternalError(
        absl::StrCat("The new rendered string does not start with the previous "
                     "rendered string. \nold_string: ",
                     old_string, "\nnew_string: ", new_string));
  }

  return new_string.substr(old_string.length());
}

absl::StatusOr<std::vector<InputData>>
Conversation::GetInputDataVectorForMessages(
    absl::Span<const Message> old_messages,
    absl::Span<const Message> new_messages, const OptionalArgs& optional_args,
    bool include_preface) {
  ASSIGN_OR_RETURN(std::string prefill_text,
                   GetPrefillTextForMessages(old_messages, new_messages,
                                             optional_args, include_preface));

  nlohmann::ordered_json prefill_messages = nlohmann::ordered_json::array();
  for (const auto& message : new_messages) {
    prefill_messages.push_back(message);
  }

  return model_data_processor_->ToInputDataVector(
      prefill_text, prefill_messages,
      optional_args.args.value_or(std::monostate()));
}

absl::StatusOr<std::vector<InputData>>
Conversation::RewindAndGetInputDataVector(const OptionalArgs& optional_args) {
  absl::MutexLock lock(history_mutex_);
  if (!checkpoint_message_index_.has_value()) {
    // If no rewind is needed, return early with empty InputData vector.
    return std::vector<InputData>();
  }

  // Rewind the session to the saved checkpoint.
  RETURN_IF_ERROR(session_->RewindToCheckpoint(kChannelContentCheckpoint));

  // Get the InputData vector for the messages from the checkpoint onward.
  ASSIGN_OR_RETURN(
      std::vector<InputData> input_data_vector,
      GetInputDataVectorForMessages(
          absl::MakeSpan(history_).subspan(0, *checkpoint_message_index_),
          absl::MakeSpan(history_).subspan(
              *checkpoint_message_index_,
              history_.size() - *checkpoint_message_index_ - 1),
          optional_args,
          /*include_preface=*/!config_.prefill_preface_on_init()));

  // Clear the checkpoint message index.
  checkpoint_message_index_ = std::nullopt;

  return input_data_vector;
}

}  // namespace litert::lm
