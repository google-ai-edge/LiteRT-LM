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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
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
#include "runtime/components/logits_processor/constrained_decoding/constraint_provider.h"
#include "runtime/components/logits_processor/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/logits_processor/constrained_decoding/constraint_provider_factory.h"
#include "runtime/components/logits_processor/no_repeat_ngram_config.h"
#include "runtime/components/logits_processor/repetition_penalty_config.h"
#include "runtime/components/logits_processor/suppress_tokens_config.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/channel_util.h"
#include "runtime/conversation/internal_callback_util.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/conversation/thinking_config.h"
#include "runtime/core/cached_session.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/model_type_utils.h"
#include "runtime/util/status_macros.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

namespace {

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

// Merges two messages, appending content and tool calls.
//
// This function merges `second` into `first`. Note the following behavior:
//
// 1. **Content Merging**:
//    - If both messages contain `content`, they are merged into a single array
//      of content blocks.
//    - Single string content is normalized to `[{"type": "text", "text":
//    "..."}]`.
//    - Single object content is normalized to `[...]`.
//    - The content blocks from `second` are appended to those of `first`.
//
// 2. **Tool Calls Merging**:
//    - If both messages contain `tool_calls` arrays, they are concatenated.
//
// 3. **Conflict Resolution for Other Fields**:
//    - For any other fields (e.g., `role`, custom metadata), fields from the
//      `first` message take precedence. If a field that isn't content or tool
//      calls exists in both, the value from `second` is **dropped**.
//
// This function is only used when
// `OptionalArgs::has_pending_message` is set to true when calling
// `SendMessage/Async`.
nlohmann::ordered_json MergeMessages(const nlohmann::ordered_json& first,
                                     const nlohmann::ordered_json& second) {
  // Initialize the merged message as a copy of the first message.
  nlohmann::ordered_json merged = first;

  // Merge "content".
  if (merged.contains("content") && second.contains("content")) {
    // Convert string or object content into an array.
    if (merged["content"].is_string()) {
      merged["content"] = nlohmann::ordered_json::array(
          {{{"type", "text"}, {"text", merged["content"]}}});
    } else if (merged["content"].is_object()) {
      merged["content"] = nlohmann::ordered_json::array({merged["content"]});
    }
    nlohmann::ordered_json second_content = second["content"];
    if (second_content.is_string()) {
      second_content = nlohmann::ordered_json::array(
          {{{"type", "text"}, {"text", second_content}}});
    } else if (second_content.is_object()) {
      second_content = nlohmann::ordered_json::array({second_content});
    }

    // Insert content from the second message into the result.
    if (merged["content"].is_array() && second_content.is_array()) {
      merged["content"].insert(merged["content"].end(), second_content.begin(),
                               second_content.end());

    } else {
      ABSL_LOG(ERROR) << "`content` field must be a string or an array.";
    }
  }

  // Merge "tool_calls".
  if (merged.contains("tool_calls") && second.contains("tool_calls") &&
      merged["tool_calls"].is_array() && second["tool_calls"].is_array()) {
    merged["tool_calls"].insert(merged["tool_calls"].end(),
                                second["tool_calls"].begin(),
                                second["tool_calls"].end());
  }

  // Copy over the remaining fields from the second message if they don't
  // already exist in the first. Since we don't know how to merge these
  // miscellaneous fields, we choose to let the first message's fields take
  // precedence if there's a conflict.
  if (second.is_object()) {
    for (const auto& [key, value] : second.items()) {
      if (!merged.contains(key)) {
        merged[key] = value;
      }
    }
  }

  return merged;
}

std::optional<ThinkingConfig> ResolveThinkingConfig(
    const ConversationConfig& config, const OptionalArgs& optional_args) {
  if (optional_args.thinking_config.has_value()) {
    return optional_args.thinking_config;
  }
  if (config.thinking_config().has_value()) {
    return config.thinking_config();
  }
  return std::nullopt;
}

absl::string_view TaskStateToString(TaskState task_state) {
  switch (task_state) {
    case TaskState::kUnknown:
      return "Unknown";
    case TaskState::kCreated:
      return "Created";
    case TaskState::kQueued:
      return "Queued";
    case TaskState::kProcessing:
      return "Processing";
    case TaskState::kDone:
      return "Done";
    case TaskState::kMaxNumTokensReached:
      return "MaxNumTokensReached";
    case TaskState::kFailed:
      return "Failed";
    case TaskState::kDependentTaskFailed:
      return "DependentTaskFailed";
    case TaskState::kCancelled:
      return "Cancelled";
    case TaskState::kDependentTaskCancelled:
      return "DependentTaskCancelled";
    case TaskState::kLastCallbackQueued:
      return "LastCallbackQueued";
  }
  return "Unknown";
}

class ClonedSessionTaskController : public SessionInterface::TaskController {
 public:
  ClonedSessionTaskController(
      std::unique_ptr<SessionInterface::TaskController> task_controller,
      std::shared_ptr<CachedSession> session)
      : task_controller_(std::move(task_controller)),
        session_(std::move(session)) {}

  absl::Status WaitUntilDone(absl::Duration timeout) override {
    if (task_controller_ != nullptr) {
      return task_controller_->WaitUntilDone(timeout);
    }
    return absl::OkStatus();
  }

  absl::Status Cancel() override {
    if (task_controller_ != nullptr) {
      return task_controller_->Cancel();
    }
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<SessionInterface::TaskController> task_controller_;
  std::shared_ptr<CachedSession> session_;
};

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
    bool filter_channel_content_from_kv_cache,
    bool return_error_on_parse_failure, bool return_error_on_max_tokens_reached,
    std::optional<ThinkingConfig> thinking_config, bool stream_tool_calls,
    const std::string& stream_tool_calls_channel_name, bool enable_rewinding) {
  if (preface.has_value() && !std::holds_alternative<JsonPreface>(*preface)) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }

  SessionConfig session_config_copy = session_config;
  session_config_copy.SetApplyPromptTemplateInSession(false);
  ABSL_RETURN_IF_ERROR(
      session_config_copy.MaybeUpdateAndValidate(engine.GetEngineSettings()));

  auto metadata = engine.GetEngineSettings().GetLlmMetadata();
  PromptTemplate prompt_template("");
  if (overwrite_prompt_template.has_value()) {
    prompt_template = *overwrite_prompt_template;
  } else if (metadata.has_value()) {
    if (metadata->has_jinja_prompt_template()) {
      prompt_template = PromptTemplate(metadata->jinja_prompt_template());
    } else if (metadata->has_prompt_templates()) {
      ABSL_ASSIGN_OR_RETURN(
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
      channels.push_back(litert::lm::Channel{
          .channel_name = channel.channel_name(),
          .start = channel.start(),
          .end = channel.end(),
          .is_reasoning_channel = channel.is_reasoning_channel()});
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
    ABSL_ASSIGN_OR_RETURN(processor_config,
                          CreateDataProcessorConfigFromLlmModelType(
                              session_config_copy.GetLlmModelType()));
  }

  return ConversationConfig(
      session_config_copy, preface.value_or(JsonPreface()), prompt_template,
      processor_config, enable_constrained_decoding, prefill_preface_on_init,
      std::move(constraint_provider_config), std::move(channels),
      filter_channel_content_from_kv_cache, return_error_on_parse_failure,
      return_error_on_max_tokens_reached, thinking_config, stream_tool_calls,
      stream_tool_calls_channel_name, enable_rewinding);
}

absl::StatusOr<DecodeConfig> Conversation::CreateDecodeConfig(
    std::optional<RepetitionPenaltyConfig> repetition_penalty_config,
    std::optional<NoRepeatNgramConfig> no_repeat_ngram_config,
    std::optional<SuppressTokensConfig> suppress_tokens_config,
    std::optional<ConstraintArg> decoding_constraint,
    std::optional<int> max_output_tokens,
    std::optional<ThinkingConfig> thinking_config,
    std::optional<absl::string_view> open_channel_name) {
  auto decode_config = DecodeConfig::CreateDefault();

  if (repetition_penalty_config.has_value()) {
    decode_config.SetRepetitionPenaltyConfig(
        *std::move(repetition_penalty_config));
  }

  if (no_repeat_ngram_config.has_value()) {
    decode_config.SetNoRepeatNgramConfig(*std::move(no_repeat_ngram_config));
  }

  if (suppress_tokens_config.has_value()) {
    decode_config.SetSuppressTokensConfig(*std::move(suppress_tokens_config));
  }

  if (max_output_tokens.has_value()) {
    decode_config.SetMaxOutputTokens(max_output_tokens.value());
  }
  if (thinking_config.has_value() && thinking_config->enable_thinking()) {
    decode_config.SetThinkingTokenBudget(
        thinking_config->thinking_token_budget());
    const Channel* thinking_channel = nullptr;
    // We assume the thinking channel is the first channel configured for the
    // conversation.
    // TODO(b/521921341): Support dynamically identifying or specifying the
    // thinking channel when multiple channels are present.
    if (!config_.GetChannels().empty()) {
      thinking_channel = &config_.GetChannels().front();
    }
    if (thinking_channel != nullptr) {
      if (open_channel_name.has_value() &&
          *open_channel_name == thinking_channel->channel_name) {
        decode_config.SetThinkingStartTokenIds({});
      } else {
        ABSL_ASSIGN_OR_RETURN(auto start_token_ids,
                              const_cast<Tokenizer&>(engine_.GetTokenizer())
                                  .TextToTokenIds(thinking_channel->start));
        decode_config.SetThinkingStartTokenIds(std::move(start_token_ids));
      }
      ABSL_ASSIGN_OR_RETURN(auto end_token_ids,
                            const_cast<Tokenizer&>(engine_.GetTokenizer())
                                .TextToTokenIds(thinking_channel->end));
      decode_config.SetThinkingEndTokenIds(std::move(end_token_ids));
    }
  } else {
    decode_config.SetThinkingTokenBudget(0);
  }
  if (decoding_constraint.has_value() && constraint_provider_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(constraint_,
                          constraint_provider_->CreateConstraint(
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
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                        engine.CreateSession(config.GetSessionConfig()));
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      CreateModelDataProcessor(config.GetProcessorConfig(), config.GetPreface(),
                               &engine.GetTokenizer(),
                               session->GetSessionConfig().GetStopTokenIds(),
                               config.constrained_decoding_enabled(),
                               config.GetPromptTemplate().GetCapabilities()));
  std::unique_ptr<ConstraintProvider> constraint_provider;
  if (config.constraint_provider_config().has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        constraint_provider,
        CreateConstraintProvider(
            config.constraint_provider_config().value(), engine.GetTokenizer(),
            session->GetSessionConfig().GetStopTokenIds()));
  }
  CachedSessionOptions cached_session_options;
  if (auto vision_props = engine.GetVisionExecutorProperties();
      vision_props.ok()) {
    cached_session_options.vision_properties = *vision_props;
  }
  if (auto audio_props = engine.GetAudioExecutorProperties();
      audio_props.ok()) {
    cached_session_options.audio_properties = *audio_props;
  }
  cached_session_options.insert_bos_token_id =
      session->GetSessionConfig().GetStartTokenId() >= 0;
  auto cached_session = std::make_unique<CachedSession>(
      std::move(session),
      const_cast<litert::support::Tokenizer*>(&engine.GetTokenizer()),
      cached_session_options);
  auto conversation = absl::WrapUnique(new Conversation(
      engine, std::move(cached_session), std::move(model_data_processor),
      config.GetPreface(), config.GetPromptTemplate(), config,
      std::move(constraint_provider)));
  if (config.prefill_preface_on_init() &&
      !IsEmptyPreface(config.GetPreface())) {
    PromptTemplateInput tmpl_input;
    ABSL_RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
        config.GetPreface(), conversation->model_data_processor_.get(),
        tmpl_input));
    if (config.thinking_config().has_value()) {
      tmpl_input.extra_context["enable_thinking"] =
          config.thinking_config()->enable_thinking();
    }
    tmpl_input.add_generation_prompt = false;
    ABSL_ASSIGN_OR_RETURN(std::string single_turn_text,
                          conversation->ApplyTemplate(tmpl_input));
    ABSL_ASSIGN_OR_RETURN(
        const auto session_inputs,
        conversation->model_data_processor_->ToInputDataVector(
            single_turn_text,
            std::get<JsonPreface>(config.GetPreface()).messages,
            std::monostate()));
    if (!session_inputs.empty()) {
      ABSL_RETURN_IF_ERROR(conversation->session_->RunPrefill(session_inputs));
    }
  }

  if (engine.GetEngineSettings().IsBenchmarkEnabled()) {
    ABSL_ASSIGN_OR_RETURN(BenchmarkInfo * benchmark_info,
                          conversation->GetMutableBenchmarkInfo());
    ABSL_RETURN_IF_ERROR(benchmark_info->InitPhaseRecord(
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
  const bool appending = optional_args.has_pending_message;

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

  // Trigger tasks to run in the execution manager. Necessary for the serial
  // executor, which lazily runs tasks only when they're waited on.
  // This should not slow down the threaded execution manager since it will
  // need to wait for all the session's tasks to complete anyway.
  ABSL_RETURN_IF_ERROR(session_->WaitUntilDone());
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
  const bool is_appending_message = optional_args.has_pending_message;

  std::vector<InputData> session_inputs;
  std::optional<std::string> open_channel_name;
  Message previous_last_message;
  size_t num_messages_added = 0;
  bool should_merge = false;
  bool was_appending_message = false;

  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    was_appending_message = is_appending_message_;
    is_appending_message_ = is_appending_message;

    should_merge = was_appending_message && !history_.empty() &&
                   history_.back().contains("role") &&
                   message.contains("role") &&
                   history_.back()["role"] == message["role"];
    if (should_merge) {
      previous_last_message = history_.back();
      history_.back() = MergeMessages(history_.back(), message);
    } else if (message.is_array()) {
      num_messages_added = message.size();
      for (size_t i = 0; i < message.size(); ++i) {
        history_.push_back(message[i]);
      }
    } else {
      num_messages_added = 1;
      history_.push_back(message);
    }

    auto prefill_data = GetInputDataVectorForMessages(
        /*old_messages=*/{}, history_, optional_args,
        /*include_preface=*/true,
        /*add_generation_prompt=*/!is_appending_message);
    if (!prefill_data.ok()) {
      if (should_merge) {
        if (!history_.empty()) {
          history_.back() = std::move(previous_last_message);
        }
      } else {
        for (size_t i = 0; i < num_messages_added && !history_.empty(); ++i) {
          history_.pop_back();
        }
      }
      is_appending_message_ = was_appending_message;
      return prefill_data.status();
    }
    open_channel_name =
        GetOpenChannelName(prefill_data->prefill_text, config_.GetChannels());
    session_inputs = std::move(prefill_data->session_inputs);
  }

  absl::Cleanup rollback = [&]() {
    absl::MutexLock lock(history_mutex_);
    if (should_merge) {
      if (!history_.empty()) {
        history_.back() = std::move(previous_last_message);
      }
    } else {
      for (size_t i = 0; i < num_messages_added && !history_.empty(); ++i) {
        history_.pop_back();
      }
    }
    is_appending_message_ = was_appending_message;
  };

  if (!config_.enable_rewinding()) {
    auto reset_res = session_->Reset();
    if (!reset_res.ok()) {
      return reset_res;
    }
  }

  if (is_appending_message) {
    auto task_controller = session_->RunPrefillAsync(
        session_inputs,
        [callback = std::move(user_callback)](
            absl::StatusOr<Responses> responses) mutable {
          if (!responses.ok()) {
            auto status = IgnoreEmptyInputError(responses.status());
            if (!status.ok()) {
              callback(status);
            } else {
              callback(Message());
            }
            return;
          }
          if (responses->GetTaskState() == TaskState::kDone) {
            callback(Message());
          } else if (IsTaskEndState(responses->GetTaskState())) {
            callback(absl::InternalError(absl::StrCat(
                "Prefill failed with task state: ",
                TaskStateToString(responses->GetTaskState()))));
          }
        });
    if (!task_controller.ok()) {
      return task_controller.status();
    }
    std::move(rollback).Cancel();
    AddTaskController(optional_args.task_group_id,
                      std::move(*task_controller));
    return absl::OkStatus();
  }

  absl::AnyInvocable<void(Message)> complete_message_callback =
      [this](const Message& complete_message) {
        absl::MutexLock lock(this->history_mutex_);
        this->history_.push_back(complete_message);
      };

  absl::AnyInvocable<void()> cancel_callback =
      [this, should_merge,
       previous_last_message = std::move(previous_last_message),
       num_messages_added]() mutable {
        absl::MutexLock lock(this->history_mutex_);
        if (should_merge) {
          if (!this->history_.empty()) {
            this->history_.back() = std::move(previous_last_message);
          }
        } else {
          for (size_t i = 0; i < num_messages_added && !this->history_.empty();
               ++i) {
            this->history_.pop_back();
          }
        }
      };

  auto internal_callback =
      std::make_shared<absl::AnyInvocable<void(absl::StatusOr<Responses>)>>(
          CreateInternalCallback(
              *model_data_processor_,
              optional_args.args.value_or(std::monostate()),
              config_.GetChannels(), std::move(user_callback),
              std::move(cancel_callback), std::move(complete_message_callback),
              open_channel_name, config_.return_error_on_max_tokens_reached(),
              config_.stream_tool_calls(),
              config_.stream_tool_calls_channel_name()));

  auto decode_config = CreateDecodeConfig(
      std::move(optional_args.repetition_penalty_config),
      std::move(optional_args.no_repeat_ngram_config),
      std::move(optional_args.suppress_tokens_config),
      std::move(optional_args.decoding_constraint),
      optional_args.max_output_tokens,
      ResolveThinkingConfig(config_, optional_args), open_channel_name);
  if (!decode_config.ok()) {
    return decode_config.status();
  }

  std::optional<std::string> task_group_id = optional_args.task_group_id;

  // This lambda contains the async calls to prefill and decode. It is called
  // immediately if refill_session_inputs is empty. If refill_session_inputs is
  // not empty, this lambda is called after refill_session_inputs is prefilled.
  auto run_prefill = [this, session_inputs = std::move(session_inputs),
                      internal_callback,
                      decode_config = *std::move(decode_config),
                      optional_args =
                          std::move(optional_args)]() -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(
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
              } else if (responses.ok() &&
                         IsTaskEndState(responses->GetTaskState()) &&
                         responses->GetTaskState() != TaskState::kDone) {
                (*callback)(responses);
              } else if (IsEmptyInputError(responses.status()) ||
                         (responses.ok() &&
                          responses->GetTaskState() == TaskState::kDone)) {
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

  // Run prefill for the input message.
  auto prefill_result = run_prefill();
  if (!prefill_result.ok()) {
    return prefill_result;
  }
  std::move(rollback).Cancel();
  return absl::OkStatus();
};

absl::StatusOr<Responses> Conversation::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    OptionalArgs optional_args) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<CachedSession> cloned_session,
                        session_->Clone());
  return cloned_session->RunTextScoring(target_text,
                                        /*store_token_lengths=*/true);
}

absl::Status Conversation::RunTextScoringAsync(
    const std::vector<absl::string_view>& target_text,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    OptionalArgs optional_args) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<CachedSession> cloned_session,
                        session_->CloneAsync(nullptr));
  auto shared_cloned_session =
      std::shared_ptr<CachedSession>(std::move(cloned_session));
  auto wrapped_callback =
      [shared_cloned_session, callback = std::move(callback)](
          absl::StatusOr<Responses> responses) mutable {
        callback(std::move(responses));
      };
  ABSL_ASSIGN_OR_RETURN(
      auto task_controller,
      shared_cloned_session->RunTextScoringAsync(target_text,
                                                 std::move(wrapped_callback),
                                                 /*store_token_lengths=*/true));
  auto cloned_task_controller =
      std::make_unique<ClonedSessionTaskController>(
          std::move(task_controller), shared_cloned_session);
  AddTaskController(optional_args.task_group_id,
                    std::move(cloned_task_controller));
  return absl::OkStatus();
}

absl::StatusOr<int> Conversation::GetTokenCount() const {
  return session_->GetCurrentStep();
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
  ABSL_ASSIGN_OR_RETURN(auto session, session_->Clone());
  ABSL_ASSIGN_OR_RETURN(
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
    ABSL_ASSIGN_OR_RETURN(constraint_provider,
                          CreateConstraintProvider(
                              config_.constraint_provider_config().value(),
                              engine_.GetTokenizer(),
                              session->GetSessionConfig().GetStopTokenIds()));
  }
  auto new_conversation = absl::WrapUnique(new Conversation(
      engine_, std::move(session), std::move(model_data_processor),
      config_.GetPreface(), config_.GetPromptTemplate(), config_,
      std::move(constraint_provider)));
  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    new_conversation->is_appending_message_ = is_appending_message_;
    new_conversation->history_ = history_;
  }
  return new_conversation;
}

absl::StatusOr<std::string> Conversation::RenderMessageIntoString(
    const Message& message, OptionalArgs optional_args) {
  absl::MutexLock lock(history_mutex_);
  std::vector<Message> message_vec;
  if (message.is_array()) {
    for (const auto& msg : message) {
      message_vec.push_back(msg);
    }
  } else {
    message_vec.push_back(message);
  }
  return GetPrefillTextForMessages(
      history_, message_vec, optional_args,
      /*include_preface=*/history_.empty() &&
          !config_.prefill_preface_on_init(),
      /*add_generation_prompt=*/!optional_args.has_pending_message);
}

absl::StatusOr<std::string> Conversation::RenderPrefaceIntoString(
    OptionalArgs optional_args) {
  PromptTemplateInput tmpl_input;
  ABSL_RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface_, model_data_processor_.get(), tmpl_input));

  std::optional<ThinkingConfig> resolved_thinking_config = std::nullopt;
  if (optional_args.thinking_config.has_value()) {
    resolved_thinking_config = optional_args.thinking_config;
  } else if (config_.thinking_config().has_value()) {
    resolved_thinking_config = config_.thinking_config();
  }

  if (resolved_thinking_config.has_value()) {
    tmpl_input.extra_context["enable_thinking"] =
        resolved_thinking_config->enable_thinking();
  }

  if (optional_args.extra_context.has_value()) {
    for (const auto& [key, value] : optional_args.extra_context->items()) {
      tmpl_input.extra_context[key] = value;
    }
  }

  tmpl_input.add_generation_prompt = false;
  return ApplyTemplate(tmpl_input);
}

absl::StatusOr<std::string> Conversation::GetPrefillTextForMessages(
    absl::Span<const Message> old_messages,
    absl::Span<const Message> new_messages, const OptionalArgs& optional_args,
    bool include_preface, bool add_generation_prompt) {
  // Create the template context for the `old` string.
  PromptTemplateInput old_context;
  old_context.add_generation_prompt = false;

  // Fill the `old` template context with the preface.
  ABSL_RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface_, model_data_processor_.get(), old_context));

  std::optional<ThinkingConfig> resolved_thinking_config = std::nullopt;
  if (optional_args.thinking_config.has_value()) {
    resolved_thinking_config = optional_args.thinking_config;
  } else if (config_.thinking_config().has_value()) {
    resolved_thinking_config = config_.thinking_config();
  }

  if (resolved_thinking_config.has_value()) {
    old_context.extra_context["enable_thinking"] =
        resolved_thinking_config->enable_thinking();
  }

  // Merge extra context for the message into the extra context provided in the
  // preface. Existing keys will be overwritten.
  if (optional_args.extra_context.has_value()) {
    for (const auto& [key, value] : optional_args.extra_context->items()) {
      old_context.extra_context[key] = value;
    }
  }

  // Add old messages to the `old` template context.
  for (const auto& message : old_messages) {
    ABSL_ASSIGN_OR_RETURN(
        nlohmann::ordered_json message_tmpl_input,
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
    ABSL_ASSIGN_OR_RETURN(old_string, ApplyTemplate(old_context));
  }

  // Copy the `old` template context to the `new` template context.
  PromptTemplateInput new_context = old_context;
  new_context.add_generation_prompt = add_generation_prompt;

  // Add new messages to the `new` template context.
  for (const auto& message : new_messages) {
    ABSL_ASSIGN_OR_RETURN(
        nlohmann::ordered_json message_tmpl_input,
        model_data_processor_->MessageToTemplateInput(message));
    new_context.messages.push_back(std::move(message_tmpl_input));
  }

  // Render the `new` string.
  ABSL_ASSIGN_OR_RETURN(std::string new_string, ApplyTemplate(new_context));

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

absl::StatusOr<Conversation::PrefillData>
Conversation::GetInputDataVectorForMessages(
    absl::Span<const Message> old_messages,
    absl::Span<const Message> new_messages, const OptionalArgs& optional_args,
    bool include_preface, bool add_generation_prompt) {
  ABSL_ASSIGN_OR_RETURN(
      std::string prefill_text,
      GetPrefillTextForMessages(old_messages, new_messages, optional_args,
                                include_preface, add_generation_prompt));

  nlohmann::ordered_json prefill_messages = nlohmann::ordered_json::array();
  if (include_preface) {
    if (auto* json_preface = std::get_if<JsonPreface>(&preface_)) {
      if (json_preface->messages.is_array()) {
        for (const auto& msg : json_preface->messages) {
          prefill_messages.push_back(msg);
        }
      }
    }
  }
  for (const auto& message : old_messages) {
    prefill_messages.push_back(message);
  }
  for (const auto& message : new_messages) {
    prefill_messages.push_back(message);
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<InputData> session_inputs,
      model_data_processor_->ToInputDataVector(
          prefill_text, prefill_messages,
          optional_args.args.value_or(std::monostate())));

  return PrefillData{
      .session_inputs = std::move(session_inputs),
      .prefill_text = std::move(prefill_text),
  };
}

absl::StatusOr<std::string> Conversation::ApplyTemplate(
    PromptTemplateInput& input) {
  StripBlobsFromTemplateInput(input);
  return prompt_template_.Apply(input);
}

}  // namespace litert::lm
