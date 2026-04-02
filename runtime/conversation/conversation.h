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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

// Configuration for the Conversation instance. This class is used to initialize
// the Conversation instance.
//
// To create a ConversationConfig, use ConversationConfig::CreateDefault() to
// create a default config, or use the ConversationConfig::Builder() to build a
// custom config.
//
// Note: Consider to remove ConversationConfig and use ConversationBuilder to
// build Conversation.
class ConversationConfig {
 public:
  // Creates a default ConversationConfig from the given Engine.
  // Args:
  // - `engine`: The Engine instance to be used for creating the default config.
  static absl::StatusOr<ConversationConfig> CreateDefault(const Engine& engine);

  // Returns the SessionConfig used for creating the ConversationConfig.
  const SessionConfig& GetSessionConfig() const { return session_config_; }

  // Returns the Preface used for creating the ConversationConfig.
  const Preface& GetPreface() const { return preface_; }

  // Returns the PromptTemplate used for creating the ConversationConfig.
  const PromptTemplate& GetPromptTemplate() const { return prompt_template_; }

  // Returns the DataProcessorConfig used for creating the ConversationConfig.
  const DataProcessorConfig& GetProcessorConfig() const {
    return processor_config_;
  }

  // Returns whether constrained decoding is enabled.
  bool constrained_decoding_enabled() const {
    return constrained_decoding_enabled_;
  }

  // Returns whether the preface should be prefilled when the Conversation is
  // created. This will make the first response faster, but take longer to
  // initialize.
  bool prefill_preface_on_init() const { return prefill_preface_on_init_; }

  // Returns the channels configured for the conversation.
  const std::vector<Channel>& GetChannels() const { return channels_; }

 public:
  // Builder class for ConversationConfig.
  //
  // Example usage:
  //   // Create a ConversationConfig instance using the Builder.
  //   ASSIGN_OR_RETURN(auto conversation_config,
  //                    ConversationConfig::Builder()
  //                        .SetEnableConstrainedDecoding(true)
  //                        .SetPrefillPrefaceOnInit(true)
  //                        .Build(*engine));
  class Builder {
   public:
    // Sets the SessionConfig to be used for creating the ConversationConfig.
    Builder& SetSessionConfig(const SessionConfig& session_config) {
      session_config_ = session_config;
      return *this;
    }

    // Sets the Preface for the conversation. The Preface provides
    // the initial background for the conversation, tool uses and extra
    // context for the conversation. If not provided, the conversation will
    // start with an empty Preface.
    Builder& SetPreface(const Preface& preface) {
      preface_ = preface;
      return *this;
    }

    // Sets the PromptTemplate instance to be used for the conversation. If
    // not provided, the conversation will use the template read from the model
    // metadata.
    Builder& SetOverwritePromptTemplate(
        const PromptTemplate& overwrite_prompt_template) {
      overwrite_prompt_template_ = overwrite_prompt_template;
      return *this;
    }

    // Sets the configuration for the model data processor. If not provided,
    // the default config for the model type's data processor will be used.
    // Most of the time, the users don't need to provide the data processor
    // config.
    Builder& SetOverwriteProcessorConfig(
        const DataProcessorConfig& overwrite_processor_config) {
      overwrite_processor_config_ = overwrite_processor_config;
      return *this;
    }

    // Sets whether to enable constrained decoding. If true, constrained
    // decoding will be used, primarily for function calling.
    Builder& SetEnableConstrainedDecoding(bool enable_constrained_decoding) {
      enable_constrained_decoding_ = enable_constrained_decoding;
      return *this;
    }

    // Sets whether to prefill the preface on init. If true, the preface will
    // be prefilled on init, which will make the first response faster, but
    // take longer to initialize.
    Builder& SetPrefillPrefaceOnInit(bool prefill_preface_on_init) {
      prefill_preface_on_init_ = prefill_preface_on_init;
      return *this;
    }

    // Sets the configuration for the constraint provider.
    Builder& SetConstraintProviderConfig(
        const ConstraintProviderConfig& constraint_provider_config) {
      constraint_provider_config_ = constraint_provider_config;
      return *this;
    }

    // Sets the channels for the conversation.
    Builder& SetChannels(const std::vector<Channel>& channels) {
      channels_ = channels;
      return *this;
    }

    absl::StatusOr<ConversationConfig> Build(const Engine& engine) {
      return ConversationConfig::CreateInternal(
          engine, session_config_, preface_, overwrite_prompt_template_,
          overwrite_processor_config_, enable_constrained_decoding_,
          prefill_preface_on_init_, constraint_provider_config_, channels_);
    }

    // Returns a unique pointer to a ConversationConfig.
    absl::StatusOr<std::unique_ptr<ConversationConfig>> BuildUnique(
        const Engine& engine) {
      ASSIGN_OR_RETURN(ConversationConfig config, Build(engine));
      return std::make_unique<ConversationConfig>(std::move(config));
    }

   private:
    SessionConfig session_config_ = SessionConfig::CreateDefault();
    std::optional<Preface> preface_;
    std::optional<PromptTemplate> overwrite_prompt_template_;
    std::optional<DataProcessorConfig> overwrite_processor_config_;
    bool enable_constrained_decoding_ = false;
    bool prefill_preface_on_init_ = false;
    std::optional<ConstraintProviderConfig> constraint_provider_config_;
    std::optional<std::vector<Channel>> channels_ = std::nullopt;
  };

  // Returns the constrained decoding config.
  const std::optional<ConstraintProviderConfig>& constraint_provider_config()
      const {
    return constraint_provider_config_;
  }

 private:
  // Creates a ConversationConfig.
  // Args:
  // - `engine`: The Engine instance to be used to validate the SessionConfig.
  // - `session_config`: The SessionConfig to be used for creating the
  //     ConversationConfig.
  // - `preface`: Optional Preface for the conversation. The Preface provides
  //     the initial background for the conversation, tool uses and extra
  //     context for the conversation. If not provided, the conversation will
  //     start with an empty Preface.
  // - `overwrite_prompt_template`: Optional PromptTemplate instance to be used
  //     for the conversation. If not provided, the conversation will use the
  //     template read from the model metadata "jinja_prompt_template". If not
  //     provided, LiteRT-LM will try to generate a default one based on the llm
  //     model type.
  // - `overwrite_processor_config`: Optional configuration for the model data
  //     processor, if not provided, the default config for the model type's
  //     data processor will be used. Most of the time, the users don't need to
  //     provide the data processor config.
  // - `enable_constrained_decoding`: Whether to enable constrained decoding. If
  //     true, constrained decoding will be used, primarily for function
  //     calling.
  // - `prefill_preface_on_init`: Whether to prefill the preface on init. If
  //     true, the preface will be prefilled on init, which will make the first
  //     response faster, but take longer to initialize.
  // - `channels`: The channels configured for the conversation.
  static absl::StatusOr<ConversationConfig> CreateInternal(
      const Engine& engine, const SessionConfig& session_config,
      std::optional<Preface> preface = std::nullopt,
      std::optional<PromptTemplate> overwrite_prompt_template = std::nullopt,
      std::optional<DataProcessorConfig> overwrite_processor_config =
          std::nullopt,
      bool enable_constrained_decoding = false,
      bool prefill_preface_on_init = false,
      std::optional<ConstraintProviderConfig> constraint_provider_config =
          std::nullopt,
      std::optional<std::vector<Channel>> channels = std::nullopt);

  explicit ConversationConfig(SessionConfig session_config, Preface preface,
                              PromptTemplate prompt_template,
                              DataProcessorConfig processor_config,
                              bool constrained_decoding_enabled = false,
                              bool prefill_preface_on_init = false,
                              std::optional<ConstraintProviderConfig>
                                  constraint_provider_config = std::nullopt,
                              std::vector<Channel> channels = {})
      : session_config_(std::move(session_config)),
        preface_(std::move(preface)),
        prompt_template_(std::move(prompt_template)),
        processor_config_(std::move(processor_config)),
        constrained_decoding_enabled_(constrained_decoding_enabled),
        prefill_preface_on_init_(prefill_preface_on_init),
        constraint_provider_config_(std::move(constraint_provider_config)),
        channels_(std::move(channels)) {}

  SessionConfig session_config_;
  Preface preface_;
  PromptTemplate prompt_template_;
  DataProcessorConfig processor_config_;
  bool constrained_decoding_enabled_;
  bool prefill_preface_on_init_;
  std::optional<ConstraintProviderConfig> constraint_provider_config_;
  std::vector<Channel> channels_;
};

// Optional arguments for sending a message to the LLM.
struct OptionalArgs {
  // Whether there is a pending message to be sent. If true, only the prefill
  // stage of LLM will be triggered, and the following decode stage will be
  // skipped. This is useful for the case where we need to append multiple
  // messages to the conversation, but only want to generate a response once.
  //
  // To also trigger the decode stage, set this field to false. Or to explicitly
  // trigger the decode stage only, set this field to false and send an empty
  // content message.
  //
  // Note: this option is only valid for model templates and
  // ModelDataProcessor that supports single turn prompt rendering.
  //
  // Example usages:
  //
  // Append multiple messages to the conversation without triggering the decode
  // stage.
  //
  // ASSERT_OK(conversation->SendMessage(
  //   Message{{"role", "user"}, {"content", "Hello world!"}},
  //   {.has_pending_message = true}));
  //
  // ASSERT_OK(conversation->SendMessage(
  //   Message{{"role", "user"}, {"content", " This is a long message."}},
  //   {.has_pending_message = true}));
  //
  // By sending a message with has_pending_message set to false, the decode
  // stage will be triggered, and the decode result will be returned.
  //
  // ASSERT_OK(conversation->SendMessage(
  //   Message{{"role", "user"}, {"content", " This is the last message."}},
  //   {.has_pending_message = false}));
  //
  // Alternatively, send an empty message with has_pending_message set to false
  // to only trigger the decode stage.
  //
  // ASSERT_OK(conversation->SendMessage(
  //   Message{{"role", "user"}, {"content", " This is the last message."}},
  //   {.has_pending_message = true}));
  //
  // ASSERT_OK(conversation->SendMessage(
  //   Message{{"role", "user"}, {"content", ""}},
  //   {.has_pending_message = false}));
  bool has_pending_message = false;

  // The constraint to be used for constrained decoding.
  std::optional<ConstraintArg> decoding_constraint = std::nullopt;

  // The arguments for the model data processor. Most of the time, the users
  // don't need to provide this argument.
  std::optional<DataProcessorArguments> args = std::nullopt;

  // The maximum number of tokens to generate during decode.
  std::optional<int> max_output_tokens = std::nullopt;

  // The task group id for asynchronous tasks. If provided, the task
  // controller will be stored and can be cancelled by calling
  // `Conversation::CancelGroup(task_group_id)`.
  std::optional<std::string> task_group_id = std::nullopt;

  // The extra template context passed into PromptTemplateInput. This extra
  // context only applies to a single message and is merged with the extra
  // context provided in the Preface, overwriting existing keys.
  std::optional<nlohmann::ordered_json> extra_context = std::nullopt;
};

// A multi-turn centric stateful Conversation API for high-level user
// interaction. Conversation maintains the history for users, so the users'
// messages will be used as the LLM context through the conversation.
//
// Conversation handles the complex data processing logic for Session usage,
// including:
// - Prompt template rendering.
// - Role-based messages handling.
// - Multimodal input processing.
// - History management.
// - Model-specific data processing.
//
// Example usage:
//
//   // Create an Engine instance.
//   ASSIGN_OR_RETURN(auto engine, Engine::Create(model_assets));
//
//   // Create a ConversationConfig instance from the Engine.
//   ASSIGN_OR_RETURN(auto conversation_config,
//                    ConversationConfig::CreateDefault(*engine));
//
//   // Create a Conversation instance.
//   ASSIGN_OR_RETURN(auto conversation,
//       Conversation::Create(*engine, conversation_config));
//
//   // Send a message to the LLM and returns the complete message.
//   ASSIGN_OR_RETURN(const Message message,
//                    conversation->SendMessage(Message{
//                        {"role", "user"}, {"content", "Hello world!"}}));
//
//   // Send a message to the LLM and process the asynchronous message results
//   // via the user_callback. The user_callback is a user-defined callback
//   // function that handles the message results.
//   EXPECT_OK(conversation->SendMessageAsync(
//       Message{{"role", "user"}, {"content", "Hello world!"}},
//       [](absl::StatusOr<Message> message) {
//         // Handle the message results.
//         if (message.ok()) {
//           std::cout << "Message: " << std::endl;
//         }
//       });
//
class Conversation {
 public:
  // Creates a Conversation instance from the the Engine and ConversationConfig.
  // Args:
  // - `engine`: The Engine instance to be used for creating the Conversation.
  // - `config`: The ConversationConfig instance to be used for creating the
  // Conversation.
  static absl::StatusOr<std::unique_ptr<Conversation>> Create(
      Engine& engine, const ConversationConfig& config);

  // Sends a message to the LLM and returns the complete message.
  // Args:
  // - `message`: The message to be sent to the LLM. If `message` is an array,
  //    each element will be treated as a separate message and be prefilled
  //    before generating the response.
  // - `optional_args`: The optional arguments for sending the message. See the
  //    definition of `OptionalArgs` for more details.
  // Returns :
  // - The complete message from the LLM.
  absl::StatusOr<Message> SendMessage(
      const Message& message, OptionalArgs optional_args = OptionalArgs());

  // Sends a message to the LLM and process the asynchronous message results via
  // the user_callback.
  // Args:
  // - `message`: The message to be sent to the LLM. If `message` is an array,
  //    each element will be treated as a separate message and be prefilled
  //    before generating the response.
  // - `user_callback`: The callback to receive the message events. The
  //    user_callback will be invoked in the following conditions:
  //    - On every new message chunk.
  //    - When the generation is complete, the user_callback will be invoked
  //      with an empty message.
  //    - When the generation is cancelled, the user_callback will be invoked
  //      with absl::CancelledError.
  //    - When an error occurs, the user_callback will be invoked with the error
  //      status.
  // - `optional_args`: The optional arguments for sending the message. See the
  //    definition of `OptionalArgs` for more details.
  // Returns :
  // - absl::OkStatus if the message is sent and processing successfully,
  //   otherwise the error status.
  absl::Status SendMessageAsync(
      const Message& message,
      absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
      OptionalArgs optional_args = OptionalArgs());

  // Scores the target text after the prefill process is done. This function
  // will run the decode process (with the existing context history) by feeding
  // in the provided target text tokens and fetch the decode output logits that
  // corresponds to the target text tokens. This is useful for running certain
  // scoring metrics, e.g. perplexity.
  // Note that the function will NOT update the conversation history or the
  // internal state of the Conversation. The existing context history will
  // remain the same after the function call.
  // Note also that the function will NOT apply any additional prompt template
  // to the target text as the goal is to get the score of the raw target text.
  // Args:
  //   - target_text: The target text to score.
  //   - returns: This function returns the score associated with each of the
  //     target texts. The scores are the log likelihood of the target text
  //     given the existing context history.
  absl::StatusOr<Responses> RunTextScoring(
      const std::vector<absl::string_view>& target_text,
      OptionalArgs optional_args = OptionalArgs());

  // Similar to the above RunTextScoring function, but this is a not blocking
  // call and the function will return right away. The processing status will
  // be signaled through the callback.
  absl::Status RunTextScoringAsync(
      const std::vector<absl::string_view>& target_text,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      OptionalArgs optional_args = OptionalArgs());

  // Returns the history of the conversation.
  // Note: the return value is a copy of the history, which may be expensive
  // for large history.
  std::vector<Message> GetHistory() const {
    absl::MutexLock lock(&history_mutex_);  // NOLINT
    return history_;
  }

  // Provides safe access to the conversation history without copying.
  // The provided visitor function is executed while the history mutex is held.
  // Args:
  // - visitor: The visitor function takes a const reference to the history
  //  vector.
  //
  // Example usage:
  //
  //   Message assistant_message;
  //   conversation->AccessHistory(
  //       [&assistant_message](const std::vector<Message>& history) {
  //         // Copy the last message to assistant_message. So we don't need to
  //         // copy the whole history, if we only need the last message.
  //         assistant_message = history.back();
  //       });
  void AccessHistory(absl::AnyInvocable<void(const std::vector<Message>&) const>
                         visitor) const {
    absl::MutexLock lock(&history_mutex_);  // NOLINT
    visitor(history_);
  }

  // Returns the configuration used for creating the Conversation.
  const ConversationConfig& GetConfig() const { return config_; }

  // Returns the benchmark info for the conversation. Under the hood, this
  // method triggers the benchmark info collection from the Session. Returns:
  // - The benchmark info for the conversation.
  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo();

  // Returns the mutable benchmark info for the conversation. Under the hood,
  // this method triggers the mutable benchmark info collection from the
  // Session. Returns:
  // - The mutable benchmark info for the conversation.
  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo();

  // Cancels the ongoing inference process, for asynchronous inference.
  // Note: the underlying Session is not rollbacked, so the message
  // from the user is actually sent to the LLM and processed for prefill.
  void CancelProcess();

  // Clones the conversation. The cloned conversation will be independent of the
  // original conversation, including the history, state, etc.
  //
  // Note that the cloned conversation will not clone the group_id of the
  // ongoing tasks.
  absl::StatusOr<std::unique_ptr<Conversation>> Clone();

  // Cancels all ongoing asynchronous tasks with the given task_group_id.
  // Args:
  // - `task_group_id`: The id of the task group to cancel.
  // Note: after the cancellation, there is no guarantee that the internal state
  // of the Conversation is intact and therefore it is recommended to not
  // continue using the Conversation after cancellation.
  void CancelGroup(absl::string_view task_group_id);

 private:
  explicit Conversation(
      Engine& engine, std::unique_ptr<Engine::Session> session,
      std::unique_ptr<ModelDataProcessor> model_data_processor, Preface preface,
      PromptTemplate prompt_template, ConversationConfig config,
      std::unique_ptr<ConstraintProvider> constraint_provider = nullptr)
      : engine_(engine),
        model_data_processor_(std::move(model_data_processor)),
        preface_(preface),
        prompt_template_(std::move(prompt_template)),
        config_(config),
        constraint_provider_(std::move(constraint_provider)),
        session_(std::move(session)) {}

  absl::StatusOr<std::string> GetSingleTurnText(
      const Message& message, const OptionalArgs& optional_args);

  absl::StatusOr<std::string> GetSingleTurnTextFromFullHistory(
      const Message& message, const OptionalArgs& optional_args);

  absl::StatusOr<std::string> GetSingleTurnTextFromSingleTurnTemplate(
      const Message& message, const OptionalArgs& optional_args);

  absl::StatusOr<DecodeConfig> CreateDecodeConfig(
      std::optional<ConstraintArg> decoding_constraint = std::nullopt,
      std::optional<int> max_output_tokens = std::nullopt);

  // Adds a task controller to the task_controllers_ map if task_group_id is
  // provided.
  // Args:
  // - `task_group_id`: The id of the task group to add the controller to.
  // - `task_controller`: The task controller to add.
  void AddTaskController(
      const std::optional<std::string>& task_group_id,
      std::unique_ptr<Engine::Session::TaskController> task_controller);

  // Keep a reference to the creator engine to enable access to the shared
  // resources that might be required for features like cloning.
  Engine& engine_;
  std::unique_ptr<ModelDataProcessor> model_data_processor_;
  Preface preface_;
  PromptTemplate prompt_template_;
  // The constraint is currently created from the tools defined in the preface,
  // if any.
  std::unique_ptr<Constraint> constraint_;
  const ConversationConfig config_;
  std::unique_ptr<ConstraintProvider> constraint_provider_ = nullptr;
  mutable absl::Mutex history_mutex_;
  std::vector<Message> history_ ABSL_GUARDED_BY(history_mutex_);

  // Whether the current conversation is in message appending state.
  bool is_appending_message_ = false;

  // Mutex for task_controllers_.
  mutable absl::Mutex task_controllers_mutex_;
  // Map of task group id to task controllers.
  absl::flat_hash_map<
      std::string,
      std::vector<std::unique_ptr<Engine::Session::TaskController>>>
      task_controllers_ ABSL_GUARDED_BY(task_controllers_mutex_);

  // Declare the session after model_data_processor_ and other members it
  // depends on so that the session is destroyed before them. This is to avoid
  // memory corruption and null-pointer deference issues.
  std::unique_ptr<Engine::Session> session_;
};
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_
