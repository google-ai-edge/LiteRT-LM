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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// A single trainable parameter (e.g., a LoRA weight tensor).
struct TrainableParameter {
  std::string name;
  float* data;
  size_t num_elements;
  bool is_bias_or_layernorm;
};

// Handle that holds trainable parameters and keeps their memory alive.
// The float* pointers in the parameters are valid as long as this handle
// exists and the owning session is alive.
class TrainableParameterHandle {
 public:
  virtual ~TrainableParameterHandle() = default;
  virtual const std::vector<TrainableParameter>& GetParameters() const = 0;
};

// Engine is the interface for the LLM runtime. It is responsible for
// - Initializing the LLM model and related resources, e.g. tokenizer,
//   embedder, etc.
// - Providing the APIs to create the Session.
//
// The Session is responsible for hosting the internal state (e.g. conversation
// history) of each separate interaction with LLM. It is created by the Engine
// and is responsible for:
// - Generating content from the input prompt/query.
// - Running the prefill and decode processes.
//
// Example usage:
//   // Create the model assets.
//   auto model_assets = ModelAssets::Create(model_path);
//   CHECK_OK(model_assets);
//
//   // Create the engine.
//   auto engine = Engine::CreateEngine(EngineSettings::CreateDefault(
//       model_assets, litert::lm::Backend::CPU));
//   CHECK_OK(engine);
//
//   // Create the session.
//   auto session = engine->CreateSession(SessionConfig::CreateDefault());
//   CHECK_OK(session);
//
//   // Run generate content.
//   auto responses = (*session)->GenerateContent({InputText("What's the tallest
//   building in the world?")});
//   CHECK_OK(responses);
//
//   // Print the response.
//   std::cout << *responses << std::endl;
class Engine {
 public:
  virtual ~Engine() = default;

  // Session is responsible for hosting the internal state (e.g. conversation
  // history) of each separate interaction with LLM.
  class Session {
   public:
    // The TaskController is responsible for controlling the async task
    // execution.
    class TaskController {
     public:
      TaskController() = default;

      // The TaskController is not copyable. This is to avoid
      // the user from accidentally copying the TaskController and calling the
      // CancelProcess function multiple times.
      TaskController(const TaskController&) = delete;
      TaskController& operator=(const TaskController&) = delete;

      // The TaskController is movable.
      TaskController(TaskController&&) = default;
      TaskController& operator=(TaskController&&) = default;

      // The TaskController destructor.
      virtual ~TaskController() = default;

      // Waits until all the tasks are done or the timeout is reached. The
      // function will return error if the timeout is reached.
      virtual absl::Status WaitUntilDone(absl::Duration timeout) {
        return absl::UnimplementedError("Not implemented.");
      };

      // Cancels the ongoing inference process. Note that if this function is
      // called after the inference process is done, the function will be a
      // no-op.
      virtual absl::Status Cancel() {
        return absl::UnimplementedError("Not implemented.");
      };
    };

    virtual ~Session() = default;

    // High-level API to generate content from the input prompt/query. This
    // function will handle the prefill and decode processes internally and
    // the usage is similar to the Gemini Text Generation API
    // (https://ai.google.dev/gemini-api/docs/text-generation).
    // - contents: The input data for generation.
    virtual absl::StatusOr<Responses> GenerateContent(
        const std::vector<InputData>& contents) = 0;

    // This is a not blocking call and the function will return right away. The
    // result will be streamed through the callback.
    //
    // - contents: The input data for generation.
    // - callback: Callback to receive streamed results.
    //   Note:
    //     - If the generation is done successfully, the callback will be
    //       called with empty responses to signal the completion.
    //     - If there is an error during the streaming process, the callback
    //       will be called with the error status and no further results will be
    //       sent.
    //     - If the generation is cancelled, the callback will be called
    //       with a Cancellation error.
    virtual absl::Status GenerateContentStream(
        const std::vector<InputData>& contents,
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

    // Same as above, but with a custom decode config.
    // - decode_config: configuration for the model decode process.
    virtual absl::Status GenerateContentStream(
        const std::vector<InputData>& contents,
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
        const DecodeConfig& decode_config) = 0;

    // Scores the target text after the prefill process is done. This function
    // will only run the decode process to fetch the decode output logits, which
    // is used to calculate the target text's score and update the model memory
    // using the target_text tokens.
    // This function should be called after the prefill process is done.
    // - target_text: The target text to score.
    // - store_token_lengths: Whether to store the token lengths of the target
    //   texts in `Responses`.
    // - returns: This function returns the score associated with the target
    // text after the model has been prefilled. The returned score is the sum of
    // the negative log probability of seeing the target text during decode.
    virtual absl::StatusOr<Responses> RunTextScoring(
        const std::vector<absl::string_view>& target_text,
        bool store_token_lengths) = 0;

    // Similar to the above RunTextScoring function, but this is a not blocking
    // call and the function will return right away. The processing status will
    // be signaled through the callback.
    // - target_text: The target text to score.
    // - callback: Callback to receive the scoring results.
    // - store_token_lengths: Whether to store the token lengths of the target
    //   texts in `Responses`.
    virtual absl::StatusOr<std::unique_ptr<TaskController>> RunTextScoringAsync(
        const std::vector<absl::string_view>& target_text,
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
        bool store_token_lengths) {
      return absl::UnimplementedError("Not implemented.");
    }

    // Adds the input prompt/query to the model for starting the prefilling
    // process. Note that the user can break down their prompt/query into
    // multiple chunks and call this function multiple times.
    //
    // This is a blocking call and the function will return when the prefill
    // process is done.
    virtual absl::Status RunPrefill(const std::vector<InputData>& contents) = 0;

    // This is a not blocking call and the function will return right away. The
    // processing status will be signaled through the callback.
    virtual absl::StatusOr<std::unique_ptr<TaskController>> RunPrefillAsync(
        const std::vector<InputData>& contents,
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
      return absl::UnimplementedError("Not implemented.");
    }

    // Starts the decoding process for the model to predict the response based
    // on the input prompt/query added after using RunPrefill* functions.
    // This is a blocking call and the function will return when the decoding
    // process is done.
    virtual absl::StatusOr<Responses> RunDecode() = 0;

    // Same as above, but with a custom decode config.
    // - decode_config: configuration for the model decode process.
    virtual absl::StatusOr<Responses> RunDecode(
        const DecodeConfig& decode_config) = 0;

    // Startes the decoding process for the model to predict the response based
    // on the input prompt/query added after using RunPrefill* functions.
    // This is a not blocking call and the function will return right away. The
    // result will be streamed through the callback.
    // - callback: Callback to receive streamed results.
    virtual absl::StatusOr<std::unique_ptr<TaskController>> RunDecodeAsync(
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
      return absl::UnimplementedError("Not implemented.");
    }

    // Same as above, but with a custom decode config.
    // - decode_config: configuration for the model decode process.
    virtual absl::StatusOr<std::unique_ptr<TaskController>> RunDecodeAsync(
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
        const DecodeConfig& decode_config) {
      return absl::UnimplementedError("Not implemented.");
    }

    // Returns the benchmark info for the session. Returns error if the
    // benchmark is not enabled.
    virtual absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() = 0;

    // Returns the mutable benchmark info for the session. Returns error if the
    // benchmark is not enabled.
    virtual absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo() = 0;

    // Cancels the ongoing inference process. Note that if this function is
    // called, the inference process will return with a kCancelled error. The
    // session could still be used after afterwards.
    virtual void CancelProcess() {
      ABSL_LOG(FATAL) << "CancelProcess is not implemented.";
    }

    // Waits until all the tasks are done or the default timeout is reached.
    virtual absl::Status WaitUntilDone() = 0;

    // Get the reference to the session config for the session.
    virtual const SessionConfig& GetSessionConfig() const = 0;

    // Get the reference to the tokenizer for the session.
    virtual const Tokenizer& GetTokenizer() const = 0;

    // Returns a handle to the trainable parameters (e.g., LoRA weights) for
    // fine-tuning. The returned float* pointers are valid as long as both the
    // handle and the session are alive.
    virtual absl::StatusOr<std::unique_ptr<TrainableParameterHandle>>
    GetTrainableParameters() {
      return absl::UnimplementedError("Not implemented.");
    }
  };

  // Method to create the Session.
  virtual absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) = 0;

  // Waits until the engine is done with all the tasks. The function will
  // return error if the timeout is reached.
  virtual absl::Status WaitUntilDone(absl::Duration timeout) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Loads a LoRA adapter from the given path and assigns it the given ID.
  // The adapter must be loaded before creating sessions that reference it.
  virtual absl::Status LoadLoRA(uint32_t lora_id,
                                const std::string& lora_path) {
    return absl::UnimplementedError("LoadLoRA not implemented.");
  }

  // Returns the EngineSettings currently used by the engine.
  virtual const EngineSettings& GetEngineSettings() const = 0;

  // Default timeout duration for the engine/session processes.
  static constexpr absl::Duration kDefaultTimeout = absl::Minutes(10);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_
