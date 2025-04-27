// This is a header file for the google internal version of the llm executor.
// It contains the implementation of the llm executor that is not intended to be
// open sourced.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_GOOGLE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_GOOGLE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/utils/llm_utils/constraint_utils.h"
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_base.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Context/state related data types should be excluded from the open source
// version.
// TODO(b/412847331): move the context/state related data types to a separate
// file.
//
// KVCache direct related context container.
class ProcessedContext {
 public:
  virtual ~ProcessedContext() = default;

  // Gets the LoRA id.
  virtual std::optional<uint32_t> lora_id() const = 0;

  // Gets the processed tokens.
  virtual std::vector<int>& mutable_processed_tokens() = 0;

 protected:
  ProcessedContext() = default;
  ProcessedContext(const ProcessedContext&) = default;
  ProcessedContext(ProcessedContext&&) noexcept = default;
  ProcessedContext& operator=(const ProcessedContext&) = default;
  ProcessedContext& operator=(ProcessedContext&&) noexcept = default;
};

// Struct to host the internal state for the executor.
// State will be changed by the executor while executing task.
// Noticed: The states here are all the internal states excluded those that are
// directly related to the KVCache.
struct RuntimeState {
  // The current time step.
  int current_step;

  // The id that is to be processed. -1 if all ids are processed.
  int next_input_token_id;

  // Random generator for sampling step.
  std::shared_ptr<std::default_random_engine> rand_gen;

  // The current state of the constraint.
  std::shared_ptr<::odml::infra::llm_utils::Constraint::State> constraint_state;
};

// A resource interface to hold the llm context.
class LlmContext {
 public:
  explicit LlmContext(std::unique_ptr<ProcessedContext> processed_context,
                      std::unique_ptr<RuntimeConfig> runtime_config,
                      std::unique_ptr<RuntimeState> runtime_state)
      : processed_context_(std::move(processed_context)),
        runtime_config_(std::move(runtime_config)),
        runtime_state_(std::move(runtime_state)) {};

  ~LlmContext() = default;

  // Gets the processed context.
  ProcessedContext& processed_context() { return *processed_context_; };

  // Gets the process state.
  RuntimeConfig& runtime_config() { return *runtime_config_; };

  // Gets the runtime state.
  RuntimeState& runtime_state() { return *runtime_state_; };

  // Retrieves the processed context, the caller will take the ownership of the
  // returned processed context and it will no longer be available in the
  // LlmContext. This is useful for non-duplicating the processed context while
  // extracting it.
  absl::StatusOr<std::unique_ptr<ProcessedContext>> RetrieveProcessedContext() {
    return std::move(processed_context_);
  };

  // Retrieves the runtime state, the caller will take the ownership of the
  // returned runtime state and it will no longer be available in the
  // LlmContext. This is useful for non-duplicating the runtime state while
  // extracting it.
  absl::StatusOr<std::unique_ptr<RuntimeState>> RetrieveRuntimeState() {
    return std::move(runtime_state_);
  };

  // Retrieves the runtime config, the caller will take the ownership of the
  // returned runtime config and it will no longer be available in the
  // LlmContext. This is useful for non-duplicating the runtime config while
  // extracting it.
  absl::StatusOr<std::unique_ptr<RuntimeConfig>> RetrieveRuntimeConfig() {
    return std::move(runtime_config_);
  };

 private:
  std::unique_ptr<ProcessedContext> processed_context_;
  std::unique_ptr<RuntimeConfig> runtime_config_;
  std::unique_ptr<RuntimeState> runtime_state_;

 protected:
  LlmContext() = default;
  LlmContext(LlmContext&&) noexcept = default;
  LlmContext& operator=(LlmContext&&) noexcept = default;
};

// Function type to create a new context.
using ContextCreationFunction = absl::StatusOr<std::unique_ptr<LlmContext>> (*)(
    std::optional<uint32_t>, const RuntimeConfig&);

// TODO(b/412847331): provide better documentation.
class LlmExecutorGoogle : public LlmExecutorBase {
 public:
  // The DecodeToLogits API is used specifically by AICore at the moment and
  // is not planned to be open sourced.
  //
  // This function will generate KV cache
  // entries for all the tokens in the input_tokens, and generate logits for
  // the tokens listed in the logits_indices. There's no direct support for
  // batching, the caller is responsible for flattening the batches into
  // virtual batches.
  // DecodeToLogits is used to generate logits for the given set of input
  // tokens.
  //
  // Args:
  // model_size: The model size is the number of tokens that can be processed
  //   in a single forward pass.
  // cache_size: The cache size is the number of tokens that can be stored in
  //   the KV cache.
  // input_tokens: 1D tensor buffer of input tokens to generate logits for.
  // local_attention_mask: 2D tensor buffer of the local attention mask for the
  //   input tokens, in shape of [input_tokens.size(), max_sequence_length].
  // global_attention_mask: 2D tensor buffer of the global attention mask for
  //   the input tokens, in shape of [input_tokens.size(), max_sequence_length].
  // seq_position: 1D tensor buffer of the sequence position for the input
  //   tokens, same size as the input_tokens.
  // logits_indices: Array of indices of the tokens for which it will generate
  //   output logits.
  // cancel: A flag to cancel the operation.
  // output_logits: 2D tensor buffer of the output logits in shape of
  //   [logits_indices.size(), VOCAB_SIZE].
  virtual absl::Status DecodeToLogits(
      const uint32_t model_size, const uint32_t cache_size,
      const ::litert::TensorBuffer& input_tokens,
      const ::litert::TensorBuffer& local_attention_mask,
      const ::litert::TensorBuffer& global_attention_mask,
      const ::litert::TensorBuffer& seq_position,
      absl::Span<const uint32_t> logits_indices, const std::atomic_bool* cancel,
      ::litert::TensorBuffer& output_logits) {
    return absl::UnimplementedError(absl::StrCat(
        "DecodeToLogits not implemented for backend: ", ExecutorBackendName()));
  };

  // State management related APIs are not used in the open source version.
  //
  // TODO b/405224841 - Simplify context-related APIs.
  // ------------State/context management APIs------------:
  // This function will return a function that can be used to create an empty
  // context. The creation of new context can then be done without the executor
  // by simply executing the returned function.
  virtual absl::StatusOr<ContextCreationFunction> GetNewContextCreateFunction()
      const {
    return absl::UnimplementedError(absl::StrCat(
        "GetNewContextCreateFunction not implemented for backend: ",
        ExecutorBackendName()));
  };

  // Performs necessary operations to clone the current llm context from the
  // executor and returns it to the caller.
  virtual absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const {
    return absl::UnimplementedError(absl::StrCat(
        "GetContext not implemented for backend: ", ExecutorBackendName()));
  };

  // Sets the llm_context and performs necessary operations to make sure the
  // model is restored with the provided llm context.
  virtual absl::Status RestoreContext(
      std::unique_ptr<LlmContext> context_data) {
    return absl::UnimplementedError(absl::StrCat(
        "RestoreContext not implemented for backend: ", ExecutorBackendName()));
  };

  // Cleans KV cache entries at specified indices
  virtual absl::Status CleanCacheAt(absl::Span<const int32_t> indices) {
    return absl::UnimplementedError(absl::StrCat(
        "CleanCacheAt not implemented for backend: ", ExecutorBackendName()));
  };

  // Resets all of the internal states (e.g. KVCache). Loaded and used LoRA
  // models are not affected (remain loaded and in use).
  virtual absl::Status Reset() {
    return absl::UnimplementedError(absl::StrCat(
        "Reset not implemented for backend: ", ExecutorBackendName()));
  };

  // Most Getter and Setter APIs are not used in the open source version (except
  // for the GetVocabSize()).
  //
  // ------------Getter and setter APIs------------:
  // Gets the runtime configuration.
  virtual absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const {
    return absl::UnimplementedError(
        absl::StrCat("GetRuntimeConfig not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Updates the runtime configuration.
  virtual absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) {
    return absl::UnimplementedError(
        absl::StrCat("UpdateRuntimeConfig not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Gets the runtime state.
  virtual absl::StatusOr<RuntimeState> GetRuntimeState() const {
    return absl::UnimplementedError(
        absl::StrCat("GetRuntimeState not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Updates the runtime state.
  virtual absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) {
    return absl::UnimplementedError(
        absl::StrCat("UpdateRuntimeState not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Gets the current step of the executor.
  virtual absl::StatusOr<int> GetCurrentStep() const {
    return absl::UnimplementedError(absl::StrCat(
        "GetCurrentStep not implemented for backend: ", ExecutorBackendName()));
  };

  // Sets the current step of the executor. The new step can only be less than
  // or equal to the current step of the executor.
  virtual absl::Status SetCurrentStep(int new_step) {
    return absl::UnimplementedError(absl::StrCat(
        "SetCurrentStep not implemented for backend: ", ExecutorBackendName()));
  };

  // Gets the processed tokens of the executor. This is used by resource manager
  // to check if processed context copying is needed.
  virtual absl::StatusOr<const std::vector<int>*> GetProcessedTokens() const {
    return absl::UnimplementedError(
        absl::StrCat("processed_tokens not implemented for backend: ",
                     ExecutorBackendName()));
  }

  virtual absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>>
  GetVariants() {
    return absl::UnimplementedError(absl::StrCat(
        "GetVariants not implemented for backend: ", ExecutorBackendName()));
  };

  // LoRA-related APIs are not used in the open source version.
  //
  // ------------LoRA APIs------------:
  // Loads the LoRA model into tensor loader, but does not use it.
  // To use the lora weights, call `RestoreContext()` with the lora_id.
  // Args:
  // lora_id: The unique id to assign to the loaded LoRA model.
  // lora_rank: The rank of the provided LoRA model. Used for validation.
  // model_assets: Contains the LoRA model to load.
  virtual absl::Status LoadLoRA(uint32_t lora_id, int lora_rank,
                                const ModelAssets& model_assets) {
    return absl::UnimplementedError(absl::StrCat(
        "LoadLoRA not implemented for backend: ", ExecutorBackendName()));
  };

  // TODO(b/382711815): remove when opencl RestoreContext is implemented.
  // DO_NOT_USE: This is a temporary solution (see b/382711815).
  virtual absl::Status UseLoRA(std::optional<uint32_t> lora_id) {
    return absl::UnimplementedError(
        "DO NOT USE: UseLoRA is a temporary solution (see b/382711815).");
  };
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_GOOGLE_H_
