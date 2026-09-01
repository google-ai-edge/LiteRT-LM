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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_MTP_DRAFTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_MTP_DRAFTER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/state_interface.h"

namespace litert::lm {

class LlmLiteRtMtpDrafter {
 public:
  ~LlmLiteRtMtpDrafter();

  // Creates an LlmLiteRtMtpDrafter from model resources.
  //
  // The executor_settings is used to create the MTP drafter model and its
  // sampler.
  // The base_model is used for verification. The model is expected to have a
  // "verify" signature and be invokable when Draft is called (i.e., not busy).
  static absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>> Create(
      Environment& env, ModelResources& resources,
      const LlmExecutorSettings& executor_settings, CompiledModel& base_model,
      EmbeddingLookupManager& embedding_manager,
      std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager,
      const proto::ExecutorMetadata* executor_metadata = nullptr);

  // Creates an LlmLiteRtMtpDrafter directly from a pre-compiled
  // litert::CompiledModel.
  //
  // This is primarily intended for use with pre-compiled models, where the
  // caller compiles the model beforehand and constructs the drafter directly.
  // In this case, the caller will also need to call UpdateCompilationOptions
  // to ensure that the compilation options are set correctly before compiling
  // the model themselves.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>> Create(
      Environment& env, CompiledModel mtp_drafter_model,
      const LlmExecutorSettings& executor_settings, CompiledModel& base_model,
      const Model& base_model_desc, EmbeddingLookupManager& embedding_manager,
      std::optional<std::reference_wrapper<EmbeddingLookupManager>>
          ple_manager = std::nullopt,
      const proto::ExecutorMetadata* executor_metadata = nullptr);

  // Draft the next set of tokens using the MTP drafter model.
  // Inputs:
  //   position: The current position of the input sequence.
  //   token_id: The id of the last input token.
  //   activations: Activations corresponding to the token_id. This is only
  //    required for the first invocation of the drafter.
  //   state: The model's runtime KV cache state.
  //   constraint: Optional decoding constraint to apply during drafting and
  //    verification.
  // Outputs:
  //   The drafted tokens from the MTP drafter model with the shape:
  //   [batch_size, num_tokens].
  absl::StatusOr<std::vector<std::vector<int>>> Draft(
      int position, int token_id, std::optional<TensorBuffer> activations,
      StateInterface& state, const Constraint* constraint = nullptr);

 private:
  LlmLiteRtMtpDrafter(
      CompiledModel mtp_drafter_model, SimpleSignature drafter_signature,
      CompiledModel& base_model, SimpleSignature verify_signature,
      const Model& base_model_desc, EmbeddingLookupManager& embedding_manager,
      std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager,
      std::unique_ptr<Sampler> drafter_sampler,
      std::unique_ptr<Sampler> verifier_sampler,
      std::vector<std::string> kv_cache_input_names,
      absl::flat_hash_map<absl::string_view, TensorBuffer>
          drafter_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>
          drafter_output_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>
          verifier_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>
          verifier_output_buffers,
      TensorBuffer drafter_id_tensor, TensorBuffer verifier_id_tensor,
      int num_draft_steps, ModelSignatures drafter_signatures,
      ModelSignatures verifier_signatures, int vocab_size,
      AttentionMaskParams attn_params)
      : mtp_drafter_model_(std::move(mtp_drafter_model)),
        drafter_signature_(std::move(drafter_signature)),
        base_model_(base_model),
        base_model_desc_(base_model_desc),
        verify_signature_(std::move(verify_signature)),
        embedding_manager_(embedding_manager),
        ple_manager_(ple_manager),
        drafter_sampler_(std::move(drafter_sampler)),
        verifier_sampler_(std::move(verifier_sampler)),
        kv_cache_input_names_(std::move(kv_cache_input_names)),
        drafter_input_buffers_(std::move(drafter_input_buffers)),
        drafter_output_buffers_(std::move(drafter_output_buffers)),
        verifier_input_buffers_(std::move(verifier_input_buffers)),
        verifier_output_buffers_(std::move(verifier_output_buffers)),
        drafter_id_tensor_(std::move(drafter_id_tensor)),
        verifier_id_tensor_(std::move(verifier_id_tensor)),
        num_draft_steps_(num_draft_steps),
        drafter_signatures_(std::move(drafter_signatures)),
        verifier_signatures_(std::move(verifier_signatures)),
        vocab_size_(vocab_size),
        attn_params_(attn_params) {
    for (const auto& [name, buffer] : drafter_input_buffers_) {
      auto expected = buffer.Duplicate();
      active_drafter_input_buffers_[name] = std::move(expected.Value());
    }
    for (const auto& [name, buffer] : drafter_output_buffers_) {
      auto expected = buffer.Duplicate();
      active_drafter_output_buffers_[name] = std::move(expected.Value());
    }
    for (const auto& [name, buffer] : verifier_input_buffers_) {
      auto expected = buffer.Duplicate();
      active_verifier_input_buffers_[name] = std::move(expected.Value());
    }
    for (const auto& [name, buffer] : verifier_output_buffers_) {
      auto expected = buffer.Duplicate();
      active_verifier_output_buffers_[name] = std::move(expected.Value());
    }
  }

  struct DraftingResult {
    std::vector<int> drafted_tokens;
    std::vector<std::unique_ptr<Constraint::State>> draft_constraint_states;
  };

  absl::Status PrepareDrafterInputBuffers(
      int position, absl::flat_hash_map<absl::string_view, TensorBuffer>&
                        output_kv_cache_buffers);

  absl::Status PrepareDrafterOutputBuffers();

  absl::StatusOr<DraftingResult> RunDraftingLoop(
      int token_id, std::optional<TensorBuffer>& activations,
      const Constraint* constraint,
      const Constraint::State* verified_constraint_state);

  absl::Status PrepareVerifierInputBuffers(
      int position, int token_id, const std::vector<int>& drafted_tokens,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          input_kv_cache_buffers);

  absl::Status PrepareVerifierOutputBuffers(
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          output_kv_cache_buffers);

  absl::StatusOr<std::vector<int>> RunVerification(
      const std::vector<std::unique_ptr<Constraint::State>>&
          draft_constraint_states);

  // The MTP drafter model.
  CompiledModel mtp_drafter_model_;
  SimpleSignature drafter_signature_;

  // The base model, used for verification. The model is owned by the base
  // LiteRtCompiledModelExecutor.
  CompiledModel& base_model_;
  const Model& base_model_desc_;
  SimpleSignature verify_signature_;

  EmbeddingLookupManager& embedding_manager_;
  std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager_;

  // Greedy sampler with batch size 1.
  std::unique_ptr<Sampler> drafter_sampler_;
  // Greedy sampler with batch size G + 1, where G is the number of draft steps.
  std::unique_ptr<Sampler> verifier_sampler_;

  // The names of the key/value cache input tensors for the MTP drafter model.
  std::vector<std::string> kv_cache_input_names_;

  // MTP drafter owned buffers. This includes tokens, positions, results.
  //   - input_position [batch, sequence_length]
  //   - mask [batch, 1, sequence_length = 1, context]
  //   - activations [batch, sequence_length = 1, hidden_size * 2]
  absl::flat_hash_map<absl::string_view, TensorBuffer> drafter_input_buffers_;
  //   - logits [batch, sequence_length, vocab_size]
  //   - projected_logits [batch, sequence_length, hidden_size]
  absl::flat_hash_map<absl::string_view, TensorBuffer> drafter_output_buffers_;

  // Verifier owned buffers.
  //   - input_position [batch, draft_steps + 1]
  //   - mask [batch, 1, draft_steps + 1, context]
  //   - embeddings [batch, draft_steps + 1, hidden_size]
  //   - per_layer_embeddings [batch, draft_steps + 1, ...]
  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_input_buffers_;
  //   - logits [batch, draft_steps + 1, vocab_size]
  //   - activations [batch, draft_steps + 1, hidden_size]
  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_output_buffers_;

  // Cached maps for Run to avoid map creation overhead.
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      active_drafter_input_buffers_;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      active_drafter_output_buffers_;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      active_verifier_input_buffers_;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      active_verifier_output_buffers_;

  // Pre-allocated temporary tensors for sampling.
  TensorBuffer drafter_id_tensor_;
  TensorBuffer verifier_id_tensor_;

  // The number of draft steps.
  const int num_draft_steps_;

  // Model signatures for drafter and verifier.
  ModelSignatures drafter_signatures_;
  ModelSignatures verifier_signatures_;

  // The index of the last verified token in the verifier output buffers.
  int last_verified_token_id_idx_ = -1;

  // Misc statistics for the MTP drafter.
  // The number of tokens drafted by the MTP drafter model, regardless of
  // whether they are verified or not - does not include the bonus token.
  int num_drafted_tokens_ = 0;
  // The number of tokens verified by the base model (i.e., accepted) - does not
  // include the bonus token.
  int num_verified_tokens_ = 0;

  // The vocabulary size for logits masking.
  int vocab_size_ = 0;

  // Attention mask params.
  AttentionMaskParams attn_params_;

  // Active constraint and verified constraint state.
  const Constraint* constraint_ = nullptr;
  std::unique_ptr<Constraint::State> constraint_state_;
};

// Updates existing LiteRT compilation options for the MTP drafter model with
// external tensor and buffer storage patterns from the executor settings.
absl::Status UpdateCompilationOptions(
    const LlmExecutorSettings& executor_settings,
    litert::Options& compilation_options);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_MTP_DRAFTER_H_
