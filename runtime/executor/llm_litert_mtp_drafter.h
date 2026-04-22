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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

class LlmLiteRtMtpDrafter {
 public:
  ~LlmLiteRtMtpDrafter();

  // Create an instance of LlmLiteRtMtpDrafter.
  // The executor_settings is used to create the MTP drafter model and its
  // sampler.
  // The base_model is used for verification. The model is expected to have
  // "verify" signature and be invokable when Draft is called (i.e., not busy).
  static absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>> Create(
      Environment& env, ModelResources& resources,
      const LlmExecutorSettings& executor_settings, CompiledModel base_model,
      EmbeddingLookupManager& embedding_manager,
      EmbeddingLookupManager& ple_manager);

  // Draft the next set of tokens using the MTP drafter model.
  // Inputs:
  //   position: The current position of the input sequence.
  //   token_id: The id of the last input token.
  //   activations: Activations corresponding to the token_id. This is only
  //    required for the first invocation of the drafter.
  //   input_kv_cache_buffers:  The key/value cache buffers for the base model,
  //    used to draft and start the verification process.
  //   output_kv_cache_buffers: The key/value cache buffers for the base model,
  //    used to store the key/value cache for the drafted tokens through
  //    verification.
  // Outputs:
  //   The drafted tokens from the MTP drafter model with the shape:
  //   [batch_size, num_tokens].
  absl::StatusOr<std::vector<std::vector<int>>> Draft(
      int position, int token_id, std::optional<TensorBuffer> activations,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          output_kv_cache_buffers);

 private:
  LlmLiteRtMtpDrafter(CompiledModel mtp_drafter_model, CompiledModel base_model,
                      SimpleSignature drafter_signature,
                      SimpleSignature verify_signature,
                      EmbeddingLookupManager& embedding_manager,
                      EmbeddingLookupManager& ple_manager,
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
                      int num_draft_steps)
      : mtp_drafter_model_(std::move(mtp_drafter_model)),
        base_model_(std::move(base_model)),
        drafter_signature_(std::move(drafter_signature)),
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
        num_draft_steps_(num_draft_steps) {
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

  absl::Status PrepareDrafterInputBuffers(
      int position, absl::flat_hash_map<absl::string_view, TensorBuffer>&
                        output_kv_cache_buffers);

  absl::Status PrepareDrafterOutputBuffers();

  absl::StatusOr<std::vector<int>> RunDraftingLoop(
      int token_id, std::optional<TensorBuffer>& activations);

  absl::Status PrepareVerifierInputBuffers(
      int position, int token_id, const std::vector<int>& drafted_tokens,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          input_kv_cache_buffers);

  absl::Status PrepareVerifierOutputBuffers(
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          output_kv_cache_buffers);

  absl::StatusOr<std::vector<int>> RunVerification();

  // The MTP drafter model.
  CompiledModel mtp_drafter_model_;

  // The base model, used for verification.
  CompiledModel base_model_;
  SimpleSignature drafter_signature_;
  SimpleSignature verify_signature_;

  EmbeddingLookupManager& embedding_manager_;
  EmbeddingLookupManager& ple_manager_;

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

  // The index of the last verified token in the verifier output buffers.
  int last_verified_token_id_idx_ = -1;

  // Misc statistics for the MTP drafter.
  // The number of tokens drafted by the MTP drafter model, regardless of
  // whether they are verified or not - does not include the bonus token.
  int num_drafted_tokens_ = 0;
  // The number of tokens verified by the base model (i.e., accepted) - does not
  // include the bonus token.
  int num_verified_tokens_ = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_MTP_DRAFTER_H_
