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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/state_interface.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Result of lenient prefix acceptance.
struct LenientAcceptResult {
  // Number of accepted draft tokens. The bonus token is
  // `verifier_ids[accepted]`.
  int accepted = 0;
  // Of `accepted`, how many were accepted only because of lenience (i.e. the
  // verifier sample disagreed with the draft).
  int lenient_accepted = 0;
};

// Lenient acceptance for greedy (one-hot) drafts with sampled verification.
//
// `verifier_ids[i]` is a sample y_i ~ p_i from the (processed) verifier
// distribution at draft position i, and `drafted[i]` is the greedy draft x_i.
// At each position the draft is accepted if y_i == x_i; otherwise it is still
// accepted with probability `lenience` (using an independent uniform
// `uniform(i)` in [0, 1)); else y_i is emitted and acceptance stops.
//
// This samples exactly from r_i = (1 - c) * p_i + c * delta(x_i):
//   P(accept) = p(x) + (1 - p(x)) * c = (1 - c) * p(x) + c = r(x),
// and on rejection the emitted token is y_i conditioned on y_i != x_i, i.e.
// p_i with x_i removed and renormalized, which is the residual
// norm(max(0, p - r)) of r. `lenience == 0` is identical to
// `CountAcceptedPrefix` and never calls `uniform`; `lenience >= 1` accepts
// every draft token.
//
// `uniform` is only called at positions where y_i != x_i.
inline LenientAcceptResult CountAcceptedPrefixWithLenience(
    absl::Span<const int> verifier_ids, absl::Span<const int> drafted,
    float lenience, absl::FunctionRef<double(int)> uniform) {
  const int limit =
      static_cast<int>(std::min(verifier_ids.size(), drafted.size()));
  LenientAcceptResult result;
  while (result.accepted < limit) {
    const int i = result.accepted;
    if (verifier_ids[i] == drafted[i]) {
      ++result.accepted;
      continue;
    }
    if (lenience > 0.0f && uniform(i) < static_cast<double>(lenience)) {
      ++result.accepted;
      ++result.lenient_accepted;
      continue;
    }
    break;
  }
  return result;
}

// Returns the number of accepted draft tokens (longest matching prefix between
// `verifier_ids` and `drafted`). The bonus token is `verifier_ids[result]`.
//
// Precondition: `verifier_ids.size() == drafted.size() + 1`. Safe (never reads
// out of bounds) if the precondition is violated.
inline int CountAcceptedPrefix(absl::Span<const int> verifier_ids,
                               absl::Span<const int> drafted) {
  const int limit =
      static_cast<int>(std::min(verifier_ids.size(), drafted.size()));
  int accepted = 0;
  while (accepted < limit && verifier_ids[accepted] == drafted[accepted]) {
    ++accepted;
  }
  return accepted;
}

// Owns a random number generator seeded with the last requested seed. The
// generator is only recreated when the seed changes, so that repeated requests
// with the same seed continue its stream instead of restarting it.
class SeededRandomEngine {
 public:
  std::shared_ptr<std::default_random_engine> Get(int32_t seed) {
    if (engine_ == nullptr || seed != seed_) {
      engine_ = std::make_shared<std::default_random_engine>(seed);
      seed_ = seed;
    }
    return engine_;
  }

 private:
  std::shared_ptr<std::default_random_engine> engine_;
  int32_t seed_ = 0;
};

// Owns the random number generator of the lenient acceptance test. It is a
// different engine than the verifier's, seeded with a salted seed, so that
// lenience doesn't perturb the sampled tokens. It restarts together with the
// verifier's stream, i.e. when the seed changes or the verifier is handed a
// different generator (e.g. a new session's), so that lenient runs with the
// same seed are reproducible. Repeated updates with the same seed and generator
// continue its stream.
class LenienceRandomEngine {
 public:
  void Update(int32_t seed,
              const std::shared_ptr<std::default_random_engine>& verifier_gen) {
    if (seed != seed_ || verifier_gen != verifier_gen_.lock()) {
      engine_.seed(SaltedSeed(seed));
      seed_ = seed;
      verifier_gen_ = verifier_gen;
    }
  }

  std::mt19937_64& engine() { return engine_; }

 private:
  static uint64_t SaltedSeed(int32_t seed) {
    constexpr uint64_t kLenienceSeedSalt = 0x4c454e49;  // "LENI"
    return static_cast<uint64_t>(static_cast<uint32_t>(seed)) ^
           kLenienceSeedSalt;
  }

  std::mt19937_64 engine_{SaltedSeed(0)};
  int32_t seed_ = 0;
  // Not owned: only used to detect a new verifier generator. An expired
  // pointer never compares equal to a live one.
  std::weak_ptr<std::default_random_engine> verifier_gen_;
};

// Statistics of the last successful `LlmLiteRtMtpDrafter::Draft()` call.
struct MtpRoundStats {
  // Number of draft steps (γ).
  int drafted_tokens = 0;

  // Number of drafted tokens accepted by the verifier.
  int accepted_tokens = 0;

  // Number of tokens emitted: `accepted_tokens` plus the bonus token.
  int emitted_tokens = 0;

  // Of `accepted_tokens`, the drafts accepted only because of lenience (see
  // `LlmLiteRtMtpDrafter::SetLenience()`).
  int lenient_accepted_tokens = 0;

  // Wall time of `RunDraftingLoop()`.
  absl::Duration drafting_time;

  // Wall time of `PrepareVerifier*()` and `RunVerification()`.
  absl::Duration verify_time;
};

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

  // Reconfigures the verifier sampler to sample from `sampler_params` with the
  // shared random number generator `rand_gen` (typically the session sampler's
  // configuration and generator), recreating it if its top-k capacity is too
  // small. If `rand_gen` is null, the verifier uses its own generator seeded
  // with `sampler_params.seed()`, which is only reseeded when the seed changes.
  // The drafter sampler stays greedy, which keeps the exact-match acceptance
  // lossless with respect to the verifier's distribution.
  absl::Status UpdateVerifierSamplerConfig(
      const proto::SamplerParameters& sampler_params,
      std::shared_ptr<std::default_random_engine> rand_gen);

  // Sets the lenience factor c in [0, 1] of the acceptance test. With c > 0,
  // a draft token that the verifier's sample disagrees with is still accepted
  // with probability c. c = 0 (the default) is exact.
  // This is a quality/speed trade-off chosen by the caller.
  absl::Status SetLenience(float lenience);
  float lenience() const { return lenience_; }

  const MtpRoundStats& last_round_stats() const { return last_round_stats_; }

  // Returns the number of tokens drafted per round (γ). A round verifies
  // `num_draft_steps() + 1` positions.
  int num_draft_steps() const { return num_draft_steps_; }

 private:
  LlmLiteRtMtpDrafter(
      CompiledModel mtp_drafter_model, SimpleSignature drafter_signature,
      CompiledModel& base_model, SimpleSignature verify_signature,
      const Model& base_model_desc, EmbeddingLookupManager& embedding_manager,
      std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager,
      std::unique_ptr<Sampler> drafter_sampler,
      std::unique_ptr<Sampler> verifier_sampler, const Environment& env,
      Backend sampler_backend, ActivationDataType verifier_logits_data_type,
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
      AttentionMaskParams attn_params,
      std::unique_ptr<LiteRtLogitMaskRunner> drafter_mask_runner,
      std::unique_ptr<LiteRtLogitMaskRunner> verifier_mask_runner)
      : mtp_drafter_model_(std::move(mtp_drafter_model)),
        drafter_signature_(std::move(drafter_signature)),
        base_model_(base_model),
        base_model_desc_(base_model_desc),
        verify_signature_(std::move(verify_signature)),
        embedding_manager_(embedding_manager),
        ple_manager_(ple_manager),
        drafter_sampler_(std::move(drafter_sampler)),
        verifier_sampler_(std::move(verifier_sampler)),
        env_(env),
        sampler_backend_(sampler_backend),
        verifier_logits_data_type_(verifier_logits_data_type),
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
        attn_params_(attn_params),
        drafter_mask_runner_(std::move(drafter_mask_runner)),
        verifier_mask_runner_(std::move(verifier_mask_runner)) {
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
    std::vector<std::unique_ptr<LogitMask>> draft_masks;
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
      DraftingResult& drafting_result);

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
  // Sampler with batch size 1 and sequence size G + 1, where G is the number
  // of draft steps. Greedy unless reconfigured by
  // `UpdateVerifierSamplerConfig()`.
  std::unique_ptr<Sampler> verifier_sampler_;
  // Arguments needed to recreate `verifier_sampler_`.
  const Environment& env_;
  const Backend sampler_backend_;
  const ActivationDataType verifier_logits_data_type_;
  // The top-k capacity of `verifier_sampler_`.
  int verifier_sampler_max_top_k_ = 1;
  // The verifier's generator when no shared generator is supplied.
  SeededRandomEngine verifier_own_rand_gen_;

  // The lenience factor of the acceptance test.
  float lenience_ = 0.0f;
  // Random number generator of the lenient acceptance test, updated by
  // `UpdateVerifierSamplerConfig()`.
  LenienceRandomEngine lenience_rng_;

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
  int64_t num_drafted_tokens_ = 0;
  // The number of tokens verified by the base model (i.e., accepted) - does not
  // include the bonus token.
  int64_t num_verified_tokens_ = 0;

  MtpRoundStats last_round_stats_;

  // The vocabulary size for logits masking.
  int vocab_size_ = 0;

  // Attention mask params.
  AttentionMaskParams attn_params_;

  // Active constraint and verified constraint state.
  const Constraint* constraint_ = nullptr;
  std::unique_ptr<Constraint::State> constraint_state_;

  std::unique_ptr<LiteRtLogitMaskRunner> drafter_mask_runner_;
  std::unique_ptr<LiteRtLogitMaskRunner> verifier_mask_runner_;
};

// Updates existing LiteRT compilation options for the MTP drafter model with
// external tensor and buffer storage patterns from the executor settings.
absl::Status UpdateCompilationOptions(
    const LlmExecutorSettings& executor_settings,
    litert::Options& compilation_options);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_MTP_DRAFTER_H_
