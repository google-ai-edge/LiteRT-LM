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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_VERIFIER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_VERIFIER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/state_interface.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Draft verification with the verify signatures of a base model:
//   - `LlmLiteRtVerifier` runs the passes of one verify signature. The MTP
//     drafter verifies each round with one pass of its `verify` signature.
//   - `LlmLiteRtDraftVerifier` verifies drafts of any length for the executor,
//     with one `LlmLiteRtVerifier` per usable verify signature (see
//     `FindVerifySignatures()`).
//   - `DraftConstraintStates` gives the rows of a pass their constraint masks.
//   - The free functions below plan the passes and decide which draft tokens
//     a pass accepts, independently of the model.
//
// Draft verification runs verify passes over the pending token (the last
// context token, not processed yet) and the draft tokens. A pass of width W
// takes the pending token at `position` and up to W - 1 draft tokens; the
// token it samples after its last input checks one more draft token. A pass
// that accepts all of these W draft tokens hands the last one over as the
// pending token of the next pass.

// Returns InvalidArgument if `draft_token_ids` is empty or holds a token id
// outside the vocabulary of `vocab_size` tokens.
absl::Status ValidateDraftTokenIds(absl::Span<const int> draft_token_ids,
                                   int vocab_size);

// Returns the width of the next pass, which takes the pending token at
// `position` and `num_drafts` draft tokens, among the verify signature widths
// `widths`: the smallest width that takes all of them, else the largest one.
// Only widths whose pass fits into the KV cache of `kv_cache_size` entries
// count. Returns nullopt if none fits.
std::optional<int> SelectVerifyWidth(absl::Span<const int> widths,
                                     int num_drafts, int position,
                                     int kv_cache_size);

// The outcome of a verify pass.
struct VerifyPassOutcome {
  // The number of draft tokens that the pass accepted.
  int num_accepted = 0;

  // The token decoded after the accepted draft tokens: a correction of the
  // first rejected draft token, or the token after the last draft token. Null
  // if the pass accepted a draft token after its last input, which becomes the
  // pending token of the next pass.
  std::optional<int> next_token;

  // Of `num_accepted`, the draft tokens accepted only because of lenience.
  int num_lenient_accepted = 0;
};

// Returns how many of the unverified draft tokens `drafts` a verify pass
// accepted: the longest prefix that matches its sampled tokens. The pass fed
// the pending token plus the first `num_draft_inputs` draft tokens (fewer than
// `drafts.size()` if the pass is too narrow), and `pass_tokens[i]` is its
// sampled token for `drafts[i]`. The last one checks one draft token beyond
// the inputs, if there is one.
// Requires `pass_tokens.size() == num_draft_inputs + 1` and
// `num_draft_inputs <= drafts.size()`.
VerifyPassOutcome EvaluateVerifyPass(absl::Span<const int> pass_tokens,
                                     absl::Span<const int> drafts,
                                     int num_draft_inputs);

// Like above, but a draft token x that differs from its sampled token is still
// accepted with probability `lenience` c, if `uniform(i)` (a uniform random
// number in [0, 1) for draft token i, only drawn then) is below c. If the
// sampled token follows p, x is accepted with probability (1 - c) * p(x) + c,
// and the new token follows (1 - c) * p + c * delta(x). c = 0 is exact.
VerifyPassOutcome EvaluateVerifyPass(absl::Span<const int> pass_tokens,
                                     absl::Span<const int> drafts,
                                     int num_draft_inputs, float lenience,
                                     absl::FunctionRef<double(int)> uniform);

// A verify signature of a base model; see `FindVerifySignatures()`.
struct VerifySignatureInfo {
  std::string key;
  // The number of tokens W of a pass, or 0 if the signature is unusable.
  int width = 0;
  // OK if draft verification can use the signature, otherwise the reason why
  // it can't.
  absl::Status status;
};

// Finds the verify signatures of `model`, i.e. the signatures whose key starts
// with "verify", in the model's order, and checks whether draft verification
// can use them with the KV cache `state` of the decode signature
// `decode_signature_key`. A verify signature is usable if
//   - `selected_signatures` is empty or holds it,
//   - it takes `input_pos [W]` and either `tokens [1, W]`, or `embeddings`
//     (and `per_layer_embeddings`) for which there are embedding lookups, but
//     not both,
//   - it returns `logits [1, W, V]` with decode's vocabulary size V,
//   - it has the same KV cache tensors, param tensor and LoRA inputs as
//     decode, no other inputs than these, the attention mask and the
//     positions, and no local attention mask (the entries of rejected tokens
//     would overwrite the context in a ring-buffer KV cache), and
//   - W doesn't exceed the KV cache size.
// Returns an error only if the model can't be inspected.
absl::StatusOr<std::vector<VerifySignatureInfo>> FindVerifySignatures(
    CompiledModel& model, absl::string_view decode_signature_key,
    const StateInterface& state,
    absl::Span<const std::string> selected_signatures,
    bool has_embedding_lookup, bool has_per_layer_embedding_lookup);

// Owns a random number generator seeded with the last requested seed. The
// generator is only recreated when the seed changes, so that repeated requests
// with the same seed continue its stream instead of restarting it.
class SeededRandomEngine {
 public:
  // Returns the generator for `seed`. The generator is created on the first
  // call and recreated whenever `seed` differs from the previous call's seed;
  // otherwise the same generator is returned.
  std::shared_ptr<std::default_random_engine> Get(int32_t seed) {
    if (engine_ == nullptr || seed != seed_) {
      engine_ = std::make_shared<std::default_random_engine>(seed);
      seed_ = seed;
    }
    return engine_;
  }

 private:
  // The current generator, or null before the first call to `Get()`.
  std::shared_ptr<std::default_random_engine> engine_;

  // The seed that `engine_` was created with.
  int32_t seed_ = 0;
};

// The constraint states along a draft, and the logit masks that they give the
// draft tokens: state i follows the first i draft tokens, and its mask applies
// to draft token i, i.e. to row i of a verify pass over the draft.
//
// The MTP drafter appends each token after sampling it under its mask.
// `LlmLiteRtVerifier::Verify()` appends the draft tokens of its pass that are
// missing, masks the rows, and returns the state that follows the accepted
// draft tokens.
class DraftConstraintStates {
 public:
  // Starts the draft at the state `start`, which must outlive this object.
  DraftConstraintStates(const Constraint& constraint,
                        const Constraint::State& start);

  // Starts the draft at the state `start`, which this object takes over.
  DraftConstraintStates(const Constraint& constraint,
                        std::unique_ptr<Constraint::State> start);

  // The number of draft tokens appended so far.
  int num_tokens() const { return static_cast<int>(states_.size()) - 1; }

  // Returns the mask of the next draft token, which is computed on the first
  // call and owned by this object.
  absl::StatusOr<const LogitMask*> NextMask();

  // Appends `token` to the draft, after computing its mask (`NextMask()`) if
  // that hasn't happened yet. Fails, leaving the draft as is, if the
  // constraint disallows `token` (see `AdvanceConstraintState()`).
  absl::Status Append(int token);

  // Returns the masks of the rows of a verify pass of width `width` over the
  // draft: the masks of the appended draft tokens and of the next one, then
  // nulls for the padding rows. Requires `num_tokens() < width`.
  absl::StatusOr<absl::Span<const std::unique_ptr<LogitMask>>> PassMasks(
      int width);

  // Moves out the state that follows the first `n` draft tokens, to finish
  // with the draft. Returns null if that is the start state and this object
  // doesn't own it.
  absl::StatusOr<std::unique_ptr<Constraint::State>> TakeState(int n);

 private:
  // The constraint, owned by the caller.
  const Constraint* constraint_;

  // The start state, owned by `states_[0]` or by the caller.
  const Constraint::State* start_;

  // `states_[i]` owns the state that follows the first i draft tokens.
  // `states_[0]` is null if the caller owns the start state.
  std::vector<std::unique_ptr<Constraint::State>> states_;

  // `masks_[i]` is the mask of draft token i, null until computed.
  std::vector<std::unique_ptr<LogitMask>> masks_;
};

// The result of `LlmLiteRtVerifier::Verify()`.
struct VerifyPassResult {
  // The accepted draft tokens and the new pending token.
  VerifyPassOutcome outcome;

  // With constraint states, the state that follows the accepted draft tokens,
  // except an accepted draft token that becomes the pending token of the next
  // pass. Null without constraint states, or if that is a start state that
  // they don't own (see `DraftConstraintStates::TakeState()`).
  std::unique_ptr<Constraint::State> constraint_state;
};

// Runs the passes of one verify signature of a base model. A verify pass
// processes up to W consecutive tokens like a prefill does, writing their KV
// cache entries, and also returns the logits of every input row. The verifier
// masks and samples these logits into the token that follows each row's
// prefix. The MTP drafter uses one directly, and `LlmLiteRtDraftVerifier` holds
// one per usable verify signature.
//
// The signature takes the positions `input_pos [W]`, either the token ids
// `tokens [1, W]` or the token embeddings `embeddings [1, W, D]` (and
// `per_layer_embeddings`, if the model has them), the attention masks and the
// base model's KV cache, and it returns `logits [1, W, V]`.
class LlmLiteRtVerifier {
 public:
  // Starts a pass of the verify signature `signature_key` with
  // `input_buffers` and `output_buffers`, which hold all the inputs and
  // outputs of the pass, including the KV cache. A runner may bind further
  // inputs, e.g. LoRA weights, before it starts the pass.
  using PassRunner = absl::AnyInvocable<absl::Status(
      absl::string_view signature_key,
      absl::flat_hash_map<absl::string_view, TensorBuffer>& input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>& output_buffers)>;

  // Creates a verifier for the signature `signature_key` of `base_model`. If
  // the signature takes embeddings, they are looked up with
  // `embedding_lookup`, and the per-layer embeddings with
  // `per_layer_embedding_lookup`; the lookups the signature needs must not be
  // null. Padding rows take the pad token id of `executor_settings` (0 if
  // unset). The sampler is greedy until `UpdateSamplerConfig()` is called.
  // The passes start with `pass_runner`, e.g. so that they run like the
  // executor's prefill and decode runs, or directly on `base_model` if it is
  // null. `env`, `base_model` and the embedding lookups must outlive the
  // verifier.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtVerifier>> Create(
      Environment& env, CompiledModel& base_model,
      absl::string_view signature_key,
      const LlmExecutorSettings& executor_settings,
      EmbeddingLookupManager* embedding_lookup,
      EmbeddingLookupManager* per_layer_embedding_lookup,
      const proto::ExecutorMetadata* executor_metadata = nullptr,
      PassRunner pass_runner = nullptr);

  // The key of the verify signature.
  absl::string_view signature_key() const { return signature_.Key(); }

  // The number of tokens W of a pass.
  int width() const { return width_; }

  // The vocabulary size V of the logits.
  int vocab_size() const { return vocab_size_; }

  // Starts the pass of `ids`, which holds between 1 and W tokens, at the
  // positions [position, position + ids.size()). `state_buffers` are the base
  // model's KV cache buffers of this pass (see
  // `LitertState::GetStateBuffers()`). The pass starts with the pass runner.
  //
  // A pass with fewer than W tokens is padded like a prefill: the padding rows
  // take the pad token id (or zero embeddings), position 0 and no attention.
  // Depending on the model, they may still write the KV cache entries of the
  // positions [position + ids.size(), position + W), so the caller must make
  // sure that position + W does not exceed the KV cache size, and must not
  // rely on these entries.
  absl::Status Run(int position, absl::Span<const int> ids,
                   const LitertState::StateBuffers& state_buffers);

  // Applies `masks` to the logits of the last pass, one mask per row (a null
  // mask leaves its row as is), and returns the sampled token of every row of
  // the pass's tokens: the i-th token follows the first i + 1 tokens of the
  // pass. `masks` is either empty (no masking) or holds W masks, including
  // the padding rows'.
  absl::StatusOr<std::vector<int>> MaskAndSample(
      absl::Span<const std::unique_ptr<LogitMask>> masks);

  // Verifies `drafts`, the draft tokens after `pending_token` at `position`,
  // in a pass that takes the pending token and up to W - 1 draft tokens (see
  // `Run()` for `state_buffers`), and accepts them like
  // `EvaluateVerifyPass()`.
  //
  // With `constraint_states`, which start at the state that predicts the
  // first draft token and may already hold some of the pass's draft tokens
  // (e.g. a drafter's), each row is masked with the mask of its draft token.
  // A draft token that the constraint disallows ends the draft.
  absl::StatusOr<VerifyPassResult> Verify(
      int position, int pending_token, absl::Span<const int> drafts,
      const LitertState::StateBuffers& state_buffers,
      DraftConstraintStates* constraint_states);

  // Like above, but accepts draft tokens with `lenience` and `uniform`, see
  // `EvaluateVerifyPass()`. A leniently accepted draft token isn't checked
  // against its mask, so `constraint_states` must already hold all of
  // `drafts`, sampled under their masks like a drafter's.
  absl::StatusOr<VerifyPassResult> Verify(
      int position, int pending_token, absl::Span<const int> drafts,
      const LitertState::StateBuffers& state_buffers,
      DraftConstraintStates* constraint_states, float lenience,
      absl::FunctionRef<double(int)> uniform);

  // Reconfigures the sampler to sample from `sampler_params` with the random
  // number generator `rand_gen` (typically the session sampler's), recreating
  // it if its top-k capacity is too small. If `rand_gen` is null, the sampler
  // uses the verifier's own generator, seeded with `sampler_params.seed()`
  // and only reseeded when the seed changes. Returns the generator that the
  // sampler uses from now on.
  absl::StatusOr<std::shared_ptr<std::default_random_engine>>
  UpdateSamplerConfig(const proto::SamplerParameters& sampler_params,
                      std::shared_ptr<std::default_random_engine> rand_gen);

  // Returns the output buffer `name` of the signature, e.g. the `activations`
  // of the last pass. The KV cache outputs are not part of the verifier.
  absl::StatusOr<TensorBuffer*> GetOutputBuffer(absl::string_view name);

 private:
  // Takes ownership of the objects that `Create()` built for the signature.
  LlmLiteRtVerifier(
      const Environment& env, SimpleSignature signature, PassRunner pass_runner,
      EmbeddingLookupManager* embedding_lookup,
      EmbeddingLookupManager* per_layer_embedding_lookup, int width,
      int vocab_size, int pad_token_id, ModelSignatures signatures,
      AttentionMaskParams attn_params, ActivationDataType logits_data_type,
      Backend sampler_backend, std::unique_ptr<Sampler> sampler,
      std::unique_ptr<LiteRtLogitMaskRunner> mask_runner,
      absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer> active_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>
          active_output_buffers,
      TensorBuffer id_tensor)
      : env_(env),
        signature_(std::move(signature)),
        pass_runner_(std::move(pass_runner)),
        embedding_lookup_(embedding_lookup),
        per_layer_embedding_lookup_(per_layer_embedding_lookup),
        width_(width),
        vocab_size_(vocab_size),
        pad_token_id_(pad_token_id),
        signatures_(std::move(signatures)),
        attn_params_(attn_params),
        logits_data_type_(logits_data_type),
        sampler_backend_(sampler_backend),
        sampler_(std::move(sampler)),
        mask_runner_(std::move(mask_runner)),
        output_buffers_(std::move(output_buffers)),
        active_input_buffers_(std::move(active_input_buffers)),
        active_output_buffers_(std::move(active_output_buffers)),
        id_tensor_(std::move(id_tensor)) {}

  // Writes `ids` into the `tokens` input, padded with the pad token id.
  absl::Status FillTokens(absl::Span<const int> ids);

  // Looks up the embeddings of `ids` into the input `name` with `lookup`, and
  // zeroes the padding rows.
  absl::Status FillEmbeddings(EmbeddingLookupManager& lookup,
                              absl::string_view name,
                              absl::Span<const int> ids);

  // The LiteRT environment, used to recreate `sampler_`. Owned by the caller.
  const Environment& env_;

  // The verify signature of the base model that the passes run.
  SimpleSignature signature_;

  // Starts the passes: the runner given to `Create()`, or one that runs them
  // directly on the base model.
  PassRunner pass_runner_;

  // Looks up the input embeddings of a pass. Owned by the caller; null if the
  // signature takes no embeddings.
  EmbeddingLookupManager* embedding_lookup_;

  // Looks up the per-layer embeddings of a pass. Owned by the caller; null if
  // the signature takes no per-layer embeddings.
  EmbeddingLookupManager* per_layer_embedding_lookup_;

  // The number of tokens W of a pass, read from the `input_pos [W]` input.
  const int width_;

  // The vocabulary size V, read from the `logits [1, W, V]` output.
  const int vocab_size_;

  // The token id of the padding rows of a `tokens` input.
  const int pad_token_id_;

  // The names of the signature's inputs and outputs by role, e.g. the
  // attention masks, the embeddings and the param tensor.
  const ModelSignatures signatures_;

  // The attention mask types used to fill the masks of a pass.
  const AttentionMaskParams attn_params_;

  // The number of tokens of the last pass.
  int num_pass_tokens_ = 0;

  // The data type of the logits, needed to recreate `sampler_`.
  const ActivationDataType logits_data_type_;

  // The backend of `sampler_`, needed to recreate it.
  const Backend sampler_backend_;

  // Sampler with batch size 1 and sequence size W.
  std::unique_ptr<Sampler> sampler_;

  // The top-k capacity of `sampler_`.
  int sampler_max_top_k_ = 1;

  // The sampler's generator when no shared generator is supplied.
  SeededRandomEngine own_rand_gen_;

  // Applies the logit masks of `MaskAndSample()` to all rows of the logits.
  std::unique_ptr<LiteRtLogitMaskRunner> mask_runner_;

  // The signature's outputs other than the KV cache, e.g.
  //   - logits [1, W, V]
  //   - activations [1, W, hidden_size]
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers_;

  // The input buffers of a pass: duplicates of the signature's own buffers,
  // e.g.
  //   - input_pos [W]
  //   - mask [1, 1, W, context]
  //   - tokens [1, W] or embeddings [1, W, hidden_size]
  //   - per_layer_embeddings [1, W, ...]
  // plus duplicates of the KV cache input buffers of the last pass.
  absl::flat_hash_map<absl::string_view, TensorBuffer> active_input_buffers_;

  // The output buffers of a pass: duplicates of `output_buffers_` plus
  // duplicates of the KV cache output buffers of the last pass.
  absl::flat_hash_map<absl::string_view, TensorBuffer> active_output_buffers_;

  // The sampled token ids [1, W].
  TensorBuffer id_tensor_;
};

// Verifies drafts of any length for the executor. Holds an `LlmLiteRtVerifier`
// for each usable verify signature, created on first use, and runs each pass
// with the narrowest one that takes the rest of the draft, else the widest,
// among those whose pass fits into the KV cache (see `SelectVerifyWidth()`).
class LlmLiteRtDraftVerifier {
 public:
  // Creates a draft verifier for the verify signatures of `base_model` that
  // can share the KV cache `state` of its decode signature
  // `decode_signature_key` (see `FindVerifySignatures()`). See
  // `LlmLiteRtVerifier::Create()` for the other parameters. `env`,
  // `base_model`, `executor_settings` and the embedding lookups must outlive
  // the draft verifier.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtDraftVerifier>> Create(
      Environment& env, CompiledModel& base_model,
      absl::string_view decode_signature_key, const StateInterface& state,
      const LlmExecutorSettings& executor_settings,
      EmbeddingLookupManager* embedding_lookup,
      EmbeddingLookupManager* per_layer_embedding_lookup,
      const proto::ExecutorMetadata* executor_metadata = nullptr,
      LlmLiteRtVerifier::PassRunner pass_runner = nullptr);

  // Returns OK if a verify signature is usable, else FAILED_PRECONDITION with
  // the reasons why none is.
  absl::Status CheckSupport() const;

  // Returns whether a pass that starts with the pending token at `position`
  // fits into the KV cache of `kv_cache_size` entries.
  bool PassFits(int position, int kv_cache_size) const;

  // Sets the sampler parameters and the random number generator of the
  // passes, see `LlmLiteRtVerifier::UpdateSamplerConfig()`. Greedy until set.
  void SetSamplerConfig(const proto::SamplerParameters& sampler_params,
                        std::shared_ptr<std::default_random_engine> rand_gen);

  // Verifies `drafts`, the draft tokens after `pending_token` at `position`,
  // in one pass with the KV cache `state` (see `LlmLiteRtVerifier::Verify()`).
  // Like `Decode()`, turns a negative (invalid) sampled token into 0, or into
  // an error with `AdvancedSettings::error_on_invalid_sampled_token_id`.
  //
  // With a `constrained_decoder`, the rows are masked with the constraint
  // states that follow its state, advanced by the pending token if
  // `feed_pending_token` (a decode step feeds its input token to the decoder
  // only if a decode step produced it). Afterwards, the decoder's state
  // follows the accepted draft tokens, except the new pending token, as after
  // a decode step.
  absl::StatusOr<VerifyPassOutcome> VerifyPass(
      int position, int pending_token, absl::Span<const int> drafts,
      LitertState& state, ConstrainedDecoder* constrained_decoder,
      bool feed_pending_token);

 private:
  LlmLiteRtDraftVerifier(Environment& env, CompiledModel& base_model,
                         const LlmExecutorSettings& executor_settings,
                         EmbeddingLookupManager* embedding_lookup,
                         EmbeddingLookupManager* per_layer_embedding_lookup,
                         const proto::ExecutorMetadata* executor_metadata,
                         LlmLiteRtVerifier::PassRunner pass_runner,
                         std::vector<VerifySignatureInfo> signatures);

  // Returns the verifier of the usable verify signature of width `width`.
  absl::StatusOr<LlmLiteRtVerifier*> GetOrCreateVerifier(int width);

  // The LiteRT environment of the verifiers.
  Environment& env_;

  // The base model, owned by the executor.
  CompiledModel& base_model_;

  // The executor's settings, read when a verifier is created and by the policy
  // for invalid sampled tokens.
  const LlmExecutorSettings& executor_settings_;

  // The embedding lookups of the verifiers, owned by the executor; null if the
  // model has none.
  EmbeddingLookupManager* embedding_lookup_;
  EmbeddingLookupManager* per_layer_embedding_lookup_;

  // The executor metadata, owned by the caller; may be null.
  const proto::ExecutorMetadata* executor_metadata_;

  // Starts the passes of all verifiers; null to run them directly on
  // `base_model_`.
  LlmLiteRtVerifier::PassRunner pass_runner_;

  // The model's verify signatures and whether draft verification can use
  // them.
  std::vector<VerifySignatureInfo> signatures_;

  // The widths of the usable verify signatures.
  std::vector<int> widths_;

  // The verifiers of the usable verify signatures by width, created on first
  // use.
  absl::flat_hash_map<int, std::unique_ptr<LlmLiteRtVerifier>> verifiers_;

  // The sampler parameters of the passes.
  proto::SamplerParameters sampler_params_;

  // The random number generator of the passes; null for the verifiers' own.
  std::shared_ptr<std::default_random_engine> rand_gen_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_VERIFIER_H_
