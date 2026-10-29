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
#include <random>
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
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
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

// Runs the passes of one verify signature of a base model. A verify pass
// processes up to W consecutive tokens like a prefill does, writing their KV
// cache entries, and also returns the logits of every input row. The verifier
// masks and samples these logits into the token that follows each row's
// prefix.
//
// The signature takes the positions `input_pos [W]`, either the token ids
// `tokens [1, W]` or the token embeddings `embeddings [1, W, D]` (and
// `per_layer_embeddings`, if the model has them), the attention masks and the
// base model's KV cache, and it returns `logits [1, W, V]`.
class LlmLiteRtVerifier {
 public:
  // Creates a verifier for the signature `signature_key` of `base_model`. If
  // the signature takes embeddings, they are looked up with
  // `embedding_lookup`, and the per-layer embeddings with
  // `per_layer_embedding_lookup`; the lookups the signature needs must not be
  // null. Padding rows take the pad token id of `executor_settings` (0 if
  // unset). The sampler is greedy until `UpdateSamplerConfig()` is called.
  // `env`, `base_model` and the embedding lookups must outlive the verifier.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtVerifier>> Create(
      Environment& env, CompiledModel& base_model,
      absl::string_view signature_key,
      const LlmExecutorSettings& executor_settings,
      EmbeddingLookupManager* embedding_lookup,
      EmbeddingLookupManager* per_layer_embedding_lookup,
      const proto::ExecutorMetadata* executor_metadata = nullptr);

  // The key of the verify signature.
  absl::string_view signature_key() const { return signature_.Key(); }

  // The number of tokens W of a pass.
  int width() const { return width_; }

  // The vocabulary size V of the logits.
  int vocab_size() const { return vocab_size_; }

  // Starts the pass of `ids`, which holds between 1 and W tokens, at the
  // positions [position, position + ids.size()). `state_buffers` are the base
  // model's KV cache buffers of this pass (see
  // `LitertState::GetStateBuffers()`).
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
      const Environment& env, CompiledModel& base_model,
      SimpleSignature signature, EmbeddingLookupManager* embedding_lookup,
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
        base_model_(base_model),
        signature_(std::move(signature)),
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

  // The base model, owned by the executor.
  CompiledModel& base_model_;

  // The verify signature of `base_model_` that the passes run.
  SimpleSignature signature_;

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

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_VERIFIER_H_
