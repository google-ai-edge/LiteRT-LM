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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_QWEN3_TTS_QWEN3_TTS_ACOUSTIC_PREDICTOR_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_QWEN3_TTS_QWEN3_TTS_ACOUSTIC_PREDICTOR_STAGE_H_

#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/base/stateful_litert_runner.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_model_config.h"

namespace litert::omni::tts {

// Derives and validates the MTP KV cache sequence length from the MTP
// attention mask tensor buffer's shape (`[1, ..., 1, cache_len]`).
//
// args
// - mask_buffer: Non-state attention mask input TensorBuffer of the MTP model.
//
// returns
// - Positive cache sequence length (>= `qwen3_tts::kNumCodeGroups`) on
//   success, or `absl::InvalidArgumentError` (or LiteRT buffer error) if the
//   tensor type or layout dimensions are invalid.
absl::StatusOr<int> DeriveMtpCacheLen(const TensorBuffer& mask_buffer);

// Output layout of the MTP drafter's per-step (`mtp_step`) signature.
enum class MtpStepOutputLayout {
  // Legacy models: the step emits the logits of all 15 residual codebook
  // heads (`15 * qwen3_tts::kMtpVocab` floats).
  kAllHeadsLogits,
  // Per-head models: the step emits the final hidden state
  // (`qwen3_tts::kHiddenDim` floats), and each codebook head is a separate
  // `head_<i>` signature.
  kHiddenState,
};

// Resolves the MTP step output layout from the packed size of the step's
// non-state output tensor.
//
// args
// - step_output_bytes: Packed byte size of the MTP step output tensor.
//
// returns
// - The matching `MtpStepOutputLayout` on success, or
//   `absl::InvalidArgumentError` if the size matches neither layout.
absl::StatusOr<MtpStepOutputLayout> ResolveMtpStepOutputLayout(
    size_t step_output_bytes);

// Validates the I/O buffers of a per-codebook MTP head signature
// (`head_<i>`): exactly one `qwen3_tts::kHiddenDim`-float input and one
// `qwen3_tts::kMtpVocab`-float logits output.
//
// args
// - signature: Name of the head signature, used in error messages.
// - input_buffers: Input buffers created for the head signature.
// - output_buffers: Output buffers created for the head signature.
//
// returns
// - `absl::OkStatus()` if the buffers match, or `absl::InvalidArgumentError`
//   (or a LiteRT buffer error) otherwise.
absl::Status ValidateMtpHeadBuffers(
    absl::string_view signature, const std::vector<TensorBuffer>& input_buffers,
    const std::vector<TensorBuffer>& output_buffers);

// Returns the argmax token of codebook head `head_idx` from a logits buffer
// that holds one or more consecutive `qwen3_tts::kMtpVocab`-float heads
// (the legacy all-heads step output, or a single `head_<i>` output with
// `head_idx = 0`).
//
// args
// - logits_buffer: Float32 buffer of `N * qwen3_tts::kMtpVocab` logits.
// - head_idx: Index of the head whose logits slice is evaluated.
//
// returns
// - Token ID in `[0, qwen3_tts::kMtpVocab)` on success, or
//   `absl::InvalidArgumentError` if `head_idx` is out of range for the
//   buffer (or a LiteRT buffer error).
absl::StatusOr<int> ArgmaxMtpHeadLogits(const TensorBuffer& logits_buffer,
                                        int head_idx);

// Stage 2: Acoustic Predictor (Talker + MTP Autoregressive RVQ Generation)
class Qwen3TtsAcousticPredictorStage
    : public SingleThreadedStageWithDeque<Qwen3TtsAcousticOutput> {
 public:
  // Creates a Qwen3TtsAcousticPredictorStage instance and initializes its
  // resources.
  //
  // args
  // - text_frontend: Stage providing input Qwen3TtsFrontendOutput prompts.
  // - config: Configuration for the Qwen3TtsAcousticPredictorStage.
  // - resources: Shared ModelResources container with LM runners and compiled
  // models.
  //
  // returns
  // - Unique pointer to created Qwen3TtsAcousticPredictorStage on success, or
  // error status.
  static absl::StatusOr<std::unique_ptr<Qwen3TtsAcousticPredictorStage>> Create(
      Stage<Qwen3TtsFrontendOutput>* absl_nonnull text_frontend,
      const Qwen3TtsModelConfig& config,
      std::shared_ptr<ModelResources> absl_nonnull resources);

  ~Qwen3TtsAcousticPredictorStage() override = default;

  // Resets the stage state back to idle and clears any pending outputs.
  void Reset() override;

 protected:
  bool NeedScheduleInternal() const override {
    return text_frontend_.HasOutput();
  }

  // Executes one step of acoustic predictor stage processing asynchronously.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status ScheduleInternal() override;

 private:
  Qwen3TtsAcousticPredictorStage(
      Stage<Qwen3TtsFrontendOutput>* absl_nonnull text_frontend,
      Qwen3TtsModelConfig config,
      std::shared_ptr<ModelResources> absl_nonnull resources)
      : text_frontend_(*text_frontend),
        config_(std::move(config)),
        resources_(std::move(resources)) {
    if (config_.seed.has_value()) {
      rng_.seed(*config_.seed);
    } else {
      rng_.seed(0);
    }
  }

  // Runs the Talker model prefill step on input prompt embeddings to populate
  // KV cache.
  //
  // args
  // - prefill: Input prompt embedding float values.
  // - p: Number of tokens in prefill prompt embeddings.
  //
  // returns
  // - absl::OkStatus() on success, or error status on execution failure.
  absl::Status RunPrefill(const std::vector<float>& prefill, int p);

  // Runs Multi-Token Predictor (MTP) model to generate remaining codebook
  // tokens.
  //
  // args
  // - hidden: Talker hidden state float vector of size 1024.
  // - cb0: Sampled codebook 0 token ID.
  //
  // returns
  // - Vector of predicted codebook token IDs on success, or error status.
  absl::StatusOr<std::vector<int>> RunMtp(const std::vector<float>& hidden,
                                          int cb0);

  // Computes the logits of MTP codebook `head_idx` (0-based, excluding
  // codebook 0) from the output of the last MTP step.
  //
  // With per-head signatures (`head_<i>`) only that head's LM projection is
  // evaluated; legacy MTP models emit all 15 heads from the step itself, in
  // which case the matching slice is read.
  //
  // args
  // - step_output: Non-state output tensor of the last MTP step.
  // - head_idx: Index of the residual codebook head to evaluate.
  //
  // returns
  // - Argmax token ID of the selected head on success, or error status.
  absl::StatusOr<int> PickMtpToken(const TensorBuffer& step_output,
                                   int head_idx);

  // Resolves the MTP output layout: per-head signatures or legacy
  // all-heads logits.
  //
  // args
  // - mtp_model: Compiled MTP model.
  //
  // returns
  // - absl::OkStatus() on success, or error status if the layout is unknown.
  absl::Status InitMtpHeads(CompiledModel& mtp_model);

  // Embeds a single audio codec token ID using the codec embedding model.
  //
  // args
  // - code_id: Codec token ID to embed.
  //
  // returns
  // - Embedding float vector of size 1024 on success, or error status.
  absl::StatusOr<std::vector<float>> EmbedCodecToken(int code_id);

  // Embeds a sequence of MTP codebook token IDs using the MTP embedding model.
  //
  // args
  // - mtp_codes: Vector of MTP codebook token IDs to embed.
  //
  // returns
  // - Embedding float vector on success, or error status.
  absl::StatusOr<std::vector<float>> EmbedMtpTokens(
      const std::vector<int>& mtp_codes);

  // Selects a token ID from a probability logits distribution by sampling or
  // argmax.
  //
  // args
  // - logits: Unnormalized logit values float vector.
  // - do_sample: Whether to sample randomly according to softmax probabilities
  //   or select greedy argmax.
  //
  // returns
  // - Selected token index integer.
  int PickToken(const std::vector<float>& logits, bool do_sample);

  Stage<Qwen3TtsFrontendOutput>& text_frontend_;
  Qwen3TtsModelConfig config_;
  std::shared_ptr<ModelResources> resources_;

  // LM Runner for Talker
  std::shared_ptr<LiteRtLmRunner> talker_runner_;

  // Stateful runner for MTP
  std::shared_ptr<CompiledModel> mtp_model_;
  std::unique_ptr<LiteRtRunner> mtp_runner_raw_;
  std::unique_ptr<StatefulLiteRtRunner> mtp_runner_;

  // Per-codebook MTP LM heads (`head_<i>` signatures). Empty for legacy MTP
  // models whose step emits the logits of all heads at once.
  struct MtpHead {
    size_t signature_index = 0;
    std::vector<TensorBuffer> input_buffers;
    std::vector<TensorBuffer> output_buffers;
  };
  std::vector<MtpHead> mtp_heads_;

  // Embedding compiled models
  std::shared_ptr<CompiledModel> codec_embedding_model_;
  std::shared_ptr<CompiledModel> mtp_embedding_model_;

  // Pre-allocated reusable buffers for embedding models
  std::vector<TensorBuffer> codec_emb_input_buffers_;
  std::vector<TensorBuffer> codec_emb_output_buffers_;
  std::vector<TensorBuffer> mtp_emb_input_buffers_;
  std::vector<TensorBuffer> mtp_emb_output_buffers_;

  // MTP KV cache sequence length derived in Create() from the mask tensor
  // shape.
  int mtp_cache_len_ = 0;
  std::mt19937_64 rng_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_QWEN3_TTS_QWEN3_TTS_ACOUSTIC_PREDICTOR_STAGE_H_
