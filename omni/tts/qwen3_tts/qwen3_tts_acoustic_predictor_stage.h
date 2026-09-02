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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_ACOUSTIC_PREDICTOR_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_ACOUSTIC_PREDICTOR_STAGE_H_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/base/stateful_litert_runner.h"
#include "omni/tts/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"

namespace litert::omni::tts {

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
  std::unique_ptr<LiteRtRunner> mtp_runner_raw_;
  std::unique_ptr<StatefulLiteRtRunner> mtp_runner_;

  // Embedding compiled models
  std::shared_ptr<CompiledModel> codec_embedding_model_;
  std::shared_ptr<CompiledModel> mtp_embedding_model_;

  // Pre-allocated reusable buffers for embedding models
  std::vector<TensorBuffer> codec_emb_input_buffers_;
  std::vector<TensorBuffer> codec_emb_output_buffers_;
  std::vector<TensorBuffer> mtp_emb_input_buffers_;
  std::vector<TensorBuffer> mtp_emb_output_buffers_;

  int mtp_cache_len_ = 32;
  std::mt19937_64 rng_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_ACOUSTIC_PREDICTOR_STAGE_H_
