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

#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"

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
  // - resources: Shared ModelResources container with compiled models.
  //
  // returns
  // - Unique pointer to created Qwen3TtsAcousticPredictorStage on success, or
  // error
  //   status on failure.
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
  struct DecodeOutput {
    std::vector<float> cb0_logits;  // Shape [3072]
    std::vector<float> hidden;      // Shape [1024]
  };

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

  Stage<Qwen3TtsFrontendOutput>& text_frontend_;

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

  // Runs one step of Talker model decode to produce codebook 0 logits and
  // hidden vector.
  //
  // args
  // - embed_1024: Input embedding float array of size 1024.
  // - pos: Current sequence position index.
  //
  // returns
  // - DecodeOutput containing logits and hidden state on success, or error
  // status.
  absl::StatusOr<DecodeOutput> RunDecode(const float* embed_1024, int pos);

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

  Qwen3TtsModelConfig config_;
  std::shared_ptr<ModelResources> resources_;

  // Required compiled models
  std::shared_ptr<CompiledModel> talker_model_;
  std::shared_ptr<CompiledModel> mtp_model_;
  std::shared_ptr<CompiledModel> codec_embedding_model_;
  std::shared_ptr<CompiledModel> mtp_embedding_model_;

  // Pre-allocated TensorBuffers for embedding models
  std::vector<TensorBuffer> codec_emb_input_buffers_;
  std::vector<TensorBuffer> codec_emb_output_buffers_;
  std::vector<TensorBuffer> mtp_emb_input_buffers_;
  std::vector<TensorBuffer> mtp_emb_output_buffers_;

  struct TalkerPrefillBuffers {
    int seq_len = 0;
    std::string signature_name;
    TensorBuffer emb_buf;
    TensorBuffer pos_buf;
    TensorBuffer mask_buf;
    absl::flat_hash_map<std::string, TensorBuffer> kv_output_bufs;
  };

  // Pre-allocated TensorBuffers for Talker prefill signatures (keyed by
  // seq_len)
  std::map<int, TalkerPrefillBuffers> talker_prefill_buffers_;

  TensorBuffer talker_decode_emb_buf_;
  TensorBuffer talker_decode_pos_buf_;
  TensorBuffer talker_decode_mask_buf_;
  TensorBuffer talker_decode_logits_buf_;
  absl::flat_hash_map<std::string, TensorBuffer> talker_kv_cache_bufs_0_;
  absl::flat_hash_map<std::string, TensorBuffer> talker_kv_cache_bufs_1_;

  // Pre-allocated TensorBuffers for MTP model
  TensorBuffer mtp_embed_buf_;
  TensorBuffer mtp_pos_buf_;
  TensorBuffer mtp_mask_buf_;
  TensorBuffer mtp_logits_out_buf_;
  absl::flat_hash_map<std::string, TensorBuffer> mtp_kv_cache_bufs_0_;
  absl::flat_hash_map<std::string, TensorBuffer> mtp_kv_cache_bufs_1_;

  std::vector<std::string> talker_kv_names_;
  lm::ModelSignatures mtp_signatures_;
  std::string mtp_signature_name_;
  std::string mtp_embed_input_name_;
  std::string mtp_pos_input_name_;
  std::string mtp_mask_input_name_;
  std::string mtp_logits_output_name_;
  std::vector<std::string> mtp_k_input_names_;
  std::vector<std::string> mtp_v_input_names_;
  std::vector<std::string> mtp_k_output_names_;
  std::vector<std::string> mtp_v_output_names_;
  int talker_cache_len_ = 512;
  int mtp_cache_len_ = 32;
  std::mt19937_64 rng_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_ACOUSTIC_PREDICTOR_STAGE_H_
