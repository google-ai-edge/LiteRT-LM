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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_STAGE_H_

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/lfm2_audio/lfm2_audio_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/utils/audio_resampler.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Represents a processed multimodal input segment or assembled turn prompt for
// `LiquidAI/LFM2.5-Audio-1.5B`.
struct Lfm2AudioEmbeddingSegment {
  // Flattened continuous embeddings of shape `[num_tokens, llm_hidden_size]`.
  std::vector<float> embeddings;

  // Token IDs (`>= 0` for text/control tokens, or
  // `config.audio_embed_placeholder_token_id` for continuous `AUDIO_IN`
  // embeddings).
  std::vector<int> token_ids;

  // Number of token positions in this segment (`token_ids.size()`).
  int num_tokens = 0;

  // True when this segment marks `EndOfInputTurn`.
  bool is_end_of_turn = false;
};

// Assembled LFM2.5-Audio turn prompt produced by `Lfm2AudioFrontendStage`.
struct Lfm2AudioFrontendOutput {
  // Flattened prompt embeddings of shape `[prompt_len, llm_hidden_size]`.
  std::vector<float> prompt_embeddings;

  // Prompt token IDs (`>= 0` for text/special tokens, or
  // `config.audio_embed_placeholder_token_id` for `AUDIO_IN` frames).
  std::vector<int> prompt_token_ids;

  // Total prompt token count.
  int prompt_len = 0;

  // Whether speech generation (`AUDIO_OUT`) is enabled for this turn.
  bool generate_audio = true;

  // Resolved generation mode (`kInterleaved` or `kSequential`) for this turn.
  Lfm2AudioGenerationMode generation_mode =
      Lfm2AudioGenerationMode::kInterleaved;
};

// Extracts NeMo `FilterbankFeatures` 128-bin log-mel spectrogram frames from
// 16 kHz mono PCM samples (`preemph = 0.97`, symmetric Hann `win_length = 400`,
// `hop_length = 160`, `n_fft = 512`, power spectrum, Slaney mel basis,
// `log(mel + 2^-24)`, and per-feature mean/std normalization over valid frames
// `0 .. seq_len - 1`).
//
// Args:
//   pcm_16k: Mono PCM audio samples at 16 kHz.
//   config: LFM2.5-Audio model configuration.
//   out_valid_frames: Output parameter receiving the number of valid mel frames
//     (`max(1, pcm_16k.size() / hop_length)`).
//
// Returns:
//   Flattened feature-major or time-major `[valid_frames, audio_mel_bins]`
//   normalized log-mel spectrogram, or an error status on invalid config.
absl::StatusOr<std::vector<float>> ExtractNemoLogMelSpectrogram(
    absl::Span<const float> pcm_16k, const Lfm2AudioModelConfig& config,
    int* absl_nonnull out_valid_frames);

// Tokenizes `text` while preserving LFM2.5-Audio special tokens (`<|pad|>`,
// `<|startoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|audio_start|>`,
// `<|text_end|>`).
//
// Args:
//   tokenizer: Tokenizer instance.
//   text: Input string possibly containing special token literals.
//   config: LFM2.5-Audio model configuration with special token IDs.
//
// Returns:
//   Token ID sequence for `text`.
absl::StatusOr<std::vector<int>> TokenizeLfm2AudioTextWithSpecialTokens(
    support::Tokenizer& tokenizer, absl::string_view text,
    const Lfm2AudioModelConfig& config);

// Fan-out stage for `LiquidAI/LFM2.5-Audio-1.5B` that routes `TextInput`,
// `AudioInput`, `AudioInputMetadata`, and `EndOfInputTurn` into modality
// queues and rejects unsupported visual inputs (`ImageInput`, `VideoInput`).
class Lfm2AudioInputFanOutStage : public FanOutStage<OmniSession::Input> {
 public:
  // Creates an `Lfm2AudioInputFanOutStage` connected to `upstream`.
  //
  // Args:
  //   upstream: Non-null upstream `Stage<OmniSession::Input>`.
  //   has_audio_encoder: Whether the model bundle includes `audio_encoder`.
  //
  // Returns:
  //   Initialized `Lfm2AudioInputFanOutStage`.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioInputFanOutStage>> Create(
      Stage<OmniSession::Input>* absl_nonnull upstream,
      bool has_audio_encoder = true);

  ~Lfm2AudioInputFanOutStage() override;

  FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue() {
    return text_queue_;
  }
  FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
  audio_queue() {
    return audio_queue_;
  }
  FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
  audio_input_metadata_queue() {
    return audio_input_metadata_queue_;
  }
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
  end_of_turn_queue() {
    return end_of_turn_queue_;
  }

 protected:
  absl::Status RouteInput(OmniSession::Input input) override;
  void ResetInternal() override;

 private:
  Lfm2AudioInputFanOutStage(Stage<OmniSession::Input>* absl_nonnull upstream,
                            bool has_audio_encoder);

  FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue_;
  FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue_;
  FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
      audio_input_metadata_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
      end_of_turn_queue_;
  bool has_audio_encoder_ = true;
  bool has_open_turn_ = false;
};

// FastConformer audio input processing stage for `LiquidAI/LFM2.5-Audio-1.5B`:
// - Consumes `Sequenced<AudioInput>` and `Sequenced<AudioInputMetadata>` from
//   `Lfm2AudioInputFanOutStage` in ascending `seqnum` order.
// - Resamples mono PCM audio to 16 kHz via `AudioResampler`.
// - Computes 128-bin NeMo `FilterbankFeatures` log-mel spectrogram features
//   (`preemph=0.97`, symmetric Hann `win_length=400`, `hop_length=160`,
//   `n_fft=512`, Slaney mel filterbank, `log(mel + 2^-24)`, per-feature
//   normalization).
// - Runs the FastConformer encoder + 2-layer `audio_adapter` MLP per chunk and
//   keeps `mel2emb_len(T) = (T + subsampling_factor - 1) / subsampling_factor`
//   continuous embeddings.
// - Outputs `Sequenced<Lfm2AudioEmbeddingSegment>` with
//   `config.audio_embed_placeholder_token_id` (`-100`) per frame (no
//   `<|audio_start|>` / `<|audio_end|>` wrapping around input audio).
class Lfm2AudioStage : public SingleThreadedStageWithDeque<
                           Sequenced<Lfm2AudioEmbeddingSegment>> {
 public:
  // Creates an `Lfm2AudioStage`.
  //
  // Args:
  //   audio_queue: Non-null queue providing `Sequenced<AudioInput>` items.
  //   audio_input_metadata_queue: Non-null queue providing
  //     `Sequenced<AudioInputMetadata>` items.
  //   audio_encoder_runner: Non-null runner executing the FastConformer audio
  //     encoder and `audio_adapter` model.
  //   config: LFM2.5-Audio model configuration.
  //
  // Returns:
  //   A newly initialized `Lfm2AudioStage` on success, or an error status if
  //   configuration parameters or runner buffer dimensions are invalid.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioStage>> Create(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      const Lfm2AudioModelConfig& config = {});

  ~Lfm2AudioStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  Lfm2AudioStage(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      int chunk_frames, bool feature_major_input,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      std::vector<TensorBuffer> audio_enc_input_buffers,
      std::vector<TensorBuffer> audio_enc_output_buffers,
      const Lfm2AudioModelConfig& config);

  absl::StatusOr<Lfm2AudioEmbeddingSegment> ProcessAudio(
      const AudioInput& audio);

  FanOutOutputQueue<OmniSession::Input, AudioInput>& audio_queue_;
  FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>&
      audio_input_metadata_queue_;
  Lfm2AudioModelConfig config_;
  int hidden_dim_;
  int mel_bins_;
  int chunk_frames_;
  bool feature_major_input_;
  int input_sample_rate_hz_;

  AudioResampler resampler_;
  std::unique_ptr<LiteRtRunner> audio_encoder_runner_;
  std::vector<TensorBuffer> audio_enc_input_buffers_;
  std::vector<TensorBuffer> audio_enc_output_buffers_;
};

// Text tokenization and embedding stage for `LiquidAI/LFM2.5-Audio-1.5B`.
// Rejects any out-of-range token ID (`< 0` or `>= config.llm_vocab_size`) with
// `absl::InvalidArgumentError` without clamping.
class Lfm2AudioTextStage : public SingleThreadedStageWithDeque<
                               Sequenced<Lfm2AudioEmbeddingSegment>> {
 public:
  // Creates an `Lfm2AudioTextStage`.
  //
  // Args:
  //   text_queue: Non-null queue providing `Sequenced<TextInput>` items.
  //   end_of_turn_queue: Non-null queue providing
  //     `Sequenced<EndOfInputTurn>` items.
  //   tokenizer: Non-null tokenizer.
  //   text_embedder_runner: Optional runner executing `tf_lite_text_embedder`.
  //   config: LFM2.5-Audio model configuration.
  //
  // Returns:
  //   Initialized `Lfm2AudioTextStage`, or an error status on invalid config.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioTextStage>> Create(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner =
          nullptr,
      const Lfm2AudioModelConfig& config = {});

  ~Lfm2AudioTextStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  Lfm2AudioTextStage(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
      std::vector<TensorBuffer> embed_input_buffers,
      std::vector<TensorBuffer> embed_output_buffers,
      const Lfm2AudioModelConfig& config);

  absl::StatusOr<std::vector<float>> EmbedTokens(
      absl::Span<const int> token_ids);

  FanOutOutputQueue<OmniSession::Input, TextInput>& text_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>& end_of_turn_queue_;
  Lfm2AudioModelConfig config_;
  int hidden_dim_;
  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtRunner> text_embedder_runner_;
  std::vector<TensorBuffer> embed_input_buffers_;
  std::vector<TensorBuffer> embed_output_buffers_;
};

// Chat-template assembly stage for `LiquidAI/LFM2.5-Audio-1.5B` (`ChatState`
// format: `<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n` +
// `<|im_start|>user\n{interleaved text and AUDIO_IN embeddings}<|im_end|>\n` +
// `<|im_start|>assistant\n`).
class Lfm2AudioFrontendStage
    : public SingleThreadedStageWithDeque<Lfm2AudioFrontendOutput> {
 public:
  // Creates an `Lfm2AudioFrontendStage`.
  //
  // Args:
  //   text_stage: Non-null `Lfm2AudioTextStage`.
  //   audio_stage: Optional `Lfm2AudioStage`.
  //   tokenizer: Non-null tokenizer.
  //   text_embedder_runner: Optional runner executing `tf_lite_text_embedder`.
  //   config: LFM2.5-Audio model configuration.
  //
  // Returns:
  //   Initialized `Lfm2AudioFrontendStage`, or an error status on invalid
  //   config.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioFrontendStage>> Create(
      SingleThreadedStageWithDeque<
          Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nonnull text_stage,
      SingleThreadedStageWithDeque<
          Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nullable audio_stage,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner =
          nullptr,
      const Lfm2AudioModelConfig& config = {});

  ~Lfm2AudioFrontendStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  Lfm2AudioFrontendStage(
      SingleThreadedStageWithDeque<
          Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nonnull text_stage,
      SingleThreadedStageWithDeque<
          Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nullable audio_stage,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
      std::vector<TensorBuffer> embed_input_buffers,
      std::vector<TensorBuffer> embed_output_buffers,
      const Lfm2AudioModelConfig& config);

  absl::StatusOr<Lfm2AudioEmbeddingSegment> EmbedControlText(
      absl::string_view text);
  absl::Status AppendSegment(const Lfm2AudioEmbeddingSegment& seg);
  absl::Status FinalizeTurn();

  SingleThreadedStageWithDeque<Sequenced<Lfm2AudioEmbeddingSegment>>&
      text_stage_;
  SingleThreadedStageWithDeque<Sequenced<Lfm2AudioEmbeddingSegment>>*
      audio_stage_;
  Lfm2AudioModelConfig config_;
  int hidden_dim_;
  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtRunner> text_embedder_runner_;
  std::vector<TensorBuffer> embed_input_buffers_;
  std::vector<TensorBuffer> embed_output_buffers_;

  uint64_t next_expected_seqnum_ = 0;
  bool is_first_turn_ = true;
  Lfm2AudioGenerationMode resolved_mode_ =
      Lfm2AudioGenerationMode::kInterleaved;
  std::vector<Lfm2AudioEmbeddingSegment> turn_segments_;
};

// Unified interleaved/sequential generation stage for
// `LiquidAI/LFM2.5-Audio-1.5B` combining:
// 1. LFM2-1.2B hybrid short-conv + GQA LLM backbone (`LiteRtLmRunner`).
// 2. `Lfm2AudioTtsProjectorModule` (`depth_linear` `[2048 -> 8192]` and
//    `audio_embedding(codes + codebook_offsets).sum(0)` `[8 -> 2048]`).
// 3. 8-codebook RQ-Transformer `Depthformer` (`layers=6`, `dim=1024`,
//    `vocab_size=2049`, `EOAudio = vocab_size - 1`).
// 4. `LFM2AudioDetokenizer` (`[1, 8, T] -> [1, T * 1920]` 24 kHz waveform).
//
// During `AUDIO_OUT` generation (triggered when the LFM2 backbone emits
// `<|audio_start|>` = `128`), each LFM2 hidden state is projected through
// `depth_linear` and decoded into 8 codebook tokens by `Depthformer`. If
// codebook 0 emits `EOAudio` (`tts_vocab_size - 1`), all 8 codes are set to
// `EOAudio` and generation switches back to `TEXT` mode; otherwise the 8-code
// frame is appended for detokenization. In both cases, `audio_embedding(codes +
// codebook_offsets).sum(0)` is fed back into `llm_runner->Decode` as the next
// continuous step embedding!
class Lfm2AudioGenerationStage : public SingleThreadedStageWithDeque<Output> {
 public:
  // Creates an `Lfm2AudioGenerationStage`.
  //
  // Args:
  //   frontend: Non-null `Stage<Lfm2AudioFrontendOutput>`.
  //   tokenizer: Non-null tokenizer.
  //   llm_runner: Non-null LFM2 backbone `LiteRtLmRunner`.
  //   tts_projector_runner: Optional runner executing `tf_lite_tts_projector`
  //     (`depth_linear` + `audio_embedding`).
  //   tts_decoder_runner: Optional runner executing `tf_lite_tts_decoder`
  //     (8-codebook RQ-Transformer `Depthformer`).
  //   audio_vocoder_runner: Optional runner executing `tf_lite_audio_vocoder`
  //     (`LFM2AudioDetokenizer`).
  //   config: LFM2.5-Audio model configuration.
  //
  // Returns:
  //   Initialized `Lfm2AudioGenerationStage`, or an error status on invalid
  //   config or incompatible runner tensor shapes.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioGenerationStage>> Create(
      Stage<Lfm2AudioFrontendOutput>* absl_nonnull frontend,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner =
          nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner = nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable audio_vocoder_runner =
          nullptr,
      const Lfm2AudioModelConfig& config = {});

  ~Lfm2AudioGenerationStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  struct AudioFrameResult {
    std::vector<int32_t> codes;
    std::vector<float> audio_embedding;
    bool is_eo_audio = false;
  };

  Lfm2AudioGenerationStage(
      Stage<Lfm2AudioFrontendOutput>* absl_nonnull frontend,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
      std::vector<TensorBuffer> proj_input_buffers,
      std::vector<TensorBuffer> proj_output_buffers,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner,
      std::vector<TensorBuffer> dec_input_buffers,
      std::vector<TensorBuffer> dec_output_buffers,
      std::unique_ptr<LiteRtRunner> absl_nullable audio_vocoder_runner,
      std::vector<TensorBuffer> voc_input_buffers,
      std::vector<TensorBuffer> voc_output_buffers, int voc_chunk_frames,
      int voc_codebooks_dim_index, const Lfm2AudioModelConfig& config);

  absl::StatusOr<AudioFrameResult> GenerateAudioFrame(
      absl::Span<const float> llm_hidden_state, bool force_eo_audio);

  absl::StatusOr<std::vector<float>> SynthesizeWaveform(
      const std::vector<std::vector<int32_t>>& audio_frames);

  Stage<Lfm2AudioFrontendOutput>& frontend_;
  Lfm2AudioModelConfig config_;
  int hidden_dim_;
  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtLmRunner> llm_runner_;

  std::unique_ptr<LiteRtRunner> tts_projector_runner_;
  std::vector<TensorBuffer> proj_input_buffers_;
  std::vector<TensorBuffer> proj_output_buffers_;

  std::unique_ptr<LiteRtRunner> tts_decoder_runner_;
  std::vector<TensorBuffer> dec_input_buffers_;
  std::vector<TensorBuffer> dec_output_buffers_;

  std::unique_ptr<LiteRtRunner> audio_vocoder_runner_;
  std::vector<TensorBuffer> voc_input_buffers_;
  std::vector<TensorBuffer> voc_output_buffers_;
  int voc_chunk_frames_ = 0;
  int voc_codebooks_dim_index_ = 1;

  std::vector<int> turn_end_token_ids_;
  Lfm2AudioGenerationMode resolved_session_mode_ =
      Lfm2AudioGenerationMode::kAuto;
  int kv_tokens_used_ = 0;
  std::mt19937_64 rng_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_STAGE_H_
