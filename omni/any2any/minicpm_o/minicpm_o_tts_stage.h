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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_TTS_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_TTS_STAGE_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Output of Stage 3 (MiniCpmOTtsStage): generated text passthrough and/or
// predicted log-mel spectrogram frames (`[num_mel_bins, num_frames]`) for the
// downstream 24 kHz mel-to-waveform neural vocoder stage.
struct MiniCpmOTtsOutput {
  std::string text;
  bool has_text = false;
  bool is_text_final = true;
  // Log-mel spectrogram features in bin-major `[num_mel_bins, num_frames]`
  // layout emitted by `tts_decoder.tflite`'s fused Token2Wav flow-matching
  // mel head and consumed by `MiniCpmOVocoderStage`.
  std::vector<float> mel_features;
  int num_mel_bins = minicpm_o::kTtsNumMelBins;
  int num_frames = 0;
  // Discrete CosyVoice2 speech token IDs (`[0, kTtsVocabSize)`) sampled from
  // `tts_decoder.tflite`'s `code_logits` head at each autoregressive step and
  // fed back as `prev_audio_code` until `kTtsEosTokenId` is reached.
  std::vector<int> audio_codes;
};

// Stage 3: Conditional TTS Projector & Stateful Autoregressive Acoustic Decoder
// Stage for MiniCPM-o 4.5.
//
// Exported TFLite sub-model contracts:
// - `tts_projector.tflite`:
//   - Input 0 (`llm_hidden`): `float[1, llm_hidden_size]`
//   - Output 0 (`cond_emb`): `float[1, tts_hidden_size]`
// - `tts_decoder.tflite` (stateful per-step subgraph maintaining internal
//   KV-cache and flow-matching state across steps, reset when `step_idx == 0`):
//   - Input 0 (`cond_emb`): `float[1, tts_hidden_size]`
//   - Input 1 (`prev_audio_code`): `int32[1, 1]` (`kTtsBosTokenId` on step 0,
//     then the previously sampled speech token ID)
//   - Input 2 (`step_idx`): `int32[1, 1]` (0-based decode step index)
//   - Output 0 (`code_logits`): `float[1, tts_vocab_size]` (CosyVoice2 speech
//     token logits; `kTtsEosTokenId` terminates generation)
//   - Output 1 (`mel_frame`): `float[1, tts_num_mel_bins]` (100-bin log-mel
//     frame synthesized by the fused Token2Wav flow-matching head)
class MiniCpmOTtsStage
    : public SingleThreadedStageWithDeque<MiniCpmOTtsOutput> {
 public:
  static absl::StatusOr<std::unique_ptr<MiniCpmOTtsStage>> Create(
      Stage<MiniCpmOLlmOutput>* absl_nonnull llm_stage,
      const MiniCpmOModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner =
          nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner = nullptr);

  ~MiniCpmOTtsStage() override = default;

 protected:
  bool NeedScheduleInternal() const override { return llm_stage_.HasOutput(); }

  absl::Status ScheduleInternal() override;

  void ResetInternal() override;

 private:
  MiniCpmOTtsStage(
      Stage<MiniCpmOLlmOutput>* absl_nonnull llm_stage,
      MiniCpmOModelConfig config,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner);

  // Projects `[num_tokens, llm_hidden_size]` to `[num_tokens, tts_hidden_size]`
  // using `tts_projector_runner_`.
  absl::StatusOr<std::vector<float>> ProjectHiddenStates(
      absl::Span<const float> llm_hidden, int num_tokens);

  // Predicts log-mel spectrogram features (`[num_mel_bins, num_frames]`)
  // from the projected TTS conditioning vectors.
  absl::StatusOr<MiniCpmOTtsOutput> PredictLogMelSpectrogram(
      const MiniCpmOLlmOutput& llm_out);

  // Runs the stateful autoregressive TTS decoder step-by-step, feeding back
  // `(cond_emb, prev_audio_code, step_idx)` at each step to predict discrete
  // speech tokens (`code_logits`) and mel spectrogram frames (`mel_frame`)
  // until `kTtsEosTokenId` or `max_tts_tokens` is reached.
  absl::Status DecodeAudioCodesAndMelFramesAutoregressively(
      absl::Span<const float> cond_hidden, int num_tokens,
      MiniCpmOTtsOutput& out);

  Stage<MiniCpmOLlmOutput>& llm_stage_;
  MiniCpmOModelConfig config_;

  std::unique_ptr<LiteRtRunner> tts_projector_runner_;
  std::unique_ptr<LiteRtRunner> tts_decoder_runner_;

  std::vector<TensorBuffer> proj_input_buffers_;
  std::vector<TensorBuffer> proj_output_buffers_;
  std::vector<TensorBuffer> decoder_input_buffers_;
  std::vector<TensorBuffer> decoder_output_buffers_;

  std::mt19937_64 rng_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_TTS_STAGE_H_
