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

#include <cstdint>
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
// discrete 25 Hz CosyVoice2 speech token IDs (`[0, kTtsVocabSize - 1)`) for the
// downstream 24 kHz Token2Wav neural vocoder stage.
struct MiniCpmOTtsOutput {
  // Generated assistant text forwarded from `MiniCpmOLlmOutput`.
  std::string text;
  // True when `text` contains non-empty assistant text to emit.
  bool has_text = false;
  // True when this output marks the end of the current assistant turn.
  bool is_text_final = true;
  // Discrete CosyVoice2 25 Hz speech token IDs (`[0, kTtsVocabSize - 1)`)
  // sampled from `tts_decoder.tflite`'s `code_logits` head across streaming
  // text chunks until `kTtsEosTokenId` is reached.
  std::vector<int> audio_codes;
};

// Stage 3: Conditional TTS Projector & KV-Cached Autoregressive Speech-Token
// Decoder Stage for MiniCPM-o 4.5.
//
// Exported TFLite sub-model contracts:
// - `tts_projector.tflite`:
//   - Input 0 (`llm_hidden`): `float[1, 1, llm_hidden_size]` (or all-zeros for
//     control tokens `kTtsTextEosTokenId` and `kTtsAudioBosTokenId`)
//   - Input 1 (`token_id`): `int32[1, 1]`
//   - Output 0 (`cond_emb`): `float[1, 1, tts_hidden_size]`
// - `tts_decoder.tflite` (20-layer CosyVoice2 LLaMA causal decoder with
//   explicit KV cache):
//   - Input 0 (`cond_emb`): `float[1, 1, tts_hidden_size]`
//   - Input 1 (`prev_audio_code`): `int32[1, 1]` (`-1` during conditioning
//     prefill steps; `0..tts_vocab_size - 1` during autoregressive audio code
//     generation steps)
//   - Input 2 (`step_idx`): `int32[1]` (or `int32[1, 1]`, 0-based KV-cache
//     position index)
//   - Input 3 (`kv_cache`): `float[1, 2 * tts_num_layers, tts_max_cache_len,
//     tts_hidden_size]`
//   - Output 0 (`code_logits`): `float[1, 1, tts_vocab_size]` (CosyVoice2
//     speech token logits; `kTtsEosTokenId` terminates chunk generation)
//   - Output 1 (`new_kv_cache`): `float[1, 2 * tts_num_layers,
//     tts_max_cache_len, tts_hidden_size]`
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

  // Projects a single `(llm_hidden, token_id)` pair to `[tts_hidden_size]`
  // using `tts_projector_runner_`.
  absl::Status ProjectTokenConditioning(absl::Span<const float> token_hidden,
                                        int32_t token_id,
                                        absl::Span<float> out_cond);

  // Runs a single step of `tts_decoder_runner_` at `step_idx`, swapping the
  // output `new_kv_cache` buffer into `decoder_input_buffers_[3]` in O(1) and
  // reading `out_logits`.
  absl::Status RunDecoderStep(absl::Span<const float> cond_emb,
                              int32_t prev_audio_code, int32_t step_idx,
                              absl::Span<float> out_logits);

  // Projects the spoken text tokens in `[chunk_start, chunk_start +
  // chunk_tokens)`, optional `<|text_eos|>` (when `is_last_text_chunk` is
  // true), and `<|audio_bos|>` into a contiguous `[cur_cond_len,
  // tts_hidden_size]` conditioning buffer.
  absl::StatusOr<std::vector<float>> BuildChunkConditioning(
      const MiniCpmOLlmOutput& llm_out, int token_offset, int chunk_start,
      int chunk_tokens, bool is_last_text_chunk,
      bool has_leading_post_think_newline);

  // Resets the decoder KV cache, replays `[prev_chunk_cond, prev_chunk_codes]`
  // and `cur_chunk_cond` (`sliding_recompute`), and autoregressively decodes
  // the current chunk's speech tokens (appending them to `all_audio_codes`).
  absl::StatusOr<std::vector<int>> DecodeChunkSpeechCodes(
      absl::Span<const float> prev_chunk_cond,
      absl::Span<const int> prev_chunk_codes,
      absl::Span<const float> cur_chunk_cond, bool is_last_text_chunk,
      int max_total_codes, std::vector<int>& all_audio_codes);

  // Predicts discrete 25 Hz speech codes from the LLM output tokens and hidden
  // states using 10-token text chunking and 2-chunk sliding KV recomputation.
  absl::StatusOr<MiniCpmOTtsOutput> GenerateSpeechCodes(
      const MiniCpmOLlmOutput& llm_out);

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
