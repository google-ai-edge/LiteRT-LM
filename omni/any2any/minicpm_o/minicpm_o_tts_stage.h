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
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/any2any/any2any_tts_decoder.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Stage 3: Conditional TTS Projector & KV-Cached Autoregressive Speech-Token
// Decoder Stage for MiniCPM-o 4.5.
//
// Delegates shared projector/decoder buffer validation, KV-cache stepping,
// frequency-window repetition penalty, and audio code sampling to
// `Any2AnyTtsDecoder`, while implementing MiniCPM-o's dynamic text chunking and
// 2-chunk `sliding_recompute` CosyVoice2 conditioning protocol.
class MiniCpmOTtsStage : public SingleThreadedStageWithDeque<Any2AnyTtsOutput> {
 public:
  // Creates a `MiniCpmOTtsStage` reading LLM outputs from `llm_stage` (which
  // must outlive the returned stage) and taking ownership of
  // `tts_projector_runner` and `tts_decoder_runner`.
  //
  // When `config.generate_audio` is `true`, both runners must be non-null;
  // otherwise returns `NotFoundError`. When `config.generate_audio` is
  // `false`, null runners are permitted and the stage passes text through
  // without generating audio codes. Returns `InvalidArgumentError` if `config`
  // dimensions or runner tensor buffers fail `Any2AnyTtsDecoder::Create`
  // validation.
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
  MiniCpmOTtsStage(Stage<MiniCpmOLlmOutput>* absl_nonnull llm_stage,
                   MiniCpmOModelConfig config,
                   std::unique_ptr<Any2AnyTtsDecoder> absl_nonnull tts_decoder);

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
  absl::StatusOr<Any2AnyTtsOutput> GenerateSpeechCodes(
      const MiniCpmOLlmOutput& llm_out);

  Stage<MiniCpmOLlmOutput>& llm_stage_;
  MiniCpmOModelConfig config_;
  std::unique_ptr<Any2AnyTtsDecoder> tts_decoder_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_TTS_STAGE_H_
