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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_TTS_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_TTS_STAGE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/any2any/any2any_tts_decoder.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Stage 3: Thinker-to-Talker Projector & KV-Cached Talker Speech-Codec Decoder
// Stage for Qwen2.5-Omni-3B.
//
// Delegates shared projector/decoder buffer validation, GQA/4D KV-cache shape
// inference, KV-cache stepping, unique-history repetition penalty, and
// NaN/Inf-safe sampling to `Any2AnyTtsDecoder`, while implementing Qwen2.5-Omni
// Talker's prompt prefill, speaker/BOS framing, invalid codec suppression, and
// 1-token-per-step text/codec interleaved decode loop.
class QwenOmniTtsStage : public SingleThreadedStageWithDeque<Any2AnyTtsOutput> {
 public:
  // Creates a `QwenOmniTtsStage` reading Thinker outputs from `llm_stage`
  // (which must outlive the returned stage) and taking ownership of
  // `tts_projector_runner` and `tts_decoder_runner`.
  //
  // When `config.generate_audio` is `true`, both runners must be non-null;
  // otherwise returns `NotFoundError`. When `config.generate_audio` is
  // `false`, null runners are permitted and the stage passes text through
  // without generating audio codes. Returns `InvalidArgumentError` if `config`
  // dimensions or runner tensor buffers fail `Any2AnyTtsDecoder::Create`
  // validation.
  static absl::StatusOr<std::unique_ptr<QwenOmniTtsStage>> Create(
      Stage<Any2AnyLlmOutput>* absl_nonnull llm_stage,
      const QwenOmniModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner =
          nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner = nullptr);

  ~QwenOmniTtsStage() override = default;

 protected:
  bool NeedScheduleInternal() const override { return llm_stage_.HasOutput(); }
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  QwenOmniTtsStage(Stage<Any2AnyLlmOutput>* absl_nonnull llm_stage,
                   QwenOmniModelConfig config,
                   std::unique_ptr<Any2AnyTtsDecoder> absl_nonnull tts_decoder);

  // Prefills the Talker KV cache with projected Thinker prompt hidden states.
  absl::Status PrefillTalkerPrompt(const Any2AnyLlmOutput& llm_out,
                                   int32_t& step_idx,
                                   absl::Span<float> cond_emb);

  // Prefills speaker conditioning and the first assistant text token into the
  // Talker KV cache.
  absl::Status PrefillTalkerSpeakerAndFirstToken(
      const Any2AnyLlmOutput& llm_out, absl::Span<const float> zero_llm_hidden,
      int32_t& step_idx, absl::Span<float> cond_emb, absl::Span<float> logits);

  // Computes the conditioning embedding for `next_text_idx`, falling back to
  // cached text-pad conditioning once assistant text tokens are exhausted.
  absl::Status ComputeNextTextConditioning(
      const Any2AnyLlmOutput& llm_out, int next_text_idx,
      absl::Span<const float> zero_llm_hidden,
      std::vector<float>& text_pad_cond, absl::Span<float> cond_emb);

  // Autoregressively decodes 50 Hz Talker speech codec tokens until EOS or
  // `max_total_codes` / KV-cache capacity is reached.
  absl::Status DecodeTalkerSpeechCodes(const Any2AnyLlmOutput& llm_out,
                                       int max_total_codes,
                                       absl::Span<const float> zero_llm_hidden,
                                       int32_t& step_idx,
                                       absl::Span<float> cond_emb,
                                       absl::Span<float> logits,
                                       std::vector<int>& audio_codes);

  // Orchestrates prompt prefill, speaker/BOS framing, and codec decoding for a
  // single Thinker turn output.
  absl::StatusOr<Any2AnyTtsOutput> GenerateSpeechCodes(
      const Any2AnyLlmOutput& llm_out);

  Stage<Any2AnyLlmOutput>& llm_stage_;
  QwenOmniModelConfig config_;
  std::unique_ptr<Any2AnyTtsDecoder> tts_decoder_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_TTS_STAGE_H_
