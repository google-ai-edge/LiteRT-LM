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

#include "omni/any2any/qwen_omni/qwen_omni_tts_stage.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/any2any/any2any_tts_decoder.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {
namespace {

bool IsMultimodalPlaceholderToken(int token_id,
                                  const QwenOmniModelConfig& config) {
  return token_id == config.audio_token_id ||
         token_id == config.image_token_id ||
         token_id == config.video_token_id ||
         token_id == qwen_omni::kAudioTokenId ||
         token_id == qwen_omni::kImageTokenId ||
         token_id == qwen_omni::kVideoTokenId;
}

// Suppresses `codec_bos_token` (`8293`) and non-EOS special codec tokens
// (`>= kVocoderCodebookSize`) matching `suppress_tokens=[codec_bos_token]` in
// `Qwen2_5OmniForConditionalGeneration.generate()`.
void SuppressInvalidCodecTokens(absl::Span<float> logits,
                                const QwenOmniModelConfig& config) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  if (config.tts_audio_bos_token_id >= 0 &&
      static_cast<size_t>(config.tts_audio_bos_token_id) < logits.size()) {
    logits[config.tts_audio_bos_token_id] = kNegInf;
  }
  if (logits.size() > static_cast<size_t>(qwen_omni::kVocoderCodebookSize)) {
    for (size_t i = qwen_omni::kVocoderCodebookSize; i < logits.size(); ++i) {
      const int id = static_cast<int>(i);
      if (id != config.tts_eos_token_id && id != config.tts_silence_token_id) {
        logits[i] = kNegInf;
      }
    }
  }
}

}  // namespace

QwenOmniTtsStage::QwenOmniTtsStage(
    Stage<Any2AnyLlmOutput>* absl_nonnull llm_stage, QwenOmniModelConfig config,
    std::unique_ptr<Any2AnyTtsDecoder> absl_nonnull tts_decoder)
    : llm_stage_(*llm_stage),
      config_(std::move(config)),
      tts_decoder_(std::move(tts_decoder)) {}

absl::StatusOr<std::unique_ptr<QwenOmniTtsStage>> QwenOmniTtsStage::Create(
    Stage<Any2AnyLlmOutput>* absl_nonnull llm_stage,
    const QwenOmniModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner) {
  Any2AnyTtsDecoder::Options decoder_options;
  decoder_options.llm_hidden_size = config.llm_hidden_size;
  decoder_options.llm_vocab_size = config.llm_vocab_size;
  decoder_options.tts_hidden_size = config.tts_hidden_size;
  decoder_options.tts_vocab_size = config.tts_vocab_size;
  decoder_options.tts_num_layers = config.tts_num_layers;
  decoder_options.tts_num_kv_heads = config.tts_num_kv_heads;
  decoder_options.tts_head_dim = config.tts_head_dim;
  decoder_options.tts_max_cache_len = config.tts_max_cache_len;
  decoder_options.generate_audio = config.generate_audio;
  decoder_options.allow_inferred_kv_shape = true;
  decoder_options.do_sample = config.tts_do_sample;
  decoder_options.temperature = config.tts_temperature;
  decoder_options.top_p = config.tts_top_p;
  decoder_options.top_k = config.tts_top_k;
  decoder_options.repetition_penalty = config.tts_repetition_penalty;
  decoder_options.repetition_penalty_mode =
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kUniqueHistory;
  decoder_options.seed = config.seed;

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Any2AnyTtsDecoder> tts_decoder,
                        Any2AnyTtsDecoder::Create(
                            decoder_options, std::move(tts_projector_runner),
                            std::move(tts_decoder_runner)));
  return absl::WrapUnique(
      new QwenOmniTtsStage(llm_stage, config, std::move(tts_decoder)));
}

absl::Status QwenOmniTtsStage::PrefillTalkerPrompt(
    const Any2AnyLlmOutput& llm_out, int32_t& step_idx,
    absl::Span<float> cond_emb) {
  // Feed `thinker_hidden_states[0] + embeds_to_talker` for each prompt token
  // (with `embeds_to_talker` zeroed at audio/image/video positions via
  // `b_token_id = -1`) and `prev_audio_code = -1` (no codec token embedding
  // added on prompt positions).
  const size_t llm_dim = static_cast<size_t>(config_.llm_hidden_size);
  const int max_cache_len = tts_decoder_->tts_max_cache_len();
  absl::Span<const float> all_prompt_hidden(llm_out.prompt_hidden_states);
  const size_t prompt_len = llm_out.prompt_token_ids.size();
  for (size_t i = 0; i < prompt_len; ++i) {
    if (step_idx + 2 >= max_cache_len) {
      ABSL_LOG(WARNING) << "QwenOmniTtsStage prompt length (" << prompt_len
                        << ") truncated at step " << step_idx
                        << " due to tts_max_cache_len (" << max_cache_len
                        << ").";
      break;
    }
    const int tok_i = llm_out.prompt_token_ids[i];
    const int32_t proj_tok_id = IsMultimodalPlaceholderToken(tok_i, config_)
                                    ? -1
                                    : static_cast<int32_t>(tok_i);
    ABSL_RETURN_IF_ERROR(tts_decoder_->ProjectTokenConditioning(
        all_prompt_hidden.subspan(i * llm_dim, llm_dim), proj_tok_id,
        cond_emb));
    ABSL_RETURN_IF_ERROR(tts_decoder_->RunDecoderStep(
        cond_emb, /*prev_audio_code=*/-1, step_idx++,
        /*out_logits=*/std::nullopt));
  }
  return absl::OkStatus();
}

absl::Status QwenOmniTtsStage::PrefillTalkerSpeakerAndFirstToken(
    const Any2AnyLlmOutput& llm_out, absl::Span<const float> zero_llm_hidden,
    int32_t& step_idx, absl::Span<float> cond_emb, absl::Span<float> logits) {
  const size_t llm_dim = static_cast<size_t>(config_.llm_hidden_size);

  // Speaker text-BOS prefill: `tts_text_bos_token_id` paired with
  // `tts_silence_token_id` (`tts_codec_pad_token_id` = 8292).
  ABSL_RETURN_IF_ERROR(tts_decoder_->ProjectTokenConditioning(
      zero_llm_hidden, config_.tts_text_bos_token_id, cond_emb));
  ABSL_RETURN_IF_ERROR(tts_decoder_->RunDecoderStep(
      cond_emb, static_cast<int32_t>(config_.tts_silence_token_id), step_idx++,
      /*out_logits=*/std::nullopt));

  // First reply token conditioning paired with `tts_audio_bos_token_id`
  // (`tts_codec_start_token_id` = 8293).
  const int32_t first_token_id =
      llm_out.generated_token_ids.empty()
          ? 0
          : static_cast<int32_t>(llm_out.generated_token_ids[0]);
  ABSL_RETURN_IF_ERROR(tts_decoder_->ProjectTokenConditioning(
      absl::MakeConstSpan(llm_out.llm_hidden_states).subspan(0, llm_dim),
      first_token_id, cond_emb));
  return tts_decoder_->RunDecoderStep(
      cond_emb, static_cast<int32_t>(config_.tts_audio_bos_token_id),
      step_idx++, logits);
}

absl::Status QwenOmniTtsStage::ComputeNextTextConditioning(
    const Any2AnyLlmOutput& llm_out, int next_text_idx,
    absl::Span<const float> zero_llm_hidden, std::vector<float>& text_pad_cond,
    absl::Span<float> cond_emb) {
  const size_t llm_dim = static_cast<size_t>(config_.llm_hidden_size);
  if (next_text_idx < llm_out.num_tokens) {
    const int32_t next_token_id =
        llm_out.generated_token_ids.empty()
            ? 0
            : static_cast<int32_t>(llm_out.generated_token_ids[next_text_idx]);
    absl::Span<const float> next_hidden =
        absl::MakeConstSpan(llm_out.llm_hidden_states)
            .subspan(static_cast<size_t>(next_text_idx) * llm_dim, llm_dim);
    return tts_decoder_->ProjectTokenConditioning(next_hidden, next_token_id,
                                                  cond_emb);
  }
  if (next_text_idx == llm_out.num_tokens) {
    return tts_decoder_->ProjectTokenConditioning(
        zero_llm_hidden, config_.tts_text_eos_token_id, cond_emb);
  }
  if (text_pad_cond.empty()) {
    text_pad_cond.resize(static_cast<size_t>(config_.tts_hidden_size), 0.0f);
    ABSL_RETURN_IF_ERROR(tts_decoder_->ProjectTokenConditioning(
        zero_llm_hidden, config_.tts_text_pad_token_id,
        absl::MakeSpan(text_pad_cond)));
  }
  absl::c_copy(text_pad_cond, cond_emb.begin());
  return absl::OkStatus();
}

absl::Status QwenOmniTtsStage::DecodeTalkerSpeechCodes(
    const Any2AnyLlmOutput& llm_out, int max_total_codes,
    absl::Span<const float> zero_llm_hidden, int32_t& step_idx,
    absl::Span<float> cond_emb, absl::Span<float> logits,
    std::vector<int>& audio_codes) {
  std::vector<float> text_pad_cond;
  const int max_cache_len = tts_decoder_->tts_max_cache_len();
  for (int code_step = 0; code_step < max_total_codes; ++code_step) {
    tts_decoder_->ApplyRepetitionPenalty(logits, audio_codes);
    SuppressInvalidCodecTokens(logits, config_);
    const int code_id = tts_decoder_->SampleAudioCode(logits);
    if (code_id == config_.tts_eos_token_id ||
        code_id == config_.tts_silence_token_id) {
      break;
    }
    audio_codes.push_back(code_id);

    if (static_cast<int>(audio_codes.size()) >= max_total_codes) {
      ABSL_LOG(WARNING) << "QwenOmniTtsStage reached max_tts_tokens cap ("
                        << max_total_codes << ").";
      break;
    }
    if (step_idx >= max_cache_len) {
      ABSL_LOG(WARNING) << "QwenOmniTtsStage reached tts_max_cache_len cap ("
                        << max_cache_len << ").";
      break;
    }

    ABSL_RETURN_IF_ERROR(
        ComputeNextTextConditioning(llm_out, /*next_text_idx=*/code_step + 1,
                                    zero_llm_hidden, text_pad_cond, cond_emb));
    ABSL_RETURN_IF_ERROR(tts_decoder_->RunDecoderStep(
        cond_emb, static_cast<int32_t>(code_id), step_idx++, logits));
  }
  return absl::OkStatus();
}

absl::StatusOr<Any2AnyTtsOutput> QwenOmniTtsStage::GenerateSpeechCodes(
    const Any2AnyLlmOutput& llm_out) {
  Any2AnyTtsOutput out;
  out.text = llm_out.text;
  out.has_text = !llm_out.text.empty();
  out.is_text_final = llm_out.is_final;

  if (!llm_out.generate_audio || !config_.generate_audio ||
      llm_out.num_tokens <= 0) {
    return out;
  }
  ABSL_RETURN_IF_ERROR(Any2AnyTtsDecoder::ValidateLlmAudioInputs(
      llm_out.num_tokens, config_.llm_hidden_size, llm_out.generated_token_ids,
      llm_out.llm_hidden_states, llm_out.prompt_token_ids,
      llm_out.prompt_hidden_states));

  if (tts_decoder_->tts_max_cache_len() < 2) {
    return absl::FailedPreconditionError(absl::StrCat(
        "QwenOmniTtsStage tts_max_cache_len (",
        tts_decoder_->tts_max_cache_len(),
        ") must be at least 2 for Talker BOS + first-token conditioning."));
  }

  const int max_total_codes = Any2AnyTtsDecoder::ComputeMaxTotalCodes(
      llm_out.num_tokens, config_.max_tts_tokens);
  const std::vector<float> zero_llm_hidden(
      static_cast<size_t>(config_.llm_hidden_size), 0.0f);
  std::vector<float> cond_emb(static_cast<size_t>(config_.tts_hidden_size),
                              0.0f);
  std::vector<float> logits(config_.tts_vocab_size, 0.0f);

  ABSL_RETURN_IF_ERROR(tts_decoder_->ClearKvCache());
  int32_t step_idx = 0;

  ABSL_RETURN_IF_ERROR(
      PrefillTalkerPrompt(llm_out, step_idx, absl::MakeSpan(cond_emb)));
  ABSL_RETURN_IF_ERROR(PrefillTalkerSpeakerAndFirstToken(
      llm_out, zero_llm_hidden, step_idx, absl::MakeSpan(cond_emb),
      absl::MakeSpan(logits)));
  ABSL_RETURN_IF_ERROR(DecodeTalkerSpeechCodes(
      llm_out, max_total_codes, zero_llm_hidden, step_idx,
      absl::MakeSpan(cond_emb), absl::MakeSpan(logits), out.audio_codes));

  return out;
}

absl::Status QwenOmniTtsStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  absl::StatusOr<Any2AnyLlmOutput> llm_out = llm_stage_.GetOutput();
  if (absl::IsNotFound(llm_out.status())) {
    return absl::OkStatus();
  }
  if (!llm_out.ok()) {
    return llm_out.status();
  }

  ABSL_ASSIGN_OR_RETURN(Any2AnyTtsOutput tts_out,
                        GenerateSpeechCodes(*llm_out));
  PushOutput(std::move(tts_out));
  return absl::OkStatus();
}

void QwenOmniTtsStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  tts_decoder_->Reset();
}

}  // namespace litert::omni::any2any
