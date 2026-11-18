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

#include "omni/any2any/minicpm_o/minicpm_o_tts_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/any2any/any2any_tts_decoder.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {
namespace {

// Minimum `tts_max_cache_len` at which dynamic KV-headroom text chunk sizing is
// enabled (smaller cache lengths are used in unit tests with explicit chunk
// sizes).
constexpr int kMinDynamicChunkingCacheLen = 64;
// Average KV-cache steps consumed per text token (1 conditioning step + ~13
// CosyVoice2 25 Hz audio code steps).
constexpr int kEstimatedKvStepsPerTextToken = 14;
// Minimum text chunk size (in tokens) when dynamically clamping chunk size to
// remaining KV-cache headroom.
constexpr int kMinDynamicChunkTokens = 4;
// Remainder window (in tokens) beyond `max_safe_tokens` within which the
// remaining text is split into two balanced chunks to avoid a tiny final chunk.
constexpr int kRemainderSplitMarginTokens = 6;
// KV-cache steps reserved on non-final chunks so the next chunk has headroom to
// replay the current chunk, feed at least `kMinDynamicChunkTokens` conditioning
// tokens (`4 + 2 = 6` steps), and decode speech tokens (`~58` steps).
constexpr int kMinNextChunkReservedSteps = 64;

// Computes the number of spoken LLM text tokens to group into the current TTS
// chunk based on remaining KV-cache headroom after `prev_chunk_steps` and
// near-end remainder balancing.
int ComputeTextChunkTokens(int remaining_tokens, int chunk_size,
                           int max_cache_len, bool has_prev_chunk,
                           int prev_chunk_steps) {
  const int min_dynamic_chunk_tokens =
      std::min(kMinDynamicChunkTokens, chunk_size);
  const int max_safe_tokens =
      (max_cache_len >= kMinDynamicChunkingCacheLen && has_prev_chunk)
          ? std::clamp((max_cache_len - prev_chunk_steps) /
                           kEstimatedKvStepsPerTextToken,
                       min_dynamic_chunk_tokens, chunk_size)
          : chunk_size;
  int chunk_tokens = std::min(max_safe_tokens, remaining_tokens);
  if (max_cache_len >= kMinDynamicChunkingCacheLen &&
      remaining_tokens > max_safe_tokens &&
      remaining_tokens < max_safe_tokens + kRemainderSplitMarginTokens) {
    chunk_tokens = (remaining_tokens + 1) / 2;
  }
  return chunk_tokens;
}

}  // namespace

MiniCpmOTtsStage::MiniCpmOTtsStage(
    Stage<MiniCpmOLlmOutput>* absl_nonnull llm_stage,
    MiniCpmOModelConfig config,
    std::unique_ptr<Any2AnyTtsDecoder> absl_nonnull tts_decoder)
    : llm_stage_(*llm_stage),
      config_(std::move(config)),
      tts_decoder_(std::move(tts_decoder)) {}

absl::StatusOr<std::unique_ptr<MiniCpmOTtsStage>> MiniCpmOTtsStage::Create(
    Stage<MiniCpmOLlmOutput>* absl_nonnull llm_stage,
    const MiniCpmOModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner) {
  Any2AnyTtsDecoder::Options decoder_options;
  decoder_options.llm_hidden_size = config.llm_hidden_size;
  // Intentionally 0 (unbounded) to preserve parity with MiniCPM-o's original
  // projector behavior, which did not bounds-check text token IDs.
  decoder_options.llm_vocab_size = 0;
  decoder_options.tts_hidden_size = config.tts_hidden_size;
  decoder_options.tts_vocab_size = config.tts_vocab_size;
  decoder_options.tts_num_layers = config.tts_num_layers;
  decoder_options.tts_max_cache_len = config.tts_max_cache_len;
  decoder_options.generate_audio = config.generate_audio;
  decoder_options.allow_inferred_kv_shape = false;
  decoder_options.do_sample = config.do_sample;
  decoder_options.temperature = config.tts_temperature;
  decoder_options.top_p = config.tts_top_p;
  decoder_options.top_k = config.tts_top_k;
  decoder_options.repetition_penalty = config.tts_repetition_penalty;
  decoder_options.repetition_window = config.tts_repetition_window;
  decoder_options.repetition_penalty_mode =
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kFrequencyWindow;
  decoder_options.seed = config.seed;

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Any2AnyTtsDecoder> tts_decoder,
                        Any2AnyTtsDecoder::Create(
                            decoder_options, std::move(tts_projector_runner),
                            std::move(tts_decoder_runner)));
  return absl::WrapUnique(
      new MiniCpmOTtsStage(llm_stage, config, std::move(tts_decoder)));
}

absl::StatusOr<std::vector<float>> MiniCpmOTtsStage::BuildChunkConditioning(
    const MiniCpmOLlmOutput& llm_out, int token_offset, int chunk_start,
    int chunk_tokens, bool is_last_text_chunk,
    bool has_leading_post_think_newline) {
  const int raw_num_tokens = llm_out.num_tokens;
  const size_t llm_dim = static_cast<size_t>(config_.llm_hidden_size);
  const size_t tts_dim = static_cast<size_t>(config_.tts_hidden_size);
  // Append `<|text_eos|>` on the final text chunk and `<|audio_bos|>` on every
  // text chunk.
  const int cur_cond_len = chunk_tokens + (is_last_text_chunk ? 2 : 1);
  std::vector<float> cur_chunk_cond(static_cast<size_t>(cur_cond_len) * tts_dim,
                                    0.0f);
  absl::Span<const float> all_llm_hidden(llm_out.llm_hidden_states);

  for (int i = 0; i < chunk_tokens; ++i) {
    const int raw_t = token_offset + chunk_start + i;
    const int32_t token_id =
        llm_out.generated_token_ids.empty()
            ? 0
            : static_cast<int32_t>(llm_out.generated_token_ids[raw_t]);
    const int hidden_t = has_leading_post_think_newline
                             ? std::min(raw_t + 1, raw_num_tokens - 1)
                             : raw_t;
    absl::Span<const float> token_hidden = all_llm_hidden.subspan(
        static_cast<size_t>(hidden_t) * llm_dim, llm_dim);
    absl::Span<float> dst =
        absl::MakeSpan(cur_chunk_cond)
            .subspan(static_cast<size_t>(i) * tts_dim, tts_dim);
    ABSL_RETURN_IF_ERROR(
        tts_decoder_->ProjectTokenConditioning(token_hidden, token_id, dst));
  }

  const std::vector<float> zero_llm_hidden(llm_dim, 0.0f);
  int cond_offset = chunk_tokens;
  if (is_last_text_chunk) {
    absl::Span<float> text_eos_dst =
        absl::MakeSpan(cur_chunk_cond)
            .subspan(static_cast<size_t>(cond_offset) * tts_dim, tts_dim);
    ABSL_RETURN_IF_ERROR(tts_decoder_->ProjectTokenConditioning(
        zero_llm_hidden, minicpm_o::kTtsTextEosTokenId, text_eos_dst));
    ++cond_offset;
  }
  absl::Span<float> audio_bos_dst =
      absl::MakeSpan(cur_chunk_cond)
          .subspan(static_cast<size_t>(cond_offset) * tts_dim, tts_dim);
  ABSL_RETURN_IF_ERROR(tts_decoder_->ProjectTokenConditioning(
      zero_llm_hidden, minicpm_o::kTtsAudioBosTokenId, audio_bos_dst));
  return cur_chunk_cond;
}

absl::StatusOr<std::vector<int>> MiniCpmOTtsStage::DecodeChunkSpeechCodes(
    absl::Span<const float> prev_chunk_cond,
    absl::Span<const int> prev_chunk_codes,
    absl::Span<const float> cur_chunk_cond, bool is_last_text_chunk,
    int max_total_codes, std::vector<int>& all_audio_codes) {
  const size_t tts_dim = static_cast<size_t>(config_.tts_hidden_size);
  const int max_cache_len = std::max(1, tts_decoder_->tts_max_cache_len());
  const int prev_cond_len = static_cast<int>(prev_chunk_cond.size() / tts_dim);
  const int cur_cond_len = static_cast<int>(cur_chunk_cond.size() / tts_dim);
  const int required_steps =
      prev_cond_len + static_cast<int>(prev_chunk_codes.size()) + cur_cond_len;
  if (required_steps > max_cache_len) {
    return absl::FailedPreconditionError(absl::StrCat(
        "MiniCpmOTtsStage chunk conditioning exceeds tts_max_cache_len (",
        prev_cond_len, " + ", prev_chunk_codes.size(), " + ", cur_cond_len,
        " = ", required_steps, " > ", max_cache_len, ")."));
  }
  const std::vector<float> zero_tts_hidden(tts_dim, 0.0f);
  std::vector<float> logits(config_.tts_vocab_size, 0.0f);

  // `sliding_recompute` (`window_size = 2`): reset the KV cache and replay
  // `[prev_chunk, cur_chunk]` so the 20-layer CosyVoice2 causal decoder has
  // contiguous rotary position indices without cache overflow.
  ABSL_RETURN_IF_ERROR(tts_decoder_->ClearKvCache());
  int32_t step_idx = 0;

  for (int i = 0; i < prev_cond_len; ++i) {
    absl::Span<const float> cond_vec =
        prev_chunk_cond.subspan(static_cast<size_t>(i) * tts_dim, tts_dim);
    ABSL_RETURN_IF_ERROR(tts_decoder_->RunDecoderStep(
        cond_vec, /*prev_audio_code=*/-1, step_idx++,
        /*out_logits=*/std::nullopt));
  }
  for (int code : prev_chunk_codes) {
    ABSL_RETURN_IF_ERROR(tts_decoder_->RunDecoderStep(
        zero_tts_hidden, static_cast<int32_t>(code), step_idx++,
        /*out_logits=*/std::nullopt));
  }
  for (int i = 0; i < cur_cond_len; ++i) {
    absl::Span<const float> cond_vec =
        cur_chunk_cond.subspan(static_cast<size_t>(i) * tts_dim, tts_dim);
    const std::optional<absl::Span<float>> out_logits =
        (i + 1 == cur_cond_len) ? std::make_optional(absl::MakeSpan(logits))
                                : std::nullopt;
    ABSL_RETURN_IF_ERROR(tts_decoder_->RunDecoderStep(
        cond_vec, /*prev_audio_code=*/-1, step_idx++, out_logits));
  }

  const int remaining_turn_codes =
      max_total_codes - static_cast<int>(all_audio_codes.size());
  const int max_cache_codes = max_cache_len - step_idx + 1;
  const int reserved_next_chunk_steps =
      std::min(kMinNextChunkReservedSteps, max_cache_len / 2);
  const int max_non_final_codes =
      std::max(1, max_cache_len - reserved_next_chunk_steps - cur_cond_len);
  const int max_cur_chunk_codes =
      is_last_text_chunk ? std::min(remaining_turn_codes, max_cache_codes)
                         : std::min({remaining_turn_codes, max_cache_codes,
                                     max_non_final_codes});

  std::vector<int> cur_chunk_codes;
  for (int code_step = 0; code_step < max_cur_chunk_codes; ++code_step) {
    if (code_step > 0) {
      ABSL_RETURN_IF_ERROR(tts_decoder_->RunDecoderStep(
          zero_tts_hidden, static_cast<int32_t>(cur_chunk_codes.back()),
          step_idx++, absl::MakeSpan(logits)));
    }
    tts_decoder_->ApplyRepetitionPenalty(absl::MakeSpan(logits),
                                         all_audio_codes);
    const int code_id = tts_decoder_->SampleAudioCode(logits);
    if (code_id == minicpm_o::kTtsEosTokenId) {
      break;
    }
    all_audio_codes.push_back(code_id);
    cur_chunk_codes.push_back(code_id);
  }

  return cur_chunk_codes;
}

absl::StatusOr<Any2AnyTtsOutput> MiniCpmOTtsStage::GenerateSpeechCodes(
    const MiniCpmOLlmOutput& llm_out) {
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
      llm_out.llm_hidden_states));

  // In live `MiniCpmOLlmStage` execution, Qwen3 emits `<think>\n\n</think>\n\n`
  // before assistant text; after `<think>...</think>` is stripped, token 0 is
  // `\n\n` (`kQwenDoubleNewlineTokenId = 271`) and `llm_hidden_states[t]`
  // holds the pre-sampling hidden state that produced `generated_token_ids[t]`,
  // so each spoken token `t`'s post-token hidden state was captured on decode
  // step `t + 1` (falling back to `raw_num_tokens - 1` for the final token
  // since the EOS decode step is not appended to `MiniCpmOLlmOutput`). When
  // `generated_token_ids` does not start with `kQwenDoubleNewlineTokenId`
  // (e.g. unit tests supplying pre-aligned `(token_id, hidden_state)` pairs),
  // index `t` is used directly without shifting.
  const bool has_leading_post_think_newline =
      llm_out.generated_token_ids.size() > 1 &&
      llm_out.generated_token_ids[0] == minicpm_o::kQwenDoubleNewlineTokenId;
  const int token_offset = has_leading_post_think_newline ? 1 : 0;
  const int num_tokens = llm_out.num_tokens - token_offset;

  const int max_total_codes = Any2AnyTtsDecoder::ComputeMaxTotalCodes(
      num_tokens, config_.max_tts_tokens);
  const int max_cache_len = std::max(1, tts_decoder_->tts_max_cache_len());
  const int chunk_size = std::max(1, config_.tts_text_chunk_size);
  const size_t tts_dim = static_cast<size_t>(config_.tts_hidden_size);

  bool has_prev_chunk = false;
  std::vector<float> prev_chunk_cond;
  std::vector<int> prev_chunk_codes;

  for (int chunk_start = 0, chunk_tokens = 0;
       chunk_start < num_tokens &&
       static_cast<int>(out.audio_codes.size()) < max_total_codes;
       chunk_start += chunk_tokens) {
    const int prev_chunk_steps =
        has_prev_chunk ? static_cast<int>(prev_chunk_cond.size() / tts_dim +
                                          prev_chunk_codes.size())
                       : 0;
    chunk_tokens =
        ComputeTextChunkTokens(num_tokens - chunk_start, chunk_size,
                               max_cache_len, has_prev_chunk, prev_chunk_steps);
    const bool is_last_text_chunk = (chunk_start + chunk_tokens == num_tokens);

    ABSL_ASSIGN_OR_RETURN(
        std::vector<float> cur_chunk_cond,
        BuildChunkConditioning(llm_out, token_offset, chunk_start, chunk_tokens,
                               is_last_text_chunk,
                               has_leading_post_think_newline));
    ABSL_ASSIGN_OR_RETURN(
        std::vector<int> cur_chunk_codes,
        DecodeChunkSpeechCodes(prev_chunk_cond, prev_chunk_codes,
                               cur_chunk_cond, is_last_text_chunk,
                               max_total_codes, out.audio_codes));

    prev_chunk_cond = std::move(cur_chunk_cond);
    prev_chunk_codes = std::move(cur_chunk_codes);
    has_prev_chunk = true;
  }

  return out;
}

absl::Status MiniCpmOTtsStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  absl::StatusOr<MiniCpmOLlmOutput> llm_out = llm_stage_.GetOutput();
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

void MiniCpmOTtsStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  tts_decoder_->Reset();
}

}  // namespace litert::omni::any2any
