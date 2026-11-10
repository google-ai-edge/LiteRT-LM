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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {
namespace {

// Threshold at or above which `config.max_tts_tokens` is treated as the default
// turn cap and scaled up for long text responses.
constexpr int kDefaultMaxTtsTokensThreshold = 1024;
// Upper-bound acoustic token budget per spoken LLM text token (~0.8s of 25 Hz
// speech per text token) when scaling `max_tts_tokens` for long responses.
constexpr int kMaxAudioCodesPerTextToken = 20;
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

// Applies CosyVoice2 / MiniCPM-o frequency-based repetition penalty
// (`CustomRepetitionPenaltyLogitsProcessorRepeat`, `penalty > 1.0f`) over all
// speech token occurrences in the last `window_size` elements of `history`,
// scaling repeated token logits by `pow(penalty, freq)`.
void ApplyRepetitionPenalty(absl::Span<float> logits,
                            absl::Span<const int> history, float penalty,
                            int window_size) {
  if (penalty <= 1.0f || window_size <= 0) {
    return;
  }
  const size_t start = (history.size() > static_cast<size_t>(window_size))
                           ? (history.size() - static_cast<size_t>(window_size))
                           : 0;
  for (size_t i = start; i < history.size(); ++i) {
    const int code = history[i];
    if (code < 0 || static_cast<size_t>(code) >= logits.size()) {
      continue;
    }
    if (logits[code] < 0.0f) {
      logits[code] *= penalty;
    } else {
      logits[code] /= penalty;
    }
  }
}

// Selects the next speech token ID from `logits` using greedy argmax when
// `!config.do_sample || config.tts_temperature <= 0.0f`, or temperature-scaled
// top-k (`config.tts_top_k`) and nucleus (`config.tts_top_p`) sampling
// otherwise.
int SampleAudioCode(absl::Span<const float> logits,
                    const MiniCpmOModelConfig& config, std::mt19937_64& rng) {
  if (logits.empty()) {
    return 0;
  }
  if (!config.do_sample || config.tts_temperature <= 0.0f) {
    return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  const double temp = std::max<double>(config.tts_temperature, 1e-6);
  const float max_logit = *std::max_element(logits.begin(), logits.end());

  std::vector<std::pair<double, int>> candidates;
  candidates.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    const double p = std::exp((logits[i] - max_logit) / temp);
    candidates.push_back({p, static_cast<int>(i)});
  }

  const int k =
      (config.tts_top_k > 0)
          ? std::min(config.tts_top_k, static_cast<int>(candidates.size()))
          : static_cast<int>(candidates.size());
  absl::c_partial_sort(
      candidates, candidates.begin() + k,
      [](const auto& a, const auto& b) { return a.first > b.first; });
  candidates.resize(k);

  double sum = 0.0;
  for (const auto& c : candidates) {
    sum += c.first;
  }
  if (sum <= 0.0) {
    return candidates.front().second;
  }

  if (config.tts_top_p > 0.0f && config.tts_top_p < 1.0f &&
      candidates.size() > 1) {
    double cumulative = 0.0;
    size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
      cumulative += candidates[keep].first / sum;
      if (cumulative >= config.tts_top_p) {
        ++keep;
        break;
      }
    }
    candidates.resize(std::max<size_t>(1, keep));
  }

  std::vector<double> weights;
  weights.reserve(candidates.size());
  for (const auto& c : candidates) {
    weights.push_back(c.first);
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return candidates[dist(rng)].second;
}

// Validates that `llm_out.llm_hidden_states` and `llm_out.generated_token_ids`
// match `llm_out.num_tokens` and `llm_hidden_size`.
absl::Status ValidateLlmAudioInputs(const MiniCpmOLlmOutput& llm_out,
                                    int llm_hidden_size) {
  const int raw_num_tokens = llm_out.num_tokens;
  const size_t expected_hidden_size =
      static_cast<size_t>(raw_num_tokens) *
      static_cast<size_t>(llm_hidden_size);
  if (llm_out.llm_hidden_states.size() != expected_hidden_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("MiniCpmOLlmOutput llm_hidden_states size (",
                     llm_out.llm_hidden_states.size(),
                     ") does not match num_tokens * llm_hidden_size (",
                     expected_hidden_size, ")."));
  }
  if (!llm_out.generated_token_ids.empty() &&
      static_cast<int>(llm_out.generated_token_ids.size()) != raw_num_tokens) {
    return absl::InvalidArgumentError(
        absl::StrCat("MiniCpmOLlmOutput generated_token_ids size (",
                     llm_out.generated_token_ids.size(),
                     ") does not match num_tokens (", raw_num_tokens, ")."));
  }
  return absl::OkStatus();
}

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
          ? std::clamp(
                (max_cache_len - prev_chunk_steps) /
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
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner)
    : llm_stage_(*llm_stage),
      config_(std::move(config)),
      tts_projector_runner_(std::move(tts_projector_runner)),
      tts_decoder_runner_(std::move(tts_decoder_runner)) {
  rng_.seed(config_.seed.value_or(42));
}

absl::StatusOr<std::unique_ptr<MiniCpmOTtsStage>> MiniCpmOTtsStage::Create(
    Stage<MiniCpmOLlmOutput>* absl_nonnull llm_stage,
    const MiniCpmOModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner) {
  if (config.llm_hidden_size <= 0 || config.tts_hidden_size <= 0 ||
      config.tts_vocab_size <= 0 || config.tts_num_layers <= 0 ||
      config.tts_max_cache_len <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOTtsStage dimensions (llm_hidden_size, tts_hidden_size, "
        "tts_vocab_size, tts_num_layers, tts_max_cache_len) must be positive.");
  }

  auto stage = absl::WrapUnique(
      new MiniCpmOTtsStage(llm_stage, config, std::move(tts_projector_runner),
                           std::move(tts_decoder_runner)));

  if (stage->tts_projector_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        stage->proj_input_buffers_,
        stage->tts_projector_runner_->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(
        stage->proj_output_buffers_,
        stage->tts_projector_runner_->CreateOutputBuffers(""));

    if (stage->proj_input_buffers_.size() != 2 ||
        stage->proj_output_buffers_.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_projector model must have 2 inputs (llm_hidden, token_id) and 1 "
          "output buffer; got ",
          stage->proj_input_buffers_.size(), " inputs and ",
          stage->proj_output_buffers_.size(), " outputs."));
    }
    LITERT_ASSIGN_OR_RETURN(size_t hidden_in_bytes,
                            stage->proj_input_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t token_in_bytes,
                            stage->proj_input_buffers_[1].Size());
    LITERT_ASSIGN_OR_RETURN(size_t out_bytes,
                            stage->proj_output_buffers_[0].Size());
    const size_t expected_hidden_in_bytes =
        static_cast<size_t>(stage->config_.llm_hidden_size) * sizeof(float);
    const size_t expected_out_bytes =
        static_cast<size_t>(stage->config_.tts_hidden_size) * sizeof(float);
    if (hidden_in_bytes != expected_hidden_in_bytes ||
        token_in_bytes != sizeof(int32_t) || out_bytes != expected_out_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_projector buffer sizes ((", hidden_in_bytes, ", ",
          token_in_bytes, ") -> ", out_bytes,
          " bytes) do not match configured (llm_hidden_size, int32) -> "
          "tts_hidden_size ((",
          expected_hidden_in_bytes, ", ", sizeof(int32_t), ") -> ",
          expected_out_bytes, " bytes)."));
    }
  } else if (stage->config_.generate_audio) {
    return absl::NotFoundError(
        "MiniCpmOTtsStage requires tts_projector_runner when generate_audio is "
        "enabled.");
  }

  if (stage->tts_decoder_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        stage->decoder_input_buffers_,
        stage->tts_decoder_runner_->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(
        stage->decoder_output_buffers_,
        stage->tts_decoder_runner_->CreateOutputBuffers(""));
    if (stage->decoder_input_buffers_.size() != 4 ||
        stage->decoder_output_buffers_.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_decoder model must have 4 inputs (cond_emb, prev_audio_code, "
          "step_idx, kv_cache) and 2 outputs (code_logits, new_kv_cache); got ",
          stage->decoder_input_buffers_.size(), " inputs and ",
          stage->decoder_output_buffers_.size(), " outputs."));
    }

    LITERT_ASSIGN_OR_RETURN(size_t cond_bytes,
                            stage->decoder_input_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t code_bytes,
                            stage->decoder_input_buffers_[1].Size());
    LITERT_ASSIGN_OR_RETURN(size_t step_bytes,
                            stage->decoder_input_buffers_[2].Size());
    LITERT_ASSIGN_OR_RETURN(size_t kv_in_bytes,
                            stage->decoder_input_buffers_[3].Size());
    LITERT_ASSIGN_OR_RETURN(size_t logits_bytes,
                            stage->decoder_output_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t kv_out_bytes,
                            stage->decoder_output_buffers_[1].Size());

    const size_t expected_cond_bytes =
        static_cast<size_t>(stage->config_.tts_hidden_size) * sizeof(float);
    const size_t expected_int32_bytes = sizeof(int32_t);
    const size_t expected_logits_bytes =
        static_cast<size_t>(stage->config_.tts_vocab_size) * sizeof(float);
    const size_t expected_kv_floats =
        static_cast<size_t>(2) * stage->config_.tts_num_layers *
        stage->config_.tts_max_cache_len * stage->config_.tts_hidden_size;
    const size_t expected_kv_bytes = expected_kv_floats * sizeof(float);

    if (std::tie(cond_bytes, code_bytes, step_bytes, kv_in_bytes, logits_bytes,
                 kv_out_bytes) !=
        std::tie(expected_cond_bytes, expected_int32_bytes,
                 expected_int32_bytes, expected_kv_bytes, expected_logits_bytes,
                 expected_kv_bytes)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_decoder buffer sizes do not match configured TTS dimensions: "
          "inputs=(",
          cond_bytes, ", ", code_bytes, ", ", step_bytes, ", ", kv_in_bytes,
          ") bytes vs expected (", expected_cond_bytes, ", ", sizeof(int32_t),
          ", ", sizeof(int32_t), ", ", expected_kv_bytes, ") bytes; outputs=(",
          logits_bytes, ", ", kv_out_bytes, ") bytes vs expected (",
          expected_logits_bytes, ", ", expected_kv_bytes, ") bytes."));
    }
  } else if (stage->config_.generate_audio) {
    return absl::NotFoundError(
        "MiniCpmOTtsStage requires tts_decoder_runner when generate_audio is "
        "enabled.");
  }

  return stage;
}

absl::Status MiniCpmOTtsStage::ProjectTokenConditioning(
    absl::Span<const float> token_hidden, int32_t token_id,
    absl::Span<float> out_cond) {
  if (tts_projector_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "MiniCpmOTtsStage requires tts_projector_runner to project token "
        "conditioning.");
  }
  LITERT_RETURN_IF_ERROR(proj_input_buffers_[0].Write<float>(token_hidden));
  LITERT_RETURN_IF_ERROR(
      proj_input_buffers_[1].Write<int32_t>(absl::MakeConstSpan(&token_id, 1)));
  ABSL_RETURN_IF_ERROR(tts_projector_runner_->Run("", proj_input_buffers_,
                                                  proj_output_buffers_));
  LITERT_RETURN_IF_ERROR(proj_output_buffers_[0].Read<float>(out_cond));
  return absl::OkStatus();
}

absl::Status MiniCpmOTtsStage::RunDecoderStep(absl::Span<const float> cond_emb,
                                              int32_t prev_audio_code,
                                              int32_t step_idx,
                                              absl::Span<float> out_logits) {
  if (tts_decoder_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "MiniCpmOTtsStage requires tts_decoder_runner to decode speech "
        "tokens.");
  }
  LITERT_RETURN_IF_ERROR(decoder_input_buffers_[0].Write<float>(cond_emb));
  LITERT_RETURN_IF_ERROR(decoder_input_buffers_[1].Write<int32_t>(
      absl::MakeConstSpan(&prev_audio_code, 1)));
  LITERT_RETURN_IF_ERROR(decoder_input_buffers_[2].Write<int32_t>(
      absl::MakeConstSpan(&step_idx, 1)));
  ABSL_RETURN_IF_ERROR(tts_decoder_runner_->Run("", decoder_input_buffers_,
                                                decoder_output_buffers_));
  LITERT_RETURN_IF_ERROR(decoder_output_buffers_[0].Read<float>(out_logits));
  std::swap(decoder_input_buffers_[3], decoder_output_buffers_[1]);
  return absl::OkStatus();
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
    ABSL_RETURN_IF_ERROR(ProjectTokenConditioning(token_hidden, token_id, dst));
  }

  const std::vector<float> zero_llm_hidden(llm_dim, 0.0f);
  int cond_offset = chunk_tokens;
  if (is_last_text_chunk) {
    absl::Span<float> text_eos_dst =
        absl::MakeSpan(cur_chunk_cond)
            .subspan(static_cast<size_t>(cond_offset) * tts_dim, tts_dim);
    ABSL_RETURN_IF_ERROR(ProjectTokenConditioning(
        zero_llm_hidden, minicpm_o::kTtsTextEosTokenId, text_eos_dst));
    ++cond_offset;
  }
  absl::Span<float> audio_bos_dst =
      absl::MakeSpan(cur_chunk_cond)
          .subspan(static_cast<size_t>(cond_offset) * tts_dim, tts_dim);
  ABSL_RETURN_IF_ERROR(ProjectTokenConditioning(
      zero_llm_hidden, minicpm_o::kTtsAudioBosTokenId, audio_bos_dst));
  return cur_chunk_cond;
}

absl::StatusOr<std::vector<int>> MiniCpmOTtsStage::DecodeChunkSpeechCodes(
    absl::Span<const float> prev_chunk_cond,
    absl::Span<const int> prev_chunk_codes,
    absl::Span<const float> cur_chunk_cond, bool is_last_text_chunk,
    int max_total_codes, std::vector<int>& all_audio_codes) {
  const size_t tts_dim = static_cast<size_t>(config_.tts_hidden_size);
  const int max_cache_len = std::max(1, config_.tts_max_cache_len);
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
  LITERT_RETURN_IF_ERROR(decoder_input_buffers_[3].Clear());
  int32_t step_idx = 0;

  for (int i = 0; i < prev_cond_len; ++i) {
    absl::Span<const float> cond_vec =
        prev_chunk_cond.subspan(static_cast<size_t>(i) * tts_dim, tts_dim);
    ABSL_RETURN_IF_ERROR(RunDecoderStep(cond_vec, /*prev_audio_code=*/-1,
                                        step_idx++, absl::MakeSpan(logits)));
  }
  for (int code : prev_chunk_codes) {
    ABSL_RETURN_IF_ERROR(RunDecoderStep(zero_tts_hidden,
                                        static_cast<int32_t>(code), step_idx++,
                                        absl::MakeSpan(logits)));
  }
  for (int i = 0; i < cur_cond_len; ++i) {
    absl::Span<const float> cond_vec =
        cur_chunk_cond.subspan(static_cast<size_t>(i) * tts_dim, tts_dim);
    ABSL_RETURN_IF_ERROR(RunDecoderStep(cond_vec, /*prev_audio_code=*/-1,
                                        step_idx++, absl::MakeSpan(logits)));
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
      ABSL_RETURN_IF_ERROR(RunDecoderStep(
          zero_tts_hidden, static_cast<int32_t>(cur_chunk_codes.back()),
          step_idx++, absl::MakeSpan(logits)));
    }
    ApplyRepetitionPenalty(absl::MakeSpan(logits), all_audio_codes,
                           config_.tts_repetition_penalty,
                           config_.tts_repetition_window);
    const int code_id = SampleAudioCode(logits, config_, rng_);
    if (code_id == minicpm_o::kTtsEosTokenId) {
      break;
    }
    all_audio_codes.push_back(code_id);
    cur_chunk_codes.push_back(code_id);
  }

  return cur_chunk_codes;
}

absl::StatusOr<MiniCpmOTtsOutput> MiniCpmOTtsStage::GenerateSpeechCodes(
    const MiniCpmOLlmOutput& llm_out) {
  MiniCpmOTtsOutput out;
  out.text = llm_out.text;
  out.has_text = !llm_out.text.empty();
  out.is_text_final = llm_out.is_final;

  if (!llm_out.generate_audio || !config_.generate_audio ||
      llm_out.num_tokens <= 0 || llm_out.llm_hidden_states.empty()) {
    return out;
  }
  ABSL_RETURN_IF_ERROR(
      ValidateLlmAudioInputs(llm_out, config_.llm_hidden_size));

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

  const int max_total_codes =
      (config_.max_tts_tokens >= kDefaultMaxTtsTokensThreshold)
          ? std::max(config_.max_tts_tokens,
                     num_tokens * kMaxAudioCodesPerTextToken)
          : std::max(1, config_.max_tts_tokens);
  const int max_cache_len = std::max(1, config_.tts_max_cache_len);
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
        has_prev_chunk
            ? static_cast<int>(prev_chunk_cond.size() / tts_dim +
                               prev_chunk_codes.size())
            : 0;
    chunk_tokens = ComputeTextChunkTokens(num_tokens - chunk_start, chunk_size,
                                          max_cache_len, has_prev_chunk,
                                          prev_chunk_steps);
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
  auto llm_out = llm_stage_.GetOutput();
  if (absl::IsNotFound(llm_out.status())) {
    return absl::OkStatus();
  }
  if (!llm_out.ok()) {
    return llm_out.status();
  }

  ABSL_ASSIGN_OR_RETURN(MiniCpmOTtsOutput tts_out,
                        GenerateSpeechCodes(*llm_out));
  PushOutput(std::move(tts_out));
  return absl::OkStatus();
}

void MiniCpmOTtsStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  rng_.seed(config_.seed.value_or(42));
}

}  // namespace litert::omni::any2any
