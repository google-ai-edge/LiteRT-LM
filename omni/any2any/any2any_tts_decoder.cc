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

#include "omni/any2any/any2any_tts_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "omni/any2any/any2any_sampler.h"
#include "omni/base/litert_runner.h"

namespace litert::omni::any2any {
namespace {

// Threshold at or above which `max_tts_tokens` is treated as the default turn
// cap and scaled up for long text responses.
constexpr int kDefaultMaxTtsTokensThreshold = 1024;
// Upper-bound acoustic token budget per spoken LLM text token when scaling
// `max_tts_tokens` for long responses.
constexpr int kMaxAudioCodesPerTextToken = 20;

// Computes the product of `factors` in `size_t`, returning `std::nullopt` on
// overflow.
std::optional<size_t> CheckedMultiply(std::initializer_list<size_t> factors) {
  size_t result = 1;
  for (size_t factor : factors) {
    if (factor == 0) {
      return 0;
    }
    if (result > std::numeric_limits<size_t>::max() / factor) {
      return std::nullopt;
    }
    result *= factor;
  }
  return result;
}

std::string FormatCheckedSize(std::optional<size_t> value) {
  return value.has_value() ? absl::StrCat(*value) : "overflow";
}

void ApplySingleRepetitionPenalty(float& logit, float penalty) {
  if (logit < 0.0f) {
    logit *= penalty;
  } else {
    logit /= penalty;
  }
}

}  // namespace

Any2AnyTtsDecoder::Any2AnyTtsDecoder(
    Options options,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner)
    : options_(std::move(options)),
      tts_projector_runner_(std::move(tts_projector_runner)),
      tts_decoder_runner_(std::move(tts_decoder_runner)),
      sampler_({
          .do_sample = options_.do_sample,
          .temperature = options_.temperature,
          .top_k = options_.top_k,
          .top_p = options_.top_p,
          .seed = options_.seed,
      }) {}

absl::StatusOr<std::unique_ptr<Any2AnyTtsDecoder>> Any2AnyTtsDecoder::Create(
    Options options,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner) {
  if (options.llm_hidden_size <= 0 || options.tts_hidden_size <= 0 ||
      options.tts_vocab_size <= 0 || options.tts_num_layers <= 0 ||
      options.tts_max_cache_len <= 0) {
    return absl::InvalidArgumentError(
        "Any2AnyTtsDecoder dimensions (llm_hidden_size, tts_hidden_size, "
        "tts_vocab_size, tts_num_layers, tts_max_cache_len) must be positive.");
  }

  auto decoder = absl::WrapUnique(
      new Any2AnyTtsDecoder(std::move(options), std::move(tts_projector_runner),
                            std::move(tts_decoder_runner)));

  if (decoder->tts_projector_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        decoder->proj_input_buffers_,
        decoder->tts_projector_runner_->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(
        decoder->proj_output_buffers_,
        decoder->tts_projector_runner_->CreateOutputBuffers(""));

    if (decoder->proj_input_buffers_.size() != 2 ||
        decoder->proj_output_buffers_.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_projector model must have 2 inputs (llm_hidden, token_id) and 1 "
          "output buffer; got ",
          decoder->proj_input_buffers_.size(), " inputs and ",
          decoder->proj_output_buffers_.size(), " outputs."));
    }
    LITERT_ASSIGN_OR_RETURN(size_t hidden_in_bytes,
                            decoder->proj_input_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t token_in_bytes,
                            decoder->proj_input_buffers_[1].Size());
    LITERT_ASSIGN_OR_RETURN(size_t out_bytes,
                            decoder->proj_output_buffers_[0].Size());
    const std::optional<size_t> expected_hidden_in_bytes =
        CheckedMultiply({static_cast<size_t>(decoder->options_.llm_hidden_size),
                         sizeof(float)});
    const std::optional<size_t> expected_out_bytes =
        CheckedMultiply({static_cast<size_t>(decoder->options_.tts_hidden_size),
                         sizeof(float)});
    if (!expected_hidden_in_bytes.has_value() ||
        !expected_out_bytes.has_value() ||
        hidden_in_bytes != *expected_hidden_in_bytes ||
        token_in_bytes != sizeof(int32_t) || out_bytes != *expected_out_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_projector buffer sizes ((", hidden_in_bytes, ", ",
          token_in_bytes, ") -> ", out_bytes,
          " bytes) do not match configured (llm_hidden_size, int32) -> "
          "tts_hidden_size ((",
          FormatCheckedSize(expected_hidden_in_bytes), ", ", sizeof(int32_t),
          ") -> ", FormatCheckedSize(expected_out_bytes), " bytes)."));
    }
  } else if (decoder->options_.generate_audio) {
    return absl::NotFoundError(
        "Any2AnyTtsDecoder requires tts_projector_runner when generate_audio "
        "is enabled.");
  }

  if (decoder->tts_decoder_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(decoder->decoder_input_buffers_,
                          decoder->tts_decoder_runner_->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(
        decoder->decoder_output_buffers_,
        decoder->tts_decoder_runner_->CreateOutputBuffers(""));
    if (decoder->decoder_input_buffers_.size() != 4 ||
        decoder->decoder_output_buffers_.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_decoder model must have 4 inputs (cond_emb, prev_audio_code, "
          "step_idx, kv_cache) and 2 outputs (code_logits, new_kv_cache); got ",
          decoder->decoder_input_buffers_.size(), " inputs and ",
          decoder->decoder_output_buffers_.size(), " outputs."));
    }

    LITERT_ASSIGN_OR_RETURN(size_t cond_bytes,
                            decoder->decoder_input_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t code_bytes,
                            decoder->decoder_input_buffers_[1].Size());
    LITERT_ASSIGN_OR_RETURN(size_t step_bytes,
                            decoder->decoder_input_buffers_[2].Size());
    LITERT_ASSIGN_OR_RETURN(size_t kv_in_bytes,
                            decoder->decoder_input_buffers_[3].Size());
    LITERT_ASSIGN_OR_RETURN(size_t logits_bytes,
                            decoder->decoder_output_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t kv_out_bytes,
                            decoder->decoder_output_buffers_[1].Size());

    const std::optional<size_t> expected_cond_bytes =
        CheckedMultiply({static_cast<size_t>(decoder->options_.tts_hidden_size),
                         sizeof(float)});
    const size_t expected_int32_bytes = sizeof(int32_t);
    const std::optional<size_t> expected_logits_bytes = CheckedMultiply(
        {static_cast<size_t>(decoder->options_.tts_vocab_size), sizeof(float)});
    const std::optional<size_t> kv_step_divisor = CheckedMultiply(
        {static_cast<size_t>(2),
         static_cast<size_t>(decoder->options_.tts_num_layers),
         static_cast<size_t>(decoder->options_.tts_max_cache_len),
         sizeof(float)});
    const bool has_gqa_config = decoder->options_.tts_num_kv_heads > 0 &&
                                decoder->options_.tts_head_dim > 0;
    const std::optional<size_t> configured_kv_dim =
        has_gqa_config
            ? CheckedMultiply(
                  {static_cast<size_t>(decoder->options_.tts_num_kv_heads),
                   static_cast<size_t>(decoder->options_.tts_head_dim)})
            : std::optional<size_t>(
                  static_cast<size_t>(decoder->options_.tts_hidden_size));
    const std::optional<size_t> expected_gqa_kv_bytes =
        (kv_step_divisor.has_value() && configured_kv_dim.has_value())
            ? CheckedMultiply({*kv_step_divisor, *configured_kv_dim})
            : std::nullopt;
    const std::optional<size_t> expected_mha_kv_bytes =
        kv_step_divisor.has_value()
            ? CheckedMultiply(
                  {*kv_step_divisor,
                   static_cast<size_t>(decoder->options_.tts_hidden_size)})
            : std::nullopt;
    if (!kv_step_divisor.has_value() ||
        (has_gqa_config ? !expected_gqa_kv_bytes.has_value()
                        : !expected_mha_kv_bytes.has_value())) {
      return absl::InvalidArgumentError(
          "Configured TTS KV-cache dimensions overflow size_t.");
    }

    LITERT_ASSIGN_OR_RETURN(RankedTensorType kv_in_type,
                            decoder->decoder_input_buffers_[3].TensorType());
    LITERT_ASSIGN_OR_RETURN(RankedTensorType kv_out_type,
                            decoder->decoder_output_buffers_[1].TensorType());
    const bool kv_types_compatible =
        (kv_in_type == kv_out_type) &&
        (kv_in_type.ElementType() == ElementType::Float32);

    bool valid_kv_bytes = kv_types_compatible &&
                          (kv_in_bytes == kv_out_bytes) &&
                          ((expected_gqa_kv_bytes.has_value() &&
                            kv_in_bytes == *expected_gqa_kv_bytes) ||
                           (expected_mha_kv_bytes.has_value() &&
                            kv_in_bytes == *expected_mha_kv_bytes));
    if (!valid_kv_bytes && decoder->options_.allow_inferred_kv_shape &&
        kv_types_compatible && kv_in_bytes == kv_out_bytes &&
        kv_in_type.Layout().Rank() == 4) {
      auto dims = kv_in_type.Layout().Dimensions();
      Expected<size_t> num_elements = kv_in_type.Layout().NumElements();
      const std::optional<size_t> inferred_bytes =
          num_elements.HasValue()
              ? CheckedMultiply({*num_elements, sizeof(float)})
              : std::nullopt;
      if (dims[0] == 1 && dims[1] >= 2 && dims[1] % 2 == 0 && dims[2] > 0 &&
          dims[3] > 0 && inferred_bytes.has_value() &&
          kv_in_bytes == *inferred_bytes) {
        decoder->options_.tts_num_layers = dims[1] / 2;
        decoder->options_.tts_max_cache_len = dims[2];
        valid_kv_bytes = true;
      }
    }

    if (!expected_cond_bytes.has_value() ||
        !expected_logits_bytes.has_value() ||
        std::tie(cond_bytes, code_bytes, step_bytes, logits_bytes) !=
            std::tie(*expected_cond_bytes, expected_int32_bytes,
                     expected_int32_bytes, *expected_logits_bytes) ||
        !valid_kv_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_decoder buffer sizes/types do not match configured TTS "
          "dimensions: inputs=(",
          cond_bytes, ", ", code_bytes, ", ", step_bytes, ", ", kv_in_bytes,
          ") bytes vs expected (", FormatCheckedSize(expected_cond_bytes), ", ",
          sizeof(int32_t), ", ", sizeof(int32_t), ", ",
          FormatCheckedSize(expected_gqa_kv_bytes), " or ",
          FormatCheckedSize(expected_mha_kv_bytes), ") bytes; outputs=(",
          logits_bytes, ", ", kv_out_bytes, ") bytes vs expected (",
          FormatCheckedSize(expected_logits_bytes), ", ",
          FormatCheckedSize(expected_gqa_kv_bytes), " or ",
          FormatCheckedSize(expected_mha_kv_bytes), ") bytes."));
    }
    if (decoder->options_.repetition_penalty_mode ==
        RepetitionPenaltyMode::kUniqueHistory) {
      decoder->seen_code_epochs_.assign(
          static_cast<size_t>(decoder->options_.tts_vocab_size), 0);
    }
  } else if (decoder->options_.generate_audio) {
    return absl::NotFoundError(
        "Any2AnyTtsDecoder requires tts_decoder_runner when generate_audio is "
        "enabled.");
  }

  return decoder;
}

int Any2AnyTtsDecoder::ComputeMaxTotalCodes(int num_tokens,
                                            int max_tts_tokens) {
  if (max_tts_tokens >= kDefaultMaxTtsTokensThreshold) {
    return std::max(max_tts_tokens,
                    std::max(0, num_tokens) * kMaxAudioCodesPerTextToken);
  }
  return std::max(1, max_tts_tokens);
}

absl::Status Any2AnyTtsDecoder::ValidateLlmAudioInputs(
    int num_tokens, int llm_hidden_size,
    absl::Span<const int> generated_token_ids,
    absl::Span<const float> llm_hidden_states,
    absl::Span<const int> prompt_token_ids,
    absl::Span<const float> prompt_hidden_states) {
  if (num_tokens < 0 || llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid LLM audio input dimensions: num_tokens (",
                     num_tokens, ") must be non-negative and llm_hidden_size (",
                     llm_hidden_size, ") must be positive."));
  }
  const std::optional<size_t> expected_hidden_size = CheckedMultiply(
      {static_cast<size_t>(num_tokens), static_cast<size_t>(llm_hidden_size)});
  if (!expected_hidden_size.has_value() ||
      llm_hidden_states.size() != *expected_hidden_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("LLM llm_hidden_states size (", llm_hidden_states.size(),
                     ") does not match num_tokens * llm_hidden_size (",
                     FormatCheckedSize(expected_hidden_size), ")."));
  }
  if (!generated_token_ids.empty() &&
      generated_token_ids.size() != static_cast<size_t>(num_tokens)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "LLM generated_token_ids size (", generated_token_ids.size(),
        ") does not match num_tokens (", num_tokens, ")."));
  }
  if (!prompt_token_ids.empty() || !prompt_hidden_states.empty()) {
    const std::optional<size_t> expected_prompt_hidden = CheckedMultiply(
        {prompt_token_ids.size(), static_cast<size_t>(llm_hidden_size)});
    if (!expected_prompt_hidden.has_value() ||
        prompt_hidden_states.size() != *expected_prompt_hidden) {
      return absl::InvalidArgumentError(absl::StrCat(
          "LLM prompt_hidden_states size (", prompt_hidden_states.size(),
          ") does not match prompt_token_ids.size() * llm_hidden_size (",
          FormatCheckedSize(expected_prompt_hidden), ")."));
    }
  }
  return absl::OkStatus();
}

void Any2AnyTtsDecoder::ApplyRepetitionPenalty(absl::Span<float> logits,
                                               absl::Span<const int> history,
                                               float penalty,
                                               RepetitionPenaltyMode mode,
                                               int window_size) {
  if (!(penalty > 1.0f) || !std::isfinite(penalty) || history.empty()) {
    return;
  }
  if (mode == RepetitionPenaltyMode::kFrequencyWindow) {
    if (window_size <= 0) {
      return;
    }
    const size_t start =
        (history.size() > static_cast<size_t>(window_size))
            ? (history.size() - static_cast<size_t>(window_size))
            : 0;
    for (size_t i = start; i < history.size(); ++i) {
      const int code = history[i];
      if (code < 0 || static_cast<size_t>(code) >= logits.size()) {
        continue;
      }
      ApplySingleRepetitionPenalty(logits[code], penalty);
    }
    return;
  }

  absl::flat_hash_set<int> seen_codes;
  seen_codes.reserve(history.size());
  for (int code : history) {
    if (code < 0 || static_cast<size_t>(code) >= logits.size()) {
      continue;
    }
    if (seen_codes.insert(code).second) {
      ApplySingleRepetitionPenalty(logits[code], penalty);
    }
  }
}

void Any2AnyTtsDecoder::ApplyRepetitionPenalty(absl::Span<float> logits,
                                               absl::Span<const int> history) {
  if (options_.repetition_penalty_mode ==
      RepetitionPenaltyMode::kUniqueHistory) {
    if (!(options_.repetition_penalty > 1.0f) ||
        !std::isfinite(options_.repetition_penalty) || history.empty()) {
      return;
    }
    if (seen_code_epochs_.size() < logits.size()) {
      seen_code_epochs_.resize(logits.size(), 0);
    }
    if (++seen_code_epoch_ == 0) {
      absl::c_fill(seen_code_epochs_, 0);
      seen_code_epoch_ = 1;
    }
    for (int code : history) {
      if (code < 0 || static_cast<size_t>(code) >= logits.size()) {
        continue;
      }
      if (seen_code_epochs_[code] == seen_code_epoch_) {
        continue;
      }
      seen_code_epochs_[code] = seen_code_epoch_;
      ApplySingleRepetitionPenalty(logits[code], options_.repetition_penalty);
    }
    return;
  }

  ApplyRepetitionPenalty(logits, history, options_.repetition_penalty,
                         options_.repetition_penalty_mode,
                         options_.repetition_window);
}

int Any2AnyTtsDecoder::SampleAudioCode(absl::Span<const float> logits) {
  return sampler_.Sample(logits);
}

absl::Status Any2AnyTtsDecoder::ProjectTokenConditioning(
    absl::Span<const float> token_hidden, int32_t token_id,
    absl::Span<float> out_cond) {
  if (tts_projector_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "Any2AnyTtsDecoder requires tts_projector_runner to project token "
        "conditioning.");
  }
  if (token_id < -1 ||
      (options_.llm_vocab_size > 0 && token_id >= options_.llm_vocab_size)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyTtsDecoder received out-of-range text token ID (", token_id,
        "); expected -1 or [0, ",
        options_.llm_vocab_size > 0 ? absl::StrCat(options_.llm_vocab_size)
                                    : "unbounded",
        ")."));
  }
  if (token_hidden.size() != static_cast<size_t>(options_.llm_hidden_size)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyTtsDecoder token_hidden size (", token_hidden.size(),
        ") does not match llm_hidden_size (", options_.llm_hidden_size, ")."));
  }
  if (out_cond.size() != static_cast<size_t>(options_.tts_hidden_size)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyTtsDecoder out_cond size (", out_cond.size(),
        ") does not match tts_hidden_size (", options_.tts_hidden_size, ")."));
  }
  LITERT_RETURN_IF_ERROR(proj_input_buffers_[0].Write<float>(token_hidden));
  LITERT_RETURN_IF_ERROR(
      proj_input_buffers_[1].Write<int32_t>(absl::MakeConstSpan(&token_id, 1)));
  ABSL_RETURN_IF_ERROR(tts_projector_runner_->Run("", proj_input_buffers_,
                                                  proj_output_buffers_));
  LITERT_RETURN_IF_ERROR(proj_output_buffers_[0].Read<float>(out_cond));
  return absl::OkStatus();
}

absl::Status Any2AnyTtsDecoder::ClearKvCache() {
  if (tts_decoder_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "Any2AnyTtsDecoder requires tts_decoder_runner to clear KV cache.");
  }
  LITERT_RETURN_IF_ERROR(decoder_input_buffers_[3].Clear());
  return absl::OkStatus();
}

absl::Status Any2AnyTtsDecoder::RunDecoderStep(
    absl::Span<const float> cond_emb, int32_t prev_audio_code, int32_t step_idx,
    std::optional<absl::Span<float>> out_logits) {
  if (tts_decoder_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "Any2AnyTtsDecoder requires tts_decoder_runner to decode speech "
        "tokens.");
  }
  if (prev_audio_code < -1 || prev_audio_code >= options_.tts_vocab_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyTtsDecoder received out-of-range prev_audio_code (",
        prev_audio_code, "); expected -1 or [0, ", options_.tts_vocab_size,
        ")."));
  }
  if (step_idx < 0 || step_idx >= options_.tts_max_cache_len) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyTtsDecoder received out-of-range step_idx (", step_idx,
        "); expected [0, ", options_.tts_max_cache_len, ")."));
  }
  if (cond_emb.size() != static_cast<size_t>(options_.tts_hidden_size)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyTtsDecoder cond_emb size (", cond_emb.size(),
        ") does not match tts_hidden_size (", options_.tts_hidden_size, ")."));
  }
  if (out_logits.has_value() &&
      out_logits->size() != static_cast<size_t>(options_.tts_vocab_size)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyTtsDecoder out_logits size (", out_logits->size(),
        ") does not match tts_vocab_size (", options_.tts_vocab_size, ")."));
  }
  LITERT_RETURN_IF_ERROR(decoder_input_buffers_[0].Write<float>(cond_emb));
  LITERT_RETURN_IF_ERROR(decoder_input_buffers_[1].Write<int32_t>(
      absl::MakeConstSpan(&prev_audio_code, 1)));
  LITERT_RETURN_IF_ERROR(decoder_input_buffers_[2].Write<int32_t>(
      absl::MakeConstSpan(&step_idx, 1)));
  ABSL_RETURN_IF_ERROR(tts_decoder_runner_->Run("", decoder_input_buffers_,
                                                decoder_output_buffers_));
  if (out_logits.has_value()) {
    LITERT_RETURN_IF_ERROR(decoder_output_buffers_[0].Read<float>(*out_logits));
  }
  std::swap(decoder_input_buffers_[3], decoder_output_buffers_[1]);
  return absl::OkStatus();
}

void Any2AnyTtsDecoder::Reset() { sampler_.Reset(); }

}  // namespace litert::omni::any2any
