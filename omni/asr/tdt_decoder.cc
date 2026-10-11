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

#include "omni/asr/tdt_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/speech_recognizer.h"
#include "omni/asr/utils.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stateful_litert_runner.h"

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kDecodeSignatureName = "decode";
constexpr absl::string_view kStatefulDecodeSignatureName = "decode_1";
constexpr size_t kNumDurations = 5;
// Hardcoded because CompileModel doesn't expose shape info.
constexpr size_t kNumFeatures = 1024;

template <typename T>
absl::StatusOr<size_t> GetNumOfElements(const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto num_bytes, buffer.PackedSize());
  return num_bytes / sizeof(T);
}

absl::StatusOr<std::vector<TensorBuffer>> GetInputBuffersForInference(
    absl::Span<const TensorBuffer> encoder_outputs,
    const TensorBuffer& token_ids_buffer,
    absl::Span<const TensorBuffer> state_buffers) {
  std::vector<TensorBuffer> inputs;
  inputs.reserve(encoder_outputs.size() + 1 + state_buffers.size());
  for (const auto& buffer : encoder_outputs) {
    LITERT_ASSIGN_OR_RETURN(auto dup, buffer.Duplicate());
    inputs.push_back(std::move(dup));
  }
  LITERT_ASSIGN_OR_RETURN(auto dup_tok, token_ids_buffer.Duplicate());
  inputs.push_back(std::move(dup_tok));
  for (const auto& buffer : state_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto dup_st, buffer.Duplicate());
    inputs.push_back(std::move(dup_st));
  }
  return inputs;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TdtDecoder>> TdtDecoder::Create(
    LiteRtRunner* absl_nonnull runner, int decode_start_token_id,
    int decode_statefully_after, int max_symbols_per_step) {
  if (max_symbols_per_step <= 0) {
    return absl::InvalidArgumentError("max_symbols_per_step must be positive.");
  }
  LITERT_ASSIGN_OR_RETURN(auto inputs,
                          runner->CreateInputBuffers(kDecodeSignatureName));
  // encode output, token ids, input LSTM state 1, input LSTM state 2.
  LITERT_RETURN_IF_ERROR(inputs.size() >= 4);
  LITERT_ASSIGN_OR_RETURN(auto outputs,
                          runner->CreateOutputBuffers(kDecodeSignatureName));
  // logits, output LSTM state 1, output LSTM state 2.
  LITERT_RETURN_IF_ERROR(outputs.size() >= 3);

  ABSL_ASSIGN_OR_RETURN(auto num_input0, GetNumOfElements<float>(inputs[0]));
  size_t max_time_index = num_input0 / kNumFeatures;
  ABSL_ASSIGN_OR_RETURN(auto num_token_ids,
                        GetNumOfElements<int32_t>(inputs[1]));
  if (max_time_index == 0 || num_token_ids == 0) {
    return absl::InvalidArgumentError(
        "Decoder input tensors must have non-zero time and token dimensions.");
  }
  ABSL_ASSIGN_OR_RETURN(auto total_num_logits,
                        GetNumOfElements<float>(outputs[0]));
  size_t num_logits_per_token =
      total_num_logits / num_token_ids / max_time_index;
  if (num_logits_per_token == 0 || decode_start_token_id < 0) {
    return absl::InvalidArgumentError(
        "decode_start_token_id and num_logits_per_token must be valid.");
  }

  std::unique_ptr<StatefulLiteRtRunner> stateful_runner;
  std::optional<TensorBuffer> stateful_decode_token_ids_buf;

  auto stateful_runner_res = StatefulLiteRtRunnerImpl::Create(
      runner, kStatefulDecodeSignatureName,
      /*num_non_state_inputs=*/2, /*num_non_state_outputs=*/1);
  if (stateful_runner_res.ok()) {
    stateful_runner = std::move(*stateful_runner_res);
    auto non_state_in = stateful_runner->GetNonStateInputBuffers();
    LITERT_RETURN_IF_ERROR(non_state_in.size() >= 2);
    LITERT_ASSIGN_OR_RETURN(stateful_decode_token_ids_buf,
                            non_state_in[1].Duplicate());

    // Wire up stateless decode outputs directly into stateful runner active
    // input states so transitioning from stateless to stateful requires zero
    // memory copies.
    auto active_states = stateful_runner->GetActiveInputStates();
    for (size_t s = 0; s < active_states.size() && (s + 1) < outputs.size();
         ++s) {
      LITERT_ASSIGN_OR_RETURN(outputs[s + 1], active_states[s].Duplicate());
    }
  }

  return std::unique_ptr<TdtDecoder>(new TdtDecoder(
      runner, std::move(stateful_runner), std::move(inputs), std::move(outputs),
      std::move(stateful_decode_token_ids_buf), max_time_index, num_token_ids,
      num_logits_per_token, decode_start_token_id, decode_statefully_after,
      max_symbols_per_step));
}

TdtDecoder::TdtDecoder(
    LiteRtRunner* absl_nonnull runner,
    std::unique_ptr<StatefulLiteRtRunner> absl_nullable stateful_runner,
    std::vector<TensorBuffer> decode_input_buffers,
    std::vector<TensorBuffer> decode_output_buffers,
    std::optional<TensorBuffer> stateful_decode_token_ids_buffer,
    size_t max_time_index, size_t num_token_ids, size_t num_logits_per_token,
    int decode_start_token_id, int decode_statefully_after,
    int max_symbols_per_step)
    : runner_(*runner),
      stateful_runner_(std::move(stateful_runner)),
      decode_input_buffers_(std::move(decode_input_buffers)),
      decode_output_buffers_(std::move(decode_output_buffers)),
      stateful_decode_token_ids_buffer_(
          std::move(stateful_decode_token_ids_buffer)),
      max_time_index_(max_time_index),
      num_token_ids_(num_token_ids),
      num_logits_per_token_(num_logits_per_token),
      decode_start_token_id_(decode_start_token_id),
      decode_statefully_after_(decode_statefully_after),
      max_symbols_per_step_(max_symbols_per_step) {}

void TdtDecoder::SetDecodeRange(const DecodeRange& range) {
  range_.start_fraction =
      (range.start_fraction > 0.0f && range.start_fraction < 1.0f)
          ? range.start_fraction
          : 0.0f;
  range_.valid_fraction =
      (range.valid_fraction > 0.0f && range.valid_fraction < 1.0f)
          ? range.valid_fraction
          : 1.0f;
}

size_t TdtDecoder::EffectiveMaxTimeIndex() const {
  if (range_.valid_fraction >= 1.0f) return max_time_index_;
  const size_t valid_frames = static_cast<size_t>(
      std::lround(range_.valid_fraction * static_cast<float>(max_time_index_)));
  return std::min(max_time_index_, valid_frames + kValidFrameMargin);
}

size_t TdtDecoder::FirstReportedTimeIndex() const {
  if (range_.start_fraction <= 0.0f) return 0;
  // Frames overlapping the boundary are kept (the merger de-duplicates them).
  return static_cast<size_t>(
      std::floor(range_.start_fraction * static_cast<float>(max_time_index_)));
}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> TdtDecoder::Decode(
    std::vector<TensorBuffer>& encoder_outputs) {
  const int blank_token_id = decode_start_token_id_;
  // Only decode encoder frames backed by real audio; the zero-padded tail of
  // a short final chunk otherwise yields repeated / hallucinated tokens. Tokens
  // before `first_reported_time_index` belong to audio already reported with
  // the previous chunk (a final chunk shifted back to fill the window).
  const size_t effective_max_time_index = EffectiveMaxTimeIndex();
  const size_t first_reported_time_index = FirstReportedTimeIndex();
  range_ = DecodeRange();

  if (stateful_runner_) {
    LITERT_RETURN_IF_ERROR(stateful_runner_->Reset());
  }
  for (size_t i = 2; i < decode_input_buffers_.size(); ++i) {
    LITERT_RETURN_IF_ERROR(decode_input_buffers_[i].Clear());
  }

  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  // Start inference with decode signature for better quality in sequence start
  // as stateless decoding can keep context better than decode_1, stateful
  // decoding.
  TensorBuffer* current_token_ids_buffer = &decode_input_buffers_[1];
  int num_inference_token_ids = num_token_ids_;
  std::vector<int32_t> token_ids(num_token_ids_, 0);
  token_ids[0] = decode_start_token_id_;
  size_t token_index = 0;
  size_t time_index = 0;
  int symbols_at_time_index = 0;
  size_t frame_tokens_start = 0;
  size_t sos_tokens_start = 0;
  size_t tokens_since_sos = 0;
  size_t last_emitted_time_index = 0;
  size_t reset_origin_time_index = 0;
  bool has_emitted_in_chunk = false;
  bool did_mid_chunk_sos_reset = false;
  bool prev_frame_had_burst = false;
  std::vector<int> frame_token_ids;

  auto reset_to_sos = [&]() -> absl::Status {
    current_token_ids_buffer = &decode_input_buffers_[1];
    num_inference_token_ids = num_token_ids_;
    token_ids.assign(num_token_ids_, 0);
    token_ids[0] = decode_start_token_id_;
    token_index = 0;
    tokens_since_sos = 0;
    sos_tokens_start = decoded_tokens.size();
    prev_frame_had_burst = false;
    if (stateful_runner_) {
      LITERT_RETURN_IF_ERROR(stateful_runner_->Reset());
    }
    for (size_t i = 2; i < decode_input_buffers_.size(); ++i) {
      LITERT_RETURN_IF_ERROR(decode_input_buffers_[i].Clear());
    }
    return absl::OkStatus();
  };

  while (time_index < effective_max_time_index) {
    LITERT_RETURN_IF_ERROR(current_token_ids_buffer->Write<int32_t>(
        absl::MakeConstSpan(token_ids)));

    const TensorBuffer* current_logits_buffer = nullptr;
    bool was_stateful_step = false;

    if (num_inference_token_ids > 1 || !stateful_runner_) {
      // Stateless decode using LiteRtRunner
      LITERT_ASSIGN_OR_RETURN(
          auto step_inputs,
          GetInputBuffersForInference(
              encoder_outputs, *current_token_ids_buffer,
              absl::MakeSpan(decode_input_buffers_).subspan(2)));
      LITERT_RETURN_IF_ERROR(runner_.Run(kDecodeSignatureName, step_inputs,
                                         decode_output_buffers_));
      current_logits_buffer = &decode_output_buffers_[0];
    } else {
      // Stateful decode using StatefulLiteRtRunner
      LITERT_ASSIGN_OR_RETURN(
          auto step_inputs,
          GetInputBuffersForInference(encoder_outputs,
                                      *current_token_ids_buffer, {}));
      LITERT_ASSIGN_OR_RETURN(
          auto step_outputs,
          stateful_runner_->Step(step_inputs, /*auto_commit_state=*/false));
      if (step_outputs.empty()) {
        return absl::InternalError("No outputs returned from stateful runner.");
      }
      current_logits_buffer = &step_outputs[0];
      was_stateful_step = true;
    }

    LITERT_ASSIGN_OR_RETURN(
        auto lock, TensorBufferScopedLock::Create<const float>(
                       *current_logits_buffer, TensorBuffer::LockMode::kRead));
    const float* logits = lock.second;

    size_t start_index_in_current_time_index =
        time_index * num_inference_token_ids * num_logits_per_token_;
    size_t start_index_of_token_id =
        start_index_in_current_time_index + token_index * num_logits_per_token_;
    size_t end_index_of_duration = start_index_in_current_time_index +
                                   (token_index + 1) * num_logits_per_token_;
    size_t end_index_of_token_id = end_index_of_duration - kNumDurations;
    auto max_token_it = std::max_element(logits + start_index_of_token_id,
                                         logits + end_index_of_token_id);
    int token_id = static_cast<int>(
        std::distance(logits + start_index_of_token_id, max_token_it));
    if (token_id == blank_token_id && token_index == 0 &&
        num_inference_token_ids > 1 && effective_max_time_index >= 50 &&
        time_index >= 2) {
      auto best_non_blank_it = [&](size_t start,
                                   size_t end) -> const float* {
        const float* best = nullptr;
        const size_t count = end - start;
        for (size_t tok = 0; tok < count; ++tok) {
          if (static_cast<int>(tok) == blank_token_id) continue;
          const float* cur = logits + start + tok;
          if (best == nullptr || *cur > *best) {
            best = cur;
          }
        }
        return best != nullptr ? best : (logits + start);
      };
      const float* best_nb_it =
          best_non_blank_it(start_index_of_token_id, end_index_of_token_id);
      const float cur_diff =
          logits[start_index_of_token_id + blank_token_id] - *best_nb_it;
      bool is_local_min = true;
      if (time_index + 1 < effective_max_time_index) {
        const size_t next_start =
            (time_index + 1) * num_inference_token_ids * num_logits_per_token_;
        const size_t next_end_tok =
            next_start + num_logits_per_token_ - kNumDurations;
        const float* next_nb_it = best_non_blank_it(next_start, next_end_tok);
        const float next_diff =
            logits[next_start + blank_token_id] - *next_nb_it;
        if (next_diff < cur_diff) {
          is_local_min = false;
        }
      }
      bool has_near_natural_sos = false;
      if (cur_diff <= 3.0f && is_local_min) {
        const size_t lookahead_end =
            std::min(effective_max_time_index, time_index + 12);
        for (size_t f = time_index + 1; f < lookahead_end; ++f) {
          const size_t f_start =
              f * num_inference_token_ids * num_logits_per_token_;
          const size_t f_end_tok =
              f_start + num_logits_per_token_ - kNumDurations;
          const float* f_nb_it = best_non_blank_it(f_start, f_end_tok);
          if (logits[f_start + blank_token_id] < *f_nb_it) {
            has_near_natural_sos = true;
            break;
          }
        }
      }
      if (cur_diff <= 3.0f && is_local_min && !has_near_natural_sos) {
        token_id = static_cast<int>(
            std::distance(logits + start_index_of_token_id, best_nb_it));
      }
    }
    auto max_duration_it = std::max_element(logits + end_index_of_token_id,
                                            logits + end_index_of_duration);
    int duration = static_cast<int>(
        std::distance(logits + end_index_of_token_id, max_duration_it));

    if (token_id != blank_token_id) {
      ++symbols_at_time_index;
      frame_token_ids.push_back(token_id);
      if (symbols_at_time_index > max_symbols_per_step_) {
        decoded_tokens.resize(frame_tokens_start);
        if (time_index + 2 >= effective_max_time_index) {
          break;
        }
        if (tokens_since_sos <= 2) {
          // A zero-duration burst immediately after the `<sos>` token
          // indicates a premature `<sos>` trigger; reset back to `<sos>` and
          // continue from the next frame.
          decoded_tokens.resize(sos_tokens_start);
          if (decoded_tokens.empty()) {
            has_emitted_in_chunk = false;
          }
          LITERT_RETURN_IF_ERROR(reset_to_sos());
        } else {
          prev_frame_had_burst = true;
        }
        symbols_at_time_index = 0;
        frame_tokens_start = decoded_tokens.size();
        frame_token_ids.clear();
        time_index += 1;
        continue;
      }
      has_emitted_in_chunk = true;
      last_emitted_time_index = time_index;
      const bool skip_post_burst_zero_dur =
          prev_frame_had_burst && duration == 0;
      if (duration > 0) {
        prev_frame_had_burst = false;
      }
      if (!skip_post_burst_zero_dur &&
          time_index >= first_reported_time_index) {
        if (TruncateOnTrailingRepetition(decoded_tokens, token_id)) {
          break;
        }
        decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
            .token_id = token_id,
            .timestamp_ms =
                static_cast<int>(time_index - first_reported_time_index)});
      }
      if (num_inference_token_ids > 1) {
        ++token_index;
        if (token_index < num_inference_token_ids) {
          // Still room in the stateless token array: keep filling it, so the
          // LSTM states handed to decode_1 below are computed from real tokens
          // only (an unfilled zero slot skews them).
        } else if (stateful_decode_token_ids_buffer_.has_value()) {
          // Switch to stateful decoding.
          current_token_ids_buffer = &(*stateful_decode_token_ids_buffer_);
          num_inference_token_ids = 1;
          token_ids.resize(1, 0);
          token_index = 0;
        } else {
          break;
        }
      }
      token_ids[token_index] = token_id;
    } else if (duration == 0 ||
               (token_index == 0 && num_inference_token_ids > 1 &&
                effective_max_time_index >= 50)) {
      duration = 1;
      prev_frame_had_burst = false;
    } else {
      prev_frame_had_burst = false;
    }

    if (duration > 0) {
      tokens_since_sos += symbols_at_time_index;
      symbols_at_time_index = 0;
      frame_tokens_start = decoded_tokens.size();
      frame_token_ids.clear();
    }
    time_index += duration;

    if (was_stateful_step && token_id != blank_token_id && stateful_runner_) {
      // Stateful RNN decoder: adopt the new LSTM states by swapping the input
      // and output state buffers — but only on a non-blank emission. In RNN-T
      // greedy decoding the prediction network advances only when a token is
      // emitted; adopting the state on blank steps re-consumes the last token
      // once per blank and degrades the transcript.
      LITERT_RETURN_IF_ERROR(stateful_runner_->CommitState());
    }

    if (!did_mid_chunk_sos_reset && has_emitted_in_chunk &&
        effective_max_time_index >= 50 && last_emitted_time_index <= 30 &&
        time_index >= 25 && time_index >= last_emitted_time_index + 16) {
      did_mid_chunk_sos_reset = true;
      reset_origin_time_index = last_emitted_time_index;
      if (last_emitted_time_index <= 8) {
        decoded_tokens.clear();
        frame_tokens_start = 0;
      }
      LITERT_RETURN_IF_ERROR(reset_to_sos());
      time_index = last_emitted_time_index + 3;
    }
  }

  if (did_mid_chunk_sos_reset && sos_tokens_start > 0 &&
      decoded_tokens.size() > sos_tokens_start &&
      (decoded_tokens.size() - sos_tokens_start <= 4 ||
       decoded_tokens.back().timestamp_ms.value_or(0) -
               decoded_tokens[sos_tokens_start].timestamp_ms.value_or(0) <=
           4 ||
       static_cast<size_t>(
           decoded_tokens[sos_tokens_start].timestamp_ms.value_or(0)) +
               first_reported_time_index >
           reset_origin_time_index + 20)) {
    decoded_tokens.resize(sos_tokens_start);
  }

  decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
      .token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
      .timestamp_ms =
          static_cast<int>(max_time_index_ - first_reported_time_index)});

  return decoded_tokens;
}

}  // namespace litert::omni::asr
