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

#include "omni/speech2text/tdt_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stateful_litert_runner.h"
#include "omni/speech2text/litert_speech_recognizer.h"
#include "omni/speech2text/speech_recognizer.h"

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kDecodeSignatureName = "decode";
constexpr absl::string_view kStatefulDecodeSignatureName = "decode_1";
constexpr size_t kNumDurations = 5;
// Hardcoded because CompileModel doesn't expose shape info.
constexpr size_t kNumFeatures = 1024;

// The streaming recovery heuristics below are tuned to parakeet-tdt-0.6b-v3's
// 80 ms encoder frames and ~62-frame 5-second streaming window:
//
// Minimum and maximum encoder window capacity (`max_time_index`) identifying a
// 5-second TDT export (4.0 s to 6.0 s at 80 ms/frame), so 10 s or 30 s exports
// do not silently inherit 5-second window heuristics.
constexpr size_t kMin5sWindowFrames = 50;
constexpr size_t kMax5sWindowFrames = 75;
// Minimum active-audio fraction of `max_time_index` (`>= 80%`, i.e. `>= 50`
// frames for a 62-frame 5 s window) required to enable full-window heuristics.
constexpr float kFullWindowMinFraction = 0.8f;
// Earliest encoder frame index (2 frames = 160 ms at 80 ms/frame) eligible for
// near-threshold `<sos>` token selection, skipping chunk-onset transients.
constexpr size_t kSosMinFrameIndex = 2;
// Maximum `blank_logit - best_non_blank_logit` gap at `<sos>` still eligible
// for local-minimum non-blank token selection.
constexpr float kSosBlankMarginThreshold = 3.0f;
// Lookahead window (12 frames = 960 ms at 80 ms/frame) checked for a natural
// non-blank `<sos>` winner before forcing a near-threshold `<sos>` token.
constexpr size_t kSosLookaheadFrames = 12;
// Remaining frames threshold (2 frames = 160 ms at 80 ms/frame) at chunk end
// below which a zero-duration burst terminates decoding instead of advancing.
constexpr size_t kBurstEndOfChunkMarginFrames = 2;
// Maximum tokens fed since `<sos>` (`<= 2` tokens) treated as a premature
// false start and rolled back to `<sos>` when a zero-duration burst occurs.
constexpr size_t kMaxPrematureSosTokens = 2;
// Maximum emission frames since a forced near-threshold `<sos>` (`<= 6`
// frames) treated as a premature forced-`<sos>` false start and rolled back to
// `<sos>` when a zero-duration burst occurs.
constexpr size_t kMaxPrematureForcedSosFrames = 6;
// Minimum consecutive blank frames (16 frames = 1.28 s at 80 ms/frame) after
// an early emission before triggering a mid-chunk `<sos>` reset.
constexpr size_t kDeadZoneMinBlankFrames = 16;
// Earliest current frame index (25 frames = 2.0 s at 80 ms/frame) at which a
// mid-chunk dead-zone `<sos>` reset may fire.
constexpr size_t kDeadZoneMinCurrentFrame = 25;
// Latest prior emission frame index (30 frames = 2.4 s at 80 ms/frame)
// eligible to anchor a mid-chunk dead-zone `<sos>` reset.
constexpr size_t kDeadZoneMaxLastEmittedFrame = 30;
// Latest emission frame index (8 frames = 640 ms at 80 ms/frame) treated as an
// early false start (covered by the overlapping previous chunk) and cleared on
// a mid-chunk `<sos>` reset.
constexpr size_t kDeadZoneClearPrefixMaxFrame = 8;
// Frame offset (`last_emitted_time_index + 3`, 240 ms at 80 ms/frame) to which
// `time_index` rewinds after a mid-chunk `<sos>` reset.
constexpr size_t kDeadZoneRewindOffsetFrames = 3;
// Maximum post-reset token count (`<= 4` tokens) treated as an abortive
// trailing fragment and trimmed when a mid-chunk `<sos>` reset kept a prefix.
constexpr size_t kAbortiveResetMaxTokens = 4;
// Maximum post-reset frame span (`last_ts - first_ts <= 4`, 320 ms at
// 80 ms/frame) treated as an abortive trailing fragment and trimmed.
constexpr int kAbortiveResetMaxFrameSpan = 4;
// Maximum frame delay (`> 20` frames = 1.6 s at 80 ms/frame) after
// `reset_origin_time_index` before a post-reset emission is treated as a late
// trailing artifact and trimmed.
constexpr size_t kAbortiveResetMaxStartDelayFrames = 20;

// TDT (Token-and-Duration Transducer) greedy decoding overview:
//
// At each step, the joint network takes the encoder frame at `time_index` and
// the prediction network state for the last emitted token (`token_index`), and
// outputs two logit heads:
//   [ token logits (vocab + blank) | duration logits (0, 1, 2, 3, 4 frames) ]
//
//   Encoder frames:   t=0       t=1       t=2       t=3
//                      |         |         |         |
//   Step 1 (t=0):    [tok="a", dur=0]
//                      | (dur=0 stays at t=0, advances prediction state)
//   Step 2 (t=0):    [tok="b", dur=2] -------------> |
//                      (emits "b", jumps +2 frames)  |
//   Step 3 (t=2):                                  [blank, dur=1] -> |
//                                                    (keeps prediction state)
//
// - Non-blank token: emit `token_id`, advance the prediction network state with
//   `token_id`, and advance `time_index += duration` (`duration == 0` emits
//   another token at the same frame, capped by `max_symbols_per_step`).
// - Blank token: keep the prediction network state unchanged and advance
//   `time_index += max(1, duration)`.
// - Stateless -> Stateful transition: decoding starts with the multi-token
//   stateless `decode` signature (filling `token_ids[0..N-1]` from `<sos>`) and
//   switches to the single-token stateful `decode_1` signature once those slots
//   are filled.
//
// Mutable decoding state tracked across encoder frames within a single chunk.
struct DecodeState {
  // Exclusive upper bound on encoder frames to decode in this chunk.
  size_t effective_max_time_index = 0;
  // First encoder frame index whose tokens are reported in `decoded_tokens`.
  size_t first_reported_time_index = 0;
  // Whether this chunk is a nearly full 5-second streaming window (`>= 80%` of
  // a 5-second `max_time_index` export) eligible for the 5-second window
  // recovery heuristics.
  bool is_full_5s_window = false;
  // Active token IDs input buffer (`decode` or `decode_1`).
  TensorBuffer* current_token_ids_buffer = nullptr;
  // Number of token slots in `current_token_ids_buffer` (> 1 for stateless
  // `decode`, 1 for stateful `decode_1`).
  int num_inference_token_ids = 0;
  // Token IDs written to `current_token_ids_buffer` before each step.
  std::vector<int32_t> token_ids;
  // Current slot index in `token_ids` being populated and read from logits.
  size_t token_index = 0;
  // Current encoder time frame index `[0, effective_max_time_index)`.
  size_t time_index = 0;
  // Number of non-blank tokens emitted at `time_index` via zero-duration steps.
  int symbols_at_time_index = 0;
  // Index in `decoded_tokens` where the current frame's emissions started,
  // used to roll back zero-duration bursts exceeding `max_symbols_per_step`.
  size_t frame_tokens_start = 0;
  // Index in `decoded_tokens` where the current `<sos>` segment started.
  size_t sos_tokens_start = 0;
  // Number of non-blank tokens fed to the prediction network at earlier frames
  // since the last `<sos>` reset.
  size_t tokens_since_sos = 0;
  // Number of encoder frames that emitted at least one non-blank token since
  // the last `<sos>` reset.
  size_t sos_emission_frames = 0;
  // Whether the current `<sos>` segment was initiated by
  // `SelectNearThresholdSosToken` rather than a natural non-blank argmax.
  bool sos_was_forced = false;
  // Most recent encoder frame index at which a non-blank token was emitted.
  size_t last_emitted_time_index = 0;
  // Value of `last_emitted_time_index` when a mid-chunk `<sos>` reset fired.
  size_t reset_origin_time_index = 0;
  // Whether any non-blank token has been emitted in the current chunk.
  bool has_emitted_in_chunk = false;
  // Whether the once-per-chunk mid-chunk dead-zone `<sos>` reset has fired.
  bool did_mid_chunk_sos_reset = false;
  // Whether the previous encoder frame rolled back a zero-duration burst.
  bool prev_frame_had_burst = false;
  // Cached output logits buffer from the most recent `RunDecodeStep` call,
  // reused across consecutive blank frames while prediction network inputs
  // (`token_ids` and LSTM state) are unchanged.
  const TensorBuffer* current_logits_buffer = nullptr;
  // Whether `current_logits_buffer` was produced by `stateful_runner`.
  bool was_stateful_step = false;
  // Output tokens accumulated for the current chunk.
  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
};

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

// Initializes `effective_max_time_index`, `first_reported_time_index`, and
// `is_full_5s_window` on `state` from `max_time_index` and `decode_range`.
// Only encoder frames backed by real audio are decoded, and tokens before
// `first_reported_time_index` (already reported with the previous chunk) are
// suppressed.
void InitDecodeBounds(
    size_t max_time_index,
    std::optional<LiteRtSpeechRecognizer::Decoder::DecodeRange> decode_range,
    DecodeState& state) {
  state.effective_max_time_index = max_time_index;
  state.first_reported_time_index = 0;
  if (decode_range.has_value()) {
    if (decode_range->valid_fraction > 0.0f &&
        decode_range->valid_fraction < 1.0f) {
      state.effective_max_time_index = std::min(
          max_time_index,
          static_cast<size_t>(std::lround(decode_range->valid_fraction *
                                          static_cast<float>(max_time_index))));
    }
    if (decode_range->start_fraction > 0.0f &&
        decode_range->start_fraction < 1.0f) {
      state.first_reported_time_index = static_cast<size_t>(std::floor(
          decode_range->start_fraction * static_cast<float>(max_time_index)));
    }
  }
  const size_t min_full_window_frames = static_cast<size_t>(
      std::lround(kFullWindowMinFraction * static_cast<float>(max_time_index)));
  state.is_full_5s_window =
      max_time_index >= kMin5sWindowFrames &&
      max_time_index <= kMax5sWindowFrames &&
      state.effective_max_time_index >= min_full_window_frames;
}

// Returns a pointer to the highest logit in `logits[start..end)` excluding
// `blank_token_id` (falling back to `logits + start` if all slots are blank).
const float* FindBestNonBlankLogit(const float* logits, size_t start,
                                   size_t end, int blank_token_id) {
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
}

// When the greedy prediction at `token_index == 0` (`<sos>`) in a full 5-second
// window (`state.is_full_5s_window`) is `blank_token_id`, checks whether the
// gap `blank_logit - best_non_blank_logit` has a local minimum
// `<= kSosBlankMarginThreshold` at `time_index` with no natural non-blank
// `<sos>` emission in the next `kSosLookaheadFrames` frames, and returns the
// best non-blank token ID if so.
int SelectNearThresholdSosToken(const float* logits, const DecodeState& state,
                                size_t num_logits_per_token,
                                size_t start_index_of_token_id,
                                size_t end_index_of_token_id,
                                int blank_token_id) {
  const float* best_nb_it = FindBestNonBlankLogit(
      logits, start_index_of_token_id, end_index_of_token_id, blank_token_id);
  const float cur_diff =
      logits[start_index_of_token_id + blank_token_id] - *best_nb_it;
  if (cur_diff > kSosBlankMarginThreshold) {
    return blank_token_id;
  }
  const size_t lookahead_end = std::min(state.effective_max_time_index,
                                        state.time_index + kSosLookaheadFrames);
  for (size_t f = state.time_index + 1; f < lookahead_end; ++f) {
    const size_t f_start =
        f * state.num_inference_token_ids * num_logits_per_token;
    const size_t f_end_tok = f_start + num_logits_per_token - kNumDurations;
    const float* f_nb_it =
        FindBestNonBlankLogit(logits, f_start, f_end_tok, blank_token_id);
    const float f_diff = logits[f_start + blank_token_id] - *f_nb_it;
    if (f_diff < (f == state.time_index + 1 ? cur_diff : 0.0f)) {
      return blank_token_id;
    }
  }
  return static_cast<int>(
      std::distance(logits + start_index_of_token_id, best_nb_it));
}

// Selects the greedy token ID and duration step `[0, kNumDurations)` from
// `logits` for `(state.time_index, state.token_index)`, applying the
// near-threshold `<sos>` fallback when `<sos>` predicts `blank_token_id` in a
// full 5-second window.
std::pair<int, int> SelectGreedyTokenAndDuration(const float* logits,
                                                 DecodeState& state,
                                                 size_t num_logits_per_token,
                                                 int blank_token_id) {
  const size_t start_index_in_current_time_index =
      state.time_index * state.num_inference_token_ids * num_logits_per_token;
  const size_t start_index_of_token_id =
      start_index_in_current_time_index +
      state.token_index * num_logits_per_token;
  const size_t end_index_of_duration =
      start_index_in_current_time_index +
      (state.token_index + 1) * num_logits_per_token;
  const size_t end_index_of_token_id = end_index_of_duration - kNumDurations;
  auto max_token_it = std::max_element(logits + start_index_of_token_id,
                                       logits + end_index_of_token_id);
  int token_id = static_cast<int>(
      std::distance(logits + start_index_of_token_id, max_token_it));
  if (token_id == blank_token_id && state.token_index == 0 &&
      state.num_inference_token_ids > 1 && state.is_full_5s_window &&
      state.time_index >= kSosMinFrameIndex) {
    token_id = SelectNearThresholdSosToken(
        logits, state, num_logits_per_token, start_index_of_token_id,
        end_index_of_token_id, blank_token_id);
    if (token_id != blank_token_id) {
      state.sos_was_forced = true;
    }
  }
  auto max_duration_it = std::max_element(logits + end_index_of_token_id,
                                          logits + end_index_of_duration);
  int duration = static_cast<int>(
      std::distance(logits + end_index_of_token_id, max_duration_it));
  return {token_id, duration};
}

// Resets the decoder prediction network to the initial `<sos>` state by
// switching back to the multi-token stateless `decode` buffer seeded with
// `blank_token_id` at slot 0, resetting `stateful_runner`, and clearing the
// stateless LSTM state input buffers.
absl::Status ResetToSos(std::vector<TensorBuffer>& decode_input_buffers,
                        StatefulLiteRtRunner* stateful_runner,
                        size_t num_token_ids, int blank_token_id,
                        DecodeState& state) {
  if (decode_input_buffers.size() <= 1) {
    return absl::InternalError("Insufficient decode input buffers.");
  }
  state.current_token_ids_buffer = &decode_input_buffers[1];
  state.num_inference_token_ids = num_token_ids;
  state.token_ids.assign(num_token_ids, 0);
  if (!state.token_ids.empty()) {
    state.token_ids[0] = blank_token_id;
  }
  state.token_index = 0;
  state.sos_tokens_start = state.decoded_tokens.size();
  state.tokens_since_sos = 0;
  state.sos_emission_frames = 0;
  state.sos_was_forced = false;
  state.prev_frame_had_burst = false;
  state.current_logits_buffer = nullptr;
  state.was_stateful_step = false;
  if (stateful_runner != nullptr) {
    LITERT_RETURN_IF_ERROR(stateful_runner->Reset());
  }
  for (size_t i = 2; i < decode_input_buffers.size(); ++i) {
    LITERT_RETURN_IF_ERROR(decode_input_buffers[i].Clear());
  }
  return absl::OkStatus();
}

// Executes one decoder step using either stateless `decode` or stateful
// `decode_1`, returning the output logits buffer and setting
// `was_stateful_step`.
absl::StatusOr<const TensorBuffer*> RunDecodeStep(
    LiteRtRunner& runner, StatefulLiteRtRunner* stateful_runner,
    absl::Span<const TensorBuffer> encoder_outputs,
    absl::Span<const TensorBuffer> decode_input_buffers,
    std::vector<TensorBuffer>& decode_output_buffers, DecodeState& state,
    bool& was_stateful_step) {
  LITERT_RETURN_IF_ERROR(state.current_token_ids_buffer->Write<int32_t>(
      absl::MakeConstSpan(state.token_ids)));
  was_stateful_step = false;
  if (state.num_inference_token_ids > 1 || stateful_runner == nullptr) {
    LITERT_ASSIGN_OR_RETURN(
        auto step_inputs, GetInputBuffersForInference(
                              encoder_outputs, *state.current_token_ids_buffer,
                              decode_input_buffers.subspan(2)));
    LITERT_RETURN_IF_ERROR(
        runner.Run(kDecodeSignatureName, step_inputs, decode_output_buffers));
    if (decode_output_buffers.empty()) {
      return absl::InternalError("No outputs returned from stateless runner.");
    }
    return &decode_output_buffers[0];
  }
  LITERT_ASSIGN_OR_RETURN(
      auto step_inputs,
      GetInputBuffersForInference(encoder_outputs,
                                  *state.current_token_ids_buffer, {}));
  LITERT_ASSIGN_OR_RETURN(
      auto step_outputs,
      stateful_runner->Step(step_inputs, /*auto_commit_state=*/false));
  if (step_outputs.empty()) {
    return absl::InternalError("No outputs returned from stateful runner.");
  }
  was_stateful_step = true;
  return &step_outputs[0];
}

// Rolls back tokens emitted during a zero-duration burst on the current frame.
// - If the burst occurs within the final `kBurstEndOfChunkMarginFrames` frames
//   of the chunk, returns false to terminate decoding immediately without
//   rolling back `sos_tokens_start`, preserving any `<= kMaxPrematureSosTokens`
//   valid tokens emitted earlier in the segment since no frames remain to
//   re-decode after a reset.
// - Otherwise, if the burst occurred near `<sos>` (`tokens_since_sos <=
//   kMaxPrematureSosTokens`), rewinds `decoded_tokens` to `sos_tokens_start`
//   and resets the prediction network to `<sos>`.
// - When `tokens_since_sos > kMaxPrematureSosTokens`, only the current frame's
//   burst outputs are removed from `decoded_tokens` while the prediction
//   network intentionally retains the tokens consumed during the burst
//   (`CommitState()` already ran for them in stateful mode and `token_ids`
//   still holds the last one), keeping subsequent frames conditioned on the
//   ongoing sequence rather than resetting to `<sos>`.
absl::StatusOr<bool> RollbackFrameBurstAndAdvance(
    std::vector<TensorBuffer>& decode_input_buffers,
    StatefulLiteRtRunner* stateful_runner, size_t num_token_ids,
    int blank_token_id, DecodeState& state) {
  const bool is_premature_forced_sos_burst =
      state.sos_was_forced &&
      state.sos_emission_frames <= kMaxPrematureForcedSosFrames;
  state.decoded_tokens.resize(is_premature_forced_sos_burst
                                  ? state.sos_tokens_start
                                  : state.frame_tokens_start);
  // Intentional early exit before the unforced premature-`<sos>` check: at the
  // end of the chunk there are no remaining frames to re-decode after a reset,
  // so keep any earlier unforced tokens from the current `<sos>` segment and
  // stop decoding.
  if (state.time_index + kBurstEndOfChunkMarginFrames >=
      state.effective_max_time_index) {
    return false;
  }
  if (is_premature_forced_sos_burst ||
      state.tokens_since_sos <= kMaxPrematureSosTokens) {
    state.decoded_tokens.resize(state.sos_tokens_start);
    state.has_emitted_in_chunk = !state.decoded_tokens.empty();
    LITERT_RETURN_IF_ERROR(ResetToSos(decode_input_buffers, stateful_runner,
                                      num_token_ids, blank_token_id, state));
  } else {
    // Intentionally keep the prediction network state (`CommitState()` and
    // `token_ids`) from the rolled-back burst and suppress `dur == 0` tokens on
    // the immediately following frame via `prev_frame_had_burst`.
    state.prev_frame_had_burst = true;
  }
  state.symbols_at_time_index = 0;
  state.frame_tokens_start = state.decoded_tokens.size();
  ++state.time_index;
  return true;
}

// Appends a non-blank token to `state.decoded_tokens` (unless suppressed by
// `first_reported_time_index` or a zero-duration step immediately following a
// frame burst) and advances the prediction network token slot, switching from
// stateless `decode` to stateful `decode_1` once the stateless token buffer is
// full. Returns false when stateless slots are exhausted and no stateful runner
// is available.
bool RecordNonBlankTokenAndAdvanceSlot(
    int token_id, int duration,
    std::optional<TensorBuffer>& stateful_decode_token_ids_buffer,
    DecodeState& state) {
  state.has_emitted_in_chunk = true;
  state.last_emitted_time_index = state.time_index;
  if (!(state.prev_frame_had_burst && duration == 0) &&
      state.time_index >= state.first_reported_time_index) {
    state.decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
        .token_id = token_id,
        .timestamp_ms = static_cast<int>(state.time_index -
                                         state.first_reported_time_index)});
  }
  if (duration > 0) {
    state.prev_frame_had_burst = false;
  }
  if (state.num_inference_token_ids > 1) {
    ++state.token_index;
    if (state.token_index <
        static_cast<size_t>(state.num_inference_token_ids)) {
      // Still room in the stateless token array: keep filling it, so the
      // LSTM states handed to decode_1 below are computed from real tokens
      // only (an unfilled zero slot skews them).
    } else if (stateful_decode_token_ids_buffer.has_value()) {
      // Switch to stateful decoding.
      state.current_token_ids_buffer = &(*stateful_decode_token_ids_buffer);
      state.num_inference_token_ids = 1;
      state.token_ids.resize(1, 0);
      state.token_index = 0;
    } else {
      return false;
    }
  }
  if (state.token_index < state.token_ids.size()) {
    state.token_ids[state.token_index] = token_id;
  }
  return true;
}

// Performs a once-per-chunk `<sos>` reset when the decoder stalls in a
// mid-chunk blank dead zone (`>= kDeadZoneMinBlankFrames` frames without
// emission after an early token at `<= kDeadZoneMaxLastEmittedFrame` in a full
// 5-second window), rewinding `time_index` to
// `last_emitted_time_index + kDeadZoneRewindOffsetFrames`.
//
// When `last_emitted_time_index <= kDeadZoneClearPrefixMaxFrame` (<= 640 ms),
// the early emission followed by a >= 1.28 s dead zone is treated as an onset
// false start (or speech already covered by the overlapping previous chunk) and
// cleared from `decoded_tokens`. `has_emitted_in_chunk` and
// `last_emitted_time_index` are intentionally left unchanged because
// `did_mid_chunk_sos_reset` prevents a second reset and
// `last_emitted_time_index` anchors the `time_index` rewind.
absl::Status MaybeResetAfterMidChunkDeadZone(
    std::vector<TensorBuffer>& decode_input_buffers,
    StatefulLiteRtRunner* stateful_runner, size_t num_token_ids,
    int blank_token_id, DecodeState& state) {
  if (state.did_mid_chunk_sos_reset || !state.has_emitted_in_chunk ||
      !state.is_full_5s_window ||
      state.last_emitted_time_index > kDeadZoneMaxLastEmittedFrame ||
      state.time_index < kDeadZoneMinCurrentFrame ||
      state.time_index <
          state.last_emitted_time_index + kDeadZoneMinBlankFrames) {
    return absl::OkStatus();
  }
  state.did_mid_chunk_sos_reset = true;
  state.reset_origin_time_index = state.last_emitted_time_index;
  // Discard isolated onset false starts (`<= 8` frames in, or `<= 2` forced
  // `<sos>` tokens, followed by `>= 16` blank frames); in streaming mode any
  // real speech in that prefix is covered by the overlapping previous chunk.
  // `tokens_since_sos` (rather than `decoded_tokens.size()`) also counts tokens
  // that were not reported, e.g. before `first_reported_time_index` on a
  // shifted-back final chunk. `has_emitted_in_chunk` and
  // `last_emitted_time_index` are intentionally preserved to anchor the rewind.
  if (state.last_emitted_time_index <= kDeadZoneClearPrefixMaxFrame ||
      (state.sos_was_forced &&
       state.tokens_since_sos <= kMaxPrematureSosTokens)) {
    state.decoded_tokens.clear();
    state.frame_tokens_start = 0;
  }
  LITERT_RETURN_IF_ERROR(ResetToSos(decode_input_buffers, stateful_runner,
                                    num_token_ids, blank_token_id, state));
  state.time_index =
      state.last_emitted_time_index + kDeadZoneRewindOffsetFrames;
  return absl::OkStatus();
}

// Trims short (`<= kAbortiveResetMaxTokens` tokens or spanning
// `<= kAbortiveResetMaxFrameSpan` frames) or late (`>
// kAbortiveResetMaxStartDelayFrames` frames after `reset_origin_time_index`)
// post-reset token fragments when a mid-chunk `<sos>` reset kept earlier prefix
// tokens (`sos_tokens_start > 0`). In 5-second overlapping streaming windows,
// genuine speech starting after a late pause near the end of the chunk is
// decoded with full right context by the next overlapping chunk, whereas short
// or late post-reset tails in the current chunk are typically truncated
// boundary artifacts.
void TrimAbortiveMidChunkSosReset(DecodeState& state) {
  if (!state.did_mid_chunk_sos_reset || state.sos_tokens_start == 0 ||
      state.decoded_tokens.size() <= state.sos_tokens_start) {
    return;
  }
  const size_t post_reset_count =
      state.decoded_tokens.size() - state.sos_tokens_start;
  const int first_ts =
      state.decoded_tokens[state.sos_tokens_start].timestamp_ms.value_or(0);
  const int last_ts = state.decoded_tokens.back().timestamp_ms.value_or(0);
  const size_t first_frame =
      static_cast<size_t>(first_ts) + state.first_reported_time_index;
  if (post_reset_count <= kAbortiveResetMaxTokens ||
      last_ts - first_ts <= kAbortiveResetMaxFrameSpan ||
      first_frame >
          state.reset_origin_time_index + kAbortiveResetMaxStartDelayFrames) {
    state.decoded_tokens.resize(state.sos_tokens_start);
  }
}

bool IsDeviceMemory(const TensorBuffer& buf) {
  auto buffer_type = buf.BufferType();
  return buffer_type.HasValue() &&
         *buffer_type != TensorBufferType::kHostMemory;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TdtDecoder>> TdtDecoder::Create(
    LiteRtRunner* absl_nonnull runner, std::vector<int> decode_start_token_ids,
    int decode_statefully_after, int max_symbols_per_step) {
  if (decode_start_token_ids.size() != 1 || decode_start_token_ids[0] < 0) {
    return absl::InvalidArgumentError(
        "TdtDecoder requires a single non-negative decode_start_token_id.");
  }
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
  if (num_logits_per_token == 0) {
    return absl::InvalidArgumentError("num_logits_per_token must be positive.");
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
      num_logits_per_token, decode_start_token_ids[0], decode_statefully_after,
      max_symbols_per_step));
}

TdtDecoder::TdtDecoder(
    LiteRtRunner* absl_nonnull runner,
    std::unique_ptr<StatefulLiteRtRunner> absl_nullable stateful_runner,
    std::vector<TensorBuffer> decode_input_buffers,
    std::vector<TensorBuffer> decode_output_buffers,
    std::optional<TensorBuffer> stateful_decode_token_ids_buffer,
    size_t max_time_index, size_t num_token_ids, size_t num_logits_per_token,
    int blank_token_id, int decode_statefully_after, int max_symbols_per_step)
    : runner_(*runner),
      stateful_runner_(std::move(stateful_runner)),
      decode_input_buffers_(std::move(decode_input_buffers)),
      decode_output_buffers_(std::move(decode_output_buffers)),
      stateful_decode_token_ids_buffer_(
          std::move(stateful_decode_token_ids_buffer)),
      max_time_index_(max_time_index),
      num_token_ids_(num_token_ids),
      num_logits_per_token_(num_logits_per_token),
      blank_token_id_(blank_token_id),
      decode_statefully_after_(decode_statefully_after),
      max_symbols_per_step_(max_symbols_per_step) {}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> TdtDecoder::Decode(
    std::vector<TensorBuffer>& encoder_outputs,
    std::optional<DecodeRange> decode_range) {
  DecodeState state;
  InitDecodeBounds(max_time_index_, decode_range, state);
  // Start inference with decode signature for better quality in sequence start
  // as stateless decoding can keep context better than decode_1, stateful
  // decoding.
  LITERT_RETURN_IF_ERROR(ResetToSos(decode_input_buffers_,
                                    stateful_runner_.get(), num_token_ids_,
                                    blank_token_id_, state));

  // If the encoder is on GPU and the decoder is on CPU, copy the encoder
  // output data into the host decode input buffers once before the decode
  // loop; otherwise, pass the encoder output buffers as input directly.
  absl::Span<const TensorBuffer> active_encoder_outputs = encoder_outputs;
  if (!encoder_outputs.empty() &&
      encoder_outputs.size() < decode_input_buffers_.size() &&
      IsDeviceMemory(encoder_outputs[0]) &&
      !IsDeviceMemory(decode_input_buffers_[0])) {
    for (size_t j = 0; j < encoder_outputs.size(); ++j) {
      LITERT_ASSIGN_OR_RETURN(auto num_bytes, encoder_outputs[j].PackedSize());
      LITERT_ASSIGN_OR_RETURN(
          auto src_lock,
          TensorBufferScopedLock::Create<const uint8_t>(
              encoder_outputs[j], TensorBuffer::LockMode::kRead));
      LITERT_RETURN_IF_ERROR(decode_input_buffers_[j].Write<uint8_t>(
          absl::MakeConstSpan(src_lock.second, num_bytes)));
    }
    active_encoder_outputs = absl::MakeConstSpan(decode_input_buffers_)
                                 .first(encoder_outputs.size());
  }

  const float* current_logits_ptr = nullptr;
  std::optional<TensorBufferScopedLock> logits_lock;
  while (state.time_index < state.effective_max_time_index) {
    if (state.current_logits_buffer == nullptr) {
      logits_lock.reset();
      current_logits_ptr = nullptr;
      LITERT_ASSIGN_OR_RETURN(
          state.current_logits_buffer,
          RunDecodeStep(runner_, stateful_runner_.get(), active_encoder_outputs,
                        decode_input_buffers_, decode_output_buffers_, state,
                        state.was_stateful_step));
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_ptr,
          TensorBufferScopedLock::Create<const float>(
              *state.current_logits_buffer, TensorBuffer::LockMode::kRead));
      logits_lock.emplace(std::move(lock_and_ptr.first));
      current_logits_ptr = lock_and_ptr.second;
    }

    auto [token_id, duration] = SelectGreedyTokenAndDuration(
        current_logits_ptr, state, num_logits_per_token_, blank_token_id_);

    if (token_id != blank_token_id_) {
      state.current_logits_buffer = nullptr;
      // When a single forced near-threshold `<sos>` token is immediately
      // followed by a single frame that dumps `max_symbols_per_step_` tokens
      // across the remaining stateless slots, treat that single-frame dump as a
      // zero-duration burst before transitioning to stateful decoding.
      const int effective_max_symbols_per_step =
          (state.sos_was_forced && state.tokens_since_sos == 1 &&
           state.num_inference_token_ids > 1 && max_symbols_per_step_ > 1)
              ? max_symbols_per_step_ - 1
              : max_symbols_per_step_;
      if (++state.symbols_at_time_index > effective_max_symbols_per_step) {
        LITERT_ASSIGN_OR_RETURN(
            bool keep_decoding,
            RollbackFrameBurstAndAdvance(decode_input_buffers_,
                                         stateful_runner_.get(), num_token_ids_,
                                         blank_token_id_, state));
        if (!keep_decoding) {
          break;
        }
        continue;
      }
      if (!RecordNonBlankTokenAndAdvanceSlot(
              token_id, duration, stateful_decode_token_ids_buffer_, state)) {
        break;
      }
    } else {
      if (duration == 0) {
        duration = 1;
      }
      state.prev_frame_had_burst = false;
    }

    if (duration > 0) {
      if (state.symbols_at_time_index > 0) {
        state.tokens_since_sos += state.symbols_at_time_index;
        ++state.sos_emission_frames;
      }
      state.symbols_at_time_index = 0;
      state.frame_tokens_start = state.decoded_tokens.size();
    }
    state.time_index += duration;

    if (state.was_stateful_step && token_id != blank_token_id_ &&
        stateful_runner_) {
      // Stateful RNN decoder: adopt the new LSTM states by swapping the input
      // and output state buffers — but only on a non-blank emission. In RNN-T
      // greedy decoding the prediction network advances only when a token is
      // emitted; adopting the state on blank steps re-consumes the last token
      // once per blank and degrades the transcript.
      LITERT_RETURN_IF_ERROR(stateful_runner_->CommitState());
    }

    LITERT_RETURN_IF_ERROR(MaybeResetAfterMidChunkDeadZone(
        decode_input_buffers_, stateful_runner_.get(), num_token_ids_,
        blank_token_id_, state));
  }

  TrimAbortiveMidChunkSosReset(state);
  state.decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
      .token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
      .timestamp_ms =
          static_cast<int>(max_time_index_ - state.first_reported_time_index)});
  return std::move(state.decoded_tokens);
}

}  // namespace litert::omni::asr
