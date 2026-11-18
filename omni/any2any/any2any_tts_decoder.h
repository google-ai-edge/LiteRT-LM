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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TTS_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TTS_DECODER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/any2any_sampler.h"
#include "omni/base/litert_runner.h"

namespace litert::omni::any2any {

// Shared conditional TTS projector and KV-cached autoregressive speech-codec
// decoder helper for Any2Any models (e.g., MiniCPM-o 4.5 and Qwen2.5-Omni).
//
// Manages:
// - `tts_projector.tflite` buffer allocation, shape validation, and single-step
//   `(llm_hidden, token_id) -> cond_emb` projection.
// - `tts_decoder.tflite` buffer allocation, GQA/MHA/4D-inferred KV-cache shape
//   validation, KV-cache clearing, and O(1) KV-buffer-swapping single-step
//   decoding.
// - Shared LLM-to-TTS input validation, dynamic acoustic token budget
//   computation, repetition penalty (frequency-windowed and unique-history),
//   and NaN/Inf-safe greedy / temperature + top-k + top-p sampling.
class Any2AnyTtsDecoder {
 public:
  // Strategy used by `ApplyRepetitionPenalty` to penalize previously emitted
  // speech codec tokens.
  enum class RepetitionPenaltyMode {
    // Scales each occurrence in the last `repetition_window` tokens
    // multiplicatively (`pow(penalty, freq)`), matching CosyVoice2 / MiniCPM-o.
    kFrequencyWindow,
    // Penalizes each unique code present anywhere in `history` once, matching
    // HuggingFace `RepetitionPenaltyLogitsProcessor` / Qwen2.5-Omni.
    kUniqueHistory,
  };

  struct Options {
    int llm_hidden_size = 0;
    // Optional upper bound on `token_id` in `ProjectTokenConditioning`;
    // validated when `> 0` (`-1` placeholder is always allowed; values `< -1`
    // are always rejected).
    int llm_vocab_size = 0;
    int tts_hidden_size = 0;
    int tts_vocab_size = 0;
    int tts_num_layers = 0;
    int tts_num_kv_heads = 0;
    int tts_head_dim = 0;
    int tts_max_cache_len = 0;
    bool generate_audio = false;
    // When true, if the decoder KV-cache buffer size does not match the
    // configured GQA/MHA byte count, attempt to infer `tts_num_layers` and
    // `tts_max_cache_len` from a 4D `[1, 2*L, max_cache_len, D]` tensor type.
    bool allow_inferred_kv_shape = false;
    bool do_sample = false;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float repetition_penalty = 1.0f;
    int repetition_window = 0;
    RepetitionPenaltyMode repetition_penalty_mode =
        RepetitionPenaltyMode::kFrequencyWindow;
    std::optional<uint64_t> seed = 42;
  };

  // Creates an `Any2AnyTtsDecoder`, taking ownership of `tts_projector_runner`
  // and `tts_decoder_runner`.
  //
  // When `options.generate_audio` is `true`, both runners must be non-null;
  // otherwise returns `NotFoundError`. When `options.generate_audio` is
  // `false`, null runners are permitted (stepping APIs then return
  // `FailedPreconditionError`).
  //
  // Allocates and validates projector/decoder input and output `TensorBuffer`
  // counts, byte sizes, and matching `Float32` KV-cache tensor types (inferring
  // `tts_num_layers` and `tts_max_cache_len` from a 4D `[1, 2*L, cache_len, D]`
  // KV tensor when `options.allow_inferred_kv_shape` is `true`).
  //
  // Returns `InvalidArgumentError` if required dimensions are non-positive,
  // KV-cache byte-size calculations overflow, or buffer counts/sizes/types do
  // not match `options`.
  static absl::StatusOr<std::unique_ptr<Any2AnyTtsDecoder>> Create(
      Options options,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner =
          nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner = nullptr);

  // Computes the turn-level acoustic token cap, scaling up the default budget
  // (`>= 1024`) linearly for long text responses.
  static int ComputeMaxTotalCodes(int num_tokens, int max_tts_tokens);

  // Validates that LLM hidden states, generated token IDs, and optional prompt
  // token IDs / prompt hidden states have mutually consistent shapes.
  static absl::Status ValidateLlmAudioInputs(
      int num_tokens, int llm_hidden_size,
      absl::Span<const int> generated_token_ids,
      absl::Span<const float> llm_hidden_states,
      absl::Span<const int> prompt_token_ids = {},
      absl::Span<const float> prompt_hidden_states = {});

  // Applies repetition penalty to `logits` in place according to `mode`.
  // Non-finite or `<= 1.0f` penalties (and empty `history`) are no-ops.
  static void ApplyRepetitionPenalty(absl::Span<float> logits,
                                     absl::Span<const int> history,
                                     float penalty, RepetitionPenaltyMode mode,
                                     int window_size = 0);

  // Applies repetition penalty using this decoder's configured options,
  // reusing an internal O(1)-reset epoch array in `kUniqueHistory` mode to
  // avoid per-step heap allocations and sorting.
  void ApplyRepetitionPenalty(absl::Span<float> logits,
                              absl::Span<const int> history);

  // Selects the next speech codec token ID from `logits` using NaN/Inf-safe
  // greedy selection or temperature + top-k + top-p sampling.
  int SampleAudioCode(absl::Span<const float> logits);

  // Projects a single `(token_hidden, token_id)` pair to `out_cond` using
  // `tts_projector_runner_`.
  //
  // Requirements:
  // - `token_hidden.size()` must equal `options.llm_hidden_size`.
  // - `out_cond.size()` must equal `options.tts_hidden_size`.
  // - `token_id` must be `-1` (multimodal placeholder sentinel) or `>= 0`
  //   (and `< options.llm_vocab_size` when `options.llm_vocab_size > 0`).
  //
  // Returns `FailedPreconditionError` if `tts_projector_runner_` is null, or
  // `InvalidArgumentError` if `token_id` or span lengths are out of range.
  absl::Status ProjectTokenConditioning(absl::Span<const float> token_hidden,
                                        int32_t token_id,
                                        absl::Span<float> out_cond);

  // Zero-clears the decoder's input KV-cache buffer. Returns
  // `FailedPreconditionError` if `tts_decoder_runner_` is null.
  absl::Status ClearKvCache();

  // Runs a single autoregressive step of `tts_decoder_runner_` at `step_idx`,
  // swapping `new_kv_cache` into `decoder_input_buffers_[3]` in O(1).
  //
  // Requirements:
  // - `cond_emb.size()` must equal `options.tts_hidden_size`.
  // - `prev_audio_code` must be `-1` (conditioning-prefill sentinel) or in
  //   `[0, options.tts_vocab_size)`.
  // - `step_idx` must be in `[0, tts_max_cache_len())`.
  // - When `out_logits` is present, `out_logits->size()` must equal
  //   `options.tts_vocab_size` and receives the step's output logits; passing
  //   `std::nullopt` skips the device-to-host logits readback (for prefill or
  //   replay steps whose logits are discarded).
  //
  // Returns `FailedPreconditionError` if `tts_decoder_runner_` is null, or
  // `InvalidArgumentError` if IDs, `step_idx`, or span lengths are out of
  // range.
  absl::Status RunDecoderStep(absl::Span<const float> cond_emb,
                              int32_t prev_audio_code, int32_t step_idx,
                              std::optional<absl::Span<float>> out_logits);

  // Reseeds the internal sampling RNG from `options_.seed`.
  void Reset();

  int tts_num_layers() const { return options_.tts_num_layers; }
  int tts_max_cache_len() const { return options_.tts_max_cache_len; }

 private:
  Any2AnyTtsDecoder(
      Options options,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner);

  Options options_;
  std::unique_ptr<LiteRtRunner> tts_projector_runner_;
  std::unique_ptr<LiteRtRunner> tts_decoder_runner_;

  std::vector<TensorBuffer> proj_input_buffers_;
  std::vector<TensorBuffer> proj_output_buffers_;
  std::vector<TensorBuffer> decoder_input_buffers_;
  std::vector<TensorBuffer> decoder_output_buffers_;

  Any2AnySampler sampler_;
  std::vector<uint32_t> seen_code_epochs_;
  uint32_t seen_code_epoch_ = 0;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TTS_DECODER_H_
