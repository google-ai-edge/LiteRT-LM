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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_VOCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_VOCODER_STAGE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/io_types.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Shared neural speech-token-to-waveform vocoder and output stage for Any2Any
// models (e.g., MiniCPM-o 4.5 and Qwen2.5-Omni).
//
// Consumes `Any2AnyTtsOutput` from an upstream TTS stage, forwards assistant
// `TextOutput`, synthesizes `AudioOutput` PCM waveforms from sliding windows of
// `history_tokens + chunk_tokens + lookahead_tokens` discrete speech codec
// tokens when audio generation is enabled, and emits `EndOfOutputTurn` when
// `is_text_final` is true.
class Any2AnyVocoderStage : public SingleThreadedStageWithDeque<Output> {
 public:
  // Padding strategy for out-of-bounds indices in the sliding token window.
  enum class PaddingMode {
    // Pad out-of-bounds leading/trailing slots with `Options::pad_token_id`.
    kConstant,
    // Clamp leading out-of-bounds slots to `audio_codes.front()` and trailing
    // out-of-bounds slots to `audio_codes.back()`.
    kEdgeClamp,
  };

  struct Options {
    int history_tokens = 0;
    int chunk_tokens = 0;
    int lookahead_tokens = 0;
    int samples_per_token = 0;
    int audio_output_sample_rate_hz = 24000;
    bool generate_audio = false;
    PaddingMode padding_mode = PaddingMode::kConstant;
    int32_t pad_token_id = 0;
    // Maximum number of history-overlap samples to linearly crossfade at chunk
    // boundaries (`0` disables seam crossfade).
    size_t max_seam_crossfade_samples = 0;
    // When `> 0`, validates every input audio code is in `[0, max_valid_code)`.
    int max_valid_code = 0;
    // When true, accepts both trimmed (`history + chunk`) and full
    // (`history + chunk + lookahead`) output buffer sizes, and falls back to
    // inferring `samples_per_token` or a non-overlapping chunk layout from the
    // model's buffer sizes.
    bool allow_inferred_layout = false;
  };

  // Creates and initializes an `Any2AnyVocoderStage` consuming
  // `Any2AnyTtsOutput` from `tts_stage`.
  //
  // `tts_stage` must outlive the returned stage. `vocoder_runner` is required
  // when `options.generate_audio` is true, and optional (ignored without
  // initializing buffers) when `options.generate_audio` is false.
  //
  // Returns:
  //   - `InvalidArgumentError` if `options` has invalid window, sample count,
  //     or sample rate dimensions, or (when `options.generate_audio` is true)
  //     if the vocoder's input/output tensor element types (`Int32`/`Float32`)
  //     or packed byte sizes do not match `options`.
  //   - `NotFoundError` if `options.generate_audio` is true and
  //     `vocoder_runner` is null.
  //   - `InternalError` if `vocoder_runner` returns empty input or output
  //     buffers.
  static absl::StatusOr<std::unique_ptr<Any2AnyVocoderStage>> Create(
      Stage<Any2AnyTtsOutput>* absl_nonnull tts_stage, Options options,
      std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner = nullptr);

  ~Any2AnyVocoderStage() override = default;

 protected:
  absl::Status FlushInternal() override;

  bool NeedScheduleInternal() const override { return tts_stage_.HasOutput(); }

  absl::Status ScheduleInternal() override;

 private:
  Any2AnyVocoderStage(
      Stage<Any2AnyTtsOutput>* absl_nonnull tts_stage, Options options,
      std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner);

  absl::Status ProcessTtsOutput(const Any2AnyTtsOutput& tts_out);

  absl::StatusOr<std::vector<float>> SynthesizeWaveform(
      const Any2AnyTtsOutput& tts_out);

  Stage<Any2AnyTtsOutput>& tts_stage_;
  Options options_;

  std::unique_ptr<LiteRtRunner> vocoder_runner_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
  size_t output_num_samples_ = 0;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_VOCODER_STAGE_H_
