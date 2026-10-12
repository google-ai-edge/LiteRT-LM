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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DUMMY_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DUMMY_PREPROCESSOR_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/utils.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {

// Dummy audio preprocessor that passes raw PCM speech audio directly as
// features.
// TODO: b/524681030 - Consider a way to pass data without copying.
class DummyPreprocessor : public AudioPreprocessor {
 public:
  // `pad_to_num_samples` is the model's fixed PCM input length. The audio
  // source emits the final chunk of a stream unpadded, so chunks shorter than
  // this are shifted back over the previous window (multi-chunk streams) or
  // padded with low-level dither (single-chunk streams); 0 disables padding.
  explicit DummyPreprocessor(
      Stage<std::vector<float>>* absl_nonnull audio_source,
      int pad_to_num_samples = 0)
      : AudioPreprocessor(audio_source),
        pad_to_num_samples_(pad_to_num_samples) {}

  ~DummyPreprocessor() override = default;

  float last_chunk_start_fraction() const override {
    return last_start_fraction_;
  }
  float last_chunk_valid_fraction() const override {
    return last_valid_fraction_;
  }

 protected:
  void ResetInternal() override {
    previous_window_.clear();
    last_start_fraction_ = 0.0f;
    last_valid_fraction_ = 1.0f;
    peak_speech_rms_ = 0.0f;
  }

  absl::Status ScheduleInternal() override {
    auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
    ABSL_ASSIGN_OR_RETURN(auto raw_speech, audio_source_.GetOutput());
    const size_t window_samples =
        pad_to_num_samples_ > 0 ? static_cast<size_t>(pad_to_num_samples_) : 0;
    const bool has_previous_window =
        window_samples > 0 && previous_window_.size() == window_samples;
    size_t overlap = 0;
    if (has_previous_window) {
      const size_t max_candidate = std::min(raw_speech.size(), window_samples);
      constexpr size_t kMinOverlap = 160;
      for (size_t candidate = max_candidate; candidate >= kMinOverlap;
           --candidate) {
        if (std::equal(previous_window_.end() - candidate,
                       previous_window_.end(), raw_speech.begin())) {
          overlap = candidate;
          break;
        }
      }
      if (overlap == raw_speech.size()) {
        PushOutput(std::vector<float>());
        return absl::OkStatus();
      }
    }
    const std::vector<float> prev_window_copy =
        (has_previous_window && raw_speech.size() < window_samples)
            ? previous_window_
            : std::vector<float>();
    if (window_samples > 0 && raw_speech.size() >= window_samples) {
      previous_window_.assign(raw_speech.end() - window_samples,
                              raw_speech.end());
    }
    // Models that consume raw PCM waveforms directly (such as moonshine-tiny)
    // apply internal normalization inside the encoder subgraph. Feeding digital
    // silence or near-silent background noise causes that internal
    // normalization to amplify near-zero inputs and make the autoregressive
    // decoder hallucinate text (e.g. "Thank you."). Emit an empty feature
    // vector on silent chunks so LiteRtSpeechRecognizer skips encoder/decoder
    // execution.
    if (IsSilentAudio(raw_speech)) {
      PushOutput(std::vector<float>());
      return absl::OkStatus();
    }

    constexpr size_t kBlockSamples = 1600;  // 100 ms at 16 kHz.
    constexpr float kTrailingSilenceMaxRms = 6e-3f;
    constexpr float kTrailingSilenceRelativeRms = 0.05f;
    const size_t num_blocks = raw_speech.size() / kBlockSamples;
    if (num_blocks > 0) {
      float max_block_rms = 0.0f;
      float max_new_block_rms = 0.0f;
      const size_t check_start_block =
          overlap >= 2 * kBlockSamples ? (overlap / kBlockSamples - 2) : 0;
      for (size_t b = 0; b < num_blocks; ++b) {
        double sum_sq = 0.0;
        const size_t offset = b * kBlockSamples;
        for (size_t i = 0; i < kBlockSamples; ++i) {
          const float s = raw_speech[offset + i];
          sum_sq += static_cast<double>(s) * s;
        }
        const float rms = static_cast<float>(std::sqrt(sum_sq / kBlockSamples));
        max_block_rms = std::max(max_block_rms, rms);
        if (b >= check_start_block) {
          max_new_block_rms = std::max(max_new_block_rms, rms);
        }
      }
      if (peak_speech_rms_ > 0.0f &&
          max_new_block_rms <=
              std::min(kTrailingSilenceMaxRms,
                       kTrailingSilenceRelativeRms * peak_speech_rms_)) {
        PushOutput(std::vector<float>());
        return absl::OkStatus();
      }
      peak_speech_rms_ = std::max(peak_speech_rms_, max_block_rms);
    }

    last_start_fraction_ = 0.0f;
    last_valid_fraction_ = 1.0f;
    if (window_samples > 0 && raw_speech.size() < window_samples) {
      if (!prev_window_copy.empty()) {
        const size_t missing = window_samples - raw_speech.size();
        std::vector<float> window;
        window.reserve(window_samples);
        window.insert(window.end(),
                      prev_window_copy.end() - overlap - missing,
                      prev_window_copy.end() - overlap);
        window.insert(window.end(), raw_speech.begin(), raw_speech.end());
        raw_speech = std::move(window);
        previous_window_ = raw_speech;
        last_start_fraction_ =
            static_cast<float>(missing) / static_cast<float>(window_samples);
      } else {
        last_valid_fraction_ = static_cast<float>(raw_speech.size()) /
                               static_cast<float>(window_samples);
        const size_t dither_start = raw_speech.size();
        raw_speech.resize(window_samples, 0.0f);
        // Exported raw-waveform encoders without an attention mask (e.g.
        // moonshine-tiny) apply Conv1d(bias=False) -> Tanh -> GroupNorm ->
        // Conv1d(bias=True) across the full window; exact zero padding produces
        // identical constant key/value vectors across all padded frames that
        // coherently hijack unmasked cross-attention. Low-level deterministic
        // zero-mean dither (~-54 dBFS) keeps padded frame keys incoherent.
        uint32_t rng = 0x9E3779B9u;
        for (size_t i = dither_start; i < window_samples; ++i) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          const float u = static_cast<float>(static_cast<int32_t>(rng)) *
                          (1.0f / 2147483648.0f);
          raw_speech[i] = u * 3.5e-3f;
        }
      }
    }
    PushOutput(std::move(raw_speech));
    return absl::OkStatus();
  }

 private:
  const int pad_to_num_samples_;
  std::vector<float> previous_window_;
  float last_start_fraction_ = 0.0f;
  float last_valid_fraction_ = 1.0f;
  float peak_speech_rms_ = 0.0f;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DUMMY_PREPROCESSOR_H_
