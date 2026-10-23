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

#include "omni/speech2text/dummy_preprocessor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/speech2text/audio_preprocessor.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/utils.h"

namespace litert::omni::asr {

DummyPreprocessor::DummyPreprocessor(
    Stage<std::vector<float>>* absl_nonnull audio_source,
    int pad_to_num_samples)
    : AudioPreprocessor(audio_source),
      pad_to_num_samples_(pad_to_num_samples) {}

DummyPreprocessor::DummyPreprocessor(AudioSource* absl_nonnull audio_source,
                                     int pad_to_num_samples)
    : AudioPreprocessor(audio_source),
      pad_to_num_samples_(pad_to_num_samples) {}

void DummyPreprocessor::ResetInternal() { previous_window_.clear(); }

absl::Status DummyPreprocessor::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto raw_speech, audio_source_.GetOutput());
  const bool is_warm_start = IsLastInputWarmStart();
  // Models that consume raw PCM waveforms directly (such as moonshine-tiny)
  // apply internal normalization inside the encoder subgraph. Feeding digital
  // silence or near-silent background noise causes that internal
  // normalization to amplify near-zero inputs and make the autoregressive
  // decoder hallucinate text (e.g. "Thank you."). Emit an empty feature
  // vector on silent chunks so LiteRtSpeechRecognizer skips encoder/decoder
  // execution.
  if (IsSilentAudio(raw_speech)) {
    previous_window_.clear();
    PushOutput(PreprocessedAudio{.is_warm_start = is_warm_start});
    return absl::OkStatus();
  }

  const size_t window_samples =
      pad_to_num_samples_ > 0 ? static_cast<size_t>(pad_to_num_samples_) : 0;
  std::optional<DecodeRange> decode_range;
  if (window_samples > 0) {
    if (raw_speech.size() >= window_samples) {
      previous_window_.assign(raw_speech.end() - window_samples,
                              raw_speech.end());
      decode_range =
          DecodeRange{.start_fraction = 0.0f, .valid_fraction = 1.0f};
    } else if (previous_window_.size() == window_samples) {
      // Shift a short follow-up chunk back over `previous_window_` so raw-PCM
      // encoders without a length mask (such as moonshine-tiny) receive a full
      // window of real speech rather than a mostly dither-padded window whose
      // utterance-level normalization distorts the active speech segment.
      const float start_fraction = ShiftShortChunkBackOverPreviousWindow(
          raw_speech, previous_window_, window_samples);
      decode_range =
          DecodeRange{.start_fraction = start_fraction, .valid_fraction = 1.0f};
    } else {
      // Pad standalone short chunks with low-level dither (~-54 dBFS) so
      // unmasked raw-waveform encoders do not produce identical constant keys
      // across padded frames.
      const size_t dither_start = raw_speech.size();
      const float valid_fraction =
          static_cast<float>(dither_start) / static_cast<float>(window_samples);
      raw_speech.resize(window_samples, 0.0f);
      uint32_t rng = 0x9E3779B9u;
      for (size_t i = dither_start; i < window_samples; ++i) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        raw_speech[i] = static_cast<float>(static_cast<int32_t>(rng)) *
                        (3.5e-3f / 2147483648.0f);
      }
      decode_range =
          DecodeRange{.start_fraction = 0.0f, .valid_fraction = valid_fraction};
    }
  }
  PushOutput(PreprocessedAudio{
      .features = std::move(raw_speech),
      .decode_range = decode_range,
      .is_warm_start = is_warm_start,
  });
  return absl::OkStatus();
}

}  // namespace litert::omni::asr
