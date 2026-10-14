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

#include "omni/asr/dummy_preprocessor.h"

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

DummyPreprocessor::DummyPreprocessor(
    Stage<std::vector<float>>* absl_nonnull audio_source,
    int pad_to_num_samples)
    : AudioPreprocessor(audio_source),
      pad_to_num_samples_(pad_to_num_samples) {}

absl::Status DummyPreprocessor::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto raw_speech, audio_source_.GetOutput());
  // Models that consume raw PCM waveforms directly (such as moonshine-tiny)
  // apply internal normalization inside the encoder subgraph. Feeding digital
  // silence or near-silent background noise causes that internal
  // normalization to amplify near-zero inputs and make the autoregressive
  // decoder hallucinate text (e.g. "Thank you."). Emit an empty feature
  // vector on silent chunks so LiteRtSpeechRecognizer skips encoder/decoder
  // execution.
  if (IsSilentAudio(raw_speech)) {
    PushOutput(PreprocessedAudio{});
    return absl::OkStatus();
  }

  const size_t window_samples =
      pad_to_num_samples_ > 0 ? static_cast<size_t>(pad_to_num_samples_) : 0;
  if (window_samples > 0 && raw_speech.size() < window_samples) {
    // Pad with low-level dither (~-54 dBFS) so unmasked raw-waveform encoders
    // do not produce identical constant keys across padded frames.
    const size_t dither_start = raw_speech.size();
    raw_speech.resize(window_samples, 0.0f);
    uint32_t rng = 0x9E3779B9u;
    for (size_t i = dither_start; i < window_samples; ++i) {
      rng ^= rng << 13;
      rng ^= rng >> 17;
      rng ^= rng << 5;
      raw_speech[i] = static_cast<float>(static_cast<int32_t>(rng)) *
                      (3.5e-3f / 2147483648.0f);
    }
  }
  PushOutput(PreprocessedAudio{.features = std::move(raw_speech)});
  return absl::OkStatus();
}

}  // namespace litert::omni::asr
