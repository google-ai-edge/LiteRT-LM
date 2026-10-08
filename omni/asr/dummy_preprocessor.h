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
  explicit DummyPreprocessor(
      Stage<std::vector<float>>* absl_nonnull audio_source)
      : AudioPreprocessor(audio_source) {}

  ~DummyPreprocessor() override = default;

 protected:
  absl::Status ScheduleInternal() override {
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
      PushOutput(std::vector<float>());
      return absl::OkStatus();
    }
    PushOutput(std::move(raw_speech));
    return absl::OkStatus();
  }
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DUMMY_PREPROCESSOR_H_
