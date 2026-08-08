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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_LATENT_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_LATENT_DECODER_H_

#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"

namespace litert::omni::tts {

// TODO: b/538727793. Refactor the outputs to a more generic structure.
// Currently these are tailored for qwen3-tts model. Generic frontend output
// Generic latent decoding output payload for continuous latent feature decoder
// or flow solver.
struct LatentOutput {
  std::vector<float> codec_features;
  std::vector<std::vector<int>> rvq_frames;
};

// Abstract interface for continuous latent feature decoder or flow solver.
class LatentDecoder : public SingleThreadedStageWithDeque<LatentOutput> {
 public:
  using LatentOutput = ::litert::omni::tts::LatentOutput;

  explicit LatentDecoder(
      Stage<AcousticPredictor::AcousticOutput>* absl_nonnull acoustic_predictor)
      : acoustic_predictor_(*acoustic_predictor) {}

  ~LatentDecoder() override = default;

  // Resets internal state for a new TTS stream.
  virtual void Reset() = 0;

 protected:
  bool NeedScheduleInternal() const override {
    return acoustic_predictor_.HasOutput();
  }

  Stage<AcousticPredictor::AcousticOutput>& acoustic_predictor_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_LATENT_DECODER_H_
