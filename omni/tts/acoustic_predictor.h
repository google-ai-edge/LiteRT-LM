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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_ACOUSTIC_PREDICTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_ACOUSTIC_PREDICTOR_H_

#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/tts/text_frontend.h"

namespace litert::omni::tts {

// TODO: b/538727793. Refactor the outputs to a more generic structure.
// Currently these are tailored for qwen3-tts model. Generic frontend output
// Generic latent decoding output payload for continuous latent feature decoder
// or flow solver.
// Generic acoustic prediction output payload for acoustic predictor.
struct AcousticOutput {
  // RVQ(residual vector quantization) frames of each codebook. Shape
  // [num_frames, num_codebooks]
  std::vector<std::vector<int>> rvq_frames;
  // Shape [num_frames, hidden_dim]
  std::vector<float> codec_features;
};

// Abstract interface for acoustic feature prediction (e.g. Talker LM /
// MTP).
class AcousticPredictor : public SingleThreadedStageWithDeque<AcousticOutput> {
 public:
  using AcousticOutput = ::litert::omni::tts::AcousticOutput;

  explicit AcousticPredictor(
      Stage<TextFrontend::FrontendOutput>* absl_nonnull text_frontend)
      : text_frontend_(*text_frontend) {}

  ~AcousticPredictor() override = default;

  // Resets internal state for a new TTS stream.
  virtual void Reset() = 0;

 protected:
  bool NeedScheduleInternal() const override {
    return text_frontend_.HasOutput();
  }

  Stage<TextFrontend::FrontendOutput>& text_frontend_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_ACOUSTIC_PREDICTOR_H_
