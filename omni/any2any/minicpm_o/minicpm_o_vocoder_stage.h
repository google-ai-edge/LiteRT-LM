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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_VOCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_VOCODER_STAGE_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_vocoder_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Stage 4: Neural Speech-Token-to-Waveform Vocoder & Output Stage for
// MiniCPM-o 4.5.
// Wraps `Any2AnyVocoderStage` with constant silence-token padding
// (`kTtsSilenceTokenId`), strict trimmed buffer validation, and a 48-sample
// seam de-clicking crossfade over 25 Hz CosyVoice2 speech tokens to synthesize
// 24 kHz `AudioOutput` PCM waveforms alongside `TextOutput` and
// `EndOfOutputTurn`.
class MiniCpmOVocoderStage : public SingleThreadedStageWithDeque<Output> {
 public:
  static absl::StatusOr<std::unique_ptr<MiniCpmOVocoderStage>> Create(
      Stage<Any2AnyTtsOutput>* absl_nonnull tts_stage,
      const MiniCpmOModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner = nullptr);

  ~MiniCpmOVocoderStage() override = default;

 protected:
  absl::Status FlushInternal() override;

  bool NeedScheduleInternal() const override;

  void ResetInternal() override;

  absl::Status ScheduleInternal() override;

 private:
  explicit MiniCpmOVocoderStage(
      std::unique_ptr<Any2AnyVocoderStage> absl_nonnull vocoder_stage);

  std::unique_ptr<Any2AnyVocoderStage> vocoder_stage_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_VOCODER_STAGE_H_
