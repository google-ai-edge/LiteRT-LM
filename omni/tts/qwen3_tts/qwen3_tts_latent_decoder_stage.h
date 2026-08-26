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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_LATENT_DECODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_LATENT_DECODER_STAGE_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/qwen3_tts_io_types.h"

namespace litert::omni::tts {

// Stage 3: Latent Decoder (Discrete RVQ Codebook Index to Feature
// Transformation).
class Qwen3TtsLatentDecoderStage
    : public SingleThreadedStageWithDeque<Qwen3TtsLatentOutput> {
 public:
  // Creates a Qwen3TtsLatentDecoderStage instance.
  //
  // args
  // - acoustic_predictor: Stage providing input Qwen3TtsAcousticOutput data.
  //
  // returns
  // - Unique pointer to created Qwen3TtsLatentDecoderStage on success, or error
  //   status on failure.
  static absl::StatusOr<std::unique_ptr<Qwen3TtsLatentDecoderStage>> Create(
      Stage<Qwen3TtsAcousticOutput>* absl_nonnull acoustic_predictor);

  explicit Qwen3TtsLatentDecoderStage(
      Stage<Qwen3TtsAcousticOutput>* absl_nonnull acoustic_predictor);
  ~Qwen3TtsLatentDecoderStage() override = default;

  // Resets the stage state back to idle and clears any pending outputs.
  void Reset() override;

 protected:
  bool NeedScheduleInternal() const override {
    return acoustic_predictor_.HasOutput();
  }

  // Executes one step of latent decoder stage processing asynchronously.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status ScheduleInternal() override;

 private:
  Stage<Qwen3TtsAcousticOutput>& acoustic_predictor_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_LATENT_DECODER_STAGE_H_
