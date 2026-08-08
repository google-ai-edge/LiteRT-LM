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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_LATENT_DECODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_LATENT_DECODER_STAGE_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/latent_decoder.h"

namespace litert::omni::tts {

// Stage 3: Latent Decoder (Discrete RVQ Codebook Index to Feature
// Transformation). A dummy stage for now.
class Qwen3LatentDecoderStage : public LatentDecoder {
 public:
  // Creates a Qwen3LatentDecoderStage instance.
  //
  // args
  // - acoustic_predictor: Stage providing input AcousticOutput data.
  //
  // returns
  // - Unique pointer to created Qwen3LatentDecoderStage on success, or error
  //   status on failure.
  static absl::StatusOr<std::unique_ptr<Qwen3LatentDecoderStage>> Create(
      Stage<AcousticOutput>* absl_nonnull acoustic_predictor);

  explicit Qwen3LatentDecoderStage(
      Stage<AcousticOutput>* absl_nonnull acoustic_predictor);
  ~Qwen3LatentDecoderStage() override = default;

  // Resets the stage state back to idle and clears any pending outputs.
  void Reset() override;

 protected:
  // Executes one step of latent decoder stage processing asynchronously.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status ScheduleInternal() override;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_LATENT_DECODER_STAGE_H_
