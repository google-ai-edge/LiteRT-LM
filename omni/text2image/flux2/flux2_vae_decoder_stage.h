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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_VAE_DECODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_VAE_DECODER_STAGE_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_denoiser_stage.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/image_decoder.h"

namespace litert::omni::text2image {

// Stage 3: Latent BatchNorm, 2x2 Unpatchify, and AutoencoderKLFlux2 VAE Decoder
// for FLUX.2 models (Bonsai-FLUX.2 and FLUX.2-klein-4B).
class Flux2VaeDecoderStage : public ImageDecoder {
 public:
  static absl::StatusOr<std::unique_ptr<Flux2VaeDecoderStage>> Create(
      Stage<Flux2DenoiserOutput>* absl_nonnull denoiser,
      const Flux2ModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner);

  ~Flux2VaeDecoderStage() override = default;

  absl::Status Flush() override { return absl::OkStatus(); }

 protected:
  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  Flux2VaeDecoderStage(Stage<Flux2DenoiserOutput>* absl_nonnull denoiser,
                       Flux2ModelConfig config,
                       std::unique_ptr<LiteRtRunner> absl_nonnull runner,
                       std::vector<TensorBuffer> input_buffers,
                       std::vector<TensorBuffer> output_buffers);

  Stage<Flux2DenoiserOutput>& denoiser_;
  const Flux2ModelConfig config_;
  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_VAE_DECODER_STAGE_H_
