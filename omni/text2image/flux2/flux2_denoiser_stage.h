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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_DENOISER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_DENOISER_STAGE_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/text_encoder_stage.h"

namespace litert::omni::text2image {

// Output of Stage 2 (Flux2DenoiserStage): denoised packed latents
// of shape [1, grid_h * grid_w, packed_ch] after FlowMatch Euler integration,
// preserving the input generation metadata (`ImageGenInputMetadata`).
struct Flux2DenoiserOutput {
  ImageGenInputMetadata metadata;
  std::vector<float> packed_latents;
};

struct Flux2DenoiserInputIndices {
  size_t hidden = 0;   // args_0: [1, tokens, 128]
  size_t enc = 1;      // args_1: [1, seq_len, 7680]
  size_t t = 2;        // args_2: [1]
  size_t img_ids = 3;  // args_3: [tokens, 4]
  size_t txt_ids = 4;  // args_4: [seq_len, 4]
};

// Stage 2: FlowMatch-Euler DiT Denoiser for monolithic FLUX.2 models
// (Bonsai-FLUX.2).
// Samples initial Gaussian noise latents of shape `[1, tokens, 128]` and runs
// the FlowMatch Euler integration loop:
//   `lat += (sigma[k+1] - sigma[k]) *
//           dit(lat, embeds, sigma[k], img_ids, txt_ids)`
class Flux2DenoiserStage
    : public SingleThreadedStageWithDeque<Flux2DenoiserOutput> {
 public:
  using InputIndices = Flux2DenoiserInputIndices;

  static absl::StatusOr<std::unique_ptr<Flux2DenoiserStage>> Create(
      Stage<TextEncoderOutput>* absl_nonnull text_encoder,
      const Flux2ModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      InputIndices input_indices = {});

  ~Flux2DenoiserStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  Flux2DenoiserStage(Stage<TextEncoderOutput>* absl_nonnull text_encoder,
                     Flux2ModelConfig config,
                     std::unique_ptr<LiteRtRunner> absl_nonnull runner,
                     InputIndices input_indices,
                     std::vector<TensorBuffer> input_buffers,
                     std::vector<TensorBuffer> output_buffers);

  Stage<TextEncoderOutput>& text_encoder_;
  const Flux2ModelConfig config_;
  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  const InputIndices input_indices_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_DENOISER_STAGE_H_
