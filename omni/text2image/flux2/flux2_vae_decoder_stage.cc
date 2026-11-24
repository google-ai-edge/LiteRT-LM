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

#include "omni/text2image/flux2/flux2_vae_decoder_stage.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/io_types.h"
#include "omni/text2image/vae_decoder_stage.h"

namespace litert::omni::text2image {

absl::StatusOr<std::unique_ptr<Flux2VaeDecoderStage>>
Flux2VaeDecoderStage::Create(
    Stage<DenoisedLatents>* absl_nonnull denoiser,
    const Flux2ModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner) {
  ABSL_RETURN_IF_ERROR(ValidateFlux2ModelConfig(config));

  const int grid_dim = config.img_size / 16;
  const size_t expected_in_elements =
      static_cast<size_t>(32) * (2 * grid_dim) * (2 * grid_dim);
  VaeDecoderStage::Config vae_config{
      .image_width = config.img_size,
      .image_height = config.img_size,
      .expected_latent_elements = expected_in_elements,
      .stage_name = "Flux2VaeDecoderStage",
  };
  UnpackLatentsFn unpack_latents =
      [grid_dim, bn_scale = config.latent_bn_scale,
       bn_shift = config.latent_bn_shift](const DenoisedLatents& denoised)
      -> absl::StatusOr<std::vector<float>> {
    return UnpatchifyLatents(denoised.packed_latents, grid_dim, grid_dim,
                             bn_scale, bn_shift);
  };

  ABSL_ASSIGN_OR_RETURN(RunnerBuffers buffers,
                        ValidateAndCreateBuffers(denoiser, vae_config,
                                                 runner.get(), unpack_latents));

  return absl::WrapUnique(new Flux2VaeDecoderStage(
      denoiser, std::move(vae_config), std::move(runner),
      std::move(unpack_latents), std::move(buffers.inputs),
      std::move(buffers.outputs)));
}

Flux2VaeDecoderStage::Flux2VaeDecoderStage(
    Stage<DenoisedLatents>* absl_nonnull denoiser, Config vae_config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    UnpackLatentsFn unpack_latents, std::vector<TensorBuffer> input_buffers,
    std::vector<TensorBuffer> output_buffers)
    : VaeDecoderStage(denoiser, std::move(vae_config), std::move(runner),
                      std::move(unpack_latents), std::move(input_buffers),
                      std::move(output_buffers)) {}

}  // namespace litert::omni::text2image
