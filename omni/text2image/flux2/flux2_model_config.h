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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MODEL_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MODEL_CONFIG_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/image_gen_model_type.pb.h"

namespace litert::omni::text2image {

// Returns the default 128-element latent BatchNorm scale vector
// (`sqrt(vae.bn.running_var + vae.config.batch_norm_eps)`) from the pretrained
// `AutoencoderKLFlux2` (`black-forest-labs/FLUX.2-klein-4B` /
// `prism-ml/Bonsai-Image-ternary-4B`) VAE checkpoint.
std::vector<float> DefaultFlux2LatentBnScale();

// Returns the default 128-element latent BatchNorm shift vector
// (`vae.bn.running_mean`) from the pretrained `AutoencoderKLFlux2`
// (`black-forest-labs/FLUX.2-klein-4B` / `prism-ml/Bonsai-Image-ternary-4B`)
// VAE checkpoint.
std::vector<float> DefaultFlux2LatentBnShift();

// Model-specific configuration settings for FLUX.2 pipelines (including
// 3-graph monolithic Bonsai-FLUX.2 and sharded FLUX.2-klein-4B).
struct Flux2ModelConfig {
  // Padded text sequence length expected by textenc and dit.
  int seq_len = 256;
  // Target output image width and height in pixels (512 or 256).
  int img_size = 512;
  // Packed latent channel count (32 VAE channels * 2x2 patchify = 128).
  int packed_ch = 128;
  // Concatenated Qwen3-4B hidden dimension (3 * 2560 = 7680).
  int prompt_dim = 7680;
  // Default number of FlowMatch Euler denoising steps.
  int steps = 4;
  // Default random seed for initial Gaussian latent noise.
  uint64_t default_seed = 42;

  // Optional per-stage backend overrides. If unset, defaults to the engine's
  // backend.
  std::optional<lm::Backend> textenc_backend;
  std::optional<lm::Backend> dit_backend;
  std::optional<lm::Backend> vae_backend;

  // Per-packed-channel BatchNorm affine parameters (128 elements each) applied
  // before 2x2 unpatchify and VAE decoding.
  std::vector<float> latent_bn_scale = DefaultFlux2LatentBnScale();
  std::vector<float> latent_bn_shift = DefaultFlux2LatentBnShift();
};

// Validates that all dimensions and step counts in `config` are within
// supported bounds (`img_size` is a positive multiple of 16, and `img_size`,
// `seq_len`, `packed_ch`, `prompt_dim`, and `steps` are positive and within
// maximum limits).
absl::Status ValidateFlux2ModelConfig(const Flux2ModelConfig& config);

// Populates `config` from `Flux2Params` / `BonsaiFlux2` / `Flux2Klein` protobuf
// messages stored in `ImageGenMetadata` inside a `.litertlm` container.
void PopulateFlux2ConfigFromProto(const lm::proto::Flux2Params& params,
                                  Flux2ModelConfig& config);
void PopulateFlux2ConfigFromProto(const lm::proto::BonsaiFlux2& proto,
                                  Flux2ModelConfig& config);
void PopulateFlux2ConfigFromProto(const lm::proto::Flux2Klein& proto,
                                  Flux2ModelConfig& config);

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MODEL_CONFIG_H_
