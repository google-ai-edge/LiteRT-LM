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

#include "omni/text2image/flux2/flux2_model_config.h"

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "runtime/proto/image_gen_model_type.pb.h"

namespace litert::omni::text2image {
namespace {

// Upper bounds for FLUX.2 configuration parameters to guard against integer
// overflow and excessive memory allocation.
constexpr int kMaxFlux2SeqLen = 4096;
constexpr int kMaxFlux2ImgSize = 4096;
constexpr int kMaxFlux2PackedChannels = 1024;
constexpr int kMaxFlux2PromptDim = 65536;
constexpr int kMaxFlux2Steps = 1000;

}  // namespace

// These 128-channel BatchNorm scale (`sqrt(vae.bn.running_var + 1e-4)`) and
// shift (`vae.bn.running_mean`) constants are extracted from the pretrained
// `AutoencoderKLFlux2` (`vae.bn`) weights of
// `black-forest-labs/FLUX.2-klein-4B` (and `prism-ml/Bonsai-Image-ternary-4B`,
// which shares the same VAE). In FLUX.2, 32-channel VAE latents are
// 2x2-patchified into 128 channels and normalized with `vae.bn`; during
// decoding (`Flux2KleinPipeline` in
// `diffusers/pipelines/flux2/pipeline_flux2_klein.py`), predicted packed
// latents are denormalized via `latents * scale + shift` before unpatchifying.
std::vector<float> DefaultFlux2LatentBnScale() {
  return {
      1.80280337f, 1.77661194f, 1.78538511f, 1.78538511f, 1.77220907f,
      1.75893434f, 1.75893434f, 1.75002857f, 1.73207967f, 1.73658429f,
      1.73207967f, 1.73207967f, 1.86248490f, 1.85407659f, 1.86248490f,
      1.85828550f, 1.75893434f, 1.75448710f, 1.75448710f, 1.75893434f,
      1.73658429f, 1.74107725f, 1.73658429f, 1.74107725f, 1.73207967f,
      1.72303511f, 1.74107725f, 1.73207967f, 1.75448710f, 1.75002857f,
      1.75448710f, 1.75002857f, 1.84562997f, 1.84562997f, 1.85407659f,
      1.85407659f, 1.82434235f, 1.78100393f, 1.78538511f, 1.79411538f,
      1.80280337f, 1.78975557f, 1.76779524f, 1.76779524f, 1.77661194f,
      1.76779524f, 1.79411538f, 1.78538511f, 1.75893434f, 1.74107725f,
      1.75002857f, 1.73658429f, 1.79411538f, 1.79846462f, 1.79411538f,
      1.79411538f, 1.77661194f, 1.77661194f, 1.77220907f, 1.77661194f,
      1.75448710f, 1.74555865f, 1.76779524f, 1.75002857f, 1.77661194f,
      1.77220907f, 1.78100393f, 1.78100393f, 1.81575742f, 1.80280337f,
      1.80280337f, 1.80280337f, 1.75448710f, 1.73207967f, 1.75448710f,
      1.75002857f, 1.75893434f, 1.75002857f, 1.76779524f, 1.76337035f,
      1.76779524f, 1.75448710f, 1.78100393f, 1.77220907f, 1.78975557f,
      1.79846462f, 1.78538511f, 1.77661194f, 1.76337035f, 1.76337035f,
      1.75448710f, 1.75893434f, 1.77661194f, 1.76779524f, 1.78100393f,
      1.78100393f, 1.71849498f, 1.70480204f, 1.72756331f, 1.70937854f,
      1.78100393f, 1.75893434f, 1.75002857f, 1.74555865f, 1.75893434f,
      1.75002857f, 1.75448710f, 1.73658429f, 1.76337035f, 1.75893434f,
      1.74555865f, 1.74107725f, 1.75002857f, 1.75893434f, 1.76337035f,
      1.76337035f, 1.72756331f, 1.71394282f, 1.73207967f, 1.72303511f,
      1.76779524f, 1.76337035f, 1.77220907f, 1.76779524f, 1.74555865f,
      1.74555865f, 1.75893434f, 1.75448710f,
  };
}

std::vector<float> DefaultFlux2LatentBnShift() {
  return {
      -0.0673828125f,      -0.0712890625f,        -0.0751953125f,
      -0.07470703125f,     0.0223388671875f,      0.0179443359375f,
      0.01422119140625f,   0.018310546875f,       -6.29425049e-05f,
      -0.006256103515625f, -0.000209808350f,      -0.003143310546875f,
      -0.0272216796875f,   -0.028076171875f,      -0.027587890625f,
      -0.029052734375f,    -0.07666015625f,       -0.0673828125f,
      -0.09033203125f,     -0.08935546875f,       0.016845703125f,
      0.01519775390625f,   0.00787353515625f,     0.00860595703125f,
      0.00836181640625f,   0.0015411376953125f,   0.000257492065f,
      -0.0042724609375f,   -0.0439453125f,        -0.0419921875f,
      -0.043701171875f,    -0.043212890625f,      -0.01025390625f,
      -0.01318359375f,     -0.006622314453125f,   -0.0047607421875f,
      -0.031005859375f,    -0.030517578125f,      -0.0279541015625f,
      -0.0179443359375f,   0.003021240234375f,    0.00150299072265625f,
      0.0125732421875f,    0.01446533203125f,     0.03466796875f,
      0.03369140625f,      0.03369140625f,        0.0283203125f,
      0.001983642578125f,  0.004730224609375f,    0.004669189453125f,
      0.004974365234375f,  0.01226806640625f,     0.00811767578125f,
      0.008056640625f,     0.01458740234375f,     0.06787109375f,
      0.06787109375f,      0.07666015625f,        0.0732421875f,
      -0.046142578125f,    -0.04736328125f,       -0.039306640625f,
      -0.051025390625f,    -0.052734375f,         -0.0478515625f,
      -0.047119140625f,    -0.0517578125f,        -0.03173828125f,
      -0.03173828125f,     -0.034423828125f,      -0.0281982421875f,
      0.051025390625f,     0.04443359375f,        0.057861328125f,
      0.0458984375f,       -0.041259765625f,      -0.0458984375f,
      -0.048828125f,       -0.046630859375f,      -0.00885009765625f,
      -0.0106201171875f,   -0.0087890625f,        -0.004608154296875f,
      -0.03759765625f,     -0.043212890625f,      -0.04345703125f,
      -0.0498046875f,      0.0118408203125f,      0.0166015625f,
      0.020263671875f,     0.0279541015625f,      0.01129150390625f,
      0.01287841796875f,   0.001556396484375f,    0.00714111328125f,
      -0.01177978515625f,  -0.00183868408203125f, -0.01416015625f,
      -0.00537109375f,     -0.00909423828125f,    -0.0137939453125f,
      -0.01446533203125f,  -0.0186767578125f,     0.0322265625f,
      0.030517578125f,     0.02587890625f,        0.0299072265625f,
      0.053955078125f,     0.0615234375f,         0.049560546875f,
      0.05908203125f,      -0.051025390625f,      -0.060302734375f,
      -0.0478515625f,      -0.052490234375f,      -0.022705078125f,
      -0.0274658203125f,   -0.015380859375f,      -0.0255126953125f,
      -0.05712890625f,     -0.056396484375f,      -0.0517578125f,
      -0.049560546875f,    0.0115966796875f,      0.00543212890625f,
      0.016357421875f,     0.0103759765625f,
  };
}

absl::Status ValidateFlux2ModelConfig(const Flux2ModelConfig& config) {
  if (config.img_size <= 0 || config.img_size > kMaxFlux2ImgSize ||
      config.img_size % 16 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Image dimensions (%dx%d) must be positive multiples of 16 and at most "
        "%d.",
        config.img_size, config.img_size, kMaxFlux2ImgSize));
  }
  if (config.seq_len <= 0 || config.seq_len > kMaxFlux2SeqLen ||
      config.packed_ch <= 0 || config.packed_ch > kMaxFlux2PackedChannels ||
      config.prompt_dim <= 0 || config.prompt_dim > kMaxFlux2PromptDim ||
      config.steps <= 0 || config.steps > kMaxFlux2Steps) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "seq_len (%d), packed_ch (%d), prompt_dim (%d), and steps (%d) must be "
        "within valid positive bounds.",
        config.seq_len, config.packed_ch, config.prompt_dim, config.steps));
  }
  return absl::OkStatus();
}

void PopulateFlux2ConfigFromProto(const lm::proto::Flux2Params& params,
                                  Flux2ModelConfig& config) {
  if (params.has_seq_len() && params.seq_len() > 0 &&
      params.seq_len() <= kMaxFlux2SeqLen) {
    config.seq_len = params.seq_len();
  }
  if (params.has_img_size() && params.img_size() > 0 &&
      params.img_size() <= kMaxFlux2ImgSize) {
    config.img_size = params.img_size();
  }
  if (params.has_packed_ch() && params.packed_ch() > 0 &&
      params.packed_ch() <= kMaxFlux2PackedChannels) {
    config.packed_ch = params.packed_ch();
  }
  if (params.has_prompt_dim() && params.prompt_dim() > 0 &&
      params.prompt_dim() <= kMaxFlux2PromptDim) {
    config.prompt_dim = params.prompt_dim();
  }
  if (params.has_default_steps() && params.default_steps() > 0 &&
      params.default_steps() <= kMaxFlux2Steps) {
    config.steps = params.default_steps();
  }
  if (!params.latent_bn_scale().empty()) {
    config.latent_bn_scale.assign(params.latent_bn_scale().begin(),
                                  params.latent_bn_scale().end());
  }
  if (!params.latent_bn_shift().empty()) {
    config.latent_bn_shift.assign(params.latent_bn_shift().begin(),
                                  params.latent_bn_shift().end());
  }
}

void PopulateFlux2ConfigFromProto(const lm::proto::BonsaiFlux2& proto,
                                  Flux2ModelConfig& config) {
  PopulateFlux2ConfigFromProto(proto.flux2_params(), config);
}

void PopulateFlux2ConfigFromProto(const lm::proto::Flux2Klein& proto,
                                  Flux2ModelConfig& config) {
  PopulateFlux2ConfigFromProto(proto.flux2_params(), config);
}

}  // namespace litert::omni::text2image
