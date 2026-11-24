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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_VAE_DECODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_VAE_DECODER_STAGE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/image_decoder.h"
#include "omni/text2image/io_types.h"

namespace litert::omni::text2image {

// Shared VAE decoder stage for text-to-image pipelines. Pulls denoised packed
// latents from an upstream denoiser stage, applies a model-specific unpack and
// scaling callback into a planar NCHW float tensor, invokes a single-input /
// single-output LiteRT VAE decoder runner, and pushes an RGB888 `ImageOutput`.
class VaeDecoderStage : public ImageDecoder {
 public:
  // Static geometry and diagnostic configuration for a `VaeDecoderStage`.
  struct Config {
    int image_width = 0;
    int image_height = 0;
    size_t expected_latent_elements = 0;
    std::string stage_name = "VaeDecoderStage";
  };

  // Transforms denoised packed latents into a planar NCHW float buffer of
  // length `expected_latent_elements` matching the VAE runner's input tensor.
  using UnpackLatentsFn = absl::AnyInvocable<absl::StatusOr<std::vector<float>>(
      const DenoisedLatents& denoised) const>;

  static absl::StatusOr<std::unique_ptr<VaeDecoderStage>> Create(
      Stage<DenoisedLatents>* absl_nonnull denoiser, Config config,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      UnpackLatentsFn unpack_latents);

  ~VaeDecoderStage() override = default;

  absl::Status Flush() override { return absl::OkStatus(); }

 protected:
  // Input and output tensor buffers created and validated for
  // `VaeDecoderStage`.
  struct RunnerBuffers {
    std::vector<TensorBuffer> inputs;
    std::vector<TensorBuffer> outputs;
  };

  static absl::StatusOr<RunnerBuffers> ValidateAndCreateBuffers(
      Stage<DenoisedLatents>* absl_nullable denoiser, const Config& config,
      LiteRtRunner* absl_nullable runner,
      const UnpackLatentsFn& unpack_latents);

  VaeDecoderStage(Stage<DenoisedLatents>* absl_nonnull denoiser, Config config,
                  std::unique_ptr<LiteRtRunner> absl_nonnull runner,
                  UnpackLatentsFn unpack_latents,
                  std::vector<TensorBuffer> input_buffers,
                  std::vector<TensorBuffer> output_buffers);

  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  Stage<DenoisedLatents>& denoiser_;
  const Config config_;
  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  UnpackLatentsFn unpack_latents_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_VAE_DECODER_STAGE_H_
