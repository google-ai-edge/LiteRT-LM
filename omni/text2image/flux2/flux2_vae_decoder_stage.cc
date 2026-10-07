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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/base/tensor_utils.h"
#include "omni/text2image/flux2/flux2_denoiser_stage.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/text2image/flux2/flux2_model_config.h"

namespace litert::omni::text2image {

absl::StatusOr<std::unique_ptr<Flux2VaeDecoderStage>>
Flux2VaeDecoderStage::Create(
    Stage<Flux2DenoiserOutput>* absl_nonnull denoiser,
    const Flux2ModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner) {
  ABSL_RETURN_IF_ERROR(ValidateFlux2ModelConfig(config));
  if (runner == nullptr) {
    return absl::InvalidArgumentError(
        "Flux2VaeDecoderStage runner must not be null.");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> input_buffers,
                        runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> output_buffers,
                        runner->CreateOutputBuffers(""));

  if (input_buffers.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Flux2VaeDecoderStage expected 1 input buffer, got %d",
                        input_buffers.size()));
  }
  if (output_buffers.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Flux2VaeDecoderStage expected 1 output buffer, got %d",
                        output_buffers.size()));
  }

  const size_t grid_dim = static_cast<size_t>(config.img_size / 16);
  const size_t expected_in_elements = 32 * (2 * grid_dim) * (2 * grid_dim);
  const size_t expected_out_elements =
      static_cast<size_t>(3) * config.img_size * config.img_size;
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      input_buffers[0], expected_in_elements, "Flux2VaeDecoderStage input"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      output_buffers[0], expected_out_elements, "Flux2VaeDecoderStage output"));

  return absl::WrapUnique(new Flux2VaeDecoderStage(
      denoiser, config, std::move(runner), std::move(input_buffers),
      std::move(output_buffers)));
}

Flux2VaeDecoderStage::Flux2VaeDecoderStage(
    Stage<Flux2DenoiserOutput>* absl_nonnull denoiser, Flux2ModelConfig config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    std::vector<TensorBuffer> input_buffers,
    std::vector<TensorBuffer> output_buffers)
    : denoiser_(*denoiser),
      config_(std::move(config)),
      runner_(std::move(runner)),
      input_buffers_(std::move(input_buffers)),
      output_buffers_(std::move(output_buffers)) {}

bool Flux2VaeDecoderStage::NeedScheduleInternal() const {
  return denoiser_.HasOutput();
}

absl::Status Flux2VaeDecoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  auto denoised = denoiser_.GetOutput();
  if (absl::IsNotFound(denoised.status())) {
    return absl::OkStatus();
  }
  if (!denoised.ok()) {
    return denoised.status();
  }

  const int width = config_.img_size;
  const int height = config_.img_size;
  const int grid_w = width / 16;
  const int grid_h = height / 16;

  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> z_nchw,
      UnpatchifyLatents(denoised->packed_latents, grid_h, grid_w,
                        config_.latent_bn_scale, config_.latent_bn_shift));

  LITERT_RETURN_IF_ERROR(
      input_buffers_[0].Write<float>(absl::MakeConstSpan(z_nchw)));
  ABSL_RETURN_IF_ERROR(runner_->Run("", input_buffers_, output_buffers_));

  const size_t nchw_elements = static_cast<size_t>(3) * height * width;
  std::vector<float> img_nchw(nchw_elements, 0.0f);
  LITERT_RETURN_IF_ERROR(
      output_buffers_[0].Read<float>(absl::MakeSpan(img_nchw)));

  ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> rgb888,
                        ConvertNchwToRgb888(img_nchw, height, width));

  ImageOutput image;
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.rgb_data = std::move(rgb888);

  PushOutput(std::move(image));
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image
