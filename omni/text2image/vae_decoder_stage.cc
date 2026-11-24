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

#include "omni/text2image/vae_decoder_stage.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/io_types.h"
#include "omni/text2image/text2image_math.h"
#include "omni/utils/tensor_utils.h"

namespace litert::omni::text2image {
namespace {

absl::string_view StageNameOrDefault(const VaeDecoderStage::Config& config) {
  return config.stage_name.empty() ? absl::string_view("VaeDecoderStage")
                                   : absl::string_view(config.stage_name);
}

}  // namespace

absl::StatusOr<VaeDecoderStage::RunnerBuffers>
VaeDecoderStage::ValidateAndCreateBuffers(
    Stage<DenoisedLatents>* absl_nullable denoiser, const Config& config,
    LiteRtRunner* absl_nullable runner, const UnpackLatentsFn& unpack_latents) {
  const absl::string_view stage_name = StageNameOrDefault(config);
  if (denoiser == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(stage_name, " denoiser must not be null."));
  }
  if (runner == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(stage_name, " runner must not be null."));
  }
  if (!unpack_latents) {
    return absl::InvalidArgumentError(
        absl::StrCat(stage_name, " unpack_latents callback must not be null."));
  }
  if (config.image_width <= 0 || config.image_height <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s image dimensions must be positive, got %dx%d.",
                        stage_name, config.image_width, config.image_height));
  }
  if (config.expected_latent_elements == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        stage_name, " expected_latent_elements must be positive."));
  }
  if (config.expected_latent_elements >
      std::numeric_limits<size_t>::max() / sizeof(float)) {
    return absl::InvalidArgumentError(absl::StrCat(
        stage_name, " expected_latent_elements overflows size_t."));
  }
  const size_t width = static_cast<size_t>(config.image_width);
  const size_t height = static_cast<size_t>(config.image_height);
  constexpr size_t kBytesPerPixel = 3 * sizeof(float);
  if (width > std::numeric_limits<size_t>::max() / kBytesPerPixel ||
      height > std::numeric_limits<size_t>::max() / (kBytesPerPixel * width)) {
    return absl::InvalidArgumentError(
        absl::StrCat(stage_name, " image dimensions overflow size_t."));
  }
  const size_t expected_out_elements = 3 * height * width;

  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> input_buffers,
                        runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> output_buffers,
                        runner->CreateOutputBuffers(""));

  if (input_buffers.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expected 1 input buffer, got %d.", stage_name,
                        input_buffers.size()));
  }
  if (output_buffers.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expected 1 output buffer, got %d.", stage_name,
                        output_buffers.size()));
  }

  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(input_buffers[0],
                                           config.expected_latent_elements,
                                           absl::StrCat(stage_name, " input")));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(output_buffers[0], expected_out_elements,
                          absl::StrCat(stage_name, " output")));

  return RunnerBuffers{
      .inputs = std::move(input_buffers),
      .outputs = std::move(output_buffers),
  };
}

absl::StatusOr<std::unique_ptr<VaeDecoderStage>> VaeDecoderStage::Create(
    Stage<DenoisedLatents>* absl_nonnull denoiser, Config config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    UnpackLatentsFn unpack_latents) {
  ABSL_ASSIGN_OR_RETURN(
      RunnerBuffers buffers,
      ValidateAndCreateBuffers(denoiser, config, runner.get(), unpack_latents));
  return absl::WrapUnique(new VaeDecoderStage(
      denoiser, std::move(config), std::move(runner), std::move(unpack_latents),
      std::move(buffers.inputs), std::move(buffers.outputs)));
}

VaeDecoderStage::VaeDecoderStage(
    Stage<DenoisedLatents>* absl_nonnull denoiser, Config config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    UnpackLatentsFn unpack_latents, std::vector<TensorBuffer> input_buffers,
    std::vector<TensorBuffer> output_buffers)
    : denoiser_(*denoiser),
      config_(std::move(config)),
      runner_(std::move(runner)),
      unpack_latents_(std::move(unpack_latents)),
      input_buffers_(std::move(input_buffers)),
      output_buffers_(std::move(output_buffers)) {}

bool VaeDecoderStage::NeedScheduleInternal() const {
  return denoiser_.HasOutput();
}

absl::Status VaeDecoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  absl::StatusOr<DenoisedLatents> denoised = denoiser_.GetOutput();
  if (absl::IsNotFound(denoised.status())) {
    return absl::OkStatus();
  }
  if (!denoised.ok()) {
    return denoised.status();
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<float> z_nchw, unpack_latents_(*denoised));
  if (z_nchw.size() != config_.expected_latent_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s unpack_latents returned %d elements, expected %d.",
                        StageNameOrDefault(config_), z_nchw.size(),
                        config_.expected_latent_elements));
  }

  LITERT_RETURN_IF_ERROR(
      input_buffers_[0].Write<float>(absl::MakeConstSpan(z_nchw)));
  ABSL_RETURN_IF_ERROR(runner_->Run("", input_buffers_, output_buffers_));

  const int width = config_.image_width;
  const int height = config_.image_height;
  const size_t nchw_elements =
      static_cast<size_t>(3) * static_cast<size_t>(height) * width;
  std::vector<uint8_t> rgb888;
  {
    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        TensorBufferScopedLock::Create<const float>(
            output_buffers_[0], TensorBuffer::LockMode::kRead));
    ABSL_ASSIGN_OR_RETURN(
        rgb888, ConvertNchwToRgb888(
                    absl::MakeConstSpan(lock_and_addr.second, nchw_elements),
                    height, width));
  }

  ImageOutput image;
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.rgb_data = std::move(rgb888);

  PushOutput(std::move(image));
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image
