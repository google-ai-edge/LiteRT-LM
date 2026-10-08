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

#include "omni/text2image/flux2/flux2_denoiser_stage.h"

#include <array>
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
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/base/tensor_utils.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/text_encoder_stage.h"

namespace litert::omni::text2image {

absl::StatusOr<std::unique_ptr<Flux2DenoiserStage>> Flux2DenoiserStage::Create(
    Stage<TextEncoderOutput>* absl_nonnull text_encoder,
    const Flux2ModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    InputIndices input_indices) {
  ABSL_RETURN_IF_ERROR(ValidateFlux2ModelConfig(config));
  if (runner == nullptr) {
    return absl::InvalidArgumentError(
        "Flux2DenoiserStage runner must not be null.");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> input_buffers,
                        runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> output_buffers,
                        runner->CreateOutputBuffers(""));

  if (input_buffers.size() < 5) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Flux2DenoiserStage expected at least 5 input buffers, got %d",
        input_buffers.size()));
  }
  const std::array<size_t, 5> indices = {
      input_indices.hidden, input_indices.enc, input_indices.t,
      input_indices.img_ids, input_indices.txt_ids};
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= input_buffers.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Flux2DenoiserStage input index %d out of bounds for %d input "
          "buffers",
          indices[i], input_buffers.size()));
    }
    for (size_t j = i + 1; j < indices.size(); ++j) {
      if (indices[i] == indices[j]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Flux2DenoiserStage input indices must be distinct, got duplicate "
            "%d",
            indices[i]));
      }
    }
  }
  if (output_buffers.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Flux2DenoiserStage expected 1 output buffer, got %d",
                        output_buffers.size()));
  }

  const size_t grid_dim = static_cast<size_t>(config.img_size / 16);
  const size_t tokens = grid_dim * grid_dim;
  const size_t latent_elements = tokens * static_cast<size_t>(config.packed_ch);
  const size_t prompt_elements =
      static_cast<size_t>(config.seq_len) * config.prompt_dim;
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(input_buffers[input_indices.hidden],
                                           latent_elements,
                                           "Flux2DenoiserStage hidden"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(input_buffers[input_indices.enc],
                                           prompt_elements,
                                           "Flux2DenoiserStage enc"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(input_buffers[input_indices.t], 1,
                                           "Flux2DenoiserStage t"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(input_buffers[input_indices.img_ids],
                                           tokens * 4,
                                           "Flux2DenoiserStage img_ids"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      input_buffers[input_indices.txt_ids],
      static_cast<size_t>(config.seq_len) * 4, "Flux2DenoiserStage txt_ids"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(output_buffers[0], latent_elements,
                                           "Flux2DenoiserStage output"));

  return absl::WrapUnique(new Flux2DenoiserStage(
      text_encoder, config, std::move(runner), input_indices,
      std::move(input_buffers), std::move(output_buffers)));
}

Flux2DenoiserStage::Flux2DenoiserStage(
    Stage<TextEncoderOutput>* absl_nonnull text_encoder,
    Flux2ModelConfig config, std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    InputIndices input_indices, std::vector<TensorBuffer> input_buffers,
    std::vector<TensorBuffer> output_buffers)
    : text_encoder_(*text_encoder),
      config_(std::move(config)),
      runner_(std::move(runner)),
      input_indices_(input_indices),
      input_buffers_(std::move(input_buffers)),
      output_buffers_(std::move(output_buffers)) {}

bool Flux2DenoiserStage::NeedScheduleInternal() const {
  return text_encoder_.HasOutput();
}

absl::Status Flux2DenoiserStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  auto cond = text_encoder_.GetOutput();
  if (absl::IsNotFound(cond.status())) {
    return absl::OkStatus();
  }
  if (!cond.ok()) {
    return cond.status();
  }

  const int width = config_.img_size;
  const int height = config_.img_size;
  const int grid_w = width / 16;
  const int grid_h = height / 16;
  const int tokens = grid_h * grid_w;
  const int packed_ch = config_.packed_ch;
  const int steps = cond->metadata.num_inference_steps > 0
                        ? cond->metadata.num_inference_steps
                        : config_.steps;
  if (steps != config_.steps) {
    Flux2ModelConfig step_config = config_;
    step_config.steps = steps;
    ABSL_RETURN_IF_ERROR(ValidateFlux2ModelConfig(step_config));
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<float> sigmas,
                        ComputeFlowMatchSigmas(steps, tokens));
  std::vector<float> img_ids = BuildImagePositionIds(grid_h, grid_w);
  std::vector<float> txt_ids = BuildTextPositionIds(config_.seq_len);

  const size_t latent_elements = static_cast<size_t>(tokens) * packed_ch;
  const uint64_t seed =
      cond->metadata.seed > 0 ? cond->metadata.seed : config_.default_seed;
  std::vector<float> lat = SampleGaussianLatents(latent_elements, seed);
  std::vector<float> velocity(latent_elements, 0.0f);

  // Write inputs that remain constant across Euler steps.
  LITERT_RETURN_IF_ERROR(input_buffers_[input_indices_.enc].Write<float>(
      absl::MakeConstSpan(cond->prompt_embeds)));
  LITERT_RETURN_IF_ERROR(input_buffers_[input_indices_.img_ids].Write<float>(
      absl::MakeConstSpan(img_ids)));
  LITERT_RETURN_IF_ERROR(input_buffers_[input_indices_.txt_ids].Write<float>(
      absl::MakeConstSpan(txt_ids)));

  for (int step = 0; step < steps; ++step) {
    const float sigma_cur = sigmas[step];
    const float sigma_next = sigmas[step + 1];
    const float dt = sigma_next - sigma_cur;

    LITERT_RETURN_IF_ERROR(input_buffers_[input_indices_.hidden].Write<float>(
        absl::MakeConstSpan(lat)));
    LITERT_RETURN_IF_ERROR(input_buffers_[input_indices_.t].Write<float>(
        absl::MakeConstSpan(&sigma_cur, 1)));

    ABSL_RETURN_IF_ERROR(runner_->Run("", input_buffers_, output_buffers_));

    LITERT_RETURN_IF_ERROR(
        output_buffers_[0].Read<float>(absl::MakeSpan(velocity)));

    for (size_t i = 0; i < latent_elements; ++i) {
      lat[i] += dt * velocity[i];
    }
  }

  Flux2DenoiserOutput output;
  output.metadata = cond->metadata;
  output.metadata.width = width;
  output.metadata.height = height;
  output.metadata.num_inference_steps = steps;
  output.metadata.seed = seed;
  output.packed_latents = std::move(lat);

  PushOutput(std::move(output));
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image
