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

#include "omni/text2image/flux2/klein_denoiser_stage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/base/tensor_utils.h"
#include "omni/text2image/flux2/flux2_denoiser_stage.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/text_encoder_stage.h"

namespace litert::omni::text2image {
namespace {

absl::StatusOr<size_t> GetBufferNumElements(const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(const size_t num_elements,
                          tensor_type.Layout().NumElements());
  return num_elements;
}

absl::Status ValidateDistinctIndices(absl::Span<const size_t> indices,
                                     size_t num_buffers,
                                     absl::string_view stage_name) {
  if (num_buffers < indices.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expected at least %d input buffers, got %d.", stage_name,
        indices.size(), num_buffers));
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= num_buffers) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s input index %d out of bounds for %d input buffers.", stage_name,
          indices[i], num_buffers));
    }
    for (size_t j = i + 1; j < indices.size(); ++j) {
      if (indices[i] == indices[j]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s input indices must be distinct, got duplicate %d.", stage_name,
            indices[i]));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<KleinDenoiserStage>> KleinDenoiserStage::Create(
    Stage<TextEncoderOutput>* absl_nonnull text_encoder,
    const Flux2ModelConfig& config, Runners runners,
    InputIndices input_indices) {
  ABSL_RETURN_IF_ERROR(ValidateFlux2ModelConfig(config));
  if (runners.initial == nullptr || runners.final_stage == nullptr) {
    return absl::InvalidArgumentError(
        "KleinDenoiserStage initial and final_stage runners must not be null.");
  }
  for (size_t i = 0; i < 2; ++i) {
    if (runners.double_blocks[i] == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "KleinDenoiserStage double_blocks[%d] runner must not be null.", i));
    }
  }
  for (size_t i = 0; i < 4; ++i) {
    if (runners.single_blocks[i] == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "KleinDenoiserStage single_blocks[%d] runner must not be null.", i));
    }
  }

  Buffers buffers;
  ABSL_ASSIGN_OR_RETURN(buffers.initial_in,
                        runners.initial->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(buffers.initial_out,
                        runners.initial->CreateOutputBuffers(""));

  const std::array<size_t, 3> init_indices = {input_indices.initial.hidden,
                                              input_indices.initial.enc,
                                              input_indices.initial.t};
  ABSL_RETURN_IF_ERROR(ValidateDistinctIndices(
      init_indices, buffers.initial_in.size(), "KleinDenoiserStage initial"));
  if (buffers.initial_out.size() != 5 && buffers.initial_out.size() != 6) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "KleinDenoiserStage initial expected 5 or 6 output buffers, got %d.",
        buffers.initial_out.size()));
  }
  const bool initial_outputs_temb = (buffers.initial_out.size() == 6);

  const size_t grid_dim = static_cast<size_t>(config.img_size / 16);
  const size_t tokens = grid_dim * grid_dim;
  const size_t seq_len = static_cast<size_t>(config.seq_len);
  const size_t joint_tokens = seq_len + tokens;
  const size_t latent_elements = tokens * static_cast<size_t>(config.packed_ch);
  const size_t prompt_elements =
      seq_len * static_cast<size_t>(config.prompt_dim);

  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.initial_in[input_indices.initial.hidden], latent_elements,
      "KleinDenoiserStage initial hidden"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.initial_in[input_indices.initial.enc], prompt_elements,
      "KleinDenoiserStage initial enc"));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.initial_in[input_indices.initial.t], 1,
                          "KleinDenoiserStage initial t"));

  ABSL_ASSIGN_OR_RETURN(const size_t img_out_elements,
                        GetBufferNumElements(buffers.initial_out[0]));
  if (img_out_elements == 0 || img_out_elements % tokens != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "KleinDenoiserStage initial image output elements (%d) must be a "
        "positive multiple of tokens (%d).",
        img_out_elements, tokens));
  }
  const int dit_hidden_dim = static_cast<int>(img_out_elements / tokens);
  const size_t img_hidden_elements = tokens * dit_hidden_dim;
  const size_t txt_hidden_elements = seq_len * dit_hidden_dim;
  const size_t joint_hidden_elements = joint_tokens * dit_hidden_dim;
  const size_t mod_double_elements = 6 * static_cast<size_t>(dit_hidden_dim);
  const size_t mod_single_elements = 3 * static_cast<size_t>(dit_hidden_dim);

  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.initial_out[0],
                                           img_hidden_elements,
                                           "KleinDenoiserStage initial image"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.initial_out[1],
                                           txt_hidden_elements,
                                           "KleinDenoiserStage initial text"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.initial_out[2], mod_double_elements,
      "KleinDenoiserStage initial mod_img"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.initial_out[3], mod_double_elements,
      "KleinDenoiserStage initial mod_txt"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.initial_out[4], mod_single_elements,
      "KleinDenoiserStage initial mod_single"));
  if (initial_outputs_temb) {
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.initial_out[5], static_cast<size_t>(dit_hidden_dim),
        "KleinDenoiserStage initial temb"));
  }

  int inferred_rope_dim = 0;
  for (size_t i = 0; i < 2; ++i) {
    const std::string label =
        absl::StrCat("KleinDenoiserStage double_block_", i);
    ABSL_ASSIGN_OR_RETURN(buffers.double_in[i],
                          runners.double_blocks[i]->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(buffers.double_out[i],
                          runners.double_blocks[i]->CreateOutputBuffers(""));

    const auto& idx = input_indices.double_blocks[i];
    const std::array<size_t, 6> indices = {idx.image, idx.text,    idx.cos,
                                           idx.sin,   idx.mod_img, idx.mod_txt};
    ABSL_RETURN_IF_ERROR(
        ValidateDistinctIndices(indices, buffers.double_in[i].size(), label));
    if (buffers.double_out[i].size() != 2) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s expected 2 output buffers, got %d.", label,
                          buffers.double_out[i].size()));
    }

    ABSL_ASSIGN_OR_RETURN(const size_t cos_elements,
                          GetBufferNumElements(buffers.double_in[i][idx.cos]));
    if (cos_elements == 0 || cos_elements % joint_tokens != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s cos elements (%d) must be a positive multiple of joint_tokens "
          "(%d).",
          label, cos_elements, joint_tokens));
    }
    const int rope_dim = static_cast<int>(cos_elements / joint_tokens);
    if (rope_dim % 8 != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rope head_dim (%d) must be a multiple of 8.", label, rope_dim));
    }
    if (i == 0) {
      inferred_rope_dim = rope_dim;
    } else if (rope_dim != inferred_rope_dim) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rope head_dim (%d) does not match double_block_0 (%d).", label,
          rope_dim, inferred_rope_dim));
    }

    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.double_in[i][idx.image], img_hidden_elements,
        absl::StrCat(label, " image")));
    ABSL_RETURN_IF_ERROR(
        ValidateFloatBuffer(buffers.double_in[i][idx.text], txt_hidden_elements,
                            absl::StrCat(label, " text")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.double_in[i][idx.cos],
                                             cos_elements,
                                             absl::StrCat(label, " cos")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.double_in[i][idx.sin],
                                             cos_elements,
                                             absl::StrCat(label, " sin")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.double_in[i][idx.mod_img], mod_double_elements,
        absl::StrCat(label, " mod_img")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.double_in[i][idx.mod_txt], mod_double_elements,
        absl::StrCat(label, " mod_txt")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.double_out[i][0], img_hidden_elements,
        absl::StrCat(label, " output image")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.double_out[i][1], txt_hidden_elements,
        absl::StrCat(label, " output text")));
  }

  const size_t rope_elements = joint_tokens * inferred_rope_dim;
  for (size_t i = 0; i < 4; ++i) {
    const std::string label =
        absl::StrCat("KleinDenoiserStage single_block_", i);
    ABSL_ASSIGN_OR_RETURN(buffers.single_in[i],
                          runners.single_blocks[i]->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(buffers.single_out[i],
                          runners.single_blocks[i]->CreateOutputBuffers(""));

    const auto& idx = input_indices.single_blocks[i];
    const std::array<size_t, 4> indices = {idx.joint, idx.cos, idx.sin,
                                           idx.mod_single};
    ABSL_RETURN_IF_ERROR(
        ValidateDistinctIndices(indices, buffers.single_in[i].size(), label));
    if (buffers.single_out[i].size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s expected 1 output buffer, got %d.", label,
                          buffers.single_out[i].size()));
    }

    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.single_in[i][idx.joint], joint_hidden_elements,
        absl::StrCat(label, " joint")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.single_in[i][idx.cos],
                                             rope_elements,
                                             absl::StrCat(label, " cos")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.single_in[i][idx.sin],
                                             rope_elements,
                                             absl::StrCat(label, " sin")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.single_in[i][idx.mod_single], mod_single_elements,
        absl::StrCat(label, " mod_single")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        buffers.single_out[i][0], joint_hidden_elements,
        absl::StrCat(label, " output joint")));
  }

  ABSL_ASSIGN_OR_RETURN(buffers.final_in,
                        runners.final_stage->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(buffers.final_out,
                        runners.final_stage->CreateOutputBuffers(""));
  const std::array<size_t, 2> final_indices = {input_indices.final_stage.joint,
                                               input_indices.final_stage.temb};
  ABSL_RETURN_IF_ERROR(ValidateDistinctIndices(
      final_indices, buffers.final_in.size(), "KleinDenoiserStage final"));
  if (buffers.final_out.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "KleinDenoiserStage final expected 1 output buffer, got %d.",
        buffers.final_out.size()));
  }
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.final_in[input_indices.final_stage.joint], joint_hidden_elements,
      "KleinDenoiserStage final joint"));
  const size_t expected_final_temb_size =
      initial_outputs_temb ? static_cast<size_t>(dit_hidden_dim) : 1;
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.final_in[input_indices.final_stage.temb],
                          expected_final_temb_size,
                          "KleinDenoiserStage final temb"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.final_out[0],
                                           latent_elements,
                                           "KleinDenoiserStage final output"));

  // Precompute 4D axial RoPE (cos, sin) for [txt_ids; img_ids] and write into
  // all double and single block runners.
  std::vector<float> txt_ids = BuildTextPositionIds(config.seq_len);
  std::vector<float> img_ids =
      BuildImagePositionIds(static_cast<int>(grid_dim),
                            static_cast<int>(grid_dim));
  std::vector<float> joint_ids;
  joint_ids.reserve(txt_ids.size() + img_ids.size());
  joint_ids.insert(joint_ids.end(), txt_ids.begin(), txt_ids.end());
  joint_ids.insert(joint_ids.end(), img_ids.begin(), img_ids.end());

  // TODO(b/568027544): Carry `axes_dim` in `Flux2ModelConfig` if a future
  // variant uses non-uniform RoPE axis dimensions.
  const int axis_dim = inferred_rope_dim / 4;
  const std::array<int, 4> axes_dim = {axis_dim, axis_dim, axis_dim, axis_dim};
  ABSL_ASSIGN_OR_RETURN(
      auto rope,
      BuildFlux2RotaryPosEmbed(joint_ids, static_cast<int>(joint_tokens),
                               axes_dim));
  for (size_t i = 0; i < 2; ++i) {
    const auto& idx = input_indices.double_blocks[i];
    LITERT_RETURN_IF_ERROR(buffers.double_in[i][idx.cos].Write<float>(
        absl::MakeConstSpan(rope.first)));
    LITERT_RETURN_IF_ERROR(buffers.double_in[i][idx.sin].Write<float>(
        absl::MakeConstSpan(rope.second)));
  }
  for (size_t i = 0; i < 4; ++i) {
    const auto& idx = input_indices.single_blocks[i];
    LITERT_RETURN_IF_ERROR(buffers.single_in[i][idx.cos].Write<float>(
        absl::MakeConstSpan(rope.first)));
    LITERT_RETURN_IF_ERROR(buffers.single_in[i][idx.sin].Write<float>(
        absl::MakeConstSpan(rope.second)));
  }

  return absl::WrapUnique(new KleinDenoiserStage(
      text_encoder, config, std::move(runners), input_indices,
      std::move(buffers), dit_hidden_dim, initial_outputs_temb));
}

KleinDenoiserStage::KleinDenoiserStage(
    Stage<TextEncoderOutput>* absl_nonnull text_encoder,
    Flux2ModelConfig config, Runners runners, InputIndices input_indices,
    Buffers buffers, int dit_hidden_dim, bool initial_outputs_temb)
    : text_encoder_(*text_encoder),
      config_(std::move(config)),
      runners_(std::move(runners)),
      input_indices_(input_indices),
      buffers_(std::move(buffers)),
      dit_hidden_dim_(dit_hidden_dim),
      initial_outputs_temb_(initial_outputs_temb) {}

bool KleinDenoiserStage::NeedScheduleInternal() const {
  return text_encoder_.HasOutput();
}

absl::Status KleinDenoiserStage::ScheduleInternal() {
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
  const int seq_len = config_.seq_len;
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

  const size_t latent_elements = static_cast<size_t>(tokens) * packed_ch;
  const size_t img_hidden_elements =
      static_cast<size_t>(tokens) * dit_hidden_dim_;
  const size_t txt_hidden_elements =
      static_cast<size_t>(seq_len) * dit_hidden_dim_;
  const size_t joint_hidden_elements =
      img_hidden_elements + txt_hidden_elements;
  const size_t mod_double_elements = 6 * static_cast<size_t>(dit_hidden_dim_);
  const size_t mod_single_elements = 3 * static_cast<size_t>(dit_hidden_dim_);

  const uint64_t seed =
      cond->metadata.seed > 0 ? cond->metadata.seed : config_.default_seed;
  std::vector<float> lat = SampleGaussianLatents(latent_elements, seed);
  std::vector<float> velocity(latent_elements, 0.0f);

  std::vector<float> image(img_hidden_elements, 0.0f);
  std::vector<float> text(txt_hidden_elements, 0.0f);
  std::vector<float> mod_img(mod_double_elements, 0.0f);
  std::vector<float> mod_txt(mod_double_elements, 0.0f);
  std::vector<float> mod_single(mod_single_elements, 0.0f);
  std::vector<float> temb(static_cast<size_t>(dit_hidden_dim_), 0.0f);
  std::vector<float> joint(joint_hidden_elements, 0.0f);

  LITERT_RETURN_IF_ERROR(
      buffers_.initial_in[input_indices_.initial.enc].Write<float>(
          absl::MakeConstSpan(cond->prompt_embeds)));

  for (int step = 0; step < steps; ++step) {
    const float sigma_cur = sigmas[step];
    const float sigma_next = sigmas[step + 1];
    const float dt = sigma_next - sigma_cur;

    // 1. Initial embedding and modulation stage.
    LITERT_RETURN_IF_ERROR(
        buffers_.initial_in[input_indices_.initial.hidden].Write<float>(
            absl::MakeConstSpan(lat)));
    LITERT_RETURN_IF_ERROR(
        buffers_.initial_in[input_indices_.initial.t].Write<float>(
            absl::MakeConstSpan(&sigma_cur, 1)));
    ABSL_RETURN_IF_ERROR(
        runners_.initial->Run("", buffers_.initial_in, buffers_.initial_out));

    LITERT_RETURN_IF_ERROR(
        buffers_.initial_out[0].Read<float>(absl::MakeSpan(image)));
    LITERT_RETURN_IF_ERROR(
        buffers_.initial_out[1].Read<float>(absl::MakeSpan(text)));
    LITERT_RETURN_IF_ERROR(
        buffers_.initial_out[2].Read<float>(absl::MakeSpan(mod_img)));
    LITERT_RETURN_IF_ERROR(
        buffers_.initial_out[3].Read<float>(absl::MakeSpan(mod_txt)));
    LITERT_RETURN_IF_ERROR(
        buffers_.initial_out[4].Read<float>(absl::MakeSpan(mod_single)));
    if (initial_outputs_temb_) {
      LITERT_RETURN_IF_ERROR(
          buffers_.initial_out[5].Read<float>(absl::MakeSpan(temb)));
    }

    // 2. Double-stream transformer block shards (0 and 1).
    for (size_t i = 0; i < 2; ++i) {
      const auto& idx = input_indices_.double_blocks[i];
      LITERT_RETURN_IF_ERROR(buffers_.double_in[i][idx.image].Write<float>(
          absl::MakeConstSpan(image)));
      LITERT_RETURN_IF_ERROR(buffers_.double_in[i][idx.text].Write<float>(
          absl::MakeConstSpan(text)));
      LITERT_RETURN_IF_ERROR(buffers_.double_in[i][idx.mod_img].Write<float>(
          absl::MakeConstSpan(mod_img)));
      LITERT_RETURN_IF_ERROR(buffers_.double_in[i][idx.mod_txt].Write<float>(
          absl::MakeConstSpan(mod_txt)));

      ABSL_RETURN_IF_ERROR(runners_.double_blocks[i]->Run(
          "", buffers_.double_in[i], buffers_.double_out[i]));

      LITERT_RETURN_IF_ERROR(
          buffers_.double_out[i][0].Read<float>(absl::MakeSpan(image)));
      LITERT_RETURN_IF_ERROR(
          buffers_.double_out[i][1].Read<float>(absl::MakeSpan(text)));
    }

    // 3. Concatenate [text; image] along token dimension into `joint`.
    std::memcpy(joint.data(), text.data(), txt_hidden_elements * sizeof(float));
    std::memcpy(joint.data() + txt_hidden_elements, image.data(),
                img_hidden_elements * sizeof(float));

    // 4. Single-stream transformer block shards (0..3).
    for (size_t i = 0; i < 4; ++i) {
      const auto& idx = input_indices_.single_blocks[i];
      LITERT_RETURN_IF_ERROR(buffers_.single_in[i][idx.joint].Write<float>(
          absl::MakeConstSpan(joint)));
      LITERT_RETURN_IF_ERROR(buffers_.single_in[i][idx.mod_single].Write<float>(
          absl::MakeConstSpan(mod_single)));

      ABSL_RETURN_IF_ERROR(runners_.single_blocks[i]->Run(
          "", buffers_.single_in[i], buffers_.single_out[i]));

      LITERT_RETURN_IF_ERROR(
          buffers_.single_out[i][0].Read<float>(absl::MakeSpan(joint)));
    }

    // 5. Final norm and output projection stage.
    LITERT_RETURN_IF_ERROR(
        buffers_.final_in[input_indices_.final_stage.joint].Write<float>(
            absl::MakeConstSpan(joint)));
    if (initial_outputs_temb_) {
      LITERT_RETURN_IF_ERROR(
          buffers_.final_in[input_indices_.final_stage.temb].Write<float>(
              absl::MakeConstSpan(temb)));
    } else {
      LITERT_RETURN_IF_ERROR(
          buffers_.final_in[input_indices_.final_stage.temb].Write<float>(
              absl::MakeConstSpan(&sigma_cur, 1)));
    }

    ABSL_RETURN_IF_ERROR(
        runners_.final_stage->Run("", buffers_.final_in, buffers_.final_out));
    LITERT_RETURN_IF_ERROR(
        buffers_.final_out[0].Read<float>(absl::MakeSpan(velocity)));

    // 6. FlowMatch Euler step update.
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
