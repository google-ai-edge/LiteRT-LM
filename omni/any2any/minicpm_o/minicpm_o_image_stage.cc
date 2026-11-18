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

#include "omni/any2any/minicpm_o/minicpm_o_image_stage.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/functional/bind_front.h"  // from @com_google_absl
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
#include "omni/any2any/any2any_image_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/minicpmv_image_preprocess.h"

namespace litert::omni::any2any {
namespace {

// Passthrough ImagePreprocessor that feeds an already-decoded RGB888 image into
// `lm::PreprocessImageSliced`.
class PassthroughDecodedImagePreprocessor : public lm::ImagePreprocessor {
 public:
  explicit PassthroughDecodedImagePreprocessor(lm::DecodedImage decoded)
      : decoded_(std::move(decoded)) {}

  absl::StatusOr<lm::DecodedImage> Decode(
      absl::string_view /*image_bytes*/) const override {
    return std::move(decoded_);
  }

 private:
  mutable lm::DecodedImage decoded_;
};

absl::Status PrepareThreeInputBuffers(
    lm::MiniCpmVSlice& slice, absl::Span<TensorBuffer> vision_input_buffers,
    const MiniCpmOImageStage::InputIndices& input_indices, int patch_size,
    int padded_num_patches) {
  if (slice.num_patches > padded_num_patches) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MiniCPM-V slice has %d patches > fused signature capacity %d.",
        slice.num_patches, padded_num_patches));
  }
  if (slice.num_patches != padded_num_patches) {
    lm::PadSliceToSignatureLength(slice, padded_num_patches, patch_size);
  }

  const lm::MiniCpmVPackedSlice packed =
      lm::PackSlicePatches(slice, patch_size);
  LITERT_RETURN_IF_ERROR(
      vision_input_buffers[input_indices.images].Write<float>(
          absl::MakeConstSpan(packed.images)));
  LITERT_RETURN_IF_ERROR(
      vision_input_buffers[input_indices.positions_xy].Write<int32_t>(
          absl::MakeConstSpan(packed.positions_xy)));
  LITERT_RETURN_IF_ERROR(
      vision_input_buffers[input_indices.vit_positions].Write<int32_t>(
          absl::MakeConstSpan(packed.vit_positions)));
  return absl::OkStatus();
}

absl::Status PrepareTwoInputBuffers(
    lm::MiniCpmVSlice& slice, absl::Span<TensorBuffer> vision_input_buffers,
    const MiniCpmOImageStage::InputIndices& input_indices, int patch_size,
    int padded_num_patches) {
  constexpr int kPatchesPerSide = lm::kMiniCpmVNumPatchesPerSide;
  constexpr int32_t kResamplerPosMultiplier = 10000;

  if (slice.num_patches > padded_num_patches) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MiniCPM-V slice has %d patches > fused signature capacity %d.",
        slice.num_patches, padded_num_patches));
  }
  if (slice.num_patches != padded_num_patches) {
    lm::PadSliceToSignatureLength(slice, padded_num_patches, patch_size);
  }

  std::vector<int32_t> position_ids(padded_num_patches, -1);
  const int tgt_w = slice.tgt_w > 0 ? slice.tgt_w : 1;
  for (int i = 0; i < padded_num_patches; ++i) {
    const int64_t vit_id = slice.position_ids[i];
    if (vit_id >= 0) {
      const int patch_col = i % tgt_w;
      const int patch_row = i / tgt_w;
      const int32_t resampler_pos_id = patch_row * kPatchesPerSide + patch_col;
      position_ids[i] = resampler_pos_id * kResamplerPosMultiplier +
                        static_cast<int32_t>(vit_id);
    }
  }

  LITERT_RETURN_IF_ERROR(
      vision_input_buffers[input_indices.images].Write<float>(
          absl::MakeConstSpan(slice.strip)));
  LITERT_RETURN_IF_ERROR(
      vision_input_buffers[input_indices.positions_xy].Write<int32_t>(
          absl::MakeConstSpan(position_ids)));
  return absl::OkStatus();
}

absl::Status PrepareSingleInputBuffer(
    lm::MiniCpmVSlice& slice, absl::Span<TensorBuffer> vision_input_buffers,
    const MiniCpmOImageStage::InputIndices& input_indices, int patch_size,
    int padded_num_patches) {
  if (slice.num_patches > padded_num_patches) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MiniCPM-V slice has %d patches > fused signature capacity %d.",
        slice.num_patches, padded_num_patches));
  }
  if (slice.num_patches != padded_num_patches) {
    lm::PadSliceToSignatureLength(slice, padded_num_patches, patch_size);
  }
  LITERT_RETURN_IF_ERROR(
      vision_input_buffers[input_indices.images].Write<float>(
          absl::MakeConstSpan(slice.strip)));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> RunVisionSlice(
    lm::MiniCpmVSlice& slice, absl::Span<TensorBuffer> vision_input_buffers,
    const MiniCpmOImageStage::InputIndices& input_indices, int patch_size,
    int padded_num_patches, Any2AnyImageStage::RunEncoderFn run_encoder) {
  if (vision_input_buffers.size() >= 3) {
    ABSL_RETURN_IF_ERROR(PrepareThreeInputBuffers(slice, vision_input_buffers,
                                                  input_indices, patch_size,
                                                  padded_num_patches));
  } else if (vision_input_buffers.size() == 2) {
    ABSL_RETURN_IF_ERROR(PrepareTwoInputBuffers(slice, vision_input_buffers,
                                                input_indices, patch_size,
                                                padded_num_patches));
  } else {
    ABSL_RETURN_IF_ERROR(PrepareSingleInputBuffer(slice, vision_input_buffers,
                                                  input_indices, patch_size,
                                                  padded_num_patches));
  }
  return run_encoder();
}

struct MiniCpmOVisionEncodeParams {
  lm::MiniCpmVSliceConfig slice_config;
  MiniCpmOImageStage::InputIndices input_indices;
  int hidden_dim = 0;
  int num_output_tokens = 0;
  int padded_num_patches = 0;
};

absl::StatusOr<lm::MiniCpmVSliced> SliceImage(
    const Any2AnyImageStage::RgbImageView& image,
    const lm::MiniCpmVSliceConfig& slice_config) {
  lm::DecodedImage decoded{
      .pixels = std::vector<uint8_t>(image.pixels.begin(), image.pixels.end()),
      .width = image.width,
      .height = image.height,
  };
  PassthroughDecodedImagePreprocessor passthrough(std::move(decoded));
  ABSL_ASSIGN_OR_RETURN(
      lm::MiniCpmVSliced sliced,
      lm::PreprocessImageSliced("", passthrough, slice_config));
  if (sliced.slices.empty()) {
    return absl::InternalError("PreprocessImageSliced produced no slices.");
  }
  return sliced;
}

absl::StatusOr<Any2AnyImageEmbeddings> EncodeMiniCpmOImage(
    const MiniCpmOVisionEncodeParams& params,
    const Any2AnyImageStage::RgbImageView& image,
    absl::Span<TensorBuffer> buffers,
    Any2AnyImageStage::RunEncoderFn run_encoder) {
  ABSL_ASSIGN_OR_RETURN(lm::MiniCpmVSliced sliced,
                        SliceImage(image, params.slice_config));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> overview_emb,
      RunVisionSlice(sliced.slices[0], buffers, params.input_indices,
                     params.slice_config.patch_size, params.padded_num_patches,
                     run_encoder));

  const bool has_grid = sliced.grid_x > 0 && sliced.grid_y > 0 &&
                        static_cast<int>(sliced.slices.size()) >=
                            1 + sliced.grid_x * sliced.grid_y;
  if (!has_grid) {
    return Any2AnyImageEmbeddings{
        .embeddings = std::move(overview_emb),
        .num_tokens = params.num_output_tokens,
        .tokens_per_slice = params.num_output_tokens,
    };
  }

  const int num_slices = 1 + sliced.grid_x * sliced.grid_y;
  std::vector<float> all_embeddings;
  all_embeddings.reserve(static_cast<size_t>(num_slices) *
                         params.num_output_tokens * params.hidden_dim);
  all_embeddings.insert(all_embeddings.end(), overview_emb.begin(),
                        overview_emb.end());
  for (int slice_idx = 1; slice_idx < num_slices; ++slice_idx) {
    ABSL_ASSIGN_OR_RETURN(
        std::vector<float> slice_emb,
        RunVisionSlice(sliced.slices[slice_idx], buffers, params.input_indices,
                       params.slice_config.patch_size,
                       params.padded_num_patches, run_encoder));
    all_embeddings.insert(all_embeddings.end(), slice_emb.begin(),
                          slice_emb.end());
  }
  return Any2AnyImageEmbeddings{
      .embeddings = std::move(all_embeddings),
      .num_tokens = num_slices * params.num_output_tokens,
      .tokens_per_slice = params.num_output_tokens,
      .grid_x = sliced.grid_x,
      .grid_y = sliced.grid_y,
  };
}

absl::StatusOr<Any2AnyImageStage::EncodeImageFn> InitMiniCpmOVisionEncoder(
    const MiniCpmOModelConfig& config,
    MiniCpmOImageStage::InputIndices input_indices,
    absl::Span<const TensorBuffer> vision_input_buffers, int hidden_dim,
    int num_output_tokens) {
  if (input_indices.images >= vision_input_buffers.size() ||
      (vision_input_buffers.size() >= 2 &&
       input_indices.positions_xy >= vision_input_buffers.size()) ||
      (vision_input_buffers.size() >= 3 &&
       input_indices.vit_positions >= vision_input_buffers.size())) {
    return absl::InvalidArgumentError(
        "MiniCpmOImageStage input_indices exceed vision_input_buffers size.");
  }

  const size_t patch_bytes = static_cast<size_t>(3) * config.vision_patch_size *
                             config.vision_patch_size * sizeof(float);
  LITERT_ASSIGN_OR_RETURN(size_t vision_in_bytes,
                          vision_input_buffers[input_indices.images].Size());
  if (vision_in_bytes == 0 || vision_in_bytes % patch_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MiniCpmOImageStage vision image input buffer size (", vision_in_bytes,
        " bytes) must be a positive multiple of 3 * vision_patch_size^2 * "
        "sizeof(float) (",
        patch_bytes, " bytes)."));
  }
  const int padded_num_patches =
      static_cast<int>(vision_in_bytes / patch_bytes);

  lm::MiniCpmVSliceConfig slice_config;
  slice_config.scale_resolution = config.vision_scale_resolution;
  slice_config.patch_size = config.vision_patch_size;
  slice_config.max_slice_nums = config.vision_max_slice_nums;
  const MiniCpmOVisionEncodeParams params{
      .slice_config = slice_config,
      .input_indices = input_indices,
      .hidden_dim = hidden_dim,
      .num_output_tokens = num_output_tokens,
      .padded_num_patches = padded_num_patches,
  };
  return absl::bind_front(EncodeMiniCpmOImage, params);
}

MiniCpmOEmbeddingSegment BuildImageSegment(Any2AnyImageEmbeddings encoded) {
  MiniCpmOEmbeddingSegment segment;
  if (encoded.num_tokens == 0) {
    return segment;
  }

  const int tokens_per_slice = encoded.tokens_per_slice;
  const bool has_grid = encoded.grid_x > 0 && encoded.grid_y > 0;
  const int num_sub_slices = has_grid ? encoded.grid_x * encoded.grid_y : 0;
  const size_t total_token_ids =
      static_cast<size_t>(tokens_per_slice + 2) * (1 + num_sub_slices) +
      (has_grid ? static_cast<size_t>(encoded.grid_y - 1) : 0);
  segment.token_ids.reserve(total_token_ids);

  segment.token_ids.push_back(minicpm_o::kImageStartTokenId);
  segment.token_ids.insert(segment.token_ids.end(), tokens_per_slice,
                           minicpm_o::kImageEmbedPlaceholderTokenId);
  segment.token_ids.push_back(minicpm_o::kImageEndTokenId);

  if (has_grid) {
    for (int gy = 0; gy < encoded.grid_y; ++gy) {
      for (int gx = 0; gx < encoded.grid_x; ++gx) {
        segment.token_ids.push_back(minicpm_o::kSliceStartTokenId);
        segment.token_ids.insert(segment.token_ids.end(), tokens_per_slice,
                                 minicpm_o::kImageEmbedPlaceholderTokenId);
        segment.token_ids.push_back(minicpm_o::kSliceEndTokenId);
      }
      if (gy + 1 < encoded.grid_y) {
        segment.token_ids.push_back(minicpm_o::kImageSliceRowBreakTokenId);
      }
    }
  }

  segment.embeddings = std::move(encoded.embeddings);
  segment.num_tokens = encoded.num_tokens;
  return segment;
}

}  // namespace

MiniCpmOImageStage::MiniCpmOImageStage(
    std::unique_ptr<Any2AnyImageStage> absl_nonnull image_stage)
    : image_stage_(std::move(image_stage)) {}

MiniCpmOImageStage::~MiniCpmOImageStage() = default;

absl::StatusOr<std::unique_ptr<MiniCpmOImageStage>> MiniCpmOImageStage::Create(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    const MiniCpmOModelConfig& config, InputIndices input_indices) {
  if (config.vision_patch_size <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOImageStage config.vision_patch_size must be positive.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Any2AnyImageStage> image_stage,
      Any2AnyImageStage::Create(
          image_queue, std::move(vision_encoder_runner),
          Any2AnyImageStage::Options{
              .llm_hidden_size = config.llm_hidden_size,
          },
          absl::bind_front(InitMiniCpmOVisionEncoder, std::cref(config),
                           input_indices)));

  return absl::WrapUnique(new MiniCpmOImageStage(std::move(image_stage)));
}

bool MiniCpmOImageStage::NeedScheduleInternal() const {
  return image_stage_->NeedSchedule() || image_stage_->HasOutput();
}

absl::Status MiniCpmOImageStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!NeedScheduleInternal()) {
    return absl::NotFoundError("Image queue has no output.");
  }
  absl::Status schedule_status = absl::OkStatus();
  if (image_stage_->NeedSchedule()) {
    schedule_status = image_stage_->Schedule();
  }
  while (image_stage_->HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(Sequenced<Any2AnyImageEmbeddings> encoded,
                          image_stage_->GetOutput());
    PushOutput(Sequenced<MiniCpmOEmbeddingSegment>{
        .seqnum = encoded.seqnum,
        .item = BuildImageSegment(std::move(encoded.item)),
    });
  }
  return schedule_status;
}

void MiniCpmOImageStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  image_stage_->Reset();
}

}  // namespace litert::omni::any2any
