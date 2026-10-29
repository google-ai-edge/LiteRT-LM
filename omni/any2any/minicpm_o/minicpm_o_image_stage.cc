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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
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

// Passthrough ImagePreprocessor used when `ImageInput` already provides raw
// interleaved RGB888 pixels (`width > 0 && height > 0`).
class PassthroughDecodedImagePreprocessor : public lm::ImagePreprocessor {
 public:
  PassthroughDecodedImagePreprocessor(std::vector<uint8_t> pixels, int width,
                                      int height)
      : pixels_(std::move(pixels)), width_(width), height_(height) {}

  absl::StatusOr<lm::DecodedImage> Decode(
      absl::string_view /*image_bytes*/) const override {
    return lm::DecodedImage{
        .pixels = std::move(pixels_),
        .width = width_,
        .height = height_,
    };
  }

 private:
  mutable std::vector<uint8_t> pixels_;
  int width_;
  int height_;
};

// Pads a MiniCPM-V slice strip to `padded_num_patches` patches of edge
// `patch_size`.
void PadSliceToSignatureLength(lm::MiniCpmVSlice& slice, int padded_num_patches,
                               int patch_size) {
  const int old_w = slice.num_patches * patch_size;
  const int new_w = padded_num_patches * patch_size;
  std::vector<float> wide(static_cast<size_t>(3) * patch_size * new_w, 0.0f);
  for (int row = 0; row < 3 * patch_size; ++row) {
    absl::c_copy(absl::MakeConstSpan(slice.strip)
                     .subspan(static_cast<size_t>(row) * old_w, old_w),
                 wide.begin() + static_cast<size_t>(row) * new_w);
  }
  slice.strip = std::move(wide);
  slice.num_patches = padded_num_patches;
  slice.position_ids.resize(padded_num_patches, -1);
}

// Converts raw RGB (3-channel), RGBA (4-channel), or grayscale (1-channel)
// `image.image_bytes` into interleaved RGB888 pixels.
absl::StatusOr<std::vector<uint8_t>> ConvertRawImageToRgb(
    const ImageInput& image) {
  const size_t num_pixels = static_cast<size_t>(image.width) * image.height;
  if (image.channels == 3 && image.image_bytes.size() == num_pixels * 3) {
    return image.image_bytes;
  }
  if (image.channels == 4 && image.image_bytes.size() == num_pixels * 4) {
    std::vector<uint8_t> rgb_pixels;
    rgb_pixels.reserve(num_pixels * 3);
    for (size_t i = 0; i < image.image_bytes.size(); i += 4) {
      rgb_pixels.push_back(image.image_bytes[i]);
      rgb_pixels.push_back(image.image_bytes[i + 1]);
      rgb_pixels.push_back(image.image_bytes[i + 2]);
    }
    return rgb_pixels;
  }
  if (image.channels == 1 && image.image_bytes.size() == num_pixels) {
    std::vector<uint8_t> rgb_pixels;
    rgb_pixels.reserve(num_pixels * 3);
    for (uint8_t g : image.image_bytes) {
      rgb_pixels.push_back(g);
      rgb_pixels.push_back(g);
      rgb_pixels.push_back(g);
    }
    return rgb_pixels;
  }
  return absl::InvalidArgumentError(
      "ImageInput dimensions/channels do not match image_bytes size.");
}

// Appends `<image>...</image>` or `<slice>...</slice>` tokens and vision query
// embeddings to `segment`.
void AppendVisionSliceTokensAndEmbeddings(absl::Span<const float> slice_emb,
                                          int hidden_dim, int start_token_id,
                                          int end_token_id,
                                          MiniCpmOEmbeddingSegment& segment) {
  const int slice_tokens = static_cast<int>(slice_emb.size()) / hidden_dim;
  segment.token_ids.push_back(start_token_id);
  segment.token_ids.insert(segment.token_ids.end(), slice_tokens,
                           minicpm_o::kImageEmbedPlaceholderTokenId);
  segment.token_ids.push_back(end_token_id);
  segment.embeddings.insert(segment.embeddings.end(), slice_emb.begin(),
                            slice_emb.end());
  segment.num_tokens += slice_tokens;
}

}  // namespace

MiniCpmOImageStage::MiniCpmOImageStage(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<lm::ImagePreprocessor> absl_nullable image_preprocessor,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    std::vector<TensorBuffer> vision_input_buffers,
    std::vector<TensorBuffer> vision_output_buffers,
    const MiniCpmOModelConfig& config, InputIndices input_indices)
    : image_queue_(*image_queue),
      config_(config),
      hidden_dim_(config.llm_hidden_size),
      image_preprocessor_(std::move(image_preprocessor)),
      vision_encoder_runner_(std::move(vision_encoder_runner)),
      vision_input_buffers_(std::move(vision_input_buffers)),
      vision_output_buffers_(std::move(vision_output_buffers)),
      input_indices_(input_indices) {}

MiniCpmOImageStage::~MiniCpmOImageStage() = default;

absl::StatusOr<std::unique_ptr<MiniCpmOImageStage>> MiniCpmOImageStage::Create(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    const MiniCpmOModelConfig& config, InputIndices input_indices) {
  if (config.llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOImageStage config.llm_hidden_size must be positive.");
  }
  if (config.vision_patch_size <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOImageStage config.vision_patch_size must be positive.");
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> vision_input_buffers,
                        vision_encoder_runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> vision_output_buffers,
                        vision_encoder_runner->CreateOutputBuffers(""));
  if (vision_input_buffers.empty() || vision_output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "MiniCpmOImageStage vision_encoder_runner must provide non-empty "
        "input and output buffers.");
  }
  if (vision_input_buffers.size() >= 3 &&
      (input_indices.images >= vision_input_buffers.size() ||
       input_indices.positions_xy >= vision_input_buffers.size() ||
       input_indices.vit_positions >= vision_input_buffers.size())) {
    return absl::InvalidArgumentError(
        "MiniCpmOImageStage input_indices exceed vision_input_buffers size.");
  }
  if (vision_input_buffers.size() == 2 &&
      (input_indices.images >= vision_input_buffers.size() ||
       input_indices.positions_xy >= vision_input_buffers.size())) {
    return absl::InvalidArgumentError(
        "MiniCpmOImageStage input_indices exceed vision_input_buffers size.");
  }

  const size_t hidden_bytes =
      static_cast<size_t>(config.llm_hidden_size) * sizeof(float);
  LITERT_ASSIGN_OR_RETURN(size_t vision_out_bytes,
                          vision_output_buffers[0].Size());
  if (vision_out_bytes == 0 || vision_out_bytes % hidden_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MiniCpmOImageStage vision output buffer size (", vision_out_bytes,
        " bytes) must be a positive multiple of llm_hidden_size * "
        "sizeof(float) (",
        hidden_bytes, " bytes)."));
  }

  return absl::WrapUnique(new MiniCpmOImageStage(
      image_queue, lm::ImagePreprocessor::Create(),
      std::move(vision_encoder_runner), std::move(vision_input_buffers),
      std::move(vision_output_buffers), config, input_indices));
}

absl::Status MiniCpmOImageStage::PrepareThreeInputBuffers(
    lm::MiniCpmVSlice& slice) {
  const size_t img_idx = input_indices_.images;
  const size_t pos_idx = input_indices_.positions_xy;
  const size_t vit_idx = input_indices_.vit_positions;

  const int patch_size = config_.vision_patch_size;
  constexpr int kPatchesPerSide = lm::kMiniCpmVNumPatchesPerSide;
  const int patch_dim = 3 * patch_size * patch_size;

  LITERT_ASSIGN_OR_RETURN(size_t img_bytes,
                          vision_input_buffers_[img_idx].Size());
  const int padded_num_patches =
      std::max(1, static_cast<int>(img_bytes / (sizeof(float) * patch_dim)));
  if (slice.num_patches > padded_num_patches) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MiniCPM-V slice has %d patches > fused signature capacity %d.",
        slice.num_patches, padded_num_patches));
  }
  if (slice.num_patches != padded_num_patches) {
    PadSliceToSignatureLength(slice, padded_num_patches, patch_size);
  }

  std::vector<float> images(static_cast<size_t>(padded_num_patches) * patch_dim,
                            0.0f);
  const int strip_w = padded_num_patches * patch_size;
  for (int p = 0; p < padded_num_patches; ++p) {
    for (int ch = 0; ch < 3; ++ch) {
      for (int py = 0; py < patch_size; ++py) {
        for (int px = 0; px < patch_size; ++px) {
          const int col = p * patch_size + px;
          images[(static_cast<size_t>(p) * patch_dim) +
                 ((ch * patch_size + py) * patch_size + px)] =
              slice
                  .strip[(static_cast<size_t>(ch) * patch_size + py) * strip_w +
                         col];
        }
      }
    }
  }

  std::vector<int32_t> positions_xy(static_cast<size_t>(padded_num_patches) * 2,
                                    -1);
  std::vector<int32_t> vit_positions(
      static_cast<size_t>(padded_num_patches) * 2, -1);
  const int tgt_w = slice.tgt_w > 0 ? slice.tgt_w : 1;
  for (int i = 0; i < padded_num_patches; ++i) {
    const int64_t id = slice.position_ids[i];
    if (id >= 0) {
      positions_xy[static_cast<size_t>(i) * 2 + 0] = i % tgt_w;
      positions_xy[static_cast<size_t>(i) * 2 + 1] = i / tgt_w;
      vit_positions[static_cast<size_t>(i) * 2 + 0] =
          static_cast<int32_t>(id % kPatchesPerSide);
      vit_positions[static_cast<size_t>(i) * 2 + 1] =
          static_cast<int32_t>(id / kPatchesPerSide);
    }
  }

  LITERT_RETURN_IF_ERROR(
      vision_input_buffers_[img_idx].Write<float>(absl::MakeConstSpan(images)));
  LITERT_RETURN_IF_ERROR(vision_input_buffers_[pos_idx].Write<int32_t>(
      absl::MakeConstSpan(positions_xy)));
  LITERT_RETURN_IF_ERROR(vision_input_buffers_[vit_idx].Write<int32_t>(
      absl::MakeConstSpan(vit_positions)));
  return absl::OkStatus();
}

absl::Status MiniCpmOImageStage::PrepareTwoInputBuffers(
    lm::MiniCpmVSlice& slice) {
  const size_t img_idx = input_indices_.images;
  const size_t pos_idx = input_indices_.positions_xy;

  const int patch_size = config_.vision_patch_size;
  constexpr int kPatchesPerSide = lm::kMiniCpmVNumPatchesPerSide;
  constexpr int32_t kResamplerPosMultiplier = 10000;
  const int patch_dim = 3 * patch_size * patch_size;

  LITERT_ASSIGN_OR_RETURN(size_t img_bytes,
                          vision_input_buffers_[img_idx].Size());
  const int padded_num_patches =
      std::max(1, static_cast<int>(img_bytes / (sizeof(float) * patch_dim)));
  if (slice.num_patches > padded_num_patches) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MiniCPM-V slice has %d patches > fused signature capacity %d.",
        slice.num_patches, padded_num_patches));
  }
  if (slice.num_patches != padded_num_patches) {
    PadSliceToSignatureLength(slice, padded_num_patches, patch_size);
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

  const size_t in_floats = img_bytes / sizeof(float);
  if (slice.strip.size() == in_floats) {
    LITERT_RETURN_IF_ERROR(vision_input_buffers_[img_idx].Write<float>(
        absl::MakeConstSpan(slice.strip)));
  } else {
    std::vector<float> input_data(in_floats, 0.0f);
    const size_t copy_count = std::min(in_floats, slice.strip.size());
    absl::c_copy(absl::MakeConstSpan(slice.strip).first(copy_count),
                 input_data.begin());
    LITERT_RETURN_IF_ERROR(vision_input_buffers_[img_idx].Write<float>(
        absl::MakeConstSpan(input_data)));
  }
  LITERT_RETURN_IF_ERROR(vision_input_buffers_[pos_idx].Write<int32_t>(
      absl::MakeConstSpan(position_ids)));
  return absl::OkStatus();
}

absl::Status MiniCpmOImageStage::PrepareSingleInputBuffer(
    lm::MiniCpmVSlice& slice) {
  const int patch_size = config_.vision_patch_size;
  const int patch_dim = 3 * patch_size * patch_size;

  LITERT_ASSIGN_OR_RETURN(size_t in_bytes, vision_input_buffers_[0].Size());
  const size_t in_floats = in_bytes / sizeof(float);
  if (in_floats >= static_cast<size_t>(patch_dim) &&
      in_floats % patch_dim == 0) {
    const int padded_num_patches = static_cast<int>(in_floats / patch_dim);
    if (slice.num_patches > padded_num_patches) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MiniCPM-V slice has %d patches > fused signature capacity %d.",
          slice.num_patches, padded_num_patches));
    }
    if (slice.num_patches != padded_num_patches) {
      PadSliceToSignatureLength(slice, padded_num_patches, patch_size);
    }
  }
  if (slice.strip.size() == in_floats) {
    LITERT_RETURN_IF_ERROR(vision_input_buffers_[0].Write<float>(
        absl::MakeConstSpan(slice.strip)));
  } else {
    std::vector<float> input_data(in_floats, 0.0f);
    const size_t copy_count = std::min(in_floats, slice.strip.size());
    absl::c_copy(absl::MakeConstSpan(slice.strip).first(copy_count),
                 input_data.begin());
    LITERT_RETURN_IF_ERROR(
        vision_input_buffers_[0].Write<float>(absl::MakeConstSpan(input_data)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> MiniCpmOImageStage::RunVisionSlice(
    lm::MiniCpmVSlice& slice) {
  if (vision_input_buffers_.size() >= 3) {
    ABSL_RETURN_IF_ERROR(PrepareThreeInputBuffers(slice));
  } else if (vision_input_buffers_.size() == 2) {
    ABSL_RETURN_IF_ERROR(PrepareTwoInputBuffers(slice));
  } else {
    ABSL_RETURN_IF_ERROR(PrepareSingleInputBuffer(slice));
  }

  ABSL_RETURN_IF_ERROR(vision_encoder_runner_->Run("", vision_input_buffers_,
                                                   vision_output_buffers_));
  LITERT_ASSIGN_OR_RETURN(size_t out_bytes, vision_output_buffers_[0].Size());
  std::vector<float> out(out_bytes / sizeof(float));
  LITERT_RETURN_IF_ERROR(
      vision_output_buffers_[0].Read<float>(absl::MakeSpan(out)));
  return out;
}

absl::StatusOr<lm::MiniCpmVSliced> MiniCpmOImageStage::SliceImage(
    const ImageInput& image) {
  if (image.width < 0 || image.height < 0 ||
      (image.width > 0) != (image.height > 0)) {
    return absl::InvalidArgumentError(
        "ImageInput width and height must both be positive for raw pixels or "
        "both be zero for encoded images.");
  }

  lm::MiniCpmVSliceConfig slice_config;
  slice_config.scale_resolution = config_.vision_scale_resolution;
  slice_config.patch_size = config_.vision_patch_size;
  slice_config.max_slice_nums = config_.vision_max_slice_nums;

  lm::MiniCpmVSliced sliced;
  if (image.width > 0 && image.height > 0) {
    ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> rgb_pixels,
                          ConvertRawImageToRgb(image));
    PassthroughDecodedImagePreprocessor passthrough(std::move(rgb_pixels),
                                                    image.width, image.height);
    ABSL_ASSIGN_OR_RETURN(
        sliced, lm::PreprocessImageSliced("", passthrough, slice_config));
  } else {
    if (image_preprocessor_ == nullptr) {
      return absl::FailedPreconditionError(
          "MiniCpmOImageStage requires an ImagePreprocessor to decode "
          "compressed image bytes.");
    }
    std::string encoded_bytes(
        reinterpret_cast<const char*>(image.image_bytes.data()),
        image.image_bytes.size());
    ABSL_ASSIGN_OR_RETURN(
        sliced, lm::PreprocessImageSliced(encoded_bytes, *image_preprocessor_,
                                          slice_config));
  }

  if (sliced.slices.empty()) {
    return absl::InternalError("PreprocessImageSliced produced no slices.");
  }
  return sliced;
}

absl::StatusOr<MiniCpmOEmbeddingSegment> MiniCpmOImageStage::ProcessImage(
    const ImageInput& image) {
  if (image.image_bytes.empty()) {
    return MiniCpmOEmbeddingSegment{};
  }

  ABSL_ASSIGN_OR_RETURN(lm::MiniCpmVSliced sliced, SliceImage(image));

  MiniCpmOEmbeddingSegment segment;
  ABSL_ASSIGN_OR_RETURN(std::vector<float> overview_emb,
                        RunVisionSlice(sliced.slices[0]));
  AppendVisionSliceTokensAndEmbeddings(overview_emb, hidden_dim_,
                                       minicpm_o::kImageStartTokenId,
                                       minicpm_o::kImageEndTokenId, segment);

  if (sliced.grid_x > 0 && sliced.grid_y > 0 &&
      static_cast<int>(sliced.slices.size()) >=
          1 + sliced.grid_x * sliced.grid_y) {
    for (int gy = 0; gy < sliced.grid_y; ++gy) {
      for (int gx = 0; gx < sliced.grid_x; ++gx) {
        const int slice_idx = 1 + gy * sliced.grid_x + gx;
        ABSL_ASSIGN_OR_RETURN(std::vector<float> slice_emb,
                              RunVisionSlice(sliced.slices[slice_idx]));
        AppendVisionSliceTokensAndEmbeddings(
            slice_emb, hidden_dim_, minicpm_o::kSliceStartTokenId,
            minicpm_o::kSliceEndTokenId, segment);
      }
      if (gy + 1 < sliced.grid_y) {
        segment.token_ids.push_back(static_cast<int>('\n'));
      }
    }
  }

  return segment;
}

bool MiniCpmOImageStage::NeedScheduleInternal() const {
  return !image_queue_.Empty();
}

absl::Status MiniCpmOImageStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (image_queue_.Empty()) {
    return absl::NotFoundError("Image queue has no output.");
  }

  while (!image_queue_.Empty()) {
    ABSL_ASSIGN_OR_RETURN(const Sequenced<ImageInput>* sequenced_input,
                          image_queue_.PeekFront());
    ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment image_emb,
                          ProcessImage(sequenced_input->item));
    const uint64_t seqnum = sequenced_input->seqnum;
    ABSL_RETURN_IF_ERROR(image_queue_.PopFront().status());
    this->PushOutput(Sequenced<MiniCpmOEmbeddingSegment>{
        .seqnum = seqnum,
        .item = std::move(image_emb),
    });
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
