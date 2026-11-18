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

#include "omni/any2any/qwen_omni/qwen_omni_image_stage.h"

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
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/any2any_image_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/preprocessor/pil_resize.h"

namespace litert::omni::any2any {
namespace {

constexpr float kQwenClipMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float kQwenClipStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};

absl::StatusOr<int> ValidateVisionInputBuffer(
    const TensorBuffer& buffer, const QwenOmniModelConfig& config) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  if (tensor_type.ElementType() != ElementType::Float32) {
    return absl::InvalidArgumentError(
        "QwenOmniImageStage vision input buffer element type must be Float32.");
  }
  const Layout& layout = tensor_type.Layout();
  const auto dims = layout.Dimensions();
  if (layout.Rank() != 4 || dims[0] != 1 || dims[1] != 3 || dims[2] <= 0 ||
      dims[2] != dims[3]) {
    return absl::InvalidArgumentError(
        "QwenOmniImageStage vision input buffer must have rank-4 NCHW shape "
        "[1, 3, image_size, image_size] with positive square spatial "
        "dimensions.");
  }

  const int image_size = static_cast<int>(dims[2]);
  const int64_t patch_unit = static_cast<int64_t>(config.vision_patch_size) *
                             config.vision_spatial_merge_size;
  if (image_size % patch_unit != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("QwenOmniImageStage vision input image_size (", image_size,
                     ") must be divisible by vision_patch_size * "
                     "vision_spatial_merge_size (",
                     patch_unit, ")."));
  }

  LITERT_ASSIGN_OR_RETURN(const size_t num_elements, layout.NumElements());
  LITERT_ASSIGN_OR_RETURN(const size_t packed_size, buffer.PackedSize());
  if (packed_size != num_elements * sizeof(float)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "QwenOmniImageStage vision input buffer packed size (", packed_size,
        " bytes) does not match [1, 3, ", image_size, ", ", image_size,
        "] Float32 (", num_elements * sizeof(float), " bytes)."));
  }
  return image_size;
}

absl::Status WriteNormalizedNchw(const Any2AnyImageStage::RgbImageView& image,
                                 int image_size, TensorBuffer& input_buffer) {
  std::vector<uint8_t> resized_rgb;
  const uint8_t* rgb_bytes = image.pixels.data();
  if (image.width != image_size || image.height != image_size) {
    resized_rgb = lm::PilResizeBicubicRgb(image.pixels.data(), image.width,
                                          image.height, image_size, image_size);
    rgb_bytes = resized_rgb.data();
  }

  const float scale[3] = {
      1.0f / (255.0f * kQwenClipStd[0]),
      1.0f / (255.0f * kQwenClipStd[1]),
      1.0f / (255.0f * kQwenClipStd[2]),
  };
  const float bias[3] = {
      -kQwenClipMean[0] / kQwenClipStd[0],
      -kQwenClipMean[1] / kQwenClipStd[1],
      -kQwenClipMean[2] / kQwenClipStd[2],
  };

  const size_t plane_size = static_cast<size_t>(image_size) * image_size;
  LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                          TensorBufferScopedLock::Create<float>(
                              input_buffer, TensorBuffer::LockMode::kWrite));
  float* dst_r = lock_and_addr.second;
  float* dst_g = dst_r + plane_size;
  float* dst_b = dst_g + plane_size;
  for (size_t i = 0; i < plane_size; ++i) {
    const size_t src_idx = i * 3;
    dst_r[i] = rgb_bytes[src_idx] * scale[0] + bias[0];
    dst_g[i] = rgb_bytes[src_idx + 1] * scale[1] + bias[1];
    dst_b[i] = rgb_bytes[src_idx + 2] * scale[2] + bias[2];
  }
  return absl::OkStatus();
}

absl::StatusOr<Any2AnyImageEmbeddings> EncodeQwenOmniImage(
    int image_size, int num_output_tokens,
    const Any2AnyImageStage::RgbImageView& image,
    absl::Span<TensorBuffer> buffers,
    Any2AnyImageStage::RunEncoderFn run_encoder) {
  ABSL_RETURN_IF_ERROR(WriteNormalizedNchw(image, image_size, buffers[0]));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> vision_embeds, run_encoder());
  return Any2AnyImageEmbeddings{
      .embeddings = std::move(vision_embeds),
      .num_tokens = num_output_tokens,
      .tokens_per_slice = num_output_tokens,
  };
}

absl::StatusOr<Any2AnyImageStage::EncodeImageFn> InitQwenOmniVisionEncoder(
    const QwenOmniModelConfig& config,
    absl::Span<const TensorBuffer> input_buffers, int /*hidden_dim*/,
    int num_output_tokens) {
  ABSL_ASSIGN_OR_RETURN(const int image_size,
                        ValidateVisionInputBuffer(input_buffers[0], config));
  return absl::bind_front(EncodeQwenOmniImage, image_size, num_output_tokens);
}

}  // namespace

QwenOmniImageStage::QwenOmniImageStage(
    std::unique_ptr<Any2AnyImageStage> absl_nonnull image_stage,
    const QwenOmniModelConfig& config)
    : image_stage_(std::move(image_stage)), config_(config) {}

QwenOmniImageStage::~QwenOmniImageStage() = default;

absl::StatusOr<std::unique_ptr<QwenOmniImageStage>> QwenOmniImageStage::Create(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    const QwenOmniModelConfig& config) {
  if (config.vision_patch_size <= 0) {
    return absl::InvalidArgumentError(
        "QwenOmniImageStage config.vision_patch_size must be positive.");
  }
  if (config.vision_spatial_merge_size <= 0) {
    return absl::InvalidArgumentError(
        "QwenOmniImageStage config.vision_spatial_merge_size must be "
        "positive.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Any2AnyImageStage> image_stage,
      Any2AnyImageStage::Create(
          image_queue, std::move(vision_encoder_runner),
          Any2AnyImageStage::Options{
              .llm_hidden_size = config.llm_hidden_size,
          },
          absl::bind_front(InitQwenOmniVisionEncoder, std::cref(config))));

  return absl::WrapUnique(
      new QwenOmniImageStage(std::move(image_stage), config));
}

bool QwenOmniImageStage::NeedScheduleInternal() const {
  return image_stage_->NeedSchedule() || image_stage_->HasOutput();
}

absl::Status QwenOmniImageStage::ScheduleInternal() {
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
    PushOutput(Sequenced<QwenOmniEmbeddingSegment>{
        .seqnum = encoded.seqnum,
        .item = BuildImageSegment(std::move(encoded.item)),
    });
  }
  return schedule_status;
}

void QwenOmniImageStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  image_stage_->Reset();
}

QwenOmniEmbeddingSegment QwenOmniImageStage::BuildImageSegment(
    Any2AnyImageEmbeddings encoded) const {
  QwenOmniEmbeddingSegment segment;
  if (encoded.num_tokens == 0) {
    return segment;
  }
  segment.modality = QwenOmniEmbeddingSegment::Modality::kVision;
  segment.token_ids.reserve(static_cast<size_t>(encoded.num_tokens) + 2);
  segment.token_ids.push_back(config_.vision_bos_token_id);
  segment.token_ids.insert(segment.token_ids.end(), encoded.num_tokens,
                           config_.image_token_id);
  segment.token_ids.push_back(config_.vision_eos_token_id);
  segment.embeddings = std::move(encoded.embeddings);
  segment.num_tokens = encoded.num_tokens;
  return segment;
}

}  // namespace litert::omni::any2any
