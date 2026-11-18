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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_IMAGE_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_IMAGE_STAGE_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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

namespace litert::omni::any2any {

// Vision modality stage for Qwen-Omni (`Qwen2.5-Omni-3B`):
// - Wraps `Any2AnyImageStage` to consume `Sequenced<ImageInput>` items from
//   `Any2AnyInputFanOutStage`.
// - Resizes decoded RGB888 images using PIL-compatible bicubic resampling to
//   `[image_size, image_size]` (skipping resizing when dimensions already
//   match) and normalizes directly into NCHW `float32[1, 3, image_size,
//   image_size]` using Qwen-VL CLIP mean and std
//   (`{0.48145466, 0.4578275, 0.40821073}` and
//   `{0.26862954, 0.26130258, 0.27577711}`).
// - Runs the `tf_lite_vision_encoder` runner (`Qwen2_5OmniVisionEncoder` with
//   `Qwen2_5_VisionPatchMerger` 2x2 spatial merge) and frames the emitted
//   embeddings into `Sequenced<QwenOmniEmbeddingSegment>` with `<|vision_bos|>`
//   (`151652`), `<|IMAGE|>` (`151655`), and `<|vision_eos|>` (`151653`)
//   markers.
class QwenOmniImageStage
    : public SingleThreadedStageWithDeque<Sequenced<QwenOmniEmbeddingSegment>> {
 public:
  // Creates and initializes a `QwenOmniImageStage` from a `LiteRtRunner`
  // executing `tf_lite_vision_encoder`.
  //
  // Args:
  //   image_queue: Non-null fan-out queue providing `Sequenced<ImageInput>`
  //     items from `Any2AnyInputFanOutStage`.
  //   vision_encoder_runner: Non-null runner executing
  //     `tf_lite_vision_encoder`.
  //   config: Qwen-Omni model configuration specifying `llm_hidden_size`,
  //     `vision_patch_size`, `vision_spatial_merge_size`, and vision
  //     boundary/placeholder token IDs.
  //
  // Returns:
  //   A newly initialized `QwenOmniImageStage` instance, or an `absl::Status`
  //   error if config parameters, tensor element type/shape, or buffer creation
  //   fail.
  static absl::StatusOr<std::unique_ptr<QwenOmniImageStage>> Create(
      FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
          image_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
      const QwenOmniModelConfig& config = {});

  ~QwenOmniImageStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  QwenOmniImageStage(
      std::unique_ptr<Any2AnyImageStage> absl_nonnull image_stage,
      const QwenOmniModelConfig& config);

  QwenOmniEmbeddingSegment BuildImageSegment(
      Any2AnyImageEmbeddings encoded) const;

  std::unique_ptr<Any2AnyImageStage> image_stage_;
  QwenOmniModelConfig config_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_IMAGE_STAGE_H_
