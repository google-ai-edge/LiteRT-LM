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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_IMAGE_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_IMAGE_STAGE_H_

#include <cstddef>
#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_image_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {

// Image input processing stage for MiniCPM-o:
// - Wraps `Any2AnyImageStage` to consume `Sequenced<ImageInput>` from
//   `Any2AnyInputFanOutStage`, decode/validate RGB888 images, slice images
//   using MiniCPM-V dynamic high-resolution grid slicing, and run the vision
//   encoder model on each slice.
// - Frames the emitted vision embeddings into
//   `Sequenced<MiniCpmOEmbeddingSegment>` with `<image>...</image>`,
//   `<slice>...</slice>`, placeholder (`-1`), and newline token IDs for
//   downstream reordering and chat-template framing in `MiniCpmOFrontendStage`.
class MiniCpmOImageStage
    : public SingleThreadedStageWithDeque<Sequenced<MiniCpmOEmbeddingSegment>> {
 public:
  // Input buffer tensor indices for the MiniCPM-o 2-input or 3-input vision
  // encoder model.
  struct InputIndices {
    // Index of the flattened image patch strip tensor (`float32`).
    size_t images = 0;
    // Index of the 2D target patch grid coordinates tensor (`int32`) in 3-input
    // models, or the combined `resampler_pos_id * 10000 + vit_pos_id`
    // `position_ids` tensor (`int32`) in 2-input models.
    size_t positions_xy = 1;
    // Index of the 2D ViT positional embedding coordinates tensor (`int32`) in
    // 3-input models.
    size_t vit_positions = 2;
  };

  // Creates a `MiniCpmOImageStage` that consumes sequenced image inputs, slices
  // images using MiniCPM-V dynamic high-resolution slicing, encodes each slice
  // with `vision_encoder_runner`, and outputs
  // `Sequenced<MiniCpmOEmbeddingSegment>`.
  //
  // Args:
  //   image_queue: Non-null output queue from `Any2AnyInputFanOutStage`
  //     providing `Sequenced<ImageInput>` items.
  //   vision_encoder_runner: Non-null runner executing the SigLIP2 vision
  //     encoder + resampler model. Owned exclusively by this stage and invoked
  //     only from this stage's single execution thread.
  //   config: Model configuration specifying `llm_hidden_size` and vision
  //     slicing parameters.
  //   input_indices: Input tensor indices for `vision_encoder_runner` when
  //     using a 3-input signature (`images`, `positions_xy`, `vit_positions`).
  //
  // Returns:
  //   A newly initialized `MiniCpmOImageStage` on success, or an error status
  //   if `config.llm_hidden_size <= 0`, if `config.vision_patch_size <= 0`, if
  //   creating input/output buffers fails, if any buffer list is empty, if
  //   output buffer byte size is not a positive multiple of
  //   `config.llm_hidden_size * sizeof(float)`, if `input_indices` exceed
  //   `vision_input_buffers.size()`, or if image input buffer size is not a
  //   positive multiple of `3 * vision_patch_size^2 * sizeof(float)`.
  static absl::StatusOr<std::unique_ptr<MiniCpmOImageStage>> Create(
      FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
          image_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
      const MiniCpmOModelConfig& config = {},
      InputIndices input_indices = DefaultInputIndices());

  ~MiniCpmOImageStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  static InputIndices DefaultInputIndices() { return InputIndices{}; }

  explicit MiniCpmOImageStage(
      std::unique_ptr<Any2AnyImageStage> absl_nonnull image_stage);

  std::unique_ptr<Any2AnyImageStage> image_stage_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_IMAGE_STAGE_H_
