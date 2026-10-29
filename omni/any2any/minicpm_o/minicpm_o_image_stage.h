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
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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

// Image input processing stage for MiniCPM-o:
// - Consumes `Sequenced<ImageInput>` from `MiniCpmOInputFanOutStage`.
// - Slices images using MiniCPM-V dynamic high-resolution grid slicing.
// - Runs the vision encoder model on each slice.
// - Outputs `Sequenced<MiniCpmOEmbeddingSegment>` containing the continuous
//   vision query embeddings and `<image>...</image>`, `<slice>...</slice>`,
//   placeholder (`-1`), and newline token IDs for downstream reordering and
//   chat-template framing in `MiniCpmOFrontendStage`.
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
  //   image_queue: Non-null output queue from `MiniCpmOInputFanOutStage`
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
  //   if `config.llm_hidden_size <= 0`, if creating input/output buffers fails,
  //   if any buffer list is empty, if output buffer byte size is not a positive
  //   multiple of `config.llm_hidden_size * sizeof(float)`, or if
  //   `input_indices` exceed `vision_input_buffers.size()`.
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

 private:
  static InputIndices DefaultInputIndices() { return InputIndices{}; }

  MiniCpmOImageStage(
      FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
          image_queue,
      std::unique_ptr<lm::ImagePreprocessor> absl_nullable image_preprocessor,
      std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
      std::vector<TensorBuffer> vision_input_buffers,
      std::vector<TensorBuffer> vision_output_buffers,
      const MiniCpmOModelConfig& config, InputIndices input_indices);

  // Slices `image` using MiniCPM-V dynamic high-resolution slicing and runs the
  // vision encoder on the overview and sub-slices.
  //
  // Args:
  //   image: Raw RGB/RGBA/grayscale pixel buffer or encoded image bytes.
  //
  // Returns:
  //   A `MiniCpmOEmbeddingSegment` containing the continuous vision query
  //   embeddings and `<image>...</image>`, `<slice>...</slice>`, placeholder
  //   (`-1`), and newline token IDs, or an error status if decoding, slicing,
  //   or model execution fails.
  absl::StatusOr<MiniCpmOEmbeddingSegment> ProcessImage(
      const ImageInput& image);

  // Decodes or converts `image` into RGB pixels and slices it into an overview
  // plus optional high-resolution sub-slices.
  absl::StatusOr<lm::MiniCpmVSliced> SliceImage(const ImageInput& image);

  // Formats and pads `slice` to match the vision encoder signature, executes
  // `vision_encoder_runner_`, and reads the resulting query embeddings.
  //
  // Args:
  //   slice: Preprocessed MiniCPM-V image slice strip and patch positions
  //     (padded in-place if the model signature expects more patches).
  //
  // Returns:
  //   Flattened vision query embeddings of size `num_queries * hidden_dim_`,
  //   or an error status if buffer I/O or runner execution fails.
  absl::StatusOr<std::vector<float>> RunVisionSlice(lm::MiniCpmVSlice& slice);

  // Populates `vision_input_buffers_` for the 3-input MiniCPM-V vision encoder
  // signature (`images`, `positions_xy`, `vit_positions`).
  absl::Status PrepareThreeInputBuffers(lm::MiniCpmVSlice& slice);

  // Populates `vision_input_buffers_` for the 2-input MiniCPM-o vision encoder
  // signature (`pixel_values` strip and `position_ids` encoding
  // `resampler_pos_id * 10000 + vit_pos_id`).
  absl::Status PrepareTwoInputBuffers(lm::MiniCpmVSlice& slice);

  // Populates `vision_input_buffers_` for the 1-input vision encoder signature
  // (`pixel_values` strip).
  absl::Status PrepareSingleInputBuffer(lm::MiniCpmVSlice& slice);

  FanOutOutputQueue<OmniSession::Input, ImageInput>& image_queue_;
  MiniCpmOModelConfig config_;
  int hidden_dim_;

  std::unique_ptr<lm::ImagePreprocessor> image_preprocessor_;
  std::unique_ptr<LiteRtRunner> vision_encoder_runner_;
  std::vector<TensorBuffer> vision_input_buffers_;
  std::vector<TensorBuffer> vision_output_buffers_;
  InputIndices input_indices_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_IMAGE_STAGE_H_
