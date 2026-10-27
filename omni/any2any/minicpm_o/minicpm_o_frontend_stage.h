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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_FRONTEND_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_FRONTEND_STAGE_H_

#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/fan_in_stage.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Frontend stage for MiniCPM-o that uses an internal
// `FanInStage<MiniCpmOEmbeddingSegment>` to collect and reorder
// `Sequenced<MiniCpmOEmbeddingSegment>` from all active modality stages
// (TextStage in CL 2, AudioStage in CL 3, ImageStage in CL 4) by `seqnum`,
// appends them in original input sequence order, and emits
// `MiniCpmOFrontendOutput` when a turn is complete.
class MiniCpmOFrontendStage
    : public SingleThreadedStageWithDeque<MiniCpmOFrontendOutput> {
 public:
  // Constructs a `MiniCpmOFrontendStage` that fans in sequenced embedding
  // segments from the provided modality stages in ascending `seqnum` order and
  // emits a `MiniCpmOFrontendOutput` at the end of each user turn.
  //
  // Args:
  //   text_stage: Optional upstream text embedding stage producing
  //     `Sequenced<MiniCpmOEmbeddingSegment>`.
  //   audio_stage: Optional upstream audio embedding stage producing
  //     `Sequenced<MiniCpmOEmbeddingSegment>`.
  //   image_stage: Optional upstream image embedding stage producing
  //     `Sequenced<MiniCpmOEmbeddingSegment>`.
  //   config: Model configuration specifying hidden dimensions and audio
  //     generation settings.
  explicit MiniCpmOFrontendStage(
      Stage<Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable text_stage,
      Stage<Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable
          audio_stage = nullptr,
      Stage<Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable
          image_stage = nullptr,
      const MiniCpmOModelConfig& config = {});
  ~MiniCpmOFrontendStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  absl::Status FlushInternal() override;
  void ResetInternal() override;

 private:
  void AppendSegment(MiniCpmOEmbeddingSegment segment);
  void FlushTurn();
  void ResetTurn();

  MiniCpmOModelConfig config_;
  FanInStage<MiniCpmOEmbeddingSegment> fan_in_;
  std::vector<float> prompt_embeddings_;
  std::vector<int> token_ids_;
  bool generate_audio_ = true;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_FRONTEND_STAGE_H_
