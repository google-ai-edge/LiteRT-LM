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
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/any2any/minicpm_o/minicpm_o_text_stage.h"
#include "omni/base/fan_in_stage.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Frontend stage for MiniCPM-o:
// - Reorders `Sequenced<MiniCpmOEmbeddingSegment>` items from parallel modality
//   stages (`MiniCpmOTextStage` and optional `MiniCpmOAudioStage`) in
//   contiguous `seqnum` order via an internal `FanInStage`.
// - Prepends MiniCPM-o chat-template turn headers (`<|im_start|>system...` on
//   the first turn and `<|im_start|>user\n` at the start of each user turn) via
//   `MiniCpmOTextStage::EmbedControlText()`.
// - Frames audio segments with `<|audio_start|>`, `<|audio_end|>`, and trailing
//   `\n` boundary token embeddings.
// - Aggregates all segments belonging to the same user turn into a single
//   turn-level prompt buffer and emits `MiniCpmOFrontendOutput` when a turn is
//   complete (`is_turn_end = true`).
class MiniCpmOFrontendStage
    : public SingleThreadedStageWithDeque<MiniCpmOFrontendOutput> {
 public:
  // Constructs a `MiniCpmOFrontendStage` that reorders sequenced embedding
  // segments from `text_stage` and optional `audio_stage` via `FanInStage`,
  // frames chat turns and audio boundary tokens, and emits a turn-level
  // `MiniCpmOFrontendOutput` at the end of each user turn.
  //
  // Args:
  //   text_stage: Non-null upstream `MiniCpmOTextStage` producing
  //     `Sequenced<MiniCpmOEmbeddingSegment>` items and providing control-token
  //     embedding lookup.
  //   audio_stage: Optional upstream `MiniCpmOAudioStage` (or test modality
  //     stage) producing `Sequenced<MiniCpmOEmbeddingSegment>` items.
  //   config: Model configuration specifying hidden dimensions, system prompt,
  //     and audio generation settings.
  explicit MiniCpmOFrontendStage(
      MiniCpmOTextStage* absl_nonnull text_stage,
      SingleThreadedStageWithDeque<
          Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable audio_stage =
          nullptr,
      const MiniCpmOModelConfig& config = {});

  ~MiniCpmOFrontendStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  absl::Status FlushInternal() override;
  void ResetInternal() override;

 private:
  absl::Status ProcessSegment(const MiniCpmOEmbeddingSegment& segment);
  absl::StatusOr<MiniCpmOEmbeddingSegment> BuildTurnStartPrefix();
  absl::StatusOr<MiniCpmOEmbeddingSegment> FrameAudioSegment(
      const MiniCpmOEmbeddingSegment& audio);
  void AppendRawSegment(const MiniCpmOEmbeddingSegment& segment);
  void FlushTurn();
  void ResetTurn();

  MiniCpmOTextStage& text_stage_;
  FanInStage<MiniCpmOEmbeddingSegment> fan_in_;
  MiniCpmOModelConfig config_;
  int hidden_dim_;
  std::vector<float> prompt_embeddings_;
  std::vector<int> token_ids_;
  bool generate_audio_ = true;

  // Tracks whether the stage is formatting the first user turn so the system
  // prompt (`<|im_start|>system...`) is prepended only once at the start of a
  // conversation (and again after `Reset()`).
  bool is_first_turn_ = true;
  // Tracks whether the `<|im_start|>user\n` header has already been emitted for
  // the current user turn so multi-chunk turns frame all chunks inside a single
  // user block before `EndOfInputTurn`.
  bool in_turn_ = false;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_FRONTEND_STAGE_H_
