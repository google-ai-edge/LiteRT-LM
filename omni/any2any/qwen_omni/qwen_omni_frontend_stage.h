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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_FRONTEND_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_FRONTEND_STAGE_H_

#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/any2any/qwen_omni/qwen_omni_text_stage.h"
#include "omni/base/fan_in_stage.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Frontend stage for Qwen2.5-Omni:
// - Reorders `Sequenced<QwenOmniEmbeddingSegment>` items from parallel modality
//   stages (`QwenOmniTextStage`, optional audio stage, and optional image
//   stage) in contiguous `seqnum` order via an internal `FanInStage`.
// - Prepends ChatML turn headers (`<|im_start|>system...` on the first turn and
//   `<|im_start|>user\n` at the start of each user turn) via
//   `QwenOmniTextStage::EmbedControlText()`.
// - Frames audio segments with `<|audio_bos|>...<|audio_eos|>` and vision
//   segments with `<|vision_bos|>...<|vision_eos|>`, matching
//   `chat_template.json` (no trailing newline after markers).
// - Aggregates all segments belonging to the same user turn into a single
//   turn-level prompt buffer and emits `Any2AnyFrontendOutput` when a turn is
//   closed (`EndOfInputTurn`, `EndOfInput`, or `Flush()`).
class QwenOmniFrontendStage
    : public SingleThreadedStageWithDeque<Any2AnyFrontendOutput> {
 public:
  // Constructs a `QwenOmniFrontendStage` that reorders sequenced embedding
  // segments from `text_stage`, optional `audio_stage`, and optional
  // `image_stage` via `FanInStage`, frames ChatML turns and audio/vision
  // boundary tokens, and emits a turn-level `Any2AnyFrontendOutput` when a
  // user turn completes.
  //
  // Args:
  //   text_stage: Non-null upstream `QwenOmniTextStage` producing
  //     `Sequenced<QwenOmniEmbeddingSegment>` items and providing control-token
  //     embedding lookup.
  //   audio_stage: Optional upstream audio modality stage producing
  //     `Sequenced<QwenOmniEmbeddingSegment>` items.
  //   image_stage: Optional upstream vision modality stage producing
  //     `Sequenced<QwenOmniEmbeddingSegment>` items.
  //   config: Model configuration specifying hidden dimensions, special token
  //     IDs, system prompt, and audio generation settings.
  explicit QwenOmniFrontendStage(
      QwenOmniTextStage* absl_nonnull text_stage,
      SingleThreadedStageWithDeque<
          Sequenced<QwenOmniEmbeddingSegment>>* absl_nullable audio_stage =
          nullptr,
      SingleThreadedStageWithDeque<
          Sequenced<QwenOmniEmbeddingSegment>>* absl_nullable image_stage =
          nullptr,
      const QwenOmniModelConfig& config = {});

  ~QwenOmniFrontendStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  absl::Status FlushInternal() override;
  void ResetInternal() override;

 private:
  absl::StatusOr<const QwenOmniEmbeddingSegment* absl_nonnull>
  GetOrEmbedBoundary(absl::string_view boundary_control_text,
                     std::optional<QwenOmniEmbeddingSegment>& cache);
  absl::StatusOr<const QwenOmniEmbeddingSegment* absl_nonnull>
  GetOrEmbedTurnSuffix();
  absl::Status EnsureTurnStarted();
  absl::Status AppendFramedModalitySegment(
      const QwenOmniEmbeddingSegment& segment, absl::string_view modality,
      int bos_token_id, int eos_token_id, int placeholder_token_id,
      absl::string_view boundary_control_text,
      std::optional<QwenOmniEmbeddingSegment>& boundary_cache);
  absl::Status CloseOpenTurn();
  absl::Status ProcessSegment(const QwenOmniEmbeddingSegment& segment);
  void AppendRawSegment(const QwenOmniEmbeddingSegment& segment);
  void FlushTurn();
  void ResetTurn();

  QwenOmniTextStage& text_stage_;
  FanInStage<QwenOmniEmbeddingSegment> fan_in_;
  QwenOmniModelConfig config_;
  int hidden_dim_;
  std::vector<float> prompt_embeddings_;
  std::vector<int> token_ids_;
  bool generate_audio_ = false;
  bool is_first_turn_ = true;
  bool in_turn_ = false;

  // Lazily cached constant control-token embeddings reused across chunks/turns
  // to avoid repeated `EmbedControlText()` runner dispatches and cross-stage
  // synchronization.
  std::optional<QwenOmniEmbeddingSegment> cached_user_turn_prefix_;
  std::optional<QwenOmniEmbeddingSegment> cached_audio_boundary_;
  std::optional<QwenOmniEmbeddingSegment> cached_vision_boundary_;
  std::optional<QwenOmniEmbeddingSegment> cached_turn_suffix_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_FRONTEND_STAGE_H_
