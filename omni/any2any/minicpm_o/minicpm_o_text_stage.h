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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_TEXT_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_TEXT_STAGE_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Text processing stage for MiniCPM-o.
// Consumes `Sequenced<TextInput>` and `Sequenced<EndOfInputTurn>` from FanOut,
// applies chat templating and tokenization, embeds token IDs into LLM hidden
// dimension space, and outputs `Sequenced<MiniCpmOEmbeddingSegment>`.
class MiniCpmOTextStage
    : public SingleThreadedStageWithDeque<Sequenced<MiniCpmOEmbeddingSegment>> {
 public:
  // Creates a `MiniCpmOTextStage` that consumes sequenced text and end-of-turn
  // inputs, applies MiniCPM-o chat templating, tokenizes text, and embeds token
  // IDs using `text_embedder_runner`.
  //
  // Args:
  //   text_queue: Non-null output queue from `MiniCpmOInputFanOutStage`
  //     providing `Sequenced<TextInput>` items.
  //   end_of_turn_queue: Non-null output queue from `MiniCpmOInputFanOutStage`
  //     providing `Sequenced<EndOfInputTurn>` items.
  //   tokenizer: Non-null tokenizer used to convert prompt text and control
  //     tokens into token IDs.
  //   text_embedder_runner: Non-null runner executing the token embedding
  //     lookup model.
  //   config: Model configuration specifying `llm_hidden_size` and
  //     `system_prompt`.
  //
  // Returns:
  //   A newly initialized `MiniCpmOTextStage` on success, or an error status if
  //   creating input/output buffers fails, if the buffers are empty, or if the
  //   output buffer byte size does not match `input_elements *
  //   config.llm_hidden_size * sizeof(float)`.
  static absl::StatusOr<std::unique_ptr<MiniCpmOTextStage>> Create(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      const MiniCpmOModelConfig& config = {});

  ~MiniCpmOTextStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  MiniCpmOTextStage(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      std::vector<TensorBuffer> text_emb_input_buffers,
      std::vector<TensorBuffer> text_emb_output_buffers,
      const MiniCpmOModelConfig& config);

  // Tokenizes prompt text and projects to LLM embeddings.
  absl::StatusOr<MiniCpmOEmbeddingSegment> ProcessText(absl::string_view text);

  // Emits the turn-closing chat template suffix (`<|im_end|>\n<|im_start|>assistant\n`)
  // and marks the segment with `is_turn_end = true`.
  absl::StatusOr<MiniCpmOEmbeddingSegment> ProcessEndOfTurn();

  absl::StatusOr<std::vector<float>> EmbedTokenIds(
      absl::Span<const int> token_ids);

  FanOutOutputQueue<OmniSession::Input, TextInput>& text_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>& end_of_turn_queue_;
  MiniCpmOModelConfig config_;
  int hidden_dim_;

  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::unique_ptr<LiteRtRunner> text_embedder_runner_;
  std::vector<TensorBuffer> text_emb_input_buffers_;
  std::vector<TensorBuffer> text_emb_output_buffers_;

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

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_TEXT_STAGE_H_
