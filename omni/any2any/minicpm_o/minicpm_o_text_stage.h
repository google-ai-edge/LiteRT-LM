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

// Text and turn-control embedding stage for MiniCPM-o:
// - Consumes `Sequenced<TextInput>`, `Sequenced<EndOfInputTurn>`, and
//   `Sequenced<EndOfInput>` from `MiniCpmOInputFanOutStage` in ascending
//   `seqnum` order among its input queues.
// - Tokenizes user prompt text via `Tokenizer::TextToTokenIds()` (without
//   parsing special control tokens in untrusted user text) and embeds token IDs
//   into LLM hidden dimension space via `text_embedder_runner`.
// - Embeds the turn-closing suffix (`<|im_end|>\n<|im_start|>assistant\n`) on
//   `EndOfInputTurn` and emits a turn-end marker on `EndOfInput`.
// - Exposes `EmbedControlText()` for `MiniCpmOFrontendStage` to embed chat
//   template headers and modality boundary tokens after `FanInStage` reorders
//   parallel modality segments.
class MiniCpmOTextStage
    : public SingleThreadedStageWithDeque<Sequenced<MiniCpmOEmbeddingSegment>> {
 public:
  // Creates a `MiniCpmOTextStage` that consumes sequenced text, end-of-turn,
  // and end-of-input items from `MiniCpmOInputFanOutStage` and embeds tokens
  // using `text_embedder_runner`.
  //
  // Args:
  //   text_queue: Non-null output queue from `MiniCpmOInputFanOutStage`
  //     providing `Sequenced<TextInput>` items.
  //   end_of_turn_queue: Non-null output queue from `MiniCpmOInputFanOutStage`
  //     providing `Sequenced<EndOfInputTurn>` items.
  //   end_of_input_queue: Non-null output queue from `MiniCpmOInputFanOutStage`
  //     providing `Sequenced<EndOfInput>` end-of-stream items.
  //   tokenizer: Non-null tokenizer used to convert prompt text and control
  //     tokens into token IDs.
  //   text_embedder_runner: Non-null runner executing the token embedding
  //     lookup model.
  //   config: Model configuration specifying `llm_hidden_size`.
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
      FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
          end_of_input_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      const MiniCpmOModelConfig& config = {});

  ~MiniCpmOTextStage() override;

  // Tokenizes `text` (recognizing MiniCPM-o special control tokens such as
  // `<|im_start|>`, `<|im_end|>`, `<|audio_start|>`, `<|audio_end|>`, etc.)
  // and embeds the resulting token IDs into LLM hidden dimension space.
  absl::StatusOr<MiniCpmOEmbeddingSegment> EmbedControlText(
      absl::string_view text);

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  MiniCpmOTextStage(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
          end_of_input_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      std::vector<TensorBuffer> text_emb_input_buffers,
      std::vector<TensorBuffer> text_emb_output_buffers,
      const MiniCpmOModelConfig& config);

  // Tokenizes plain user prompt text and projects to LLM embeddings.
  absl::StatusOr<MiniCpmOEmbeddingSegment> ProcessText(absl::string_view text);

  // Emits the turn-closing chat template suffix
  // (`<|im_end|>\n<|im_start|>assistant\n`) and marks the segment with
  // `is_turn_end = true`.
  absl::StatusOr<MiniCpmOEmbeddingSegment> ProcessEndOfTurn();

  FanOutOutputQueue<OmniSession::Input, TextInput>& text_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>& end_of_turn_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInput>& end_of_input_queue_;
  MiniCpmOModelConfig config_;
  int hidden_dim_;

  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::unique_ptr<LiteRtRunner> text_embedder_runner_;
  std::vector<TensorBuffer> text_emb_input_buffers_;
  std::vector<TensorBuffer> text_emb_output_buffers_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_TEXT_STAGE_H_
