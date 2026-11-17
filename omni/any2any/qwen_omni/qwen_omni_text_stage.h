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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_TEXT_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_TEXT_STAGE_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/qwen_omni/qwen_omni_io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Text and turn-control embedding stage for Qwen2.5-Omni:
// - Consumes `Sequenced<TextInput>`, `Sequenced<EndOfInputTurn>`, and
//   `Sequenced<EndOfInput>` from `Any2AnyInputFanOutStage` in ascending
//   `seqnum` order among its input queues.
// - Tokenizes plain user text via `Tokenizer::TextToTokenIds()` (escaping
//   `<|...|>` delimiters in untrusted user text) and embeds token IDs into
//   Thinker LLM hidden dimension space via `text_embedder_runner`.
// - Exposes `EmbedControlText()` for `QwenOmniFrontendStage` to embed ChatML
//   turn headers and audio/vision boundary tokens.
class QwenOmniTextStage
    : public SingleThreadedStageWithDeque<Sequenced<QwenOmniEmbeddingSegment>> {
 public:
  // Creates a `QwenOmniTextStage` that consumes sequenced text, end-of-turn,
  // and end-of-input items from `Any2AnyInputFanOutStage` and embeds tokens
  // using `text_embedder_runner`.
  //
  // Args:
  //   text_queue: Non-null output queue from `Any2AnyInputFanOutStage`
  //     providing `Sequenced<TextInput>` items.
  //   end_of_turn_queue: Non-null output queue from `Any2AnyInputFanOutStage`
  //     providing `Sequenced<EndOfInputTurn>` items.
  //   end_of_input_queue: Non-null output queue from `Any2AnyInputFanOutStage`
  //     providing `Sequenced<EndOfInput>` end-of-stream items.
  //   tokenizer: Non-null tokenizer used to convert prompt text and control
  //     tokens into token IDs.
  //   text_embedder_runner: Non-null runner executing the token embedding
  //     lookup model.
  //   config: Model configuration specifying `llm_hidden_size`,
  //     `llm_vocab_size`, and fallback special token IDs.
  //
  // Returns:
  //   A newly initialized `QwenOmniTextStage` on success, or an error status if
  //   creating input/output buffers fails, if the buffers are empty, if
  //   `config.llm_hidden_size` is non-positive, or if the output buffer byte
  //   size does not match `input_elements * config.llm_hidden_size *
  //   sizeof(float)`.
  static absl::StatusOr<std::unique_ptr<QwenOmniTextStage>> Create(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
          end_of_input_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      const QwenOmniModelConfig& config = {});

  ~QwenOmniTextStage() override = default;

  // Tokenizes trusted control text `text` and embeds the resulting token IDs
  // into Thinker LLM hidden dimension space.
  //
  // Unlike `ProcessText()`, this method recognizes and maps Qwen2.5-Omni
  // `<|...|>` special control tokens (`<|im_start|>`, `<|im_end|>`,
  // `<|endoftext|>`, `<|AUDIO|>`, `<|audio_bos|>`, `<|audio_eos|>`,
  // `<|vision_bos|>`, `<|vision_eos|>`, `<|IMAGE|>`, `<|VIDEO|>`) rather than
  // escaping them.
  //
  // Thread safety: blocks until the stage transitions from `State::kIdle` to
  // `State::kRunning` (restoring `State::kIdle` before returning) to serialize
  // access to `tokenizer_`, `text_embedder_runner_`, and reusable buffers with
  // `Schedule()`. Must NOT be called from inside
  // `QwenOmniTextStage::ScheduleInternal()` or it will deadlock.
  //
  // Args:
  //   text: Trusted control template string to tokenize and embed.
  //
  // Returns:
  //   A `QwenOmniEmbeddingSegment` with modality `kText` containing the
  //   embedded tokens, or an error status if tokenization or embedding fails.
  absl::StatusOr<QwenOmniEmbeddingSegment> EmbedControlText(
      absl::string_view text);

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  QwenOmniTextStage(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
          end_of_input_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      std::vector<TensorBuffer> text_emb_input_buffers,
      std::vector<TensorBuffer> text_emb_output_buffers,
      const QwenOmniModelConfig& config);

  // Tokenizes and embeds `text` as control text. Callers must already hold a
  // non-idle stage state (e.g., `State::kScheduling` or `State::kRunning`) to
  // serialize access to `tokenizer_`, `text_embedder_runner_`, and buffers.
  absl::StatusOr<QwenOmniEmbeddingSegment> EmbedControlTextInternal(
      absl::string_view text);
  absl::StatusOr<std::vector<float>> EmbedTokens(
      absl::Span<const int> token_ids);
  absl::StatusOr<QwenOmniEmbeddingSegment> ProcessText(absl::string_view text);

  FanOutOutputQueue<OmniSession::Input, TextInput>& text_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>& end_of_turn_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInput>& end_of_input_queue_;
  QwenOmniModelConfig config_;
  int hidden_dim_;

  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::unique_ptr<LiteRtRunner> text_embedder_runner_;
  std::vector<TensorBuffer> text_emb_input_buffers_;
  std::vector<TensorBuffer> text_emb_output_buffers_;
  std::optional<QwenOmniEmbeddingSegment> cached_turn_suffix_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_TEXT_STAGE_H_
