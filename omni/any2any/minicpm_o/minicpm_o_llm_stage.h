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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_LLM_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_LLM_STAGE_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Output of Stage 2 (MiniCpmOLlmStage): generated text and conditioning
// representations for the downstream TTS decoder stage.
struct MiniCpmOLlmOutput {
  // Detokenized assistant text response generated for the turn.
  std::string text;
  // Sequence of generated LLM token IDs (excluding any stop token).
  std::vector<int> generated_token_ids;
  // Flattened hidden states for the generated tokens in LLM hidden dimension
  // space with shape `[num_tokens, llm_hidden_size]`.
  std::vector<float> llm_hidden_states;
  // Total number of generated tokens represented in `generated_token_ids` and
  // `llm_hidden_states`.
  int num_tokens = 0;
  // Whether downstream TTS and vocoder stages should synthesize speech audio
  // for this output.
  bool generate_audio = true;
  // Whether this output marks the final LLM chunk for the current turn.
  bool is_final = true;
};

// Stage 2: Autoregressive Multimodal LLM Backbone Stage (Qwen3-8B in
// MiniCPM-o 4.5). Consumes multimodal prompt embeddings from
// `MiniCpmOFrontendStage`, runs prefill and token decoding via
// `LiteRtLmRunner`, detokenizes the assistant text response, and collects
// per-token hidden states for downstream TTS audio generation.
class MiniCpmOLlmStage
    : public SingleThreadedStageWithDeque<MiniCpmOLlmOutput> {
 public:
  // Creates a `MiniCpmOLlmStage` with the given upstream frontend stage,
  // tokenizer, and stateful LLM backbone runner.
  //
  // Args:
  //   frontend: Non-null upstream stage producing `MiniCpmOFrontendOutput`
  //     prompt embeddings.
  //   tokenizer: Non-null tokenizer used to resolve stop token IDs and
  //     detokenize generated token IDs into text.
  //   llm_runner: Non-null stateful `LiteRtLmRunner` for KV-cached prefill and
  //     decode execution.
  //   config: Model configuration specifying hidden dimensions, vocabulary
  //     size, and sampling parameters.
  //
  // Returns:
  //   A newly initialized `MiniCpmOLlmStage` on success, or an error status if
  //   `config.llm_hidden_size <= 0` or `config.llm_vocab_size <= 0`.
  static absl::StatusOr<std::unique_ptr<MiniCpmOLlmStage>> Create(
      Stage<MiniCpmOFrontendOutput>* absl_nonnull frontend,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
      const MiniCpmOModelConfig& config = {});

  ~MiniCpmOLlmStage() override;

 protected:
  void ResetInternal() override;

  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  MiniCpmOLlmStage(Stage<MiniCpmOFrontendOutput>* absl_nonnull frontend,
                   std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
                   std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
                   MiniCpmOModelConfig config);

  Stage<MiniCpmOFrontendOutput>& frontend_;
  MiniCpmOModelConfig config_;
  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtLmRunner> llm_runner_;

  absl::flat_hash_set<int> stop_token_ids_;
  std::vector<int> turn_end_token_ids_;
  int hidden_dim_ = minicpm_o::kLlmHiddenSize;
  std::mt19937_64 rng_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_LLM_STAGE_H_
