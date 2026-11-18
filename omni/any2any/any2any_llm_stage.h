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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_LLM_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_LLM_STAGE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_sampler.h"
#include "omni/any2any/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Concrete autoregressive multimodal LLM backbone stage shared across Any2Any
// model families:
// - Consumes `Any2AnyFrontendOutput` interleaved multimodal prompt embeddings.
// - Runs prompt prefill and autoregressive token decoding via `LiteRtLmRunner`
//   with greedy or temperature/top-k/top-p sampling and Float32/Float16 output
//   buffer decoding.
// - Extracts last-layer hidden states for downstream TTS conditioning using
//   either `HiddenStateMode::kPreToken` (MiniCPM-o) or
//   `HiddenStateMode::kPostTokenWithPrompt` (Qwen2.5-Omni).
// - Commits assistant turn-closing tokens (`<|im_end|>\n`) into the runner's
//   KV cache and optionally detokenizes generated token IDs into
//   `Any2AnyLlmOutput::text`.
class Any2AnyLlmStage : public SingleThreadedStageWithDeque<Any2AnyLlmOutput> {
 public:
  // Controls how last-layer hidden states are collected from `llm_runner` when
  // audio conditioning is enabled.
  enum class HiddenStateMode {
    // Batch-prefills the prompt and pairs each generated token
    // `generated_token_ids[j]` with the hidden state from the decode step that
    // predicted it (used by MiniCPM-o).
    kPreToken,
    // When audio conditioning is enabled, prefills the prompt token-by-token to
    // collect `prompt_hidden_states` across all `prompt_len` tokens, then pairs
    // each generated token `generated_token_ids[j]` with the hidden state from
    // the decode step where `generated_token_ids[j]` is the input (used by
    // Qwen2.5-Omni).
    kPostTokenWithPrompt,
  };

  // Configuration for `Any2AnyLlmStage`, shared across Any2Any model families
  // (e.g., MiniCPM-o, Qwen2.5-Omni).
  struct Options {
    // Hidden state dimension of the LLM backbone.
    int llm_hidden_size = 0;
    // Vocabulary size of the LLM backbone.
    int llm_vocab_size = 0;
    // Maximum number of text tokens generated per turn (clamped to >= 1).
    int max_new_tokens = 512;
    // Whether to use stochastic sampling (temperature/top-k/top-p) instead of
    // greedy argmax decoding.
    bool do_sample = false;
    // Sampling temperature (non-positive values fall back to greedy decoding).
    float temperature = 0.7f;
    // Top-k candidate count for stochastic sampling (<= 0 disables top-k).
    int top_k = 20;
    // Nucleus (top-p) cumulative probability threshold (values outside (0, 1)
    // disable nucleus truncation).
    float top_p = 0.8f;
    // Optional RNG seed for reproducible stochastic sampling (defaults to 42).
    std::optional<uint64_t> seed = 42;
    // Whether audio conditioning hidden states may be collected when
    // `Any2AnyFrontendOutput::generate_audio` is true.
    bool enable_audio_conditioning = true;
    // Hidden-state extraction alignment mode.
    HiddenStateMode hidden_state_mode = HiddenStateMode::kPreToken;
    // Fallback `<|im_end|>` token ID used if the tokenizer does not resolve
    // `"<|im_end|>"`.
    int default_im_end_token_id = 151645;
    // Fallback EOS token ID added to the stop token set.
    int default_eos_token_id = 151643;
    // Stop token strings resolved via `tokenizer->TokenToId(...)`.
    std::vector<std::string> stop_token_strings = {
        "<|im_end|>", "<|endoftext|>", "<|tts_eos|>", "<tts_eos>", "</s>",
    };
    // Whether to detokenize `generated_token_ids` into `Any2AnyLlmOutput::text`
    // (with leading ASCII whitespace stripped). Wrapper stages that strip
    // reasoning blocks (e.g., `<think>...</think>`) from `generated_token_ids`
    // before detokenizing may set this to `false` to avoid redundant
    // `Tokenizer::TokenIdsToText()` calls.
    bool detokenize_output = true;
  };

  // Creates an `Any2AnyLlmStage` with the given upstream frontend stage,
  // tokenizer, stateful LLM runner, and options.
  //
  // Returns `InvalidArgumentError` if `options.llm_hidden_size <= 0` or
  // `options.llm_vocab_size <= 0`.
  static absl::StatusOr<std::unique_ptr<Any2AnyLlmStage>> Create(
      Stage<Any2AnyFrontendOutput>* absl_nonnull frontend,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner, Options options);

  ~Any2AnyLlmStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  Any2AnyLlmStage(Stage<Any2AnyFrontendOutput>* absl_nonnull frontend,
                  std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
                  std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
                  Options options);

  Stage<Any2AnyFrontendOutput>& frontend_;
  Options options_;
  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtLmRunner> llm_runner_;
  int hidden_dim_;
  absl::flat_hash_set<int> stop_token_ids_;
  std::vector<int> turn_end_token_ids_;
  Any2AnySampler sampler_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_LLM_STAGE_H_
