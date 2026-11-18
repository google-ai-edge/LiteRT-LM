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

#include "omni/any2any/any2any_llm_stage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::omni::any2any {
namespace {

absl::StatusOr<std::vector<float>> ReadTensorBufferAsFloatVector(
    const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  const ElementType elem_type = tensor_type.ElementType();
  if (elem_type == ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(std::vector<float> out,
                            support::CopyFromTensorBuffer<float>(buffer));
    return out;
  }
  if (elem_type == ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(
        std::vector<tflite::half> fp16,
        support::CopyFromTensorBuffer<tflite::half>(buffer));
    return std::vector<float>(fp16.begin(), fp16.end());
  }
  return absl::InvalidArgumentError(
      "Unsupported TensorBuffer element type; expected Float32 or Float16.");
}

struct DecodeStepOutput {
  std::vector<float> logits;
  std::vector<float> hidden_state;
};

// Splits a raw decode output buffer into `[llm_vocab_size]` logits and an
// optional trailing `[llm_hidden_size]` last-layer hidden state vector.
absl::StatusOr<DecodeStepOutput> SplitDecodeOutput(std::vector<float> raw_out,
                                                   int vocab_size,
                                                   int hidden_dim) {
  if (raw_out.empty()) {
    return absl::InternalError("LLM Decode returned an empty TensorBuffer.");
  }
  DecodeStepOutput step_out;
  const int raw_len = static_cast<int>(raw_out.size());
  if (raw_len == vocab_size + hidden_dim) {
    step_out.hidden_state.assign(raw_out.begin() + vocab_size, raw_out.end());
    raw_out.resize(vocab_size);
  } else if (raw_len != vocab_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("LLM Decode output size (", raw_len,
                     ") does not match llm_vocab_size (", vocab_size,
                     ") or llm_vocab_size + llm_hidden_size (",
                     vocab_size + hidden_dim, ")."));
  }
  step_out.logits = std::move(raw_out);
  return step_out;
}

absl::StatusOr<DecodeStepOutput> RunDecodeWithToken(LiteRtLmRunner& llm_runner,
                                                    TensorBuffer& token_buf,
                                                    int32_t input_token,
                                                    int vocab_size,
                                                    int hidden_dim) {
  LITERT_RETURN_IF_ERROR(
      token_buf.Write<int32_t>(absl::MakeConstSpan(&input_token, 1)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs decode_inputs,
                        CreateExecutorInputsWithText(token_buf));
  ABSL_ASSIGN_OR_RETURN(TensorBuffer decode_buf,
                        llm_runner.Decode(decode_inputs));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> raw_out,
                        ReadTensorBufferAsFloatVector(decode_buf));
  return SplitDecodeOutput(std::move(raw_out), vocab_size, hidden_dim);
}

absl::StatusOr<DecodeStepOutput> RunDecodeWithEmbedding(
    LiteRtLmRunner& llm_runner, TensorBuffer& emb_buf,
    absl::Span<const float> token_emb, int vocab_size, int hidden_dim) {
  LITERT_RETURN_IF_ERROR(emb_buf.Write<float>(token_emb));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs decode_inputs,
                        CreateExecutorInputsWithAudio(emb_buf));
  ABSL_ASSIGN_OR_RETURN(TensorBuffer decode_buf,
                        llm_runner.Decode(decode_inputs));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> raw_out,
                        ReadTensorBufferAsFloatVector(decode_buf));
  return SplitDecodeOutput(std::move(raw_out), vocab_size, hidden_dim);
}

absl::Status DecodeTokenDiscardOutput(LiteRtLmRunner& llm_runner,
                                      TensorBuffer& token_buf,
                                      int32_t input_token) {
  LITERT_RETURN_IF_ERROR(
      token_buf.Write<int32_t>(absl::MakeConstSpan(&input_token, 1)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs decode_inputs,
                        CreateExecutorInputsWithText(token_buf));
  return llm_runner.Decode(decode_inputs).status();
}

bool CompareCandidateWeightDesc(const std::pair<double, int>& a,
                                const std::pair<double, int>& b) {
  return a.first > b.first;
}

// Selects the first `+Inf` logit if present, else the argmax among finite
// logits, or `0` if all logits are `NaN` or `-Inf`.
int SelectGreedyToken(absl::Span<const float> logits) {
  int best_idx = 0;
  float best_val = -std::numeric_limits<float>::infinity();
  bool found_finite = false;
  for (size_t i = 0; i < logits.size(); ++i) {
    const float v = logits[i];
    if (std::isinf(v) && v > 0.0f) {
      return static_cast<int>(i);
    }
    if (std::isfinite(v) && (!found_finite || v > best_val)) {
      best_val = v;
      best_idx = static_cast<int>(i);
      found_finite = true;
    }
  }
  return best_idx;
}

// Selects the next token ID from `logits` using greedy argmax when
// `!options.do_sample || options.temperature <= 0.0f`, or temperature-scaled
// top-k and nucleus (top-p) sampling otherwise, guarding against `NaN`/`±Inf`.
int SampleToken(absl::Span<const float> logits,
                const Any2AnyLlmStage::Options& options, std::mt19937_64& rng) {
  if (!options.do_sample || options.temperature <= 0.0f) {
    return SelectGreedyToken(logits);
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  bool found_finite = false;
  for (size_t i = 0; i < logits.size(); ++i) {
    const float v = logits[i];
    if (std::isinf(v) && v > 0.0f) {
      return static_cast<int>(i);
    }
    if (std::isfinite(v) && (!found_finite || v > max_logit)) {
      max_logit = v;
      found_finite = true;
    }
  }
  if (!found_finite) {
    return 0;
  }

  const double temp = std::max<double>(options.temperature, 1e-6);
  std::vector<std::pair<double, int>> candidates;
  candidates.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    const double p = std::isfinite(logits[i])
                         ? std::exp((logits[i] - max_logit) / temp)
                         : 0.0;
    candidates.push_back({p, static_cast<int>(i)});
  }

  const int k =
      (options.top_k > 0)
          ? std::min(options.top_k, static_cast<int>(candidates.size()))
          : static_cast<int>(candidates.size());
  absl::c_partial_sort(candidates, candidates.begin() + k,
                       CompareCandidateWeightDesc);
  candidates.resize(k);

  double sum = 0.0;
  for (const auto& c : candidates) {
    sum += c.first;
  }
  if (!std::isfinite(sum) || sum <= 0.0) {
    return candidates.front().second;
  }

  if (options.top_p > 0.0f && options.top_p < 1.0f && candidates.size() > 1) {
    double cumulative = 0.0;
    size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
      cumulative += candidates[keep].first / sum;
      if (cumulative >= options.top_p) {
        ++keep;
        break;
      }
    }
    candidates.resize(std::max<size_t>(1, keep));
  }

  std::vector<double> weights;
  weights.reserve(candidates.size());
  for (const auto& c : candidates) {
    weights.push_back(c.first);
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return candidates[dist(rng)].second;
}

absl::Status ValidateFrontendPrompt(const Any2AnyFrontendOutput& prompt,
                                    int hidden_dim) {
  if (prompt.prompt_len <= 0 || prompt.prompt_embeddings.empty()) {
    return absl::InvalidArgumentError(
        "Any2AnyFrontendOutput has empty prompt_embeddings.");
  }
  const size_t expected_size =
      static_cast<size_t>(prompt.prompt_len) * hidden_dim;
  if (prompt.prompt_embeddings.size() != expected_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Any2AnyFrontendOutput prompt_embeddings size (",
                     prompt.prompt_embeddings.size(),
                     ") does not match prompt_len * llm_hidden_size (",
                     expected_size, ")."));
  }
  return absl::OkStatus();
}

absl::Status AppendHiddenStateOrFail(const DecodeStepOutput& step,
                                     std::vector<float>& dst) {
  if (step.hidden_state.empty()) {
    return absl::FailedPreconditionError(
        "Any2AnyLlmStage requires hidden states from llm_runner when "
        "generate_audio is enabled.");
  }
  dst.insert(dst.end(), step.hidden_state.begin(), step.hidden_state.end());
  return absl::OkStatus();
}

// Prefills the full prompt embedding sequence in a single `Prefill` call.
absl::Status PrefillPromptBatch(LiteRtLmRunner& llm_runner,
                                const Any2AnyFrontendOutput& prompt,
                                int hidden_dim) {
  LITERT_ASSIGN_OR_RETURN(
      auto prefill_buf,
      support::CreateTensorBuffer<float>({1, prompt.prompt_len, hidden_dim}));
  LITERT_RETURN_IF_ERROR(
      prefill_buf.Write<float>(absl::MakeConstSpan(prompt.prompt_embeddings)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs prefill_inputs,
                        CreateExecutorInputsWithAudio(prefill_buf));
  return llm_runner.Prefill(prefill_inputs);
}

// Prefills the prompt token-by-token so the runner emits last-layer hidden
// states for every prompt position, returning logits after the final token.
absl::StatusOr<std::vector<float>> PrefillPromptWithHiddenStates(
    LiteRtLmRunner& llm_runner, TensorBuffer& token_buf,
    const Any2AnyFrontendOutput& prompt, int vocab_size, int hidden_dim,
    std::vector<float>& prompt_hidden_states) {
  const size_t dim = static_cast<size_t>(hidden_dim);
  prompt_hidden_states.reserve(prompt.prompt_embeddings.size());

  LITERT_ASSIGN_OR_RETURN(
      auto emb_1_buf, support::CreateTensorBuffer<float>({1, 1, hidden_dim}));
  absl::Span<const float> all_prompt_emb(prompt.prompt_embeddings);

  // Token 0: 1-token Prefill stages token 0 as pending input, and Decode(`-1`)
  // runs `decode` at position 0 and returns `[logits_0, prompt_hidden_0]`.
  LITERT_RETURN_IF_ERROR(
      emb_1_buf.Write<float>(all_prompt_emb.subspan(0, dim)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs prefill_0_inputs,
                        CreateExecutorInputsWithAudio(emb_1_buf));
  ABSL_RETURN_IF_ERROR(llm_runner.Prefill(prefill_0_inputs));
  ABSL_ASSIGN_OR_RETURN(
      DecodeStepOutput step_0,
      RunDecodeWithToken(llm_runner, token_buf, /*input_token=*/-1, vocab_size,
                         hidden_dim));
  ABSL_RETURN_IF_ERROR(AppendHiddenStateOrFail(step_0, prompt_hidden_states));
  std::vector<float> last_logits = std::move(step_0.logits);

  for (int i = 1; i < prompt.prompt_len; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        DecodeStepOutput step_i,
        RunDecodeWithEmbedding(
            llm_runner, emb_1_buf,
            all_prompt_emb.subspan(static_cast<size_t>(i) * dim, dim),
            vocab_size, hidden_dim));
    ABSL_RETURN_IF_ERROR(AppendHiddenStateOrFail(step_i, prompt_hidden_states));
    last_logits = std::move(step_i.logits);
  }
  return last_logits;
}

absl::Status FeedTurnEndTokens(LiteRtLmRunner& llm_runner,
                               TensorBuffer& token_buf,
                               absl::Span<const int> turn_end_token_ids) {
  for (int token_id : turn_end_token_ids) {
    ABSL_RETURN_IF_ERROR(DecodeTokenDiscardOutput(
        llm_runner, token_buf, static_cast<int32_t>(token_id)));
  }
  return absl::OkStatus();
}

// Commits any un-decoded final assistant token (when `max_new_tokens` was
// reached without hitting a stop token) followed by the assistant turn-closing
// tokens (`<|im_end|>\n`) into `llm_runner`'s KV cache.
absl::Status CommitAssistantTurnEnd(LiteRtLmRunner& llm_runner,
                                    TensorBuffer& token_buf,
                                    bool stopped_on_eos,
                                    int32_t last_sampled_token,
                                    absl::Span<const int> turn_end_token_ids) {
  if (!stopped_on_eos && last_sampled_token >= 0) {
    ABSL_RETURN_IF_ERROR(
        DecodeTokenDiscardOutput(llm_runner, token_buf, last_sampled_token));
  }
  return FeedTurnEndTokens(llm_runner, token_buf, turn_end_token_ids);
}

// Runs `HiddenStateMode::kPreToken` autoregressive decoding (pairing each
// emitted token with the hidden state from the step that predicted it) and
// commits the assistant turn end into the KV cache.
absl::Status DecodePreTokenLoop(LiteRtLmRunner& llm_runner,
                                TensorBuffer& token_buf,
                                const Any2AnyLlmStage::Options& options,
                                const absl::flat_hash_set<int>& stop_token_ids,
                                absl::Span<const int> turn_end_token_ids,
                                bool generate_audio, std::mt19937_64& rng,
                                Any2AnyLlmOutput& out) {
  int32_t current_token = -1;
  const int max_steps = std::max(1, options.max_new_tokens);
  bool stopped_on_eos = false;
  for (int step = 0; step < max_steps; ++step) {
    ABSL_ASSIGN_OR_RETURN(
        DecodeStepOutput step_out,
        RunDecodeWithToken(llm_runner, token_buf, current_token,
                           options.llm_vocab_size, options.llm_hidden_size));
    const int token_id = SampleToken(step_out.logits, options, rng);
    if (stop_token_ids.contains(token_id)) {
      stopped_on_eos = true;
      break;
    }
    if (generate_audio) {
      ABSL_RETURN_IF_ERROR(
          AppendHiddenStateOrFail(step_out, out.llm_hidden_states));
    }
    out.generated_token_ids.push_back(token_id);
    ++out.num_tokens;
    current_token = token_id;
  }
  return CommitAssistantTurnEnd(llm_runner, token_buf, stopped_on_eos,
                                current_token, turn_end_token_ids);
}

// Runs `HiddenStateMode::kPostTokenWithPrompt` autoregressive decoding (pairing
// each emitted token with the hidden state from the step where that token is
// the input) and feeds the assistant turn-closing tokens into the KV cache.
absl::Status DecodePostTokenLoop(LiteRtLmRunner& llm_runner,
                                 TensorBuffer& token_buf,
                                 std::vector<float> last_logits,
                                 const Any2AnyLlmStage::Options& options,
                                 const absl::flat_hash_set<int>& stop_token_ids,
                                 absl::Span<const int> turn_end_token_ids,
                                 bool generate_audio, std::mt19937_64& rng,
                                 Any2AnyLlmOutput& out) {
  const int max_steps = std::max(1, options.max_new_tokens);
  for (int step = 0; step < max_steps; ++step) {
    const int token_id = SampleToken(last_logits, options, rng);
    if (stop_token_ids.contains(token_id)) {
      break;
    }
    out.generated_token_ids.push_back(token_id);
    ++out.num_tokens;

    ABSL_ASSIGN_OR_RETURN(
        DecodeStepOutput next_step,
        RunDecodeWithToken(llm_runner, token_buf,
                           static_cast<int32_t>(token_id),
                           options.llm_vocab_size, options.llm_hidden_size));
    if (generate_audio) {
      ABSL_RETURN_IF_ERROR(
          AppendHiddenStateOrFail(next_step, out.llm_hidden_states));
    }
    last_logits = std::move(next_step.logits);
  }
  return FeedTurnEndTokens(llm_runner, token_buf, turn_end_token_ids);
}

}  // namespace

Any2AnyLlmStage::Any2AnyLlmStage(
    Stage<Any2AnyFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner, Options options)
    : frontend_(*frontend),
      options_(std::move(options)),
      tokenizer_(std::move(tokenizer)),
      llm_runner_(std::move(llm_runner)),
      hidden_dim_(options_.llm_hidden_size) {
  rng_.seed(options_.seed.value_or(42));
}

Any2AnyLlmStage::~Any2AnyLlmStage() = default;

absl::StatusOr<std::unique_ptr<Any2AnyLlmStage>> Any2AnyLlmStage::Create(
    Stage<Any2AnyFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner, Options options) {
  if (options.llm_hidden_size <= 0 || options.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "Any2AnyLlmStage options.llm_hidden_size and options.llm_vocab_size "
        "must be positive.");
  }

  auto stage = absl::WrapUnique(
      new Any2AnyLlmStage(frontend, std::move(tokenizer), std::move(llm_runner),
                          std::move(options)));

  stage->stop_token_ids_.insert(stage->options_.default_im_end_token_id);
  stage->stop_token_ids_.insert(stage->options_.default_eos_token_id);
  for (const std::string& stop_str : stage->options_.stop_token_strings) {
    if (auto tok_id = stage->tokenizer_->TokenToId(stop_str); tok_id.ok()) {
      stage->stop_token_ids_.insert(*tok_id);
    }
  }

  int im_end_id = stage->options_.default_im_end_token_id;
  if (auto tok_id = stage->tokenizer_->TokenToId("<|im_end|>"); tok_id.ok()) {
    im_end_id = *tok_id;
  }
  stage->turn_end_token_ids_.push_back(im_end_id);
  if (auto nl_ids = stage->tokenizer_->TextToTokenIds("\n");
      nl_ids.ok() && !nl_ids->empty()) {
    stage->turn_end_token_ids_.insert(stage->turn_end_token_ids_.end(),
                                      nl_ids->begin(), nl_ids->end());
  } else if (auto nl_id = stage->tokenizer_->TokenToId("\n"); nl_id.ok()) {
    stage->turn_end_token_ids_.push_back(*nl_id);
  }

  return stage;
}

bool Any2AnyLlmStage::NeedScheduleInternal() const {
  return frontend_.HasOutput();
}

absl::Status Any2AnyLlmStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_ASSIGN_OR_RETURN(Any2AnyFrontendOutput prompt, frontend_.GetOutput());
  ABSL_RETURN_IF_ERROR(ValidateFrontendPrompt(prompt, hidden_dim_));

  const bool generate_audio =
      prompt.generate_audio && options_.enable_audio_conditioning;
  Any2AnyLlmOutput out;
  out.generate_audio = generate_audio;
  out.is_final = true;

  LITERT_ASSIGN_OR_RETURN(auto token_buf,
                          support::CreateTensorBuffer<int32_t>({1, 1}));

  if (options_.hidden_state_mode == HiddenStateMode::kPreToken) {
    ABSL_RETURN_IF_ERROR(PrefillPromptBatch(*llm_runner_, prompt, hidden_dim_));
    ABSL_RETURN_IF_ERROR(
        DecodePreTokenLoop(*llm_runner_, token_buf, options_, stop_token_ids_,
                           turn_end_token_ids_, generate_audio, rng_, out));
  } else {
    std::vector<float> last_logits;
    if (generate_audio) {
      out.prompt_token_ids = prompt.token_ids;
      ABSL_ASSIGN_OR_RETURN(
          last_logits,
          PrefillPromptWithHiddenStates(*llm_runner_, token_buf, prompt,
                                        options_.llm_vocab_size, hidden_dim_,
                                        out.prompt_hidden_states));
    } else {
      ABSL_RETURN_IF_ERROR(
          PrefillPromptBatch(*llm_runner_, prompt, hidden_dim_));
      ABSL_ASSIGN_OR_RETURN(
          DecodeStepOutput first_step,
          RunDecodeWithToken(*llm_runner_, token_buf, /*input_token=*/-1,
                             options_.llm_vocab_size, hidden_dim_));
      last_logits = std::move(first_step.logits);
    }
    ABSL_RETURN_IF_ERROR(DecodePostTokenLoop(
        *llm_runner_, token_buf, std::move(last_logits), options_,
        stop_token_ids_, turn_end_token_ids_, generate_audio, rng_, out));
  }

  if (options_.detokenize_output && !out.generated_token_ids.empty()) {
    ABSL_ASSIGN_OR_RETURN(std::string raw_text,
                          tokenizer_->TokenIdsToText(out.generated_token_ids));
    out.text = std::string(absl::StripLeadingAsciiWhitespace(raw_text));
  }
  PushOutput(std::move(out));
  return absl::OkStatus();
}

void Any2AnyLlmStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  rng_.seed(options_.seed.value_or(42));
  (void)llm_runner_->Reset();
}

}  // namespace litert::omni::any2any
