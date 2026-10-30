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

#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
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

// Strips `<think>...</think>` blocks emitted by Qwen3 reasoning models from
// both the detokenized text and the generated token/hidden-state sequences so
// downstream TTS conditioning stays aligned with the spoken text.
absl::Status FinalizeLlmOutputTextAndHiddenStates(support::Tokenizer& tokenizer,
                                                  int hidden_dim,
                                                  MiniCpmOLlmOutput& out) {
  if (out.generated_token_ids.empty()) {
    return absl::OkStatus();
  }
  constexpr absl::string_view kThinkStart = "<think>";
  constexpr absl::string_view kThinkEnd = "</think>";

  int think_start_id = minicpm_o::kThinkStartTokenId;
  if (auto id = tokenizer.TokenToId(kThinkStart); id.ok()) {
    think_start_id = *id;
  }
  int think_end_id = minicpm_o::kThinkEndTokenId;
  if (auto id = tokenizer.TokenToId(kThinkEnd); id.ok()) {
    think_end_id = *id;
  }

  auto start_it = absl::c_find(out.generated_token_ids, think_start_id);
  if (start_it != out.generated_token_ids.end()) {
    auto end_it =
        std::find(start_it + 1, out.generated_token_ids.end(), think_end_id);
    if (end_it != out.generated_token_ids.end()) {
      const size_t start_idx = static_cast<size_t>(
          std::distance(out.generated_token_ids.begin(), start_it));
      const size_t end_idx = static_cast<size_t>(std::distance(
                                 out.generated_token_ids.begin(), end_it)) +
                             1;
      const size_t prev_tokens = out.generated_token_ids.size();
      out.generated_token_ids.erase(start_it, end_it + 1);
      if (hidden_dim > 0 &&
          out.llm_hidden_states.size() == prev_tokens * hidden_dim) {
        out.llm_hidden_states.erase(
            out.llm_hidden_states.begin() + start_idx * hidden_dim,
            out.llm_hidden_states.begin() + end_idx * hidden_dim);
      }
      out.num_tokens = static_cast<int>(out.generated_token_ids.size());
    }
  }

  if (out.generated_token_ids.empty()) {
    out.text.clear();
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(std::string raw_text,
                        tokenizer.TokenIdsToText(out.generated_token_ids));
  const size_t start_pos = raw_text.find(kThinkStart);
  if (start_pos != std::string::npos) {
    const size_t end_pos =
        raw_text.find(kThinkEnd, start_pos + kThinkStart.size());
    if (end_pos != std::string::npos) {
      absl::string_view view(raw_text);
      absl::string_view prefix = view.substr(0, start_pos);
      absl::string_view suffix = absl::StripLeadingAsciiWhitespace(
          view.substr(end_pos + kThinkEnd.size()));
      out.text = absl::StrCat(prefix, suffix);
      return absl::OkStatus();
    }
  }
  out.text = std::string(absl::StripLeadingAsciiWhitespace(raw_text));
  return absl::OkStatus();
}

// Selects the next token ID from `logits` using greedy argmax when
// `!config.do_sample || config.temperature <= 0.0f`, or temperature-scaled
// top-k and nucleus (top-p) sampling otherwise.
int SampleToken(absl::Span<const float> logits,
                const MiniCpmOModelConfig& config, std::mt19937_64& rng) {
  if (logits.empty()) return 0;
  if (!config.do_sample || config.temperature <= 0.0f) {
    return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  const double temp = std::max<double>(config.temperature, 1e-6);
  const float max_logit = *std::max_element(logits.begin(), logits.end());

  std::vector<std::pair<double, int>> candidates;
  candidates.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    const double p = std::exp((logits[i] - max_logit) / temp);
    candidates.push_back({p, static_cast<int>(i)});
  }

  const int k =
      (config.top_k > 0)
          ? std::min(config.top_k, static_cast<int>(candidates.size()))
          : static_cast<int>(candidates.size());
  absl::c_partial_sort(
      candidates, candidates.begin() + k,
      [](const auto& a, const auto& b) { return a.first > b.first; });
  candidates.resize(k);

  double sum = 0.0;
  for (const auto& c : candidates) sum += c.first;
  if (sum <= 0.0) return candidates.front().second;

  if (config.top_p > 0.0f && config.top_p < 1.0f && candidates.size() > 1) {
    double cumulative = 0.0;
    size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
      cumulative += candidates[keep].first / sum;
      if (cumulative >= config.top_p) {
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

// Validates `prompt` dimensions and runs `Prefill` on `llm_runner` with the
// full prompt embedding sequence (`LlmLiteRtCompiledModelExecutorBase::Prefill`
// handles work-group splitting across `prefill_<N>` signatures internally).
absl::Status PrefillPrompt(LiteRtLmRunner& llm_runner,
                           const MiniCpmOFrontendOutput& prompt,
                           int hidden_dim) {
  if (prompt.prompt_len <= 0 || prompt.prompt_embeddings.empty()) {
    return absl::InvalidArgumentError(
        "MiniCpmOFrontendOutput has empty prompt_embeddings.");
  }
  if (prompt.prompt_embeddings.size() !=
      static_cast<size_t>(prompt.prompt_len) * hidden_dim) {
    return absl::InvalidArgumentError(
        absl::StrCat("MiniCpmOFrontendOutput prompt_embeddings size (",
                     prompt.prompt_embeddings.size(),
                     ") does not match prompt_len * llm_hidden_size (",
                     prompt.prompt_len * hidden_dim, ")."));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto prefill_buf,
      support::CreateTensorBuffer<float>({1, prompt.prompt_len, hidden_dim}));
  LITERT_RETURN_IF_ERROR(
      prefill_buf.Write<float>(absl::MakeConstSpan(prompt.prompt_embeddings)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs prefill_inputs,
                        CreateExecutorInputsWithAudio(prefill_buf));
  return llm_runner.Prefill(prefill_inputs);
}

struct DecodeStepOutput {
  std::vector<float> logits;
  std::vector<float> hidden_state;
};

// Runs one autoregressive decode step on `llm_runner` with `input_token` and
// splits the output buffer into vocabulary logits and optional last-layer
// hidden states.
absl::StatusOr<DecodeStepOutput> RunDecodeStep(LiteRtLmRunner& llm_runner,
                                               TensorBuffer& token_buf,
                                               int32_t input_token,
                                               int vocab_size, int hidden_dim) {
  LITERT_RETURN_IF_ERROR(
      token_buf.Write<int32_t>(absl::MakeConstSpan(&input_token, 1)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs decode_inputs,
                        CreateExecutorInputsWithText(token_buf));
  ABSL_ASSIGN_OR_RETURN(TensorBuffer decode_buf,
                        llm_runner.Decode(decode_inputs));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> raw_out,
                        ReadTensorBufferAsFloatVector(decode_buf));
  if (raw_out.empty()) {
    return absl::InternalError("LLM Decode returned an empty TensorBuffer.");
  }

  DecodeStepOutput step_out;
  // Custom MiniCPM-o TFLite prefill/decode exports for speech-enabled
  // pipelines concatenate the [1, llm_vocab_size] LM head logits and
  // [1, llm_hidden_size] last-layer hidden state into a single output buffer
  // so the downstream TTS projector receives hidden states without a second
  // graph output binding.
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

// Commits any un-decoded final assistant token (when `max_new_tokens` was
// reached without hitting a stop token) followed by the assistant turn-closing
// tokens (`<|im_end|>\n`) into `llm_runner`'s KV cache so subsequent
// conversation turns stay aligned with the MiniCPM-o chat template.
absl::Status CommitAssistantTurnEnd(LiteRtLmRunner& llm_runner,
                                    TensorBuffer& token_buf,
                                    bool stopped_on_eos,
                                    int32_t last_sampled_token,
                                    absl::Span<const int> turn_end_token_ids) {
  auto decode_token = [&](int32_t token_id) -> absl::Status {
    LITERT_RETURN_IF_ERROR(
        token_buf.Write<int32_t>(absl::MakeConstSpan(&token_id, 1)));
    ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs decode_inputs,
                          CreateExecutorInputsWithText(token_buf));
    ABSL_RETURN_IF_ERROR(llm_runner.Decode(decode_inputs).status());
    return absl::OkStatus();
  };

  if (!stopped_on_eos && last_sampled_token >= 0) {
    ABSL_RETURN_IF_ERROR(decode_token(last_sampled_token));
  }
  for (int token_id : turn_end_token_ids) {
    ABSL_RETURN_IF_ERROR(decode_token(static_cast<int32_t>(token_id)));
  }
  return absl::OkStatus();
}

}  // namespace

MiniCpmOLlmStage::MiniCpmOLlmStage(
    Stage<MiniCpmOFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    MiniCpmOModelConfig config)
    : frontend_(*frontend),
      config_(std::move(config)),
      tokenizer_(std::move(tokenizer)),
      llm_runner_(std::move(llm_runner)),
      hidden_dim_(config_.llm_hidden_size) {
  rng_.seed(config_.seed.value_or(42));
}

MiniCpmOLlmStage::~MiniCpmOLlmStage() = default;

absl::StatusOr<std::unique_ptr<MiniCpmOLlmStage>> MiniCpmOLlmStage::Create(
    Stage<MiniCpmOFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    const MiniCpmOModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOLlmStage config.llm_hidden_size and config.llm_vocab_size "
        "must be positive.");
  }

  auto stage = absl::WrapUnique(new MiniCpmOLlmStage(
      frontend, std::move(tokenizer), std::move(llm_runner), config));

  stage->stop_token_ids_.insert(minicpm_o::kImEndTokenId);
  stage->stop_token_ids_.insert(minicpm_o::kEosTokenId);
  for (absl::string_view stop_str :
       {"<|im_end|>", "<|endoftext|>", "<|tts_eos|>", "<tts_eos>", "</s>"}) {
    auto tok_id = stage->tokenizer_->TokenToId(stop_str);
    if (tok_id.ok()) {
      stage->stop_token_ids_.insert(*tok_id);
    }
  }

  int im_end_id = minicpm_o::kImEndTokenId;
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

bool MiniCpmOLlmStage::NeedScheduleInternal() const {
  return frontend_.HasOutput();
}

absl::Status MiniCpmOLlmStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_ASSIGN_OR_RETURN(MiniCpmOFrontendOutput prompt, frontend_.GetOutput());

  ABSL_RETURN_IF_ERROR(PrefillPrompt(*llm_runner_, prompt, hidden_dim_));

  // TODO(b/568027544): Support streaming partial token/hidden-state outputs per
  // decode step for low-latency voice assistant turns.
  MiniCpmOLlmOutput out;
  out.generate_audio = prompt.generate_audio;
  out.is_final = true;

  // LlmLiteRtCompiledModelExecutorBase::Prefill records the final prompt token
  // in `ProcessedTokens` via `AddPendingInputToken`. Passing -1 on step 0
  // instructs Decode to consume that pending input token; subsequent steps pass
  // the previously sampled token ID via `CreateExecutorInputsWithText`.
  int32_t current_token = -1;
  LITERT_ASSIGN_OR_RETURN(auto token_buf,
                          support::CreateTensorBuffer<int32_t>({1, 1}));
  const int max_steps = std::max(1, config_.max_new_tokens);
  bool stopped_on_eos = false;

  for (int step = 0; step < max_steps; ++step) {
    ABSL_ASSIGN_OR_RETURN(
        DecodeStepOutput step_out,
        RunDecodeStep(*llm_runner_, token_buf, current_token,
                      config_.llm_vocab_size, hidden_dim_));
    const int token_id = SampleToken(step_out.logits, config_, rng_);
    if (stop_token_ids_.contains(token_id)) {
      stopped_on_eos = true;
      break;
    }
    if (step_out.hidden_state.empty() && prompt.generate_audio) {
      return absl::FailedPreconditionError(
          "MiniCpmOLlmStage requires hidden states from llm_runner when "
          "generate_audio is enabled.");
    }
    out.generated_token_ids.push_back(token_id);
    out.llm_hidden_states.insert(out.llm_hidden_states.end(),
                                 step_out.hidden_state.begin(),
                                 step_out.hidden_state.end());
    ++out.num_tokens;
    current_token = token_id;
  }

  ABSL_RETURN_IF_ERROR(CommitAssistantTurnEnd(
      *llm_runner_, token_buf, stopped_on_eos, current_token,
      turn_end_token_ids_));
  ABSL_RETURN_IF_ERROR(
      FinalizeLlmOutputTextAndHiddenStates(*tokenizer_, hidden_dim_, out));
  PushOutput(std::move(out));
  return absl::OkStatus();
}

void MiniCpmOLlmStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  rng_.seed(config_.seed.value_or(42));
  (void)llm_runner_->Reset();
}

}  // namespace litert::omni::any2any
