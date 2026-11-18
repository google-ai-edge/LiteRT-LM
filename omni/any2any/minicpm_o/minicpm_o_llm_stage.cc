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
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

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
#include "omni/any2any/any2any_llm_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace {

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

}  // namespace

MiniCpmOLlmStage::MiniCpmOLlmStage(
    std::unique_ptr<Any2AnyLlmStage> absl_nonnull llm_stage,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer, int hidden_dim)
    : llm_stage_(std::move(llm_stage)),
      tokenizer_(std::move(tokenizer)),
      hidden_dim_(hidden_dim) {}

MiniCpmOLlmStage::~MiniCpmOLlmStage() = default;

absl::StatusOr<std::unique_ptr<MiniCpmOLlmStage>> MiniCpmOLlmStage::Create(
    Stage<Any2AnyFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    const MiniCpmOModelConfig& config) {
  Any2AnyLlmStage::Options options{
      .llm_hidden_size = config.llm_hidden_size,
      .llm_vocab_size = config.llm_vocab_size,
      .max_new_tokens = config.max_new_tokens,
      .do_sample = config.do_sample,
      .temperature = config.temperature,
      .top_k = config.top_k,
      .top_p = config.top_p,
      .seed = config.seed,
      .enable_audio_conditioning = true,
      .hidden_state_mode = Any2AnyLlmStage::HiddenStateMode::kPreToken,
      .default_im_end_token_id = minicpm_o::kImEndTokenId,
      .default_eos_token_id = minicpm_o::kEosTokenId,
      .detokenize_output = false,
  };
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Any2AnyLlmStage> llm_stage,
      Any2AnyLlmStage::Create(frontend, tokenizer, std::move(llm_runner),
                              std::move(options)));
  return absl::WrapUnique(new MiniCpmOLlmStage(
      std::move(llm_stage), std::move(tokenizer), config.llm_hidden_size));
}

bool MiniCpmOLlmStage::NeedScheduleInternal() const {
  return llm_stage_->NeedSchedule();
}

absl::Status MiniCpmOLlmStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_RETURN_IF_ERROR(llm_stage_->Schedule());
  ABSL_ASSIGN_OR_RETURN(Any2AnyLlmOutput llm_out, llm_stage_->GetOutput());

  MiniCpmOLlmOutput out{
      .text = std::move(llm_out.text),
      .generated_token_ids = std::move(llm_out.generated_token_ids),
      .llm_hidden_states = std::move(llm_out.llm_hidden_states),
      .num_tokens = llm_out.num_tokens,
      .generate_audio = llm_out.generate_audio,
      .is_final = llm_out.is_final,
  };
  ABSL_RETURN_IF_ERROR(
      FinalizeLlmOutputTextAndHiddenStates(*tokenizer_, hidden_dim_, out));
  PushOutput(std::move(out));
  return absl::OkStatus();
}

void MiniCpmOLlmStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  llm_stage_->Reset();
}

}  // namespace litert::omni::any2any
