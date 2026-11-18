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

#include "omni/any2any/qwen_omni/qwen_omni_llm_stage.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_llm_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

QwenOmniLlmStage::QwenOmniLlmStage(
    std::unique_ptr<Any2AnyLlmStage> absl_nonnull llm_stage)
    : llm_stage_(std::move(llm_stage)) {}

QwenOmniLlmStage::~QwenOmniLlmStage() = default;

absl::StatusOr<std::unique_ptr<QwenOmniLlmStage>> QwenOmniLlmStage::Create(
    Stage<Any2AnyFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    const QwenOmniModelConfig& config) {
  Any2AnyLlmStage::Options options{
      .llm_hidden_size = config.llm_hidden_size,
      .llm_vocab_size = config.llm_vocab_size,
      .max_new_tokens = config.max_new_tokens,
      .do_sample = config.do_sample,
      .temperature = config.temperature,
      .top_k = config.top_k,
      .top_p = config.top_p,
      .seed = config.seed,
      .enable_audio_conditioning = config.generate_audio,
      .hidden_state_mode =
          Any2AnyLlmStage::HiddenStateMode::kPostTokenWithPrompt,
      .default_im_end_token_id = qwen_omni::kImEndTokenId,
      .default_eos_token_id = qwen_omni::kEosTokenId,
      .detokenize_output = true,
  };
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Any2AnyLlmStage> llm_stage,
      Any2AnyLlmStage::Create(frontend, std::move(tokenizer),
                              std::move(llm_runner), std::move(options)));
  return absl::WrapUnique(new QwenOmniLlmStage(std::move(llm_stage)));
}

bool QwenOmniLlmStage::NeedScheduleInternal() const {
  return llm_stage_->NeedSchedule();
}

void QwenOmniLlmStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  llm_stage_->Reset();
}

absl::Status QwenOmniLlmStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_RETURN_IF_ERROR(llm_stage_->Schedule());
  ABSL_ASSIGN_OR_RETURN(Any2AnyLlmOutput out, llm_stage_->GetOutput());
  PushOutput(std::move(out));
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
