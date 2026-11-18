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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_LLM_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_LLM_STAGE_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_llm_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Thinker LLM stage for Qwen2.5-Omni:
// - Wraps `Any2AnyLlmStage` in `HiddenStateMode::kPostTokenWithPrompt` mode.
// - Collects last-layer hidden states across all `prompt_len` prompt tokens
//   (`prompt_hidden_states`, matching `thinker_result.hidden_states[0][-1]`)
//   when `generate_audio` is enabled.
// - Pairs each generated reply token `generated_token_ids[j]` with the Thinker
//   last-layer hidden state from the step where `generated_token_ids[j]` is
//   the INPUT (`thinker_result.hidden_states[j + 1][-1]` in
//   `Qwen2_5OmniForConditionalGeneration.generate`).
class QwenOmniLlmStage : public SingleThreadedStageWithDeque<Any2AnyLlmOutput> {
 public:
  static absl::StatusOr<std::unique_ptr<QwenOmniLlmStage>> Create(
      Stage<Any2AnyFrontendOutput>* absl_nonnull frontend,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
      const QwenOmniModelConfig& config = {});

  ~QwenOmniLlmStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  void ResetInternal() override;
  absl::Status ScheduleInternal() override;

 private:
  explicit QwenOmniLlmStage(
      std::unique_ptr<Any2AnyLlmStage> absl_nonnull llm_stage);

  std::unique_ptr<Any2AnyLlmStage> llm_stage_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_LLM_STAGE_H_
