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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_PROCESSED_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_PROCESSED_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/state_interface.h"

namespace litert::lm {

// Stores data for a processed context in `LlmLiteRtCompiledModelExecutor`.
// This includes data that is directly relevant to a processed context,
// including the processed token IDs and the LoRA ID.
class LlmProcessedContext : public ProcessedContext {
 public:
  explicit LlmProcessedContext(
      std::optional<uint32_t> lora_id, std::unique_ptr<StateInterface> state,
      ::litert::lm::ProcessedTokens processed_tokens = {})
      : lora_id_(lora_id),
        processed_tokens_(std::move(processed_tokens)),
        state_(std::move(state)) {};

  std::optional<uint32_t> lora_id() const override { return lora_id_; }
  void set_lora_id(std::optional<uint32_t> lora_id) override {
    lora_id_ = lora_id;
  }
  ProcessedTokens& processed_tokens() override { return processed_tokens_; }

  std::unique_ptr<StateInterface>& state() { return state_; }

 private:
  std::optional<uint32_t> lora_id_;
  ProcessedTokens processed_tokens_;
  std::unique_ptr<StateInterface> state_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_PROCESSED_CONTEXT_H_
