// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_LLM_GPU_ARTISAN_PROCESSED_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_LLM_GPU_ARTISAN_PROCESSED_CONTEXT_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"

namespace odml::infra {

class LlmGpuArtisanProcessedContext : public ::litert::lm::ProcessedContext {
 public:
  explicit LlmGpuArtisanProcessedContext(
      ml_drift::ContextData context_data,
      ::litert::lm::ProcessedTokens processed_tokens = {})
      : context_data_(std::move(context_data)),
        processed_tokens_(processed_tokens) {};

  absl::StatusOr<bool> HasKVCache() {
    return !context_data_.key_tensors.empty() &&
           !context_data_.value_tensors.empty();
  };

  std::optional<uint32_t> lora_id() const override {
    return context_data_.lora_id;
  }

  void set_lora_id(std::optional<uint32_t> lora_id) override {
    context_data_.lora_id = lora_id;
  }

  ::litert::lm::ProcessedTokens& processed_tokens() override {
    return processed_tokens_;
  };

  ml_drift::ContextData& mutable_context_data() { return context_data_; }

 private:
  ml_drift::ContextData context_data_;

  ::litert::lm::ProcessedTokens processed_tokens_;
};

}  // namespace odml::infra

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_LLM_GPU_ARTISAN_PROCESSED_CONTEXT_H_
