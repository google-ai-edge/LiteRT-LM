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
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"

namespace litert::lm {

// Stores data for a processed context in `LlmLiteRtCompiledModelExecutor`.
// This includes data that is directly relevant to a processed context,
// including the processed token IDs and the LoRA ID.
class LlmProcessedContext : public ProcessedContext {
 public:
  explicit LlmProcessedContext(
      std::optional<uint32_t> lora_id,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          kv_cache_buffers,
      ::litert::lm::ProcessedTokens processed_tokens = {})
      : lora_id_(lora_id),
        processed_tokens_(std::move(processed_tokens)),
        kv_cache_buffers_(std::move(kv_cache_buffers)) {};

  std::optional<uint32_t> lora_id() const override { return lora_id_; }
  void set_lora_id(std::optional<uint32_t> lora_id) override {
    lora_id_ = lora_id;
  }
  ProcessedTokens& processed_tokens() override { return processed_tokens_; }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
  kv_cache_buffers() {
    return kv_cache_buffers_;
  }

 private:
  std::optional<uint32_t> lora_id_;
  ProcessedTokens processed_tokens_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      kv_cache_buffers_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_PROCESSED_CONTEXT_H_
