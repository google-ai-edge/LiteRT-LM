// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_PROCESSED_CONTEXT_LITERT_PROCESSED_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_PROCESSED_CONTEXT_LITERT_PROCESSED_CONTEXT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/utils/tflite_utils/litert_kv_cache.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"

namespace litert::lm {

// LiteRTProcessedContext is a wrapper of the LiteRTKVCache and the
// processed tokens.
class LiteRTProcessedContext : public ::litert::lm::ProcessedContext {
 public:
  explicit LiteRTProcessedContext(
      std::unique_ptr<odml::infra::tflite_utils::LiteRTKVCache> kv_cache,
      std::optional<uint32_t> lora_id,
      ::litert::lm::ProcessedTokens processed_tokens = {})
      : kv_cache_(std::move(kv_cache)),
        lora_id_(lora_id),
        processed_tokens_(std::move(processed_tokens)) {};

  std::optional<uint32_t> lora_id() const override {
    return lora_id_;
  }

  void set_lora_id(std::optional<uint32_t> lora_id) override {
    lora_id_ = lora_id;
  }

  ::litert::lm::ProcessedTokens& processed_tokens() override {
    return processed_tokens_;
  }

  absl::StatusOr<
      std::reference_wrapper<odml::infra::tflite_utils::LiteRTKVCache>>
  mutable_kv_cache() {
    if (!kv_cache_) {
      return absl::FailedPreconditionError("KV cache is not initialized.");
    }
    return *kv_cache_;
  }

  std::unique_ptr<odml::infra::tflite_utils::LiteRTKVCache> TakeKVCache() {
    return std::move(kv_cache_);
  }

 private:
  // The KV cache.
  std::unique_ptr<odml::infra::tflite_utils::LiteRTKVCache> kv_cache_;

  // The LoRA id.
  std::optional<uint32_t> lora_id_;

  // The processed tokens.
  ::litert::lm::ProcessedTokens processed_tokens_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_PROCESSED_CONTEXT_LITERT_PROCESSED_CONTEXT_H_
