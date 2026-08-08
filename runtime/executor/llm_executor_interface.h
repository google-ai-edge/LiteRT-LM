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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_INTERFACE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_INTERFACE_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/state_interface.h"

namespace litert::lm {

// The contract between LiteRT-LM and the executor implementation.
// The executor is expected to be stateless. Thread-safety is not required.
class LlmExecutorBaseInterface {
 public:
  virtual ~LlmExecutorBaseInterface() = default;

  // Creates a KV cache with the appropriate configurations.
  virtual absl::StatusOr<std::unique_ptr<StateInterface>> CreateKVCache() = 0;

  // Synchronous prefill operation. The executor is expected to update the KV
  // Cache with the provided input data.
  virtual absl::Status Prefill(ExecutorInputs&& input_data,
                               StateInterface& kv_cache,
                               std::optional<int> lora_id) = 0;

  // Loads a LoRA adapter with the provided model assets. Returns the ID of the
  // loaded LoRA adapter.
  virtual absl::StatusOr<int> LoadLoRA(const ModelAssets& model_assets) = 0;

  // Unloads the LoRA adapter with the provided ID.
  virtual absl::Status UnloadLoRA(int lora_id) = 0;

  // Best effort cancellation of any ongoing operations. If no operation is
  // ongoing, the cancellation is a no-op.
  virtual absl::Status Cancel() = 0;
};

class LlmExecutorExternalSamplerInterface : public LlmExecutorBaseInterface {
 public:
  // Performs a single decode step synchronously. The executor is expected to
  // update the KV Cache with the provided input data. The returned value is
  // logits for the provided input.
  virtual absl::StatusOr<TensorBuffer> Step(ExecutorInputs&& input_data,
                                            StateInterface& kv_cache,
                                            std::optional<int> lora_id) = 0;
};

class LlmExecutorInternalSamplerInterface : public LlmExecutorBaseInterface {
 public:
  // Performs N decode steps synchronously. The executor is expected to update
  // the KV Cache with the provided input data. The internal sampling allows to
  // minimize data movement. Additionally, the grouped num_steps allows
  // scheduling multiple back to back decode steps.
  // The function returns the sampled token ids.
  virtual absl::StatusOr<std::vector<int>> SampleTokens(
      int num_steps, ExecutorInputs&& input_data, StateInterface& kv_cache,
      std::optional<int> lora_id) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_INTERFACE_H_
