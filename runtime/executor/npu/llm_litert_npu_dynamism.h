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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_DYNAMISM_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_DYNAMISM_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert

namespace litert::lm {

// Helper utility for dynamic KV cache resizing on NPU models.
//
// Usage workflow:
// 1) Call `HasDynamicKVCache(text_decoder_model)` on the text decoder Model to
//    check if dynamic KV cache resizing is supported.
// 2) Call `ResizeDynamicInputs(text_decoder_compiled_model, signature_name,
//    target_context_size)` on the text decoder CompiledModel to resize the KV
//    cache to a target context length. On success, this returns the actual
//    resized context size, which is >= target_context_size (due to hardware
//    bucketing), or an error status on failure.
// 3) Call `ResizeDynamicMaskInputs(aux_compiled_model, signature_name,
//    actual_context_size)` on the auxiliary CompiledModel, passing in the
//    actual context size returned from step (2) to ensure mask inputs match
//    the allocated KV cache capacity.
struct NpuDynamismHelper {
  // Checks if the given model contains dynamic KV cache dimensions (e.g. -1).
  static bool HasDynamicKVCache(const litert::Model& model);

  // Resizes dynamic input tensors (KV cache) for the given signature to
  // `target_context_size`.
  // Returns the actual resized context dimension (which is >=
  // target_context_size due to hardware bucketing) on success, or an error.
  static absl::StatusOr<int> ResizeDynamicInputs(
      ::litert::CompiledModel& compiled_model, absl::string_view signature_name,
      int target_context_size);

  // Resizes dynamic mask input tensors (local_context_length,
  // global_context_length) on the auxiliary compiled model.
  // Note: `actual_context_size` should be the actual context size returned
  // by `ResizeDynamicInputs`.
  static absl::Status ResizeDynamicMaskInputs(
      ::litert::CompiledModel& aux_compiled_model,
      absl::string_view signature_name, int actual_context_size);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_DYNAMISM_H_
