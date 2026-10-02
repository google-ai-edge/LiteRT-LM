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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_mtp_drafter.h"

namespace litert::lm {

// Creates an LlmExecutor for a LiteRT model, loading and compiling the model
// from `resources`. Supports CPU, GPU and NPU backends, and both statically
// and dynamically shaped models (the right executor is picked automatically).
//
// This is the default entry point and should be preferred in most cases: the
// executor owns the whole setup (model compilation, embedding lookups, MTP
// drafter, etc.) based on `executor_settings` and the contents of `resources`.
//
// Args:
//   executor_settings: Settings for the executor (backend, cache, etc.).
//   lrt_env: The LiteRT environment.
//   resources: The model resources to load and compile the model(s) from.
absl::StatusOr<std::unique_ptr<LlmExecutor>>
CreateLlmLiteRtCompiledModelExecutor(LlmExecutorSettings executor_settings,
                                     Environment& lrt_env,
                                     ModelResources& resources);

// Creates an LlmExecutor from a LiteRT model that the caller has already
// compiled. Supports CPU and GPU backends only (NPU returns an error), and
// both statically and dynamically shaped models.
//
// Use this overload only when the caller needs to control compilation of the
// main prefill/decode model itself, e.g. when weights are streamed in and the
// model is compiled before the full model file is available, or when the
// compiled model (or its auxiliary components) is shared or built elsewhere.
// Otherwise, prefer the ModelResources& overload above.
//
// Optional components are resolved as follows:
//   - Executor metadata is read from `resources`, if provided.
//   - If `embedding_lookup` is null and `resources` is provided, both the text
//     and per-layer embedding lookups are loaded from `resources` (the passed
//     `per_layer_embedding_lookup` is then ignored). If `embedding_lookup` is
//     provided, `per_layer_embedding_lookup` is used as-is (may be null).
//   - If `mtp_drafter` is provided, it is used directly. Otherwise, when
//     speculative decoding is enabled and `resources` is provided, a drafter
//     is created from `compiled_mtp_drafter_model` if given, or else by
//     loading and compiling the drafter model from `resources`.
//
// Args:
//   executor_settings: Settings for the executor. Backend must be CPU or GPU.
//   lrt_env: The LiteRT environment.
//   compiled_model: The compiled main prefill/decode model. Must not be null.
//   resources: Optional model resources; see above.
//   embedding_lookup: Optional pre-built text embedding lookup.
//   per_layer_embedding_lookup: Optional pre-built per-layer embedding lookup.
//   compiled_mtp_drafter_model: Optional pre-compiled MTP drafter model. Only
//     used when `mtp_drafter` is null and `resources` is non-null.
//   mtp_drafter: Optional pre-built MTP drafter. Takes precedence over
//     `compiled_mtp_drafter_model`.
absl::StatusOr<std::unique_ptr<LlmExecutor>>
CreateLlmLiteRtCompiledModelExecutor(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    std::unique_ptr<CompiledModel> compiled_model,
    ModelResources* resources = nullptr,
    std::unique_ptr<EmbeddingLookupManager> embedding_lookup = nullptr,
    std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup =
        nullptr,
    std::unique_ptr<CompiledModel> compiled_mtp_drafter_model = nullptr,
    std::unique_ptr<LlmLiteRtMtpDrafter> mtp_drafter = nullptr);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_FACTORY_H_
