// Copyright 2026 The MediaPipe Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_WEBGPU_DEBUG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_WEBGPU_DEBUG_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift

// Internal-only debugging utilities for LlmWebGpu, for inspecting intermediate
// activations. These are only built and called when llm_webgpu is built with
// `--define=web_llm_webgpu_debug_mldrift_tensors=true`. On the web, the dumped
// tensors are also offered as download links on the page.
//
// The dump state is per process, so these assume a single LlmWebGpu instance.

namespace ml_drift {
namespace webgpu {
namespace llm_debug {

using DebugTensors =
    absl::flat_hash_map<std::string, GpuModelBuilder::TensorHandle>;

// Downloads, logs, and offers for download (as raw float32 files named after
// each tensor) `debug_tensors` from `context` for one batch of the first
// prefill in this process (see the implementation for which batch). Does
// nothing for other batches and prefills.
absl::Status MaybeDumpPrefillTensors(ExecutionEnvironment& env,
                                     InferenceContext& context,
                                     const DebugTensors& debug_tensors,
                                     int batch_index, int num_batches);

}  // namespace llm_debug
}  // namespace webgpu
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_WEBGPU_DEBUG_H_
