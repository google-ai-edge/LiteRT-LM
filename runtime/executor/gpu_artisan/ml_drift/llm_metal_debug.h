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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_METAL_DEBUG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_METAL_DEBUG_H_

#import <Metal/Metal.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/metal/inference_context.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"

// Internal-only debugging utilities for LlmMetal, for inspecting intermediate
// activations, logits, and KV caches. These are only built and called when
// llm_metal is built with `--define=llm_metal_debug_mldrift_tensors=true`.
//
// Files are written into the directory given by the
// LLM_METAL_DEBUG_MLDRIFT_TENSORS_DIR environment variable (default:
// /tmp/llm_metal_debug_tensors). Debug tensors are raw float32 files named
// after the tensor.
//
// The dump state is per process, so these assume a single LlmMetal instance.

namespace ml_drift {
namespace metal {
namespace llm_debug {

using DebugTensors =
    absl::flat_hash_map<std::string, GpuModelBuilder::TensorHandle>;

// Dumps `debug_tensors` from `context` for one batch of the first prefill in
// this process (see the implementation for which batch), waiting for all work
// on `command_queue` first. Does nothing for other batches and prefills.
absl::Status MaybeDumpPrefillTensors(id<MTLCommandQueue> command_queue,
                                     id<MTLDevice> device,
                                     InferenceContext& context,
                                     const DebugTensors& debug_tensors,
                                     int batch_index, int num_batches);

// Extra post-processing model outputs that are kept for logits debugging.
struct PostProcessingOutputs {
  GpuModelBuilder::TensorHandle pre_softmax;
  GpuModelBuilder::TensorHandle topk_tokens;
  GpuModelBuilder::TensorHandle topk_probabilities;
};

// Records the tensor ids of `outputs` after the post-processing model has been
// created, enabling MaybeDumpPostProcessingLogits().
void RegisterPostProcessingOutputs(const PostProcessingOutputs& outputs);

// For the first N sampling steps (N defaults to 64 and can be overridden with
// the LLM_METAL_DEBUG_MLDRIFT_LOGITS_STEPS environment variable), waits for
// `command_buffer`, which must contain the post-processing pass for sequence
// position `position`, then saves the full-vocabulary logits fed into TopK as
// `logits_step<NNN>_pos<NNNNN>` and prints the top few candidates. The logits
// at `position` predict the token at `position + 1`.
absl::Status MaybeDumpPostProcessingLogits(id<MTLCommandBuffer> command_buffer,
                                           id<MTLDevice> device,
                                           InferenceContext& post_processing,
                                           int position);

// If the LLM_METAL_DEBUG_MLDRIFT_DECODE_POS environment variable equals
// `position`, dumps the single-token model's `debug_tensors` into the
// `decode_pos<N>` subdirectory, plus the raw KV caches and a `meta.txt`
// describing them into its `kv` subdirectory.
absl::Status MaybeDumpDecodeStepTensors(
    id<MTLCommandQueue> command_queue, id<MTLDevice> device,
    InferenceContext& context, const DebugTensors& debug_tensors,
    const TransformerConfig& config,
    const std::vector<std::unique_ptr<MetalSpatialTensor>>& key_caches,
    const std::vector<std::unique_ptr<MetalSpatialTensor>>& value_caches,
    int position);

}  // namespace llm_debug
}  // namespace metal
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_METAL_DEBUG_H_
