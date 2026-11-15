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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// A sampler that samples token ids from logits.
//
// A sampler owns no randomness: the sampled ids are a pure function of the
// logits, the sampler parameters and the caller-supplied uniform values that
// pick the tokens (see `SampleToIdAndScoreBuffer()`). Callers draw the
// uniforms from their own random number generators, e.g. with
// `DrawSamplingUniforms()` in `sampling_uniforms.h`.
// Optionally, it may be able to handle input tensors. If so, the sampler can
// fill input tensors by itself, e.g. input tokens from output tokens, input
// positions with one-incremented from the previous step, then, runs inference
// for the next step. If the backend is an independent processing unit like GPU,
// the inference is being done asynchronously while the sampler returns the
// sample ID for the previous step.
class Sampler {
 public:
  virtual ~Sampler() = default;

  // Given a batch of logits, samples a batch of token ids.
  // The expected shape of the logits is [batch_size, sequence_size,
  // vocab_size]. The output is a 2D litert::TensorBuffer of shape [batch_size,
  // sequence_size]. The scores_tensor is optional. If it is not nullptr, the
  // sampled scores are also written to it (in the same shape as the
  // ids_tensor). The scores are the log of the probability of the sampled
  // token.
  //
  // The token of batch `b` and position `s` is picked with the uniform value
  // `uniforms[b * sequence_size + s]` by inverse transform sampling.
  // `uniforms` must have batch_size * sequence_size values in [0, 1).
  // Samplers that sample deterministically (e.g. greedy sampling) ignore the
  // values.
  virtual absl::Status SampleToIdAndScoreBuffer(
      const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
      TensorBuffer* scores_tensor, absl::Span<const float> uniforms) = 0;

  // Updates the configs of the sampler.
  virtual absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size) = 0;

  // Whether the sampler can handle inputs as well. If true, the sampler can
  // fill input tensors by itself, e.g. input tokens from output tokens,
  // input positions with one-incremented from the previous step, etc.
  virtual bool CanHandleInput() const { return false; }

  // Whether the sampler handles the input.
  //
  // It must be true when `CanHandleInput()` is true and
  // `SetInferenceFuncAndInputTensors()` returned OK for non-null
  // `run_inference_func`.
  //
  // It must be false
  //  1) when `CanHandleInput()` is false,
  //  2) when `CanHandleInput()` is true but `SetInferenceFuncAndInputTensors()`
  //     has not been called,
  //  3) when `CanHandleInput()` is true but `SetInferenceFuncAndInputTensors()`
  //     was called with null `run_inference_func` last time, or
  //  4) when `CanHandleInput()` is true but `SetInferenceFuncAndInputTensors()`
  //     returned non-OK status last time.
  virtual bool HandlesInput() const { return false; }

  // Sets `run_inference_func` with `arg` and input tensors to handle inputs.
  //
  // If `run_inference_func` is not nullptr, it will be called within
  // `SampleToIdAndScoreBuffer()` to run inference with the given input tensors
  // before `SampleToIdAndScoreBuffer()` returns. `HandlesInput()` will be true
  // after this call.
  //
  // If `run_inference_func` is nullptr, all other arguments are ignored, and
  // `HandlesInput()` will be false after this call.
  //
  // It returns `UnimplementedError` if `CanHandleInput()` is false.
  virtual absl::Status SetInferenceFuncAndInputTensors(
      int (*run_inference_func)(void* arg), void* arg,
      const TensorBuffer* ids_tensor,
      const TensorBuffer* prev_input_positions_tensor,
      const TensorBuffer* input_positions_tensor,
      const TensorBuffer* prev_mask_tensor, const TensorBuffer* mask_tensor,
      const TensorBuffer* prev_param_tensor, const TensorBuffer* param_tensor) {
    return absl::UnimplementedError(
        "SetInferenceFuncAndInputTensors is not implemented.");
  }
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_H_
