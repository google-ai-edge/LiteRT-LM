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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_GPU_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_GPU_SAMPLER_H_

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "runtime/components/sampler.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// The GPU interface of LiteRT Sampler.
// The backend implementation defines the creation function.
class TopKGpuSampler : public Sampler {
 public:
  virtual ~TopKGpuSampler() = default;

  // Normalizes `sampler_params` (see `NormalizeSamplerParams`) and stores them
  // with `batch_size`. Returns an UnimplementedError if their type is
  // unrecognized, or an InvalidArgumentError if `k`, `p`, or `temperature` is
  // out of bounds, `k` exceeds the k the sampler was created with, or
  // `batch_size` is outside [1, the batch size the sampler was created with].
  absl::Status UpdateConfig(const proto::SamplerParameters& sampler_params,
                            int batch_size) override;

  // Draws the uniforms of a sampling call with the current parameters from
  // `rng`: one value per (batch, position) of the `batch_size` of
  // `UpdateConfig()` in batch-major order, all 0 without advancing `rng` for
  // greedy sampling. This is how the samplers drew from their own generators
  // before they took caller-supplied uniforms; the C API entry point without
  // uniforms still samples this way (see `TopKGpuSamplerCApiHandle`).
  std::vector<float> DrawLegacyUniforms(std::default_random_engine& rng) const;

 protected:
  struct LlmRuntimeParams {
    int topk;

    // indexes of values in linear int32 param buffer
    static constexpr int kTopKIndex = 0;
    static constexpr int kTotalParamsCount = 1;
  };

  // Layout of the linear float32 param buffer:
  //   [random_0, ..., random_{N-1}, 1 / temperature, top_p]
  // where N = batch_size * sequence_size and random_i are uniform samples in
  // [0, 1) used to pick the token of each (batch, position).
  struct FloatParams {
    // Offsets of the non-random values, relative to N.
    static constexpr int kInvTemperatureOffset = 0;
    static constexpr int kTopPOffset = 1;
    static constexpr int kNonRandomParamsCount = 2;
  };

  struct TransformerConfig {
    int batch_size;
    int sequence_size;
    int vocab_size;
    int max_top_k;
  };

  TopKGpuSampler(::ml_drift::GpuInfo gpu_info,
                 ::ml_drift::CreateGpuModelInfo create_info,
                 proto::SamplerParameters sampler_params, int batch_size,
                 int sequence_size, int vocab_size, int max_top_k)
      : gpu_info_(std::move(gpu_info)),
        create_info_(std::move(create_info)),
        config_{batch_size, sequence_size, vocab_size, max_top_k},
        kv_cache_batch_size_(batch_size),
        sampler_params_(std::move(sampler_params)) {}

  // Initializes the gpu sampler.
  absl::Status Initialize();

  // Creates the sampling model, and bind the input pointers to the model
  // inputs/output tensors.
  absl::StatusOr<::ml_drift::GpuModel> CreateSamplingModel(
      ::ml_drift::GpuModelBuilder::TensorHandle* src_logits,
      ::ml_drift::GpuModelBuilder::TensorHandle* constraint_mask_handle,
      ::ml_drift::GpuModelBuilder::TensorHandle* params_i32_handle,
      ::ml_drift::GpuModelBuilder::TensorHandle* params_f32_handle,
      ::ml_drift::GpuModelBuilder::TensorHandle* output_tokens,
      ::ml_drift::DataType logits_data_type);

  // Initializes the sampling model and related operations.
  virtual absl::Status InitSampling() = 0;

  // Gets the descriptors for model tensor creation.
  ::ml_drift::TensorDescriptor GetParamsTensorDescriptor() const;
  ::ml_drift::TensorDescriptor GetTokensTensorDescriptor() const;

  // Gets the params count for the I32 and F32 model input params.
  int GetI32ParamsCount() const { return LlmRuntimeParams::kTotalParamsCount; }
  int GetF32ParamsCount(int sequence_size) const {
    return kv_cache_batch_size_ * sequence_size +
           FloatParams::kNonRandomParamsCount;
  }

  // Creates the int params (mainly user input parameters) for the gpu models.
  LlmRuntimeParams CreateLlmRuntimeParams(
      const proto::SamplerParameters& sampler_params) const;
  // Creates the float params (see FloatParams) for the gpu models from the
  // caller-supplied `uniforms` (see `Sampler::SampleToIdAndScoreBuffer()`),
  // which must have one value in [0, 1) per (batch, position) of the
  // `batch_size` of `UpdateConfig()`, in batch-major order. The params of the
  // remaining batches of the model are 0.
  absl::StatusOr<std::vector<float>> CreateFloatParams(
      const proto::SamplerParameters& sampler_params,
      absl::Span<const float> uniforms, int params_count) const;

  // Creates the input handling model, and bind the input pointers to the model
  // inputs/output tensors.
  absl::StatusOr<::ml_drift::GpuModel> CreateInputHandlingModel(
      const ::ml_drift::TensorDescriptor& last_output_token_desc,
      const ::ml_drift::TensorDescriptor& input_token_desc,
      const ::ml_drift::TensorDescriptor& input_position_desc,
      const ::ml_drift::TensorDescriptor& mask_desc,
      const ::ml_drift::TensorDescriptor& param_desc,
      ::ml_drift::GpuModelBuilder::TensorHandle* last_output_token,
      ::ml_drift::GpuModelBuilder::TensorHandle* input_token,
      ::ml_drift::GpuModelBuilder::TensorHandle* prev_input_position,
      ::ml_drift::GpuModelBuilder::TensorHandle* input_position,
      ::ml_drift::GpuModelBuilder::TensorHandle* prev_mask,
      ::ml_drift::GpuModelBuilder::TensorHandle* mask,
      ::ml_drift::GpuModelBuilder::TensorHandle* prev_param,
      ::ml_drift::GpuModelBuilder::TensorHandle* param);

  // Model builder for building the sampler model.
  ::ml_drift::GpuModelBuilder model_builder_;

  // Necessary information for building the sampler model.
  ::ml_drift::GpuInfo gpu_info_;
  ::ml_drift::CreateGpuModelInfo create_info_;
  TransformerConfig config_;
  // This is the number of output heads.
  int kv_cache_batch_size_;

  proto::SamplerParameters sampler_params_;
};

// The object behind the opaque sampler handle of the GPU sampler C APIs
// (`LiteRtTopK{OpenCl,WebGpu,Metal}Sampler_Sampler*`).
template <typename SamplerT>
struct TopKGpuSamplerCApiHandle {
  std::unique_ptr<SamplerT> sampler;
  // The random number generator of the C API entry point without uniforms
  // (`..._SampleToIdAndScoreBuffer()`), which picks the tokens with
  // `sampler->DrawLegacyUniforms(*legacy_rand_gen)`. Kept for callers that
  // don't pass uniforms (e.g. runtimes built before the uniforms symbol was
  // added); the sampler itself owns no randomness. Seeded from the creation
  // parameters, and replaced by a successful `..._UpdateConfig()` whose
  // `rand_gen_shared_ptr` points to a non-null generator, so never null.
  std::shared_ptr<std::default_random_engine> legacy_rand_gen;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_GPU_SAMPLER_H_
