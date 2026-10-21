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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_WEBGPU_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_WEBGPU_SAMPLER_H_

#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/webgpu/compute_task.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/top_k_gpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

#ifndef SAMPLER_EXPORT
#if defined(_WIN32)
#define SAMPLER_EXPORT __declspec(dllexport)
#else
#define SAMPLER_EXPORT __attribute__((visibility("default")))
#endif  // defined(_WIN32)
#endif  // SAMPLER_EXPORT

extern "C" {

typedef void LiteRtTopKWebGpuSampler_Sampler;
typedef void LiteRtTopKWebGpuSampler_SamplerParameters;
typedef void LiteRtTopKWebGpuSampler_ActivationDataType;

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_Create(
    const struct LiteRtRuntimeCApiStruct* absl_nullable runtime_c_api,
    LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKWebGpuSampler_ActivationDataType* absl_nullable
        activation_data_type,
    const LiteRtTopKWebGpuSampler_SamplerParameters* absl_nullable
        sampler_params,
    LiteRtTopKWebGpuSampler_Sampler** sampler_out,
    char** absl_nullable error_msg);

SAMPLER_EXPORT void LiteRtTopKWebGpuSampler_Destroy(
    LiteRtTopKWebGpuSampler_Sampler* sampler);

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer(
    LiteRtTopKWebGpuSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor,
    const LiteRtTensorBuffer* absl_nullable scores_tensor,
    char** absl_nullable error_msg);

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_UpdateConfig(
    LiteRtTopKWebGpuSampler_Sampler* sampler,
    const LiteRtTopKWebGpuSampler_SamplerParameters* sampler_params,
    int batch_size, void* rand_gen_shared_ptr, char** error_msg);

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_CanHandleInput(
    LiteRtTopKWebGpuSampler_Sampler* sampler);

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_HandlesInput(
    LiteRtTopKWebGpuSampler_Sampler* sampler);

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors(
    LiteRtTopKWebGpuSampler_Sampler* sampler,
    int (*run_inference_func)(void* arg), void* arg,
    LiteRtTensorBuffer absl_nullable ids_tensor,
    LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
    LiteRtTensorBuffer absl_nullable input_positions_tensor,
    LiteRtTensorBuffer absl_nullable prev_mask_tensor,
    LiteRtTensorBuffer absl_nullable mask_tensor,
    LiteRtTensorBuffer absl_nullable prev_param_tensor,
    LiteRtTensorBuffer absl_nullable param_tensor,
    char** absl_nullable error_msg);

}  // extern "C"

namespace litert::lm {

// WebGPU implementation of TopK GPU sampler interface.
// Interface:
//  - Create() - Create the Sampler. Not that user should provide a LiteRT
//  environment.
//  - SampleToIdAndScoreBuffer() - Sample the token ids and scores from logits.
//  GPU backend does not support scoring for now.
class TopKWebGpuSampler : public TopKGpuSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKWebGpuSampler>> Create(
      Environment& env, int batch_size, int sequence_size, int vocab_size,
      std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params);

  // TopKGpuSampler implementation:
  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override;
  absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size,
      std::shared_ptr<std::default_random_engine> rand_gen) override;
  bool CanHandleInput() const override;
  bool HandlesInput() const override;
  absl::Status SetInferenceFuncAndInputTensors(
      int (*run_inference_func)(void* arg), void* arg,
      const TensorBuffer* ids_tensor,
      const TensorBuffer* prev_input_positions_tensor,
      const TensorBuffer* input_positions_tensor,
      const TensorBuffer* prev_mask_tensor, const TensorBuffer* mask_tensor,
      const TensorBuffer* prev_param_tensor,
      const TensorBuffer* param_tensor) override;

  const Environment& GetEnvironment() const { return litert_env_; }

 private:
  // Transformer parameters for writing params to GPU.
  struct TransformerParams {
    // The tensor to hold the i32 params.
    std::unique_ptr<ml_drift::webgpu::SpatialTensor> params_i32;
    // The operation to write the i32 params to GPU.
    std::unique_ptr<ml_drift::webgpu::ComputeTask> write_i32_params;
  };

  TopKWebGpuSampler(Environment&& litert_env,
                    std::unique_ptr<ml_drift::webgpu::ExecutionEnvironment> env,
                    ml_drift::GpuInfo gpu_info,
                    ml_drift::CreateGpuModelInfo create_info,
                    proto::SamplerParameters sampler_params, int batch_size,
                    int sequence_size, int vocab_size, int max_top_k,
                    ml_drift::DataType logits_data_type)
      : TopKGpuSampler(std::move(gpu_info), std::move(create_info),
                       std::move(sampler_params), batch_size, sequence_size,
                       vocab_size, max_top_k),
        litert_env_(std::move(litert_env)),
        env_(std::move(env)),
        logits_data_type_(logits_data_type) {}

  // Initializes the sampling model and related operations.
  absl::Status InitSampling() override;

  // Initializes write params helper ops.
  absl::Status InitHelperOps(ml_drift::webgpu::ExecutionEnvironment* env);

  // Executes the write int params helper ops.
  absl::Status ExecuteUpdateIntParams(
      ml_drift::webgpu::ExecutionEnvironment* env, TransformerParams& params,
      const LlmRuntimeParams& param_vals);

  // Executes the write float params helper ops.
  absl::Status ExecuteUpdateParams(ml_drift::webgpu::ExecutionEnvironment* env,
                                   ml_drift::webgpu::SpatialTensor* tensor,
                                   const std::vector<float>& params);

  // Copies the sampled token ids from GPU to the mappable buffer.
  absl::Status CopyTokenIdsToMappableBuffer();

  // Downloads the sampled ids from GPU to CPU.
  // The input pointer should have enough space >= output_batch_size *
  // sizeof(int32_t).
  absl::Status DownloadSampledIds(void* dst);

  Environment litert_env_;
  std::unique_ptr<ml_drift::webgpu::ExecutionEnvironment> env_;

  // Parameters and tensors for holding and writing the input parameters.
  TransformerParams text_params_;
  std::unique_ptr<ml_drift::webgpu::SpatialTensor> tokens_ids_;
  std::unique_ptr<ml_drift::webgpu::SpatialTensor> params_f32_;
  std::unique_ptr<ml_drift::webgpu::ComputeTask> write_f32_params_;

  // Holds the invokable sampling gpu model.
  std::unique_ptr<ml_drift::webgpu::InferenceContext> sampling_;
  // The value id and descriptor of the logits tensor in the sampling model.
  ml_drift::ValueId logits_id_;
  ml_drift::TensorDescriptor logits_tensor_desc_;

  // Holds the constraint mask tensor.
  std::unique_ptr<ml_drift::webgpu::SpatialTensor> constraint_mask_;

  // WebGPU mappable buffer to download the sampled ids from GPU to CPU.
  std::unique_ptr<wgpu::Buffer> tokens_ids_mappable_buffer_;

  ml_drift::DataType logits_data_type_;

  // Members for input handling model.
  std::unique_ptr<ml_drift::webgpu::InferenceContext> input_handling_;
  // Value IDs of input_handling_ in the known order.
  std::vector<ml_drift::ValueId> input_handling_ids_;
  // May hold tensors bound to the input handling model when typeinfos don't
  // match with ones provided by SetInferenceFuncAndInputTensors().
  std::vector<ml_drift::webgpu::SpatialTensor> shared_tensors_;
  // Callback to run inference after input handling.
  int (*run_inference_func_)(void* arg) = nullptr;
  void* run_inference_arg_ = nullptr;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_WEBGPU_SAMPLER_H_
