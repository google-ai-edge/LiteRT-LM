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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_METAL_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_METAL_SAMPLER_H_

#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/ml_drift/common/gpu_info.h"
#include "third_party/ml_drift/common/gpu_model.h"
#include "third_party/ml_drift/common/model.h"
#include "third_party/ml_drift/common/task/tensor_desc.h"
#include "third_party/ml_drift/metal/compute_task.h"
#include "third_party/ml_drift/metal/environment.h"
#include "third_party/ml_drift/metal/inference_context.h"
#include "third_party/ml_drift/metal/metal_spatial_tensor.h"
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/top_k_gpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

#ifndef SAMPLER_EXPORT
#define SAMPLER_EXPORT __attribute__((visibility("default")))
#endif  // SAMPLER_EXPORT

extern "C" {

typedef void LiteRtTopKMetalSampler_Sampler;
typedef void LiteRtTopKMetalSampler_SamplerParameters;
typedef void LiteRtTopKMetalSampler_ActivationDataType;

SAMPLER_EXPORT int LiteRtTopKMetalSampler_Create(
    LiteRtEnvironment env, int batch_size, int vocab_size,
    const LiteRtTopKMetalSampler_ActivationDataType* absl_nullable activation_data_type,
    const LiteRtTopKMetalSampler_SamplerParameters* absl_nullable sampler_params,
    LiteRtTopKMetalSampler_Sampler** sampler_out, char** absl_nullable error_msg);

SAMPLER_EXPORT void LiteRtTopKMetalSampler_Destroy(LiteRtTopKMetalSampler_Sampler* sampler);

SAMPLER_EXPORT int LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer(
    LiteRtTopKMetalSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* absl_nullable scores_tensor,
    char** absl_nullable error_msg);

SAMPLER_EXPORT int LiteRtTopKMetalSampler_UpdateConfig(
    LiteRtTopKMetalSampler_Sampler* sampler,
    const LiteRtTopKMetalSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg);

}  // extern "C"

namespace litert::lm {

// Metal implementation of TopK GPU sampler interface.
class TopKMetalSampler : public TopKGpuSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKMetalSampler>> Create(
      Environment* env, int batch_size, int vocab_size,
      std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params);

  // TopKGpuSampler implementation:
  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override;
  absl::Status UpdateConfig(const proto::SamplerParameters& sampler_params, int batch_size,
                            std::shared_ptr<std::default_random_engine> rand_gen) override;

 private:
  struct TransformerParams {
    std::unique_ptr<ml_drift::metal::MetalSpatialTensor> params_i32;
    std::unique_ptr<ml_drift::metal::ComputeTask> write_i32_params;
  };

  TopKMetalSampler(std::unique_ptr<ml_drift::metal::Environment> env, ml_drift::GpuInfo gpu_info,
                   ml_drift::CreateGpuModelInfo create_info,
                   litert::lm::proto::SamplerParameters sampler_params, TransformerConfig config,
                   ml_drift::DataType logits_data_type, id<MTLCommandQueue> command_queue)
      : TopKGpuSampler(std::move(gpu_info), std::move(create_info), sampler_params,
                       config.batch_size, config.sequence_size, config.vocab_size,
                       config.max_top_k),
        env_(std::move(env)),
        logits_data_type_(logits_data_type) {
    if (command_queue) {
      command_queue_ = command_queue;
    } else {
      command_queue_ = [env_->device() newCommandQueue];
    }
  }
  absl::Status InitSampling() override;
  absl::Status InitHelperOps(ml_drift::metal::Environment* env);

  absl::Status ExecuteUpdateIntParams(id<MTLCommandBuffer> command_buffer,
                                      TransformerParams& params,
                                      const LlmRuntimeParams& param_vals);

  absl::Status ExecuteUpdateParams(id<MTLCommandBuffer> command_buffer,
                                   ml_drift::metal::MetalSpatialTensor* tensor,
                                   const std::vector<float>& params);

  absl::Status DownloadSampledIds(void* dst);

  std::unique_ptr<ml_drift::metal::Environment> env_;
  id<MTLCommandQueue> command_queue_;

  TransformerParams text_params_;
  std::unique_ptr<ml_drift::metal::MetalSpatialTensor> tokens_ids_;
  std::unique_ptr<ml_drift::metal::MetalSpatialTensor> params_f32_;
  std::unique_ptr<ml_drift::metal::ComputeTask> write_f32_params_;

  id<MTLBuffer> staging_logits_buffer_;
  id<MTLBuffer> staging_ids_buffer_;

  std::unique_ptr<ml_drift::metal::InferenceContext> sampling_;
  ml_drift::ValueId logits_id_;
  ml_drift::TensorDescriptor logits_tensor_desc_;
  std::unique_ptr<ml_drift::metal::MetalSpatialTensor> logits_metal_tensor_;

  std::unique_ptr<ml_drift::metal::MetalSpatialTensor> constraint_mask_;

  ml_drift::DataType logits_data_type_;

  std::unique_ptr<ml_drift::metal::InferenceContext> input_handling_;
  std::vector<ml_drift::ValueId> input_handling_ids_;
  std::vector<ml_drift::metal::MetalSpatialTensor> shared_tensors_;

  int (*run_inference_func_)(void* arg) = nullptr;
  void* run_inference_arg_ = nullptr;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_METAL_SAMPLER_H_
