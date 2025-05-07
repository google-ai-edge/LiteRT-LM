#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_OPENCL_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_OPENCL_SAMPLER_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/ml_drift/cl/cl_command_queue.h"
#include "third_party/ml_drift/cl/cl_event.h"
#include "third_party/ml_drift/cl/cl_operation.h"
#include "third_party/ml_drift/cl/environment.h"
#include "third_party/ml_drift/cl/inference_context.h"
#include "third_party/ml_drift/cl/tensor.h"
#include "third_party/ml_drift/common/gpu_info.h"
#include "third_party/ml_drift/common/gpu_model.h"
#include "third_party/ml_drift/common/model.h"
#include "third_party/ml_drift/common/task/tensor_desc.h"
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/top_k_gpu_sampler.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// OpenCL implementation of TopK GPU sampler interface.
// Interface:
//  - Create() - Create the Sampler. Not that user should provide a LiteRT
//  environment.
//  - SampleToIdAndScoreBuffer() - Sample the token ids and scores from logits.
//  GPU backend does not support scoring for now.
class TopKOpenClSampler : public TopKGpuSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKOpenClSampler>> Create(
      Environment* env, int batch_size, int cache_size, int vocab_size,
      std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params);

 private:
  // Transformer parameters for writing params to GPU. Directly forked from
  // third_party/odml/infra/genai/inference/ml_drift/llm/llm_opencl.h
  struct TransformerParams {
    // The tensor to hold the i32 params.
    std::unique_ptr<ml_drift::cl::Tensor> params_i32;
    // The operation to write the i32 params to GPU.
    std::unique_ptr<ml_drift::cl::ClOperation> write_i32_params;
  };

  TopKOpenClSampler(std::unique_ptr<ml_drift::cl::Environment> env,
                    ml_drift::GpuInfo gpu_info,
                    ml_drift::CreateGpuModelInfo create_info,
                    proto::SamplerParameters sampler_params, int batch_size,
                    int cache_size, int sequence_size, int vocab_size,
                    int max_top_k)
      : TopKGpuSampler(std::move(gpu_info), std::move(create_info),
                       std::move(sampler_params), batch_size, cache_size,
                       sequence_size, vocab_size, max_top_k),
        env_(std::move(env)) {}

  // Initialize the sampling model and related operations.
  absl::Status InitSampling() override;

  // Initialize write params helper ops.
  absl::Status InitHelperOps(ml_drift::cl::Environment* env);

  // Execute the write int params helper ops.
  absl::Status ExecuteUpdateIntParams(ml_drift::cl::Environment* env,
                                      TransformerParams& params,
                                      const LlmRuntimeParams& param_vals);

  // Execute the write float params helper ops.
  absl::Status ExecuteUpdateParams(ml_drift::cl::Environment* env,
                                   ml_drift::cl::Tensor* tensor,
                                   const std::vector<float>& params);

  std::unique_ptr<ml_drift::cl::Environment> env_;

  // Parameters and tensors for holding and writing the input parameters.
  TransformerParams text_params_;
  std::unique_ptr<ml_drift::cl::Tensor> tokens_ids_;
  std::unique_ptr<ml_drift::cl::Tensor> params_f32_;
  std::unique_ptr<ml_drift::cl::ClOperation> write_f32_params_;

  // Holds the invokable sampling gpu model.
  std::unique_ptr<ml_drift::cl::InferenceContext> sampling_;
  // The value id and descriptor of the logits tensor in the sampling model.
  ml_drift::ValueId logits_id_;
  ml_drift::TensorDescriptor logits_tensor_desc_;

  // The queue for reading the sampled ids.
  ml_drift::cl::CLCommandQueue reading_queue_;
  // The event for the sampling operation. We should wait for this event before
  // downloading the sampled ids.
  ml_drift::cl::CLEvent sample_event_;

  // Holds the constraint mask tensor.
  std::unique_ptr<ml_drift::cl::Tensor> constraint_mask_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_OPENCL_SAMPLER_H_
