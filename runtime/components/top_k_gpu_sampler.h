#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_GPU_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_GPU_SAMPLER_H_

#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/ml_drift/common/gpu_info.h"
#include "third_party/ml_drift/common/gpu_model.h"
#include "third_party/ml_drift/common/gpu_model_builder.h"
#include "third_party/ml_drift/common/task/tensor_desc.h"
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// The GPU interface of LiteRT Sampler.
// The backend implementation defines the creation function.
class TopKGpuSampler : public Sampler {
 public:
  virtual ~TopKGpuSampler() = default;

  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override {
    return absl::UnimplementedError("Not implemented yet.");
  }

 protected:
  struct LlmRuntimeParams {
    int token_index_offset;
    std::optional<int> active_tokens;  // for cached self-attention
    int topk;                          // for sampling during post-processing
    int mask_time_step;                // (for attention_mask only)
    int mask_batch_size;               // (for attention_mask only)
    std::optional<int> local_attention_start_index;

    // indexes of values in linear int32 param buffer
    static constexpr int kTokenOffsetIndex = 0;
    static constexpr int kActiveTokensIndex = 1;
    static constexpr int kActiveTokensAlignedIndex =
        2;  // stores active tokens aligned to some value
    static constexpr int kTopKIndex = 3;
    static constexpr int kMaskTimeStepIndex = 4;
    static constexpr int kMaskBatchSizeIndex = 5;
    static constexpr int kLocalAttentionStartIndex = 6;
    static constexpr int kTotalParamsCount = 7;
  };

  struct TransformerConfig {
    int batch_size;
    int cache_size;
    int sequence_size;
    int vocab_size;
    int max_top_k;
  };

  TopKGpuSampler(ml_drift::GpuInfo gpu_info,
                 ml_drift::CreateGpuModelInfo create_info,
                 proto::SamplerParameters sampler_params, int batch_size,
                 int cache_size, int sequence_size, int vocab_size,
                 int max_top_k)
      : gpu_info_(std::move(gpu_info)),
        create_info_(std::move(create_info)),
        config_(batch_size, cache_size, sequence_size, vocab_size, max_top_k),
        kv_cache_batch_size_(batch_size),
        sampler_params_(std::move(sampler_params)) {
    rand_gen_ =
        std::make_shared<std::default_random_engine>(sampler_params_.seed());
  }

  // Initialize the gpu sampler.
  absl::Status Initialize();

  // Create the sampling model, and bind the input pointers to the model
  // inputs/output tensors.
  absl::StatusOr<ml_drift::GpuModel> CreateSamplingModel(
      ml_drift::GpuModelBuilder::TensorHandle* src_logits,
      ml_drift::GpuModelBuilder::TensorHandle* constraint_mask_handle,
      ml_drift::GpuModelBuilder::TensorHandle* params_i32_handle,
      ml_drift::GpuModelBuilder::TensorHandle* params_f32_handle,
      ml_drift::GpuModelBuilder::TensorHandle* output_tokens);

  // Initialize the sampling model and related operations.
  virtual absl::Status InitSampling() = 0;

  // Get the descriptors for model tensor creation.
  ml_drift::TensorDescriptor GetParamsTensorDescriptor() const;
  ml_drift::TensorDescriptor GetTokensTensorDescriptor() const;

  // Get the params count for the I32 and F32 model input params.
  int GetI32ParamsCount() const { return LlmRuntimeParams::kTotalParamsCount; }
  int GetF32ParamsCount(int sequence_size) const {
    return kv_cache_batch_size_ * sequence_size + 1;
  }

  // Create the int params (mainly user input parameters) for the gpu models.
  LlmRuntimeParams CreateLlmRuntimeParams(
      const proto::SamplerParameters& sampler_params, int time_step,
      int output_batch_size) const;
  // Create the float params (mainly random-generated) for the gpu models.
  std::vector<float> CreateFloatParams(
      const proto::SamplerParameters& sampler_params,
      std::shared_ptr<std::default_random_engine> rand_gen,
      int params_count) const;

  // Random generator for sampling.
  std::shared_ptr<std::default_random_engine> rand_gen_ =
      std::make_shared<std::default_random_engine>(0);

  // Model builder for building the sampler model.
  ml_drift::GpuModelBuilder model_builder_;

  // Necessary information for building the sampler model.
  ml_drift::GpuInfo gpu_info_;
  ml_drift::CreateGpuModelInfo create_info_;
  TransformerConfig config_;
  // This is the number of output heads.
  int kv_cache_batch_size_;

  proto::SamplerParameters sampler_params_;

 private:
  // Make the sampling GPU operation.
  void MakeSampling(ml_drift::GpuModelBuilder::TensorHandle& src_logits,
                    ml_drift::GpuModelBuilder::TensorHandle& src_indices,
                    ml_drift::GpuModelBuilder::TensorHandle& params_i32_handle,
                    ml_drift::GpuModelBuilder::TensorHandle& params_f32_handle,
                    ml_drift::GpuModelBuilder::TensorHandle& output_tokens);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_GPU_SAMPLER_H_
