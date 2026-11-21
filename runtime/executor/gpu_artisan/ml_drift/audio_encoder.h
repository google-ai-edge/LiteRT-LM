// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_AUDIO_ENCODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_AUDIO_ENCODER_H_

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"

namespace ml_drift {

using TensorHandle = GpuModelBuilder::TensorHandle;

enum class SubSamplePaddingType {
  UNSPECIFIED = 0,
  CAUSAL = 1,
  REVERSE_CAUSAL = 2,
  SAME = 3,
};

struct AudioContextData {
  std::vector<std::unique_ptr<GpuSpatialTensor>> key_tensors;
  std::vector<std::unique_ptr<GpuSpatialTensor>> value_tensors;
  std::unique_ptr<GpuSpatialTensor> stream_cumsum;
  std::vector<std::unique_ptr<GpuSpatialTensor>> stream_depthwise_tensors;
};

struct HWPadding {
  SubSamplePaddingType h;
  SubSamplePaddingType w;
};

struct ConvConfig {
  int channel_size;
  HW kernel_size;
  HW stride;
  HWPadding padding;
  bool bias;
};

struct SubSampleConvProjectionConfig {
  int input_dims;
  int model_dims;
  std::vector<ConvConfig> convs;
  bool is_causal;
  bool proj_bias;
  bool norm_bias;
  int shrink_factor;
  int take_every_n;
  int pad_for_streaming;     // How much data to grab from the previous chunk.
  int tokens_per_frame = 1;  // Number of soft tokens to output
};

inline std::ostream& operator<<(std::ostream& os,
                                const SubSampleConvProjectionConfig& config) {
  return os << "input_dims: " << config.input_dims
            << " model_dims: " << config.model_dims
            << " convs: " << config.convs.size()
            << " is_causal: " << config.is_causal
            << " proj_bias: " << config.proj_bias
            << " norm_bias: " << config.norm_bias
            << " shrink_factor: " << config.shrink_factor
            << " take_every_n: " << config.take_every_n
            << " pad_for_streaming: " << config.pad_for_streaming
            << " tokens_per_frame: " << config.tokens_per_frame;
}

struct MLPAdapterConfig {
  int num_layers = 0;
  int output_dim = 0;
  bool initial_layer_norm = true;
};

inline std::ostream& operator<<(std::ostream& os,
                                const MLPAdapterConfig& config) {
  return os << "num_layers: " << config.num_layers
            << " output_dim: " << config.output_dim
            << " initial_layer_norm: " << config.initial_layer_norm;
}

struct ProjectionConfig {
  int output_dim = 0;
};

inline std::ostream& operator<<(std::ostream& os,
                                const ProjectionConfig& config) {
  return os << "output_dim: " << config.output_dim;
}

// Configuration of audio encoder.
// TODO: b/355725539 - Introduce AudioEncoderParams in proto. Then migrate
// multimodal config values to well_known_models.
// Only MLPAdapter or Projection should be defined. Projection is used for
// Gemini v3.
struct AudioEncoderConfig {
  SubSampleConvProjectionConfig feature;
  TransformerConfig transformer;
  // Adapter at end of v1/v2 models to convert from
  // audio model dims -> text model dims
  MLPAdapterConfig adapter;
  // Adapter at end of v3+ models to convert from
  // audio model dims -> text model dims
  ProjectionConfig projection;

  // Optional projection parameters for the projection layer at the end of the
  // USM encoder.
  // This is typically used for smaller USM encoders (e.g., < 700M parameters)
  // and is applied before `projection_parameters`.
  ProjectionConfig usm_projection;
};

// Either adapter or projection should be defined.
// Whichever one isn't defined will have output_dim = 0.
// Hence applying the max will return the correct output_dim.
inline int GetOutputDim(const AudioEncoderConfig& config) {
  return std::max(config.adapter.output_dim, config.projection.output_dim);
}

inline std::ostream& operator<<(std::ostream& os,
                                const AudioEncoderConfig& config) {
  return os << "feature: (" << config.feature << ") adapter: ("
            << config.adapter << ") projection: (" << config.projection
            << ") usm projection: (" << config.usm_projection
            << ") transformer: (" << config.transformer << ")";
}

GPUOperation CreateNormalizeVar(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const TensorDescriptor& src, const TensorDescriptor& mean,
    const TensorDescriptor& variance, const TensorDescriptor& dst,
    const Tensor<Linear, DataType::kFloat32>& gamma,
    const Tensor<Linear, DataType::kFloat32>& beta, float epsilon);

class AudioEncoderBuilder : public TransformerBuilder {
 public:
  explicit AudioEncoderBuilder(
      const AudioEncoderConfig& audio_encoder_config, int kv_cache_batch_size,
      const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
      ::odml::infra::gpu::LlmTensorLoader* tensor_loader)
      : TransformerBuilder(audio_encoder_config.transformer,
                           kv_cache_batch_size, gpu_info, create_info,
                           tensor_loader),
        feature_config_(audio_encoder_config.feature),
        adapter_config_(audio_encoder_config.adapter),
        projection_config_(audio_encoder_config.projection),
        usm_projection_config_(audio_encoder_config.usm_projection) {};

  TensorDescriptor GetFeatureDescriptor() const;
  TensorDescriptor GetMaskDescriptor() const;
  TensorDescriptor GetCumsumStreamDescriptor() const;
  TensorDescriptor GetDepthwiseStreamDescriptor() const;
  TensorDescriptor GetSignalDescriptor() const;
  TensorDescriptor GetKCacheConformerTensorDescriptor() const;
  TensorDescriptor GetVCacheConformerTensorDescriptor() const;

  bool IsStreaming() const;

  absl::StatusOr<GpuModel> CreateAudioEncoderModel(
      TensorHandle* feature_handle, TensorHandle* mask_handle,
      TensorHandle* params_i32_handle,
      std::vector<TensorHandle>* key_cached_tensors,
      std::vector<TensorHandle>* value_cached_tensors,
      TensorHandle* stream_cumsum,
      std::vector<TensorHandle>* stream_depthwise_tensors, TensorHandle* signal,
      TensorHandle* feature_output_handle);

  absl::StatusOr<GpuModel> CreateCleanCacheModelv3(
      std::vector<TensorHandle>* key_cached_tensors,
      std::vector<TensorHandle>* value_cached_tensors,
      TensorHandle& stream_cumsum,
      std::vector<TensorHandle>* stream_depthwise_tensors);

  // See sequence_layers/jax/utils:get_timing_signal_1d_pos
  TensorFloat32 GetTimingSignal1dPos(int min_timescale, int max_timescale,
                                     int past_horizon, int future_horizon);

 protected:
  // Perform the gather op along the WIDTH axis.
  TensorHandle Gather(const TensorHandle& src, std::vector<int>& indices);

  // Get first N slices of src along the WIDTH axis.
  absl::StatusOr<TensorHandle> GetFirstN(const TensorHandle& src, int len);

  // Get last N slices of src along the WIDTH axis.
  absl::StatusOr<TensorHandle> GetLastN(const TensorHandle& src, int len);

  TensorHandle ComputeOutputPadding(const TensorHandle& padding_tensor,
                                    const HW& filter_stride);

  absl::StatusOr<std::vector<TensorHandle>> MakeConv2DWithMask(
      const TensorHandle& feature, const TensorHandle& mask,
      const OHWI& filter_shape, const ConvConfig& conv_config,
      const std::string& name);

  TensorHandle UpdateCumsum(const TensorHandle& input, int layer, int index);

  TensorHandle HWCMaskedGroupNorm(
      TensorHandle src, TensorHandle mask, const TensorHandle& stream_cumsum,
      int layer, float epsilon, const Tensor<Linear, DataType::kFloat32>& gamma,
      const Tensor<Linear, DataType::kFloat32>& beta);

  absl::StatusOr<TensorHandle> MakeMaskedGroupNorm(
      const TensorHandle& src, const TensorHandle& mask,
      const TensorHandle& stream_cumsum, int layer, const std::string& name);

  absl::StatusOr<std::vector<TensorHandle>> MakeSubSampleConvProjection(
      const TensorHandle& feature, const TensorHandle& mask,
      const TensorHandle& segment_pos_handle, const TensorHandle& stream_cumsum,
      const std::string& name);

  GPUOperation CreateReducerOp(const TensorDescriptor& src,
                               const TensorDescriptor& dst, int take_every_n);

  absl::StatusOr<TensorHandle> MakeAudioFFLayerv3(const TensorHandle& src,
                                                  const std::string& name);

  absl::StatusOr<TensorHandle> MakeLightConv1d(
      const TensorHandle& input, const TensorHandle& stream_depthwise,
      const std::string& name);

  absl::StatusOr<TensorHandle> MakeSoftmaxMatMul(const TensorHandle& logits,
                                                 const TensorHandle& v);

  absl::StatusOr<TensorHandle> MakeRelativeEmbedding(const TensorHandle& q,
                                                     const TensorHandle& k,
                                                     const TensorHandle& signal,
                                                     const std::string& name);

  absl::StatusOr<TensorHandle> ExtractBlockContextNoPad(
      const TensorHandle& src);

  absl::StatusOr<TensorHandle> ExtractBlockContext(const TensorHandle& src);

  absl::StatusOr<TensorHandle> MakeConformerAttention(
      const TensorHandle& src, const TensorHandle& attention_mask,
      const TensorHandle& segment_pos, const TensorHandle& key_tensor,
      const TensorHandle& value_tensor, const TensorHandle& signal,
      const std::string& name);

  absl::StatusOr<TensorHandle> MakeConformerLayer(
      const TensorHandle& inputs, const TensorHandle& attention_mask,
      const TensorHandle& segment_pos, const TensorHandle& key_tensor,
      const TensorHandle& value_tensor, const TensorHandle& params_tensor,
      const TensorHandle& signal, const TensorHandle& stream_depthwise,
      const std::string& name);

  absl::StatusOr<TensorHandle> MakeAttentionMaskv2(
      const TensorHandle& mask, const TensorHandle& params_i32_handle);

  absl::StatusOr<TensorHandle> ExtractMaskStreaming(
      const TensorHandle& src, const TensorHandle& params_i32_handle);

  absl::StatusOr<TensorHandle> MakeAttentionMaskv3(
      const TensorHandle& mask, const TensorHandle& params_i32_handle);

  absl::StatusOr<TensorHandle> MakeEncoderLayer(
      const TensorHandle& feature_handle, const TensorHandle& mask_handle,
      std::vector<TensorHandle>* key_cached_tensors,
      std::vector<TensorHandle>* value_cached_tensors,
      const TensorHandle& params_i32_handle, const TensorHandle& stream_cumsum,
      std::vector<TensorHandle>* stream_depthwise_tensors, TensorHandle* signal,
      const std::string& name);

  absl::StatusOr<TensorHandle> MakeMLPAdapterLayer(
      const TensorHandle& feature_handle, const std::string& name);

  absl::StatusOr<TensorHandle> MakeOutputProjection(
      const TensorHandle& feature_handle, const std::string& name);

  absl::StatusOr<TensorHandle> MakeProjectionV3(
      const TensorHandle& feature_handle, const std::string& name);

 private:
  // For unit tests only.
  friend class AudioEncoderSpouse;

  SubSampleConvProjectionConfig feature_config_;
  MLPAdapterConfig adapter_config_;
  ProjectionConfig projection_config_;
  ProjectionConfig usm_projection_config_;
};

bool IsModelv3(const TransformerConfig& config);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_AUDIO_ENCODER_H_
