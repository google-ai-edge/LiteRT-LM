// Copyright 2024 The MediaPipe Authors.
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

#include "runtime/executor/gpu_artisan/ml_drift/llm_metal.h"

#if LLM_METAL_DEBUG_MLDRIFT_TENSORS
#include "runtime/executor/gpu_artisan/ml_drift/llm_metal_debug.h"
#endif  // LLM_METAL_DEBUG_MLDRIFT_TENSORS

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/bind_front.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift/metal/buffer.h"  // from @ml_drift
#include "ml_drift/metal/common.h"  // from @ml_drift
#include "ml_drift/metal/compute_task.h"  // from @ml_drift
#include "ml_drift/metal/environment.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/byom/byom_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/group_size.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/name_check.h"
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_file_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace ml_drift {
namespace metal {
namespace {

using ::odml::infra::gpu::GetRecommendedGroupSizes;
using ::odml::infra::gpu::GroupSize;
using ::odml::infra::gpu::LlmFileTensorLoader;
using ::odml::infra::gpu::LlmTensorLoader;
using ::odml::infra::proto::SamplerParameters;

absl::Status InitMetalOp(MetalDevice* device,
                         std::unique_ptr<GPUOperation>&& gpu_op,
                         ComputeTask* metal_op) {
  ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(device->GetInfo()));
  metal_op->Init(std::move(gpu_op));
  ABSL_RETURN_IF_ERROR(metal_op->Compile(device));
  return absl::OkStatus();
}

absl::Status InitMetalOp(MetalDevice* device, GPUOperation&& operation,
                         ComputeTask* metal_op) {
  return InitMetalOp(
      device, std::make_unique<GPUOperation>(std::move(operation)), metal_op);
}

absl::Status UploadDescriptorData(id<MTLCommandQueue> command_queue,
                                  bool require_load_with_queue,
                                  const TensorDescriptor& desc,
                                  MetalSpatialTensor* tensor) {
  if (tensor->GetStorageType() == TensorStorageType::kBuffer &&
      !require_load_with_queue) {
    memcpy(tensor->GetBufferPointer(), desc.GetData().data(),
           desc.GetData().size());
    return absl::OkStatus();
  } else {
    return tensor->UploadDescriptorData(desc, command_queue,
                                        /*wait_for_completion=*/false);
  }
}

absl::Status UploadEmbeddingsToGPU(
    const std::vector<odml::infra::gpu::EmbeddingWeights>& weights,
    InferenceContext* context, id<MTLCommandQueue> command_queue,
    bool require_load_with_queue) {
  for (const auto& weight : weights) {
    auto* tensor = context->GetTensor(weight.weights.id);
    TensorDescriptor desc = tensor->GetDescriptor();
    std::vector<uint8_t> weights_copy = weight.weights_data;
    desc.SetData(std::move(weights_copy));
    ABSL_RETURN_IF_ERROR(UploadDescriptorData(
        command_queue, require_load_with_queue, desc, tensor));

    auto* scale_tensor = context->GetTensor(weight.scales.id);
    TensorDescriptor scale_desc = scale_tensor->GetDescriptor();
    scale_desc.UploadData(weight.scales_data.data());
    ABSL_RETURN_IF_ERROR(UploadDescriptorData(
        command_queue, require_load_with_queue, scale_desc, scale_tensor));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<MetalSpatialTensor>>>
InitKvTensorsLocalGlobal(MetalDevice& device,
                         const TensorDescriptor& local_desc,
                         std::optional<TensorDescriptor> global_desc,
                         int num_local_layers_per_global, int num_cache_layers,
                         std::optional<int> global_layer_offset) {
  std::vector<std::unique_ptr<MetalSpatialTensor>> tensors(num_cache_layers);

  int pattern_offset = global_layer_offset.has_value()
                           ? *global_layer_offset
                           : num_local_layers_per_global;

  for (int i = 0; i < num_cache_layers; ++i) {
    const bool is_global =
        (i % (num_local_layers_per_global + 1) == pattern_offset);
    if (is_global && !global_desc.has_value()) {
      continue;
    }
    tensors[i] = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(device.device(),
                                      is_global ? *global_desc : local_desc,
                                      tensors[i].get()));
  }
  return tensors;
}

absl::Status CopyTensor(id<MTLBlitCommandEncoder> encoder,
                        GpuSpatialTensor* src, GpuSpatialTensor* dst) {
  if (!src || !dst) return absl::OkStatus();
  auto* metal_src = dynamic_cast<MetalSpatialTensor*>(src);
  auto* metal_dst = dynamic_cast<MetalSpatialTensor*>(dst);
  if (!metal_src || !metal_dst) {
    return absl::InternalError("Not a MetalSpatialTensor");
  }

  const auto src_storage_type = metal_src->GetStorageType();
  const auto dst_storage_type = metal_dst->GetStorageType();

  if (src_storage_type != dst_storage_type) {
    return absl::InternalError(
        "Source and destination storage types do not match");
  }

  if (src_storage_type == TensorStorageType::kBuffer ||
      src_storage_type == TensorStorageType::kImageBuffer) {
    id<MTLBuffer> src_buf = metal_src->GetBufferHandle();
    id<MTLBuffer> dst_buf = metal_dst->GetBufferHandle();
    if (src_buf && dst_buf) {
      [encoder copyFromBuffer:src_buf
                 sourceOffset:0
                     toBuffer:dst_buf
            destinationOffset:0
                         size:metal_src->GetMemorySizeInBytes()];
    }
  } else {
    id<MTLTexture> src_tex = metal_src->GetTextureHandle();
    id<MTLTexture> dst_tex = metal_dst->GetTextureHandle();
    if (src_tex && dst_tex) {
      [encoder copyFromTexture:src_tex toTexture:dst_tex];
    }
  }
  return absl::OkStatus();
}

template <typename RefTensorType>
absl::StatusOr<AudioContextData> AllocateAudioContextData(
    MetalDevice& device,
    const std::vector<std::unique_ptr<RefTensorType>>& ref_k,
    const std::vector<std::unique_ptr<RefTensorType>>& ref_v,
    const std::vector<std::unique_ptr<RefTensorType>>& ref_depthwise,
    const std::unique_ptr<RefTensorType>& ref_cumsum, bool is_model_v3) {
  AudioContextData context_data;
  const int num_layers = ref_k.size();
  context_data.key_tensors.resize(num_layers);
  context_data.value_tensors.resize(num_layers);

  for (int i = 0; i < num_layers; ++i) {
    auto key_tensor = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        device.device(), ref_k[i]->GetDescriptor(), key_tensor.get()));
    context_data.key_tensors[i] = std::move(key_tensor);

    auto value_tensor = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        device.device(), ref_v[i]->GetDescriptor(), value_tensor.get()));
    context_data.value_tensors[i] = std::move(value_tensor);

    if (is_model_v3 && i < ref_depthwise.size()) {
      auto depthwise_tensor = std::make_unique<MetalSpatialTensor>();
      ABSL_RETURN_IF_ERROR(CreateTensor(device.device(),
                                        ref_depthwise[i]->GetDescriptor(),
                                        depthwise_tensor.get()));
      context_data.stream_depthwise_tensors.push_back(
          std::move(depthwise_tensor));
    }
  }

  if (ref_cumsum) {
    auto cumsum_tensor = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        device.device(), ref_cumsum->GetDescriptor(), cumsum_tensor.get()));
    context_data.stream_cumsum = std::move(cumsum_tensor);
  }

  return context_data;
}

template <typename SrcTensorType, typename DstTensorType>
absl::Status CopyAudioContextDataTensors(
    id<MTLCommandQueue> command_queue,
    const std::vector<std::unique_ptr<SrcTensorType>>& src_k,
    const std::vector<std::unique_ptr<SrcTensorType>>& src_v,
    const std::vector<std::unique_ptr<SrcTensorType>>& src_depthwise,
    const std::unique_ptr<SrcTensorType>& src_cumsum,
    const std::vector<std::unique_ptr<DstTensorType>>& dst_k,
    const std::vector<std::unique_ptr<DstTensorType>>& dst_v,
    const std::vector<std::unique_ptr<DstTensorType>>& dst_depthwise,
    const std::unique_ptr<DstTensorType>& dst_cumsum, bool is_model_v3) {
  const int num_layers = src_k.size();
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder =
        [command_buffer blitCommandEncoder];
    for (int i = 0; i < num_layers; ++i) {
      if (i < dst_k.size()) {
        ABSL_RETURN_IF_ERROR(
            CopyTensor(blit_encoder, src_k[i].get(), dst_k[i].get()));
      }
      if (i < dst_v.size()) {
        ABSL_RETURN_IF_ERROR(
            CopyTensor(blit_encoder, src_v[i].get(), dst_v[i].get()));
      }
      if (is_model_v3 && i < src_depthwise.size() && i < dst_depthwise.size()) {
        ABSL_RETURN_IF_ERROR(CopyTensor(blit_encoder, src_depthwise[i].get(),
                                        dst_depthwise[i].get()));
      }
    }
    if (src_cumsum && dst_cumsum) {
      ABSL_RETURN_IF_ERROR(
          CopyTensor(blit_encoder, src_cumsum.get(), dst_cumsum.get()));
    }
    [blit_encoder endEncoding];
    [command_buffer commit];
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status LlmMetal::Init(const LlmConfig& config, int outputs_count,
                            const std::string& weights_path,
                            const CreateGpuModelInfo& create_info,
                            MetalDevice* device) {
  auto tensor_loader = LlmTensorLoader::MakeCaching(
      std::make_unique<LlmFileTensorLoader>(weights_path));
  return Init(config, outputs_count, std::move(tensor_loader), create_info,
              device);
}

absl::Status LlmMetal::Init(const LlmConfig& config, int outputs_count,
                            std::unique_ptr<LlmTensorLoader> tensor_loader,
                            const CreateGpuModelInfo& create_info,
                            MetalDevice* device) {
  tensor_loader_ = std::move(tensor_loader);
  device_ = device->device();
  create_info_ = create_info;
  suppressed_tokens_ = config.suppressed_tokens;
  ABSL_CHECK(config.text_decoder.has_value()) << "Text decoder is not set.";
  text_config_ = config.text_decoder.value();  // use below a copy
  epsilon_ = GetEpsilon(create_info.precision, device->GetInfo());
  max_top_k_ = text_config_.max_top_k;

  // default sampler params to be greedy.
  sampler_params_ = SamplerParameters();
  sampler_params_.set_type(SamplerParameters::GREEDY);
  sampler_params_.set_k(1);

  if (create_info.hints.Check(ModelHints::kEnableHostMappedPointer)) {
    tensor_loader_->SetCreateTensorFn(
        absl::bind_front(&LlmMetal::CreateModelTensorMmaped, this, device,
                         tensor_loader_.get(), &model_tensors_));
  } else {
    tensor_loader_->SetCreateTensorFn(absl::bind_front(
        &LlmMetal::CreateModelTensor, this, device, &model_tensors_));
  }
  output_batch_size_ = outputs_count;
  if (group_sizes_.empty()) {
    if (text_config_.sequence_size == 0) {
      group_sizes_ = GetRecommendedGroupSizes(device->GetInfo());
    } else if (text_config_.sequence_size == 1) {
      group_sizes_ = {GroupSize{1, 0.0f}};
    } else {
      group_sizes_ = {
          GroupSize{text_config_.sequence_size, 0.0f},
          GroupSize{1, 1.0f},
      };
    }
  }

  const auto& gpu_info = device->GetInfo();
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<TransformerBuilder> text_decoder_builder,
      odml_byom::CreateTransformerBuilder(text_config_, config.model_type,
                                          output_batch_size_, gpu_info,
                                          create_info, tensor_loader_.get()));

  if (text_config_.use_autosized_ringbuffers &&
      text_config_.sliding_window_size.has_value()) {
    // Set ringbuffer size to max(group_sizes) + sliding_window_size
    auto max_group_size =
        std::max_element(group_sizes_.begin(), group_sizes_.end(),
                         [](const GroupSize& a, const GroupSize& b) {
                           return a.sequence_size < b.sequence_size;
                         });
    text_config_.local_ring_cache_size =
        max_group_size->sequence_size +
        text_config_.sliding_window_size.value();
    // Need to set this value for text_decoder_builder as well, since it uses a
    // copy of the config.
    text_decoder_builder->SetLocalRingCacheSize(
        *text_config_.local_ring_cache_size);
  }

  {  // set multimodal cache size
    int audio_emb_cache_size = 0;
    int vision_emb_cache_size = 0;
    if (text_config_.vision_tokens_num > 0 &&
        text_config_.max_num_images > 0) {  // has vision
      vision_emb_cache_size =
          text_config_.vision_tokens_num * text_config_.max_num_images;
    }
    if (config.audio_encoder.has_value()) {
      audio_emb_cache_size =
          DivideRoundUp(config.audio_encoder.value().transformer.cache_size,
                        config.audio_encoder.value().feature.take_every_n);
    }
    // Audio embeddings from an audio encoder that is not part of this model
    // are written after the slots for images.
    media_cache_size_ =
        std::max(audio_emb_cache_size,
                 vision_emb_cache_size + text_config_.max_num_audio_tokens);
    text_decoder_builder->SetMediaCacheSize(media_cache_size_);
  }

  const int num_cache_layers =
      text_config_.kv_cache_shared_start_layer.has_value()
          ? *text_config_.kv_cache_shared_start_layer
          : text_config_.stack_size;

  ABSL_RETURN_IF_ERROR(InitSharedTensors(text_decoder_builder.get(), cache_,
                                         params_, num_cache_layers, device));
  // Init multimodal tensors
  if (media_cache_size_ > 0) {  // is multimodal
    if (text_config_.per_layer_embedding_dimension.has_value()) {
      if (text_decoder_builder->UseGoogleInternalModelV2()) {
        vision_ple_tensors_.reserve(text_config_.stack_size -
                                    text_config_.extra_stack_size);
        for (int i = 0;
             i < text_config_.stack_size - text_config_.extra_stack_size; ++i) {
          vision_ple_tensors_.push_back(std::make_unique<MetalSpatialTensor>());
          ABSL_RETURN_IF_ERROR(
              CreateTensor(device->device(),
                           text_decoder_builder->GetVisionPleTensorDescriptor(),
                           vision_ple_tensors_[i].get()));
        }
      }
    }
    media_embeddings_ = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(
        CreateTensor(device->device(),
                     text_decoder_builder->GetMediaEmbeddingsTensorDescriptor(),
                     media_embeddings_.get()));
  }
  ABSL_RETURN_IF_ERROR(
      InitCleanCacheModel(text_decoder_builder.get(), cache_, params_, device));
  for (int i = 0; i < group_sizes_.size(); ++i) {
    const int sequence_size = group_sizes_[i].sequence_size;
    const int batch_size = sequence_size == 1 ? output_batch_size_ : 1;
    ABSL_RETURN_IF_ERROR(InitModel(text_decoder_builder.get(), sequence_size,
                                   batch_size, device));
  }
  model_one_ = &models_[1];

  ABSL_ASSIGN_OR_RETURN(
      ple_embeddings_,
      GetDynamicPleEmbeddings(text_config_, tensor_loader_.get()));
  ABSL_ASSIGN_OR_RETURN(
      ngrammer_embeddings_,
      GetDynamicNgrammerEmbeddings(text_config_, tensor_loader_.get()));
  if (text_decoder_builder->UseGemmaModelV3NPlus()) {
    // Replace negative tokens with 0. This is to match the behavior of the
    // v3 model. Note this is for dynamically updated PLE only. For
    // non-dynamically updated PLE, the negative tokens are removed in the
    // transformer GPU model.
    tokens_transform_fn_ = [](int token) { return std::max(0, token); };
  }

  ABSL_RETURN_IF_ERROR(
      InitPostProcessingModel(text_decoder_builder.get(), device));

  {
    int buffer_size = text_config_.cache_size * sizeof(int32_t);
    input_token_ids_ = std::make_unique<Buffer>(Buffer());
    ABSL_RETURN_IF_ERROR(CreateBuffer(buffer_size, nullptr, device->device(),
                                      input_token_ids_.get()));
  }

  ABSL_RETURN_IF_ERROR(InitHelperOps(device));
  if (text_config_.kv_cache_shared_start_layer) {
    shared_kv_cache_tag_ = kSharedKvCacheLayersOptionTag;
    SetupOptionalNodes();
  }
  return absl::OkStatus();
}

absl::Status LlmMetal::InitAudioEncoder(
    const AudioEncoderConfig& config,
    std::unique_ptr<LlmTensorLoader> tensor_loader,
    const CreateGpuModelInfo& create_info, int output_batch_size,
    id<MTLCommandQueue> command_queue, Environment* env) {
  if (!IsModelv3(config.transformer)) {
    return absl::InvalidArgumentError("Audio encoder only supports v3 models.");
  }
  tensor_loader_ = std::move(tensor_loader);
  create_info_ = create_info;
  audio_params_.config = config;
  if (text_config_.model_dimension != 0) {
    if (GetOutputDim(config) != text_config_.model_dimension) {
      return absl::InvalidArgumentError(
          "Audio encoder output dimension must match text decoder model "
          "dimension.");
    }
  }
  if (create_info.hints.Check(ModelHints::kEnableHostMappedPointer)) {
    tensor_loader_->SetCreateTensorFn(
        absl::bind_front(&LlmMetal::CreateModelTensorMmaped, this, env,
                         tensor_loader_.get(), &model_tensors_));
  } else {
    tensor_loader_->SetCreateTensorFn(absl::bind_front(
        &LlmMetal::CreateModelTensor, this, env, &model_tensors_));
  }
  const auto& gpu_info = env->GetInfo();
  auto audio_encoder_builder = std::make_unique<AudioEncoderBuilder>(
      config, output_batch_size, gpu_info, create_info, tensor_loader_.get());

  // Init tensors
  int num_cache_layers = audio_params_.config.transformer.stack_size;
  if (audio_encoder_builder->UseGemmaModelV3NPlus() &&
      audio_params_.config.transformer.kv_cache_shared_start_layer
          .has_value()) {
    num_cache_layers =
        *audio_params_.config.transformer.kv_cache_shared_start_layer;
  }

  audio_params_.transformer_params.params_i32 =
      std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      env->device(), audio_encoder_builder->GetParamsTensorDescriptor(),
      audio_params_.transformer_params.params_i32.get()));
  // only need kv cache for streaming
  if (audio_encoder_builder->IsStreaming()) {
    audio_params_.cache.key_cached_tensors =
        std::vector<std::unique_ptr<MetalSpatialTensor>>(num_cache_layers);
    audio_params_.cache.value_cached_tensors =
        std::vector<std::unique_ptr<MetalSpatialTensor>>(num_cache_layers);
    for (int i = 0; i < num_cache_layers; ++i) {
      audio_params_.cache.key_cached_tensors[i] =
          std::make_unique<MetalSpatialTensor>();
      ABSL_RETURN_IF_ERROR(CreateTensor(
          env->device(),
          audio_encoder_builder->GetKCacheConformerTensorDescriptor(),
          audio_params_.cache.key_cached_tensors[i].get()));
      audio_params_.cache.value_cached_tensors[i] =
          std::make_unique<MetalSpatialTensor>();
      ABSL_RETURN_IF_ERROR(CreateTensor(
          env->device(),
          audio_encoder_builder->GetVCacheConformerTensorDescriptor(),
          audio_params_.cache.value_cached_tensors[i].get()));
    }
  }

  audio_params_.feature = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env->device(), audio_encoder_builder->GetFeatureDescriptor(),
                   audio_params_.feature.get()));
  audio_params_.mask = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(env->device(),
                                    audio_encoder_builder->GetMaskDescriptor(),
                                    audio_params_.mask.get()));
  if (audio_encoder_builder->IsStreaming()) {
    audio_params_.stream_cumsum = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->device(), audio_encoder_builder->GetCumsumStreamDescriptor(),
        audio_params_.stream_cumsum.get()));
    audio_params_.stream_depthwise_tensors =
        std::vector<std::unique_ptr<MetalSpatialTensor>>(num_cache_layers);
    for (int i = 0; i < num_cache_layers; ++i) {
      audio_params_.stream_depthwise_tensors[i] =
          std::make_unique<MetalSpatialTensor>();
      ABSL_RETURN_IF_ERROR(CreateTensor(
          env->device(), audio_encoder_builder->GetDepthwiseStreamDescriptor(),
          audio_params_.stream_depthwise_tensors[i].get()));
    }
  }
  audio_params_.signal = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env->device(), audio_encoder_builder->GetSignalDescriptor(),
                   audio_params_.signal.get()));
  // Init models
  ABSL_RETURN_IF_ERROR(
      InitAudioEncoderModel(audio_encoder_builder.get(), command_queue, env));
  // Since we have different cache tensors for v3, we need to update the
  // clean cache model.
  // no kv cache for v3 non-streaming
  if (audio_encoder_builder->IsStreaming()) {
    std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
    std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
    GpuModelBuilder::TensorHandle stream_cumsum_handle;
    std::vector<GpuModelBuilder::TensorHandle> stream_depthwise_tensor_handles;
    ABSL_ASSIGN_OR_RETURN(
        GpuModel gpu_model,
        audio_encoder_builder->CreateCleanCacheModelv3(
            &key_cached_tensor_handles, &value_cached_tensor_handles,
            stream_cumsum_handle, &stream_depthwise_tensor_handles));
    CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
        create_info_, audio_encoder_builder->GetExternalImmutableTensors(),
        audio_encoder_builder->GetExternalMutableTensors());
    for (int i = 0; i < key_cached_tensor_handles.size(); ++i) {
      create_info_copy
          .external_immutable_tensors[key_cached_tensor_handles[i].id] =
          audio_params_.cache.key_cached_tensors[i].get();
    }
    for (int i = 0; i < value_cached_tensor_handles.size(); ++i) {
      create_info_copy
          .external_immutable_tensors[value_cached_tensor_handles[i].id] =
          audio_params_.cache.value_cached_tensors[i].get();
    }
    create_info_copy.external_immutable_tensors[stream_cumsum_handle.id] =
        audio_params_.stream_cumsum.get();
    for (int i = 0; i < stream_depthwise_tensor_handles.size(); ++i) {
      create_info_copy
          .external_immutable_tensors[stream_depthwise_tensor_handles[i].id] =
          audio_params_.stream_depthwise_tensors[i].get();
    }

    audio_params_.cache.clean_cache =
        std::make_unique<InferenceContext>(InferenceContext());
    ABSL_RETURN_IF_ERROR(audio_params_.cache.clean_cache->InitFromGpuModel(
        create_info_copy, &gpu_model, env->device(), nullptr));
  }
  // Init ops
  if (audio_params_.transformer_params.params_i32) {
    audio_params_.transformer_params.write_i32_params =
        std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(InitMetalOp(
        env,
        CreateWriteParamsOp(
            audio_params_.transformer_params.params_i32->GetDescriptor(),
            audio_params_.transformer_params.params_i32->Channels()),
        audio_params_.transformer_params.write_i32_params.get()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<MetalSpatialTensor>>>
LlmMetal::InitKvTensors(MetalDevice& device,
                        const TensorDescriptor& tensor_desc,
                        int num_cache_layers) {
  return InitKvTensorsLocalGlobal(device, tensor_desc,
                                  /*global_desc=*/tensor_desc,
                                  /*num_local_layers_per_global=*/1,
                                  num_cache_layers,
                                  /*global_layer_offset=*/std::nullopt);
}

absl::Status LlmMetal::InitSharedTensors(
    TransformerBuilder* transformer_builder, TransformerCache& cache,
    TransformerParams& params, int num_cache_layers, MetalDevice* device) {
  params.params_i32 = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      device->device(), transformer_builder->GetParamsTensorDescriptor(),
      params.params_i32.get()));
  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    ABSL_ASSIGN_OR_RETURN(
        cache.key_cached_tensors,
        InitKvTensorsLocalGlobal(
            *device, transformer_builder->GetKCacheTensorDescriptor(),
            transformer_builder->GetKCacheTensorDescriptor(/*is_global=*/true),
            text_config_.num_local_layers_per_global, num_cache_layers,
            text_config_.global_layer_offset));
    ABSL_ASSIGN_OR_RETURN(
        cache.value_cached_tensors,
        InitKvTensorsLocalGlobal(
            *device, transformer_builder->GetVCacheTensorDescriptor(),
            transformer_builder->GetVCacheTensorDescriptor(/*is_global=*/true),
            text_config_.num_local_layers_per_global, num_cache_layers,
            text_config_.global_layer_offset));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        cache.key_cached_tensors,
        InitKvTensors(*device, transformer_builder->GetKCacheTensorDescriptor(),
                      num_cache_layers));
    ABSL_ASSIGN_OR_RETURN(
        cache.value_cached_tensors,
        InitKvTensors(*device, transformer_builder->GetVCacheTensorDescriptor(),
                      num_cache_layers));
  }

  tokens_ids_ = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      device->device(), transformer_builder->GetTokensTensorDescriptor(),
      tokens_ids_.get()));
  return absl::OkStatus();
}

absl::Status LlmMetal::InitModel(TransformerBuilder* transformer_builder,
                                 int sequence_size, int batch_size,
                                 MetalDevice* device) {
  auto& model = models_[sequence_size];
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle media_handle;
  std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> vision_ple_tensor_handles;
  GpuModel gpu_model;
  if (transformer_builder->UseGoogleInternalModelV2()) {
    ABSL_ASSIGN_OR_RETURN(
        gpu_model,
        transformer_builder->CreateStackedTransformerModelV2(
            sequence_size, batch_size, &model.input,
            media_cache_size_ > 0 ? &media_handle : nullptr, &tokens_ids_handle,
            &params_i32_handle, &key_cached_tensor_handles,
            &value_cached_tensor_handles, &model.output, &model.ple_weights,
            &model.ngrammer_weights,
            media_cache_size_ > 0 ? &vision_ple_tensor_handles : nullptr,
            /*segment_pos_tensor=*/nullptr,
            /*global_mask_tensor=*/nullptr,
            /*local_mask_tensor=*/nullptr));
  } else if (transformer_builder->UseGemmaModelV3NPlus()) {
    ABSL_ASSIGN_OR_RETURN(
        gpu_model, transformer_builder->CreateStackedTransformerModelV3(
                       sequence_size, batch_size, &model.input,
                       media_cache_size_ > 0 ? &media_handle : nullptr,
                       &tokens_ids_handle, &params_i32_handle,
                       &key_cached_tensor_handles, &value_cached_tensor_handles,
                       &model.output, &model.ple_weights, nullptr,
                       /*segment_pos_tensor=*/nullptr,
                       /*global_mask_tensor=*/nullptr,
                       /*local_mask_tensor=*/nullptr,
                       /*external_embeddings_tensor=*/nullptr,
                       /*external_ple_embeddings_tensor=*/nullptr,
                       /*input_activations_handle=*/nullptr,
                       /*output_activations_handle=*/nullptr, kGemmaBaseName));
    model.debug_tensors = transformer_builder->GetDebugTensors();
  } else {
    ABSL_ASSIGN_OR_RETURN(gpu_model,
                          transformer_builder->CreateStackedTransformerModel(
                              sequence_size, batch_size, &model.input,
                              media_cache_size_ > 0 ? &media_handle : nullptr,
                              &params_i32_handle, &key_cached_tensor_handles,
                              &value_cached_tensor_handles, &model.output));
  }

  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  transformer_builder->ChangePrecisionIfBfloat16Fix(
      CalculationsPrecision::kF32);

  model.lora_tensors = transformer_builder->lora_tensors();
  {  // Providing binding info for external shared tensors
    if (transformer_builder->UseGoogleInternalModelV2() ||
        transformer_builder->UseGemmaModelV3NPlus()) {
      create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
          tokens_ids_.get();
    }
    create_info_copy.external_immutable_tensors[params_i32_handle.id] =
        params_.params_i32.get();
    for (int i = 0; i < key_cached_tensor_handles.size(); ++i) {
      create_info_copy
          .external_immutable_tensors[key_cached_tensor_handles[i].id] =
          cache_.key_cached_tensors[i].get();
    }
    for (int i = 0; i < value_cached_tensor_handles.size(); ++i) {
      create_info_copy
          .external_immutable_tensors[value_cached_tensor_handles[i].id] =
          cache_.value_cached_tensors[i].get();
    }
    if (media_cache_size_ > 0) {  // has multimodal
      create_info_copy.external_immutable_tensors[media_handle.id] =
          media_embeddings_.get();
      if (text_config_.per_layer_embedding_dimension.has_value()) {
        if (transformer_builder->UseGoogleInternalModelV2()) {
          for (int i = 0;
               i < text_config_.stack_size - text_config_.extra_stack_size;
               ++i) {
            create_info_copy
                .external_immutable_tensors[vision_ple_tensor_handles[i].id] =
                vision_ple_tensors_[i].get();
          }
        }
      }
    }
  }
  model.seq_size = sequence_size;
  model.context = std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(model.context->InitFromGpuModel(
      create_info_copy, &gpu_model, device->device(), nullptr));
  return absl::OkStatus();
}

absl::Status LlmMetal::InitPostProcessingModel(
    TransformerBuilder* transformer_builder, MetalDevice* device) {
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle stacked_transformer_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle params_f32_handle;
  GpuModelBuilder::TensorHandle constraint_mask_handle;
  GpuModelBuilder::TensorHandle* pre_softmax_handle_ptr = nullptr;
  GpuModelBuilder::TensorHandle* topk_tokens_handle_ptr = nullptr;
  GpuModelBuilder::TensorHandle* topk_probabilities_handle_ptr = nullptr;
#if LLM_METAL_DEBUG_MLDRIFT_TENSORS
  // Keep the logits and TopK results as model outputs so they can be dumped.
  llm_debug::PostProcessingOutputs debug_outputs;
  pre_softmax_handle_ptr = &debug_outputs.pre_softmax;
  topk_tokens_handle_ptr = &debug_outputs.topk_tokens;
  topk_probabilities_handle_ptr = &debug_outputs.topk_probabilities;
#endif  // LLM_METAL_DEBUG_MLDRIFT_TENSORS
  ABSL_ASSIGN_OR_RETURN(
      GpuModel gpu_model,
      transformer_builder->CreatePostProcessingModel(
          &input_handle, &params_i32_handle, &params_f32_handle,
          &tokens_ids_handle, &stacked_transformer_input_handle,
          pre_softmax_handle_ptr, topk_tokens_handle_ptr,
          topk_probabilities_handle_ptr, &constraint_mask_handle,
          /*sequence_size=*/1));
#if LLM_METAL_DEBUG_MLDRIFT_TENSORS
  llm_debug::RegisterPostProcessingOutputs(debug_outputs);
#endif  // LLM_METAL_DEBUG_MLDRIFT_TENSORS

  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  create_info_copy.external_immutable_tensors[input_handle.id] =
      model_one_->context->GetTensor(model_one_->output.id);

  params_f32_ = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      device->device(), params_f32_handle.tensor_desc, params_f32_.get()));

  create_info_copy.external_immutable_tensors[params_i32_handle.id] =
      params_.params_i32.get();
  create_info_copy.external_immutable_tensors[params_f32_handle.id] =
      params_f32_.get();
  create_info_copy
      .external_immutable_tensors[stacked_transformer_input_handle.id] =
      model_one_->context->GetTensor(model_one_->input.id);
  create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
      tokens_ids_.get();

  constraint_mask_ = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(device->device(),
                                    constraint_mask_handle.tensor_desc,
                                    constraint_mask_.get()));
  ABSL_RETURN_IF_ERROR(
      UpdateConstraintMask(odml::infra::llm_utils::AllAllowedBitmap()));
  create_info_copy.external_immutable_tensors[constraint_mask_handle.id] =
      constraint_mask_.get();
  post_processing_ = std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(post_processing_->InitFromGpuModel(
      create_info_copy, &gpu_model, device->device(), nullptr));
  return absl::OkStatus();
}

absl::Status LlmMetal::InitCleanCacheModel(
    TransformerBuilder* transformer_builder, TransformerCache& cache,
    TransformerParams& params, MetalDevice* device) {
  std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
  GpuModelBuilder::TensorHandle params_i32_handle;
  ABSL_ASSIGN_OR_RETURN(GpuModel gpu_model,
                        transformer_builder->CreateCleanCacheModel(
                            &key_cached_tensor_handles,
                            &value_cached_tensor_handles, &params_i32_handle));
  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  for (int i = 0; i < key_cached_tensor_handles.size(); ++i) {
    create_info_copy
        .external_immutable_tensors[key_cached_tensor_handles[i].id] =
        cache.key_cached_tensors[i].get();
  }
  for (int i = 0; i < value_cached_tensor_handles.size(); ++i) {
    create_info_copy
        .external_immutable_tensors[value_cached_tensor_handles[i].id] =
        cache.value_cached_tensors[i].get();
  }
  create_info_copy.external_immutable_tensors[params_i32_handle.id] =
      params.params_i32.get();

  cache.clean_cache = std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(cache.clean_cache->InitFromGpuModel(
      create_info_copy, &gpu_model, device->device(), nullptr));
  return absl::OkStatus();
}

absl::Status LlmMetal::InitAudioEncoderModel(
    AudioEncoderBuilder* audio_encoder_builder,
    id<MTLCommandQueue> command_queue, Environment* env) {
  TensorHandle audio_feature_handle, audio_mask_handle, params_i32_handle,
      audio_cumsum_stream_handle, signal_handle;
  std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> stream_depthwise_tensor_handles;

  ABSL_ASSIGN_OR_RETURN(
      GpuModel gpu_model,
      audio_encoder_builder->CreateAudioEncoderModel(
          &audio_feature_handle, &audio_mask_handle, &params_i32_handle,
          &key_cached_tensor_handles, &value_cached_tensor_handles,
          &audio_cumsum_stream_handle, &stream_depthwise_tensor_handles,
          &signal_handle, &audio_params_.output_handle));
  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, audio_encoder_builder->GetExternalImmutableTensors(),
      audio_encoder_builder->GetExternalMutableTensors());
  {  // Providing binding info for external shared tensors
    create_info_copy.external_immutable_tensors[audio_feature_handle.id] =
        audio_params_.feature.get();
    create_info_copy.external_immutable_tensors[audio_mask_handle.id] =
        audio_params_.mask.get();
    create_info_copy.external_immutable_tensors[params_i32_handle.id] =
        audio_params_.transformer_params.params_i32.get();
    if (audio_encoder_builder->IsStreaming()) {
      create_info_copy
          .external_immutable_tensors[audio_cumsum_stream_handle.id] =
          audio_params_.stream_cumsum.get();
    }
    if (audio_encoder_builder->IsStreaming()) {
      for (int i = 0; i < key_cached_tensor_handles.size(); ++i) {
        create_info_copy
            .external_immutable_tensors[key_cached_tensor_handles[i].id] =
            audio_params_.cache.key_cached_tensors[i].get();
      }
      for (int i = 0; i < value_cached_tensor_handles.size(); ++i) {
        create_info_copy
            .external_immutable_tensors[value_cached_tensor_handles[i].id] =
            audio_params_.cache.value_cached_tensors[i].get();
      }
    }
    create_info_copy.external_immutable_tensors[signal_handle.id] =
        audio_params_.signal.get();
    if (audio_encoder_builder->IsStreaming()) {
      for (int i = 0; i < stream_depthwise_tensor_handles.size(); ++i) {
        create_info_copy
            .external_immutable_tensors[stream_depthwise_tensor_handles[i].id] =
            audio_params_.stream_depthwise_tensors[i].get();
      }
    }
  }
  // Upload values for rel pos emb
  if (IsModelv3(audio_params_.config.transformer)) {
    TensorDescriptor signal_desc = audio_params_.signal->GetDescriptor();
    TransformerConfig::RelativePositionalEmbeddingConfig rel_pe_cfg =
        audio_params_.config.transformer.relative_positional_embedding_config
            .value();
    signal_desc.UploadData(audio_encoder_builder->GetTimingSignal1dPos(
        rel_pe_cfg.min_timescale, rel_pe_cfg.max_timescale,
        rel_pe_cfg.past_horizon, rel_pe_cfg.future_horizon));
    ABSL_RETURN_IF_ERROR(audio_params_.signal->UploadDescriptorData(
        signal_desc, command_queue, /*wait_for_completion=*/false));
  }
  audio_params_.context =
      std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(audio_params_.context->InitFromGpuModel(
      create_info_copy, &gpu_model, env->device(), nullptr));
  return absl::OkStatus();
}

absl::Status LlmMetal::InitHelperOps(MetalDevice* device) {
  const auto& gpu_info = device->GetInfo();
  if (params_.params_i32) {
    params_.write_i32_params = std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(
        InitMetalOp(device,
                    CreateWriteParamsOp(params_.params_i32->GetDescriptor(),
                                        params_.params_i32->Channels()),
                    params_.write_i32_params.get()));
  }
  if (params_f32_) {
    write_f32_params_ = std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(
        InitMetalOp(device,
                    CreateWriteParamsOp(params_f32_->GetDescriptor(),
                                        params_f32_->Channels()),
                    write_f32_params_.get()));
  }
  for (auto& kv : models_) {
    auto& model = kv.second;
    model.copy_tokens_op = std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(InitMetalOp(
        device, CreateCopyTokensOrPositionsOp(model.input.tensor_desc),
        model.copy_tokens_op.get()));
    if (kv.first != 1 && model_one_->context) {
      model.copy_nth_op = std::make_unique<ComputeTask>();
      ABSL_RETURN_IF_ERROR(InitMetalOp(
          device,
          CreateCopyNthToSingleOp(gpu_info, model.output.tensor_desc,
                                  model_one_->output.tensor_desc),
          model.copy_nth_op.get()));
    }
  }
  copy_cache_ = std::make_unique<ComputeTask>();
  if (cache_.key_cached_tensors.empty() ||
      cache_.value_cached_tensors.empty()) {
    return absl::InternalError("Key or value cached tensors are empty.");
  }
  ABSL_RETURN_IF_ERROR(InitMetalOp(
      device,
      CreateCopyCacheOp(cache_.key_cached_tensors[0]->GetDescriptor(),
                        cache_.value_cached_tensors[0]->GetDescriptor(),
                        text_config_, output_batch_size_),
      copy_cache_.get()));
  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    const int global_id = text_config_.global_layer_offset.value_or(
        text_config_.num_local_layers_per_global);
    if (global_id < cache_.key_cached_tensors.size()) {
      copy_cache_global_ = std::make_unique<ComputeTask>();
      ABSL_RETURN_IF_ERROR(InitMetalOp(
          device,
          CreateCopyCacheOp(
              cache_.key_cached_tensors[global_id]->GetDescriptor(),
              cache_.value_cached_tensors[global_id]->GetDescriptor(),
              text_config_, output_batch_size_, /*is_global=*/true),
          copy_cache_global_.get()));
    }
  }

  if (media_embeddings_) {
    multimodal_embeddings_op_ = std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(InitMetalOp(
        device,
        CreateCopyMediaEmbeddingsOp(media_embeddings_->GetDescriptor(),
                                    media_embeddings_->GetDescriptor()),
        multimodal_embeddings_op_.get()));
  }

  return absl::OkStatus();
}

absl::Status LlmMetal::InitInputTokens(id<MTLCommandQueue> command_queue,
                                       const std::vector<int>& input_tokens,
                                       int time_step) {
  // If a single token is given as input and it exactly matches the previous
  // input, short circuit the initialization. The LLM engine will send a single
  // token to make sure the state is correct before generating output, but in
  // this case we know the state is already right.
  if (ready_tokens_ == 1 && num_output_tokens_ == 0 &&
      input_tokens.size() == 1 && input_tokens_.size() == time_step + 1 &&
      input_tokens_.back() == input_tokens[0] &&
      !constraint_mask_changed_since_last_input_) {
    return absl::OkStatus();
  }
  constraint_mask_changed_since_last_input_ = false;
  @autoreleasepool {
    ABSL_RETURN_IF_ERROR(CleanCache(command_queue, time_step));
    if (input_tokens_.size() < time_step &&
        !model_one_->ngrammer_weights.empty()) {
      return absl::InternalError("time_step is too large.");
    }
    input_tokens_.resize(time_step);
    ABSL_RETURN_IF_ERROR(
        InitInputTokensWithModelN(command_queue, input_tokens));
    input_tokens_.insert(input_tokens_.end(), input_tokens.begin(),
                         input_tokens.end());
    {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      {
        int token_index = input_tokens_.size() - 1;
        LlmRuntimeParams llm_params;
        llm_params.token_index_offset = token_index;
        llm_params.active_tokens = token_index + 1;
        llm_params.topk = sampler_params_.k();
        ABSL_RETURN_IF_ERROR(
            ExecuteUpdateIntParams(encoder, text_config_, params_, llm_params));
      }
      ABSL_RETURN_IF_ERROR(ExecutePostProcessing(encoder));
      [encoder endEncoding];
      [command_buffer commit];
#if LLM_METAL_DEBUG_MLDRIFT_TENSORS
      ABSL_RETURN_IF_ERROR(llm_debug::MaybeDumpPostProcessingLogits(
          command_buffer, device_, *post_processing_,
          /*position=*/static_cast<int>(input_tokens_.size()) - 1));
#endif  // LLM_METAL_DEBUG_MLDRIFT_TENSORS
    }
  }
  ready_tokens_ = 1;
  num_output_tokens_ = 0;
  need_sync_for_peek_token_ = true;
  return absl::OkStatus();
}

// @autoreleasepool in InitInputTokens covers this private method too.
absl::Status LlmMetal::InitInputTokensWithModelN(
    id<MTLCommandQueue> command_queue, const std::vector<int>& input_tokens) {
  if (input_tokens.empty()) {
    return absl::OkStatus();
  }
  ABSL_RETURN_IF_ERROR(input_token_ids_->WriteData(
      command_queue, absl::MakeConstSpan(input_tokens),
      /*wait_for_completion=*/false));
  const bool has_optional_input_layers = shared_kv_cache_tag_.has_value();
  shared_kv_cache_tag_ = std::nullopt;
  SetupOptionalNodes();
  // Create a cleanup fn to make sure state is restored correctly.
  auto enable_all_layers = absl::MakeCleanup([&] {
    if (has_optional_input_layers) {
      shared_kv_cache_tag_ = kSharedKvCacheLayersOptionTag;
      SetupOptionalNodes();
    }
  });
  size_t input_size = input_tokens.size();
  if (has_optional_input_layers) {
    // Reduce input size by 1 so a single token can be processed at the end of
    // input with all layers.
    input_size--;
  }
  ABSL_ASSIGN_OR_RETURN(auto input_groups,
                        GetGroupSizesForInput(input_size, group_sizes_));
  if (has_optional_input_layers) {
    input_groups.push_back(1);
  }
  int token_index_offset = 0;
  for (int i = 0; i < input_groups.size(); ++i) {
    int group_size = input_groups[i];
    // Last input token processed needs to enable all layers in case the output
    // tensor is needed after.
    if (has_optional_input_layers && i == input_groups.size() - 1) {
      RET_CHECK_EQ(group_size, 1);
      // The last group may have processed beyond the last token, so reset.
      token_index_offset = input_tokens.size() - group_size;
      std::move(enable_all_layers).Invoke();  // NOLINT(bugprone-use-after-move)
    }
    auto& model_n = models_[group_size];

    int previous_token = 0;
    if (token_index_offset > 0) {
      previous_token = input_tokens[token_index_offset - 1];
    } else if (!input_tokens_.empty()) {
      previous_token = input_tokens_.back();
    }
    ABSL_RETURN_IF_ERROR(UpdateEmbeddings(
        command_queue, model_n, previous_token,
        absl::MakeSpan(input_tokens).subspan(token_index_offset, group_size)));

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    {  // Prepare input.
      auto& copy_tokens_op = model_n.copy_tokens_op;
      ABSL_RETURN_IF_ERROR(
          copy_tokens_op->SetSrcBuffer(input_token_ids_.get(), 0));
      ABSL_RETURN_IF_ERROR(copy_tokens_op->SetDstTensor(
          model_n.context->GetTensor(model_n.input.id), 0));
      ABSL_RETURN_IF_ERROR(
          copy_tokens_op->SetInt("token_offset", token_index_offset));
      ABSL_RETURN_IF_ERROR(
          copy_tokens_op->SetInt("tokens_count", input_tokens.size()));
      ABSL_RETURN_IF_ERROR(copy_tokens_op->UpdateParams());
      copy_tokens_op->Encode(encoder);
    }
    const int active_positions =
        std::min(token_index_offset + model_n.seq_size,
                 static_cast<int>(input_tokens.size()));
    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = input_tokens_.size() + token_index_offset;
    llm_params.active_tokens = input_tokens_.size() + active_positions;
    llm_params.topk = sampler_params_.k();
    ABSL_RETURN_IF_ERROR(
        ExecuteUpdateIntParams(encoder, text_config_, params_, llm_params));
    [encoder endEncoding];
    [command_buffer commit];
    if (model_n.seq_size == 1) {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      model_n.context->EncodeWithEncoder(encoder);
      [encoder endEncoding];
      [command_buffer commit];
    } else {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      model_n.context->EncodeWithCommandBuffer(command_buffer,
                                               /*tasks_per_encoder=*/32);
      [command_buffer commit];
    }
#if LLM_METAL_DEBUG_MLDRIFT_TENSORS
    ABSL_RETURN_IF_ERROR(llm_debug::MaybeDumpPrefillTensors(
        command_queue, device_, *model_n.context, model_n.debug_tensors,
        /*batch_index=*/i, /*num_batches=*/input_groups.size()));
#endif  // LLM_METAL_DEBUG_MLDRIFT_TENSORS
    token_index_offset += group_size;
  }
  // Copy model_n last token from output to model_one output.
  auto& last_model = models_[input_groups.back()];
  if (last_model.seq_size != 1) {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    auto& copy_nth_op = last_model.copy_nth_op;
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetSrcTensor(
        last_model.context->GetTensor(last_model.output.id), 0));
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetDstTensor(
        model_one_->context->GetTensor(model_one_->output.id), 0));
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetInt(
        "nth", (input_tokens.size() - 1) % last_model.seq_size));
    ABSL_RETURN_IF_ERROR(copy_nth_op->UpdateParams());
    copy_nth_op->Encode(encoder);
    [encoder endEncoding];
    [command_buffer commit];
  }
  return absl::OkStatus();
}

absl::Status LlmMetal::GetNextTokens(id<MTLCommandQueue> command_queue,
                                     std::vector<int>* output_tokens) {
  std::vector<std::vector<int>> temp(1);
  temp[0].resize(output_tokens->size());
  ABSL_RETURN_IF_ERROR(GetNextTokens(command_queue, &temp));
  *output_tokens = std::move(temp[0]);
  return absl::OkStatus();
}

absl::StatusOr<id<MTLCommandBuffer>> LlmMetal::ProcessNextOutputToken(
    id<MTLCommandQueue> command_queue) {
  int time_step = input_tokens_.size() + ready_tokens_ - 1;
  id<MTLCommandBuffer> cb_for_sync;
  if (!model_one_->ple_weights.empty() ||
      !model_one_->ngrammer_weights.empty()) {
    cb_for_sync = [command_queue commandBuffer];
    [cb_for_sync commit];
  }

  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  {
    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = time_step;
    llm_params.active_tokens = time_step + 1;
    llm_params.topk = sampler_params_.k();
    ABSL_RETURN_IF_ERROR(
        ExecuteUpdateIntParams(encoder, text_config_, params_, llm_params));
  }
  model_one_->context->EncodeWithEncoder(encoder);
  ABSL_RETURN_IF_ERROR(ExecutePostProcessing(encoder));
  [encoder endEncoding];
  if (!model_one_->ple_weights.empty() ||
      !model_one_->ngrammer_weights.empty()) {
    // Waiting as late as possible to allow maximum parallelism.
    [cb_for_sync waitUntilCompleted];
    RET_CHECK(tokens_ids_->GetStorageType() == TensorStorageType::kBuffer)
        << "The tokens_ids_ tensor must be a buffer based.";
    int32_t* output_data = (int32_t*)[tokens_ids_->GetBufferHandle() contents];
    ABSL_RETURN_IF_ERROR(UpdateEmbeddings(
        command_queue, *model_one_, output_data[time_step - 1],
        {output_data[time_step]}, /*require_load_with_queue=*/false));
  }
  [command_buffer commit];
#if LLM_METAL_DEBUG_MLDRIFT_TENSORS
  // Waits for this step to finish (serializing decode) while dumping is
  // enabled, so the next step does not overwrite the logits before download.
  ABSL_RETURN_IF_ERROR(llm_debug::MaybeDumpPostProcessingLogits(
      command_buffer, device_, *post_processing_, /*position=*/time_step));
  ABSL_RETURN_IF_ERROR(llm_debug::MaybeDumpDecodeStepTensors(
      command_queue, device_, *model_one_->context, model_one_->debug_tensors,
      text_config_, cache_.key_cached_tensors, cache_.value_cached_tensors,
      /*position=*/time_step));
#endif  // LLM_METAL_DEBUG_MLDRIFT_TENSORS
  ready_tokens_++;
  return command_buffer;
}

absl::Status LlmMetal::GetNextTokens(
    id<MTLCommandQueue> command_queue,
    std::vector<std::vector<int>>* output_tokens) {
  if (output_tokens->size() > 1 && (!model_one_->ple_weights.empty() ||
                                    !model_one_->ngrammer_weights.empty())) {
    // TODO: b/355494120 - Support multi output with dynamic PLE.
    return absl::InternalError(
        "Multi-output is not supported with dynamic embedding loading.");
  }
  const int batch_size = output_tokens->size();
  const int next_tokens_count = (*output_tokens)[0].size();

  // Unlike WebGPU, Metal requires an explicit waitUntilCompleted before we can
  // bring data back from GPU to CPU. So we must make sure at least one such
  // call is made for any tokens still processing.
  id<MTLCommandBuffer> cb_for_sync = nullptr;
  if (need_sync_for_peek_token_ &&
      (ready_tokens_ >= next_tokens_count + num_output_tokens_)) {
    // We have all tokens processing or being processed. Add a sync to make sure
    // they have finished.
    @autoreleasepool {
      cb_for_sync = [command_queue commandBuffer];
      [cb_for_sync commit];
    }
    need_sync_for_peek_token_ = false;
  }
  while (ready_tokens_ < next_tokens_count + num_output_tokens_) {
    @autoreleasepool {
      ABSL_ASSIGN_OR_RETURN(auto command_buffer,
                            ProcessNextOutputToken(command_queue));
      if (ready_tokens_ == next_tokens_count + num_output_tokens_) {
        cb_for_sync = command_buffer;
        need_sync_for_peek_token_ = false;
      }
    }
  }

  @autoreleasepool {
    // Queue the peek token for the NEXT round before waiting, which allows the
    // GPU to compute the peek token while the CPU downloads data.
    // Note that time_stemp for this `ProcessNextOutputToken` call uses
    // `ready_tokens_` instead of `num_output_tokens_`.
    if (ready_tokens_ == next_tokens_count + num_output_tokens_) {
      ABSL_RETURN_IF_ERROR(ProcessNextOutputToken(command_queue).status());
      // We're not returning this token immediately, so we don't call
      // `waitUntilCompleted`. This avoids double-syncing, but means we'll need
      // to ensure a sync before we try to download this token, either through
      // `PeekNextToken` or through the next call of this function.
      need_sync_for_peek_token_ = true;
    }
  }

  if (cb_for_sync != nullptr) {
    [cb_for_sync waitUntilCompleted];
  }

  const MetalSpatialTensor* gpu_tensor = tokens_ids_.get();

  Tensor<BHWC, DataType::kInt32> output;
  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, device_));
  desc.DownloadData(&output);
  for (int b = 0; b < batch_size; ++b) {
    auto& tokens = (*output_tokens)[b];
    for (int i = 0; i < tokens.size(); ++i) {
      const int time_step = i + input_tokens_.size() + num_output_tokens_;
      tokens[i] = output.data[time_step * batch_size + b];
    }
  }

  num_output_tokens_ += next_tokens_count;

  // A single output stream can be treated as input tokens. This allows models
  // with dynamic embeddings to continue processing off of previous output.
  if (output_tokens->size() == 1) {
    const auto& output = (*output_tokens)[0];
    input_tokens_.insert(input_tokens_.end(), output.begin(), output.end());
    ready_tokens_ -= output.size();
    num_output_tokens_ -= output.size();
  }
  return absl::OkStatus();
}

absl::Status LlmMetal::FillAudioEmbeddingsWithTensorBuffer(
    MetalDevice& device, id<MTLCommandQueue> command_queue,
    const ::litert::TensorBuffer& audio_tensor_buffer, int index) {
  TensorDescriptor desc(
      DataType::kFloat16,
      ml_drift::metal::GetFastestStorageType(device.GetInfo()), Layout::kBHWDC);

  LITERT_ASSIGN_OR_RETURN(auto tensor_type, audio_tensor_buffer.TensorType());
  if (tensor_type.Layout().Dimensions().size() != 5) {
    return absl::InvalidArgumentError(
        "Audio embeddings must have 5 dimensions [b, h, w, d, c].");
  }
  desc.SetBHWDCShape(BHWDC(tensor_type.Layout().Dimensions()[0],
                           tensor_type.Layout().Dimensions()[1],
                           tensor_type.Layout().Dimensions()[2],
                           tensor_type.Layout().Dimensions()[3],
                           tensor_type.Layout().Dimensions()[4]));

  LITERT_ASSIGN_OR_RETURN(
      auto span,
      ::litert::lm::ReferTensorBufferAsSpan<float>(audio_tensor_buffer));
  desc.UploadData(span.data());

  MetalSpatialTensor metal_audio_tensor;
  ABSL_RETURN_IF_ERROR(
      CreateTensor(device.device(), desc, &metal_audio_tensor));

  ABSL_RETURN_IF_ERROR(metal_audio_tensor.UploadDescriptorData(
      desc, command_queue, /*wait_for_completion=*/false));

  return FillAudioEmbeddings(metal_audio_tensor, index, command_queue);
}

absl::Status LlmMetal::FillAudioEmbeddings(MetalSpatialTensor& audio_tensor,
                                           int index,
                                           id<MTLCommandQueue> command_queue) {
  if (media_cache_size_ == 0) {
    return absl::InvalidArgumentError("Media embedding cache was not set.");
  }
  return UpdateMultimodalEmbeddings(command_queue, &audio_tensor,
                                    media_embeddings_.get(), index);
}

absl::Status LlmMetal::FillVisionEmbeddings(
    MetalDevice& device, id<MTLCommandQueue> command_queue,
    const TensorFloat32& vision_embedding_tensor,
    const std::vector<TensorFloat32>& vision_ple_tensors, int image_index) {
  if (media_cache_size_ == 0) {
    return absl::InvalidArgumentError("Media embedding cache was not set.");
  }
  if (image_index >= text_config_.max_num_images) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Image index %d must be less than %d.", image_index,
                        text_config_.max_num_images));
  }
  RET_CHECK(vision_embedding_tensor.shape.b == 1)
      << "The batch size of the vision embedding tensor must be 1.";
  RET_CHECK(vision_embedding_tensor.shape.h == 1)
      << "The height of the vision embedding tensor must be 1.";
  RET_CHECK_EQ(vision_embedding_tensor.shape.w % text_config_.vision_tokens_num,
               0)
      << "The number of vision tokens should be a multiple of "
      << text_config_.vision_tokens_num;
  RET_CHECK(vision_embedding_tensor.shape.c == text_config_.model_dimension)
      << "The channel of the vision embedding tensor must be "
      << text_config_.model_dimension;
  const int index = image_index * text_config_.vision_tokens_num;
  ABSL_RETURN_IF_ERROR(FillMediaEmbeddings(device, command_queue,
                                           vision_embedding_tensor, index));
  if (!vision_ple_tensors.empty()) {
    return FillVisionPleEmbeddings(device, command_queue, vision_ple_tensors,
                                   index);
  }
  return absl::OkStatus();
}

absl::Status LlmMetal::UpdateSamplerParams(
    const SamplerParameters& new_params,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  if (new_params.k() > max_top_k_) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Top-K value ", new_params.k(), " must be <= ", max_top_k_));
  }
  sampler_params_ = new_params;
  rand_gen_ = rand_gen;
  // If the temperature is 0, then sampling collapses to ArgMax.
  if (sampler_params_.temperature() < epsilon_) {
    sampler_params_.set_temperature(1.0f);
    sampler_params_.set_k(1);
  }
  return absl::OkStatus();
}

absl::Status LlmMetal::CleanCache(id<MTLCommandQueue> command_queue,
                                  int time_step,
                                  bool skip_ringbuffer_delta_check) {
  LlmRuntimeParams llm_params;
  llm_params.token_index_offset = time_step;
  llm_params.active_tokens =
      time_step == 0 ? 0 : input_tokens_.size();  // For ringbuffer indexing

  int local_cache_mem =
      TransformerBuilder::GetCacheSize(text_config_, /*is_global=*/false);
  // If we have a ringbuffer mid-use...
  if (!skip_ringbuffer_delta_check &&
      llm_params.active_tokens > local_cache_mem) {
    // ... make sure we don't try to "jump back" too far. Just output an error
    // for now -- this should be handled in the engines.
    int history_buffer_size =
        local_cache_mem - *text_config_.sliding_window_size;
    if (history_buffer_size <= *llm_params.active_tokens - time_step) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cannot rewind to time_step %d from %d. Ringbuffer size is %d with "
          "sliding window of %d",
          time_step, *llm_params.active_tokens, local_cache_mem,
          *text_config_.sliding_window_size));
    }
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    ABSL_RETURN_IF_ERROR(
        ExecuteUpdateIntParams(encoder, text_config_, params_, llm_params));
    cache_.clean_cache->EncodeWithEncoder(encoder);
    [encoder endEncoding];
    [command_buffer commit];
  }
  if (time_step < input_tokens_.size()) {
    input_tokens_.resize(time_step);
  }
  return absl::OkStatus();
}

absl::Status LlmMetal::CleanAudioCache(id<MTLCommandQueue> command_queue,
                                       int time_step) {
  if (audio_params_.config.transformer.relative_positional_embedding_config
          .has_value() &&
      audio_params_.config.feature.pad_for_streaming == 0) {
    return absl::InvalidArgumentError("No KV cache for v3 non-streaming.");
  }
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = time_step;
    ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
        encoder, text_config_, audio_params_.transformer_params, llm_params));
    [encoder endEncoding];
    [command_buffer commit];
  }
  return absl::OkStatus();
}

absl::StatusOr<ContextData> LlmMetal::SaveContext(
    MetalDevice& device, id<MTLCommandQueue> command_queue, bool local_only) {
  const int num_cache_layers = cache_.key_cached_tensors.size();
  // Init kv tensors to save to.
  if (cache_.key_cached_tensors.empty()) {
    return absl::InternalError("Key or value cached tensors are empty.");
  }
  // NOTE: This assumes we don't start off an architecture with two global
  // layers. That has always been the case so far, but if it changes, our logic
  // here will need to change too.
  const int local_id = IsLayerGlobal(0) ? 1 : 0;
  const DataType data_type =
      cache_.key_cached_tensors[local_id]->GetDescriptor().GetDataType();
  const size_t local_cache_size =
      TransformerBuilder::GetCacheSize(text_config_, /*is_global=*/false);
  const size_t local_alloc_size =
      std::max<size_t>(1, std::min<size_t>(input_tokens_.size(), local_cache_size));
  std::vector<std::unique_ptr<MetalSpatialTensor>> key_tensors;
  std::vector<std::unique_ptr<MetalSpatialTensor>> value_tensors;
  const TensorDescriptor k_local_desc =
      TransformerBuilder::GetKCacheTensorDescriptor(
          text_config_, output_batch_size_, local_alloc_size, data_type);
  const TensorDescriptor v_local_desc =
      TransformerBuilder::GetVCacheTensorDescriptor(
          text_config_, output_batch_size_, local_alloc_size, data_type);
  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    std::optional<TensorDescriptor> k_global_desc;
    std::optional<TensorDescriptor> v_global_desc;
    if (!local_only) {
      const size_t global_cache_size =
          TransformerBuilder::GetCacheSize(text_config_, /*is_global=*/true);
      const size_t global_alloc_size = std::max<size_t>(
          1, std::min<size_t>(input_tokens_.size(), global_cache_size));
      k_global_desc = TransformerBuilder::GetKCacheTensorDescriptor(
          text_config_, output_batch_size_, global_alloc_size, data_type,
          /*is_global=*/true);
      v_global_desc = TransformerBuilder::GetVCacheTensorDescriptor(
          text_config_, output_batch_size_, global_alloc_size, data_type,
          /*is_global=*/true);
    }
    ABSL_ASSIGN_OR_RETURN(
        key_tensors, InitKvTensorsLocalGlobal(
                         device, k_local_desc, k_global_desc,
                         text_config_.num_local_layers_per_global,
                         num_cache_layers, text_config_.global_layer_offset));
    ABSL_ASSIGN_OR_RETURN(
        value_tensors, InitKvTensorsLocalGlobal(
                           device, v_local_desc, v_global_desc,
                           text_config_.num_local_layers_per_global,
                           num_cache_layers, text_config_.global_layer_offset));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        key_tensors, InitKvTensors(device, k_local_desc, num_cache_layers));
    ABSL_ASSIGN_OR_RETURN(
        value_tensors, InitKvTensors(device, v_local_desc, num_cache_layers));
  }
  @autoreleasepool {
    // Copy kv tensors to new kv tensors.
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    for (int i = 0; i < num_cache_layers; ++i) {
      const bool is_global = IsLayerGlobal(i);
      if (local_only && is_global) {
        continue;
      }
      const size_t layer_cache_size =
          TransformerBuilder::GetCacheSize(text_config_, is_global);
      const size_t layer_copy_size =
          std::min<size_t>(input_tokens_.size(), layer_cache_size);
      ABSL_RETURN_IF_ERROR(ExecuteCopyCache(
          encoder, layer_cache_size, cache_.key_cached_tensors[i].get(),
          cache_.value_cached_tensors[i].get(), layer_copy_size,
          key_tensors[i].get(), value_tensors[i].get(), is_global));
    }
    [encoder endEncoding];
    [command_buffer commit];
  }
  // Place everything inside context data.
  ContextData context_data;
  std::move(key_tensors.begin(), key_tensors.end(),
            std::back_inserter(context_data.key_tensors));
  std::move(value_tensors.begin(), value_tensors.end(),
            std::back_inserter(context_data.value_tensors));
  context_data.lora_id = current_lora_id_;
  context_data.tokens = input_tokens_;
  return context_data;
}

absl::Status LlmMetal::RestoreContext(id<MTLCommandQueue> command_queue,
                                      const ContextData& context_data,
                                      bool local_only) {
  // Check that the context data is valid.
  if (context_data.lora_id != current_lora_id_) {
    return absl::InternalError("LoRA ID for context does not match.");
  }
  if (context_data.key_tensors.empty()) {
    return absl::InternalError("Context data key tensors are empty.");
  }
  if (cache_.key_cached_tensors.empty()) {
    return absl::InternalError("Cache key tensors are empty.");
  }
  // NOTE: This assumes we don't start off an architecture with two global
  // layers. That has always been the case so far, but if it changes, our logic
  // here will need to change too.
  const int local_id = IsLayerGlobal(0) ? 1 : 0;
  if (context_data.key_tensors[local_id]->GetDescriptor().GetDataType() !=
      cache_.key_cached_tensors[local_id]->GetDataType()) {
    return absl::InternalError("Data types do not match.");
  }
  if (!local_only) {
    // Copy over misc data.
    input_tokens_ = context_data.tokens;
    ready_tokens_ = 0;
    num_output_tokens_ = 0;
    // We probably don't need this extra sync, because we have no tokens pending
    // on GPU but since we're changing out the full context, it's safer to be
    // extra-careful and sync anyways.
    need_sync_for_peek_token_ = true;

    TensorDescriptor desc = tokens_ids_->GetDescriptor();
    std::vector<int> tokens_data = context_data.tokens;
    tokens_data.resize(desc.GetBHWDCShape().DimensionsProduct());
    desc.UploadData(tokens_data.data());
    ABSL_RETURN_IF_ERROR(tokens_ids_->UploadDescriptorData(
        desc, command_queue, /*wait_for_completion=*/false));
  }

  @autoreleasepool {
    // Copy over the cache.
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    for (int i = 0; i < context_data.key_tensors.size(); ++i) {
      const bool is_global = IsLayerGlobal(i);
      if (local_only && is_global) {
        continue;
      }
      const size_t layer_cache_size =
          TransformerBuilder::GetCacheSize(text_config_, is_global);
      const size_t layer_copy_size =
          std::min<size_t>(context_data.tokens.size(), layer_cache_size);
      ABSL_RETURN_IF_ERROR(ExecuteCopyCache(
          encoder, layer_copy_size,
          dynamic_cast<MetalSpatialTensor*>(context_data.key_tensors[i].get()),
          dynamic_cast<MetalSpatialTensor*>(
              context_data.value_tensors[i].get()),
          layer_cache_size, cache_.key_cached_tensors[i].get(),
          cache_.value_cached_tensors[i].get(), is_global));
    }
    [encoder endEncoding];
    [command_buffer commit];
  }
  return absl::OkStatus();
}

absl::StatusOr<ContextData> LlmMetal::CloneContext(
    MetalDevice& device, id<MTLCommandQueue> command_queue,
    const ContextData& context_data, bool local_only) {
  if (context_data.key_tensors.empty()) {
    return absl::InternalError("Context data key tensors are empty.");
  }
  // Init new kv tensors.
  const int num_cache_layers = context_data.key_tensors.size();
  const int local_id = IsLayerGlobal(0) ? 1 : 0;
  std::vector<std::unique_ptr<MetalSpatialTensor>> key_tensors;
  std::vector<std::unique_ptr<MetalSpatialTensor>> value_tensors;
  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    const int global_id = text_config_.global_layer_offset.value_or(
        text_config_.num_local_layers_per_global);
    std::optional<TensorDescriptor> k_global_desc;
    std::optional<TensorDescriptor> v_global_desc;
    if (!local_only && global_id < context_data.key_tensors.size() &&
        context_data.key_tensors[global_id] != nullptr) {
      k_global_desc = context_data.key_tensors[global_id]->GetDescriptor();
      v_global_desc = context_data.value_tensors[global_id]->GetDescriptor();
    }
    ABSL_ASSIGN_OR_RETURN(
        key_tensors,
        InitKvTensorsLocalGlobal(
            device, context_data.key_tensors[local_id]->GetDescriptor(),
            k_global_desc, text_config_.num_local_layers_per_global,
            num_cache_layers, text_config_.global_layer_offset));
    ABSL_ASSIGN_OR_RETURN(
        value_tensors,
        InitKvTensorsLocalGlobal(
            device, context_data.value_tensors[local_id]->GetDescriptor(),
            v_global_desc, text_config_.num_local_layers_per_global,
            num_cache_layers, text_config_.global_layer_offset));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        key_tensors,
        InitKvTensors(device,
                      context_data.key_tensors[local_id]->GetDescriptor(),
                      num_cache_layers));

    ABSL_ASSIGN_OR_RETURN(
        value_tensors,
        InitKvTensors(device,
                      context_data.value_tensors[local_id]->GetDescriptor(),
                      num_cache_layers));
  }
  @autoreleasepool {
    // Copy kv tensors to new kv tensors.
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    for (int i = 0; i < num_cache_layers; ++i) {
      const bool is_global = IsLayerGlobal(i);
      if ((local_only && is_global) || context_data.key_tensors[i] == nullptr) {
        continue;
      }
      const size_t layer_cache_size =
          TransformerBuilder::GetCacheSize(text_config_, is_global);
      const size_t layer_copy_size =
          std::min<size_t>(context_data.tokens.size(), layer_cache_size);
      ABSL_RETURN_IF_ERROR(ExecuteCopyCache(
          encoder, layer_copy_size,
          dynamic_cast<MetalSpatialTensor*>(context_data.key_tensors[i].get()),
          dynamic_cast<MetalSpatialTensor*>(
              context_data.value_tensors[i].get()),
          layer_copy_size, key_tensors[i].get(), value_tensors[i].get(),
          is_global));
    }
    [encoder endEncoding];
    [command_buffer commit];
  }
  // Store everything in the cloned data.
  ContextData cloned_data;
  std::move(key_tensors.begin(), key_tensors.end(),
            std::back_inserter(cloned_data.key_tensors));
  std::move(value_tensors.begin(), value_tensors.end(),
            std::back_inserter(cloned_data.value_tensors));
  cloned_data.lora_id = context_data.lora_id;
  cloned_data.tokens = context_data.tokens;
  return cloned_data;
}

absl::StatusOr<AudioContextData> LlmMetal::SaveAudioContext(
    MetalDevice& device, id<MTLCommandQueue> command_queue) {
  bool is_model_v3 = IsModelv3(audio_params_.config.transformer);
  ABSL_ASSIGN_OR_RETURN(
      AudioContextData context_data,
      AllocateAudioContextData(device, audio_params_.cache.key_cached_tensors,
                               audio_params_.cache.value_cached_tensors,
                               audio_params_.stream_depthwise_tensors,
                               audio_params_.stream_cumsum, is_model_v3));

  ABSL_RETURN_IF_ERROR(CopyAudioContextDataTensors(
      command_queue, audio_params_.cache.key_cached_tensors,
      audio_params_.cache.value_cached_tensors,
      audio_params_.stream_depthwise_tensors, audio_params_.stream_cumsum,
      context_data.key_tensors, context_data.value_tensors,
      context_data.stream_depthwise_tensors, context_data.stream_cumsum,
      is_model_v3));

  return context_data;
}

absl::Status LlmMetal::RestoreAudioContext(
    id<MTLCommandQueue> command_queue, const AudioContextData& context_data) {
  bool is_model_v3 = IsModelv3(audio_params_.config.transformer);
  ABSL_RETURN_IF_ERROR(CopyAudioContextDataTensors(
      command_queue, context_data.key_tensors, context_data.value_tensors,
      context_data.stream_depthwise_tensors, context_data.stream_cumsum,
      audio_params_.cache.key_cached_tensors,
      audio_params_.cache.value_cached_tensors,
      audio_params_.stream_depthwise_tensors, audio_params_.stream_cumsum,
      is_model_v3));
  return absl::OkStatus();
}

absl::StatusOr<AudioContextData> LlmMetal::CloneAudioContext(
    MetalDevice& device, id<MTLCommandQueue> command_queue,
    const AudioContextData& context_data) {
  bool is_model_v3 = IsModelv3(audio_params_.config.transformer);
  ABSL_ASSIGN_OR_RETURN(
      AudioContextData cloned_data,
      AllocateAudioContextData(device, context_data.key_tensors,
                               context_data.value_tensors,
                               context_data.stream_depthwise_tensors,
                               context_data.stream_cumsum, is_model_v3));

  ABSL_RETURN_IF_ERROR(CopyAudioContextDataTensors(
      command_queue, context_data.key_tensors, context_data.value_tensors,
      context_data.stream_depthwise_tensors, context_data.stream_cumsum,
      cloned_data.key_tensors, cloned_data.value_tensors,
      cloned_data.stream_depthwise_tensors, cloned_data.stream_cumsum,
      is_model_v3));

  return cloned_data;
}

absl::Status LlmMetal::LoadLoRA(
    MetalDevice* device, uint32_t lora_id, int lora_rank,
    std::unique_ptr<LlmTensorLoader> tensor_loader) {
  auto tensors = model_one_->lora_tensors.find(lora_rank);
  if (tensors == model_one_->lora_tensors.end() || tensors->second.empty()) {
    return absl::InternalError(
        absl::StrCat("Model does not support LoRA with rank: ", lora_rank));
  }

  auto lora_loader = std::make_unique<LoRALoader>();
  tensor_loader->SetCreateTensorFn(absl::bind_front(
      &LlmMetal::CreateModelTensor, this, device, &lora_loader->tensors));
  tensor_loader->SetGpuInfo(device->GetInfo());
  tensor_loader->SetDataType(tensors->second.begin()->second.weights_desc.type);
  lora_loader->lora_rank = lora_rank;
  lora_loader->loader = std::move(tensor_loader);
  lora_loaders_[lora_id] = std::move(lora_loader);
  return absl::OkStatus();
}

absl::Status LlmMetal::UnloadLoRA(id<MTLCommandQueue> command_queue,
                                  uint32_t lora_id) {
  lora_loaders_.erase(lora_id);
  if (current_lora_id_ == lora_id) {
    return UseLoRA(command_queue, std::nullopt);
  }
  return absl::OkStatus();
}

absl::Status LlmMetal::UseLoRA(id<MTLCommandQueue> command_queue,
                               std::optional<uint32_t> lora_id) {
  if (lora_id == current_lora_id_) {
    return absl::OkStatus();
  }

  current_lora_id_ = lora_id;
  if (!lora_id.has_value()) {
    lora_tags_.clear();
    SetupOptionalNodes();
    return absl::OkStatus();
  }

  auto loader_it = lora_loaders_.find(*lora_id);
  if (loader_it == lora_loaders_.end()) {
    return absl::InternalError("LoRA loader not found.");
  }
  int lora_rank = loader_it->second->lora_rank;
  lora_tags_.clear();
  SetupOptionalNodes();

  auto& loader = loader_it->second->loader;
  for (auto& [seq_size, model] : models_) {
    for (const auto& [name, info] : model.lora_tensors[lora_rank]) {
      if (!loader->HasTensor(name)) {
        continue;
      }
      if (odml::infra::gpu::IsAttentionTensorName(name)) {
        lora_tags_.insert(lora_rank | kLoRAAttentionOptionTag);
      } else {
        lora_tags_.insert(lora_rank | kLoRAFeedForwardOptionTag);
      }
      ABSL_ASSIGN_OR_RETURN(auto* new_tensor,
                            loader->LoadWeights(name, info.weights_desc,
                                                info.shape, info.swap_dims));
      ABSL_RETURN_IF_ERROR(model.context->SetTensor(
          info.id, static_cast<MetalSpatialTensor*>(new_tensor)));
    }
  }
  SetupOptionalNodes();
  return absl::OkStatus();
}

absl::Status LlmMetal::UseLayers(id<MTLCommandQueue> command_queue,
                                 const absl::flat_hash_set<int>& layers,
                                 int final_ln) {
  absl::flat_hash_set<int> tags;
  tags.insert(final_ln | kFinalLnOptionTag);
  for (int layer : layers) {
    tags.insert(layer | kTransformerLayerOptionTag);
  }
  if (tags == layer_tags_) {
    return absl::OkStatus();
  }
  layer_tags_ = std::move(tags);

  SetupOptionalNodes();
  return absl::OkStatus();
}

void LlmMetal::SetupOptionalNodes() {
  absl::flat_hash_set<int> tags = layer_tags_;
  tags.insert(lora_tags_.begin(), lora_tags_.end());
  if (shared_kv_cache_tag_) {
    tags.insert(*shared_kv_cache_tag_);
  }

  for (auto& kv : models_) {
    kv.second.context->EnableOptionalNodes(tags);
  }
}

absl::Status LlmMetal::ExecuteCopyCache(id<MTLComputeCommandEncoder> encoder,
                                        int src_size, MetalSpatialTensor* src_k,
                                        MetalSpatialTensor* src_v, int dst_size,
                                        MetalSpatialTensor* dst_k,
                                        MetalSpatialTensor* dst_v,
                                        bool is_global) {
  if (src_k == nullptr || src_v == nullptr) {
    return absl::InternalError(
        "Src k or v tensor is null. Most likely caused by dynamic casting.");
  }
  auto& copy_cache = (is_global ? copy_cache_global_ : copy_cache_);
  ABSL_RETURN_IF_ERROR(copy_cache->SetInt("src_size", src_size));
  ABSL_RETURN_IF_ERROR(copy_cache->SetInt("dst_size", dst_size));
  ABSL_RETURN_IF_ERROR(copy_cache->SetSrcTensor(src_k, 0));
  ABSL_RETURN_IF_ERROR(copy_cache->SetSrcTensor(src_v, 1));
  ABSL_RETURN_IF_ERROR(copy_cache->SetDstTensor(dst_k, 0));
  ABSL_RETURN_IF_ERROR(copy_cache->SetDstTensor(dst_v, 1));
  ABSL_RETURN_IF_ERROR(copy_cache->UpdateParams());
  copy_cache->Encode(encoder);
  return absl::OkStatus();
}

bool LlmMetal::IsLayerGlobal(int layer_number) const {
  int global_layer_offset = text_config_.num_local_layers_per_global;
  if (text_config_.global_layer_offset.has_value()) {
    global_layer_offset = *text_config_.global_layer_offset;
  }
  return (text_config_.global_head_dimension.has_value() ||
          TransformerBuilder::UseRingBuffer(text_config_)) &&
         ((layer_number % (text_config_.num_local_layers_per_global + 1)) ==
          global_layer_offset);
}

absl::Status LlmMetal::ExecuteUpdateIntParams(
    id<MTLComputeCommandEncoder> encoder, TransformerConfig& config,
    TransformerParams& params, const LlmRuntimeParams& param_vals) {
  const auto params_i32 = GetLlmRuntimeParamsBuffer(param_vals, config);
  for (int i = 0; i < params_i32.size(); ++i) {
    const std::string param_name = "param" + std::to_string(i);
    ABSL_RETURN_IF_ERROR(
        params.write_i32_params->SetInt(param_name, params_i32[i]));
  }
  ABSL_RETURN_IF_ERROR(
      params.write_i32_params->SetDstTensor(params.params_i32.get(), 0));
  ABSL_RETURN_IF_ERROR(params.write_i32_params->UpdateParams());
  params.write_i32_params->Encode(encoder);
  return absl::OkStatus();
}

absl::Status LlmMetal::ExecuteUpdateParams(id<MTLComputeCommandEncoder> encoder,
                                           MetalSpatialTensor* tensor,
                                           const std::vector<float>& params) {
  ABSL_RETURN_IF_ERROR(write_f32_params_->SetDstTensor(tensor, 0));
  for (int i = 0; i < params.size(); ++i) {
    const std::string param_name = "param" + std::to_string(i);
    ABSL_RETURN_IF_ERROR(write_f32_params_->SetFloat(param_name, params[i]));
  }
  ABSL_RETURN_IF_ERROR(write_f32_params_->UpdateParams());
  write_f32_params_->Encode(encoder);
  return absl::OkStatus();
}

absl::Status LlmMetal::ExecutePostProcessing(
    id<MTLComputeCommandEncoder> encoder) {
  {
    std::uniform_real_distribution<float> distribution =
        std::uniform_real_distribution<float>(0.0f, sampler_params_.p());
    std::vector<float> params(params_f32_->Channels(), 0.0f);

    for (int i = 0; i < params.size() - 1; ++i) {
      params[i] = distribution(*rand_gen_);
    }

    params[params.size() - 1] = 1.0f / sampler_params_.temperature();
    ABSL_RETURN_IF_ERROR(
        ExecuteUpdateParams(encoder, params_f32_.get(), params));
  }
  post_processing_->EncodeWithEncoder(encoder);
  return absl::OkStatus();
}

absl::StatusOr<GpuSpatialTensor*> LlmMetal::CreateModelTensor(
    MetalDevice* device,
    std::vector<std::unique_ptr<MetalSpatialTensor>>* model_tensors,
    const TensorDescriptor& tensor_desc) {
  auto& tensor = model_tensors->emplace_back(new MetalSpatialTensor());
  ABSL_RETURN_IF_ERROR(
      tensor->CreateFromDescriptor(tensor_desc, device->device()));
  return tensor.get();
}

absl::StatusOr<GpuSpatialTensor*> LlmMetal::CreateModelTensorMmaped(
    MetalDevice* device, LlmTensorLoader* tensor_loader,
    std::vector<std::unique_ptr<MetalSpatialTensor>>* model_tensors,
    const TensorDescriptor& tensor_desc) {
  auto& tensor = model_tensors->emplace_back(new MetalSpatialTensor());
  absl::Span<const uint8_t> data = tensor_desc.GetData();
  auto owned_weights = tensor_loader->GetOwnedData(data.data());
  if (owned_weights &&
      tensor_desc.GetStorageType() == TensorStorageType::kBuffer) {
    @autoreleasepool {
      absl::Span<unsigned char> owned_data = owned_weights->GetRawData();
      const size_t offset =
          tensor_desc.GetData().data() - owned_weights->GetRawData().data();
      id<MTLBuffer> buffer_base = [device->device()
          newBufferWithBytesNoCopy:owned_data.data()
                            length:owned_data.size()
                           options:MTLResourceStorageModeShared
                       deallocator:^(void* pointer, NSUInteger length){
                       }];
      mmaped_buffers_.push_back(
          std::make_unique<Buffer>(buffer_base, owned_data.size()));
      ABSL_RETURN_IF_ERROR(CreateTensorSharedBuffer(buffer_base, tensor_desc,
                                                    tensor.get(), offset));
    }
  } else {
    ABSL_RETURN_IF_ERROR(
        tensor->CreateFromDescriptor(tensor_desc, device->device()));
  }
  return tensor.get();
}

absl::Status LlmMetal::UpdateEmbeddings(id<MTLCommandQueue> command_queue,
                                        StackedTransformerInference& model,
                                        int previous_token,
                                        absl::Span<const int> tokens,
                                        bool require_load_with_queue) {
  odml::infra::gpu::UpdateNgrammerEmbeddings(
      ngrammer_embeddings_, model.ngrammer_weights, previous_token, tokens,
      text_config_.vocabulary_size,
      text_config_.ngram_vocabulary_size.value_or(0));
  odml::infra::gpu::UpdateEmbeddings(ple_embeddings_, model.ple_weights, tokens,
                                     text_config_.vocabulary_size,
                                     tokens_transform_fn_);

  ABSL_RETURN_IF_ERROR(UploadEmbeddingsToGPU(model.ngrammer_weights,
                                             model.context.get(), command_queue,
                                             require_load_with_queue));
  ABSL_RETURN_IF_ERROR(UploadEmbeddingsToGPU(model.ple_weights,
                                             model.context.get(), command_queue,
                                             require_load_with_queue));
  return absl::OkStatus();
}

// src has sh: [1, 1, x, emb_dim]
// dst has sh: [1, 1, media_cache_size_, emb_dim]
// we perform: dst[0, 0, index:index + x, :] = src
// expect dst to be either media_embeddings_ or vision_ple_tensors_
absl::Status LlmMetal::UpdateMultimodalEmbeddings(
    id<MTLCommandQueue> command_queue, MetalSpatialTensor* src,
    MetalSpatialTensor* dst, int index) {
  const BHWC src_sh = src->GetDescriptor().GetBHWCShape();
  if (src_sh.b != 1 || src_sh.h != 1) {
    return absl::InvalidArgumentError("Invalid src tensor shape.");
  }
  if (index + src_sh.w > media_cache_size_) {
    return absl::InvalidArgumentError("Media embeddings will overflow cache.");
  }
  const BHWC dst_sh = dst->GetDescriptor().GetBHWCShape();
  if (dst_sh.b != 1 || dst_sh.h != 1 || dst_sh.w != media_cache_size_) {
    return absl::InvalidArgumentError("Invalid dst shape.");
  }
  if (src_sh.c != dst_sh.c) {
    return absl::InvalidArgumentError("src and dst channel count mismatch.");
  }
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    ABSL_RETURN_IF_ERROR(multimodal_embeddings_op_->SetSrcTensor(src, 0));
    ABSL_RETURN_IF_ERROR(multimodal_embeddings_op_->SetDstTensor(dst, 0));
    ABSL_RETURN_IF_ERROR(multimodal_embeddings_op_->SetInt("index", index));
    ABSL_RETURN_IF_ERROR(multimodal_embeddings_op_->UpdateParams());
    multimodal_embeddings_op_->Encode(encoder);
    [encoder endEncoding];
    [command_buffer commit];
  }
  return absl::OkStatus();
}

absl::StatusOr<GpuSpatialTensor*> LlmMetal::ExecuteAudioEncoder(
    const TensorFloat32& audio_tensor, const TensorFloat32& mask_tensor,
    int time_step, id<MTLCommandQueue> command_queue) {
  // Check input shapes
  ABSL_CHECK_EQ(audio_tensor.shape,
                audio_params_.feature->GetDescriptor().GetBHWCShape());
  ABSL_CHECK_EQ(mask_tensor.shape,
                audio_params_.mask->GetDescriptor().GetBHWCShape());
  // Set llm_params
  LlmRuntimeParams llm_params;
  const int audio_to_reduce =
      (audio_tensor.shape.h - audio_params_.config.feature.pad_for_streaming);
  ABSL_CHECK_EQ(audio_to_reduce % audio_params_.config.feature.shrink_factor,
                0);
  llm_params.token_index_offset =
      time_step * audio_params_.config.feature.take_every_n;
  llm_params.active_tokens =
      llm_params.token_index_offset +
      (audio_to_reduce / audio_params_.config.feature.shrink_factor);
  ABSL_CHECK_LE(*llm_params.active_tokens,
                audio_params_.config.transformer.cache_size);
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    ABSL_RETURN_IF_ERROR(
        ExecuteUpdateIntParams(encoder, audio_params_.config.transformer,
                               audio_params_.transformer_params, llm_params));

    // Set audio encoder inputs.
    TensorDescriptor feature_desc = audio_params_.feature->GetDescriptor();
    feature_desc.UploadData(audio_tensor);
    ABSL_RETURN_IF_ERROR(audio_params_.feature->UploadDescriptorData(
        feature_desc, command_queue, /*wait_for_completion=*/false));
    TensorDescriptor mask_desc = audio_params_.mask->GetDescriptor();
    mask_desc.UploadData(mask_tensor);
    ABSL_RETURN_IF_ERROR(audio_params_.mask->UploadDescriptorData(
        mask_desc, command_queue, /*wait_for_completion=*/false));
    audio_params_.context->EncodeWithEncoder(encoder);
    [encoder endEncoding];
    [command_buffer commit];
  }

  return audio_params_.context->GetTensor(audio_params_.output_handle.id);
}

// vision_ple(src) should be a vector of size (text_config_.stack_size)
// and each entry should have shape (1, 1, x, ple_emb_dim)
// vision_ple_tensors_ (dst) should have size
// (text_config_.stack_size - text_config_.extra_stack_size) each entry should
// have shape (1, 1, media_cache_size, ple_emb_dim) This op performs the
// following: for each i in range(text_config_.stack_size -
// text_config_.extra_stack_size):
//   dst[i][0, 0, index:index + x, :] = src[i]
absl::Status LlmMetal::FillVisionPleEmbeddings(
    MetalDevice& device, id<MTLCommandQueue> command_queue,
    const std::vector<TensorFloat32>& vision_ple, int index) {
  if (vision_ple.size() < text_config_.stack_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("passed vision_ple tensors must have at least size %d.",
                        text_config_.stack_size));
  }
  if (vision_ple_tensors_.size() !=
      text_config_.stack_size - text_config_.extra_stack_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "stored vision_ple tensors must have size %d.",
        text_config_.stack_size - text_config_.extra_stack_size));
  }
  for (int i = 0; i < text_config_.stack_size - text_config_.extra_stack_size;
       ++i) {
    TensorDescriptor desc = vision_ple_tensors_[i]->GetDescriptor();
    desc.SetBHWCShape(vision_ple[i].shape);
    desc.UploadData(vision_ple[i]);
    auto src = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(device.device(), desc, src.get()));
    ABSL_RETURN_IF_ERROR(
        src->UploadDescriptorData(desc, command_queue,
                                  /*wait_for_completion=*/false));
    ABSL_RETURN_IF_ERROR(UpdateMultimodalEmbeddings(
        command_queue, src.get(), vision_ple_tensors_[i].get(), index));
  }
  return absl::OkStatus();
}

absl::Status LlmMetal::FillMediaEmbeddings(MetalDevice& device,
                                           id<MTLCommandQueue> command_queue,
                                           const TensorFloat32& media_tensor,
                                           int index) {
  TensorDescriptor desc = media_embeddings_->GetDescriptor();
  desc.SetBHWCShape(media_tensor.shape);
  desc.UploadData(media_tensor);
  auto media_gpu = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(device.device(), desc, media_gpu.get()));
  ABSL_RETURN_IF_ERROR(media_gpu->UploadDescriptorData(
      desc, command_queue, /*wait_for_completion=*/false));
  return UpdateMultimodalEmbeddings(command_queue, media_gpu.get(),
                                    media_embeddings_.get(), index);
}

absl::StatusOr<ml_drift::speculative_utils::TokenAtTimeStep>
LlmMetal::PeekNextToken(id<MTLCommandQueue> command_queue) {
  if (output_batch_size_ > 1) {
    return absl::UnimplementedError(
        "Multiple output heads not supported for PeekNextToken.");
  }

  // Sync to ensure token ready, if needed
  if (need_sync_for_peek_token_) {
    @autoreleasepool {
      id<MTLCommandBuffer> cb_for_sync = [command_queue commandBuffer];
      [cb_for_sync commit];
      [cb_for_sync waitUntilCompleted];
    }
    need_sync_for_peek_token_ = false;
  }

  // Download tokens from GPU to CPU.
  const MetalSpatialTensor* gpu_tensor = tokens_ids_.get();
  Tensor<BHWC, DataType::kInt32> output;
  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, device_));
  desc.DownloadData(&output);

  // Grab the next one and return it.
  const int time_step = input_tokens_.size() + (ready_tokens_ - 1);
  int token_id = output.data[time_step * output_batch_size_];
  return speculative_utils::TokenAtTimeStep(time_step, token_id);
}

// TODO(b/523389424): Clean up UpdateConstraintMask as it is almost identical
// across all 3 of our backends.
absl::Status LlmMetal::UpdateConstraintMask(
    const odml::infra::llm_utils::Bitmap& constraint_bitmap) {
  std::vector<bool> new_constraint_mask;
  new_constraint_mask.resize(text_config_.vocabulary_size);
  for (int i = 0; i < text_config_.vocabulary_size; ++i) {
    new_constraint_mask[i] = constraint_bitmap.Get(i);
  }
  for (int token : suppressed_tokens_) {
    if (token >= 0 && token < text_config_.vocabulary_size) {
      new_constraint_mask[token] = false;
    }
  }
  // Check if the constraint has changed. If not, no need to upload again.
  if (new_constraint_mask == last_constraint_mask_) {
    return absl::OkStatus();
  }

  TensorDescriptor desc = constraint_mask_->GetDescriptor();
  std::vector<float> float_mask;
  float_mask.resize(desc.GetBHWCShape().DimensionsProduct());
  RET_CHECK_EQ(float_mask.size(),
               output_batch_size_ * text_config_.vocabulary_size);
  for (int b = 0; b < output_batch_size_; ++b) {
    for (int i = 0; i < text_config_.vocabulary_size; ++i) {
      float_mask[b * text_config_.vocabulary_size + i] =
          static_cast<float>(!new_constraint_mask[i]);
    }
  }
  constraint_mask_changed_since_last_input_ = true;
  last_constraint_mask_ = std::move(new_constraint_mask);
  desc.UploadData(float_mask.data());
  return constraint_mask_->UploadDescriptorData(desc, device_);
}

}  // namespace metal
}  // namespace ml_drift
