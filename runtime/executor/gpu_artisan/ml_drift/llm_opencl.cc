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

#include "runtime/executor/gpu_artisan/ml_drift/llm_opencl.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/bind_front.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_errors.h"  // from @ml_drift
#include "ml_drift/cl/cl_event.h"  // from @ml_drift
#include "ml_drift/cl/cl_operation.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/cl/util.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
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
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/byom/byom_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/group_size.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/name_check.h"
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_file_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/util/status_macros.h"
#include "support/util/memory_mapped_file.h"

namespace ml_drift {
namespace cl {
namespace {

using ::odml::infra::gpu::GetRecommendedGroupSizes;
using ::odml::infra::gpu::GroupSize;
using ::odml::infra::gpu::LlmFileTensorLoader;
using ::odml::infra::gpu::LlmTensorLoader;
using ::odml::infra::proto::SamplerParameters;

absl::Status InitClOp(Environment* env, std::unique_ptr<GPUOperation>&& gpu_op,
                      ClOperation* cl_op) {
  ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(env->device().GetInfo()));

  cl_op->Init(std::move(gpu_op));
  ABSL_RETURN_IF_ERROR(cl_op->Compile(env->GetDevicePtr(), &env->context(),
                                      env->program_cache()));
  return absl::OkStatus();
}

absl::Status InitClOp(Environment* env, GPUOperation&& operation,
                      ClOperation* cl_op) {
  return InitClOp(env, std::make_unique<GPUOperation>(std::move(operation)),
                  cl_op);
}

absl::Status UploadEmbeddingsToGPU(
    const std::vector<odml::infra::gpu::EmbeddingWeights>& weights,
    InferenceContext* context, CLCommandQueue* queue, bool async) {
  for (const auto& weight : weights) {
    auto* tensor = context->GetTensor(weight.weights.id);
    ABSL_RETURN_IF_ERROR(
        tensor->WriteData(weight.weights_data.data(), queue, async));

    auto* scale_tensor = context->GetTensor(weight.scales.id);
    ABSL_RETURN_IF_ERROR(
        scale_tensor->WriteData(weight.scales_data.data(), queue, async));
  }
  return absl::OkStatus();
}

absl::Status CopyTensor(const Tensor& src, const Tensor& dst,
                        CLCommandQueue* queue) {
  auto src_storage = src.GetStorageType();
  if (src_storage == TensorStorageType::kBuffer ||
      src_storage == TensorStorageType::kImageBuffer) {
    return GetOpenCLError(
        clEnqueueCopyBuffer(queue->queue(), src.GetMemoryPtrForWriting(),
                            dst.GetMemoryPtrForWriting(), 0, 0,
                            src.GetMemorySizeInBytes(), 0, nullptr, nullptr));
  } else if (src_storage == TensorStorageType::kTexture2D ||
             src_storage == TensorStorageType::kSingleTexture2D) {
    size_t origin[3] = {0, 0, 0};
    auto region = src.GetDescriptor().GetFullTensorRegion();
    size_t region_size[3] = {static_cast<size_t>(region.x),
                             static_cast<size_t>(region.y),
                             static_cast<size_t>(region.z)};
    return GetOpenCLError(clEnqueueCopyImage(queue->queue(), src.GetMemoryPtr(),
                                             dst.GetMemoryPtr(), origin, origin,
                                             region_size, 0, nullptr, nullptr));
  } else {
    return absl::InvalidArgumentError("Unsupported storage type for copy.");
  }
}

absl::StatusOr<std::vector<std::unique_ptr<Tensor>>> InitKvTensorsLocalGlobal(
    const Environment& env, const TensorDescriptor& local_desc,
    std::optional<TensorDescriptor> global_desc,
    int num_local_layers_per_global, int num_cache_layers,
    std::optional<int> global_layer_offset) {
  std::vector<std::unique_ptr<Tensor>> tensors(num_cache_layers);

  int pattern_offset = global_layer_offset.has_value()
                           ? *global_layer_offset
                           : num_local_layers_per_global;

  for (int i = 0; i < num_cache_layers; ++i) {
    const bool is_global =
        (i % (num_local_layers_per_global + 1) == pattern_offset);
    if (is_global && !global_desc.has_value()) {
      continue;
    }
    tensors[i] = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(env.context(),
                                      is_global ? *global_desc : local_desc,
                                      tensors[i].get()));
  }
  return tensors;
}

bool UseGpuMmapedTensors(const GpuInfo& gpu_info) {
  return gpu_info.IsIntel() || gpu_info.IsPowerVR();
}

}  // namespace

absl::Status LlmOpenCl::Init(const LlmConfig& config, int outputs_count,
                             const std::string& weights_path,
                             const CreateGpuModelInfo& create_info,
                             Environment* env) {
  auto tensor_loader = LlmTensorLoader::MakeCaching(
      std::make_unique<LlmFileTensorLoader>(weights_path));
  return Init(config, outputs_count, std::move(tensor_loader), create_info,
              env);
}

absl::Status LlmOpenCl::Init(const LlmConfig& config, int outputs_count,
                             std::unique_ptr<LlmTensorLoader> tensor_loader,
                             const CreateGpuModelInfo& create_info,
                             Environment* env) {
  const auto& gpu_info = env->device().GetInfo();
  create_info_ = create_info;
  ABSL_CHECK(config.text_decoder.has_value()) << "Text decoder is not set.";
  text_config_ = config.text_decoder.value();  // use below a copy
  enable_decode_logits_ = text_config_.enable_decode_logits;
  enable_external_embeddings_ = text_config_.enable_external_embeddings;
  suppressed_tokens_ = config.suppressed_tokens;
  epsilon_ = GetEpsilon(create_info_.precision, gpu_info);
  max_top_k_ = text_config_.max_top_k;
  output_batch_size_ = outputs_count;
  max_tokens_ = text_config_.cache_size;
  output_tokens_.resize(output_batch_size_ * max_tokens_);

  // default sampler params to be greedy.
  sampler_params_ = SamplerParameters();
  sampler_params_.set_type(SamplerParameters::GREEDY);
  sampler_params_.set_k(1);

  if (UseGpuMmapedTensors(gpu_info) &&
      create_info.hints.Check(ModelHints::kEnableHostMappedPointer)) {
    tensor_loader->SetMemoryMappedFileAccess(
        ::litert::support::MemoryMappedFile::Access::kRead);
    tensor_loader->SetCreateTensorFn(absl::bind_front(
        &LlmOpenCl::CreateModelTensorMmaped, this, &env->context(), gpu_info,
        tensor_loader.get(), &model_tensors_));
  } else {
    tensor_loader->SetCreateTensorFn(absl::bind_front(
        &LlmOpenCl::CreateModelTensor, this, &env->context(), &model_tensors_));
  }
  if (text_config_.sequence_size == 0) {
    group_sizes_ = GetRecommendedGroupSizes(gpu_info);
  } else if (text_config_.sequence_size == 1) {
    group_sizes_ = {GroupSize{1, 0.0f}};
  } else {
    group_sizes_ = {
        GroupSize{text_config_.sequence_size, 0.0f},
        GroupSize{1, 1.0f},
    };
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<TransformerBuilder> text_decoder_builder,
      odml_byom::CreateTransformerBuilder(text_config_, config.model_type,
                                          output_batch_size_, gpu_info,
                                          create_info_, tensor_loader.get()));

  if (text_config_.use_autosized_ringbuffers &&
      text_config_.sliding_window_size.has_value()) {
    // Set ring buffer size to max(group_sizes) + sliding_window_size
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
  ABSL_RETURN_IF_ERROR(InitSharedTensors(text_decoder_builder.get(),
                                         text_cache_, text_transformer_params_,
                                         text_config_.stack_size, env));
  tokens_ids_ = std::make_unique<Tensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      env->context(), text_decoder_builder->GetTokensTensorDescriptor(),
      tokens_ids_.get()));

  // Init multimodal tensor
  if (media_cache_size_ > 0) {  // is multimodal
    if (text_config_.per_layer_embedding_dimension.has_value()) {
      if (text_decoder_builder->UseGoogleInternalModelV2()) {
        vision_ple_tensors_.reserve(text_config_.stack_size -
                                    text_config_.extra_stack_size);
        for (int i = 0;
             i < text_config_.stack_size - text_config_.extra_stack_size; ++i) {
          vision_ple_tensors_.push_back(std::make_unique<Tensor>());
          ABSL_RETURN_IF_ERROR(
              CreateTensor(env->context(),
                           text_decoder_builder->GetVisionPleTensorDescriptor(),
                           vision_ple_tensors_[i].get()));
        }
      }
    }
    media_embeddings_ = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env->context(),
                     text_decoder_builder->GetMediaEmbeddingsTensorDescriptor(),
                     media_embeddings_.get()));
  }

  {
    int buffer_size =
        text_config_.max_indices_count_in_clean_cache_at * sizeof(int32_t);
    text_cache_.ids_to_remove_buffer = std::make_unique<Buffer>(Buffer());
    ABSL_RETURN_IF_ERROR(CreateReadWriteBuffer(
        buffer_size, &env->context(), text_cache_.ids_to_remove_buffer.get()));
  }

  ABSL_RETURN_IF_ERROR(InitCleanCacheModel(
      text_decoder_builder.get(), text_cache_, text_transformer_params_, env));
  for (int i = 0; i < group_sizes_.size(); ++i) {
    const int sequence_size = group_sizes_[i].sequence_size;
    const int batch_size = sequence_size == 1 ? output_batch_size_ : 1;
    ABSL_RETURN_IF_ERROR(
        InitModel(text_decoder_builder.get(), sequence_size, batch_size, env));
  }
  model_one_ = &models_[1];

  ABSL_ASSIGN_OR_RETURN(
      ple_embeddings_,
      GetDynamicPleEmbeddings(text_config_, tensor_loader.get()));
  ABSL_ASSIGN_OR_RETURN(
      ngrammer_embeddings_,
      GetDynamicNgrammerEmbeddings(text_config_, tensor_loader.get()));
  if (text_decoder_builder->UseGemmaModelV3NPlus()) {
    // Replace negative tokens with 0. This is to match the behavior of the
    // v3 model. Note this is for dynamically updated PLE only. For
    // non-dynamically updated PLE, the negative tokens are removed in the
    // transformer GPU model.
    tokens_transform_fn_ = [](int token) { return std::max(0, token); };
  }

  if (enable_decode_logits_) {
    ABSL_RETURN_IF_ERROR(
        InitFinalEmbeddingToLogitsModel(text_decoder_builder.get(), env));
  } else {
    ABSL_RETURN_IF_ERROR(
        InitPostProcessingModel(text_decoder_builder.get(), env));
  }

  {
    int buffer_size = text_config_.cache_size * sizeof(int32_t);
    input_token_ids_ = std::make_unique<Buffer>(Buffer());
    ABSL_RETURN_IF_ERROR(CreateReadWriteBuffer(buffer_size, &env->context(),
                                               input_token_ids_.get()));
  }

  if (enable_external_embeddings_) {
    int embedding_buffer_size =
        text_config_.cache_size * text_config_.model_dimension * sizeof(float);
    input_embeddings_ = std::make_unique<Buffer>(Buffer());
    ABSL_RETURN_IF_ERROR(CreateReadWriteBuffer(
        embedding_buffer_size, &env->context(), input_embeddings_.get()));

    int ple_embedding_buffer_size =
        text_config_.cache_size * text_config_.stack_size *
        (*text_config_.per_layer_embedding_dimension) * sizeof(float);
    input_ple_embeddings_ = std::make_unique<Buffer>(Buffer());
    ABSL_RETURN_IF_ERROR(CreateReadWriteBuffer(ple_embedding_buffer_size,
                                               &env->context(),
                                               input_ple_embeddings_.get()));
  }

  if (enable_decode_logits_) {
    // Needed only for ProcessTokens.
    int buffer_size = text_config_.cache_size * sizeof(int32_t);
    input_sequence_position_ = std::make_unique<Buffer>(Buffer());
    ABSL_RETURN_IF_ERROR(CreateReadWriteBuffer(buffer_size, &env->context(),
                                               input_sequence_position_.get()));
    int mask_buffer_size =
        text_config_.cache_size * text_config_.cache_size * sizeof(float);
    input_global_mask_ = std::make_unique<Buffer>(Buffer());
    ABSL_RETURN_IF_ERROR(CreateReadWriteBuffer(
        mask_buffer_size, &env->context(), input_global_mask_.get()));
    input_local_mask_ = std::make_unique<Buffer>(Buffer());
    ABSL_RETURN_IF_ERROR(CreateReadWriteBuffer(
        mask_buffer_size, &env->context(), input_local_mask_.get()));
  }

  CLCommandQueueOptions queue_options;
  ABSL_RETURN_IF_ERROR(CreateCLCommandQueue(env->device(), env->context(),
                                            &reading_queue_, queue_options));

  if (text_config_.kv_cache_shared_start_layer) {
    shared_kv_cache_tag_ = kSharedKvCacheLayersOptionTag;
    SetupOptionalNodes();
  }
  return InitHelperOps(env);
}

absl::Status LlmOpenCl::InitAudioEncoder(
    const AudioEncoderConfig& config,
    std::unique_ptr<LlmTensorLoader> tensor_loader,
    const CreateGpuModelInfo& create_info, int output_batch_size,
    Environment* env) {
  create_info_ = create_info;
  audio_params_.config = config;
  if (text_config_.model_dimension != 0) {
    if (GetOutputDim(config) != text_config_.model_dimension) {
      return absl::InvalidArgumentError(
          "Audio encoder output dimension must match text decoder model "
          "dimension.");
    }
  }
  const auto& gpu_info = env->GetDevicePtr()->GetInfo();
  if (UseGpuMmapedTensors(gpu_info) &&
      create_info.hints.Check(ModelHints::kEnableHostMappedPointer)) {
    tensor_loader->SetMemoryMappedFileAccess(
        ::litert::support::MemoryMappedFile::Access::kRead);
    tensor_loader->SetCreateTensorFn(absl::bind_front(
        &LlmOpenCl::CreateModelTensorMmaped, this, &env->context(), gpu_info,
        tensor_loader.get(), &model_tensors_));
  } else {
    tensor_loader->SetCreateTensorFn(absl::bind_front(
        &LlmOpenCl::CreateModelTensor, this, &env->context(), &model_tensors_));
  }
  auto audio_encoder_builder = std::make_unique<AudioEncoderBuilder>(
      config, output_batch_size, gpu_info, create_info_, tensor_loader.get());

  // Init tensors
  int num_cache_layers = audio_params_.config.transformer.stack_size;
  if (audio_encoder_builder->UseGemmaModelV3NPlus() &&
      audio_params_.config.transformer.kv_cache_shared_start_layer
          .has_value()) {
    num_cache_layers =
        *audio_params_.config.transformer.kv_cache_shared_start_layer;
  }
  if (IsModelv3(audio_params_.config
                    .transformer)) {  // need different kv cache tensors
    audio_params_.transformer_params.params_i32 = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->context(), audio_encoder_builder->GetParamsTensorDescriptor(),
        audio_params_.transformer_params.params_i32.get()));
    // only need kv cache for streaming
    if (audio_encoder_builder->IsStreaming()) {
      audio_params_.cache.key_cached_tensors =
          std::vector<std::unique_ptr<Tensor>>(num_cache_layers);
      audio_params_.cache.value_cached_tensors =
          std::vector<std::unique_ptr<Tensor>>(num_cache_layers);
      for (int i = 0; i < num_cache_layers; ++i) {
        audio_params_.cache.key_cached_tensors[i] = std::make_unique<Tensor>();
        ABSL_RETURN_IF_ERROR(CreateTensor(
            env->context(),
            audio_encoder_builder->GetKCacheConformerTensorDescriptor(),
            audio_params_.cache.key_cached_tensors[i].get()));
        audio_params_.cache.value_cached_tensors[i] =
            std::make_unique<Tensor>();
        ABSL_RETURN_IF_ERROR(CreateTensor(
            env->context(),
            audio_encoder_builder->GetVCacheConformerTensorDescriptor(),
            audio_params_.cache.value_cached_tensors[i].get()));
      }
    }
  } else {
    ABSL_RETURN_IF_ERROR(InitSharedTensors(
        audio_encoder_builder.get(), audio_params_.cache,
        audio_params_.transformer_params, num_cache_layers, env));
  }
  audio_params_.feature = std::make_unique<Tensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      env->context(), audio_encoder_builder->GetFeatureDescriptor(),
      audio_params_.feature.get()));
  audio_params_.mask = std::make_unique<Tensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(env->context(),
                                    audio_encoder_builder->GetMaskDescriptor(),
                                    audio_params_.mask.get()));
  if (audio_encoder_builder->IsStreaming()) {
    audio_params_.stream_cumsum = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->context(), audio_encoder_builder->GetCumsumStreamDescriptor(),
        audio_params_.stream_cumsum.get()));
    if (IsModelv3(audio_params_.config.transformer)) {
      audio_params_.stream_depthwise_tensors =
          std::vector<std::unique_ptr<Tensor>>(num_cache_layers);
      for (int i = 0; i < num_cache_layers; ++i) {
        audio_params_.stream_depthwise_tensors[i] = std::make_unique<Tensor>();
        ABSL_RETURN_IF_ERROR(
            CreateTensor(env->context(),
                         audio_encoder_builder->GetDepthwiseStreamDescriptor(),
                         audio_params_.stream_depthwise_tensors[i].get()));
      }
    }
  }
  if (IsModelv3(audio_params_.config.transformer)) {
    audio_params_.signal = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->context(), audio_encoder_builder->GetSignalDescriptor(),
        audio_params_.signal.get()));
  }
  // Init models
  ABSL_RETURN_IF_ERROR(InitAudioEncoderModel(audio_encoder_builder.get(), env));
  // Since we have different cache tensors for v3, we need to update the
  // clean cache model.
  if (IsModelv3(audio_params_.config.transformer)) {
    // no kv cache for v3 non-streaming
    if (audio_encoder_builder->IsStreaming()) {
      std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
      std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
      GpuModelBuilder::TensorHandle stream_cumsum_handle;
      std::vector<GpuModelBuilder::TensorHandle>
          stream_depthwise_tensor_handles;
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
          create_info_copy, &gpu_model, env, nullptr));
    }
  } else {  // v1 or v2
    ABSL_RETURN_IF_ERROR(
        InitCleanCacheModel(audio_encoder_builder.get(), audio_params_.cache,
                            audio_params_.transformer_params, env));
  }
  // Init ops
  if (audio_params_.transformer_params.params_i32) {
    audio_params_.transformer_params.write_i32_params =
        std::make_unique<ClOperation>();
    ABSL_RETURN_IF_ERROR(InitClOp(
        env,
        CreateWriteParamsOp(
            audio_params_.transformer_params.params_i32->GetDescriptor(),
            audio_params_.transformer_params.params_i32->Channels()),
        audio_params_.transformer_params.write_i32_params.get()));
  }
  if (IsModelv3(audio_params_.config.transformer) &&
      audio_encoder_builder->IsStreaming()) {
    // Need to clean audio cache b/c the cache tensors are not
    // guaranteed to be zeroed out.
    ABSL_RETURN_IF_ERROR(CleanAudioCache(env, 0));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<Tensor>>> LlmOpenCl::InitKvTensors(
    const Environment& env, const TensorDescriptor& tensor_desc,
    int num_cache_layers) {
  return InitKvTensorsLocalGlobal(env, tensor_desc,
                                  /*global_desc=*/tensor_desc,
                                  /*num_local_layers_per_global=*/1,
                                  num_cache_layers,
                                  /*global_layer_offset=*/std::nullopt);
}

absl::Status LlmOpenCl::InitSharedTensors(
    TransformerBuilder* transformer_builder, TransformerCache& cache,
    TransformerParams& params, int num_cache_layers, Environment* env) {
  params.params_i32 = std::make_unique<Tensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      env->context(), transformer_builder->GetParamsTensorDescriptor(),
      params.params_i32.get()));
  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    ABSL_ASSIGN_OR_RETURN(
        cache.key_cached_tensors,
        InitKvTensorsLocalGlobal(
            *env, transformer_builder->GetKCacheTensorDescriptor(),
            transformer_builder->GetKCacheTensorDescriptor(/*is_global=*/true),
            text_config_.num_local_layers_per_global, num_cache_layers,
            text_config_.global_layer_offset));
    ABSL_ASSIGN_OR_RETURN(
        cache.value_cached_tensors,
        InitKvTensorsLocalGlobal(
            *env, transformer_builder->GetVCacheTensorDescriptor(),
            transformer_builder->GetVCacheTensorDescriptor(/*is_global=*/true),
            text_config_.num_local_layers_per_global, num_cache_layers,
            text_config_.global_layer_offset));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        cache.key_cached_tensors,
        InitKvTensors(*env, transformer_builder->GetKCacheTensorDescriptor(),
                      num_cache_layers));
    ABSL_ASSIGN_OR_RETURN(
        cache.value_cached_tensors,
        InitKvTensors(*env, transformer_builder->GetVCacheTensorDescriptor(),
                      num_cache_layers));
  }
  return absl::OkStatus();
}

absl::Status LlmOpenCl::InitModel(TransformerBuilder* transformer_builder,
                                  int sequence_size, int batch_size,
                                  Environment* env) {
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
            enable_decode_logits_ ? &model.positions : nullptr,
            enable_decode_logits_ ? &model.global_mask : nullptr,
            enable_decode_logits_ ? &model.local_mask : nullptr));
  } else if (transformer_builder->UseGemmaModelV3NPlus()) {
    if (enable_external_embeddings_) {
      // Only support text models with external embeddings.
      ABSL_ASSIGN_OR_RETURN(
          gpu_model,
          transformer_builder->CreateStackedTransformerModelV3(
              sequence_size, batch_size, nullptr, nullptr, nullptr,
              &params_i32_handle, &key_cached_tensor_handles,
              &value_cached_tensor_handles, &model.output, nullptr, nullptr,
              enable_decode_logits_ ? &model.positions : nullptr,
              enable_decode_logits_ ? &model.global_mask : nullptr,
              enable_decode_logits_ ? &model.local_mask : nullptr,
              &model.external_embeddings, &model.external_ple_embeddings,
              nullptr, nullptr, kGemmaBaseName));
    } else {
      ABSL_ASSIGN_OR_RETURN(
          gpu_model,
          transformer_builder->CreateStackedTransformerModelV3(
              sequence_size, batch_size, &model.input,
              (media_cache_size_ > 0) ? &media_handle : nullptr,
              &tokens_ids_handle, &params_i32_handle,
              &key_cached_tensor_handles, &value_cached_tensor_handles,
              &model.output, &model.ple_weights, nullptr,
              enable_decode_logits_ ? &model.positions : nullptr,
              enable_decode_logits_ ? &model.global_mask : nullptr,
              enable_decode_logits_ ? &model.local_mask : nullptr, nullptr,
              nullptr, nullptr, nullptr, kGemmaBaseName));
    }
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
        text_transformer_params_.params_i32.get();
    for (int i = 0; i < key_cached_tensor_handles.size(); ++i) {
      create_info_copy
          .external_immutable_tensors[key_cached_tensor_handles[i].id] =
          text_cache_.key_cached_tensors[i].get();
    }
    for (int i = 0; i < value_cached_tensor_handles.size(); ++i) {
      create_info_copy
          .external_immutable_tensors[value_cached_tensor_handles[i].id] =
          text_cache_.value_cached_tensors[i].get();
    }
    if (media_cache_size_ > 0) {  // has multimodal
      create_info_copy.external_immutable_tensors[media_handle.id] =
          media_embeddings_.get();
      if (transformer_builder->UseGoogleInternalModelV2()) {
        for (int i = 0;
             i < text_config_.stack_size - text_config_.extra_stack_size; ++i) {
          create_info_copy
              .external_immutable_tensors[vision_ple_tensor_handles[i].id] =
              vision_ple_tensors_[i].get();
        }
      }
    }
  }

  model.seq_size = sequence_size;
  model.context = std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(model.context->InitFromGpuModel(
      create_info_copy, &gpu_model, env, nullptr));
  if (enable_external_embeddings_) {
    ABSL_RETURN_IF_ERROR(InitEmbeddingLookupModel(
        transformer_builder, env, model, sequence_size, batch_size));
  }

  return absl::OkStatus();
}

absl::Status LlmOpenCl::InitFinalEmbeddingToLogitsModel(
    TransformerBuilder* transformer_builder, Environment* env) {
  ABSL_ASSIGN_OR_RETURN(GpuModel gpu_model,
                        transformer_builder->CreateFinalEmbeddingToLogitsModel(
                            &final_embedding_, &logits_, /*sequence_size=*/1));
  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  final_embedding_to_logits_ =
    std::make_unique<InferenceContext>(InferenceContext());
  return final_embedding_to_logits_->InitFromGpuModel(
    create_info_copy, &gpu_model, env, nullptr);
}

absl::Status LlmOpenCl::InitPostProcessingModel(
    TransformerBuilder* transformer_builder, Environment* env) {
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle stacked_transformer_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle params_f32_handle;
  GpuModelBuilder::TensorHandle constraint_mask_handle;
  ABSL_ASSIGN_OR_RETURN(
      GpuModel gpu_model,
      transformer_builder->CreatePostProcessingModel(
          &input_handle, &params_i32_handle, &params_f32_handle,
          &tokens_ids_handle, &stacked_transformer_input_handle, &logits_,
          /*topk_tokens_handle=*/nullptr,
          /*topk_probabilities_handle=*/nullptr, &constraint_mask_handle,
          /*sequence_size=*/1));
  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  create_info_copy.external_immutable_tensors[input_handle.id] =
      model_one_->context->GetTensor(model_one_->output.id);
  params_f32_ = std::make_unique<Tensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      env->context(), params_f32_handle.tensor_desc, params_f32_.get()));

  create_info_copy.external_immutable_tensors[params_i32_handle.id] =
      text_transformer_params_.params_i32.get();
  create_info_copy.external_immutable_tensors[params_f32_handle.id] =
      params_f32_.get();
  create_info_copy
      .external_immutable_tensors[stacked_transformer_input_handle.id] =
      model_one_->context->GetTensor(model_one_->input.id);
  create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
      tokens_ids_.get();

  constraint_mask_ = std::make_unique<Tensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(env->context(),
                                    constraint_mask_handle.tensor_desc,
                                    constraint_mask_.get()));
  ABSL_RETURN_IF_ERROR(
      UpdateConstraintMask(odml::infra::llm_utils::AllAllowedBitmap(), env));
  create_info_copy.external_immutable_tensors[constraint_mask_handle.id] =
      constraint_mask_.get();

  post_processing_ = std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(post_processing_->InitFromGpuModel(
      create_info_copy, &gpu_model, env, nullptr));

  // Create softmax model.
  ABSL_ASSIGN_OR_RETURN(
      GpuModel softmax_model,
      transformer_builder->CreateSoftmaxModel(&logits_, &token_probabilities_));

  create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  create_info_copy.external_immutable_tensors[logits_.id] =
      post_processing_->GetTensor(logits_.id);

  softmax_ = std::make_unique<InferenceContext>();
  return softmax_->InitFromGpuModel(create_info_copy, &softmax_model, env,
                                    nullptr);
}

absl::Status LlmOpenCl::InitCleanCacheModel(
    TransformerBuilder* transformer_builder, TransformerCache& cache,
    TransformerParams& params, Environment* env) {
  // Basic clean cache model.
  {
    std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
    std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
    GpuModelBuilder::TensorHandle params_i32_handle;

    ABSL_ASSIGN_OR_RETURN(
        GpuModel gpu_model,
        transformer_builder->CreateCleanCacheModel(
            &key_cached_tensor_handles, &value_cached_tensor_handles,
            &params_i32_handle, nullptr));
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
        create_info_copy, &gpu_model, env, nullptr));
  }

  // Clean cache at indices model.
  {
    std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
    std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
    GpuModelBuilder::TensorHandle params_i32_handle;

    ABSL_ASSIGN_OR_RETURN(
        GpuModel gpu_model,
        transformer_builder->CreateCleanCacheModel(
            &key_cached_tensor_handles, &value_cached_tensor_handles,
            &params_i32_handle, &cache.ids_to_remove));
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

    cache.clean_cache_at_ids =
        std::make_unique<InferenceContext>(InferenceContext());
    return cache.clean_cache_at_ids->InitFromGpuModel(create_info_copy,
                                                      &gpu_model, env, nullptr);
  }

  return absl::OkStatus();
}

absl::Status LlmOpenCl::InitEmbeddingLookupModel(
    TransformerBuilder* transformer_builder, Environment* env,
    StackedTransformerInference& model, const int sequence_size,
    const int batch_size) {
  ABSL_ASSIGN_OR_RETURN(
      GpuModel gpu_model,
      transformer_builder->CreateEmbeddingLookupModel(
          sequence_size, batch_size, &model.embedding_lookup_model.input_tokens,
          &model.embedding_lookup_model.output_embeddings,
          &model.embedding_lookup_model.ple_weights,
          &model.embedding_lookup_model.output_ple_embeddings));

  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  EmbeddingLookupModel& embedding_model = model.embedding_lookup_model;
  embedding_model.context = std::make_unique<InferenceContext>();
  ABSL_RETURN_IF_ERROR(embedding_model.context->InitFromGpuModel(
      create_info_copy, &gpu_model, env, nullptr));

  embedding_model.copy_tokens_op = std::make_unique<ClOperation>(ClOperation());
  ABSL_RETURN_IF_ERROR(InitClOp(
      env,
      CreateCopyTokensOrPositionsOp(embedding_model.input_tokens.tensor_desc),
      embedding_model.copy_tokens_op.get()));
  return absl::OkStatus();
}

absl::Status LlmOpenCl::InitAudioEncoderModel(
    AudioEncoderBuilder* audio_encoder_builder, Environment* env) {
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
    if (!IsModelv3(audio_params_.config.transformer) ||
        audio_encoder_builder->IsStreaming()) {
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
    if (IsModelv3(audio_params_.config.transformer)) {
      create_info_copy.external_immutable_tensors[signal_handle.id] =
          audio_params_.signal.get();
      if (audio_encoder_builder->IsStreaming()) {
        for (int i = 0; i < stream_depthwise_tensor_handles.size(); ++i) {
          create_info_copy.external_immutable_tensors
              [stream_depthwise_tensor_handles[i].id] =
              audio_params_.stream_depthwise_tensors[i].get();
        }
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
    ABSL_RETURN_IF_ERROR(
        audio_params_.signal->UploadDescriptorData(signal_desc, env->queue()));
  }
  audio_params_.context =
      std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(audio_params_.context->InitFromGpuModel(
      create_info_copy, &gpu_model, env, nullptr));
  return absl::OkStatus();
}

absl::Status LlmOpenCl::InitHelperOps(Environment* env) {
  const auto& gpu_info = env->GetDevicePtr()->GetInfo();
  if (text_transformer_params_.params_i32) {
    text_transformer_params_.write_i32_params = std::make_unique<ClOperation>();
    ABSL_RETURN_IF_ERROR(
        InitClOp(env,
                 CreateWriteParamsOp(
                     text_transformer_params_.params_i32->GetDescriptor(),
                     text_transformer_params_.params_i32->Channels()),
                 text_transformer_params_.write_i32_params.get()));
  }
  if (params_f32_) {
    write_f32_params_ = std::make_unique<ClOperation>(ClOperation());
    ABSL_RETURN_IF_ERROR(
        InitClOp(env,
                 CreateWriteParamsOp(params_f32_->GetDescriptor(),
                                     params_f32_->Channels()),
                 write_f32_params_.get()));
  }
  for (auto& kv : models_) {
    auto& model = kv.second;

    if (enable_external_embeddings_) {
      model.copy_external_embeddings_op =
          std::make_unique<ClOperation>(ClOperation());
      ABSL_RETURN_IF_ERROR(InitClOp(
          env,
          CreateCopyExternalEmbeddingsOp(model.external_embeddings.tensor_desc),
          model.copy_external_embeddings_op.get()));
    } else {
      model.copy_tokens_op = std::make_unique<ClOperation>(ClOperation());
      ABSL_RETURN_IF_ERROR(
          InitClOp(env, CreateCopyTokensOrPositionsOp(model.input.tensor_desc),
                   model.copy_tokens_op.get()));
    }
    if (enable_decode_logits_) {
      if (enable_external_embeddings_) {
        model.copy_positions_op = std::make_unique<ClOperation>(ClOperation());
        ABSL_RETURN_IF_ERROR(InitClOp(
            env, CreateCopyTokensOrPositionsOp(model.positions.tensor_desc),
            model.copy_positions_op.get()));
      } else {
        model.copy_tokens_and_positions_op =
            std::make_unique<ClOperation>(ClOperation());
        ABSL_RETURN_IF_ERROR(
            InitClOp(env,
                     CreateCopyTokensAndPositionsOp(
                         model.input.tensor_desc, model.positions.tensor_desc),
                     model.copy_tokens_and_positions_op.get()));
      }
      model.copy_masks_op = std::make_unique<ClOperation>(ClOperation());
      ABSL_RETURN_IF_ERROR(
          InitClOp(env,
                   CreateCopyMasksOp(model.local_mask.tensor_desc,
                                     model.global_mask.tensor_desc,
                                     text_config_.cache_size),
                   model.copy_masks_op.get()));
    }

    if ((kv.first != 1 || enable_decode_logits_) && model_one_->context) {
      model.copy_nth_op = std::make_unique<ClOperation>(ClOperation());
      ABSL_RETURN_IF_ERROR(
          InitClOp(env,
                   CreateCopyNthToSingleOp(gpu_info, model.output.tensor_desc,
                                           model_one_->output.tensor_desc),
                   model.copy_nth_op.get()));
    }
  }
  // Init copy cache op
  copy_cache_ = std::make_unique<ClOperation>(ClOperation());
  ABSL_RETURN_IF_ERROR(InitClOp(
      env,
      CreateCopyCacheOp(text_cache_.key_cached_tensors[0]->GetDescriptor(),
                        text_cache_.value_cached_tensors[0]->GetDescriptor(),
                        text_config_, output_batch_size_),
      copy_cache_.get()));

  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    copy_cache_global_ = std::make_unique<ClOperation>(ClOperation());
    int global_idx = text_config_.global_layer_offset.value_or(
        text_config_.num_local_layers_per_global);
    if (global_idx < text_cache_.key_cached_tensors.size()) {
      ABSL_RETURN_IF_ERROR(InitClOp(
          env,
          CreateCopyCacheOp(
              text_cache_.key_cached_tensors[global_idx]->GetDescriptor(),
              text_cache_.value_cached_tensors[global_idx]->GetDescriptor(),
              text_config_, output_batch_size_, /*is_global=*/true),
          copy_cache_global_.get()));
    }
  }

  if (media_cache_size_ > 0) {
    multimodal_embeddings_op_ = std::make_unique<ClOperation>(ClOperation());
    ABSL_RETURN_IF_ERROR(InitClOp(
        env,
        CreateCopyMediaEmbeddingsOp(media_embeddings_->GetDescriptor(),
                                    media_embeddings_->GetDescriptor()),
        multimodal_embeddings_op_.get()));
  }

  return absl::OkStatus();
}

absl::Status LlmOpenCl::InitInputTokens(Environment* env,
                                        const std::vector<int>& input_tokens,
                                        int time_step) {
  // If the input token is empty, we don't need to do anything.
  if (input_tokens.empty()) {
    return absl::OkStatus();
  }
  // If the input token is the last token in the input tokens, we don't need to
  // do anything.
  if (input_tokens.size() == 1 && ready_tokens_ == 1 &&
      input_tokens[0] == input_tokens_[input_tokens_.size() - 1] &&
      time_step == input_tokens_.size() - 1 &&
      !constraint_mask_changed_since_last_input_) {
    return absl::OkStatus();
  }
  constraint_mask_changed_since_last_input_ = false;
  ABSL_RETURN_IF_ERROR(CleanCache(env, time_step));
  if (input_tokens_.size() < time_step &&
      !model_one_->ngrammer_weights.empty()) {
    return absl::InternalError("time_step is too large.");
  }
  input_tokens_.resize(time_step);
  ABSL_RETURN_IF_ERROR(InitInputTokensWithModelN(env, input_tokens));
  input_tokens_.insert(input_tokens_.end(), input_tokens.begin(),
                       input_tokens.end());
  {
    int token_index = input_tokens_.size() - 1;
    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = token_index;
    llm_params.active_tokens = token_index + 1;
    llm_params.topk = sampler_params_.k();
    ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
        env, text_config_, text_transformer_params_, llm_params));
  }

  ABSL_RETURN_IF_ERROR(ExecutePostProcessing(env, &last_token_enqueued_event_));
  ready_tokens_ = 1;
  return absl::OkStatus();
}

absl::Status LlmOpenCl::InitInputTokensWithModelN(
    Environment* env, const std::vector<int>& input_tokens) {
  if (input_tokens.empty()) {
    return absl::OkStatus();
  }
  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      input_token_ids_->GetMemoryPtr(), input_tokens.size() * sizeof(int32_t),
      input_tokens.data(), /*async=*/true));
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
        env, model_n, previous_token,
        absl::MakeSpan(input_tokens).subspan(token_index_offset, group_size)));

    {  // Prepare input.
      auto& copy_tokens_op = model_n.copy_tokens_op;
      ABSL_RETURN_IF_ERROR(
          copy_tokens_op->SetSrcBuffer(0, input_token_ids_.get()));
      ABSL_RETURN_IF_ERROR(copy_tokens_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.input.id)));
      ABSL_RETURN_IF_ERROR(
          copy_tokens_op->SetInt("token_offset", token_index_offset));
      ABSL_RETURN_IF_ERROR(
          copy_tokens_op->SetInt("tokens_count", input_tokens.size()));
      ABSL_RETURN_IF_ERROR(copy_tokens_op->UpdateParams());
      ABSL_RETURN_IF_ERROR(copy_tokens_op->AddToQueue(env->queue()));
    }
    const int active_positions =
        std::min(token_index_offset + model_n.seq_size,
                 static_cast<int>(input_tokens.size()));
    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = input_tokens_.size() + token_index_offset;
    llm_params.active_tokens = input_tokens_.size() + active_positions;
    llm_params.topk = sampler_params_.k();
    ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
        env, text_config_, text_transformer_params_, llm_params));
    ABSL_RETURN_IF_ERROR(model_n.context->AddToQueue(env->queue()));
    token_index_offset += group_size;
  }
  // Copy model_n last token from output to model_one output.
  auto& last_model = models_[input_groups.back()];
  if (last_model.seq_size != 1) {
    auto& copy_nth_op = last_model.copy_nth_op;
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetSrcTensor(
        0, last_model.context->GetTensor(last_model.output.id)));
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetDstTensor(
        0, model_one_->context->GetTensor(model_one_->output.id)));
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetInt(
        "nth", (input_tokens.size() - 1) % last_model.seq_size));
    ABSL_RETURN_IF_ERROR(copy_nth_op->UpdateParams());
    ABSL_RETURN_IF_ERROR(copy_nth_op->AddToQueue(env->queue()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>>
LlmOpenCl::GetVariants(Environment* env) {
  std::vector<std::pair<uint32_t, uint32_t>> variants;
  variants.reserve(models_.size());
  for (const auto& kv : models_) {
    variants.push_back(std::pair<uint32_t, uint32_t>{
        static_cast<uint32_t>(kv.first),
        static_cast<uint32_t>(text_config_.cache_size)});
  }
  return variants;
}

absl::StatusOr<std::vector<float>> LlmOpenCl::DecodeToLogits(
    Environment* env, const uint32_t model_size, const uint32_t cache_size,
    absl::Span<const int> input_tokens,
    absl::Span<const float> local_attention_mask,
    absl::Span<const float> global_attention_mask,
    absl::Span<const int> seq_position,
    absl::Span<const uint32_t> logits_indices, const std::atomic_bool* cancel) {
  if (enable_external_embeddings_) {
    return absl::FailedPreconditionError(
        "DecodeToLogits (token input)"
        "called but enable_external_embeddings is true. "
        "Use DecodeEmbeddingsToLogits instead.");
  }
  std::vector<float> result;
  if (!enable_decode_logits_) {
    return absl::InvalidArgumentError(
        "LlmOpenCl::DecodeToLogits is not allowed without "
        "enable_decode_logits_=true");
  }

  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      input_token_ids_->GetMemoryPtr(), input_tokens.size() * sizeof(int32_t),
      input_tokens.data(), /*async=*/true));

  ABSL_RETURN_IF_ERROR(
      env->queue()->EnqueueWriteBuffer(input_sequence_position_->GetMemoryPtr(),
                                       seq_position.size() * sizeof(int32_t),
                                       seq_position.data(), /*async=*/true));

  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      input_global_mask_->GetMemoryPtr(),
      global_attention_mask.size() * sizeof(float),
      global_attention_mask.data(), /*async=*/true));

  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      input_local_mask_->GetMemoryPtr(),
      local_attention_mask.size() * sizeof(float), local_attention_mask.data(),
      /*async=*/true));
  {
    int group_size = model_size;
    auto& model_n = models_[group_size];

    int previous_token = 0;
    if (!input_tokens_.empty()) {
      previous_token = input_tokens_.back();
    }

    ABSL_RETURN_IF_ERROR(
        UpdateEmbeddings(env, model_n, previous_token,
                         absl::MakeSpan(input_tokens).subspan(0, group_size)));

    // Copy tokens and positions
    {
      auto& copy_raw_op = model_n.copy_tokens_and_positions_op;
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(0, input_token_ids_.get()));
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(1, input_sequence_position_.get()));

      ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.input.id)));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          1, model_n.context->GetTensor(model_n.positions.id)));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetInt("token_offset", 0));
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetInt("tokens_count", input_tokens.size()));
      ABSL_RETURN_IF_ERROR(copy_raw_op->UpdateParams());
      ABSL_RETURN_IF_ERROR(copy_raw_op->AddToQueue(env->queue()));
    }

    // Copy masks
    {
      auto& copy_raw_op = model_n.copy_masks_op;
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(0, input_local_mask_.get()));
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(1, input_global_mask_.get()));

      ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.local_mask.id)));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          1, model_n.context->GetTensor(model_n.global_mask.id)));
      ABSL_RETURN_IF_ERROR(copy_raw_op->UpdateParams());
      ABSL_RETURN_IF_ERROR(copy_raw_op->AddToQueue(env->queue()));
    }

    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = input_tokens_.size();
    llm_params.active_tokens = input_tokens_.size() + input_tokens.size();
    llm_params.topk = sampler_params_.k();
    ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
        env, text_config_, text_transformer_params_, llm_params));
    ABSL_RETURN_IF_ERROR(model_n.context->AddToQueue(env->queue()));
  }

  for (uint32_t gen_logits_index : logits_indices) {
    auto& last_model = models_[model_size];
    auto& copy_nth_op = last_model.copy_nth_op;

    ABSL_RETURN_IF_ERROR(copy_nth_op->SetSrcTensor(
        0, last_model.context->GetTensor(last_model.output.id)));
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetDstTensor(
        0, final_embedding_to_logits_->GetTensor(final_embedding_.id)));
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetInt("nth", gen_logits_index));
    ABSL_RETURN_IF_ERROR(copy_nth_op->UpdateParams());
    ABSL_RETURN_IF_ERROR(copy_nth_op->AddToQueue(env->queue()));
    ABSL_RETURN_IF_ERROR(final_embedding_to_logits_->AddToQueue(env->queue()));

    // Download logits
    TensorDescriptor desc;
    ABSL_RETURN_IF_ERROR(final_embedding_to_logits_->GetTensor(logits_.id)
                             ->ToDescriptor(&desc, env->queue()));
    TensorFloat32 output;
    desc.DownloadData(&output);
    result.insert(result.end(), output.data.begin(), output.data.end());
  }

  // Update processed tokens
  input_tokens_.insert(input_tokens_.end(), input_tokens.begin(),
                       input_tokens.end());

  return result;
}

absl::Status LlmOpenCl::ComputeTokenEmbeddings(
    Environment* env, absl::Span<const int> input_tokens,
    absl::Span<float> output_embeddings,
    absl::Span<float> output_ple_embeddings) {
  size_t input_size = input_tokens.size();
  if (input_size == 0) {
    return absl::InvalidArgumentError("Input tokens cannot be empty.");
  }

  // Copy input tokens to input_token_ids_.
  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      input_token_ids_->GetMemoryPtr(), input_size * sizeof(int32_t),
      input_tokens.data(), /*async=*/true));

  ABSL_ASSIGN_OR_RETURN(auto input_groups,
                        GetGroupSizesForInput(input_size, group_sizes_));

  int token_index_offset = 0;
  size_t result_write_offset = 0;
  size_t ple_result_write_offset = 0;

  for (int i = 0; i < input_groups.size(); ++i) {
    int group_size = input_groups[i];
    auto& model_n = models_[group_size];

    const int active_token_size =
        std::min(static_cast<int>(input_size - token_index_offset), group_size);

    ABSL_RETURN_IF_ERROR(UpdatePleEmbeddings(
        env, model_n,
        input_tokens.subspan(token_index_offset, active_token_size)));

    EmbeddingLookupModel& embedding_model = model_n.embedding_lookup_model;
    // Generate embeddings for the group.
    {
      // Prepare input.
      auto& copy_tokens_op = embedding_model.copy_tokens_op;
      ABSL_RETURN_IF_ERROR(
          copy_tokens_op->SetSrcBuffer(0, input_token_ids_.get()));
      ABSL_RETURN_IF_ERROR(copy_tokens_op->SetDstTensor(
          0,
          embedding_model.context->GetTensor(embedding_model.input_tokens.id)));
      ABSL_RETURN_IF_ERROR(
          copy_tokens_op->SetInt("token_offset", token_index_offset));
      ABSL_RETURN_IF_ERROR(copy_tokens_op->SetInt(
          "tokens_count", token_index_offset + active_token_size));
      ABSL_RETURN_IF_ERROR(copy_tokens_op->UpdateParams());
      ABSL_RETURN_IF_ERROR(copy_tokens_op->AddToQueue(env->queue()));
    }
    ABSL_RETURN_IF_ERROR(embedding_model.context->AddToQueue(env->queue()));

    // --- Download embeddings ---
    TensorDescriptor embedding_desc;
    ABSL_RETURN_IF_ERROR(
        embedding_model.context->GetTensor(embedding_model.output_embeddings.id)
            ->ToDescriptor(&embedding_desc, env->queue()));
    TensorFloat32 output;
    embedding_desc.DownloadData(&output);

    size_t embeddings_to_copy =
      active_token_size * text_config_.model_dimension;

    // Bounds check for safety
    if (result_write_offset + embeddings_to_copy > output_embeddings.size()) {
      return absl::InternalError(
          "ComputeTokenEmbeddings: output_embeddings buffer is too small.");
    }

    // Copy directly to the output span
    std::copy_n(output.data.begin(), embeddings_to_copy,
                output_embeddings.begin() + result_write_offset);
    result_write_offset += embeddings_to_copy;

    // --- Download PLE embeddings ---
    const int per_layer_embedding_dimension =
        *text_config_.per_layer_embedding_dimension;
    const size_t num_ple_layers = embedding_model.output_ple_embeddings.size();
    const size_t total_ple_dim_per_token =
        num_ple_layers * per_layer_embedding_dimension;

    // Temporary buffer to handle the Layer-Major to Token-Major transpose
    // for this group.
    std::vector<float> group_ple_buffer(
      active_token_size * total_ple_dim_per_token);

    for (int layer_idx = 0; layer_idx < num_ple_layers; ++layer_idx) {
      TensorDescriptor ple_output_desc;
      ABSL_RETURN_IF_ERROR(
          embedding_model.context
              ->GetTensor(embedding_model.output_ple_embeddings[layer_idx].id)
              ->ToDescriptor(&ple_output_desc, env->queue()));
      TensorFloat32 ple_output_data;
      ple_output_desc.DownloadData(&ple_output_data);
      RET_CHECK_EQ(ple_output_data.data.size(),
                   per_layer_embedding_dimension * group_size);

      // Distribute layer data to the correct token slots in the temporary
      absl::Span<const float> src_span =
          absl::MakeConstSpan(ple_output_data.data);
      absl::Span<float> dst_span = absl::MakeSpan(group_ple_buffer);

      for (int t = 0; t < active_token_size; ++t) {
        // Source: This layer's data for token 't'
        absl::Span<const float> token_layer_src = src_span.subspan(
            t * per_layer_embedding_dimension, per_layer_embedding_dimension);

        // Dest: Token 't' block, offset by current layer
        absl::Span<float> token_layer_dst = dst_span.subspan(
            (t * total_ple_dim_per_token) +
                (layer_idx * per_layer_embedding_dimension),
            per_layer_embedding_dimension);

        absl::c_copy(token_layer_src,
                  token_layer_dst.begin());
      }
    }

    // Copy fully re-ordered group buffer to the final ple_result span
    size_t ple_data_to_copy = group_ple_buffer.size();

    // Bounds check for safety
    if (ple_result_write_offset + ple_data_to_copy >
        output_ple_embeddings.size()) {
      return absl::InternalError(
          "ComputeTokenEmbeddings: output_ple_embeddings buffer is too small.");
    }

    std::copy_n(group_ple_buffer.begin(), ple_data_to_copy,
                output_ple_embeddings.begin() + ple_result_write_offset);
    ple_result_write_offset += ple_data_to_copy;

    token_index_offset += active_token_size;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> LlmOpenCl::GetNextTokenProbabilities(
    Environment* env) {
  ABSL_RETURN_IF_ERROR(softmax_->AddToQueue(env->queue()));
  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(softmax_->GetTensor(token_probabilities_.id)
                           ->ToDescriptor(&desc, env->queue()));

  TensorFloat32 output;
  desc.DownloadData(&output);
  return output.data;
}

absl::StatusOr<std::vector<float>> LlmOpenCl::DecodeToLogits(
    Environment* env, const uint32_t model_size, const uint32_t cache_size,
    absl::Span<const float> embeddings, absl::Span<const float> ple_embeddings,
    absl::Span<const float> local_attention_mask,
    absl::Span<const float> global_attention_mask,
    absl::Span<const int> seq_position,
    absl::Span<const uint32_t> logits_indices, const std::atomic_bool* cancel) {
  if (!enable_external_embeddings_) {
    return absl::InvalidArgumentError(
        "LlmOpenCl::DecodeToLogits with embeddings is not allowed without "
        "enable_external_embeddings_=false");
  }
  if (!enable_decode_logits_) {
    return absl::InvalidArgumentError(
        "LlmOpenCl::DecodeToLogits is not allowed without "
        "enable_decode_logits_=true");
  }

  std::vector<float> result;
  // The ple_embedding is model_size * stack_size *
  // per_layer_embedding_dimension while the input to the model is
  // size stack_size * model_size * embedding_dimension.
  // Thus, we need the conversion to happen
  std::vector<float> ple_input_data;
  const int per_layer_embedding_dimension =
      *text_config_.per_layer_embedding_dimension;
  for (int i = 0; i < text_config_.stack_size; ++i) {
    for (int j = 0; j < model_size; ++j) {
      // The start index of the PLE embeddings for token j at layer i.
      int start_index =
          j * text_config_.stack_size * per_layer_embedding_dimension +
          i * per_layer_embedding_dimension;
      ple_input_data.insert(
          ple_input_data.end(), ple_embeddings.begin() + start_index,
          ple_embeddings.begin() + start_index + per_layer_embedding_dimension);
    }
  }

  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      input_embeddings_->GetMemoryPtr(), embeddings.size() * sizeof(float),
      embeddings.data(), /*async=*/true));

  ABSL_RETURN_IF_ERROR(
      env->queue()->EnqueueWriteBuffer(input_ple_embeddings_->GetMemoryPtr(),
                                       ple_input_data.size() * sizeof(float),
                                       ple_input_data.data(), /*async=*/true));

  ABSL_RETURN_IF_ERROR(
      env->queue()->EnqueueWriteBuffer(input_sequence_position_->GetMemoryPtr(),
                                       seq_position.size() * sizeof(int32_t),
                                       seq_position.data(), /*async=*/true));

  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      input_global_mask_->GetMemoryPtr(),
      global_attention_mask.size() * sizeof(float),
      global_attention_mask.data(), /*async=*/true));

  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      input_local_mask_->GetMemoryPtr(),
      local_attention_mask.size() * sizeof(float), local_attention_mask.data(),
      /*async=*/true));

  {
    int group_size = model_size;
    auto& model_n = models_[group_size];
    // Copy embeddings
    {
      auto& copy_raw_op = model_n.copy_external_embeddings_op;
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(0, input_embeddings_.get()));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.external_embeddings.id)));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetInt("embedding_offset", 0));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetInt("embedding_dimension",
                                               text_config_.model_dimension));
      ABSL_RETURN_IF_ERROR(copy_raw_op->UpdateParams());
      ABSL_RETURN_IF_ERROR(copy_raw_op->AddToQueue(env->queue()));
    }

    // Copy PLE embeddings
    {
      int embedding_offset = 0;
      for (int i = 0; i < model_n.external_ple_embeddings.size(); ++i) {
        auto& copy_raw_op = model_n.copy_external_embeddings_op;
        ABSL_RETURN_IF_ERROR(
            copy_raw_op->SetSrcBuffer(0, input_ple_embeddings_.get()));
        ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
            0,
            model_n.context->GetTensor(model_n.external_ple_embeddings[i].id)));
        ABSL_RETURN_IF_ERROR(
            copy_raw_op->SetInt("embedding_dimension",
                                (*text_config_.per_layer_embedding_dimension)));
        ABSL_RETURN_IF_ERROR(
            copy_raw_op->SetInt("embedding_offset", embedding_offset));
        ABSL_RETURN_IF_ERROR(copy_raw_op->UpdateParams());
        ABSL_RETURN_IF_ERROR(copy_raw_op->AddToQueue(env->queue()));
        embedding_offset +=
            (*text_config_.per_layer_embedding_dimension) * model_size;
      }
    }

    // Copy positions
    {
      auto& copy_raw_op = model_n.copy_positions_op;
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(0, input_sequence_position_.get()));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.positions.id)));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetInt("token_offset", 0));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetInt("tokens_count", model_size));
      ABSL_RETURN_IF_ERROR(copy_raw_op->UpdateParams());
      ABSL_RETURN_IF_ERROR(copy_raw_op->AddToQueue(env->queue()));
    }

    // Copy masks
    {
      auto& copy_raw_op = model_n.copy_masks_op;
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(0, input_local_mask_.get()));
      ABSL_RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(1, input_global_mask_.get()));

      ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.local_mask.id)));
      ABSL_RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          1, model_n.context->GetTensor(model_n.global_mask.id)));
      ABSL_RETURN_IF_ERROR(copy_raw_op->UpdateParams());
      ABSL_RETURN_IF_ERROR(copy_raw_op->AddToQueue(env->queue()));
    }

    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = input_tokens_.size();
    llm_params.active_tokens = input_tokens_.size() + model_size;
    llm_params.topk = sampler_params_.k();
    ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
        env, text_config_, text_transformer_params_, llm_params));
    ABSL_RETURN_IF_ERROR(model_n.context->AddToQueue(env->queue()));
  }

  for (uint32_t gen_logits_index : logits_indices) {
    auto& last_model = models_[model_size];
    auto& copy_nth_op = last_model.copy_nth_op;

    ABSL_RETURN_IF_ERROR(copy_nth_op->SetSrcTensor(
        0, last_model.context->GetTensor(last_model.output.id)));
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetDstTensor(
        0, final_embedding_to_logits_->GetTensor(final_embedding_.id)));
    ABSL_RETURN_IF_ERROR(copy_nth_op->SetInt("nth", gen_logits_index));
    ABSL_RETURN_IF_ERROR(copy_nth_op->UpdateParams());
    ABSL_RETURN_IF_ERROR(copy_nth_op->AddToQueue(env->queue()));
    ABSL_RETURN_IF_ERROR(final_embedding_to_logits_->AddToQueue(env->queue()));

    // Download logits
    TensorDescriptor desc;
    ABSL_RETURN_IF_ERROR(final_embedding_to_logits_->GetTensor(logits_.id)
                             ->ToDescriptor(&desc, env->queue()));
    TensorFloat32 output;
    desc.DownloadData(&output);
    result.insert(result.end(), output.data.begin(), output.data.end());
  }

  // Insert 0s to make sure input_tokens_ size is correct.
  input_tokens_.insert(input_tokens_.end(), model_size, 0);
  return result;
}

absl::Status LlmOpenCl::GetNextTokens(Environment* env,
                                      std::vector<int>* output_tokens) {
  std::vector<std::vector<int>> temp(1);
  temp[0].resize(output_tokens->size());
  ABSL_RETURN_IF_ERROR(GetNextTokens(env, &temp));
  *output_tokens = std::move(temp[0]);
  return absl::OkStatus();
}

absl::Status LlmOpenCl::GetNextTokens(
    Environment* env, std::vector<std::vector<int>>* output_tokens) {
  if (output_tokens->size() > 1 && (!model_one_->ple_weights.empty() ||
                                    !model_one_->ngrammer_weights.empty())) {
    // TODO: b/355494120 - Support multi output with dynamic PLE.
    return absl::InternalError(
        "Multi-output is not supported with dynamic embedding loading.");
  }
  const int next_tokens_count = (*output_tokens)[0].size();
  if (input_tokens_.size() + next_tokens_count > max_tokens_) {
    return absl::OutOfRangeError(absl::StrCat(
        "Not possible to enqueue all requested tokens. Already "
        "processed tokens: ",
        input_tokens_.size(), " requested tokens: ", next_tokens_count,
        " max tokens: ", max_tokens_));
  }
  // Retrieve the event from the previous call.
  if (!last_token_enqueued_event_.is_valid()) {
    return absl::UnknownError("Unexpected invalid event");
  }
  CLEvent last_token_enqueued_event = std::move(last_token_enqueued_event_);
  cl_event last_token_enqueued_cl_event = last_token_enqueued_event.event();
  for (int i = 0; i < next_tokens_count; ++i) {
    int time_step = i + input_tokens_.size();
    // The last 2 EnqueueNextToken calls are special. The second to last call is
    // the last event before the output is ready, thus if it exists, we need to
    // wait for it instead of the event from previous call. The last call is the
    // last task to make sure the command queue can continue processing, and the
    // event should be stored for the next GetNextTokens call.
    CLEvent* event_ptr =
        i >= next_tokens_count - 2 ? &last_token_enqueued_event_ : nullptr;
    ABSL_RETURN_IF_ERROR(EnqueueNextToken(env, time_step, event_ptr));
    // If this is the second to last token, we need to wait for this event
    // instead of the event from previous call.
    if (i == next_tokens_count - 2) {
      if (!last_token_enqueued_event_.is_valid()) {
        return absl::UnknownError("Unexpected invalid event");
      }
      last_token_enqueued_event = std::move(last_token_enqueued_event_);
      last_token_enqueued_cl_event = last_token_enqueued_event.event();
    }
  }

  const int first_output = input_tokens_.size();
  cl_int error_code = clEnqueueReadBuffer(
      reading_queue_.queue(), tokens_ids_->GetMemoryPtr(), CL_TRUE,
      first_output * output_batch_size_ * sizeof(int32_t),
      next_tokens_count * output_batch_size_ * sizeof(int32_t),
      output_tokens_.data() + first_output * output_batch_size_,
      /*num_events_in_wait_list=*/1, &last_token_enqueued_cl_event, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to clEnqueueReadBuffer - ",
                                           CLErrorCodeToString(error_code)));
  }
  for (int b = 0; b < output_batch_size_; ++b) {
    auto& tokens = (*output_tokens)[b];
    for (int i = 0; i < tokens.size(); ++i) {
      const int global_index = i + first_output;
      tokens[i] = output_tokens_[global_index * output_batch_size_ + b];
    }
  }
  ready_tokens_ += next_tokens_count;

  // A single output stream can be treated as input tokens. This allows models
  // with dynamic embeddings to continue processing off of previous output.
  if (output_tokens->size() == 1) {
    const auto& output = (*output_tokens)[0];
    input_tokens_.insert(input_tokens_.end(), output.begin(), output.end());
    ready_tokens_ -= output.size();
  }
  return absl::OkStatus();
}

absl::Status LlmOpenCl::EnqueueNextToken(Environment* env, int time_step,
                                         CLEvent* event) {
  if (time_step >= max_tokens_) {
    return absl::OutOfRangeError("Not possible to enqueue more tokens");
  }

  int32_t out_tokens[2];
  CLEvent read_event;
  if (!model_one_->ple_weights.empty() ||
      !model_one_->ngrammer_weights.empty()) {
    const int offset = std::max(time_step - 1, 0) * sizeof(int32_t);
    cl_int error_code = clEnqueueReadBuffer(
        env->queue()->queue(), tokens_ids_->GetMemoryPtr(), CL_FALSE, offset,
        2 * sizeof(int32_t), out_tokens, 0, nullptr, read_event.GetEventPtr());
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat("Failed to clEnqueueReadBuffer - ",
                                             CLErrorCodeToString(error_code)));
    }
    clFlush(env->queue()->queue());
  }

  LlmRuntimeParams llm_params;
  llm_params.token_index_offset = time_step;
  llm_params.active_tokens = time_step + 1;
  llm_params.topk = sampler_params_.k();
  ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
      env, text_config_, text_transformer_params_, llm_params));
  if (!model_one_->ple_weights.empty() ||
      !model_one_->ngrammer_weights.empty()) {
    // Active wait for the read event to complete.
    // With read_event.Wait(); performance on all tested devices is much worse
    // than active waiting.
    ABSL_RETURN_IF_ERROR(read_event.ActiveWait(env->queue()->queue()));
    ABSL_RETURN_IF_ERROR(
        UpdateEmbeddings(env, *model_one_, out_tokens[0], {out_tokens[1]}));
  }
  ABSL_RETURN_IF_ERROR(model_one_->context->AddToQueue(env->queue(), event));
  ABSL_RETURN_IF_ERROR(ExecutePostProcessing(env, event));
  if (env->device().GetInfo().opencl_info.IsCLVK()) {
    clFlush(env->queue()->queue());
  }
  return absl::OkStatus();
}

absl::StatusOr<GpuSpatialTensor*> LlmOpenCl::ExecuteAudioEncoder(
    const TensorFloat32& audio_tensor, const TensorFloat32& mask_tensor,
    int time_step, Environment* env) {
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
  ABSL_RETURN_IF_ERROR(
      ExecuteUpdateIntParams(env, audio_params_.config.transformer,
                             audio_params_.transformer_params, llm_params));

  // Set audio encoder inputs.
  TensorDescriptor feature_desc = audio_params_.feature->GetDescriptor();
  feature_desc.UploadData(audio_tensor);
  ABSL_RETURN_IF_ERROR(
      audio_params_.feature->UploadDescriptorData(feature_desc, env->queue()));
  TensorDescriptor mask_desc = audio_params_.mask->GetDescriptor();
  mask_desc.UploadData(mask_tensor);
  ABSL_RETURN_IF_ERROR(
      audio_params_.mask->UploadDescriptorData(mask_desc, env->queue()));

  // Run audio encoder
  ABSL_RETURN_IF_ERROR(audio_params_.context->AddToQueue(
      env->queue(), &last_token_enqueued_event_));
  return audio_params_.context->GetTensor(audio_params_.output_handle.id);
}

absl::Status LlmOpenCl::FillVisionEmbeddings(
    const TensorFloat32& vision_embedding_tensor,
    const std::vector<TensorFloat32>& vision_ple_tensors, int image_index,
    Environment* env) {
  if (media_cache_size_ == 0) {
    return absl::InvalidArgumentError("Media embedding cache was not set.");
  }
  if (image_index >= text_config_.max_num_images) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Image index %d must be less than %d.", image_index,
                        text_config_.max_num_images));
  }
  BHWC expected_vision_emb_shape{1, 1, text_config_.vision_tokens_num,
                                 text_config_.model_dimension};
  if (vision_embedding_tensor.shape != expected_vision_emb_shape) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Vision embedding shape %s must match %s",
                        ToString(vision_embedding_tensor.shape),
                        ToString(expected_vision_emb_shape)));
  }
  const int index = image_index * text_config_.vision_tokens_num;
  ABSL_RETURN_IF_ERROR(
      FillMediaEmbeddings(vision_embedding_tensor, index, env));
  if (!vision_ple_tensors.empty()) {
    ABSL_RETURN_IF_ERROR(
        FillVisionPleEmbeddings(vision_ple_tensors, index, env));
  }
  return absl::OkStatus();
}

absl::Status LlmOpenCl::FillAudioEmbeddings(Tensor& audio_tensor, int index,
                                            Environment* env) {
  if (media_cache_size_ == 0) {
    return absl::InvalidArgumentError("Media embedding cache was not set.");
  }
  return UpdateMultimodalEmbeddings(&audio_tensor, media_embeddings_.get(),
                                    index, env);
}

absl::Status LlmOpenCl::ClearVisionPle(Environment* env) {
  std::vector<TensorFloat32> vision_ple_tensors;
  vision_ple_tensors.reserve(vision_ple_tensors_.size());
  for (int i = 0; i < vision_ple_tensors_.size(); ++i) {
    const BHWC shape = vision_ple_tensors_[i]->GetDescriptor().GetBHWCShape();
    TensorFloat32 tensor;
    tensor.shape = shape;
    tensor.data.resize(shape.DimensionsProduct());
    vision_ple_tensors.push_back(std::move(tensor));
  }
  return FillVisionPleEmbeddings(vision_ple_tensors, 0, env);
}

absl::Status LlmOpenCl::UpdateSamplerParams(
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

absl::Status LlmOpenCl::CleanCache(Environment* env, int time_step,
                                   bool skip_ringbuffer_delta_check) {
  LlmRuntimeParams llm_params;
  llm_params.token_index_offset = time_step;
  llm_params.active_tokens =
      time_step == 0 ? 0 : input_tokens_.size();  // For ringbuffer indexing
  int local_cache_mem =
      TransformerBuilder::GetCacheSize(text_config_, /*is_global=*/false);
  // If we have a ring buffer mid-use...
  if (!skip_ringbuffer_delta_check &&
      llm_params.active_tokens > local_cache_mem) {
    // ... make sure we don't try to "jump back" too far. Just output an error
    // for now -- this should be handled in the engines.
    int history_buffer_size =
        local_cache_mem - *text_config_.sliding_window_size;
    if (history_buffer_size <= *llm_params.active_tokens - time_step) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Cannot rewind to time_step %d from %d. Ringbuffer "
                          "size is %d with sliding window of %d",
                          time_step, *llm_params.active_tokens, local_cache_mem,
                          *text_config_.sliding_window_size));
    }
  }
  ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
      env, text_config_, text_transformer_params_, llm_params));
  ABSL_RETURN_IF_ERROR(text_cache_.clean_cache->AddToQueue(env->queue()));

  if (time_step < input_tokens_.size()) {
    input_tokens_.resize(time_step);
  }
  return absl::OkStatus();
}

absl::Status LlmOpenCl::CleanCacheAt(Environment* env,
                                     absl::Span<const int32_t> to_remove) {
  LlmRuntimeParams llm_params;
  llm_params.active_tokens = input_tokens_.size();  // For ringbuffer indexing

  if (to_remove.size() > text_config_.max_indices_count_in_clean_cache_at) {
    return absl::InvalidArgumentError(
        absl::StrCat("Too many tokens to clean cache at: ", to_remove.size(),
                     " > ", text_config_.max_indices_count_in_clean_cache_at));
  }
  // Copy indices to remove to ids_to_remove_buffer_.
  ABSL_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
      text_cache_.ids_to_remove_buffer->GetMemoryPtr(),
      to_remove.size() * sizeof(int32_t), to_remove.data(), /*async=*/true));

  {  // Copy indices in ids_to_remove_buffer_ to the ids_to_remove_ input
    // tensor.
    auto& copy_tokens_op = model_one_->copy_tokens_op;
    ABSL_RETURN_IF_ERROR(copy_tokens_op->SetSrcBuffer(
        0, text_cache_.ids_to_remove_buffer.get()));
    ABSL_RETURN_IF_ERROR(copy_tokens_op->SetDstTensor(
        0, text_cache_.clean_cache->GetTensor(text_cache_.ids_to_remove.id)));
    ABSL_RETURN_IF_ERROR(copy_tokens_op->SetInt("token_offset", 0));
    ABSL_RETURN_IF_ERROR(
        copy_tokens_op->SetInt("tokens_count", to_remove.size()));
    ABSL_RETURN_IF_ERROR(copy_tokens_op->UpdateParams());
    ABSL_RETURN_IF_ERROR(copy_tokens_op->AddToQueue(env->queue()));
  }

  // top_k is used as the length of remove_indices inside clean_cache_at
  // op.
  llm_params.topk = to_remove.size();
  ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
      env, text_config_, text_transformer_params_, llm_params));
  ABSL_RETURN_IF_ERROR(
      text_cache_.clean_cache_at_ids->AddToQueue(env->queue()));

  return absl::OkStatus();
}

absl::Status LlmOpenCl::CleanAudioCache(Environment* env, int time_step) {
  if (audio_params_.config.transformer.relative_positional_embedding_config
          .has_value() &&
      audio_params_.config.feature.pad_for_streaming == 0) {
    return absl::InvalidArgumentError("No KV cache for v3 non-streaming.");
  }
  LlmRuntimeParams llm_params;
  llm_params.token_index_offset = time_step;
  ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
      env, text_config_, audio_params_.transformer_params, llm_params));
  ABSL_RETURN_IF_ERROR(
      audio_params_.cache.clean_cache->AddToQueue(env->queue()));
  return absl::OkStatus();
}

absl::StatusOr<ContextData> LlmOpenCl::SaveContext(Environment* env,
                                                   bool local_only) {
  const int num_cache_layers = text_cache_.key_cached_tensors.size();
  std::vector<std::unique_ptr<Tensor>> key_tensors(num_cache_layers);
  std::vector<std::unique_ptr<Tensor>> value_tensors(num_cache_layers);
  // Init kv tensors to save to.
  // NOTE: This assumes we don't start off an architecture with two global
  // layers. That has always been the case so far, but if it changes, our logic
  // here will need to change too.
  const int local_id = IsLayerGlobal(0) ? 1 : 0;
  const DataType data_type =
      text_cache_.key_cached_tensors[local_id]->GetDescriptor().GetDataType();
  const size_t local_cache_size =
      TransformerBuilder::GetCacheSize(text_config_, /*is_global=*/false);
  const size_t local_alloc_size = std::max<size_t>(
      1, std::min<size_t>(input_tokens_.size(), local_cache_size));
  const TensorDescriptor k_local_desc =
      TransformerBuilder::GetKCacheTensorDescriptor(
          text_config_, output_batch_size_, local_alloc_size, data_type,
          /*is_global=*/false);
  const TensorDescriptor v_local_desc =
      TransformerBuilder::GetVCacheTensorDescriptor(
          text_config_, output_batch_size_, local_alloc_size, data_type,
          /*is_global=*/false);
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
                         *env, k_local_desc, k_global_desc,
                         text_config_.num_local_layers_per_global,
                         num_cache_layers, text_config_.global_layer_offset));
    ABSL_ASSIGN_OR_RETURN(
        value_tensors, InitKvTensorsLocalGlobal(
                           *env, v_local_desc, v_global_desc,
                           text_config_.num_local_layers_per_global,
                           num_cache_layers, text_config_.global_layer_offset));
  } else {
    ABSL_ASSIGN_OR_RETURN(key_tensors,
                          InitKvTensors(*env, k_local_desc, num_cache_layers));
    ABSL_ASSIGN_OR_RETURN(value_tensors,
                          InitKvTensors(*env, v_local_desc, num_cache_layers));
  }
  // Copy kv tensors to new kv tensors.
  int last_copied_layer = -1;
  for (int i = 0; i < num_cache_layers; ++i) {
    if (!local_only || !IsLayerGlobal(i)) {
      last_copied_layer = i;
    }
  }
  CLEvent* event = nullptr;
  for (int i = 0; i < num_cache_layers; ++i) {
    const bool is_global = IsLayerGlobal(i);
    if (local_only && is_global) {
      continue;
    }
    if (i == last_copied_layer) {
      event = &last_token_enqueued_event_;
    }
    const size_t layer_cache_size =
        TransformerBuilder::GetCacheSize(text_config_, is_global);
    const size_t layer_copy_size =
        std::min<size_t>(input_tokens_.size(), layer_cache_size);
    ABSL_RETURN_IF_ERROR(ExecuteCopyCache(
        env, layer_cache_size, text_cache_.key_cached_tensors[i].get(),
        text_cache_.value_cached_tensors[i].get(), layer_copy_size,
        key_tensors[i].get(), value_tensors[i].get(), is_global, event));
  }
  // Place everything inside context data.
  ContextData context_data;
  context_data.key_tensors.resize(num_cache_layers);
  context_data.value_tensors.resize(num_cache_layers);
  for (int i = 0; i < num_cache_layers; ++i) {
    context_data.key_tensors[i] = std::move(key_tensors[i]);
    context_data.value_tensors[i] = std::move(value_tensors[i]);
  }
  context_data.lora_id = current_lora_id_;
  context_data.tokens = input_tokens_;
  return context_data;
}

absl::Status LlmOpenCl::RestoreContext(Environment* env,
                                       const ContextData& context_data,
                                       bool local_only) {
  // Check that the context data is valid.
  if (context_data.lora_id != current_lora_id_) {
    return absl::InternalError("LoRA ID for context does not match.");
  }
  // NOTE: This assumes we don't start off an architecture with two global
  // layers. That has always been the case so far, but if it changes, our logic
  // here will need to change too.
  const int local_id = IsLayerGlobal(0) ? 1 : 0;
  if (context_data.key_tensors[local_id]->GetDescriptor().GetDataType() !=
      text_cache_.key_cached_tensors[local_id]->GetDataType()) {
    return absl::InternalError("Data types do not match.");
  }
  if (!local_only) {
    // Copy over misc data.
    input_tokens_ = context_data.tokens;
    ready_tokens_ = 0;

    TensorDescriptor desc = tokens_ids_->GetDescriptor();
    std::vector<int> tokens_data = context_data.tokens;
    tokens_data.resize(desc.GetBHWDCShape().DimensionsProduct());
    desc.UploadData(tokens_data.data());
    ABSL_RETURN_IF_ERROR(tokens_ids_->UploadDescriptorData(desc, env->queue()));
  }

  // Copy over the cache.
  int last_copied_layer = -1;
  for (int i = 0; i < context_data.key_tensors.size(); ++i) {
    if (!local_only || !IsLayerGlobal(i)) {
      last_copied_layer = i;
    }
  }
  CLEvent* event = nullptr;
  for (int i = 0; i < context_data.key_tensors.size(); ++i) {
    const bool is_global = IsLayerGlobal(i);
    if (local_only && is_global) {
      continue;
    }
    if (i == last_copied_layer) {
      event = &last_token_enqueued_event_;
    }
    const size_t layer_cache_size =
        TransformerBuilder::GetCacheSize(text_config_, is_global);
    const size_t layer_copy_size =
        std::min<size_t>(context_data.tokens.size(), layer_cache_size);
    ABSL_RETURN_IF_ERROR(ExecuteCopyCache(
        env, layer_copy_size,
        dynamic_cast<Tensor*>(context_data.key_tensors[i].get()),
        dynamic_cast<Tensor*>(context_data.value_tensors[i].get()),
        layer_cache_size, text_cache_.key_cached_tensors[i].get(),
        text_cache_.value_cached_tensors[i].get(), is_global, event));
  }
  return absl::OkStatus();
}

absl::StatusOr<ContextData> LlmOpenCl::CloneContext(
    Environment* env, const ContextData& context_data, bool local_only) {
  // Init new kv tensors.
  const int num_of_cache_layers = context_data.key_tensors.size();
  const int local_id = IsLayerGlobal(0) ? 1 : 0;
  std::vector<std::unique_ptr<Tensor>> key_tensors(num_of_cache_layers);
  std::vector<std::unique_ptr<Tensor>> value_tensors(num_of_cache_layers);
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
            *env, context_data.key_tensors[local_id]->GetDescriptor(),
            k_global_desc, text_config_.num_local_layers_per_global,
            num_of_cache_layers, text_config_.global_layer_offset));
    ABSL_ASSIGN_OR_RETURN(
        value_tensors,
        InitKvTensorsLocalGlobal(
            *env, context_data.value_tensors[local_id]->GetDescriptor(),
            v_global_desc, text_config_.num_local_layers_per_global,
            num_of_cache_layers, text_config_.global_layer_offset));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        key_tensors,
        InitKvTensors(*env, context_data.key_tensors[local_id]->GetDescriptor(),
                      num_of_cache_layers));
    ABSL_ASSIGN_OR_RETURN(
        value_tensors,
        InitKvTensors(*env,
                      context_data.value_tensors[local_id]->GetDescriptor(),
                      num_of_cache_layers));
  }
  // Copy kv tensors to new kv tensors.
  int last_copied_layer = -1;
  for (int i = 0; i < num_of_cache_layers; ++i) {
    if ((!local_only || !IsLayerGlobal(i)) &&
        context_data.key_tensors[i] != nullptr) {
      last_copied_layer = i;
    }
  }
  CLEvent* event = nullptr;
  for (int i = 0; i < num_of_cache_layers; ++i) {
    const bool is_global = IsLayerGlobal(i);
    if ((local_only && is_global) || context_data.key_tensors[i] == nullptr) {
      continue;
    }
    if (i == last_copied_layer) {
      event = &last_token_enqueued_event_;
    }
    const size_t layer_cache_size =
        TransformerBuilder::GetCacheSize(text_config_, is_global);
    const size_t layer_copy_size =
        std::min<size_t>(context_data.tokens.size(), layer_cache_size);
    ABSL_RETURN_IF_ERROR(ExecuteCopyCache(
        env, layer_copy_size,
        dynamic_cast<Tensor*>(context_data.key_tensors[i].get()),
        dynamic_cast<Tensor*>(context_data.value_tensors[i].get()),
        layer_copy_size, key_tensors[i].get(), value_tensors[i].get(),
        is_global, event));
  }
  // Store everything in the cloned data.
  ContextData cloned_data;
  cloned_data.key_tensors.resize(num_of_cache_layers);
  cloned_data.value_tensors.resize(num_of_cache_layers);
  for (int i = 0; i < num_of_cache_layers; ++i) {
    cloned_data.key_tensors[i] = std::move(key_tensors[i]);
    cloned_data.value_tensors[i] = std::move(value_tensors[i]);
  }
  cloned_data.lora_id = context_data.lora_id;
  cloned_data.tokens = context_data.tokens;
  return cloned_data;
}

absl::StatusOr<AudioContextData> LlmOpenCl::SaveAudioContext(Environment* env) {
  AudioContextData context_data;
  const int num_layers = audio_params_.cache.key_cached_tensors.size();
  context_data.key_tensors.resize(num_layers);
  context_data.value_tensors.resize(num_layers);

  for (int i = 0; i < num_layers; ++i) {
    auto key_tensor = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env->context(),
                     audio_params_.cache.key_cached_tensors[i]->GetDescriptor(),
                     key_tensor.get()));
    ABSL_RETURN_IF_ERROR(CopyTensor(*audio_params_.cache.key_cached_tensors[i],
                                    *key_tensor, env->queue()));
    context_data.key_tensors[i] = std::move(key_tensor);

    auto value_tensor = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->context(),
        audio_params_.cache.value_cached_tensors[i]->GetDescriptor(),
        value_tensor.get()));
    ABSL_RETURN_IF_ERROR(
        CopyTensor(*audio_params_.cache.value_cached_tensors[i], *value_tensor,
                   env->queue()));
    context_data.value_tensors[i] = std::move(value_tensor);

    if (IsModelv3(audio_params_.config.transformer)) {
      auto depthwise_tensor = std::make_unique<Tensor>();
      ABSL_RETURN_IF_ERROR(CreateTensor(
          env->context(),
          audio_params_.stream_depthwise_tensors[i]->GetDescriptor(),
          depthwise_tensor.get()));
      ABSL_RETURN_IF_ERROR(
          CopyTensor(*audio_params_.stream_depthwise_tensors[i],
                     *depthwise_tensor, env->queue()));
      context_data.stream_depthwise_tensors.push_back(
          std::move(depthwise_tensor));
    }
  }

  if (audio_params_.stream_cumsum) {
    auto cumsum_tensor = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->context(), audio_params_.stream_cumsum->GetDescriptor(),
        cumsum_tensor.get()));
    ABSL_RETURN_IF_ERROR(
        CopyTensor(*audio_params_.stream_cumsum, *cumsum_tensor, env->queue()));
    context_data.stream_cumsum = std::move(cumsum_tensor);
  }

  return context_data;
}

absl::Status LlmOpenCl::RestoreAudioContext(
    Environment* env, const AudioContextData& context_data) {
  for (int i = 0; i < context_data.key_tensors.size(); ++i) {
    auto key_tensor = dynamic_cast<Tensor*>(context_data.key_tensors[i].get());
    if (key_tensor == nullptr) {
      return absl::InternalError("Key tensor is null.");
    }
    auto value_tensor =
        dynamic_cast<Tensor*>(context_data.value_tensors[i].get());
    if (value_tensor == nullptr) {
      return absl::InternalError("Value tensor is null.");
    }
    ABSL_RETURN_IF_ERROR(CopyTensor(
        *key_tensor, *audio_params_.cache.key_cached_tensors[i], env->queue()));
    ABSL_RETURN_IF_ERROR(
        CopyTensor(*value_tensor, *audio_params_.cache.value_cached_tensors[i],
                   env->queue()));

    if (!audio_params_.stream_depthwise_tensors.empty()) {
      auto depthwise_tensor =
          dynamic_cast<Tensor*>(context_data.stream_depthwise_tensors[i].get());
      if (depthwise_tensor == nullptr) {
        return absl::InternalError("Stream depthwise tensor is null.");
      }
      ABSL_RETURN_IF_ERROR(
          CopyTensor(*depthwise_tensor,
                     *audio_params_.stream_depthwise_tensors[i], env->queue()));
    }
  }
  if (audio_params_.stream_cumsum) {
    auto cumsum_tensor =
        dynamic_cast<Tensor*>(context_data.stream_cumsum.get());
    if (cumsum_tensor == nullptr) {
      return absl::InternalError("Stream cumsum tensor is null.");
    }
    ABSL_RETURN_IF_ERROR(
        CopyTensor(*cumsum_tensor, *audio_params_.stream_cumsum, env->queue()));
  }

  return absl::OkStatus();
}

absl::StatusOr<AudioContextData> LlmOpenCl::CloneAudioContext(
    Environment* env, const AudioContextData& context_data) {
  AudioContextData cloned_data;
  const int num_layers = context_data.key_tensors.size();
  cloned_data.key_tensors.resize(num_layers);
  cloned_data.value_tensors.resize(num_layers);

  for (int i = 0; i < num_layers; ++i) {
    auto key_tensor = std::make_unique<Tensor>();
    if (context_data.key_tensors[i] == nullptr) {
      return absl::InternalError("Key tensor is null.");
    }
    auto context_key_tensor =
        dynamic_cast<Tensor*>(context_data.key_tensors[i].get());
    if (context_key_tensor == nullptr) {
      return absl::InternalError("Key tensor is null.");
    }
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->context(), context_data.key_tensors[i]->GetDescriptor(),
        key_tensor.get()));
    ABSL_RETURN_IF_ERROR(
        CopyTensor(*context_key_tensor, *key_tensor, env->queue()));
    cloned_data.key_tensors[i] = std::move(key_tensor);
    auto value_tensor = std::make_unique<Tensor>();
    if (context_data.value_tensors[i] == nullptr) {
      return absl::InternalError("Value tensor is null.");
    }
    auto context_value_tensor =
        dynamic_cast<Tensor*>(context_data.value_tensors[i].get());
    if (context_value_tensor == nullptr) {
      return absl::InternalError("Value tensor is null.");
    }
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->context(), context_data.value_tensors[i]->GetDescriptor(),
        value_tensor.get()));
    ABSL_RETURN_IF_ERROR(
        CopyTensor(*context_value_tensor, *value_tensor, env->queue()));
    cloned_data.value_tensors[i] = std::move(value_tensor);
  }

  if (context_data.stream_cumsum) {
    auto cumsum_tensor = std::make_unique<Tensor>();
    auto context_cumsum_tensor =
        dynamic_cast<Tensor*>(context_data.stream_cumsum.get());
    if (context_cumsum_tensor == nullptr) {
      return absl::InternalError("Stream cumsum tensor is null.");
    }
    ABSL_RETURN_IF_ERROR(CreateTensor(
        env->context(), context_data.stream_cumsum->GetDescriptor(),
        cumsum_tensor.get()));
    ABSL_RETURN_IF_ERROR(
        CopyTensor(*context_cumsum_tensor, *cumsum_tensor, env->queue()));
    cloned_data.stream_cumsum = std::move(cumsum_tensor);
  }

  if (!context_data.stream_depthwise_tensors.empty()) {
    cloned_data.stream_depthwise_tensors.resize(
        context_data.stream_depthwise_tensors.size());
    for (int i = 0; i < cloned_data.stream_depthwise_tensors.size(); ++i) {
      auto depthwise_tensor = std::make_unique<Tensor>();
      auto context_depthwise_tensor =
          dynamic_cast<Tensor*>(context_data.stream_depthwise_tensors[i].get());
      if (context_depthwise_tensor == nullptr) {
        return absl::InternalError("Stream depthwise tensor is null.");
      }
      ABSL_RETURN_IF_ERROR(CreateTensor(
          env->context(),
          context_data.stream_depthwise_tensors[i]->GetDescriptor(),
          depthwise_tensor.get()));
      ABSL_RETURN_IF_ERROR(CopyTensor(*context_depthwise_tensor,
                                      *depthwise_tensor, env->queue()));
      cloned_data.stream_depthwise_tensors[i] = std::move(depthwise_tensor);
    }
  }

  return cloned_data;
}

absl::Status LlmOpenCl::LoadLoRA(
    Environment* env, uint32_t lora_id, int lora_rank,
    std::unique_ptr<LlmTensorLoader> tensor_loader) {
  ABSL_LOG(INFO) << "LlmOpenCl::LoadLoRA id: " << lora_id
                 << ", rank: " << lora_rank;
  auto tensors = model_one_->lora_tensors.find(lora_rank);
  if (tensors == model_one_->lora_tensors.end() || tensors->second.empty()) {
    return absl::InternalError(
        absl::StrCat("Model does not support LoRA with rank: ", lora_rank));
  }

  auto lora_loader = std::make_unique<LoRALoader>();
  tensor_loader->SetCreateTensorFn(
      absl::bind_front(&LlmOpenCl::CreateModelTensor, this, &env->context(),
                       &lora_loader->tensors));
  tensor_loader->SetGpuInfo(env->GetDevicePtr()->GetInfo());
  tensor_loader->SetDataType(tensors->second.begin()->second.weights_desc.type);
  lora_loader->lora_rank = lora_rank;
  lora_loader->loader = std::move(tensor_loader);
  lora_loaders_[lora_id] = std::move(lora_loader);
  return absl::OkStatus();
}

absl::Status LlmOpenCl::UnloadLoRA(Environment* env, uint32_t lora_id) {
  lora_loaders_.erase(lora_id);
  if (current_lora_id_ == lora_id) {
    return UseLoRA(env, std::nullopt);
  }
  return absl::OkStatus();
}

absl::Status LlmOpenCl::UseLoRA(Environment* env,
                                std::optional<uint32_t> lora_id) {
  if (lora_id == current_lora_id_) {
    return absl::OkStatus();
  }

  current_lora_id_ = lora_id;
  if (!lora_id.has_value()) {
    ABSL_LOG(INFO) << "LlmOpenCl LoRA cleared.";
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
      ABSL_RETURN_IF_ERROR(
          model.context->SetTensor(info.id, static_cast<Tensor*>(new_tensor)));
    }
  }
  SetupOptionalNodes();
  ABSL_LOG(INFO) << "LlmOpenCl using LoRA id: " << *lora_id;
  return absl::OkStatus();
}

absl::Status LlmOpenCl::UseLayers(Environment* env,
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

void LlmOpenCl::SetupOptionalNodes() {
  absl::flat_hash_set<int> tags = layer_tags_;
  tags.insert(lora_tags_.begin(), lora_tags_.end());
  if (shared_kv_cache_tag_) {
    tags.insert(*shared_kv_cache_tag_);
  }

  for (auto& kv : models_) {
    kv.second.context->EnableOptionalNodes(tags);
  }
}

absl::Status LlmOpenCl::ExecuteCopyCache(Environment* env, int src_size,
                                         Tensor* src_k, Tensor* src_v,
                                         int dst_size, Tensor* dst_k,
                                         Tensor* dst_v, bool is_global,
                                         CLEvent* event) {
  if (src_k == nullptr || src_v == nullptr) {
    return absl::InternalError(
        "Src k or v tensor is null. Most likely caused by dynamic casting.");
  }
  auto& copy_op = is_global ? copy_cache_global_ : copy_cache_;
  ABSL_RETURN_IF_ERROR(copy_op->SetInt("src_size", src_size));
  ABSL_RETURN_IF_ERROR(copy_op->SetInt("dst_size", dst_size));
  ABSL_RETURN_IF_ERROR(copy_op->SetSrcTensor(0, src_k));
  ABSL_RETURN_IF_ERROR(copy_op->SetSrcTensor(1, src_v));
  ABSL_RETURN_IF_ERROR(copy_op->SetDstTensor(0, dst_k));
  ABSL_RETURN_IF_ERROR(copy_op->SetDstTensor(1, dst_v));
  ABSL_RETURN_IF_ERROR(copy_op->UpdateParams());
  ABSL_RETURN_IF_ERROR(copy_op->AddToQueue(env->queue(), event));
  return absl::OkStatus();
}

bool LlmOpenCl::IsLayerGlobal(int layer_index) const {
  int global_layer_offset = text_config_.num_local_layers_per_global;
  if (text_config_.global_layer_offset.has_value()) {
    global_layer_offset = *text_config_.global_layer_offset;
  }
  return (text_config_.global_head_dimension.has_value() ||
          TransformerBuilder::UseRingBuffer(text_config_)) &&
         ((layer_index % (text_config_.num_local_layers_per_global + 1)) ==
          global_layer_offset);
}

absl::Status LlmOpenCl::ExecuteUpdateIntParams(
    Environment* env, TransformerConfig& config, TransformerParams& params,
    const LlmRuntimeParams& param_vals) {
  const auto params_i32 = GetLlmRuntimeParamsBuffer(param_vals, config);
  for (int i = 0; i < params_i32.size(); ++i) {
    const std::string param_name = "param" + std::to_string(i);
    ABSL_RETURN_IF_ERROR(
        params.write_i32_params->SetInt(param_name, params_i32[i]));
  }
  ABSL_RETURN_IF_ERROR(
      params.write_i32_params->SetDstTensor(0, params.params_i32.get()));
  ABSL_RETURN_IF_ERROR(params.write_i32_params->UpdateParams());
  return params.write_i32_params->AddToQueue(env->queue());
}

absl::Status LlmOpenCl::ExecuteUpdateParams(Environment* env, Tensor* tensor,
                                            const std::vector<float>& params) {
  ABSL_RETURN_IF_ERROR(write_f32_params_->SetDstTensor(0, tensor));
  for (int i = 0; i < params.size(); ++i) {
    const std::string param_name = "param" + std::to_string(i);
    ABSL_RETURN_IF_ERROR(write_f32_params_->SetFloat(param_name, params[i]));
  }
  ABSL_RETURN_IF_ERROR(write_f32_params_->UpdateParams());
  return write_f32_params_->AddToQueue(env->queue());
}

absl::Status LlmOpenCl::ExecutePostProcessing(Environment* env) {
  return ExecutePostProcessing(env, nullptr);
}

absl::Status LlmOpenCl::ExecutePostProcessing(Environment* env,
                                              CLEvent* event) {
  std::uniform_real_distribution<float> distribution =
      std::uniform_real_distribution<float>(0.0f, sampler_params_.p());
  std::vector<float> params(params_f32_->Channels(), 0.0f);

  for (int i = 0; i < params.size() - 1; ++i) {
    params[i] = distribution(*rand_gen_);
  }

  params[params.size() - 1] = 1.0f / sampler_params_.temperature();
  ABSL_RETURN_IF_ERROR(ExecuteUpdateParams(env, params_f32_.get(), params));
  ABSL_RETURN_IF_ERROR(post_processing_->AddToQueue(env->queue(), event));
  return absl::OkStatus();
}

absl::StatusOr<GpuSpatialTensor*> LlmOpenCl::CreateModelTensor(
    CLContext* context, std::vector<std::unique_ptr<Tensor>>* model_tensors,
    const TensorDescriptor& tensor_desc) {
  auto& tensor = model_tensors->emplace_back(new Tensor());
  ABSL_RETURN_IF_ERROR(tensor->CreateFromDescriptor(tensor_desc, *context));
  return tensor.get();
}

absl::StatusOr<GpuSpatialTensor*> LlmOpenCl::CreateModelTensorMmaped(
    CLContext* context, const GpuInfo& gpu_info, LlmTensorLoader* tensor_loader,
    std::vector<std::unique_ptr<Tensor>>* model_tensors,
    const TensorDescriptor& tensor_desc) {
  absl::Span<const uint8_t> data = tensor_desc.GetData();
  auto owned_weights = tensor_loader->GetOwnedData(data.data());
  if (owned_weights) {
    absl::Span<unsigned char> owned_data = owned_weights->GetRawData();
    const size_t offset =
        tensor_desc.GetData().data() - owned_weights->GetRawData().data();
    cl_int error_code = CL_SUCCESS;
    cl_mem cl_buffer = clCreateBuffer(
        context->context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        owned_data.size(), owned_data.data(), &error_code);
    if (error_code != CL_SUCCESS || cl_buffer == nullptr) {
      return absl::InternalError(absl::StrCat(
          "Failed to create buffer: ", cl::CLErrorCodeToString(error_code)));
    }

    mmaped_buffers_.push_back(
        std::make_unique<Buffer>(cl_buffer, owned_data.size()));
    Buffer* buffer_ptr = mmaped_buffers_.back().get();
    if (offset != 0) {
      ABSL_ASSIGN_OR_RETURN(
          Buffer sub_buffer,
          CreateSubBuffer(*buffer_ptr, offset,
                          tensor_desc.GetMemorySizeInBytes(), context));
      mmaped_buffers_.push_back(
          std::make_unique<Buffer>(std::move(sub_buffer)));
      buffer_ptr = mmaped_buffers_.back().get();
    }

    auto& tensor = model_tensors->emplace_back(new Tensor());
    if (tensor_desc.GetStorageType() == TensorStorageType::kTexture2D) {
      const size_t width_pixel_alignment =
          gpu_info.opencl_info.image_pitch_alignment;
      ABSL_RETURN_IF_ERROR(CreateTensorSharedImage2DBuffer(
          *context, buffer_ptr->GetMemoryPtr(), tensor_desc,
          width_pixel_alignment, tensor.get()));
    } else {
      ABSL_RETURN_IF_ERROR(CreateTensorShared(
          *context, buffer_ptr->GetMemoryPtr(), tensor_desc, tensor.get()));
    }
    owned_weights_.push_back(std::move(owned_weights));
    return tensor.get();
  }

  auto& tensor = model_tensors->emplace_back(new Tensor());
  ABSL_RETURN_IF_ERROR(tensor->CreateFromDescriptor(tensor_desc, *context));
  return tensor.get();
}

absl::Status LlmOpenCl::UpdateEmbeddings(Environment* env,
                                         StackedTransformerInference& model,
                                         int previous_token,
                                         absl::Span<const int> tokens) {
  odml::infra::gpu::UpdateNgrammerEmbeddings(
      ngrammer_embeddings_, model.ngrammer_weights, previous_token, tokens,
      text_config_.vocabulary_size,
      text_config_.ngram_vocabulary_size.value_or(0));
  odml::infra::gpu::UpdateEmbeddings(ple_embeddings_, model.ple_weights, tokens,
                                     text_config_.vocabulary_size,
                                     tokens_transform_fn_);

  const auto& gpu_info = env->GetDevicePtr()->GetInfo();
  const bool async = !gpu_info.IsAdreno();

  ABSL_RETURN_IF_ERROR(UploadEmbeddingsToGPU(
      model.ngrammer_weights, model.context.get(), env->queue(), async));
  ABSL_RETURN_IF_ERROR(UploadEmbeddingsToGPU(
      model.ple_weights, model.context.get(), env->queue(), async));
  if (async) {
    ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());
  }
  return absl::OkStatus();
}

absl::Status LlmOpenCl::UpdatePleEmbeddings(Environment* env,
                                            StackedTransformerInference& model,
                                            absl::Span<const int> tokens) {
  odml::infra::gpu::UpdateEmbeddings(
      ple_embeddings_, model.embedding_lookup_model.ple_weights, tokens,
      text_config_.vocabulary_size, tokens_transform_fn_);

  const auto& gpu_info = env->GetDevicePtr()->GetInfo();
  const bool async = !gpu_info.IsAdreno();

  ABSL_RETURN_IF_ERROR(UploadEmbeddingsToGPU(
      model.embedding_lookup_model.ple_weights,
      model.embedding_lookup_model.context.get(), env->queue(), async));
  if (async) {
    ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());
  }
  return absl::OkStatus();
}

// src has sh: [1, 1, x, emb_dim]
// dst has sh: [1, 1, media_cache_size_, emb_dim]
// we perform: dst[0, 0, index:index + x, :] = src
// expect dst to be either media_embeddings_ or vision_ple_tensors_
absl::Status LlmOpenCl::UpdateMultimodalEmbeddings(Tensor* src, Tensor* dst,
                                                   int index,
                                                   Environment* env) {
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

  ABSL_RETURN_IF_ERROR(multimodal_embeddings_op_->SetSrcTensor(0, src));
  ABSL_RETURN_IF_ERROR(multimodal_embeddings_op_->SetDstTensor(0, dst));
  ABSL_RETURN_IF_ERROR(multimodal_embeddings_op_->SetInt("index", index));
  ABSL_RETURN_IF_ERROR(multimodal_embeddings_op_->UpdateParams());
  return multimodal_embeddings_op_->AddToQueue(env->queue());
}

// vision_ple(src) should be a vector of size (text_config_.stack_size)
// and each entry should have shape (1, 1, x, ple_emb_dim)
// vision_ple_tensors_ (dst) should have size
// (text_config_.stack_size - text_config_.extra_stack_size) each entry should
// have shape (1, 1, media_cache_size, ple_emb_dim) This op performs the
// following: for each i in range(text_config_.stack_size -
// text_config_.extra_stack_size):
//   dst[i][0, 0, index:index + x, :] = src[i]
absl::Status LlmOpenCl::FillVisionPleEmbeddings(
    const std::vector<TensorFloat32>& vision_ple, int index, Environment* env) {
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
    auto src = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(CreateTensor(env->context(), desc, src.get()));
    ABSL_RETURN_IF_ERROR(src->UploadDescriptorData(desc, env->queue()));
    ABSL_RETURN_IF_ERROR(UpdateMultimodalEmbeddings(
        src.get(), vision_ple_tensors_[i].get(), index, env));
  }
  return absl::OkStatus();
}

absl::Status LlmOpenCl::FillMediaEmbeddings(const TensorFloat32& media_tensor,
                                            int index, Environment* env) {
  TensorDescriptor desc = media_embeddings_->GetDescriptor();
  desc.SetBHWCShape(media_tensor.shape);
  desc.UploadData(media_tensor);
  auto media_gpu = std::make_unique<Tensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(env->context(), desc, media_gpu.get()));
  ABSL_RETURN_IF_ERROR(media_gpu->UploadDescriptorData(desc, env->queue()));
  return UpdateMultimodalEmbeddings(media_gpu.get(), media_embeddings_.get(),
                                    index, env);
}

// TODO(b/523389424): Clean up UpdateConstraintMask as it is almost identical
// across all 3 of our backends.
absl::Status LlmOpenCl::UpdateConstraintMask(
    const odml::infra::llm_utils::Bitmap& constraint_bitmap, Environment* env) {
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
  return constraint_mask_->UploadDescriptorData(desc, env->queue());
}

}  // namespace cl
}  // namespace ml_drift
