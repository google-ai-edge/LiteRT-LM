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

#include "runtime/executor/gpu_artisan/ml_drift/llm_webgpu.h"

#if WEB_LLM_WEBGPU_DEBUG_MLDRIFT_TENSORS
#include "runtime/executor/gpu_artisan/ml_drift/llm_webgpu_debug.h"
#endif  // WEB_LLM_WEBGPU_DEBUG_MLDRIFT_TENSORS

#include <stdlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <errhandlingapi.h>
#endif

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/bind_front.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift/webgpu/buffer.h"  // from @ml_drift
#include "ml_drift/webgpu/compute_task.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_api_util.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift  // IWYU pragma: export
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/byom/byom_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/executor_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/group_size.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/name_check.h"
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/gpu_optimized_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_file_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/placeholder_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/status_macros.h"
// IWYU pragma: no_include "third_party/dawn/include/webgpu/webgpu.h"

#ifndef __EMSCRIPTEN__
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#endif  // !__EMSCRIPTEN__

namespace ml_drift {
namespace webgpu {
namespace {

using ::odml::infra::gpu::GetRecommendedGroupSizes;
using ::odml::infra::gpu::GpuOptimizedTensorLoader;
using ::odml::infra::gpu::GroupSize;
using ::odml::infra::gpu::LlmFileTensorLoader;
using ::odml::infra::gpu::LlmTensorLoader;
using ::odml::infra::llm_utils::DataHolder;
using ::odml::infra::proto::SamplerParameters;

// A helper class to handle creating encoders/submitting.
class ComputePassHelper {
 public:
  explicit ComputePassHelper(const ExecutionEnvironment* env) : env_(env) {
    command_encoder_ = env_->device().CreateCommandEncoder();
    compute_pass_encoder_ = command_encoder_.BeginComputePass();
  }

  const wgpu::ComputePassEncoder& compute_pass_encoder() const {
    return compute_pass_encoder_;
  }

  wgpu::CommandBuffer FinalizeCommandBuffer() {
    compute_pass_encoder_.End();
    return command_encoder_.Finish();
  }

  void Submit() {
    wgpu::CommandBuffer cb = FinalizeCommandBuffer();
    env_->queue().Submit(1, &cb);
  }

 private:
  const ExecutionEnvironment* env_;
  wgpu::CommandEncoder command_encoder_;
  wgpu::ComputePassEncoder compute_pass_encoder_;
};

#ifndef __EMSCRIPTEN__
class NotifiedTensor : public SpatialTensor {
 public:
  void set_upload_notification(
      std::shared_ptr<absl::Notification> upload_notification) {
    upload_notification_ = std::move(upload_notification);
  }

  std::shared_ptr<absl::Notification> upload_notification() const {
    return upload_notification_;
  }

 private:
  std::shared_ptr<absl::Notification> upload_notification_;
};
#endif  // !__EMSCRIPTEN__

struct LazyCommandBuffer {
  wgpu::CommandBuffer cb;
  std::vector<SpatialTensor*> src_tensors;
};

#ifndef __EMSCRIPTEN__
// Runs commands in small batches at a time, checking to see if the input has
// been canceled.
absl::Status RunPendingCommands(ExecutionEnvironment* env,
                                std::deque<LazyCommandBuffer>& cbs,
                                LlmWebGpu::CancelFn canceled,
                                absl::Duration timeout) {
  absl::Time start = absl::Now();
  std::vector<wgpu::FutureWaitInfo> futures;
  wgpu::Instance instance = env->device().GetAdapter().GetInstance();
  while (!canceled() && !cbs.empty()) {
    if (absl::Now() - start > timeout) {
      return absl::AbortedError(
          "The timeout was reached while processing input.");
    }

    // Wait for a maximum of 100 milliseconds to make sure the cancel bit gets
    // checked and we can return early.
    const uint64_t wait_nanos = absl::Milliseconds(100) / absl::Nanoseconds(1);
    instance.WaitAny(futures.size(), futures.data(), wait_nanos);
    futures.erase(std::remove_if(futures.begin(), futures.end(),
                                 [](const wgpu::FutureWaitInfo& f) {
                                   return f.completed;
                                 }),
                  futures.end());

    // Run 2 commands at a time to make sure we always have something in the GPU
    // queue. A small batch size here allows low latency cancellation.
    while (futures.size() < 2 && !cbs.empty()) {
      const LazyCommandBuffer& cb = cbs.front();
      for (const auto* tensor : cb.src_tensors) {
        auto* notified_tensor = dynamic_cast<const NotifiedTensor*>(tensor);
        if (notified_tensor && notified_tensor->upload_notification()) {
          notified_tensor->upload_notification()->WaitForNotification();
        }
      }
      env->queue().Submit(1, &cb.cb);
      cbs.pop_front();
      wgpu::Future future = env->queue().OnSubmittedWorkDone(
          wgpu::CallbackMode::WaitAnyOnly,
          [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
      futures.push_back(wgpu::FutureWaitInfo{.future = future});
    }
  }
  return absl::OkStatus();
}
#endif  // !__EMSCRIPTEN__

// Runs commands in small batches at a time, checking to see if the all inputs
// have been loaded.
absl::Status ExecuteLazy(const ExecutionEnvironment& environment,
                         InferenceContext& context) {
  std::vector<InferenceContext::CommandBufferInfo> command_buffer_infos;
  ASSIGN_OR_RETURN(
      auto command_buffers,
      context.CreateCommandBuffers(environment, /*num_nodes_per_encoder=*/64,
                                   &command_buffer_infos));
  for (int i = 0; i < command_buffers.size(); ++i) {
#ifndef __EMSCRIPTEN__
    for (auto tensor_id : command_buffer_infos[i].src_tensors) {
      auto tensor = context.GetTensor(tensor_id);
      auto* notified_tensor = dynamic_cast<const NotifiedTensor*>(tensor);
      if (notified_tensor && notified_tensor->upload_notification()) {
        notified_tensor->upload_notification()->WaitForNotification();
      }
    }
#endif  // !__EMSCRIPTEN__
    // Submit every buffer separately to avoid hangs
    environment.queue().Submit(1, &command_buffers[i]);
  }
  return absl::OkStatus();
}

absl::Status InitComputeTask(ExecutionEnvironment* env,
                             std::unique_ptr<GPUOperation> gpu_op,
                             ComputeTask* task) {
  RETURN_IF_ERROR(gpu_op->AssembleCode(env->GetInfo()));
  RETURN_IF_ERROR(task->Init(*env, std::move(gpu_op)));
  RETURN_IF_ERROR(task->Compile(*env, env->GetComputePipelineCache()));
  return absl::OkStatus();
}

absl::Status InitComputeTask(ExecutionEnvironment* env,
                             GPUOperation&& operation, ComputeTask* task) {
  return InitComputeTask(
      env, std::make_unique<GPUOperation>(std::move(operation)), task);
}

absl::Status CopyTensor(const SpatialTensor& src, const SpatialTensor& dst,
                        const wgpu::CommandEncoder& command_encoder) {
  if (wgpu::Buffer handle = src.GetBufferHandle()) {
    command_encoder.CopyBufferToBuffer(
        handle, src.offset(), dst.GetBufferHandle(), dst.offset(),
        std::min(src.GetMemorySizeInBytes(), dst.GetMemorySizeInBytes()));
  } else if (wgpu::Texture handle = src.GetTextureHandle()) {
    wgpu::TexelCopyTextureInfo src_texture{
        .texture = handle,
    };
    wgpu::TexelCopyTextureInfo dst_texture{
        .texture = dst.GetTextureHandle(),
    };
    wgpu::Extent3D extent{
        .width = handle.GetWidth(),
        .height = handle.GetHeight(),
        .depthOrArrayLayers = handle.GetDepthOrArrayLayers(),
    };
    command_encoder.CopyTextureToTexture(&src_texture, &dst_texture, &extent);
  } else {
    return absl::InternalError("Unsupported tensor type.");
  }
  return absl::OkStatus();
}

#ifndef __EMSCRIPTEN__
wgpu::Buffer CreateOwnedDataBuffer(
    ExecutionEnvironment* env,
    std::unique_ptr<DataHolder<uint8_t>> owned_weights, bool is_buffer) {
  absl::Span<unsigned char> owned_data = owned_weights->GetRawData();
  // If an integrated GPU is being used, the host mapped buffer can be
  // directly used instead of copying to individual GPU buffers.
  const bool is_integrated_gpu = env->GetInfo().webgpu_info.is_integrated_gpu;
  wgpu::BufferHostMappedPointer host_mapped_desc;
  host_mapped_desc.pointer = owned_data.data();
  host_mapped_desc.disposeCallback = [](void* userdata) {
    delete reinterpret_cast<DataHolder<uint8_t>*>(userdata);
  };
  host_mapped_desc.userdata = owned_weights.release();
  wgpu::BufferDescriptor buffer_desc = {
      .nextInChain = &host_mapped_desc,
      .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
      .size = owned_data.size(),
  };
  if (is_integrated_gpu && is_buffer) {
    buffer_desc.usage = wgpu::BufferUsage::CopyDst |
                        wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage;
  }

  return env->device().CreateBuffer(&buffer_desc);
}

// Holds in-memory data which is allocated in a way that allows passing to
// CreateBuffer() using the HostMappedPointer feature.
class InMemoryDataHolder : public DataHolder<uint8_t> {
 public:
  explicit InMemoryDataHolder(uint64_t size)
      : size_(AlignByN(size, GpuOptimizedTensorLoader::kAlignment)) {
#if defined(_WIN32)
    data_ = static_cast<uint8_t*>(
        VirtualAlloc(nullptr, size_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    ABSL_CHECK(data_) << "Error calling VirtualAlloc: " << GetLastError();
#else
    data_ = static_cast<uint8_t*>(
        std::aligned_alloc(GpuOptimizedTensorLoader::kAlignment, size_));
#endif
  }

  ~InMemoryDataHolder() override {
#if defined(_WIN32)
    VirtualFree(data_, 0, MEM_RELEASE);
#else
    free(data_);
#endif
  }

  absl::Span<uint8_t> GetData() const override {
    return absl::Span<uint8_t>(static_cast<uint8_t*>(data_), size_);
  }

 private:
  uint8_t* data_;
  uint64_t size_;
};

// Information needed to correctly copy tensor data for WebGPU.
struct TensorCopyInfo {
  uint32_t bytes_per_row;
  uint32_t bytes_per_row_aligned;
  int3 region;

  uint32_t AlignedSize() const {
    return bytes_per_row_aligned * region.y * region.z;
  }

  bool NeedsCopy() const { return bytes_per_row != bytes_per_row_aligned; }
};

TensorCopyInfo GetTensorCopyInfo(const TensorDescriptor& desc) {
  if (desc.GetStorageType() == TensorStorageType::kBuffer) {
    return TensorCopyInfo{
        .bytes_per_row = 1,
        .bytes_per_row_aligned = 1,
        .region = {1, 1, static_cast<int>(desc.GetMemorySizeInBytes())},
    };
  }
  int pixel_size = SizeOf(desc.GetDataType()) * 4;
  int3 region = desc.GetFullTensorRegion();
  const uint32_t bytes_per_row = pixel_size * region.x;
  // from spec: "texelCopyBufferInfo.bytesPerRow must be a multiple of 256."
  const uint32_t bytes_per_row_aligned = AlignByN(bytes_per_row, 256);
  return TensorCopyInfo{
      .bytes_per_row = bytes_per_row,
      .bytes_per_row_aligned = bytes_per_row_aligned,
      .region = region,
  };
}

// Copies tensor data from `src` to `dst` based on the info in `copy_info`.
void CopyTensorData(const TensorCopyInfo& copy_info,
                    absl::Span<const uint8_t> src, absl::Span<uint8_t> dst) {
  if (copy_info.NeedsCopy()) {
    for (int r = 0; r < copy_info.region.y * copy_info.region.z; ++r) {
      memcpy(dst.data() + r * copy_info.bytes_per_row_aligned,
             src.data() + r * copy_info.bytes_per_row, copy_info.bytes_per_row);
    }
  } else {
    memcpy(dst.data(), src.data(), src.size());
  }
}

absl::Status CopyBufferToBuffer(
    ExecutionEnvironment* env, const TensorDescriptor& desc,
    std::unique_ptr<DataHolder<uint8_t>> owned_weights, SpatialTensor* tensor) {
  const size_t offset =
      desc.GetData().data() - owned_weights->GetRawData().data();
  wgpu::Buffer owned_data_buffer =
      CreateOwnedDataBuffer(env, std::move(owned_weights), /*is_buffer=*/true);
  TensorDescriptor desc_no_data;
  desc.CopyWithoutData(&desc_no_data);
  if (owned_data_buffer.GetUsage() & wgpu::BufferUsage::Storage) {
    RETURN_IF_ERROR(
        tensor->CreateFromBuffer(owned_data_buffer, offset, desc_no_data));
  } else {
    RETURN_IF_ERROR(tensor->CreateFromDescriptor(env->device(), desc_no_data));
    wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
    encoder.CopyBufferToBuffer(owned_data_buffer, offset,
                               tensor->GetBufferHandle(), 0,
                               desc.GetData().size());
    wgpu::CommandBuffer cb = encoder.Finish();
    env->queue().Submit(1, &cb);
  }
  return absl::OkStatus();
}
#endif  // !__EMSCRIPTEN__

absl::Status UploadEmbeddingsToGPU(
    const std::vector<odml::infra::gpu::EmbeddingWeights>& weights,
    InferenceContext* context, const wgpu::Queue& queue) {
  for (const auto& weight : weights) {
    auto* tensor = context->GetTensor(weight.weights.id);
    TensorDescriptor desc = tensor->GetDescriptor();
    std::vector<uint8_t> weights_copy = weight.weights_data;
    desc.SetData(std::move(weights_copy));
    RETURN_IF_ERROR(tensor->UploadDescriptorData(queue, desc));

    auto* scale_tensor = context->GetTensor(weight.scales.id);
    TensorDescriptor scale_desc = scale_tensor->GetDescriptor();
    scale_desc.UploadData(weight.scales_data.data());
    RETURN_IF_ERROR(scale_tensor->UploadDescriptorData(queue, scale_desc));
  }
  return absl::OkStatus();
}

// Initializes KV cache tensors for `num_cache_layers`, packed into as few
// shared buffers as possible. If `global_desc` is std::nullopt, only local
// cache layers are initialized (leaving global layer entries as nullptr).
absl::StatusOr<std::vector<std::unique_ptr<SpatialTensor>>> InitKvTensors(
    const ExecutionEnvironment& env, const TensorDescriptor& local_desc,
    std::optional<TensorDescriptor> global_desc,
    int num_local_layers_per_global, int num_cache_layers,
    std::optional<int> global_layer_offset) {
  std::vector<std::unique_ptr<SpatialTensor>> tensors(num_cache_layers);
  // Allocate memory for all cache layers in a single buffer to avoid the
  // overhead of smaller buffers. This saves quite a bit of VRAM. However, if
  // the total size exceeds the device's max buffer size (e.g. for very large
  // context sizes), split the layers across multiple buffers ("chunks"), each
  // no larger than the limit.
  constexpr size_t kAlignment = 256;
  const size_t local_size =
      AlignByN(local_desc.GetMemorySizeInBytes(), kAlignment);
  const size_t global_size =
      global_desc.has_value()
          ? AlignByN(global_desc->GetMemorySizeInBytes(), kAlignment)
          : 0;
  const int pattern_offset = global_layer_offset.has_value()
                                 ? *global_layer_offset
                                 : num_local_layers_per_global;
  auto is_global_layer = [&](int layer_number) {
    return layer_number % (num_local_layers_per_global + 1) == pattern_offset;
  };

  uint64_t max_buffer_size = env.GetInfo().webgpu_info.max_buffer_size;
  if (max_buffer_size == 0) {
    max_buffer_size = std::numeric_limits<uint64_t>::max();
  }

  // Assign each layer to a (chunk, offset) pair, greedily packing layers into
  // chunks in order.
  struct Placement {
    int chunk = -1;
    size_t offset = 0;
  };
  std::vector<Placement> placements(num_cache_layers);
  std::vector<size_t> chunk_sizes;
  for (int i = 0; i < num_cache_layers; ++i) {
    const bool is_global = is_global_layer(i);
    if (is_global && !global_desc.has_value()) {
      continue;
    }
    const size_t layer_size = is_global ? global_size : local_size;
    if (layer_size > max_buffer_size) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "KV cache tensor for layer ", i, " requires ", layer_size,
          " bytes, which exceeds the device's max buffer size of ",
          max_buffer_size, " bytes. Try reducing the context size."));
    }
    if (chunk_sizes.empty() ||
        chunk_sizes.back() + layer_size > max_buffer_size) {
      chunk_sizes.push_back(0);
    }
    placements[i] = {.chunk = static_cast<int>(chunk_sizes.size()) - 1,
                     .offset = chunk_sizes.back()};
    chunk_sizes.back() += layer_size;
  }

  std::vector<wgpu::Buffer> buffers;
  buffers.reserve(chunk_sizes.size());
  for (size_t chunk_size : chunk_sizes) {
    wgpu::BufferDescriptor buffer_desc = {
        .usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst |
                 wgpu::BufferUsage::Storage,
        .size = chunk_size,
    };
    buffers.push_back(env.device().CreateBuffer(&buffer_desc));
  }

  for (int i = 0; i < num_cache_layers; ++i) {
    const Placement& placement = placements[i];
    if (placement.chunk < 0) {
      continue;
    }
    const TensorDescriptor& desc =
        is_global_layer(i) ? *global_desc : local_desc;
    const wgpu::Buffer& buffer = buffers[placement.chunk];
    std::unique_ptr<SpatialTensor>& tensor = tensors[i];
    tensor = std::make_unique<SpatialTensor>();
    // First tensor in each chunk owns that chunk's buffer, the rest share.
    if (placement.offset == 0) {
      RETURN_IF_ERROR(tensor->CreateFromBuffer(buffer, placement.offset, desc));
    } else {
      RETURN_IF_ERROR(
          CreateSharedTensor(buffer, placement.offset, desc, tensor.get()));
    }
  }
  return tensors;
}

absl::StatusOr<std::vector<std::unique_ptr<SpatialTensor>>> InitKvTensors(
    const ExecutionEnvironment& env, const TensorDescriptor& desc,
    int num_cache_layers) {
  return InitKvTensors(env, desc, /*global_desc=*/desc,
                       /*num_local_layers_per_global=*/1, num_cache_layers,
                       /*global_layer_offset=*/std::nullopt);
}

// Returns an upper bound on the size of the attention scores (Q*K^T) tensor for
// a step of `sequence_size` tokens. This tensor spans the full (global) cache
// size, so for long contexts it is typically the largest intermediate tensor in
// the model. `activation_type` is the data type of the model's activations,
// which the attention scores tensor inherits.
uint64_t GetMaxAttentionScoresBytes(const TransformerConfig& config,
                                    int sequence_size, int batch_size,
                                    DataType activation_type) {
  return static_cast<uint64_t>(batch_size) *
         static_cast<uint64_t>(config.number_of_heads) *
         static_cast<uint64_t>(sequence_size) *
         static_cast<uint64_t>(AlignByN(config.cache_size, 4)) *
         static_cast<uint64_t>(SizeOf(activation_type));
}

// Removes group sizes (other than 1) from `group_sizes` whose attention scores
// tensor would exceed the device's max buffer size. A tensor that large can't
// be allocated or bound, so the model for that group size can't run.
absl::Status RemoveGroupSizesExceedingMaxBufferSize(
    const TransformerConfig& config, const GpuInfo& gpu_info,
    CalculationsPrecision precision, int output_batch_size,
    std::vector<GroupSize>& group_sizes) {
  const uint64_t max_buffer_size = gpu_info.GetMaxBufferSize();
  if (max_buffer_size == 0) {
    return absl::OkStatus();
  }
  // Matches the activation data type used by TransformerBuilder.
  const DataType activation_type = DeduceDataTypeFromPrecision(precision);
  std::vector<GroupSize> kept_group_sizes;
  for (const GroupSize& group_size : group_sizes) {
    const int sequence_size = group_size.sequence_size;
    const int batch_size = sequence_size == 1 ? output_batch_size : 1;
    const uint64_t scores_bytes = GetMaxAttentionScoresBytes(
        config, sequence_size, batch_size, activation_type);
    if (scores_bytes <= max_buffer_size) {
      kept_group_sizes.push_back(group_size);
      continue;
    }
    if (sequence_size == 1) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "Attention scores for context size ", config.cache_size, " require ",
          scores_bytes,
          " bytes, which exceeds the device's max buffer size of ",
          max_buffer_size, " bytes. Try reducing the context size."));
    }
    ABSL_LOG(WARNING)
        << "Skipping group size " << sequence_size << " for context size "
        << config.cache_size << ": attention scores would require "
        << scores_bytes
        << " bytes, which exceeds the device's max buffer size of "
        << max_buffer_size << " bytes.";
  }
  group_sizes = std::move(kept_group_sizes);
  return absl::OkStatus();
}

}  // namespace

bool LlmWebGpu::IsLayerGlobal(int layer_number) {
  int global_layer_offset = text_config_.num_local_layers_per_global;
  if (text_config_.global_layer_offset.has_value()) {
    global_layer_offset = *text_config_.global_layer_offset;
  }
  return (text_config_.global_head_dimension.has_value() ||
          TransformerBuilder::UseRingBuffer(text_config_)) &&
         ((layer_number % (text_config_.num_local_layers_per_global + 1)) ==
          global_layer_offset);
}

absl::Status LlmWebGpu::Init(const LlmConfig& config, int outputs_count,
                             const std::string& weights_path,
                             const CreateGpuModelInfo& create_info, int gamma,
                             ExecutionEnvironment* env,
                             bool init_audio_encoder) {
  auto tensor_loader = LlmTensorLoader::MakeCaching(
      std::make_unique<LlmFileTensorLoader>(weights_path));
  return Init(config, outputs_count, std::move(tensor_loader), create_info,
              /*executor_factory=*/nullptr, /*gamma=*/gamma, env,
              init_audio_encoder);
}

absl::Status LlmWebGpu::Init(
    const LlmConfig& config, int outputs_count,
    std::unique_ptr<LlmTensorLoader> tensor_loader,
    const CreateGpuModelInfo& create_info,
    std::shared_ptr<odml::infra::gpu::MlDriftExecutorFactory> executor_factory,
    int gamma, ExecutionEnvironment* env, bool init_audio_encoder) {
  create_info_ = create_info;
  gamma_ = gamma;
  suppressed_tokens_ = config.suppressed_tokens;
  ABSL_CHECK(config.text_decoder.has_value()) << "Text decoder is not set.";
  text_config_ = config.text_decoder.value();  // use below a copy
  enable_decode_logits_ = text_config_.enable_decode_logits;
  executor_factory_ = std::move(executor_factory);
  epsilon_ = GetEpsilon(create_info.precision, env->GetInfo());
  max_top_k_ = text_config_.max_top_k;

  // default sampler params to be greedy.
  sampler_params_ = SamplerParameters();
  sampler_params_.set_type(SamplerParameters::GREEDY);
  sampler_params_.set_k(1);

  tensor_loader->SetCreateTensorFn(
      absl::bind_front(&LlmWebGpu::CreateModelTensor, this, env,
                       tensor_loader.get(), &model_tensors_));
  output_batch_size_ = outputs_count;
  if (group_sizes_.empty()) {
    if (text_config_.sequence_size == 0) {
      group_sizes_ = GetRecommendedGroupSizes(env->GetInfo());
    } else if (text_config_.sequence_size == 1) {
      group_sizes_ = {GroupSize{1, 0.0f}};
    } else {
      group_sizes_ = {
          GroupSize{text_config_.sequence_size, 0.0f},
          GroupSize{1, 1.0f},
      };
    }
  }

  const GpuInfo& gpu_info = env->GetInfo();
  RETURN_IF_ERROR(RemoveGroupSizesExceedingMaxBufferSize(
      text_config_, gpu_info, create_info.precision, output_batch_size_,
      group_sizes_));
  odml::infra::gpu::LlmTensorLoader* llm_builder_tensor_loader =
      tensor_loader.get();
  std::unique_ptr<odml::infra::gpu::PlaceholderTensorLoader> placeholder_loader;
  if (text_config_.use_placeholder_for_layers) {
    placeholder_loader =
        std::make_unique<odml::infra::gpu::PlaceholderTensorLoader>(
            *tensor_loader);
    llm_builder_tensor_loader = placeholder_loader.get();
  }
  ASSIGN_OR_RETURN(std::unique_ptr<TransformerBuilder> text_decoder_builder,
                   odml_byom::CreateTransformerBuilder(
                       text_config_, config.model_type, output_batch_size_,
                       gpu_info, create_info, llm_builder_tensor_loader));

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

  layer_names_ = text_decoder_builder->GetLayerNames();
  if (text_config_.use_placeholder_for_layers &&
      placeholder_loader != nullptr) {
    placeholder_loader->EnablePlaceholderMode();
    placeholder_loader->SetReloadGroupNames(layer_names_);
  }

  const int num_cache_layers =
      text_config_.kv_cache_shared_start_layer.has_value()
          ? *text_config_.kv_cache_shared_start_layer
          : text_config_.stack_size;

  RETURN_IF_ERROR(InitParams(text_decoder_builder.get(), text_params_, env));
  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    RETURN_IF_ERROR(InitCache(text_decoder_builder.get(), text_cache_,
                              text_config_.num_local_layers_per_global,
                              num_cache_layers,
                              text_config_.global_layer_offset, env));
  } else {
    RETURN_IF_ERROR(InitCache(text_decoder_builder.get(), text_cache_,
                              num_cache_layers, env));
  }
  RETURN_IF_ERROR(
      InitSharedTensors(text_decoder_builder.get(), text_config_, env));
  RETURN_IF_ERROR(InitCleanCacheModel(text_decoder_builder.get(), text_cache_,
                                      text_params_, env));

  for (int i = 0; i < group_sizes_.size(); ++i) {
    const int sequence_size = group_sizes_[i].sequence_size;
    const int batch_size = sequence_size == 1 ? output_batch_size_ : 1;
    models_[sequence_size].seq_size = sequence_size;
    RETURN_IF_ERROR(InitModel(text_decoder_builder.get(), batch_size,
                              placeholder_loader.get(), models_[sequence_size],
                              env));
  }
  model_one_ = &models_[1];
  ABSL_CHECK(model_one_->context);

  for (auto& [sequence_size, model] : models_) {
    if (sequence_size == 1 && !text_config_.enable_decode_logits) {
      continue;
    }

    model.copy_nth_op = std::make_unique<ComputeTask>();
    RETURN_IF_ERROR(InitComputeTask(
        env,
        CreateCopyNthToSingleOp(gpu_info, model.output.tensor_desc,
                                model_one_->output.tensor_desc),
        model.copy_nth_op.get()));
    temporary_output_tensor_ = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(CreateTensor(env->device(), model_one_->output.tensor_desc,
                                 temporary_output_tensor_.get()));
  }

  ASSIGN_OR_RETURN(ple_embeddings_,
                   GetDynamicPleEmbeddings(text_config_, tensor_loader.get()));
  ASSIGN_OR_RETURN(
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
    RETURN_IF_ERROR(
        InitFinalEmbeddingToLogitsModel(text_decoder_builder.get(), env));
  } else {
    RETURN_IF_ERROR(InitPostProcessingModel(text_decoder_builder.get(), env));
  }

  if (enable_decode_logits_) {
    int embedding_buffer_size =
        text_config_.cache_size * text_config_.model_dimension * sizeof(float);
    input_embeddings_ = std::make_unique<Buffer>(CreateBuffer(
        env->device(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        embedding_buffer_size));

    int ple_embedding_buffer_size =
        text_config_.cache_size * text_config_.stack_size *
        (*text_config_.per_layer_embedding_dimension) * sizeof(float);
    input_ple_embeddings_ = std::make_unique<Buffer>(CreateBuffer(
        env->device(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        ple_embedding_buffer_size));

    int sequence_position_buffer_size =
        text_config_.cache_size * sizeof(int32_t);
    input_sequence_position_ = std::make_unique<Buffer>(CreateBuffer(
        env->device(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        sequence_position_buffer_size));

    int mask_buffer_size =
        text_config_.cache_size * text_config_.cache_size * sizeof(float);
    input_global_mask_ = std::make_unique<Buffer>(CreateBuffer(
        env->device(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        mask_buffer_size));
    input_local_mask_ = std::make_unique<Buffer>(CreateBuffer(
        env->device(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        mask_buffer_size));
  }

  {
    int buffer_size = text_config_.cache_size * sizeof(int32_t);
    input_token_ids_ = std::make_unique<Buffer>(CreateBuffer(
        env->device(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        buffer_size));
  }

  if (text_config_.use_optional_transformer_layers) {
    tensor_loader_ = std::move(tensor_loader);
  }

  // Initialize any speculative decoding related models/inference context.
  if (gamma_ > 0) {
    // Init the main scoring transformer model and batch post processing models.
    scoring_model_.seq_size = gamma_ + 1;
    if (output_batch_size_ > 1) {
      return absl::UnimplementedError(
          "Speculative decoding with multiple output heads is not supported.");
    }
    RETURN_IF_ERROR(InitModel(text_decoder_builder.get(),
                              /*batch_size=*/output_batch_size_,
                              placeholder_loader.get(), scoring_model_, env));
    RETURN_IF_ERROR(
        InitScoringPostProcessingModel(text_decoder_builder.get(), env));

    scoring_model_.copy_tokens_op = std::make_unique<ComputeTask>();
    RETURN_IF_ERROR(InitComputeTask(
        env, CreateCopyTokensOrPositionsOp(scoring_model_.input.tensor_desc),
        scoring_model_.copy_tokens_op.get()));
    // We need a separate write_scoring_params_f32_ task because the
    // scoring_params_f32_ tensor has different shape than the params_f32_
    // tensor.
    write_scoring_params_f32_ = std::make_unique<ComputeTask>();
    RETURN_IF_ERROR(InitComputeTask(
        env,
        CreateWriteParamsOp(scoring_params_f32_->GetDescriptor(),
                            scoring_params_f32_->Channels()),
        write_scoring_params_f32_.get()));
  }
  if (init_audio_encoder && config.audio_encoder.has_value()) {
    CreateGpuModelInfo create_info_copy = create_info;
    RETURN_IF_ERROR(InitAudioEncoder(config.audio_encoder.value(),
                                     std::move(tensor_loader), create_info_copy,
                                     output_batch_size_, env));
  }
  if (text_config_.kv_cache_shared_start_layer) {
    shared_kv_cache_tag_ = kSharedKvCacheLayersOptionTag;
    SetupOptionalNodes();
  }
  return absl::OkStatus();
}

absl::Status LlmWebGpu::InitFinalEmbeddingToLogitsModel(
    TransformerBuilder* transformer_builder, ExecutionEnvironment* env) {
  ASSIGN_OR_RETURN(GpuModel gpu_model,
                   transformer_builder->CreateFinalEmbeddingToLogitsModel(
                       &final_embedding_, &logits_, /*sequence_size=*/1));

  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  final_embedding_to_logits_ =
      std::make_unique<InferenceContext>(&memory_manager_);
  return final_embedding_to_logits_->InitFromGpuModel(*env, create_info_copy,
                                                      &gpu_model);
}

absl::Status LlmWebGpu::InitEmbeddingLookupModel(
    TransformerBuilder* transformer_builder, ExecutionEnvironment* env,
    StackedTransformerInference& model, const int sequence_size,
    const int batch_size) {
  ASSIGN_OR_RETURN(
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
  embedding_model.context =
      std::make_unique<InferenceContext>(&memory_manager_);
  RETURN_IF_ERROR(embedding_model.context->InitFromGpuModel(
      *env, create_info_copy, &gpu_model));

  embedding_model.copy_tokens_op = std::make_unique<ComputeTask>();
  RETURN_IF_ERROR(InitComputeTask(
      env,
      CreateCopyTokensOrPositionsOp(embedding_model.input_tokens.tensor_desc),
      embedding_model.copy_tokens_op.get()));
  return absl::OkStatus();
}

absl::Status LlmWebGpu::InitAudioEncoder(
    const AudioEncoderConfig& config,
    std::unique_ptr<LlmTensorLoader> tensor_loader,
    CreateGpuModelInfo& create_info, int output_batch_size,
    ExecutionEnvironment* env) {
  // If the model dimension is not 0, meaning the text decoder is loaded within
  // the same webgpu runner, we need to confirm the text decoder's model
  // dimension is the same as the audio encoder's output dimension.
  if (text_config_.model_dimension != 0) {
    RET_CHECK_EQ(text_config_.model_dimension, GetOutputDim(config))
        << absl::StrFormat(
               "Text decoder's model dimension %d is not the same as the "
               "audio encoder's output dimension %d.",
               text_config_.model_dimension, GetOutputDim(config));
  }

  create_info_ = create_info;
  audio_params_.config = config;

  tensor_loader->SetCreateTensorFn(
      absl::bind_front(&LlmWebGpu::CreateModelTensor, this, env,
                       tensor_loader.get(), &model_tensors_));
  const auto& gpu_info = env->GetInfo();
  // By default, we use BUFFER storage type for apple for text decoder. But,
  // for audio encoder TEXTURE_2D is faster, so we override it here.
  create_info.storage_type = TensorStorageType::kTexture2D;
  auto audio_encoder_builder = std::make_unique<AudioEncoderBuilder>(
      config, output_batch_size, gpu_info, create_info, tensor_loader.get());

  int num_cache_layers = audio_params_.config.transformer.stack_size;
  if (audio_encoder_builder->UseGemmaModelV3NPlus() &&
      config.transformer.kv_cache_shared_start_layer.has_value()) {
    // TODO(magicpig): moving this into parameter of transformer instead of
    // hardcoding in config when we are to add audio encoder work.
    num_cache_layers = *config.transformer.kv_cache_shared_start_layer;
  }
  // Init tensors
  audio_params_.feature = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(),
                               audio_encoder_builder->GetFeatureDescriptor(),
                               audio_params_.feature.get()));
  audio_params_.mask = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(),
                               audio_encoder_builder->GetMaskDescriptor(),
                               audio_params_.mask.get()));
  RETURN_IF_ERROR(InitParams(audio_encoder_builder.get(),
                             audio_params_.transformer_params, env));
  // v1 and v2 have normal kv cache
  if (!IsModelv3(audio_params_.config.transformer)) {
    RETURN_IF_ERROR(InitCache(audio_encoder_builder.get(), audio_params_.cache,
                              num_cache_layers, env));
  }
  // v3 only has kv cache for streaming, and different tensor descriptors
  if (IsModelv3(audio_params_.config.transformer) &&
      audio_encoder_builder->IsStreaming()) {
    audio_params_.cache.key_cached_tensors =
        std::vector<std::unique_ptr<SpatialTensor>>(num_cache_layers);
    audio_params_.cache.value_cached_tensors =
        std::vector<std::unique_ptr<SpatialTensor>>(num_cache_layers);
    for (int i = 0; i < num_cache_layers; ++i) {
      audio_params_.cache.key_cached_tensors[i] =
          std::make_unique<SpatialTensor>();
      audio_params_.cache.value_cached_tensors[i] =
          std::make_unique<SpatialTensor>();
      RETURN_IF_ERROR(CreateTensor(
          env->device(),
          audio_encoder_builder->GetKCacheConformerTensorDescriptor(),
          audio_params_.cache.key_cached_tensors[i].get()));
      RETURN_IF_ERROR(CreateTensor(
          env->device(),
          audio_encoder_builder->GetVCacheConformerTensorDescriptor(),
          audio_params_.cache.value_cached_tensors[i].get()));
    }
  }
  if (audio_encoder_builder->IsStreaming()) {
    audio_params_.stream_cumsum =
        std::make_unique<SpatialTensor>(SpatialTensor());
    RETURN_IF_ERROR(CreateTensor(
        env->device(), audio_encoder_builder->GetCumsumStreamDescriptor(),
        audio_params_.stream_cumsum.get()));
    if (IsModelv3(audio_params_.config.transformer)) {
      audio_params_.stream_depthwise_tensors =
          std::vector<std::unique_ptr<SpatialTensor>>(num_cache_layers);
      for (int i = 0; i < num_cache_layers; ++i) {
        audio_params_.stream_depthwise_tensors[i] =
            std::make_unique<SpatialTensor>();
        RETURN_IF_ERROR(
            CreateTensor(env->device(),
                         audio_encoder_builder->GetDepthwiseStreamDescriptor(),
                         audio_params_.stream_depthwise_tensors[i].get()));
      }
    }
  }
  if (IsModelv3(audio_params_.config.transformer)) {
    audio_params_.signal = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(CreateTensor(env->device(),
                                 audio_encoder_builder->GetSignalDescriptor(),
                                 audio_params_.signal.get()));
  }
  RETURN_IF_ERROR(InitAudioEncoderModel(audio_encoder_builder.get(), env));
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
      ASSIGN_OR_RETURN(
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
      RETURN_IF_ERROR(audio_params_.cache.clean_cache->InitFromGpuModel(
          *env, create_info_copy, &gpu_model));
    }
  } else {  // v1 or v2
    RETURN_IF_ERROR(InitCleanCacheModel(audio_encoder_builder.get(),
                                        audio_params_.cache,
                                        audio_params_.transformer_params, env));
  }
  if (IsModelv3(audio_params_.config.transformer) &&
      audio_encoder_builder->IsStreaming()) {
    // Need to clean audio cache b/c the cache tensors are not
    // guaranteed to be zeroed out.
    RETURN_IF_ERROR(CleanAudioCache(env, 0));
  }
  return absl::OkStatus();
}

absl::Status LlmWebGpu::InitParams(TransformerBuilder* transformer_builder,
                                   TransformerParams& params,
                                   ExecutionEnvironment* env) {
  params.params_i32 = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(),
                               transformer_builder->GetParamsTensorDescriptor(),
                               params.params_i32.get()));
  params.write_i32_params = std::make_unique<ComputeTask>();
  RETURN_IF_ERROR(
      InitComputeTask(env,
                      CreateWriteParamsOp(params.params_i32->GetDescriptor(),
                                          params.params_i32->Channels()),
                      params.write_i32_params.get()));
  return absl::OkStatus();
}

absl::Status LlmWebGpu::InitCache(TransformerBuilder* transformer_builder,
                                  TransformerCache& cache, int num_cache_layers,
                                  ExecutionEnvironment* env) {
  ASSIGN_OR_RETURN(
      cache.key_cached_tensors,
      InitKvTensors(*env, transformer_builder->GetKCacheTensorDescriptor(),
                    num_cache_layers));
  ASSIGN_OR_RETURN(
      cache.value_cached_tensors,
      InitKvTensors(*env, transformer_builder->GetVCacheTensorDescriptor(),
                    num_cache_layers));
  return absl::OkStatus();
}

absl::Status LlmWebGpu::InitCache(TransformerBuilder* transformer_builder,
                                  TransformerCache& cache,
                                  int num_local_layers_per_global,
                                  int num_cache_layers,
                                  std::optional<int> global_layer_offset,
                                  ExecutionEnvironment* env) {
  ASSIGN_OR_RETURN(
      cache.key_cached_tensors,
      InitKvTensors(
          *env, transformer_builder->GetKCacheTensorDescriptor(),
          transformer_builder->GetKCacheTensorDescriptor(/*is_global*/ true),
          num_local_layers_per_global, num_cache_layers, global_layer_offset));
  ASSIGN_OR_RETURN(
      cache.value_cached_tensors,
      InitKvTensors(
          *env, transformer_builder->GetVCacheTensorDescriptor(),
          transformer_builder->GetVCacheTensorDescriptor(/*is_global*/ true),
          num_local_layers_per_global, num_cache_layers, global_layer_offset));
  return absl::OkStatus();
}

absl::Status LlmWebGpu::InitSharedTensors(
    TransformerBuilder* transformer_builder, const TransformerConfig& config,
    ExecutionEnvironment* env) {
  copy_cache_ = std::make_unique<ComputeTask>();
  RETURN_IF_ERROR(InitComputeTask(
      env,
      CreateCopyCacheOp(text_cache_.key_cached_tensors[0]->GetDescriptor(),
                        text_cache_.value_cached_tensors[0]->GetDescriptor(),
                        config, output_batch_size_),
      copy_cache_.get()));

  // Separate copy op for global layers when necessary
  if (config.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(config)) {
    const int global_id =
        config.global_layer_offset.value_or(config.num_local_layers_per_global);
    if (global_id < text_cache_.key_cached_tensors.size()) {
      copy_cache_global_ = std::make_unique<ComputeTask>();
      RETURN_IF_ERROR(InitComputeTask(
          env,
          CreateCopyCacheOp(
              text_cache_.key_cached_tensors[global_id]->GetDescriptor(),
              text_cache_.value_cached_tensors[global_id]->GetDescriptor(),
              config, output_batch_size_, /*is_global=*/true),
          copy_cache_global_.get()));
    }
  }

  tokens_ids_ = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(),
                               transformer_builder->GetTokensTensorDescriptor(),
                               tokens_ids_.get()));
  // Init multimodal tensors
  if (media_cache_size_ > 0) {  // is multimodal
    if (config.per_layer_embedding_dimension.has_value()) {
      if (transformer_builder->UseGoogleInternalModelV2()) {
        vision_ple_tensors_.reserve(config.stack_size -
                                    config.extra_stack_size);
        for (int i = 0; i < config.stack_size - config.extra_stack_size; ++i) {
          vision_ple_tensors_.push_back(std::make_unique<SpatialTensor>());
          RETURN_IF_ERROR(
              CreateTensor(env->device(),
                           transformer_builder->GetVisionPleTensorDescriptor(),
                           vision_ple_tensors_[i].get()));
        }
      }
    }
    media_embeddings_ = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(
        CreateTensor(env->device(),
                     transformer_builder->GetMediaEmbeddingsTensorDescriptor(),
                     media_embeddings_.get()));

    multimodal_embeddings_op_ = std::make_unique<ComputeTask>();
    RETURN_IF_ERROR(InitComputeTask(
        env,
        CreateCopyMediaEmbeddingsOp(media_embeddings_->GetDescriptor(),
                                    media_embeddings_->GetDescriptor()),
        multimodal_embeddings_op_.get()));
  }

  return absl::OkStatus();
}

absl::Status LlmWebGpu::InitModel(
    TransformerBuilder* transformer_builder, int batch_size,
    odml::infra::gpu::PlaceholderTensorLoader* placeholder_loader,
    StackedTransformerInference& model_n, ExecutionEnvironment* env) {
  model_n.copy_tokens_op = std::make_unique<ComputeTask>();
  RETURN_IF_ERROR(
      InitComputeTask(env,
                      CreateCopyTokensOrPositionsOp(
                          transformer_builder->GetInputTensorDescriptor(
                              batch_size, model_n.seq_size)),
                      model_n.copy_tokens_op.get()));
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle media_handle;
  std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> vision_ple_tensor_handles;
  GpuModel gpu_model;
  if (transformer_builder->UseGoogleInternalModelV2()) {
    ASSIGN_OR_RETURN(
        gpu_model,
        transformer_builder->CreateStackedTransformerModelV2(
            model_n.seq_size, batch_size, &model_n.input,
            media_cache_size_ > 0 ? &media_handle : nullptr, &tokens_ids_handle,
            &params_i32_handle, &key_cached_tensor_handles,
            &value_cached_tensor_handles, &model_n.output, &model_n.ple_weights,
            &model_n.ngrammer_weights,
            media_cache_size_ > 0 ? &vision_ple_tensor_handles : nullptr,
            /*segment_pos_tensor=*/nullptr,
            /*global_mask_tensor=*/nullptr,
            /*local_mask_tensor=*/nullptr));
  } else if (transformer_builder->UseGemmaModelV3NPlus()) {
    if (text_config_.enable_decode_logits) {
      // Only support text models with external embeddings.
      RET_CHECK(text_config_.enable_external_embeddings)
          << "external_embeddings_handle must enabled when"
          << "enable_decode_logits is true.";
      ASSIGN_OR_RETURN(
          gpu_model,
          transformer_builder->CreateStackedTransformerModelV3(
              model_n.seq_size, batch_size,
              /*input_tensor=*/nullptr,
              /*media_tensor=*/nullptr,
              /*tokens_ids_tensor=*/nullptr, &params_i32_handle,
              &key_cached_tensor_handles, &value_cached_tensor_handles,
              &model_n.output,
              /*ple_weights=*/nullptr,
              /*vision_ple_tensors=*/nullptr, &model_n.positions,
              &model_n.global_mask, &model_n.local_mask,
              &model_n.external_embeddings, &model_n.external_ple_embeddings,
              /*input_activations_handle=*/nullptr,
              /*output_activations_handle=*/nullptr, kGemmaBaseName));
    } else {
      ASSIGN_OR_RETURN(
          gpu_model,
          transformer_builder->CreateStackedTransformerModelV3(
              model_n.seq_size, batch_size, &model_n.input,
              media_cache_size_ > 0 ? &media_handle : nullptr,
              &tokens_ids_handle, &params_i32_handle,
              &key_cached_tensor_handles, &value_cached_tensor_handles,
              &model_n.output, &model_n.ple_weights,
              /*vision_ple_tensors=*/nullptr,
              /*segment_pos_tensor=*/nullptr,
              /*global_mask_tensor=*/nullptr,
              /*local_mask_tensor=*/nullptr,
              /*external_embeddings_tensor=*/nullptr,
              /*external_ple_embeddings_tensor=*/nullptr,
              /*input_activations_handle=*/nullptr,
              /*output_activations_handle=*/nullptr, kGemmaBaseName));
    }
    model_n.debug_tensors = transformer_builder->GetDebugTensors();
  } else {
    ASSIGN_OR_RETURN(gpu_model,
                     transformer_builder->CreateStackedTransformerModel(
                         model_n.seq_size, batch_size, &model_n.input,
                         media_cache_size_ > 0 ? &media_handle : nullptr,
                         &params_i32_handle, &key_cached_tensor_handles,
                         &value_cached_tensor_handles, &model_n.output));
  }

  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  transformer_builder->ChangePrecisionIfBfloat16Fix(
      CalculationsPrecision::kF32);

  model_n.lora_tensors = transformer_builder->lora_tensors();
  const auto external_immutable_tensors =
      create_info_copy.external_immutable_tensors;
  create_info_copy.external_immutable_tensors.clear();
  for (const auto& [id, tensor] : external_immutable_tensors) {
    if (auto placeholder_tensor =
            dynamic_cast<odml::infra::gpu::PlaceholderTensor*>(tensor)) {
      placeholder_tensor->SetId(id);
      create_info_copy.external_mutable_tensors[id] = tensor->GetDescriptor();
    } else {
      create_info_copy.external_immutable_tensors[id] = tensor;
    }
  }

  if (text_config_.enable_decode_logits) {
    if (!transformer_builder->UseGemmaModelV3NPlus()) {
      return absl::InvalidArgumentError(
          "enable_decode_logits is only supported for V3 models.");
    }
    model_n.copy_external_embeddings_op = std::make_unique<ComputeTask>();
    RETURN_IF_ERROR(InitComputeTask(
        env,
        CreateCopyExternalEmbeddingsOp(model_n.external_embeddings.tensor_desc),
        model_n.copy_external_embeddings_op.get()));

    model_n.copy_positions_op = std::make_unique<ComputeTask>();
    RETURN_IF_ERROR(InitComputeTask(
        env, CreateCopyTokensOrPositionsOp(model_n.positions.tensor_desc),
        model_n.copy_positions_op.get()));
    model_n.copy_masks_op = std::make_unique<ComputeTask>();
    RETURN_IF_ERROR(
        InitComputeTask(env,
                        CreateCopyMasksOp(model_n.local_mask.tensor_desc,
                                          model_n.global_mask.tensor_desc,
                                          text_config_.cache_size),
                        model_n.copy_masks_op.get()));
    RETURN_IF_ERROR(InitEmbeddingLookupModel(transformer_builder, env, model_n,
                                             model_n.seq_size, batch_size));
  }

  {  // Providing binding info for external shared tensors
    if (transformer_builder->UseGoogleInternalModelV2() ||
        transformer_builder->UseGemmaModelV3NPlus()) {
      create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
          tokens_ids_.get();
    }
    create_info_copy.external_immutable_tensors[params_i32_handle.id] =
        text_params_.params_i32.get();
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

  model_n.context = std::make_unique<InferenceContext>(&memory_manager_);
  RETURN_IF_ERROR(
      model_n.context->InitFromGpuModel(*env, create_info_copy, &gpu_model));

  if (placeholder_loader) {
    for (const auto& [group_name, group_load_fns] :
         placeholder_loader->GetTensorReloadFns()) {
      std::vector<odml::infra::gpu::LoadTensorFn>& dst =
          model_n.tensor_reload_fns[group_name];
      for (auto& [name, tensor_load_fn] : group_load_fns) {
        dst.insert(dst.end(), tensor_load_fn);
      }
    }
    placeholder_loader->GetTensorReloadFns().clear();
  }

  return absl::OkStatus();
}

absl::Status LlmWebGpu::InitPostProcessingModel(
    TransformerBuilder* transformer_builder, ExecutionEnvironment* env) {
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle stacked_transformer_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle params_f32_handle;
  GpuModelBuilder::TensorHandle topk_tokens_handle;
  GpuModelBuilder::TensorHandle topk_probabilities_handle;
  GpuModelBuilder::TensorHandle pre_softmax_handle;
  GpuModelBuilder::TensorHandle constraint_mask_handle;
  ASSIGN_OR_RETURN(GpuModel gpu_model,
                   transformer_builder->CreatePostProcessingModel(
                       &input_handle, &params_i32_handle, &params_f32_handle,
                       &tokens_ids_handle, &stacked_transformer_input_handle,
                       &pre_softmax_handle, &topk_tokens_handle,
                       &topk_probabilities_handle, &constraint_mask_handle,
                       /*sequence_size=*/1));
  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  create_info_copy.external_immutable_tensors[input_handle.id] =
      model_one_->context->GetTensor(model_one_->output.id);
  params_f32_ = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(), params_f32_handle.tensor_desc,
                               params_f32_.get()));
  write_f32_params_ = std::make_unique<ComputeTask>();
  RETURN_IF_ERROR(
      InitComputeTask(env,
                      CreateWriteParamsOp(params_f32_->GetDescriptor(),
                                          params_f32_->Channels()),
                      write_f32_params_.get()));

  create_info_copy.external_immutable_tensors[params_i32_handle.id] =
      text_params_.params_i32.get();
  create_info_copy.external_immutable_tensors[params_f32_handle.id] =
      params_f32_.get();
  create_info_copy
      .external_immutable_tensors[stacked_transformer_input_handle.id] =
      model_one_->context->GetTensor(model_one_->input.id);
  create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
      tokens_ids_.get();
  constraint_mask_ = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(),
                               constraint_mask_handle.tensor_desc,
                               constraint_mask_.get()));
  RETURN_IF_ERROR(
      UpdateConstraintMask(odml::infra::llm_utils::AllAllowedBitmap(), env));
  create_info_copy.external_immutable_tensors[constraint_mask_handle.id] =
      constraint_mask_.get();

  topk_tokens_ = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(), topk_tokens_handle.tensor_desc,
                               topk_tokens_.get()));
  create_info_copy.external_immutable_tensors[topk_tokens_handle.id] =
      topk_tokens_.get();
  topk_probabilities_ = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(),
                               topk_probabilities_handle.tensor_desc,
                               topk_probabilities_.get()));
  create_info_copy.external_immutable_tensors[topk_probabilities_handle.id] =
      topk_probabilities_.get();

  post_processing_ = std::make_unique<InferenceContext>();
  RETURN_IF_ERROR(
      post_processing_->InitFromGpuModel(*env, create_info_copy, &gpu_model));

  // Create softmax model.
  ASSIGN_OR_RETURN(GpuModel softmax_model,
                   transformer_builder->CreateSoftmaxModel(
                       &pre_softmax_handle, &token_probabilities_));

  create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  create_info_copy.external_immutable_tensors[pre_softmax_handle.id] =
      post_processing_->GetTensor(pre_softmax_handle.id);

  softmax_ = std::make_unique<InferenceContext>();
  return softmax_->InitFromGpuModel(*env, create_info_copy, &softmax_model);
}

absl::Status LlmWebGpu::InitScoringPostProcessingModel(
    TransformerBuilder* transformer_builder, ExecutionEnvironment* env) {
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle stacked_transformer_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle params_f32_handle;
  GpuModelBuilder::TensorHandle topk_tokens_handle;
  GpuModelBuilder::TensorHandle topk_probabilities_handle;
  ASSIGN_OR_RETURN(GpuModel gpu_model,
                   transformer_builder->CreatePostProcessingModel(
                       &input_handle, &params_i32_handle, &params_f32_handle,
                       &tokens_ids_handle, &stacked_transformer_input_handle,
                       /*pre_softmax_handle=*/nullptr, &topk_tokens_handle,
                       &topk_probabilities_handle,
                       /*constraint_mask_handle=*/nullptr,
                       /*sequence_size=*/gamma_ + 1));
  CreateGpuModelInfo create_info_copy = GetCreateGpuModelInfo(
      create_info_, transformer_builder->GetExternalImmutableTensors(),
      transformer_builder->GetExternalMutableTensors());
  create_info_copy.external_immutable_tensors[input_handle.id] =
      scoring_model_.context->GetTensor(scoring_model_.output.id);

  scoring_params_f32_ = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(), params_f32_handle.tensor_desc,
                               scoring_params_f32_.get()));

  create_info_copy.external_immutable_tensors[params_i32_handle.id] =
      text_params_.params_i32.get();
  // We need a separate scoring_params_f32_ because sampling multiple tokens
  // requires multiple probabilities and thus a different shape for f32_params_.
  create_info_copy.external_immutable_tensors[params_f32_handle.id] =
      scoring_params_f32_.get();

  // While we do not directly use the stacked_transformer_input_handle, it
  // dictates the shape of the Sampling operation, and thus we must set it to
  // the corresponding tensor with correct shape.
  create_info_copy
      .external_immutable_tensors[stacked_transformer_input_handle.id] =
      scoring_model_.context->GetTensor(scoring_model_.input.id);
  create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
      tokens_ids_.get();

  scoring_topk_tokens_ = std::make_unique<SpatialTensor>();
  scoring_topk_probabilities_ = std::make_unique<SpatialTensor>();
  RETURN_IF_ERROR(CreateTensor(env->device(), topk_tokens_handle.tensor_desc,
                               scoring_topk_tokens_.get()));
  RETURN_IF_ERROR(CreateTensor(env->device(),
                               topk_probabilities_handle.tensor_desc,
                               scoring_topk_probabilities_.get()));
  create_info_copy.external_immutable_tensors[topk_tokens_handle.id] =
      scoring_topk_tokens_.get();
  create_info_copy.external_immutable_tensors[topk_probabilities_handle.id] =
      scoring_topk_probabilities_.get();
  scoring_post_processing_ = std::make_unique<InferenceContext>();
  return scoring_post_processing_->InitFromGpuModel(*env, create_info_copy,
                                                    &gpu_model);
}

absl::Status LlmWebGpu::InitCleanCacheModel(
    TransformerBuilder* transformer_builder, TransformerCache& cache,
    TransformerParams& params, ExecutionEnvironment* env) {
  std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
  GpuModelBuilder::TensorHandle params_i32_handle;
  ASSIGN_OR_RETURN(GpuModel gpu_model,
                   transformer_builder->CreateCleanCacheModel(
                       &key_cached_tensor_handles, &value_cached_tensor_handles,
                       &params_i32_handle));
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

  cache.clean_cache = std::make_unique<InferenceContext>();
  return cache.clean_cache->InitFromGpuModel(*env, create_info_copy,
                                             &gpu_model);
}

absl::Status LlmWebGpu::InitAudioEncoderModel(
    AudioEncoderBuilder* audio_encoder_builder, ExecutionEnvironment* env) {
  TensorHandle audio_feature_handle, audio_mask_handle, params_i32_handle,
      audio_stream_cumsum_handle, signal_handle;
  std::vector<GpuModelBuilder::TensorHandle> key_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> value_cached_tensor_handles;
  std::vector<GpuModelBuilder::TensorHandle> stream_depthwise_tensor_handles;

  ASSIGN_OR_RETURN(
      GpuModel gpu_model,
      audio_encoder_builder->CreateAudioEncoderModel(
          &audio_feature_handle, &audio_mask_handle, &params_i32_handle,
          &key_cached_tensor_handles, &value_cached_tensor_handles,
          &audio_stream_cumsum_handle, &stream_depthwise_tensor_handles,
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
          .external_immutable_tensors[audio_stream_cumsum_handle.id] =
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
    RETURN_IF_ERROR(
        audio_params_.signal->UploadDescriptorData(env->queue(), signal_desc));
  }
  audio_params_.context =
      std::make_unique<InferenceContext>(InferenceContext());
  RETURN_IF_ERROR(audio_params_.context->InitFromGpuModel(
      *env, create_info_copy, &gpu_model));
  return absl::OkStatus();
}

#ifdef __EMSCRIPTEN__
absl::Status LlmWebGpu::InitInputTokens(ExecutionEnvironment* env,
                                        const std::vector<int>& input_tokens,
                                        int time_step) {
#else
absl::Status LlmWebGpu::InitInputTokens(ExecutionEnvironment* env,
                                        const std::vector<int>& input_tokens,
                                        int time_step, CancelFn canceled) {
#endif  // __EMSCRIPTEN__
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
  RETURN_IF_ERROR(CleanCache(env, time_step));
  if (input_tokens_.size() < time_step &&
      !model_one_->ngrammer_weights.empty()) {
    return absl::InternalError("time_step is too large.");
  }
  input_tokens_.resize(time_step);
#ifdef __EMSCRIPTEN__
  RETURN_IF_ERROR(InitInputTokensWithModelN(env, input_tokens));
#else
  RETURN_IF_ERROR(InitInputTokensWithModelN(env, input_tokens, canceled));
  if (canceled()) {
    return absl::OkStatus();
  }
#endif  // __EMSCRIPTEN__
  input_tokens_.insert(input_tokens_.end(), input_tokens.begin(),
                       input_tokens.end());
  {
    int token_index = input_tokens_.size() - 1;
    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = token_index;
    llm_params.active_tokens = token_index + 1;
    llm_params.topk = sampler_params_.k();
    RETURN_IF_ERROR(
        ExecuteUpdateIntParams(env, text_config_, text_params_, llm_params));
  }
  RETURN_IF_ERROR(ExecutePostProcessing(env));
  if (gamma_ != 0) {
    // Need to pull top k probabilities out because speculative decoding is
    // enabled, and speculative decoding requires the top K probabilities to
    // operate.
    RETURN_IF_ERROR(PullTopKProbabilities(env, input_tokens_.size()));
  }
  ready_tokens_ = 1;
  num_output_tokens_ = 0;
  needs_init_input_tokens_ = false;
  return absl::OkStatus();
}

#ifdef __EMSCRIPTEN__
absl::Status LlmWebGpu::InitInputTokensWithModelN(
    ExecutionEnvironment* env, const std::vector<int>& input_tokens) {
#else
absl::Status LlmWebGpu::InitInputTokensWithModelN(
    ExecutionEnvironment* env, const std::vector<int>& input_tokens,
    CancelFn canceled) {
#endif  // __EMSCRIPTEN__
  if (input_tokens.empty()) {
    return absl::OkStatus();
  }
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
  ASSIGN_OR_RETURN(std::vector<int> input_groups,
                   GetGroupSizesForInput(input_size, group_sizes_));
  if (has_optional_input_layers) {
    input_groups.push_back(1);
  }
  if (input_groups.empty()) {
    return absl::OkStatus();
  }
  WriteDataToBuffer(env->queue(), input_token_ids_->GetMemoryHandle(),
                    input_tokens.size() * sizeof(int), input_tokens.data());
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
    std::deque<LazyCommandBuffer> pending_commands;
    StackedTransformerInference& model_n = models_[group_size];
    int previous_token = 0;
    if (token_index_offset > 0) {
      previous_token = input_tokens[token_index_offset - 1];
    } else if (!input_tokens_.empty()) {
      previous_token = input_tokens_.back();
    }
    RETURN_IF_ERROR(UpdateEmbeddings(
        env, model_n, previous_token,
        absl::MakeSpan(input_tokens).subspan(token_index_offset, group_size)));

    ComputePassHelper helper(env);
    RETURN_IF_ERROR(WriteInputTokensForModel(*env, model_n, input_tokens,
                                             token_index_offset,
                                             &helper.compute_pass_encoder()));

    const int active_positions =
        std::min(token_index_offset + model_n.seq_size,
                 static_cast<int>(input_tokens.size()));
    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = input_tokens_.size() + token_index_offset;
    llm_params.active_tokens = input_tokens_.size() + active_positions;
    llm_params.topk = sampler_params_.k();
    RETURN_IF_ERROR(EncodeUpdateParams(env, text_config_, text_params_,
                                       llm_params,
                                       helper.compute_pass_encoder()));

    pending_commands.push_back({helper.FinalizeCommandBuffer(), {}});

    // Commands are batched according to group size since a larger group size
    // will mean each command on average takes longer. This value is chosen to
    // give 1k group size a 16 batch size, and smaller group sizes will get
    // proportionally larger command batch sizes.
    const int command_batch_size = (16 * 1024) / group_size;
    std::vector<InferenceContext::CommandBufferInfo> command_buffer_infos;
    ASSIGN_OR_RETURN(std::vector<wgpu::CommandBuffer> cbs,
                     model_n.context->CreateCommandBuffers(
                         *env, command_batch_size, &command_buffer_infos));
    for (int cb_index = 0; cb_index < cbs.size(); ++cb_index) {
      LazyCommandBuffer lazy_cb;
      lazy_cb.cb = cbs[cb_index];
      const auto& cb_info = command_buffer_infos[cb_index];
      lazy_cb.src_tensors.reserve(cb_info.src_tensors.size());
      for (const auto& tensor_id : cb_info.src_tensors) {
        lazy_cb.src_tensors.push_back(model_n.context->GetTensor(tensor_id));
      }
      pending_commands.push_back(lazy_cb);
    }
    token_index_offset += group_size;

#ifdef __EMSCRIPTEN__
    // Run pending commands without cancellation logic.
    for (auto& cb : pending_commands) {
      env->queue().Submit(1, &cb.cb);
    }
#else
    // Timeout is 30 seconds + group_size / 5. If we're processing fewer than 5
    // tokens per second, something is probably wrong.
    RETURN_IF_ERROR(RunPendingCommands(env, pending_commands, canceled,
                                       absl::Seconds(30 + group_size / 5)));
    if (canceled()) {
      return absl::OkStatus();
    }
#endif  // __EMSCRIPTEN__

#if WEB_LLM_WEBGPU_DEBUG_MLDRIFT_TENSORS
    // Internal-only debugging for web.
    RETURN_IF_ERROR(llm_debug::MaybeDumpPrefillTensors(
        *env, *model_n.context, model_n.debug_tensors,
        /*batch_index=*/i, /*num_batches=*/input_groups.size()));
#endif  // WEB_LLM_WEBGPU_DEBUG_MLDRIFT_TENSORS
  }

  // Copy model_n last token from output to model_one output.
  StackedTransformerInference& last_model = models_[input_groups.back()];
  if (last_model.seq_size != 1) {
    ComputeTask& copy_nth_op = *last_model.copy_nth_op;
    RETURN_IF_ERROR(copy_nth_op.SetSrcTensor(
        0, last_model.context->GetTensor(last_model.output.id)));
    RETURN_IF_ERROR(
        copy_nth_op.SetDstTensor(0, temporary_output_tensor_.get()));
    RETURN_IF_ERROR(copy_nth_op.SetInt(
        "nth", (input_tokens.size() - 1) % last_model.seq_size));
    RETURN_IF_ERROR(copy_nth_op.Update(env->device()));
    RETURN_IF_ERROR(copy_nth_op.Execute(*env));

    // Copy to a temp tensor first since the output tensors of the last model
    // and model_one_ may be the same, which causes an error.
    wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
    RETURN_IF_ERROR(CopyTensor(
        *temporary_output_tensor_,
        *model_one_->context->GetTensor(model_one_->output.id), encoder));
    wgpu::CommandBuffer cb = encoder.Finish();
    env->queue().Submit(1, &cb);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> LlmWebGpu::GetNextTokenProbabilities(
    ExecutionEnvironment* env) {
  RETURN_IF_ERROR(ExecuteLazy(*env, *softmax_.get()));
  TensorDescriptor desc;
  RETURN_IF_ERROR(softmax_->GetTensor(token_probabilities_.id)
                      ->ToDescriptor(env->device(), &desc));

  TensorFloat32 output;
  desc.DownloadData(&output);
  return output.data;
}

absl::Status LlmWebGpu::GetNextTokens(ExecutionEnvironment* env,
                                      std::vector<int>* output_tokens) {
  std::vector<std::vector<int>> temp(1);
  temp[0].resize(output_tokens->size());
  RETURN_IF_ERROR(GetNextTokens(env, &temp, /*update_constraint_fn=*/{}));
  *output_tokens = std::move(temp[0]);
  return absl::OkStatus();
}

absl::Status LlmWebGpu::ProcessNextOutputToken(ExecutionEnvironment* env,
                                               bool pull_top_k_probabilities) {
  int time_step = input_tokens_.size() + ready_tokens_ - 1;
  LlmRuntimeParams llm_params;
  llm_params.token_index_offset = time_step;
  llm_params.active_tokens = time_step + 1;
  llm_params.topk = sampler_params_.k();
  RETURN_IF_ERROR(
      ExecuteUpdateIntParams(env, text_config_, text_params_, llm_params));
  RETURN_IF_ERROR(ExecuteLazy(*env, *model_one_->context.get()));
  RETURN_IF_ERROR(ExecutePostProcessing(env));
  ready_tokens_++;

  if (pull_top_k_probabilities) {
    RETURN_IF_ERROR(PullTopKProbabilities(env, time_step + 1));
  }

  return absl::OkStatus();
}

absl::Status LlmWebGpu::GetNextTokens(
    ExecutionEnvironment* env, std::vector<std::vector<int>>* output_tokens,
    UpdateConstraintFn update_constraint_fn) {
  return GetNextTokensWithPullTopKProbabilities(
      env, output_tokens, false, std::move(update_constraint_fn));
}

absl::Status LlmWebGpu::GetNextTokensWithPullTopKProbabilities(
    ExecutionEnvironment* env, std::vector<std::vector<int>>* output_tokens,
    bool pull_top_k_probabilities, UpdateConstraintFn update_constraint_fn) {
  RET_CHECK(!pull_top_k_probabilities || output_tokens->size() == 1)
      << "Getting the Top K probabilities with multiple output head is not "
         "supported.";

  const bool has_embeddings =
      !model_one_->ple_weights.empty() || !model_one_->ngrammer_weights.empty();

  if (output_tokens->size() > 1 && has_embeddings) {
    // TODO: b/355494120 - Support multi output with dynamic PLE.
    return absl::InternalError(
        "Multi-output is not supported with dynamic embedding loading.");
  }

  if (needs_init_input_tokens_) {
    return absl::InternalError("Must call InitInputTokens() first.");
  }
  const int next_tokens_count = (*output_tokens)[0].size();

  // Only add as many tokens as are needed for this round of output.
  while (next_tokens_count + num_output_tokens_ > ready_tokens_) {
    // Dynamic PLE loading requires getting the token value and uploading the
    // next weights to the GPU.
    // TODO: b/355494120 - Avoid a sync each token if possible.
    if (update_constraint_fn || has_embeddings) {
      ASSIGN_OR_RETURN(TensorInt32 previous_tokens, DownloadTokenIds(env));
      const int last_token = input_tokens_.size() + ready_tokens_ - 1;
      if (update_constraint_fn) {
        ASSIGN_OR_RETURN(auto bitmap, update_constraint_fn(
                                          previous_tokens.data[last_token]));
        RETURN_IF_ERROR(UpdateConstraintMask(*bitmap, env));
      }
      if (has_embeddings) {
        RETURN_IF_ERROR(UpdateEmbeddings(env, *model_one_,
                                         previous_tokens.data[last_token - 1],
                                         {previous_tokens.data[last_token]}));
      }
    }

    RETURN_IF_ERROR(ProcessNextOutputToken(env, pull_top_k_probabilities));
  }

  ASSIGN_OR_RETURN(TensorInt32 output, DownloadTokenIds(env));
  for (int b = 0; b < output_batch_size_; ++b) {
    std::vector<int>& tokens = output_tokens->at(b);
    for (int i = 0; i < tokens.size(); ++i) {
      const int time_step = i + input_tokens_.size() + num_output_tokens_;
      if (token_override_.contains(time_step)) {
        tokens[i] = token_override_[time_step];
      } else {
        int index = time_step * output_batch_size_ + b;
        if (index >= output.data.size()) {
          return absl::InternalError("Too many tokens requested.");
        }
        tokens[i] = output.data[index];
      }
    }
  }

  if (!model_one_->ngrammer_weights.empty() &&
      absl::MakeSpan(output.data).first(input_tokens_.size()) !=
          input_tokens_) {
    return absl::InternalError("Input tokens are out of sync with GPU.");
  }
  num_output_tokens_ += next_tokens_count;
  if (update_constraint_fn || has_embeddings) {
    const int last_token = input_tokens_.size() + num_output_tokens_ - 1;
    if (update_constraint_fn) {
      ASSIGN_OR_RETURN(auto bitmap,
                       update_constraint_fn(output.data[last_token]));
      RETURN_IF_ERROR(UpdateConstraintMask(*bitmap, env));
    }
    if (has_embeddings) {
      RETURN_IF_ERROR(UpdateEmbeddings(env, *model_one_,
                                       output.data[last_token - 1],
                                       {output.data[last_token]}));
    }
  }

  // Get at least one token processing for the next round.
  RETURN_IF_ERROR(ProcessNextOutputToken(env, pull_top_k_probabilities));

  // A single output stream can be treated as input tokens. This allows models
  // with dynamic embeddings to continue processing off of previous output.
  if (output_tokens->size() == 1) {
    const std::vector<int>& output = output_tokens->at(0);
    input_tokens_.insert(input_tokens_.end(), output.begin(), output.end());
    num_output_tokens_ -= output.size();
    ready_tokens_ -= output.size();
  }
  return absl::OkStatus();
}

absl::Status LlmWebGpu::PullTopKProbabilities(ExecutionEnvironment* env,
                                              int time_step) {
  TensorDescriptor topk_prob_desc;
  RETURN_IF_ERROR(
      topk_probabilities_->ToDescriptor(env->device(), &topk_prob_desc));
  TensorFloat32 topk_prob_out;
  topk_prob_desc.DownloadData(&topk_prob_out);

  TensorDescriptor topk_token_desc;
  RETURN_IF_ERROR(topk_tokens_->ToDescriptor(env->device(), &topk_token_desc));
  TensorInt32 topk_token_out;
  topk_token_desc.DownloadData(&topk_token_out);

  // Note that the TopK GPU Operation operates on 4 items at a time, so top K
  // not multiples of 4 will result in extra elements we can skip.
  for (int i = 0; i < topk_token_out.data.size() && i < sampler_params_.k();
       i++) {
    ASSIGN_OR_RETURN(speculative_utils::TokenProbability token_probability,
                     speculative_utils::TokenProbability::Create(
                         topk_token_out.data[i], topk_prob_out.data[i]));
    probability_cache_[time_step].push_back(token_probability);
  }

  return absl::OkStatus();
}

absl::StatusOr<GpuSpatialTensor*> LlmWebGpu::ExecuteAudioEncoder(
    const TensorFloat32& audio_tensor, const TensorFloat32& mask_tensor,
    int time_step, ExecutionEnvironment* env) {
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
  RETURN_IF_ERROR(ExecuteUpdateIntParams(env, audio_params_.config.transformer,
                                         audio_params_.transformer_params,
                                         llm_params));

  // Set audio encoder inputs.
  TensorDescriptor feature_desc = audio_params_.feature->GetDescriptor();
  feature_desc.UploadData(audio_tensor);
  RETURN_IF_ERROR(
      audio_params_.feature->UploadDescriptorData(env->queue(), feature_desc));
  TensorDescriptor mask_desc = audio_params_.mask->GetDescriptor();
  mask_desc.UploadData(mask_tensor);
  RETURN_IF_ERROR(
      audio_params_.mask->UploadDescriptorData(env->queue(), mask_desc));

  // Run audio encoder
  RETURN_IF_ERROR(ExecuteLazy(*env, *audio_params_.context.get()));
  return audio_params_.context->GetTensor(audio_params_.output_handle.id);
}

absl::Status LlmWebGpu::FillVisionEmbeddings(
    const TensorFloat32& vision_embedding_tensor,
    const std::vector<TensorFloat32>& vision_ple_tensors, int image_index,
    ExecutionEnvironment* env) {
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
  wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
  std::vector<std::unique_ptr<SpatialTensor>> temp_tensors;
  temp_tensors.reserve(1 + vision_ple_tensors.size());
  RETURN_IF_ERROR(FillMediaEmbeddings(vision_embedding_tensor, index, env,
                                      encoder, temp_tensors));
  if (!vision_ple_tensors.empty()) {
    RETURN_IF_ERROR(FillVisionPleEmbeddings(vision_ple_tensors, index, env,
                                            encoder, temp_tensors));
  }
  wgpu::CommandBuffer cb = encoder.Finish();
  env->queue().Submit(1, &cb);
  return absl::OkStatus();
}

absl::Status LlmWebGpu::FillAudioEmbeddings(SpatialTensor& audio_tensor,
                                            int index,
                                            ExecutionEnvironment* env) {
  if (media_cache_size_ == 0) {
    return absl::InvalidArgumentError("Media embedding cache was not set.");
  }
  wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
  RETURN_IF_ERROR(UpdateMultimodalEmbeddings(
      &audio_tensor, media_embeddings_.get(), index, env, encoder));
  wgpu::CommandBuffer cb = encoder.Finish();
  env->queue().Submit(1, &cb);
  return absl::OkStatus();
}

absl::Status LlmWebGpu::ClearVisionPle(ExecutionEnvironment* env) {
  std::vector<TensorFloat32> vision_ple_tensors;
  vision_ple_tensors.reserve(vision_ple_tensors_.size());
  for (int i = 0; i < vision_ple_tensors_.size(); ++i) {
    const BHWC shape = vision_ple_tensors_[i]->GetDescriptor().GetBHWCShape();
    TensorFloat32 tensor;
    tensor.shape = shape;
    tensor.data.resize(shape.DimensionsProduct());
    vision_ple_tensors.push_back(std::move(tensor));
  }
  wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
  std::vector<std::unique_ptr<SpatialTensor>> temp_tensors;
  temp_tensors.reserve(vision_ple_tensors.size());
  RETURN_IF_ERROR(FillVisionPleEmbeddings(vision_ple_tensors, 0, env, encoder,
                                          temp_tensors));
  wgpu::CommandBuffer cb = encoder.Finish();
  env->queue().Submit(1, &cb);
  return absl::OkStatus();
}

absl::Status LlmWebGpu::CleanCache(ExecutionEnvironment* env, int time_step,
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
      return absl::InvalidArgumentError(
          absl::StrFormat("Cannot rewind to time_step %d from %d. Ringbuffer "
                          "size is %d with sliding window of %d",
                          time_step, *llm_params.active_tokens, local_cache_mem,
                          *text_config_.sliding_window_size));
    }
  }
  RETURN_IF_ERROR(
      ExecuteUpdateIntParams(env, text_config_, text_params_, llm_params));
  RETURN_IF_ERROR(ExecuteLazy(*env, *text_cache_.clean_cache.get()));
  if (time_step < input_tokens_.size()) {
    input_tokens_.resize(time_step);
  }
  return absl::OkStatus();
}

absl::Status LlmWebGpu::CleanAudioCache(ExecutionEnvironment* env,
                                        int time_step) {
  if (audio_params_.config.transformer.relative_positional_embedding_config
          .has_value() &&
      audio_params_.config.feature.pad_for_streaming == 0) {
    return absl::InvalidArgumentError("No KV cache for v3 non-streaming.");
  }
  LlmRuntimeParams llm_params;
  llm_params.token_index_offset = time_step;
  RETURN_IF_ERROR(ExecuteUpdateIntParams(env, audio_params_.config.transformer,
                                         audio_params_.transformer_params,
                                         llm_params));
  RETURN_IF_ERROR(ExecuteLazy(*env, *audio_params_.cache.clean_cache.get()));
  return absl::OkStatus();
}

absl::StatusOr<ContextData> LlmWebGpu::SaveContext(ExecutionEnvironment* env,
                                                   bool local_only) {
  ContextData context_data;
  context_data.tokens = input_tokens_;
  context_data.lora_id = current_lora_id_;

  std::vector<std::unique_ptr<SpatialTensor>> key_tensors;
  std::vector<std::unique_ptr<SpatialTensor>> value_tensors;
  // NOTE: This assumes we don't start off an architecture with two global
  // layers. That has always been the case so far, but if it changes, our logic
  // here will need to change too.
  const int local_id = IsLayerGlobal(0) ? 1 : 0;
  DataType data_type =
      text_cache_.key_cached_tensors[local_id]->GetDescriptor().GetDataType();
  const size_t local_cache_size =
      TransformerBuilder::GetCacheSize(text_config_, /*is_global=*/false);
  const size_t local_alloc_size = std::max<size_t>(
      1, std::min<size_t>(input_tokens_.size(), local_cache_size));

  auto k_desc = TransformerBuilder::GetKCacheTensorDescriptor(
      text_config_, output_batch_size_, local_alloc_size, data_type);
  auto v_desc = TransformerBuilder::GetVCacheTensorDescriptor(
      text_config_, output_batch_size_, local_alloc_size, data_type);

  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    std::optional<TensorDescriptor> k_desc_global;
    std::optional<TensorDescriptor> v_desc_global;
    if (!local_only) {
      const size_t global_cache_size =
          TransformerBuilder::GetCacheSize(text_config_, /*is_global=*/true);
      const size_t global_alloc_size = std::max<size_t>(
          1, std::min<size_t>(input_tokens_.size(), global_cache_size));
      k_desc_global = TransformerBuilder::GetKCacheTensorDescriptor(
          text_config_, output_batch_size_, global_alloc_size, data_type,
          /*is_global=*/true);
      v_desc_global = TransformerBuilder::GetVCacheTensorDescriptor(
          text_config_, output_batch_size_, global_alloc_size, data_type,
          /*is_global=*/true);
    }
    ASSIGN_OR_RETURN(key_tensors,
                     InitKvTensors(*env, k_desc, k_desc_global,
                                   text_config_.num_local_layers_per_global,
                                   text_cache_.key_cached_tensors.size(),
                                   text_config_.global_layer_offset));
    ASSIGN_OR_RETURN(value_tensors,
                     InitKvTensors(*env, v_desc, v_desc_global,
                                   text_config_.num_local_layers_per_global,
                                   text_cache_.value_cached_tensors.size(),
                                   text_config_.global_layer_offset));
  } else {
    ASSIGN_OR_RETURN(
        key_tensors,
        InitKvTensors(*env, k_desc, text_cache_.key_cached_tensors.size()));
    ASSIGN_OR_RETURN(
        value_tensors,
        InitKvTensors(*env, v_desc, text_cache_.value_cached_tensors.size()));
  }

  ComputePassHelper helper(env);
  for (int i = 0; i < key_tensors.size(); ++i) {
    bool is_global = IsLayerGlobal(i);
    if (local_only && is_global) {
      continue;
    }
    const size_t layer_cache_size =
        TransformerBuilder::GetCacheSize(text_config_, is_global);
    const size_t layer_copy_size =
        std::min<size_t>(input_tokens_.size(), layer_cache_size);
    RETURN_IF_ERROR(EncodeCopyCache(
        env, layer_cache_size, text_cache_.key_cached_tensors[i].get(),
        text_cache_.value_cached_tensors[i].get(), layer_copy_size,
        key_tensors[i].get(), value_tensors[i].get(),
        helper.compute_pass_encoder(), is_global));
  }
  helper.Submit();

  context_data.key_tensors.resize(key_tensors.size());
  for (int i = 0; i < key_tensors.size(); ++i) {
    context_data.key_tensors[i] = std::move(key_tensors[i]);
  }
  context_data.value_tensors.resize(value_tensors.size());
  for (int i = 0; i < value_tensors.size(); ++i) {
    context_data.value_tensors[i] = std::move(value_tensors[i]);
  }
  return context_data;
}

absl::Status LlmWebGpu::RestoreContext(ExecutionEnvironment* env,
                                       const ContextData& context_data,
                                       bool local_only) {
  if (context_data.lora_id != current_lora_id_) {
    return absl::InternalError("LoRA ID for context does not match.");
  }

  ComputePassHelper helper(env);
  for (int i = 0; i < context_data.key_tensors.size(); ++i) {
    bool is_global = IsLayerGlobal(i);
    if (local_only && is_global) {
      continue;
    }
    const size_t layer_cache_size =
        TransformerBuilder::GetCacheSize(text_config_, is_global);
    const size_t layer_copy_size =
        std::min<size_t>(context_data.tokens.size(), layer_cache_size);
    RETURN_IF_ERROR(EncodeCopyCache(
        env, layer_copy_size,
        dynamic_cast<SpatialTensor*>(context_data.key_tensors[i].get()),
        dynamic_cast<SpatialTensor*>(context_data.value_tensors[i].get()),
        layer_cache_size, text_cache_.key_cached_tensors[i].get(),
        text_cache_.value_cached_tensors[i].get(),
        helper.compute_pass_encoder(), is_global));
  }
  helper.Submit();

  if (local_only) {
    return absl::OkStatus();
  }

  input_tokens_ = context_data.tokens;
  ready_tokens_ = 0;
  num_output_tokens_ = 0;
  needs_init_input_tokens_ = true;

  TensorDescriptor desc = tokens_ids_->GetDescriptor();
  std::vector<int> tokens_data = context_data.tokens;
  tokens_data.resize(desc.GetBHWDCShape().DimensionsProduct());
  desc.UploadData(tokens_data.data());
  return tokens_ids_->UploadDescriptorData(env->queue(), desc);
}

absl::StatusOr<ContextData> LlmWebGpu::CloneContext(
    ExecutionEnvironment* env, const ContextData& context_data,
    bool local_only) {
  ContextData cloned_data;
  cloned_data.tokens = context_data.tokens;
  cloned_data.lora_id = context_data.lora_id;

  const int local_id = IsLayerGlobal(0) ? 1 : 0;
  std::vector<std::unique_ptr<SpatialTensor>> key_tensors;
  std::vector<std::unique_ptr<SpatialTensor>> value_tensors;
  if (text_config_.global_head_dimension.has_value() ||
      TransformerBuilder::UseRingBuffer(text_config_)) {
    const int global_id = text_config_.global_layer_offset.value_or(
        text_config_.num_local_layers_per_global);
    std::optional<TensorDescriptor> k_desc_global;
    std::optional<TensorDescriptor> v_desc_global;
    if (!local_only && global_id < context_data.key_tensors.size() &&
        context_data.key_tensors[global_id] != nullptr) {
      k_desc_global = context_data.key_tensors[global_id]->GetDescriptor();
      v_desc_global = context_data.value_tensors[global_id]->GetDescriptor();
    }
    ASSIGN_OR_RETURN(
        key_tensors,
        InitKvTensors(*env, context_data.key_tensors[local_id]->GetDescriptor(),
                      k_desc_global, text_config_.num_local_layers_per_global,
                      context_data.key_tensors.size(),
                      text_config_.global_layer_offset));
    ASSIGN_OR_RETURN(
        value_tensors,
        InitKvTensors(*env,
                      context_data.value_tensors[local_id]->GetDescriptor(),
                      v_desc_global, text_config_.num_local_layers_per_global,
                      context_data.value_tensors.size(),
                      text_config_.global_layer_offset));
  } else {
    ASSIGN_OR_RETURN(
        key_tensors,
        InitKvTensors(*env, context_data.key_tensors[local_id]->GetDescriptor(),
                      context_data.key_tensors.size()));
    ASSIGN_OR_RETURN(
        value_tensors,
        InitKvTensors(*env,
                      context_data.value_tensors[local_id]->GetDescriptor(),
                      context_data.value_tensors.size()));
  }

  ComputePassHelper helper(env);
  for (int i = 0; i < context_data.key_tensors.size(); ++i) {
    bool is_global = IsLayerGlobal(i);
    if ((local_only && is_global) || context_data.key_tensors[i] == nullptr) {
      continue;
    }
    const size_t layer_cache_size =
        TransformerBuilder::GetCacheSize(text_config_, is_global);
    const size_t layer_copy_size =
        std::min<size_t>(context_data.tokens.size(), layer_cache_size);
    RETURN_IF_ERROR(EncodeCopyCache(
        env, layer_copy_size,
        dynamic_cast<SpatialTensor*>(context_data.key_tensors[i].get()),
        dynamic_cast<SpatialTensor*>(context_data.value_tensors[i].get()),
        layer_copy_size, key_tensors[i].get(), value_tensors[i].get(),
        helper.compute_pass_encoder(), is_global));
  }
  helper.Submit();

  cloned_data.key_tensors.resize(key_tensors.size());
  for (int i = 0; i < key_tensors.size(); ++i) {
    cloned_data.key_tensors[i] = std::move(key_tensors[i]);
  }
  cloned_data.value_tensors.resize(value_tensors.size());
  for (int i = 0; i < value_tensors.size(); ++i) {
    cloned_data.value_tensors[i] = std::move(value_tensors[i]);
  }
  return cloned_data;
}

absl::StatusOr<AudioContextData> LlmWebGpu::SaveAudioContext(
    ExecutionEnvironment* env) {
  AudioContextData context_data;
  const int num_layers = audio_params_.cache.key_cached_tensors.size();
  context_data.key_tensors.resize(num_layers);
  context_data.value_tensors.resize(num_layers);

  wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();

  for (int i = 0; i < num_layers; ++i) {
    auto key_tensor = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(key_tensor->CreateFromDescriptor(
        env->device(),
        audio_params_.cache.key_cached_tensors[i]->GetDescriptor()));
    RETURN_IF_ERROR(CopyTensor(*audio_params_.cache.key_cached_tensors[i],
                               *key_tensor, encoder));
    context_data.key_tensors[i] = std::move(key_tensor);

    auto value_tensor = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(value_tensor->CreateFromDescriptor(
        env->device(),
        audio_params_.cache.value_cached_tensors[i]->GetDescriptor()));
    RETURN_IF_ERROR(CopyTensor(*audio_params_.cache.value_cached_tensors[i],
                               *value_tensor, encoder));
    context_data.value_tensors[i] = std::move(value_tensor);

    if (IsModelv3(audio_params_.config.transformer)) {
      auto depthwise_tensor = std::make_unique<SpatialTensor>();
      RETURN_IF_ERROR(depthwise_tensor->CreateFromDescriptor(
          env->device(),
          audio_params_.stream_depthwise_tensors[i]->GetDescriptor()));
      RETURN_IF_ERROR(CopyTensor(*audio_params_.stream_depthwise_tensors[i],
                                 *depthwise_tensor, encoder));
      context_data.stream_depthwise_tensors.push_back(
          std::move(depthwise_tensor));
    }
  }

  if (audio_params_.stream_cumsum) {
    auto cumsum_tensor = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(cumsum_tensor->CreateFromDescriptor(
        env->device(), audio_params_.stream_cumsum->GetDescriptor()));
    RETURN_IF_ERROR(
        CopyTensor(*audio_params_.stream_cumsum, *cumsum_tensor, encoder));
    context_data.stream_cumsum = std::move(cumsum_tensor);
  }

  wgpu::CommandBuffer command_buffer = encoder.Finish();
  env->queue().Submit(1, &command_buffer);

  return context_data;
}

absl::Status LlmWebGpu::RestoreAudioContext(
    ExecutionEnvironment* env, const AudioContextData& context_data) {
  wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
  for (int i = 0; i < context_data.key_tensors.size(); ++i) {
    RETURN_IF_ERROR(CopyTensor(
        *static_cast<SpatialTensor*>(context_data.key_tensors[i].get()),
        *audio_params_.cache.key_cached_tensors[i], encoder));
    RETURN_IF_ERROR(CopyTensor(
        *static_cast<SpatialTensor*>(context_data.value_tensors[i].get()),
        *audio_params_.cache.value_cached_tensors[i], encoder));

    if (!audio_params_.stream_depthwise_tensors.empty()) {
      RETURN_IF_ERROR(
          CopyTensor(*static_cast<SpatialTensor*>(
                         context_data.stream_depthwise_tensors[i].get()),
                     *audio_params_.stream_depthwise_tensors[i], encoder));
    }
  }
  if (audio_params_.stream_cumsum) {
    RETURN_IF_ERROR(CopyTensor(
        *static_cast<SpatialTensor*>(context_data.stream_cumsum.get()),
        *audio_params_.stream_cumsum, encoder));
  }

  wgpu::CommandBuffer command_buffer = encoder.Finish();
  env->queue().Submit(1, &command_buffer);
  return absl::OkStatus();
}

absl::StatusOr<AudioContextData> LlmWebGpu::CloneAudioContext(
    ExecutionEnvironment* env, const AudioContextData& context_data) {
  AudioContextData cloned_data;
  const int num_layers = context_data.key_tensors.size();
  cloned_data.key_tensors.resize(num_layers);
  cloned_data.value_tensors.resize(num_layers);

  wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();

  for (int i = 0; i < num_layers; ++i) {
    auto key_tensor = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(key_tensor->CreateFromDescriptor(
        env->device(), context_data.key_tensors[i]->GetDescriptor()));
    RETURN_IF_ERROR(CopyTensor(
        *static_cast<SpatialTensor*>(context_data.key_tensors[i].get()),
        *key_tensor, encoder));
    cloned_data.key_tensors[i] = std::move(key_tensor);

    auto value_tensor = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(value_tensor->CreateFromDescriptor(
        env->device(), context_data.value_tensors[i]->GetDescriptor()));
    RETURN_IF_ERROR(CopyTensor(
        *static_cast<SpatialTensor*>(context_data.value_tensors[i].get()),
        *value_tensor, encoder));
    cloned_data.value_tensors[i] = std::move(value_tensor);
  }

  if (context_data.stream_cumsum) {
    auto cumsum_tensor = std::make_unique<SpatialTensor>();
    RETURN_IF_ERROR(cumsum_tensor->CreateFromDescriptor(
        env->device(), context_data.stream_cumsum->GetDescriptor()));
    RETURN_IF_ERROR(CopyTensor(
        *static_cast<SpatialTensor*>(context_data.stream_cumsum.get()),
        *cumsum_tensor, encoder));
    cloned_data.stream_cumsum = std::move(cumsum_tensor);
  }

  if (!context_data.stream_depthwise_tensors.empty()) {
    cloned_data.stream_depthwise_tensors.resize(
        context_data.stream_depthwise_tensors.size());
    for (int i = 0; i < cloned_data.stream_depthwise_tensors.size(); ++i) {
      auto depthwise_tensor = std::make_unique<SpatialTensor>();
      RETURN_IF_ERROR(depthwise_tensor->CreateFromDescriptor(
          env->device(),
          context_data.stream_depthwise_tensors[i]->GetDescriptor()));
      RETURN_IF_ERROR(
          CopyTensor(*static_cast<SpatialTensor*>(
                         context_data.stream_depthwise_tensors[i].get()),
                     *depthwise_tensor, encoder));
      cloned_data.stream_depthwise_tensors[i] = std::move(depthwise_tensor);
    }
  }

  wgpu::CommandBuffer command_buffer = encoder.Finish();
  env->queue().Submit(1, &command_buffer);

  return cloned_data;
}

absl::Status LlmWebGpu::LoadLoRA(
    ExecutionEnvironment* env, uint32_t lora_id, int lora_rank,
    std::unique_ptr<LlmTensorLoader> tensor_loader) {
  auto tensors = model_one_->lora_tensors.find(lora_rank);
  if (tensors == model_one_->lora_tensors.end() || tensors->second.empty()) {
    return absl::InternalError(
        absl::StrCat("Model does not support LoRA with rank: ", lora_rank));
  }

  auto lora_loader = std::make_unique<LoRALoader>();
  tensor_loader->SetCreateTensorFn(
      absl::bind_front(&LlmWebGpu::CreateModelTensor, this, env,
                       tensor_loader.get(), &lora_loader->tensors));
  tensor_loader->SetGpuInfo(env->GetInfo());
  tensor_loader->SetDataType(tensors->second.begin()->second.weights_desc.type);
  lora_loader->lora_rank = lora_rank;
  lora_loader->loader = std::move(tensor_loader);
  lora_loaders_[lora_id] = std::move(lora_loader);
  return absl::OkStatus();
}

absl::Status LlmWebGpu::UnloadLoRA(ExecutionEnvironment* env,
                                   uint32_t lora_id) {
  lora_loaders_.erase(lora_id);
  if (current_lora_id_ == lora_id) {
    return UseLoRA(env, std::nullopt);
  }
  return absl::OkStatus();
}

absl::Status LlmWebGpu::UseLoRA(ExecutionEnvironment* env,
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
  odml::infra::gpu::LlmTensorLoader& loader = *loader_it->second->loader;
  for (auto& [seq_size, model] : models_) {
    for (const auto& [name, info] : model.lora_tensors[lora_rank]) {
      if (!loader.HasTensor(name)) {
        continue;
      }
      if (odml::infra::gpu::IsAttentionTensorName(name)) {
        lora_tags_.insert(lora_rank | kLoRAAttentionOptionTag);
      } else {
        lora_tags_.insert(lora_rank | kLoRAFeedForwardOptionTag);
      }
      ASSIGN_OR_RETURN(ml_drift::GpuSpatialTensor * new_tensor,
                       loader.LoadWeights(name, info.weights_desc, info.shape,
                                          info.swap_dims));
      RETURN_IF_ERROR(model.context->SetTensor(
          info.id, static_cast<SpatialTensor*>(new_tensor)));
    }
    RETURN_IF_ERROR(model.context->UpdateMutableObjectsBindings(*env));
  }
  SetupOptionalNodes();
  return absl::OkStatus();
}

absl::Status LlmWebGpu::UseLayers(ExecutionEnvironment* env,
                                  const absl::flat_hash_set<int>& layers,
                                  int final_ln) {
  if (!tensor_loader_) {
    return absl::InternalError("Tensor loader not found.");
  }

  absl::flat_hash_set<int> tags;
  tags.insert(final_ln | kFinalLnOptionTag);
  for (int layer : layers) {
    tags.insert(layer | kTransformerLayerOptionTag);
  }
  if (tags == layer_tags_) {
    return absl::OkStatus();
  }
  layer_tags_ = std::move(tags);

  for (const auto& [sequence_size, model] : models_) {
    for (int layer : layers) {
      const std::string layer_name = layer_names_[layer];
      auto it = model.tensor_reload_fns.find(layer_name);
      if (it == model.tensor_reload_fns.end()) {
        continue;
      }
      const std::vector<odml::infra::gpu::LoadTensorFn>& load_fns = it->second;
      for (const auto& load_fn : load_fns) {
        ASSIGN_OR_RETURN(odml::infra::gpu::TensorsWithIds new_tensors,
                         load_fn(tensor_loader_.get()));
        for (auto& [new_tensor, id] : new_tensors) {
          RETURN_IF_ERROR(model.context->SetTensor(
              id, static_cast<SpatialTensor*>(new_tensor)));
        }
      }
    }
  }

  SetupOptionalNodes();

  WaitForWeightUploads();

  return absl::OkStatus();
}

void LlmWebGpu::SetupOptionalNodes() {
  absl::flat_hash_set<int> tags = layer_tags_;
  tags.insert(lora_tags_.begin(), lora_tags_.end());
  if (shared_kv_cache_tag_) {
    tags.insert(*shared_kv_cache_tag_);
  }

  for (auto& [_, model] : models_) {
    model.context->EnableOptionalNodes(tags);
  }
  if (scoring_model_.context) {
    scoring_model_.context->EnableOptionalNodes(tags);
  }
}

absl::Status LlmWebGpu::UpdateSamplerParams(
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

// TODO(b/523389424): Clean up UpdateConstraintMask as it is almost identical
// across all 3 of our backends.
absl::Status LlmWebGpu::UpdateConstraintMask(
    const odml::infra::llm_utils::Bitmap& constraint_bitmap,
    ExecutionEnvironment* env) {
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
  return constraint_mask_->UploadDescriptorData(env->queue(), desc);
}

absl::Status LlmWebGpu::EncodeCopyCache(
    ExecutionEnvironment* env, int src_size, SpatialTensor* src_k,
    SpatialTensor* src_v, int dst_size, SpatialTensor* dst_k,
    SpatialTensor* dst_v, wgpu::ComputePassEncoder encoder, bool is_global) {
  if (src_k == nullptr || src_v == nullptr) {
    return absl::InternalError(
        "Src k or v tensor is null. Most likely caused by dynamic casting.");
  }
  auto& copy_cache = (is_global ? copy_cache_global_ : copy_cache_);
  RETURN_IF_ERROR(copy_cache->SetInt("src_size", src_size));
  RETURN_IF_ERROR(copy_cache->SetInt("dst_size", dst_size));
  RETURN_IF_ERROR(copy_cache->SetSrcTensor(0, src_k));
  RETURN_IF_ERROR(copy_cache->SetSrcTensor(1, src_v));
  RETURN_IF_ERROR(copy_cache->SetDstTensor(0, dst_k));
  RETURN_IF_ERROR(copy_cache->SetDstTensor(1, dst_v));
  RETURN_IF_ERROR(copy_cache->Update(env->device()));
  copy_cache->UpdateGpuObjectBindings(env->device());
  return copy_cache->Encode(encoder);
}

absl::Status LlmWebGpu::EncodeUpdateParams(ExecutionEnvironment* env,
                                           TransformerConfig& config,
                                           TransformerParams& params,
                                           const LlmRuntimeParams& param_vals,
                                           wgpu::ComputePassEncoder encoder) {
  const auto params_i32 = GetLlmRuntimeParamsBuffer(param_vals, config);
  for (int i = 0; i < params_i32.size(); ++i) {
    const std::string param_name = "param" + std::to_string(i);
    RETURN_IF_ERROR(params.write_i32_params->SetInt(param_name, params_i32[i]));
  }
  RETURN_IF_ERROR(
      params.write_i32_params->SetDstTensor(0, params.params_i32.get()));
  RETURN_IF_ERROR(params.write_i32_params->Update(env->device()));
  params.write_i32_params->UpdateGpuObjectBindings(env->device());
  return params.write_i32_params->Encode(encoder);
}

absl::Status LlmWebGpu::ExecuteUpdateIntParams(
    ExecutionEnvironment* env, TransformerConfig& config,
    TransformerParams& params, const LlmRuntimeParams& param_vals) {
  ComputePassHelper helper(env);
  RETURN_IF_ERROR(EncodeUpdateParams(env, config, params, param_vals,
                                     helper.compute_pass_encoder()));
  helper.Submit();
  return absl::OkStatus();
}

absl::Status LlmWebGpu::ExecuteUpdateParams(ExecutionEnvironment* env,
                                            SpatialTensor* tensor,
                                            const std::vector<float>& params) {
  RETURN_IF_ERROR(write_f32_params_->SetDstTensor(0, tensor));
  for (int i = 0; i < params.size(); ++i) {
    const std::string param_name = "param" + std::to_string(i);
    RETURN_IF_ERROR(write_f32_params_->SetFloat(param_name, params[i]));
  }
  RETURN_IF_ERROR(write_f32_params_->Update(env->device()));
  RETURN_IF_ERROR(write_f32_params_->Execute(*env));
  return absl::OkStatus();
}

absl::Status LlmWebGpu::ExecuteUpdateScoringParams(
    ExecutionEnvironment* env, SpatialTensor* tensor,
    const std::vector<float>& params) {
  RETURN_IF_ERROR(write_scoring_params_f32_->SetDstTensor(0, tensor));
  for (int i = 0; i < params.size(); ++i) {
    const std::string param_name = "param" + std::to_string(i);
    RETURN_IF_ERROR(write_scoring_params_f32_->SetFloat(param_name, params[i]));
  }
  RETURN_IF_ERROR(write_scoring_params_f32_->Update(env->device()));
  RETURN_IF_ERROR(write_scoring_params_f32_->Execute(*env));
  return absl::OkStatus();
}

absl::Status LlmWebGpu::ExecutePostProcessing(ExecutionEnvironment* env) {
  if (params_f32_) {
    std::uniform_real_distribution<float> distribution =
        std::uniform_real_distribution<float>(0.0f, sampler_params_.p());
    std::vector<float> params(params_f32_->Channels(), 0.0f);

    for (int i = 0; i < params.size() - 1; ++i) {
      params[i] = distribution(*rand_gen_);
    }

    params[params.size() - 1] = 1.0f / sampler_params_.temperature();
    RETURN_IF_ERROR(ExecuteUpdateParams(env, params_f32_.get(), params));
  }
  RETURN_IF_ERROR(ExecuteLazy(*env, *post_processing_.get()));
  return absl::OkStatus();
}

absl::StatusOr<GpuSpatialTensor*> LlmWebGpu::CreateModelTensor(
    ExecutionEnvironment* env, LlmTensorLoader* tensor_loader,
    std::vector<std::unique_ptr<SpatialTensor>>* model_tensors,
    const TensorDescriptor& tensor_desc) {
#ifdef __EMSCRIPTEN__
  SpatialTensor& tensor =
      *model_tensors->emplace_back(std::make_unique<SpatialTensor>());
  // Grab the owned data so it will be freed after upload.
  std::unique_ptr<DataHolder<uint8_t>> owned_weights =
      tensor_loader->GetOwnedData(tensor_desc.GetData().data());
  RETURN_IF_ERROR(tensor.CreateFromDescriptor(env->device(), tensor_desc));
#else
  SpatialTensor& tensor =
      *model_tensors->emplace_back(std::make_unique<NotifiedTensor>());
  // Tick the device to allow freeing any staging buffers.
  env->device().Tick();
  absl::Span<const uint8_t> data = tensor_desc.GetData();
  std::unique_ptr<DataHolder<uint8_t>> owned_weights =
      tensor_loader->GetOwnedData(data.data());
  if (owned_weights &&
      env->device().HasFeature(wgpu::FeatureName::HostMappedPointer)) {
    RETURN_IF_ERROR(CopyBufferToTensor(env, tensor_desc,
                                       std::move(owned_weights), &tensor));
  } else {
    RETURN_IF_ERROR(tensor.CreateFromDescriptor(env->device(), tensor_desc));
  }
  env->device().Tick();
#endif  // __EMSCRIPTEN__
  return &tensor;
}

#ifndef __EMSCRIPTEN__
absl::Status LlmWebGpu::CopyBufferToTensor(
    ExecutionEnvironment* env, const TensorDescriptor& desc,
    std::unique_ptr<DataHolder<uint8_t>> owned_weights, SpatialTensor* tensor) {
  TensorCopyInfo copy_info = GetTensorCopyInfo(desc);
  // The Metal API imports host pointers from mmap'd data quickly, so no need
  // to copy the data. Vulkan and D3D12 both import memory much more quickly
  // from non-mmap'd memory, so in those APIs it's faster to copy all data
  // before importing.
#if defined(__APPLE__)
  const bool should_copy = copy_info.NeedsCopy();
#else
  const bool should_copy = true;
#endif
  if (!should_copy && desc.GetStorageType() == TensorStorageType::kBuffer) {
    return CopyBufferToBuffer(env, desc, std::move(owned_weights), tensor);
  }
  // First create the texture object with no data. This will allow creating the
  // bind groups and other things for the texture. The actual data will be
  // uploaded later on a background thread.
  TensorDescriptor desc_no_data;
  desc.CopyWithoutData(&desc_no_data);
  RETURN_IF_ERROR(tensor->CreateFromDescriptor(env->device(), desc_no_data));

  // Grab a few variables for the closure below.
  absl::Span<const uint8_t> weights_data = desc.GetData();
  wgpu::Texture dst_texture = tensor->GetTextureHandle();
  wgpu::Buffer dst_buffer = tensor->GetBufferHandle();
  wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();

  if (!buffer_upload_executor_ && executor_factory_) {
    ASSIGN_OR_RETURN(buffer_upload_executor_,
                     executor_factory_->CreateExecutor(2));
  }

  auto upload_notification = std::make_shared<absl::Notification>();
  auto* notified_tensor = static_cast<NotifiedTensor*>(tensor);
  notified_tensor->set_upload_notification(upload_notification);

  // Run the buffer upload on a background thread. This will need to be joined
  // before the data is used during inference.
  auto fn = [=, weights_ptr = owned_weights.release()] {
    std::unique_ptr<DataHolder<uint8_t>> owned_weights(weights_ptr);

    uint32_t offset = 0;
    if (should_copy) {
      auto in_memory_weights =
          std::make_unique<InMemoryDataHolder>(copy_info.AlignedSize());
      CopyTensorData(copy_info, weights_data, in_memory_weights->GetData());
      owned_weights = std::move(in_memory_weights);
    } else {
      offset = weights_data.data() - owned_weights->GetRawData().data();
    }

    wgpu::Buffer buffer = CreateOwnedDataBuffer(env, std::move(owned_weights),
                                                /*is_buffer=*/false);

    if (dst_texture) {
      wgpu::Extent3D copy_size = {
          .width = static_cast<uint32_t>(copy_info.region.x),
          .height = static_cast<uint32_t>(copy_info.region.y),
          .depthOrArrayLayers = static_cast<uint32_t>(copy_info.region.z),
      };
      wgpu::TexelCopyTextureInfo copy_texture = {
          .texture = dst_texture,
      };
      wgpu::TexelCopyBufferInfo copy_buffer = {
          .layout =
              {
                  .offset = offset,
                  .bytesPerRow = copy_info.bytes_per_row_aligned,
                  .rowsPerImage = static_cast<uint32_t>(copy_info.region.y),
              },
          .buffer = buffer,
      };

      encoder.CopyBufferToTexture(&copy_buffer, &copy_texture, &copy_size);
    } else {
      encoder.CopyBufferToBuffer(buffer, offset, dst_buffer, 0,
                                 copy_info.AlignedSize());
    }
    wgpu::CommandBuffer cb = encoder.Finish();
    env->queue().Submit(1, &cb);
    upload_notification->Notify();
    env->device().Tick();
  };
  if (buffer_upload_executor_) {
    buffer_upload_executor_->Schedule(std::move(fn));
  } else {
    fn();
  }
  return absl::OkStatus();
}
#endif  // !__EMSCRIPTEN__

absl::Status LlmWebGpu::UpdateEmbeddings(ExecutionEnvironment* env,
                                         StackedTransformerInference& model,
                                         int previous_token,
                                         absl::Span<const int> tokens) {
  odml::infra::gpu::UpdateNgrammerEmbeddings(
      ngrammer_embeddings_, model.ngrammer_weights, previous_token,
      tokens, text_config_.vocabulary_size,
      text_config_.ngram_vocabulary_size.value_or(0));
  odml::infra::gpu::UpdateEmbeddings(
      ple_embeddings_, model.ple_weights, tokens,
      text_config_.vocabulary_size, tokens_transform_fn_);
  RETURN_IF_ERROR(UploadEmbeddingsToGPU(model.ngrammer_weights,
                                        model.context.get(), env->queue()));
  RETURN_IF_ERROR(UploadEmbeddingsToGPU(model.ple_weights, model.context.get(),
                                        env->queue()));
  return absl::OkStatus();
}

absl::StatusOr<speculative_utils::TokenAtTimeStep> LlmWebGpu::PeekNextToken(
    ExecutionEnvironment* env) {
  if (output_batch_size_ > 1) {
    return absl::UnimplementedError(
        "Multiple output head not supported for PeekNextToken.");
  }

  const int time_step = input_tokens_.size() + (ready_tokens_ - 1);
  int token_id = -1;
  if (token_override_.contains(time_step)) {
    token_id = token_override_[time_step];
  } else {
    ASSIGN_OR_RETURN(TensorInt32 output, DownloadTokenIds(env));
    token_id = output.data[time_step * output_batch_size_];
  }
  return speculative_utils::TokenAtTimeStep(time_step, token_id);
}

absl::Status LlmWebGpu::OverwriteOneInputToken(
    ExecutionEnvironment* env,
    const speculative_utils::TokenAtTimeStep& token) {
  RET_CHECK_GT(token.time_step(), 0)
      << "Time step must be positive for OverwriteOneInputToken.";

  RETURN_IF_ERROR(WriteInputTokensForModel(*env, *model_one_, {token.token()}));
  token_override_[token.time_step()] = token.token();
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> LlmWebGpu::DraftTokens(
    ExecutionEnvironment* env) {
  const int draft_size = gamma_ + 1;
  const int time_step = input_tokens_.size() + (ready_tokens_ - 1);
  if (time_step + draft_size > text_config_.cache_size) {
    return std::vector<int>{};
  }
  std::vector<std::vector<int>> output_tokens(1, std::vector<int>(draft_size));

  RETURN_IF_ERROR(
      GetNextTokensWithPullTopKProbabilities(env, &output_tokens, true, {}));
  return output_tokens[0];
}

absl::StatusOr<std::vector<speculative_utils::TokenProbability>>
LlmWebGpu::GetProbabilityCache(int time_step) {
  if (!probability_cache_.contains(time_step)) {
    return absl::NotFoundError("Probability cache not found at time step: " +
                               std::to_string(time_step));
  }
  return probability_cache_[time_step];
}

absl::Status LlmWebGpu::ScoreTokens(ExecutionEnvironment* env,
                                    const std::vector<int>& tokens) {
  const int draft_size = gamma_ + 1;
  RET_CHECK(tokens.size() == draft_size)
      << "ScoreToken requires the tokens length to be gamma + 1";

  // Send data to GPU.
  RETURN_IF_ERROR(WriteInputTokensForModel(*env, scoring_model_, tokens));

  // Update parameters for scoring model.
  const int time_step = input_tokens_.size() + (ready_tokens_ - 1);

  // TODO: b/355494120 - Avoid a sync each time if possible.
  if (!scoring_model_.ple_weights.empty() ||
      !scoring_model_.ngrammer_weights.empty()) {
    ASSIGN_OR_RETURN(TensorInt32 previous_tokens, DownloadTokenIds(env));
    int previous_token =
        time_step > 1 ? previous_tokens.data[time_step - 2] : 0;
    RETURN_IF_ERROR(
        UpdateEmbeddings(env, scoring_model_, previous_token, tokens));
  }

  LlmRuntimeParams llm_params;
  llm_params.token_index_offset = time_step;
  llm_params.active_tokens = time_step + draft_size;
  llm_params.topk = sampler_params_.k();
  RETURN_IF_ERROR(
      ExecuteUpdateIntParams(env, text_config_, text_params_, llm_params));

  // Run scoring model.
  RETURN_IF_ERROR(ExecuteLazy(*env, *scoring_model_.context.get()));

  if (scoring_params_f32_) {
    std::uniform_real_distribution<float> distribution =
        std::uniform_real_distribution<float>(0.0f, sampler_params_.p());
    std::vector<float> params(scoring_params_f32_->Channels(), 0.0f);

    for (int i = 0; i < params.size() - 1; ++i) {
      params[i] = distribution(*rand_gen_);
    }

    params[params.size() - 1] = 1.0f / sampler_params_.temperature();
    RETURN_IF_ERROR(
        ExecuteUpdateScoringParams(env, scoring_params_f32_.get(), params));
  }

  RETURN_IF_ERROR(ExecuteLazy(*env, *scoring_post_processing_.get()));

  RETURN_IF_ERROR(PullScoringTopKProbabilities(env, time_step + 1));

  ready_tokens_ += draft_size;
  return absl::OkStatus();
}

absl::Status LlmWebGpu::PullScoringTopKProbabilities(ExecutionEnvironment* env,
                                                     int time_step) {
  const BHWC dst_shape = scoring_topk_tokens_->GetDescriptor().GetBHWCShape();

  TensorDescriptor topk_prob_desc;
  RETURN_IF_ERROR(scoring_topk_probabilities_->ToDescriptor(env->device(),
                                                            &topk_prob_desc));
  TensorFloat32 topk_prob_out;
  topk_prob_desc.DownloadData(&topk_prob_out);
  TensorDescriptor topk_token_desc;
  RETURN_IF_ERROR(
      scoring_topk_tokens_->ToDescriptor(env->device(), &topk_token_desc));
  TensorInt32 topk_token_out;
  topk_token_desc.DownloadData(&topk_token_out);

  const int draft_size = gamma_ + 1;
  RET_CHECK_EQ(dst_shape.b * dst_shape.w, draft_size)
      << "Scoring post processing inference context's TopK probability batch "
         "is not Gamma + 1";
  for (int i = 0; i < draft_size; i++) {
    // Note that the TopK GPU Operation operates on 4 items at a time, so top K
    // not multiples of 4 will result in extra elements we can skip.
    for (int j = 0; j < dst_shape.c && j < sampler_params_.k(); j++) {
      ASSIGN_OR_RETURN(speculative_utils::TokenProbability token_probability,
                       speculative_utils::TokenProbability::Create(
                           topk_token_out.data[i * dst_shape.c + j],
                           topk_prob_out.data[i * dst_shape.c + j]));
      probability_cache_[time_step + i].push_back(token_probability);
    }
  }

  // The output of the scoring model is not connected to model_one_, so we need
  // to pull it out specifically.
  TensorDescriptor transformer_input_desc;
  RETURN_IF_ERROR(scoring_model_.context->GetTensor(scoring_model_.input.id)
                      ->ToDescriptor(env->device(), &transformer_input_desc));
  TensorInt32 transformer_input_out;
  transformer_input_desc.DownloadData(&transformer_input_out);
  RETURN_IF_ERROR(OverwriteOneInputToken(
      env, speculative_utils::TokenAtTimeStep(
               time_step + draft_size - 1,
               transformer_input_out.data[draft_size - 1])));
  return absl::OkStatus();
}

absl::Status LlmWebGpu::RejectAfterResample(
    ExecutionEnvironment* env,
    const speculative_utils::TokenAtTimeStep& resampled) {
  int initial_time_step = input_tokens_.size() - 1 + ready_tokens_;

  // time_step is the position at which the token is invalid,
  // ie. the next token to be sampled.
  RETURN_IF_ERROR(CleanCache(env, resampled.time_step() + 1));
  ready_tokens_ = resampled.time_step() - (input_tokens_.size() - 1);
  num_output_tokens_ = ready_tokens_ - 1;
  RETURN_IF_ERROR(OverwriteOneInputToken(env, resampled));

  // Clear the probability cache for the time step that was overwritten and the
  // following time steps.
  for (int i = resampled.time_step(); i <= initial_time_step; ++i) {
    probability_cache_.erase(i);
  }
  return absl::OkStatus();
}

absl::StatusOr<TensorInt32> LlmWebGpu::DownloadTokenIds(
    ExecutionEnvironment* env) {
  TensorDescriptor desc;
  RETURN_IF_ERROR(tokens_ids_->ToDescriptor(env->device(), &desc));
  TensorInt32 output;
  desc.DownloadData(&output);
  return output;
}

// src has sh: [1, 1, x, emb_dim]
// dst has sh: [1, 1, media_cache_size_, emb_dim]
// we perform: dst[0, 0, index:index + x, :] = src
// expect dst to be either media_embeddings_ or vision_ple_tensors_
absl::Status LlmWebGpu::UpdateMultimodalEmbeddings(
    SpatialTensor* src, SpatialTensor* dst, int index,
    ExecutionEnvironment* env, const wgpu::CommandEncoder& encoder) {
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

  wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  RETURN_IF_ERROR(multimodal_embeddings_op_->SetInt("index", index));
  RETURN_IF_ERROR(multimodal_embeddings_op_->SetSrcTensor(0, src));
  RETURN_IF_ERROR(multimodal_embeddings_op_->SetDstTensor(0, dst));
  RETURN_IF_ERROR(multimodal_embeddings_op_->Update(env->device()));
  multimodal_embeddings_op_->UpdateGpuObjectBindings(env->device());
  RETURN_IF_ERROR(multimodal_embeddings_op_->Encode(pass));
  pass.End();
  return absl::OkStatus();
}

// vision_ple(src) should be a vector of size (text_config_.stack_size)
// and each entry should have shape (1, 1, x, ple_emb_dim)
// vision_ple_tensors_ (dst) should have size
// (text_config_.stack_size - text_config_.extra_stack_size) each entry should
// have shape (1, 1, media_cache_size, ple_emb_dim) This op performs the
// following: for each i in range(text_config_.stack_size -
// text_config_.extra_stack_size):
//   dst[i][0, 0, index:index + x, :] = src[i]
absl::Status LlmWebGpu::FillVisionPleEmbeddings(
    const std::vector<TensorFloat32>& vision_ple, int index,
    ExecutionEnvironment* env, const wgpu::CommandEncoder& encoder,
    std::vector<std::unique_ptr<SpatialTensor>>& temp_tensors) {
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
    auto& src = temp_tensors.emplace_back(std::make_unique<SpatialTensor>());
    RETURN_IF_ERROR(CreateTensor(env->device(), desc, src.get()));
    RETURN_IF_ERROR(src->UploadDescriptorData(env->queue(), desc));
    RETURN_IF_ERROR(UpdateMultimodalEmbeddings(
        src.get(), vision_ple_tensors_[i].get(), index, env, encoder));
  }
  return absl::OkStatus();
}

absl::Status LlmWebGpu::FillMediaEmbeddings(
    const TensorFloat32& media_tensor, int index, ExecutionEnvironment* env,
    const wgpu::CommandEncoder& encoder,
    std::vector<std::unique_ptr<SpatialTensor>>& temp_tensors) {
  TensorDescriptor desc = media_embeddings_->GetDescriptor();
  desc.SetBHWCShape(media_tensor.shape);
  desc.UploadData(media_tensor);
  auto& media_gpu =
      temp_tensors.emplace_back(std::make_unique<SpatialTensor>());
  RETURN_IF_ERROR(CreateTensor(env->device(), desc, media_gpu.get()));
  RETURN_IF_ERROR(media_gpu->UploadDescriptorData(env->queue(), desc));
  return UpdateMultimodalEmbeddings(media_gpu.get(), media_embeddings_.get(),
                                    index, env, encoder);
}

absl::Status LlmWebGpu::WriteInputTokensForModel(
    const ExecutionEnvironment& env, StackedTransformerInference& model,
    absl::Span<const int> input_tokens, int token_index_offset,
    const wgpu::ComputePassEncoder* compute_pass_helper) {
  // If `compute_pass_helper` is provided, we assume that multiple operations
  // are combined in a single compute pass and updating `input_token_ids_` is
  // handled by the caller.
  if (!compute_pass_helper) {
    WriteDataToBuffer(
        env.queue(), input_token_ids_->GetMemoryHandle(),
        input_tokens.size() * sizeof(decltype(input_tokens)::element_type),
        input_tokens.data());
  }

  ComputeTask& copy_tokens_op = *model.copy_tokens_op;
  RETURN_IF_ERROR(copy_tokens_op.SetSrcBuffer(0, input_token_ids_.get()));
  RETURN_IF_ERROR(
      copy_tokens_op.SetDstTensor(0, model.context->GetTensor(model.input.id)));
  RETURN_IF_ERROR(copy_tokens_op.SetInt("token_offset", token_index_offset));
  RETURN_IF_ERROR(copy_tokens_op.SetInt("tokens_count", input_tokens.size()));
  RETURN_IF_ERROR(copy_tokens_op.Update(env.device()));
  copy_tokens_op.UpdateGpuObjectBindings(env.device());

  if (compute_pass_helper) {
    RETURN_IF_ERROR(copy_tokens_op.Encode(*compute_pass_helper));
  } else {
    RETURN_IF_ERROR(copy_tokens_op.Execute(env));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> LlmWebGpu::DecodeToLogits(
    ExecutionEnvironment* env, const uint32_t model_size,
    const uint32_t cache_size, absl::Span<const float> embeddings,
    absl::Span<const float> ple_embeddings,
    absl::Span<const float> local_attention_mask,
    absl::Span<const float> global_attention_mask,
    absl::Span<const int> seq_position,
    absl::Span<const uint32_t> logits_indices, const std::atomic_bool* cancel) {
  if (!text_config_.enable_decode_logits) {
    return absl::InvalidArgumentError(
        "LlmWebGpu::DecodeToLogits is not allowed without "
        "enable_decode_logits_=true");
  }

  std::vector<float> result;
  // The ple_embedding is model_size * stack_size *
  // per_layer_embedding_dimension while the input to the model is
  // size stack_size * model_size * per_layer_embedding_dimension.
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
  WriteDataToBuffer(env->queue(), input_embeddings_->GetMemoryHandle(),
                    embeddings.size() * sizeof(float), embeddings.data());
  WriteDataToBuffer(env->queue(), input_ple_embeddings_->GetMemoryHandle(),
                    ple_input_data.size() * sizeof(float),
                    ple_input_data.data());
  WriteDataToBuffer(env->queue(), input_sequence_position_->GetMemoryHandle(),
                    seq_position.size() * sizeof(int32_t), seq_position.data());
  WriteDataToBuffer(env->queue(), input_global_mask_->GetMemoryHandle(),
                    global_attention_mask.size() * sizeof(float),
                    global_attention_mask.data());
  WriteDataToBuffer(env->queue(), input_local_mask_->GetMemoryHandle(),
                    local_attention_mask.size() * sizeof(float),
                    local_attention_mask.data());
  {
    int group_size = model_size;
    auto& model_n = models_[group_size];
    // Copy embeddings
    {
      auto& copy_raw_op = model_n.copy_external_embeddings_op;
      RETURN_IF_ERROR(copy_raw_op->SetSrcBuffer(0, input_embeddings_.get()));
      RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.external_embeddings.id)));
      RETURN_IF_ERROR(copy_raw_op->SetInt("embedding_offset", 0));
      RETURN_IF_ERROR(copy_raw_op->SetInt("embedding_dimension",
                                          text_config_.model_dimension));
      RETURN_IF_ERROR(copy_raw_op->Update(env->device()));
      RETURN_IF_ERROR(copy_raw_op->Execute(*env));
    }

    // Copy PLE embeddings
    {
      int embedding_offset = 0;
      for (int i = 0; i < model_n.external_ple_embeddings.size(); ++i) {
        auto& copy_raw_op = model_n.copy_external_embeddings_op;
        RETURN_IF_ERROR(
            copy_raw_op->SetSrcBuffer(0, input_ple_embeddings_.get()));
        RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
            0,
            model_n.context->GetTensor(model_n.external_ple_embeddings[i].id)));
        RETURN_IF_ERROR(
            copy_raw_op->SetInt("embedding_dimension",
                                (*text_config_.per_layer_embedding_dimension)));
        RETURN_IF_ERROR(
            copy_raw_op->SetInt("embedding_offset", embedding_offset));
        RETURN_IF_ERROR(copy_raw_op->Update(env->device()));
        RETURN_IF_ERROR(copy_raw_op->Execute(*env));
        embedding_offset +=
            (*text_config_.per_layer_embedding_dimension) * model_size;
      }
    }

    // Copy positions
    {
      auto& copy_raw_op = model_n.copy_positions_op;
      RETURN_IF_ERROR(
          copy_raw_op->SetSrcBuffer(0, input_sequence_position_.get()));
      RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.positions.id)));
      RETURN_IF_ERROR(copy_raw_op->SetInt("token_offset", 0));
      RETURN_IF_ERROR(copy_raw_op->SetInt("tokens_count", model_size));
      RETURN_IF_ERROR(copy_raw_op->Update(env->device()));
      RETURN_IF_ERROR(copy_raw_op->Execute(*env));
    }

    // Copy masks
    {
      auto& copy_raw_op = model_n.copy_masks_op;
      RETURN_IF_ERROR(copy_raw_op->SetSrcBuffer(0, input_local_mask_.get()));
      RETURN_IF_ERROR(copy_raw_op->SetSrcBuffer(1, input_global_mask_.get()));

      RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          0, model_n.context->GetTensor(model_n.local_mask.id)));
      RETURN_IF_ERROR(copy_raw_op->SetDstTensor(
          1, model_n.context->GetTensor(model_n.global_mask.id)));
      RETURN_IF_ERROR(copy_raw_op->Update(env->device()));
      RETURN_IF_ERROR(copy_raw_op->Execute(*env));
    }

    LlmRuntimeParams llm_params;
    llm_params.token_index_offset = input_tokens_.size();
    llm_params.active_tokens = input_tokens_.size() + model_size;
    llm_params.topk = sampler_params_.k();
    RETURN_IF_ERROR(
        ExecuteUpdateIntParams(env, text_config_, text_params_, llm_params));
    RETURN_IF_ERROR(ExecuteLazy(*env, *model_n.context));
  }

  for (uint32_t gen_logits_index : logits_indices) {
    auto& last_model = models_[model_size];

    ComputeTask& copy_nth_op = *last_model.copy_nth_op;
    RETURN_IF_ERROR(copy_nth_op.SetSrcTensor(
        0, last_model.context->GetTensor(last_model.output.id)));
    // Write to the temporary tensor first, matching the WebGPU pattern
    RETURN_IF_ERROR(
        copy_nth_op.SetDstTensor(0, temporary_output_tensor_.get()));
    RETURN_IF_ERROR(copy_nth_op.SetInt("nth", gen_logits_index));
    RETURN_IF_ERROR(copy_nth_op.Update(env->device()));
    RETURN_IF_ERROR(copy_nth_op.Execute(*env));

    // Copy from temp tensor to the final embedding model input
    wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
    RETURN_IF_ERROR(CopyTensor(
        *temporary_output_tensor_,
        *final_embedding_to_logits_->GetTensor(final_embedding_.id), encoder));
    wgpu::CommandBuffer cb = encoder.Finish();
    env->queue().Submit(1, &cb);

    RETURN_IF_ERROR(ExecuteLazy(*env, *final_embedding_to_logits_));

    // Download logits
    TensorDescriptor desc;
    RETURN_IF_ERROR(final_embedding_to_logits_->GetTensor(logits_.id)
                        ->ToDescriptor(env->device(), &desc));
    TensorFloat32 output;
    desc.DownloadData(&output);
    result.insert(result.end(), output.data.begin(), output.data.end());
  }

  // Insert 0s to make sure input_tokens_ size is correct.
  input_tokens_.insert(input_tokens_.end(), model_size, 0);
  return result;
}

absl::Status LlmWebGpu::ComputeTokenEmbeddings(
    ExecutionEnvironment* env, absl::Span<const int> input_tokens,
    absl::Span<float> output_embeddings,
    absl::Span<float> output_ple_embeddings) {
  size_t input_size = input_tokens.size();
  if (input_size == 0) {
    return absl::InvalidArgumentError("Input tokens cannot be empty.");
  }

  // Copy input tokens to input_token_ids_.
  WriteDataToBuffer(env->queue(), input_token_ids_->GetMemoryHandle(),
                    input_size * sizeof(int32_t), input_tokens.data());

  ASSIGN_OR_RETURN(auto input_groups,
                   GetGroupSizesForInput(input_size, group_sizes_));

  int token_index_offset = 0;
  size_t result_write_offset = 0;
  size_t ple_result_write_offset = 0;

  for (int i = 0; i < input_groups.size(); ++i) {
    int group_size = input_groups[i];
    auto& model_n = models_[group_size];

    const int active_token_size =
        std::min(static_cast<int>(input_size - token_index_offset), group_size);

    RETURN_IF_ERROR(UpdatePleEmbeddings(
        env, model_n,
        input_tokens.subspan(token_index_offset, active_token_size)));

    EmbeddingLookupModel& embedding_model = model_n.embedding_lookup_model;
    // Generate embeddings for the group.
    {
      // Prepare input.
      auto& copy_tokens_op = embedding_model.copy_tokens_op;
      RETURN_IF_ERROR(copy_tokens_op->SetSrcBuffer(0, input_token_ids_.get()));
      RETURN_IF_ERROR(copy_tokens_op->SetDstTensor(
          0,
          embedding_model.context->GetTensor(embedding_model.input_tokens.id)));
      RETURN_IF_ERROR(
          copy_tokens_op->SetInt("token_offset", token_index_offset));
      RETURN_IF_ERROR(copy_tokens_op->SetInt(
          "tokens_count", token_index_offset + active_token_size));
      RETURN_IF_ERROR(copy_tokens_op->Update(env->device()));
      RETURN_IF_ERROR(copy_tokens_op->Execute(*env));
    }
    RETURN_IF_ERROR(ExecuteLazy(*env, *embedding_model.context));

    // --- Download embeddings ---
    TensorDescriptor embedding_desc;
    RETURN_IF_ERROR(
        embedding_model.context->GetTensor(embedding_model.output_embeddings.id)
            ->ToDescriptor(env->device(), &embedding_desc));
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
    std::vector<float> group_ple_buffer(active_token_size *
                                        total_ple_dim_per_token);

    for (int layer_idx = 0; layer_idx < num_ple_layers; ++layer_idx) {
      TensorDescriptor ple_output_desc;
      RETURN_IF_ERROR(
          embedding_model.context
              ->GetTensor(embedding_model.output_ple_embeddings[layer_idx].id)
              ->ToDescriptor(env->device(), &ple_output_desc));
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
        absl::Span<float> token_layer_dst =
            dst_span.subspan((t * total_ple_dim_per_token) +
                                 (layer_idx * per_layer_embedding_dimension),
                             per_layer_embedding_dimension);

        absl::c_copy(token_layer_src, token_layer_dst.begin());
      }
    }

    // Copy fully re-ordered group buffer to the final ple_result span
    size_t ple_data_to_copy = group_ple_buffer.size();

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

absl::Status LlmWebGpu::UpdatePleEmbeddings(ExecutionEnvironment* env,
                                            StackedTransformerInference& model,
                                            absl::Span<const int> tokens) {
  odml::infra::gpu::UpdateEmbeddings(
      ple_embeddings_, model.embedding_lookup_model.ple_weights,
      tokens, text_config_.vocabulary_size, tokens_transform_fn_);
  RETURN_IF_ERROR(UploadEmbeddingsToGPU(
      model.embedding_lookup_model.ple_weights,
      model.embedding_lookup_model.context.get(), env->queue()));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>>
LlmWebGpu::GetVariants(ExecutionEnvironment* env) {
  std::vector<std::pair<uint32_t, uint32_t>> variants;
  variants.reserve(models_.size());
  for (const auto& kv : models_) {
    variants.push_back(std::pair<uint32_t, uint32_t>{
        static_cast<uint32_t>(kv.first),
        static_cast<uint32_t>(text_config_.cache_size)});
  }
  return variants;
}

}  // namespace webgpu
}  // namespace ml_drift
