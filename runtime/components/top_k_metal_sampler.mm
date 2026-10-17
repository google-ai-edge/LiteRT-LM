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

#include "runtime/components/top_k_metal_sampler.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift/metal/common.h"  // from @ml_drift
#include "ml_drift/metal/compute_task.h"  // from @ml_drift
#include "ml_drift/metal/environment.h"  // from @ml_drift
#include "ml_drift/metal/inference_context.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "litert/cc/internal/litert_handle.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_event.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "ml_drift_delegate/delegate/custom_event_metal.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

#define SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(expr, error_msg) \
  do {                                                          \
    auto status = (expr);                                       \
    if (!status.ok()) {                                         \
      if (error_msg) {                                          \
        *error_msg = strdup(status.ToString().c_str());         \
      }                                                         \
      return static_cast<int>(status.code());                   \
    }                                                           \
  } while (0)

namespace litert::lm {
namespace {

using ::litert::lm::proto::SamplerParameters;
using ::ml_drift::BHWC;
using ::ml_drift::CreateGpuModelInfo;
using ::ml_drift::DataType;
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::GPUOperation;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorToGrid;
using ::ml_drift::metal::ComputeTask;
using MetalEnv = ::ml_drift::metal::Environment;
using ::ml_drift::metal::CreateTensor;
using ::ml_drift::metal::GetFastestStorageType;
using ::ml_drift::metal::InferenceContext;
using ::ml_drift::metal::MetalSpatialTensor;

// A helper class to handle creating encoders/submitting.
class ComputePassHelper {
 public:
  explicit ComputePassHelper(const MetalEnv* env, id<MTLCommandQueue> queue) : env_(env) {
    command_buffer_ = [queue commandBuffer];
    command_encoder_ = [command_buffer_ computeCommandEncoder];
  }

  id<MTLComputeCommandEncoder> compute_pass_encoder() const { return command_encoder_; }

  void Submit() {
    [command_encoder_ endEncoding];
    [command_buffer_ commit];
    // [command_buffer_ waitUntilCompleted]; // Removed to allow pipelining
  }

 private:
  const MetalEnv* env_;
  // id<MTLCommandQueue> command_queue_; // Removed
  id<MTLCommandBuffer> command_buffer_;
  id<MTLComputeCommandEncoder> command_encoder_;
};

absl::Status InitComputeTask(MetalEnv* env, std::unique_ptr<GPUOperation>&& gpu_op,
                             ComputeTask* task) {
  ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(env->GetInfo()));
  task->Init(std::move(gpu_op), /*use_argument_buffer=*/false);
  return task->Compile(env);
}

absl::Status InitComputeTask(MetalEnv* env, GPUOperation&& operation, ComputeTask* task) {
  return InitComputeTask(env, std::make_unique<GPUOperation>(std::move(operation)), task);
}

GPUOperation CreateWriteParamsOp(const TensorDescriptor& dst, int params_count) {
  GPUOperation op;
  op.AddDstTensor("dst", dst);
  for (int i = 0; i < params_count; ++i) {
    const std::string param_name = absl::StrCat("param", i);
    if (dst.GetDataType() == DataType::kFloat32) {
      op.args_.AddFloat(param_name, 0);
    } else if (dst.GetDataType() == DataType::kInt32) {
      op.args_.AddInt(param_name, 0);
    }
  }
  if (dst.GetDataType() == DataType::kFloat32) {
    op.args_.AddFloat("zero_value", 0);
  } else if (dst.GetDataType() == DataType::kInt32) {
    op.args_.AddInt("zero_value", 0);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.code_info_.uses_global_id = true;

  std::string c;
  c += R"(
MAIN_FUNCTION($0) {
int X = ucl::GetGlobalId<0>();
int Y = ucl::GetGlobalId<1>();
int S = ucl::GetGlobalId<2>();
if (X != 0 || Y != 0 || S != 0) return;
args.dst::type result;
)";
  std::string postfixes[4] = {".x", ".y", ".z", ".w"};
  for (int s = 0; s < ::ml_drift::DivideRoundUp(params_count, 4); ++s) {
    for (int ch = 0; ch < 4; ++ch) {
      std::string param_name = absl::StrCat("args.param", s * 4 + ch);
      if (s * 4 + ch >= params_count) {
        param_name = "args.zero_value";
      }
      c += "  result" + postfixes[ch] + " = " + param_name + ";\n";
    }
    c += "  args.dst.Write(result, 0, 0, " + std::to_string(s) + ", 0);\n";
  }
  c += "}\n";
  op.code_ = std::move(c);
  return op;
}

absl::StatusOr<std::unique_ptr<MetalEnv>> CreateMetalEnvFromLiteRtEnv(const Environment& env) {
  LITERT_ASSIGN_OR_RETURN(auto env_options, env.GetOptions());

  id<MTLDevice> device = nullptr;
  auto device_option = env_options.GetOption(litert::EnvironmentOptions::Tag::kMetalDevice);
  if (device_option.HasValue()) {
    if (std::holds_alternative<void*>(*device_option)) {
      device = (__bridge id<MTLDevice>)(std::get<void*>(*device_option));
    }
  }

  if (!device) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return absl::InternalError("Failed to create Metal device.");
    }
  }

  return std::make_unique<MetalEnv>(device);
}

::ml_drift::TensorDescriptor GetTensorDescriptor(const TensorBuffer* tensor_buffer) {
  if (!tensor_buffer) {
    return ::ml_drift::TensorDescriptor();
  }
  auto tensor_type_status = tensor_buffer->TensorType();
  if (!tensor_type_status.HasValue()) {
    return ::ml_drift::TensorDescriptor();
  }
  const auto& tensor_type = *tensor_type_status;
  ::ml_drift::DataType type;
  if (tensor_type.ElementType() == litert::ElementType::Float32) {
    type = ::ml_drift::DataType::kFloat32;
  } else if (tensor_type.ElementType() == litert::ElementType::Float16) {
    type = ::ml_drift::DataType::kFloat16;
  } else if (tensor_type.ElementType() == litert::ElementType::Int32) {
    type = ::ml_drift::DataType::kInt32;
  } else if (tensor_type.ElementType() == litert::ElementType::Bool) {
    type = ::ml_drift::DataType::kBool;
  } else {
    // Unsupported type.
    ABSL_LOG(ERROR) << "Unsupported tensor type: " << static_cast<int>(tensor_type.ElementType());
    return ::ml_drift::TensorDescriptor();
  }
  auto dims = tensor_type.Layout().Dimensions();
  ::ml_drift::BHWC shape(
      dims.size() > 3 ? dims[dims.size() - 4] : 1, dims.size() > 2 ? dims[dims.size() - 3] : 1,
      dims.size() > 1 ? dims[dims.size() - 2] : 1, dims.size() > 0 ? dims[dims.size() - 1] : 1);
  ::ml_drift::TensorDescriptor td(
      type, ::ml_drift::TensorStorageType::kBuffer,
      shape.b == 1 ? ::ml_drift::Layout::kHWC : ::ml_drift::Layout::kBHWC);
  td.SetBHWCShape(shape);
  return td;
}

absl::Status BindTensor(const TensorBuffer& tensor_buffer, ::ml_drift::ValueId tensor_id,
                        ::ml_drift::metal::InferenceContext& inference_context,
                        std::vector<::ml_drift::metal::MetalSpatialTensor>& shared_tensors) {
  LITERT_ASSIGN_OR_RETURN(auto metal_buffer_handle, tensor_buffer.GetMetalBuffer());
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metal_buffer_handle;
  shared_tensors.push_back(::ml_drift::metal::MetalSpatialTensor());
  auto& shared_tensor = shared_tensors.back();
  ABSL_RETURN_IF_ERROR(::ml_drift::metal::CreateTensorSharedBuffer(
      buffer, GetTensorDescriptor(&tensor_buffer), &shared_tensor));
  return inference_context.SetTensor(tensor_id, &shared_tensor);
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<TopKMetalSampler>> TopKMetalSampler::Create(
    Environment& env, int batch_size, int sequence_size, int vocab_size,
    std::optional<ActivationDataType> activation_data_type, SamplerParameters sampler_params) {
  RET_CHECK_GT(batch_size, 0) << "Batch size must be positive.";
  RET_CHECK_GT(vocab_size, 0) << "Vocabulary size must be positive.";

  ABSL_ASSIGN_OR_RETURN(auto metal_env, CreateMetalEnvFromLiteRtEnv(env));
  auto gpu_info = metal_env->GetInfo();

  ActivationDataType activation_data_type_copy;
  if (activation_data_type.has_value()) {
    activation_data_type_copy = *activation_data_type;
  } else {
    activation_data_type_copy =
        gpu_info.SupportsFP16() ? ActivationDataType::FLOAT16 : ActivationDataType::FLOAT32;
  }

  CreateGpuModelInfo create_info;
  if (activation_data_type_copy == ActivationDataType::FLOAT16) {
    create_info.precision = ::ml_drift::CalculationsPrecision::kF16;
  } else {
    create_info.precision = ::ml_drift::CalculationsPrecision::kF32;
  }
  create_info.hints.Add(::ml_drift::ModelHints::kFastTuning);
  create_info.hints.Add(::ml_drift::ModelHints::kPreferTextureWeights);
  create_info.hints.Add(::ml_drift::ModelHints::kAllowSpecialKernels);
  create_info.storage_type = GetFastestStorageType(gpu_info);

  ::ml_drift::DataType logits_data_type = (activation_data_type_copy == ActivationDataType::FLOAT16)
                                              ? ::ml_drift::DataType::kFloat16
                                              : ::ml_drift::DataType::kFloat32;

  id<MTLCommandQueue> command_queue = nil;
  LITERT_ASSIGN_OR_RETURN(auto env_options, env.GetOptions());
  auto queue_option = env_options.GetOption(litert::EnvironmentOptions::Tag::kMetalCommandQueue);
  if (queue_option.HasValue()) {
    if (std::holds_alternative<void*>(*queue_option)) {
      command_queue = (__bridge id<MTLCommandQueue>)(std::get<void*>(*queue_option));
    }
  }

  TransformerConfig config = {
      .batch_size = batch_size,
      .sequence_size = sequence_size,
      .vocab_size = vocab_size,
      .max_top_k = sampler_params.k(),
  };

  auto handler = absl::WrapUnique(new TopKMetalSampler(
      std::move(env), std::move(metal_env), std::move(gpu_info), std::move(create_info),
      sampler_params, config, logits_data_type, command_queue));
  ABSL_RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status TopKMetalSampler::InitSampling() {
  GpuModelBuilder::TensorHandle src_logits;
  GpuModelBuilder::TensorHandle constraint_mask_handle;
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle params_f32_handle;
  ABSL_ASSIGN_OR_RETURN(auto gpu_model, CreateSamplingModel(&src_logits, &constraint_mask_handle,
                                                            &params_i32_handle, &params_f32_handle,
                                                            &tokens_ids_handle, logits_data_type_));
  auto create_info_copy = create_info_;

  logits_id_ = src_logits.id;
  logits_tensor_desc_ = src_logits.tensor_desc;

  logits_metal_tensor_ = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(
      CreateTensor(metal_env_->device(), logits_tensor_desc_, logits_metal_tensor_.get()));
  create_info_copy.external_mutable_tensors[logits_id_] = logits_tensor_desc_;

  staging_logits_buffer_ =
      [metal_env_->device() newBufferWithLength:logits_metal_tensor_->GetMemorySizeInBytes()
                                        options:MTLResourceStorageModeShared];

  constraint_mask_ = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(metal_env_->device(), constraint_mask_handle.tensor_desc,
                                    constraint_mask_.get()));
  std::vector<uint8_t> mask_data(constraint_mask_->GetMemorySizeInBytes(), 0);
  ABSL_RETURN_IF_ERROR(constraint_mask_->WriteData(
      command_queue_, mask_data.data(), /*wait_for_completion=*/true));
  create_info_copy.external_immutable_tensors[constraint_mask_handle.id] = constraint_mask_.get();

  tokens_ids_ = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(
      CreateTensor(metal_env_->device(), GetTokensTensorDescriptor(), tokens_ids_.get()));

  staging_ids_buffer_ =
      [metal_env_->device() newBufferWithLength:tokens_ids_->GetMemorySizeInBytes()
                                        options:MTLResourceStorageModeShared];

  text_params_.params_i32 = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(metal_env_->device(), GetParamsTensorDescriptor(),
                                    text_params_.params_i32.get()));
  params_f32_ = std::make_unique<MetalSpatialTensor>();
  ABSL_RETURN_IF_ERROR(
      CreateTensor(metal_env_->device(), params_f32_handle.tensor_desc, params_f32_.get()));

  create_info_copy.external_immutable_tensors[params_i32_handle.id] = text_params_.params_i32.get();
  create_info_copy.external_immutable_tensors[params_f32_handle.id] = params_f32_.get();
  create_info_copy.external_immutable_tensors[tokens_ids_handle.id] = tokens_ids_.get();

  sampling_ = std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(sampling_->InitFromGpuModel(create_info_copy, &gpu_model, metal_env_.get()));

  return InitHelperOps(metal_env_.get());
}

absl::Status TopKMetalSampler::InitHelperOps(MetalEnv* env) {
  if (text_params_.params_i32) {
    text_params_.write_i32_params = std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(
        InitComputeTask(env,
                        CreateWriteParamsOp(text_params_.params_i32->GetDescriptor(),
                                            text_params_.params_i32->Channels()),
                        text_params_.write_i32_params.get()));
  }
  if (params_f32_) {
    write_f32_params_ = std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(InitComputeTask(
        env, CreateWriteParamsOp(params_f32_->GetDescriptor(), params_f32_->Channels()),
        write_f32_params_.get()));
  }
  return absl::OkStatus();
}

absl::Status TopKMetalSampler::ExecuteUpdateIntParams(id<MTLCommandBuffer> command_buffer,
                                                      TransformerParams& params,
                                                      const LlmRuntimeParams& param_vals) {
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  ABSL_RETURN_IF_ERROR(params.write_i32_params->SetDstTensor(params.params_i32.get(), 0));
  ABSL_RETURN_IF_ERROR(params.write_i32_params->SetInt(
      absl::StrCat("param", LlmRuntimeParams::kTopKIndex), param_vals.topk));
  ABSL_RETURN_IF_ERROR(params.write_i32_params->UpdateParams());

  params.write_i32_params->Encode(encoder);
  [encoder endEncoding];
  return absl::OkStatus();
}

absl::Status TopKMetalSampler::ExecuteUpdateParams(id<MTLCommandBuffer> command_buffer,
                                                   MetalSpatialTensor* tensor,
                                                   const std::vector<float>& params) {
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  ABSL_RETURN_IF_ERROR(write_f32_params_->SetDstTensor(tensor, 0));
  for (int i = 0; i < params.size(); ++i) {
    const std::string param_name = absl::StrCat("param", i);
    ABSL_RETURN_IF_ERROR(write_f32_params_->SetFloat(param_name, params[i]));
  }
  ABSL_RETURN_IF_ERROR(write_f32_params_->UpdateParams());
  write_f32_params_->Encode(encoder);
  [encoder endEncoding];
  return absl::OkStatus();
}

absl::Status TopKMetalSampler::SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                                        TensorBuffer& ids_tensor,
                                                        TensorBuffer* scores_tensor) {
  RET_CHECK_EQ(scores_tensor, nullptr) << "This backend does not support scoring for now.";

  // Try direct Metal buffer access to avoid CPU roundtrip
  MetalSpatialTensor shared_logits_tensor;
  bool use_zero_copy = false;
  auto metal_buffer_handle = logits_tensor.GetMetalBuffer();
  if (metal_buffer_handle.HasValue()) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metal_buffer_handle.Value();
    if (absl::Status status = ::ml_drift::metal::CreateTensorSharedBuffer(
            buffer, logits_tensor_desc_, &shared_logits_tensor);
        status.ok()) {
      ABSL_RETURN_IF_ERROR(sampling_->SetTensor(logits_id_, &shared_logits_tensor));
      use_zero_copy = true;
    }
  }

  @autoreleasepool {
    // Use persistent command queue
    id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];

    if (logits_tensor.IsMetalMemory() && logits_tensor.HasEvent()) {
      LITERT_ASSIGN_OR_RETURN(auto event, logits_tensor.GetEvent());
      LITERT_ASSIGN_OR_RETURN(void* native_event, event.GetCustomNativeEvent());
      static_cast<litert::ml_drift::CustomEventMetal*>(native_event)->EncodeWait(command_buffer);
    }

    // Fallback to copy if zero-copy is not possible
    if (!use_zero_copy) {
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          TensorBufferScopedLock::Create(logits_tensor, TensorBuffer::LockMode::kRead));
      const void* logits_data = lock_and_addr.second;
      memcpy([staging_logits_buffer_ contents], logits_data,
             logits_metal_tensor_->GetMemorySizeInBytes());

      id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];
      [blit_encoder copyFromBuffer:staging_logits_buffer_
                      sourceOffset:0
                          toBuffer:logits_metal_tensor_->GetBufferHandle()
                 destinationOffset:0
                              size:logits_metal_tensor_->GetMemorySizeInBytes()];
      [blit_encoder endEncoding];

      ABSL_RETURN_IF_ERROR(sampling_->SetTensor(logits_id_, logits_metal_tensor_.get()));
    }

    // Update params
    ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(command_buffer, text_params_,
                                                CreateLlmRuntimeParams(sampler_params_)));
    ABSL_RETURN_IF_ERROR(ExecuteUpdateParams(
        command_buffer, params_f32_.get(),
        CreateFloatParams(sampler_params_, rand_gen_, params_f32_->Channels())));

    // Execute sampling
    sampling_->EncodeWithCommandBuffer(command_buffer);

    // Copy results to staging buffer
    id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:tokens_ids_->GetBufferHandle()
                    sourceOffset:0
                        toBuffer:staging_ids_buffer_
               destinationOffset:0
                            size:tokens_ids_->GetMemorySizeInBytes()];
    [blit_encoder endEncoding];

    // Encode next input handling if available.
    litert::ml_drift::CustomEventMetal* sampling_done_event = nullptr;
    if (input_handling_ != nullptr) {
      sampling_done_event = new litert::ml_drift::CustomEventMetal(metal_env_->device());
      sampling_done_event->EncodeSignal(command_buffer);

      input_handling_->EncodeWithCommandBuffer(command_buffer);
      if (run_inference_func_ != nullptr) {
        // Associate input_handling_done_event to next_ids_tensor, i.e. to signal the event to the
        // beginning of next decode step when next_ids_tensor_ is ready to read.
        auto* input_handling_done_event =
            new litert::ml_drift::CustomEventMetal(metal_env_->device());
        input_handling_done_event->EncodeSignal(command_buffer);
        LITERT_ASSIGN_OR_RETURN(auto litert_event, litert::Event::CreateManaged(
                                                       litert_env_, litert::Event::Type::kCustom));
        // Pass the ownership of input_handling_done_event to litert_event.
        litert_event.SetCustomEvent(input_handling_done_event);
        input_handling_done_event->Release(input_handling_done_event);  // Decrement the refcount.
        LITERT_RETURN_IF_ERROR(next_ids_tensor_.has_value());
        next_ids_tensor_->SetEvent(std::move(litert_event));
      }
    }

    [command_buffer commit];

    if (input_handling_ != nullptr) {
      if (run_inference_func_ != nullptr) {
        int ret = run_inference_func_(run_inference_arg_);
        if (ret != 0) {
          return absl::Status(static_cast<absl::StatusCode>(ret), "Failed to run inference.");
        }
      }
      // Wait for sampling complete before reading back results
      sampling_done_event->Wait(sampling_done_event, /*timeout_in_ms=*/-1);
      sampling_done_event->Release(sampling_done_event);  // Decrement the refcount.
    } else {
      // Wait for all GPU work to complete before reading back results.
      [command_buffer waitUntilCompleted];
    }
  }

  // Download results
  LITERT_ASSIGN_OR_RETURN(auto lock_and_addr, TensorBufferScopedLock::Create(
                                                  ids_tensor, TensorBuffer::LockMode::kWrite));
  auto* ids_data = lock_and_addr.second;
  ABSL_RETURN_IF_ERROR(DownloadSampledIds(ids_data));

  return absl::OkStatus();
}

absl::Status TopKMetalSampler::DownloadSampledIds(void* dst) {
  if (!tokens_ids_) {
    return absl::InternalError("tokens_ids_ tensor is not initialized.");
  }
  if (!staging_ids_buffer_) {
    return absl::InternalError("staging_ids_buffer_ is not initialized.");
  }
  memcpy(dst, [staging_ids_buffer_ contents], tokens_ids_->GetMemorySizeInBytes());
  return absl::OkStatus();
}

absl::Status TopKMetalSampler::UpdateConfig(const proto::SamplerParameters& sampler_params,
                                            int batch_size,
                                            std::shared_ptr<std::default_random_engine> rand_gen) {
  ABSL_RETURN_IF_ERROR(CheckTopKWithinMax(sampler_params));
  sampler_params_ = sampler_params;
  if (rand_gen != nullptr) {
    rand_gen_ = rand_gen;
  }
  config_.batch_size = batch_size;
  return absl::OkStatus();
}

bool TopKMetalSampler::CanHandleInput() const { return true; }

bool TopKMetalSampler::HandlesInput() const {
  return input_handling_ != nullptr && run_inference_func_ != nullptr;
}

absl::Status TopKMetalSampler::SetInferenceFuncAndInputTensors(
    int (*run_inference_func)(void* arg), void* arg, const TensorBuffer* ids_tensor,
    const TensorBuffer* prev_input_positions_tensor, const TensorBuffer* input_positions_tensor,
    const TensorBuffer* prev_mask_tensor, const TensorBuffer* mask_tensor,
    const TensorBuffer* prev_param_tensor, const TensorBuffer* param_tensor) {
  if (run_inference_func == nullptr) {
    run_inference_func_ = nullptr;
    run_inference_arg_ = nullptr;
    next_ids_tensor_.reset();
    return absl::OkStatus();
  }

  LITERT_RETURN_IF_ERROR(ids_tensor != nullptr);
  next_ids_tensor_ = TensorBuffer::WrapCObject(
      litert_env_.GetHolder(), ids_tensor->Get(), litert::OwnHandle::kNo);
  LITERT_RETURN_IF_ERROR(prev_input_positions_tensor != nullptr);
  LITERT_RETURN_IF_ERROR(input_positions_tensor != nullptr);
  if (mask_tensor == nullptr) {
    LITERT_RETURN_IF_ERROR(prev_mask_tensor == nullptr);
  } else {
    LITERT_RETURN_IF_ERROR(prev_mask_tensor != nullptr);
  }
  if (param_tensor == nullptr) {
    LITERT_RETURN_IF_ERROR(prev_param_tensor == nullptr);
  } else {
    LITERT_RETURN_IF_ERROR(prev_param_tensor != nullptr);
  }

  if (input_handling_ == nullptr) {
    ::ml_drift::GpuModelBuilder::TensorHandle last_output_token_handle;
    ::ml_drift::GpuModelBuilder::TensorHandle input_token_handle;
    ::ml_drift::GpuModelBuilder::TensorHandle prev_input_position_handle;
    ::ml_drift::GpuModelBuilder::TensorHandle input_position_handle;
    ::ml_drift::GpuModelBuilder::TensorHandle prev_mask_handle;
    ::ml_drift::GpuModelBuilder::TensorHandle mask_handle;
    ::ml_drift::GpuModelBuilder::TensorHandle prev_param_handle;
    ::ml_drift::GpuModelBuilder::TensorHandle param_handle;
    ABSL_ASSIGN_OR_RETURN(
        auto input_handling_model,
        CreateInputHandlingModel(
            tokens_ids_->GetDescriptor(), GetTensorDescriptor(ids_tensor),
            GetTensorDescriptor(input_positions_tensor), GetTensorDescriptor(mask_tensor),
            GetTensorDescriptor(param_tensor), &last_output_token_handle, &input_token_handle,
            &prev_input_position_handle, &input_position_handle,
            mask_tensor ? &prev_mask_handle : nullptr, mask_tensor ? &mask_handle : nullptr,
            param_tensor ? &prev_param_handle : nullptr, param_tensor ? &param_handle : nullptr));

    ::ml_drift::CreateGpuModelInfo input_handling_create_info = create_info_;
    input_handling_create_info.external_immutable_tensors.clear();
    input_handling_create_info.external_mutable_tensors.clear();
    input_handling_create_info.external_mutable_tensors[last_output_token_handle.id] =
        last_output_token_handle.tensor_desc;
    input_handling_create_info.external_mutable_tensors[input_token_handle.id] =
        input_token_handle.tensor_desc;
    input_handling_create_info.external_mutable_tensors[prev_input_position_handle.id] =
        prev_input_position_handle.tensor_desc;
    input_handling_create_info.external_mutable_tensors[input_position_handle.id] =
        input_position_handle.tensor_desc;
    if (mask_tensor) {
      input_handling_create_info.external_mutable_tensors[prev_mask_handle.id] =
          prev_mask_handle.tensor_desc;
      input_handling_create_info.external_mutable_tensors[mask_handle.id] = mask_handle.tensor_desc;
    }
    if (param_tensor) {
      input_handling_create_info.external_mutable_tensors[prev_param_handle.id] =
          prev_param_handle.tensor_desc;
      input_handling_create_info.external_mutable_tensors[param_handle.id] =
          param_handle.tensor_desc;
    }

    auto input_handling = std::make_unique<::ml_drift::metal::InferenceContext>(
        ::ml_drift::metal::InferenceContext());
    ABSL_RETURN_IF_ERROR(input_handling->InitFromGpuModel(input_handling_create_info,
                                                          &input_handling_model, metal_env_.get()));
    input_handling_ = std::move(input_handling);
    input_handling_ids_ = {last_output_token_handle.id, input_token_handle.id,
                           prev_input_position_handle.id, input_position_handle.id};
    if (mask_tensor) {
      input_handling_ids_.push_back(prev_mask_handle.id);
      input_handling_ids_.push_back(mask_handle.id);
    }
    if (param_tensor) {
      input_handling_ids_.push_back(prev_param_handle.id);
      input_handling_ids_.push_back(param_handle.id);
    }
  }

  shared_tensors_.clear();
  shared_tensors_.reserve(input_handling_ids_.size());
  ABSL_RETURN_IF_ERROR(input_handling_->SetTensor(input_handling_ids_[0], tokens_ids_.get()));
  ABSL_RETURN_IF_ERROR(
      BindTensor(*ids_tensor, input_handling_ids_[1], *input_handling_, shared_tensors_));
  ABSL_RETURN_IF_ERROR(BindTensor(*prev_input_positions_tensor, input_handling_ids_[2],
                                  *input_handling_, shared_tensors_));
  ABSL_RETURN_IF_ERROR(BindTensor(*input_positions_tensor, input_handling_ids_[3], *input_handling_,
                                  shared_tensors_));
  size_t idx = 4;
  if (mask_tensor != nullptr) {
    ABSL_RETURN_IF_ERROR(
        BindTensor(*prev_mask_tensor, input_handling_ids_[idx], *input_handling_, shared_tensors_));
    ABSL_RETURN_IF_ERROR(
        BindTensor(*mask_tensor, input_handling_ids_[idx + 1], *input_handling_, shared_tensors_));
    idx += 2;
  }
  if (param_tensor != nullptr) {
    ABSL_RETURN_IF_ERROR(BindTensor(*prev_param_tensor, input_handling_ids_[idx], *input_handling_,
                                    shared_tensors_));
    ABSL_RETURN_IF_ERROR(
        BindTensor(*param_tensor, input_handling_ids_[idx + 1], *input_handling_, shared_tensors_));
  }
  run_inference_func_ = run_inference_func;
  run_inference_arg_ = arg;

  return absl::OkStatus();
}

}  // namespace litert::lm

// C API implementations for TopKMetalSampler.
int LiteRtTopKMetalSampler_Create(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env, int batch_size,
    int sequence_size, int vocab_size,
    const LiteRtTopKMetalSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKMetalSampler_SamplerParameters* sampler_params,
    LiteRtTopKMetalSampler_Sampler** sampler_out, char** error_msg) {
  if (!runtime_c_api) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Input LiteRT C API not provided."), error_msg);
  }
  if (!env) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Input LiteRT Environment not provided."), error_msg);
  }
  if (!sampler_out) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Output sampler pointer not provided."), error_msg);
  }
  auto cpp_env = litert::Environment::WrapCObject(runtime_c_api, env, litert::OwnHandle::kNo);
  litert::lm::proto::SamplerParameters cpp_sampler_params;
  if (sampler_params) {
    cpp_sampler_params =
        *reinterpret_cast<const litert::lm::proto::SamplerParameters*>(sampler_params);
  }
  std::optional<litert::lm::ActivationDataType> cpp_activation_data_type =
      activation_data_type
          ? std::make_optional(
                *reinterpret_cast<const litert::lm::ActivationDataType*>(activation_data_type))
          : std::nullopt;

  auto sampler = litert::lm::TopKMetalSampler::Create(
      cpp_env, batch_size, sequence_size, vocab_size, std::move(cpp_activation_data_type),
      std::move(cpp_sampler_params));
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(sampler.status(), error_msg);
  *sampler_out = reinterpret_cast<LiteRtTopKMetalSampler_Sampler*>(sampler->release());
  return 0;
}

void LiteRtTopKMetalSampler_Destroy(LiteRtTopKMetalSampler_Sampler* sampler) {
  delete reinterpret_cast<litert::lm::TopKMetalSampler*>(sampler);
}

int LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer(LiteRtTopKMetalSampler_Sampler* sampler,
                                                    LiteRtTensorBuffer logits_tensor,
                                                    LiteRtTensorBuffer ids_tensor,
                                                    const LiteRtTensorBuffer* scores_tensor,
                                                    char** error_msg) {
  if (!sampler) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(absl::InvalidArgumentError("Sampler not provided."),
                                           error_msg);
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKMetalSampler*>(sampler);
  if (!logits_tensor) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Input logits tensor not provided."), error_msg);
  }
  if (!ids_tensor) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Input ids tensor not provided."), error_msg);
  }
  auto env_holder = cpp_sampler->GetEnvironment().GetHolder();
  auto cpp_logits_tensor =
      litert::TensorBuffer::WrapCObject(env_holder, logits_tensor, litert::OwnHandle::kNo);
  auto cpp_ids_tensor =
      litert::TensorBuffer::WrapCObject(env_holder, ids_tensor, litert::OwnHandle::kNo);
  std::optional<litert::TensorBuffer> cpp_scores_tensor;
  if (scores_tensor) {
    cpp_scores_tensor =
        litert::TensorBuffer::WrapCObject(env_holder, *scores_tensor, litert::OwnHandle::kNo);
  }
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
      cpp_sampler->SampleToIdAndScoreBuffer(cpp_logits_tensor, cpp_ids_tensor,
                                            cpp_scores_tensor ? &*cpp_scores_tensor : nullptr),
      error_msg);
  return 0;
}

int LiteRtTopKMetalSampler_UpdateConfig(
    LiteRtTopKMetalSampler_Sampler* sampler,
    const LiteRtTopKMetalSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg) {
  if (!sampler) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(absl::InvalidArgumentError("Sampler not provided."),
                                           error_msg);
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKMetalSampler*>(sampler);
  if (!sampler_params) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Sampler params not provided."), error_msg);
  }
  auto cpp_sampler_params =
      *reinterpret_cast<const litert::lm::proto::SamplerParameters*>(sampler_params);
  std::shared_ptr<std::default_random_engine> cpp_rand_gen;
  if (rand_gen_shared_ptr) {
    cpp_rand_gen =
        *reinterpret_cast<std::shared_ptr<std::default_random_engine>*>(rand_gen_shared_ptr);
  }
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
      cpp_sampler->UpdateConfig(cpp_sampler_params, batch_size, cpp_rand_gen), error_msg);
  return 0;
}

int LiteRtTopKMetalSampler_CanHandleInput(LiteRtTopKMetalSampler_Sampler* sampler) {
  if (!sampler) {
    return false;
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKMetalSampler*>(sampler);
  return cpp_sampler->CanHandleInput();
}

int LiteRtTopKMetalSampler_HandlesInput(LiteRtTopKMetalSampler_Sampler* sampler) {
  if (!sampler) {
    return false;
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKMetalSampler*>(sampler);
  return cpp_sampler->HandlesInput();
}

int LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors(
    LiteRtTopKMetalSampler_Sampler* sampler, int (*run_inference_func)(void* arg), void* arg,
    LiteRtTensorBuffer ids_tensor, LiteRtTensorBuffer prev_input_positions_tensor,
    LiteRtTensorBuffer input_positions_tensor, LiteRtTensorBuffer prev_mask_tensor,
    LiteRtTensorBuffer mask_tensor, LiteRtTensorBuffer prev_param_tensor,
    LiteRtTensorBuffer param_tensor, char** error_msg) {
  if (!sampler) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(absl::InvalidArgumentError("Sampler not provided."),
                                           error_msg);
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKMetalSampler*>(sampler);
  auto env_holder = cpp_sampler->GetEnvironment().GetHolder();
  std::optional<litert::TensorBuffer> cpp_ids_tensor;
  if (ids_tensor) {
    cpp_ids_tensor =
        litert::TensorBuffer::WrapCObject(env_holder, ids_tensor, litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_prev_input_positions_tensor;
  if (prev_input_positions_tensor) {
    cpp_prev_input_positions_tensor = litert::TensorBuffer::WrapCObject(
        env_holder, prev_input_positions_tensor, litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_input_positions_tensor;
  if (input_positions_tensor) {
    cpp_input_positions_tensor = litert::TensorBuffer::WrapCObject(
        env_holder, input_positions_tensor, litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_prev_mask_tensor;
  if (prev_mask_tensor) {
    cpp_prev_mask_tensor =
        litert::TensorBuffer::WrapCObject(env_holder, prev_mask_tensor, litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_mask_tensor;
  if (mask_tensor) {
    cpp_mask_tensor =
        litert::TensorBuffer::WrapCObject(env_holder, mask_tensor, litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_prev_param_tensor;
  if (prev_param_tensor) {
    cpp_prev_param_tensor =
        litert::TensorBuffer::WrapCObject(env_holder, prev_param_tensor, litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_param_tensor;
  if (param_tensor) {
    cpp_param_tensor =
        litert::TensorBuffer::WrapCObject(env_holder, param_tensor, litert::OwnHandle::kNo);
  }
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
      cpp_sampler->SetInferenceFuncAndInputTensors(
          run_inference_func, arg, cpp_ids_tensor ? &*cpp_ids_tensor : nullptr,
          cpp_prev_input_positions_tensor ? &*cpp_prev_input_positions_tensor : nullptr,
          cpp_input_positions_tensor ? &*cpp_input_positions_tensor : nullptr,
          cpp_prev_mask_tensor ? &*cpp_prev_mask_tensor : nullptr,
          cpp_mask_tensor ? &*cpp_mask_tensor : nullptr,
          cpp_prev_param_tensor ? &*cpp_prev_param_tensor : nullptr,
          cpp_param_tensor ? &*cpp_param_tensor : nullptr),
      error_msg);
  return 0;
}

#if defined(LITERT_LM_USE_STATIC_LINKED_GPU_SAMPLER)
// Function pointers defined in sampler_factory.cc.
extern "C" int (*LiteRtTopKMetalSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env, int batch_size,
    int sequence_size, int vocab_size,
    const LiteRtTopKMetalSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKMetalSampler_SamplerParameters* sampler_params,
    LiteRtTopKMetalSampler_Sampler** sampler_out, char** error_msg);

extern "C" void (*LiteRtTopKMetalSampler_Destroy_Static)(LiteRtTopKMetalSampler_Sampler* sampler);

extern "C" int (*LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKMetalSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor, char** error_msg);

extern "C" int (*LiteRtTopKMetalSampler_UpdateConfig_Static)(
    LiteRtTopKMetalSampler_Sampler* sampler,
    const LiteRtTopKMetalSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg);

extern "C" int (*LiteRtTopKMetalSampler_CanHandleInput_Static)(
    LiteRtTopKMetalSampler_Sampler* sampler);

extern "C" int (*LiteRtTopKMetalSampler_HandlesInput_Static)(
    LiteRtTopKMetalSampler_Sampler* sampler);

extern "C" int (*LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors_Static)(
    LiteRtTopKMetalSampler_Sampler* sampler, int (*run_inference_func)(void* arg), void* arg,
    LiteRtTensorBuffer ids_tensor, LiteRtTensorBuffer prev_input_positions_tensor,
    LiteRtTensorBuffer input_positions_tensor, LiteRtTensorBuffer prev_mask_tensor,
    LiteRtTensorBuffer mask_tensor, LiteRtTensorBuffer prev_param_tensor,
    LiteRtTensorBuffer param_tensor, char** error_msg);

namespace {

// Used for C API static linking.
// If defined, will statically link the C API functions at initialization.
class StaticMetalSamplerInitializer {
 public:
  StaticMetalSamplerInitializer() {
    LiteRtTopKMetalSampler_Create_Static = &LiteRtTopKMetalSampler_Create;
    LiteRtTopKMetalSampler_Destroy_Static = &LiteRtTopKMetalSampler_Destroy;
    LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer_Static =
        &LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer;
    LiteRtTopKMetalSampler_UpdateConfig_Static = &LiteRtTopKMetalSampler_UpdateConfig;
    LiteRtTopKMetalSampler_CanHandleInput_Static = &LiteRtTopKMetalSampler_CanHandleInput;
    LiteRtTopKMetalSampler_HandlesInput_Static = &LiteRtTopKMetalSampler_HandlesInput;
    LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors_Static =
        &LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors;
  }
};

static StaticMetalSamplerInitializer static_metal_sampler_initializer;

}  // namespace
#endif  // LITERT_LM_USE_STATIC_LINKED_GPU_SAMPLER
