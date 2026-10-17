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

#include "runtime/components/top_k_opencl_sampler.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_device.h"  // from @ml_drift
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
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "litert/cc/internal/litert_handle.h"  // from @litert
#include "litert/cc/litert_any.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"  // from @litert

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
using ::ml_drift::Layout;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::TensorToGrid;
using ::ml_drift::cl::CLCommandQueue;
using ::ml_drift::cl::CLCommandQueueOptions;
using ::ml_drift::cl::CLContext;
using ::ml_drift::cl::CLDevice;
using ::ml_drift::cl::CLErrorCodeToString;
using ::ml_drift::cl::ClOperation;
using ::ml_drift::cl::CreateTensorShared;
using ::ml_drift::cl::GetFastestStorageType;
using ::ml_drift::cl::InferenceContext;
using ::ml_drift::cl::ProfilingCommandQueue;
using ::ml_drift::cl::Tensor;
using ClEnv = ::ml_drift::cl::Environment;

absl::Status InitClOp(ClEnv* env, std::unique_ptr<GPUOperation>&& gpu_op,
                      ClOperation* cl_op) {
  ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(env->device().GetInfo()));

  cl_op->Init(std::move(gpu_op));
  ABSL_RETURN_IF_ERROR(cl_op->Compile(env->GetDevicePtr(), &env->context(),
                                      env->program_cache()));
  return absl::OkStatus();
}

absl::Status InitClOp(ClEnv* env, GPUOperation&& operation,
                      ClOperation* cl_op) {
  return InitClOp(env, std::make_unique<GPUOperation>(std::move(operation)),
                  cl_op);
}

GPUOperation CreateWriteParamsOp(const TensorDescriptor& dst,
                                 int params_count) {
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
  for (int s = 0; s < ml_drift::DivideRoundUp(params_count, 4); ++s) {
    for (int ch = 0; ch < 4; ++ch) {
      std::string param_name = absl::StrCat("args.param", s * 4 + ch);
      if (s * 4 + ch >= params_count) {
        param_name = "args.zero_value";
      }
      c += "  result" + postfixes[ch] + " = " + param_name + ";\n";
    }
    c += "  args.dst.Write(result, 0, 0, " + std::to_string(s) + ");\n";
  }
  c += "}\n";
  op.code_ = std::move(c);
  return op;
}

absl::StatusOr<std::unique_ptr<ClEnv>> CreateClEnvFromLiteRtEnv(
    const Environment& env) {
  LITERT_ASSIGN_OR_RETURN(auto env_options, env.GetOptions());

  LITERT_ASSIGN_OR_RETURN(
      auto opencl_device_id_any,
      env_options.GetOption(EnvironmentOptions::Tag::kOpenClDeviceId));
  auto opencl_device_id = reinterpret_cast<cl_device_id>(
      std::get<int64_t>(litert::LiteRtVariant(opencl_device_id_any)));
  LITERT_ASSIGN_OR_RETURN(
      auto opencl_platform_id_any,
      env_options.GetOption(EnvironmentOptions::Tag::kOpenClPlatformId));
  auto opencl_platform_id = reinterpret_cast<cl_platform_id>(
      std::get<int64_t>(litert::LiteRtVariant(opencl_platform_id_any)));
  CLDevice cl_device(opencl_device_id, opencl_platform_id);

  LITERT_ASSIGN_OR_RETURN(
      auto opencl_context_any,
      env_options.GetOption(EnvironmentOptions::Tag::kOpenClContext));
  auto opencl_context = reinterpret_cast<cl_context>(
      std::get<int64_t>(std::any_cast<LiteRtVariant>(opencl_context_any)));
  CLContext cl_context(opencl_context, /*has_ownership=*/false);

  LITERT_ASSIGN_OR_RETURN(
      auto opencl_command_queue_any,
      env_options.GetOption(EnvironmentOptions::Tag::kOpenClCommandQueue));
  auto opencl_command_queue = reinterpret_cast<cl_command_queue>(
      std::get<int64_t>(litert::LiteRtVariant(opencl_command_queue_any)));
  CLCommandQueue cl_command_queue(opencl_command_queue,
                                  /*has_ownership=*/false);

  ProfilingCommandQueue profiling_command_queue;
  ABSL_RETURN_IF_ERROR(CreateProfilingCommandQueue(cl_device, cl_context,
                                                   &profiling_command_queue));

  auto cl_env = std::make_unique<ClEnv>(
      std::move(cl_device), std::move(cl_context), std::move(cl_command_queue),
      std::move(profiling_command_queue));
  ABSL_RETURN_IF_ERROR(cl_env->Init());

  return cl_env;
}

TensorDescriptor GetTensorDescriptor(const TensorBuffer* tensor_buffer) {
  if (!tensor_buffer) {
    return TensorDescriptor();
  }
  auto tensor_type_status = tensor_buffer->TensorType();
  if (!tensor_type_status.HasValue()) {
    return TensorDescriptor();
  }
  const auto& tensor_type = *tensor_type_status;
  DataType type;
  if (tensor_type.ElementType() == litert::ElementType::Float32) {
    type = DataType::kFloat32;
  } else if (tensor_type.ElementType() == litert::ElementType::Float16) {
    type = DataType::kFloat16;
  } else if (tensor_type.ElementType() == litert::ElementType::Int32) {
    type = DataType::kInt32;
  } else if (tensor_type.ElementType() == litert::ElementType::Bool) {
    type = DataType::kBool;
  } else {
    ABSL_LOG(ERROR) << "Unsupported tensor type: "
                    << static_cast<int>(tensor_type.ElementType());
    return TensorDescriptor();
  }
  auto dims = tensor_type.Layout().Dimensions();
  BHWC shape(dims.size() > 3 ? dims[dims.size() - 4] : 1,
             dims.size() > 2 ? dims[dims.size() - 3] : 1,
             dims.size() > 1 ? dims[dims.size() - 2] : 1,
             !dims.empty() ? dims[dims.size() - 1] : 1);
  TensorDescriptor td(type, TensorStorageType::kBuffer,
                      shape.b == 1 ? Layout::kHWC : Layout::kBHWC);
  td.SetBHWCShape(shape);
  return td;
}

absl::Status BindTensor(const TensorBuffer& tensor_buffer,
                        const CLContext& context, ml_drift::ValueId tensor_id,
                        InferenceContext& inference_context,
                        std::vector<Tensor>& shared_tensors) {
  LITERT_ASSIGN_OR_RETURN(cl_mem cl_mem_handle,
                          tensor_buffer.GetOpenClMemory());
  shared_tensors.push_back(Tensor());
  auto& shared_tensor = shared_tensors.back();
  ABSL_RETURN_IF_ERROR(CreateTensorShared(context, cl_mem_handle,
                                          GetTensorDescriptor(&tensor_buffer),
                                          &shared_tensor));
  return inference_context.SetTensor(tensor_id, &shared_tensor);
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<TopKOpenClSampler>> TopKOpenClSampler::Create(
    Environment& env, int batch_size, int sequence_size, int vocab_size,
    std::optional<ActivationDataType> activation_data_type,
    SamplerParameters sampler_params) {
  RET_CHECK_GT(batch_size, 0) << "Batch size must be positive.";
  RET_CHECK_GT(vocab_size, 0) << "Vocabulary size must be positive.";
  ABSL_ASSIGN_OR_RETURN(auto cl_env, CreateClEnvFromLiteRtEnv(env));
  auto gpu_info = cl_env->device().GetInfo();
  ActivationDataType activation_data_type_copy;
  if (activation_data_type.has_value()) {
    activation_data_type_copy = *activation_data_type;
  } else {
    activation_data_type_copy = gpu_info.SupportsFP16()
                                    ? ActivationDataType::FLOAT16
                                    : ActivationDataType::FLOAT32;
  }

  CreateGpuModelInfo create_info;
  if (activation_data_type_copy == ActivationDataType::FLOAT16) {
    create_info.precision = ml_drift::CalculationsPrecision::kF16;
  } else {
    create_info.precision = ml_drift::CalculationsPrecision::kF32;
  }
  create_info.hints.Add(ml_drift::ModelHints::kFastTuning);
  create_info.hints.Add(ml_drift::ModelHints::kPreferTextureWeights);
  create_info.hints.Add(ml_drift::ModelHints::kAllowSpecialKernels);
  create_info.storage_type = GetFastestStorageType(gpu_info);

  ml_drift::DataType logits_data_type =
      (activation_data_type_copy == ActivationDataType::FLOAT16)
          ? ml_drift::DataType::kFloat16
          : ml_drift::DataType::kFloat32;

  auto [runtime_c_api, env_handle] = env.GetHolderForCApi();
  auto litert_env = litert::Environment::WrapCObject(runtime_c_api, env_handle,
                                                     litert::OwnHandle::kNo);
  auto handler = absl::WrapUnique(new TopKOpenClSampler(
      std::move(litert_env), std::move(cl_env), std::move(gpu_info),
      std::move(create_info), sampler_params, batch_size, sequence_size,
      vocab_size, sampler_params.k(), logits_data_type));
  ABSL_RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status TopKOpenClSampler::InitSampling() {
  GpuModelBuilder::TensorHandle src_logits;
  GpuModelBuilder::TensorHandle constraint_mask_handle;
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle params_f32_handle;
  ABSL_ASSIGN_OR_RETURN(
      auto gpu_model,
      CreateSamplingModel(&src_logits, &constraint_mask_handle,
                          &params_i32_handle, &params_f32_handle,
                          &tokens_ids_handle, logits_data_type_));
  auto create_info_copy = create_info_;

  // Register mutable tensor for logits, such that we can provide logits tensor
  // later at inference. This ensures flexibility and interface unification of
  // the sampler across backends.
  logits_id_ = src_logits.id;
  logits_tensor_desc_ = src_logits.tensor_desc;
  create_info_copy.external_mutable_tensors[logits_id_] = logits_tensor_desc_;

  constraint_mask_ = std::make_unique<Tensor>(Tensor());
  ABSL_RETURN_IF_ERROR(CreateTensor(env_->context(),
                                    constraint_mask_handle.tensor_desc,
                                    constraint_mask_.get()));
  // b/404612147: We need to properly initialize the constraint mask tensor with
  // zeros for OpenCL backend.
  std::vector<uint8_t> mask_data(constraint_mask_->GetMemorySizeInBytes(), 0);
  ABSL_RETURN_IF_ERROR(
      constraint_mask_->WriteData(mask_data.data(), env_->queue()));
  create_info_copy.external_immutable_tensors[constraint_mask_handle.id] =
      constraint_mask_.get();

  tokens_ids_ = std::make_unique<Tensor>(Tensor());
  ABSL_RETURN_IF_ERROR(CreateTensor(
      env_->context(), GetTokensTensorDescriptor(), tokens_ids_.get()));

  text_params_.params_i32 = std::make_unique<Tensor>(Tensor());
  ABSL_RETURN_IF_ERROR(CreateTensor(env_->context(),
                                    GetParamsTensorDescriptor(),
                                    text_params_.params_i32.get()));
  params_f32_ = std::make_unique<Tensor>(Tensor());
  ABSL_RETURN_IF_ERROR(CreateTensor(
      env_->context(), params_f32_handle.tensor_desc, params_f32_.get()));

  create_info_copy.external_immutable_tensors[params_i32_handle.id] =
      text_params_.params_i32.get();
  create_info_copy.external_immutable_tensors[params_f32_handle.id] =
      params_f32_.get();
  create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
      tokens_ids_.get();

  sampling_ = std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(
      sampling_->InitFromGpuModel(create_info_copy, &gpu_model, env_.get()));

  ABSL_RETURN_IF_ERROR(InitHelperOps(env_.get()));

  CLCommandQueueOptions queue_options;
  ABSL_RETURN_IF_ERROR(CreateCLCommandQueue(env_->device(), env_->context(),
                                            &reading_queue_, queue_options));

  return absl::OkStatus();
}

absl::Status TopKOpenClSampler::InitHelperOps(ClEnv* env) {
  if (text_params_.params_i32) {
    text_params_.write_i32_params =
        std::make_unique<ClOperation>(ClOperation());
    ABSL_RETURN_IF_ERROR(
        InitClOp(env,
                 CreateWriteParamsOp(text_params_.params_i32->GetDescriptor(),
                                     text_params_.params_i32->Channels()),
                 text_params_.write_i32_params.get()));
  }
  if (params_f32_) {
    write_f32_params_ = std::make_unique<ClOperation>(ClOperation());
    ABSL_RETURN_IF_ERROR(
        InitClOp(env,
                 CreateWriteParamsOp(params_f32_->GetDescriptor(),
                                     params_f32_->Channels()),
                 write_f32_params_.get()));
  }
  return absl::OkStatus();
}

absl::Status TopKOpenClSampler::ExecuteUpdateIntParams(
    ClEnv* env, TransformerParams& params, const LlmRuntimeParams& param_vals) {
  ABSL_RETURN_IF_ERROR(
      params.write_i32_params->SetDstTensor(0, params.params_i32.get()));
  ABSL_RETURN_IF_ERROR(params.write_i32_params->SetInt(
      absl::StrCat("param", LlmRuntimeParams::kTopKIndex), param_vals.topk));
  ABSL_RETURN_IF_ERROR(params.write_i32_params->UpdateParams());
  ABSL_RETURN_IF_ERROR(params.write_i32_params->AddToQueue(env->queue()));
  return absl::OkStatus();
}

absl::Status TopKOpenClSampler::ExecuteUpdateParams(
    ClEnv* env, Tensor* tensor, const std::vector<float>& params) {
  ABSL_RETURN_IF_ERROR(write_f32_params_->SetDstTensor(0, tensor));
  for (int i = 0; i < params.size(); ++i) {
    const std::string param_name = absl::StrCat("param", i);
    ABSL_RETURN_IF_ERROR(write_f32_params_->SetFloat(param_name, params[i]));
  }
  ABSL_RETURN_IF_ERROR(write_f32_params_->UpdateParams());
  ABSL_RETURN_IF_ERROR(write_f32_params_->AddToQueue(env->queue()));
  return absl::OkStatus();
}

absl::Status TopKOpenClSampler::SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
    TensorBuffer* scores_tensor) {
  RET_CHECK_EQ(scores_tensor, nullptr)
      << "This backend does not support scoring for now.";

  if (logits_tensor.HasEvent()) {
    LITERT_ASSIGN_OR_RETURN(auto event, logits_tensor.GetEvent());
    LITERT_ASSIGN_OR_RETURN(auto cl_event, event.GetOpenClEvent());
    if (ml_drift::cl::clEnqueueBarrierWithWaitList(
            env_->queue()->queue(), 1, &cl_event, nullptr) != CL_SUCCESS) {
      return absl::InternalError("Failed to enqueue barrier for logits tensor");
    }
  }

  // Create CL tensor from TensorBuffer, and set the tensor as sampler input.
  LITERT_ASSIGN_OR_RETURN(cl_mem logits_mem, logits_tensor.GetOpenClMemory());
  // This shared tensor can be local since downloading is blocking.
  auto shared_logits_tensor = std::make_unique<Tensor>(Tensor());
  ABSL_RETURN_IF_ERROR(CreateTensorShared(env_->context(), logits_mem,
                                          logits_tensor_desc_,
                                          shared_logits_tensor.get()));
  ABSL_RETURN_IF_ERROR(
      sampling_->SetTensor(logits_id_, shared_logits_tensor.get()));

  ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
      env_.get(), text_params_, CreateLlmRuntimeParams(sampler_params_)));

  ABSL_RETURN_IF_ERROR(ExecuteUpdateParams(
      env_.get(), params_f32_.get(),
      CreateFloatParams(sampler_params_, rand_gen_, params_f32_->Channels())));

  ABSL_RETURN_IF_ERROR(sampling_->AddToQueue(env_->queue(), &sample_event_));

  if (input_handling_ != nullptr) {
    ABSL_RETURN_IF_ERROR(input_handling_->AddToQueue(env_->queue()));
    if (run_inference_func_ != nullptr) {
      int ret = run_inference_func_(run_inference_arg_);
      if (ret != 0) {
        return absl::Status(static_cast<absl::StatusCode>(ret),
                            "Failed to run inference.");
      }
    }
  }

  // Download the sampled ids from GPU to CPU.
  LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                          TensorBufferScopedLock::Create(
                              ids_tensor, TensorBuffer::LockMode::kWrite));
  auto* ids_data = lock_and_addr.second;
  ABSL_RETURN_IF_ERROR(DownloadSampledIds(ids_data));
  return absl::OkStatus();
}

absl::Status TopKOpenClSampler::UpdateConfig(
    const proto::SamplerParameters& sampler_params, int batch_size,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  ABSL_RETURN_IF_ERROR(CheckTopKWithinMax(sampler_params));
  sampler_params_ = sampler_params;
  if (rand_gen != nullptr) {
    rand_gen_ = rand_gen;
  }
  config_.batch_size = batch_size;
  return absl::OkStatus();
}

bool TopKOpenClSampler::CanHandleInput() const { return true; }

bool TopKOpenClSampler::HandlesInput() const {
  return input_handling_ != nullptr && run_inference_func_ != nullptr;
}

absl::Status TopKOpenClSampler::SetInferenceFuncAndInputTensors(
    int (*run_inference_func)(void* arg), void* arg,
    const TensorBuffer* ids_tensor,
    const TensorBuffer* prev_input_positions_tensor,
    const TensorBuffer* input_positions_tensor,
    const TensorBuffer* prev_mask_tensor, const TensorBuffer* mask_tensor,
    const TensorBuffer* prev_param_tensor, const TensorBuffer* param_tensor) {
  if (run_inference_func == nullptr) {
    run_inference_func_ = nullptr;
    run_inference_arg_ = nullptr;
    return absl::OkStatus();
  }

  LITERT_RETURN_IF_ERROR(ids_tensor != nullptr);
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
    GpuModelBuilder::TensorHandle last_output_token_handle;
    GpuModelBuilder::TensorHandle input_token_handle;
    GpuModelBuilder::TensorHandle prev_input_position_handle;
    GpuModelBuilder::TensorHandle input_position_handle;
    GpuModelBuilder::TensorHandle prev_mask_handle;
    GpuModelBuilder::TensorHandle mask_handle;
    GpuModelBuilder::TensorHandle prev_param_handle;
    GpuModelBuilder::TensorHandle param_handle;
    ABSL_ASSIGN_OR_RETURN(
        auto input_handling_model,
        CreateInputHandlingModel(
            tokens_ids_->GetDescriptor(), GetTensorDescriptor(ids_tensor),
            GetTensorDescriptor(input_positions_tensor),
            GetTensorDescriptor(mask_tensor), GetTensorDescriptor(param_tensor),
            &last_output_token_handle, &input_token_handle,
            &prev_input_position_handle, &input_position_handle,
            mask_tensor ? &prev_mask_handle : nullptr,
            mask_tensor ? &mask_handle : nullptr,
            param_tensor ? &prev_param_handle : nullptr,
            param_tensor ? &param_handle : nullptr));

    CreateGpuModelInfo input_handling_create_info = create_info_;
    input_handling_create_info.external_immutable_tensors.clear();
    input_handling_create_info.external_mutable_tensors.clear();
    input_handling_create_info
        .external_mutable_tensors[last_output_token_handle.id] =
        last_output_token_handle.tensor_desc;
    input_handling_create_info.external_mutable_tensors[input_token_handle.id] =
        input_token_handle.tensor_desc;
    input_handling_create_info
        .external_mutable_tensors[prev_input_position_handle.id] =
        prev_input_position_handle.tensor_desc;
    input_handling_create_info
        .external_mutable_tensors[input_position_handle.id] =
        input_position_handle.tensor_desc;
    if (mask_tensor) {
      input_handling_create_info.external_mutable_tensors[prev_mask_handle.id] =
          prev_mask_handle.tensor_desc;
      input_handling_create_info.external_mutable_tensors[mask_handle.id] =
          mask_handle.tensor_desc;
    }
    if (param_tensor) {
      input_handling_create_info
          .external_mutable_tensors[prev_param_handle.id] =
          prev_param_handle.tensor_desc;
      input_handling_create_info.external_mutable_tensors[param_handle.id] =
          param_handle.tensor_desc;
    }

    auto input_handling =
        std::make_unique<InferenceContext>(InferenceContext());
    ABSL_RETURN_IF_ERROR(input_handling->InitFromGpuModel(
        input_handling_create_info, &input_handling_model, env_.get()));
    input_handling_ = std::move(input_handling);
    input_handling_ids_ = {last_output_token_handle.id, input_token_handle.id,
                           prev_input_position_handle.id,
                           input_position_handle.id};
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
  ABSL_RETURN_IF_ERROR(
      input_handling_->SetTensor(input_handling_ids_[0], tokens_ids_.get()));
  ABSL_RETURN_IF_ERROR(BindTensor(*ids_tensor, env_->context(),
                                  input_handling_ids_[1], *input_handling_,
                                  shared_tensors_));
  ABSL_RETURN_IF_ERROR(BindTensor(*prev_input_positions_tensor, env_->context(),
                                  input_handling_ids_[2], *input_handling_,
                                  shared_tensors_));
  ABSL_RETURN_IF_ERROR(BindTensor(*input_positions_tensor, env_->context(),
                                  input_handling_ids_[3], *input_handling_,
                                  shared_tensors_));
  size_t idx = 4;
  if (mask_tensor != nullptr) {
    ABSL_RETURN_IF_ERROR(BindTensor(*prev_mask_tensor, env_->context(),
                                    input_handling_ids_[idx], *input_handling_,
                                    shared_tensors_));
    ABSL_RETURN_IF_ERROR(BindTensor(*mask_tensor, env_->context(),
                                    input_handling_ids_[idx + 1],
                                    *input_handling_, shared_tensors_));
    idx += 2;
  }
  if (param_tensor != nullptr) {
    ABSL_RETURN_IF_ERROR(BindTensor(*prev_param_tensor, env_->context(),
                                    input_handling_ids_[idx], *input_handling_,
                                    shared_tensors_));
    ABSL_RETURN_IF_ERROR(BindTensor(*param_tensor, env_->context(),
                                    input_handling_ids_[idx + 1],
                                    *input_handling_, shared_tensors_));
  }
  run_inference_func_ = run_inference_func;
  run_inference_arg_ = arg;

  return absl::OkStatus();
}

absl::Status TopKOpenClSampler::DownloadSampledIds(void* dst) {
  if (!sample_event_.is_valid()) {
    return absl::FailedPreconditionError(
        "No valid sampling event created before downloading.");
  }
  cl_event event = sample_event_.event();
  cl_command_queue transfer_queue =
      // Using same queue for reading on CLVK to avoid stall.
      // For performance reasons, ml_drift does not flush after queueing
      // kernels when using CLVK.
      // But when using different queues for enqueue and reading, flush is
      // needed after enqueue:
      //
      // To use event objects that refer to commands enqueued in a
      // command-queue as event objects to wait on by commands enqueued in a
      // different command-queue, the application must call a clFlush or any
      // blocking commands that perform an implicit flush of the
      // command-queue where the commands that refer to these event objects
      // are enqueued.
      // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_flush_and_finish
      env_->device().GetInfo().opencl_info.IsCLVK() ? env_->queue()->queue()
                                                    : reading_queue_.queue();

  cl_int error_code = ml_drift::cl::clEnqueueReadBuffer(
      transfer_queue, tokens_ids_->GetMemoryPtr(), CL_TRUE,
      /*time_step=*/0 * config_.batch_size * sizeof(int32_t),
      /*size=*/config_.sequence_size * config_.batch_size * sizeof(int32_t),
      static_cast<int32_t*>(dst),
      /*num_events_in_wait_list=*/1, &event, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to clEnqueueReadBuffer - ",
                                           CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}
}  // namespace litert::lm

// C API implementations for TopKOpenClSampler.
int LiteRtTopKOpenClSampler_Create(
    const struct LiteRtRuntimeCApiStruct* absl_nullable runtime_c_api,
    LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKOpenClSampler_ActivationDataType* absl_nullable
        activation_data_type,
    const LiteRtTopKOpenClSampler_SamplerParameters* absl_nullable
        sampler_params,
    LiteRtTopKOpenClSampler_Sampler** sampler_out,
    char** absl_nullable error_msg) {
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(ml_drift::cl::LoadOpenCL(), error_msg);
  if (!runtime_c_api) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Input LiteRT C API not provided."),
        error_msg);
  }
  if (!env) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Input LiteRT Environment not provided."),
        error_msg);
  }
  if (!sampler_out) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Output sampler pointer not provided."),
        error_msg);
  }
  // Wrap the C env into a C++ Environment object.
  auto cpp_env = litert::Environment::WrapCObject(runtime_c_api, env,
                                                  litert::OwnHandle::kNo);
  litert::lm::proto::SamplerParameters cpp_sampler_params;
  if (sampler_params) {
    // Manually copy the sampler params information.
    // In oss, a direct copy like cpp_sampler_params = *reinterpret_cast<const
    // litert::lm::proto::SamplerParameters*>(sampler_params); doesn't work.
    auto sampler_params_ptr =
        reinterpret_cast<const litert::lm::proto::SamplerParameters*>(
            sampler_params);
    if (!sampler_params_ptr) {
      SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
          absl::InvalidArgumentError("Failed to reinterpret sampler params."),
          error_msg);
    } else {
      cpp_sampler_params.set_k(sampler_params_ptr->k());
      cpp_sampler_params.set_p(sampler_params_ptr->p());
      cpp_sampler_params.set_temperature(sampler_params_ptr->temperature());
      cpp_sampler_params.set_seed(sampler_params_ptr->seed());
      cpp_sampler_params.set_type(sampler_params_ptr->type());
    }
  }
  std::optional<litert::lm::ActivationDataType> cpp_activation_data_type =
      activation_data_type
          ? std::make_optional(
                *reinterpret_cast<const litert::lm::ActivationDataType*>(
                    activation_data_type))
          : std::nullopt;

  auto sampler = litert::lm::TopKOpenClSampler::Create(
      cpp_env, batch_size, sequence_size, vocab_size,
      std::move(cpp_activation_data_type), std::move(cpp_sampler_params));
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(sampler.status(), error_msg);
  *sampler_out =
      reinterpret_cast<LiteRtTopKOpenClSampler_Sampler*>(sampler->release());
  return 0;
}

void LiteRtTopKOpenClSampler_Destroy(LiteRtTopKOpenClSampler_Sampler* sampler) {
  delete reinterpret_cast<litert::lm::TopKOpenClSampler*>(sampler);
}

int LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer(
    LiteRtTopKOpenClSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg) {
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(tflite::gpu::cl::LoadOpenCL(),
                                         error_msg);
  if (!sampler) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Sampler not provided."), error_msg);
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKOpenClSampler*>(sampler);
  if (!logits_tensor) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Input logits tensor not provided."),
        error_msg);
  }
  if (!ids_tensor) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Input ids tensor not provided."),
        error_msg);
  }
  auto env_holder = cpp_sampler->GetEnvironment().GetHolder();
  auto cpp_logits_tensor = litert::TensorBuffer::WrapCObject(
      env_holder, logits_tensor, litert::OwnHandle::kNo);
  auto cpp_ids_tensor = litert::TensorBuffer::WrapCObject(
      env_holder, ids_tensor, litert::OwnHandle::kNo);
  std::optional<litert::TensorBuffer> cpp_scores_tensor;
  if (scores_tensor) {
    cpp_scores_tensor = litert::TensorBuffer::WrapCObject(
        env_holder, *scores_tensor, litert::OwnHandle::kNo);
  }
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
      cpp_sampler->SampleToIdAndScoreBuffer(
          cpp_logits_tensor, cpp_ids_tensor,
          cpp_scores_tensor ? &*cpp_scores_tensor : nullptr),
      error_msg);
  return 0;
}

int LiteRtTopKOpenClSampler_UpdateConfig(
    LiteRtTopKOpenClSampler_Sampler* sampler,
    const LiteRtTopKOpenClSampler_SamplerParameters* sampler_params,
    int batch_size, void* rand_gen_shared_ptr, char** error_msg) {
  if (!sampler) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Sampler not provided."), error_msg);
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKOpenClSampler*>(sampler);
  if (!sampler_params) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Sampler params not provided."), error_msg);
  }
  auto cpp_sampler_params =
      *reinterpret_cast<const litert::lm::proto::SamplerParameters*>(
          sampler_params);
  std::shared_ptr<std::default_random_engine> cpp_rand_gen;
  if (rand_gen_shared_ptr) {
    cpp_rand_gen =
        *reinterpret_cast<std::shared_ptr<std::default_random_engine>*>(
            rand_gen_shared_ptr);
  }
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
      cpp_sampler->UpdateConfig(cpp_sampler_params, batch_size, cpp_rand_gen),
      error_msg);
  return 0;
}

int LiteRtTopKOpenClSampler_CanHandleInput(
    LiteRtTopKOpenClSampler_Sampler* sampler) {
  if (!sampler) {
    return 0;
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKOpenClSampler*>(sampler);
  return cpp_sampler->CanHandleInput() ? 1 : 0;
}

int LiteRtTopKOpenClSampler_HandlesInput(
    LiteRtTopKOpenClSampler_Sampler* sampler) {
  if (!sampler) {
    return 0;
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKOpenClSampler*>(sampler);
  return cpp_sampler->HandlesInput() ? 1 : 0;
}

int LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors(
    LiteRtTopKOpenClSampler_Sampler* sampler,
    int (*run_inference_func)(void* arg), void* arg,
    LiteRtTensorBuffer absl_nullable ids_tensor,
    LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
    LiteRtTensorBuffer absl_nullable input_positions_tensor,
    LiteRtTensorBuffer absl_nullable prev_mask_tensor,
    LiteRtTensorBuffer absl_nullable mask_tensor,
    LiteRtTensorBuffer absl_nullable prev_param_tensor,
    LiteRtTensorBuffer absl_nullable param_tensor,
    char** absl_nullable error_msg) {
  if (!sampler) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Sampler not provided."), error_msg);
  }
  if (run_inference_func) {
    if (!ids_tensor) {
      SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
          absl::InvalidArgumentError("Input ids tensor not provided."),
          error_msg);
    }
    if (!prev_input_positions_tensor) {
      SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
          absl::InvalidArgumentError("Input positions tensor not provided."),
          error_msg);
    }
    if (!input_positions_tensor) {
      SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
          absl::InvalidArgumentError("Input positions tensor not provided."),
          error_msg);
    }
    if (prev_mask_tensor && !mask_tensor) {
      SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
          absl::InvalidArgumentError(
              "Previous mask tensor provided but mask tensor not provided."),
          error_msg);
    }
    if (!prev_mask_tensor && mask_tensor) {
      SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
          absl::InvalidArgumentError(
              "Mask tensor provided but previous mask tensor not provided."),
          error_msg);
    }
    if (prev_param_tensor && !param_tensor) {
      SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
          absl::InvalidArgumentError(
              "Previous param tensor provided but param tensor not provided."),
          error_msg);
    }
    if (!prev_param_tensor && param_tensor) {
      SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
          absl::InvalidArgumentError(
              "Param tensor provided but previous param tensor not provided."),
          error_msg);
    }
  }

  auto cpp_sampler = reinterpret_cast<litert::lm::TopKOpenClSampler*>(sampler);
  auto env_holder = cpp_sampler->GetEnvironment().GetHolder();
  std::optional<litert::TensorBuffer> cpp_ids_tensor;
  if (ids_tensor) {
    cpp_ids_tensor = litert::TensorBuffer::WrapCObject(env_holder, ids_tensor,
                                                       litert::OwnHandle::kNo);
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
    cpp_prev_mask_tensor = litert::TensorBuffer::WrapCObject(
        env_holder, prev_mask_tensor, litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_mask_tensor;
  if (mask_tensor) {
    cpp_mask_tensor = litert::TensorBuffer::WrapCObject(env_holder, mask_tensor,
                                                        litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_prev_param_tensor;
  if (prev_param_tensor) {
    cpp_prev_param_tensor = litert::TensorBuffer::WrapCObject(
        env_holder, prev_param_tensor, litert::OwnHandle::kNo);
  }
  std::optional<litert::TensorBuffer> cpp_param_tensor;
  if (param_tensor) {
    cpp_param_tensor = litert::TensorBuffer::WrapCObject(
        env_holder, param_tensor, litert::OwnHandle::kNo);
  }
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
      cpp_sampler->SetInferenceFuncAndInputTensors(
          run_inference_func, arg, cpp_ids_tensor ? &*cpp_ids_tensor : nullptr,
          cpp_prev_input_positions_tensor ? &*cpp_prev_input_positions_tensor
                                          : nullptr,
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
extern "C" int (*LiteRtTopKOpenClSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env,
    int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKOpenClSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKOpenClSampler_SamplerParameters* sampler_params,
    LiteRtTopKOpenClSampler_Sampler** sampler_out, char** error_msg);

extern "C" void (*LiteRtTopKOpenClSampler_Destroy_Static)(
    LiteRtTopKOpenClSampler_Sampler* sampler);

extern "C" int (*LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKOpenClSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg);

extern "C" int (*LiteRtTopKOpenClSampler_UpdateConfig_Static)(
    LiteRtTopKOpenClSampler_Sampler* sampler,
    const LiteRtTopKOpenClSampler_SamplerParameters* sampler_params,
    int batch_size, void* rand_gen_shared_ptr, char** error_msg);

extern "C" int (*LiteRtTopKOpenClSampler_CanHandleInput_Static)(
    LiteRtTopKOpenClSampler_Sampler* sampler);

extern "C" int (*LiteRtTopKOpenClSampler_HandlesInput_Static)(
    LiteRtTopKOpenClSampler_Sampler* sampler);

extern "C" int (
    *LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static)(
    LiteRtTopKOpenClSampler_Sampler* sampler,
    int (*run_inference_func)(void* arg), void* arg,
    LiteRtTensorBuffer absl_nullable ids_tensor,
    LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
    LiteRtTensorBuffer absl_nullable input_positions_tensor,
    LiteRtTensorBuffer absl_nullable prev_mask_tensor,
    LiteRtTensorBuffer absl_nullable mask_tensor,
    LiteRtTensorBuffer absl_nullable prev_param_tensor,
    LiteRtTensorBuffer absl_nullable param_tensor, char** error_msg);

namespace {

// Used for C API static linking.
// If defined, will statically link the C API functions at initialization.
class StaticGpuSamplerInitializer {
 public:
  StaticGpuSamplerInitializer() {
    LiteRtTopKOpenClSampler_Create_Static =
        [](const struct LiteRtRuntimeCApiStruct* runtime_c_api,
           LiteRtEnvironment env, int batch_size, int sequence_size,
           int vocab_size,
           const LiteRtTopKOpenClSampler_ActivationDataType*
               activation_data_type,
           const LiteRtTopKOpenClSampler_SamplerParameters* sampler_params,
           LiteRtTopKOpenClSampler_Sampler** sampler_out, char** error_msg) {
          return LiteRtTopKOpenClSampler_Create(
              runtime_c_api, env, batch_size, sequence_size, vocab_size,
              activation_data_type, sampler_params, sampler_out, error_msg);
        };
    LiteRtTopKOpenClSampler_Destroy_Static =
        [](LiteRtTopKOpenClSampler_Sampler* sampler) {
          LiteRtTopKOpenClSampler_Destroy(sampler);
        };
    LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static =
        [](LiteRtTopKOpenClSampler_Sampler* sampler,
           LiteRtTensorBuffer logits_tensor, LiteRtTensorBuffer ids_tensor,
           const LiteRtTensorBuffer* scores_tensor, char** error_msg) {
          return LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer(
              sampler, logits_tensor, ids_tensor, scores_tensor, error_msg);
        };
    LiteRtTopKOpenClSampler_UpdateConfig_Static =
        [](LiteRtTopKOpenClSampler_Sampler* sampler,
           const LiteRtTopKOpenClSampler_SamplerParameters* sampler_params,
           int batch_size, void* rand_gen_shared_ptr, char** error_msg) {
          return LiteRtTopKOpenClSampler_UpdateConfig(
              sampler, sampler_params, batch_size, rand_gen_shared_ptr,
              error_msg);
        };
    LiteRtTopKOpenClSampler_CanHandleInput_Static =
        [](LiteRtTopKOpenClSampler_Sampler* sampler) {
          return LiteRtTopKOpenClSampler_CanHandleInput(sampler);
        };
    LiteRtTopKOpenClSampler_HandlesInput_Static =
        [](LiteRtTopKOpenClSampler_Sampler* sampler) {
          return LiteRtTopKOpenClSampler_HandlesInput(sampler);
        };
    LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static =
        [](LiteRtTopKOpenClSampler_Sampler* sampler,
           int (*run_inference_func)(void* arg), void* arg,
           LiteRtTensorBuffer absl_nullable ids_tensor,
           LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
           LiteRtTensorBuffer absl_nullable input_positions_tensor,
           LiteRtTensorBuffer absl_nullable prev_mask_tensor,
           LiteRtTensorBuffer absl_nullable mask_tensor,
           LiteRtTensorBuffer absl_nullable prev_param_tensor,
           LiteRtTensorBuffer absl_nullable param_tensor, char** error_msg) {
          return LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors(
              sampler, run_inference_func, arg, ids_tensor,
              prev_input_positions_tensor, input_positions_tensor,
              prev_mask_tensor, mask_tensor, prev_param_tensor, param_tensor,
              error_msg);
        };
  }
};

// Statically initialize and link the C API functions pointers.
StaticGpuSamplerInitializer static_gpu_sampler_initializer;

}  // namespace

#endif  // defined(LITERT_LM_USE_STATIC_LINKED_GPU_SAMPLER)
