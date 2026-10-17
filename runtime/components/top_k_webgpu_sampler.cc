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

#include "runtime/components/top_k_webgpu_sampler.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
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
#include "ml_drift/webgpu/compute_task.h"  // from @ml_drift
#include "ml_drift/webgpu/environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift
#include "ml_drift/webgpu/instance.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "litert/cc/internal/litert_handle.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif  // __EMSCRIPTEN__

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
using ::ml_drift::webgpu::ComputeTask;
using WebGpuEnv = ::ml_drift::webgpu::Environment;
using ::ml_drift::webgpu::CreateTensor;
using ::ml_drift::webgpu::GetFastestStorageType;
using ::ml_drift::webgpu::InferenceContext;
using ::ml_drift::webgpu::SpatialTensor;

// A helper class to handle creating encoders/submitting.
class ComputePassHelper {
 public:
  explicit ComputePassHelper(const WebGpuEnv* env) : env_(env) {
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
  const WebGpuEnv* env_;
  wgpu::CommandEncoder command_encoder_;
  wgpu::ComputePassEncoder compute_pass_encoder_;
};

absl::Status InitComputeTask(WebGpuEnv* env,
                             std::unique_ptr<GPUOperation>&& gpu_op,
                             ComputeTask* task) {
  ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(env->GetInfo()));
  ABSL_RETURN_IF_ERROR(task->Init(*env, std::move(gpu_op)));
  ABSL_RETURN_IF_ERROR(task->Compile(*env, env->GetComputePipelineCache()));
  return absl::OkStatus();
}

absl::Status InitComputeTask(WebGpuEnv* env, GPUOperation&& operation,
                             ComputeTask* task) {
  return InitComputeTask(
      env, std::make_unique<GPUOperation>(std::move(operation)), task);
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

absl::StatusOr<std::unique_ptr<WebGpuEnv>> CreateWebGpuEnvFromLiteRtEnv(
    const Environment* env) {
  LITERT_ASSIGN_OR_RETURN(auto env_options, env->GetOptions());

#ifdef LITERT_SAMPLER_LINKED_DYNAMICALLY
  LITERT_ASSIGN_OR_RETURN(
      auto instance_ptr,
      env_options.GetOption(EnvironmentOptions::Tag::kWebGpuInstance));
  wgpu::Instance instance = reinterpret_cast<WGPUInstance>(  // NOLINT
      std::get<int64_t>(instance_ptr));
  ABSL_RETURN_IF_ERROR(ml_drift::webgpu::Instance::Set(instance));
#endif  // LITERT_SAMPLER_LINKED_DYNAMICALLY

  LITERT_ASSIGN_OR_RETURN(
      auto wgpu_device_ptr,
      env_options.GetOption(EnvironmentOptions::Tag::kWebGpuDevice));
  wgpu::Device device = reinterpret_cast<WGPUDevice>(  // NOLINT
      std::get<int64_t>(wgpu_device_ptr));

  wgpu::AdapterInfo adapter_info;
  device.GetAdapterInfo(&adapter_info);

  auto wgpu_env = std::make_unique<WebGpuEnv>(
#if defined(__APPLE__)
      wgpu::BackendType::Metal
#elif defined(_WIN32)
      wgpu::BackendType::D3D12
#elif defined(__EMSCRIPTEN__)
      wgpu::BackendType::WebGPU
#else
      wgpu::BackendType::Vulkan
#endif
  );
  ABSL_RETURN_IF_ERROR(wgpu_env->Initialize(device, adapter_info));

  return wgpu_env;
}

// Binds a tensor to the inference context for the given ID.
// shared_tensors is used to hold any tensors that need to be kept alive for the
// duration of the inference.
absl::Status BindTensor(const TensorBuffer& tensor_buffer, ml_drift::ValueId id,
                        InferenceContext& inference_context,
                        std::vector<SpatialTensor>& shared_tensors) {
  LITERT_ASSIGN_OR_RETURN(auto wgpu_buffer, tensor_buffer.GetWebGpuBuffer());
  auto* shared_tensor = reinterpret_cast<SpatialTensor*>(wgpu_buffer);
#if (defined(__APPLE__) || defined(__ANDROID__)) && \
    LITERT_SAMPLER_LINKED_DYNAMICALLY
  // On MacOS, runtime typeinfo may be different in each shared lib. Build a
  // logits tensor here with underlying WebGPU buffer to match typeinfos.
  shared_tensors.push_back(SpatialTensor());
  auto& shared_tensor_copy = shared_tensors.back();
  if (shared_tensor->GetBufferHandle()) {
    ABSL_RETURN_IF_ERROR(CreateSharedTensor(shared_tensor->GetBufferHandle(),
                                            shared_tensor->GetDescriptor(),
                                            &shared_tensor_copy));
  } else {
    ABSL_RETURN_IF_ERROR(CreateSharedTensor(
        shared_tensor->GetTextureHandle(),
        shared_tensor->GetTextureViewHandle(), shared_tensor->GetDescriptor(),
        &shared_tensor_copy));
  }
  shared_tensor = &shared_tensor_copy;
#endif  // (defined(__APPLE__) || defined(__ANDROID__)) &&
        // LITERT_SAMPLER_LINKED_DYNAMICALLY
  return inference_context.SetTensor(id, shared_tensor);
}

TensorDescriptor GetTensorDescriptor(const TensorBuffer* tensor_buffer) {
  if (!tensor_buffer) {
    return TensorDescriptor();
  }
  auto wgpu_buffer = tensor_buffer->GetWebGpuBuffer();
  if (!wgpu_buffer) {
    return TensorDescriptor();
  }
  return reinterpret_cast<SpatialTensor*>(*wgpu_buffer)->GetDescriptor();
}

size_t GetMappableBufferSize(const TensorDescriptor& tensor_desc) {
  return ml_drift::AlignByN(tensor_desc.GetMemorySizeInBytes(), 4);
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<TopKWebGpuSampler>> TopKWebGpuSampler::Create(
    Environment& env, int batch_size, int sequence_size, int vocab_size,
    std::optional<ActivationDataType> activation_data_type,
    SamplerParameters sampler_params) {
  RET_CHECK_GT(batch_size, 0) << "Batch size must be positive.";
  RET_CHECK_GT(vocab_size, 0) << "Vocabulary size must be positive.";
  ABSL_ASSIGN_OR_RETURN(auto wgpu_env, CreateWebGpuEnvFromLiteRtEnv(&env));
  auto gpu_info = wgpu_env->GetInfo();
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
  auto handler = absl::WrapUnique(new TopKWebGpuSampler(
      std::move(litert_env), std::move(wgpu_env), std::move(gpu_info),
      std::move(create_info), sampler_params, batch_size, sequence_size,
      vocab_size, sampler_params.k(), logits_data_type));
  ABSL_RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status TopKWebGpuSampler::InitSampling() {
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

  constraint_mask_ = std::make_unique<SpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(env_->device(),
                                    constraint_mask_handle.tensor_desc,
                                    constraint_mask_.get()));
  std::vector<uint8_t> mask_data(constraint_mask_->GetMemorySizeInBytes(), 0);
  ABSL_RETURN_IF_ERROR(
      constraint_mask_->WriteData(env_->queue(), mask_data.data()));
  create_info_copy.external_immutable_tensors[constraint_mask_handle.id] =
      constraint_mask_.get();

  tokens_ids_ = std::make_unique<SpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(env_->device(), GetTokensTensorDescriptor(),
                                    tokens_ids_.get()));

  wgpu::BufferDescriptor tokens_ids_mappable_buffer_desc = {
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = GetMappableBufferSize(tokens_ids_->GetDescriptor()),
  };
  tokens_ids_mappable_buffer_ = std::make_unique<wgpu::Buffer>(
      env_->device().CreateBuffer(&tokens_ids_mappable_buffer_desc));

  text_params_.params_i32 = std::make_unique<SpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(env_->device(), GetParamsTensorDescriptor(),
                                    text_params_.params_i32.get()));
  params_f32_ = std::make_unique<SpatialTensor>();
  ABSL_RETURN_IF_ERROR(CreateTensor(
      env_->device(), params_f32_handle.tensor_desc, params_f32_.get()));

  create_info_copy.external_immutable_tensors[params_i32_handle.id] =
      text_params_.params_i32.get();
  create_info_copy.external_immutable_tensors[params_f32_handle.id] =
      params_f32_.get();
  create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
      tokens_ids_.get();

  sampling_ = std::make_unique<InferenceContext>(InferenceContext());
  ABSL_RETURN_IF_ERROR(
      sampling_->InitFromGpuModel(create_info_copy, &gpu_model, env_.get()));

  return InitHelperOps(env_.get());
}

absl::Status TopKWebGpuSampler::InitHelperOps(WebGpuEnv* env) {
  if (text_params_.params_i32) {
    text_params_.write_i32_params = std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(InitComputeTask(
        env,
        CreateWriteParamsOp(text_params_.params_i32->GetDescriptor(),
                            text_params_.params_i32->Channels()),
        text_params_.write_i32_params.get()));
  }
  if (params_f32_) {
    write_f32_params_ = std::make_unique<ComputeTask>();
    ABSL_RETURN_IF_ERROR(
        InitComputeTask(env,
                        CreateWriteParamsOp(params_f32_->GetDescriptor(),
                                            params_f32_->Channels()),
                        write_f32_params_.get()));
  }
  return absl::OkStatus();
}

absl::Status TopKWebGpuSampler::ExecuteUpdateIntParams(
    WebGpuEnv* env, TransformerParams& params,
    const LlmRuntimeParams& param_vals) {
  ComputePassHelper helper(env);
  auto encoder = helper.compute_pass_encoder();
  ABSL_RETURN_IF_ERROR(
      params.write_i32_params->SetDstTensor(0, params.params_i32.get()));
  ABSL_RETURN_IF_ERROR(params.write_i32_params->SetInt(
      absl::StrCat("param", LlmRuntimeParams::kTopKIndex), param_vals.topk));
  ABSL_RETURN_IF_ERROR(params.write_i32_params->Update(env->device()));
  params.write_i32_params->UpdateGpuObjectBindings(env->device());
  ABSL_RETURN_IF_ERROR(params.write_i32_params->Encode(encoder));
  helper.Submit();
  return absl::OkStatus();
}

absl::Status TopKWebGpuSampler::ExecuteUpdateParams(
    WebGpuEnv* env, SpatialTensor* tensor, const std::vector<float>& params) {
  ABSL_RETURN_IF_ERROR(write_f32_params_->SetDstTensor(0, tensor));
  for (int i = 0; i < params.size(); ++i) {
    const std::string param_name = absl::StrCat("param", i);
    ABSL_RETURN_IF_ERROR(write_f32_params_->SetFloat(param_name, params[i]));
  }
  ABSL_RETURN_IF_ERROR(write_f32_params_->Update(env->device()));
  ABSL_RETURN_IF_ERROR(write_f32_params_->Execute(*env));
  return absl::OkStatus();
}

absl::Status TopKWebGpuSampler::SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
    TensorBuffer* scores_tensor) {
  RET_CHECK_EQ(scores_tensor, nullptr)
      << "This backend does not support scoring for now.";

  std::vector<SpatialTensor> shared_tensors;
  ABSL_RETURN_IF_ERROR(
      BindTensor(logits_tensor, logits_id_, *sampling_, shared_tensors));
  ABSL_RETURN_IF_ERROR(ExecuteUpdateIntParams(
      env_.get(), text_params_, CreateLlmRuntimeParams(sampler_params_)));
  ABSL_RETURN_IF_ERROR(ExecuteUpdateParams(
      env_.get(), params_f32_.get(),
      CreateFloatParams(sampler_params_, rand_gen_, params_f32_->Channels())));

  ABSL_RETURN_IF_ERROR(sampling_->Execute(*env_));
  ABSL_RETURN_IF_ERROR(CopyTokenIdsToMappableBuffer());

  if (input_handling_ != nullptr) {
    ABSL_RETURN_IF_ERROR(input_handling_->Execute(*env_));
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

absl::Status TopKWebGpuSampler::CopyTokenIdsToMappableBuffer() {
  // Code is copied from ml_drift::webgpu::ReadDataFromBuffer().
  wgpu::CommandEncoder encoder = env_->device().CreateCommandEncoder();
  encoder.CopyBufferToBuffer(
      tokens_ids_->GetBufferHandle(), 0, *tokens_ids_mappable_buffer_, 0,
      GetMappableBufferSize(tokens_ids_->GetDescriptor()));
  wgpu::CommandBuffer cb = encoder.Finish();
  env_->queue().Submit(1, &cb);
  return absl::OkStatus();
}

absl::Status TopKWebGpuSampler::UpdateConfig(
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

absl::Status TopKWebGpuSampler::DownloadSampledIds(void* dst) {
  size_t data_size = GetMappableBufferSize(tokens_ids_->GetDescriptor());

  wgpu::MapAsyncStatus status;
  std::string message;
  wgpu::Future future = tokens_ids_mappable_buffer_->MapAsync(
      wgpu::MapMode::Read, 0, data_size, wgpu::CallbackMode::WaitAnyOnly,
      [&status, &message](wgpu::MapAsyncStatus s, wgpu::StringView msg) {
        status = s;
        message = std::string(msg);
      });

#ifdef __EMSCRIPTEN__
  // Keep busy-waiting for Emscripten as it needs to yield to JS to process the
  // result of the `.mapAsync` call and return it to us.
  // This likely won't be necessary if running in a pthread.
  wgpu::FutureWaitInfo wait_info = {};
  wait_info.future = future;
  wgpu::WaitStatus wait_status = wgpu::WaitStatus::TimedOut;
  absl::Time timeout = absl::Now() + absl::Seconds(120);
  while (wait_status == wgpu::WaitStatus::TimedOut) {
    ml_drift::webgpu::Instance::ProcessEvents();
    wait_status = ml_drift::webgpu::Instance::Get().WaitAny(1u, &wait_info, 0);
    if (absl::Now() > timeout) {
      return absl::AbortedError(
          "The timeout was reached while reading back data.");
    }
    if (wait_status == wgpu::WaitStatus::TimedOut) {
      emscripten_sleep(1);
    }
  }
  ABSL_RETURN_IF_ERROR(ml_drift::webgpu::VerifyWaitStatus(wait_status));
#else
  ABSL_RETURN_IF_ERROR(ml_drift::webgpu::Instance::Wait(env_->device(), future,
                                                        absl::Seconds(120)));
#endif

  if (status != wgpu::MapAsyncStatus::Success) {
    return absl::InternalError(
        message.empty() ? absl::StrCat("MapAsync error: ", status) : message);
  }

  absl::Cleanup cleanup = [this]() { tokens_ids_mappable_buffer_->Unmap(); };

  auto* ptr = tokens_ids_mappable_buffer_->GetConstMappedRange(0, data_size);
  if (!ptr) {
    return absl::InternalError("GetConstMappedRange failed.");
  }

  size_t logical_size =
      kv_cache_batch_size_ * config_.sequence_size * sizeof(int32_t);
  // No need to conversion since the data is the vector of int32_t.
  std::memcpy(dst, ptr, logical_size);
  return absl::OkStatus();
}

bool TopKWebGpuSampler::CanHandleInput() const { return true; }

bool TopKWebGpuSampler::HandlesInput() const {
  return input_handling_ != nullptr && run_inference_func_ != nullptr;
}

absl::Status TopKWebGpuSampler::SetInferenceFuncAndInputTensors(
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

    ml_drift::CreateGpuModelInfo input_handling_create_info = create_info_;
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
  ABSL_RETURN_IF_ERROR(BindTensor(*ids_tensor, input_handling_ids_[1],
                                  *input_handling_, shared_tensors_));
  ABSL_RETURN_IF_ERROR(BindTensor(*prev_input_positions_tensor,
                                  input_handling_ids_[2], *input_handling_,
                                  shared_tensors_));
  ABSL_RETURN_IF_ERROR(BindTensor(*input_positions_tensor,
                                  input_handling_ids_[3], *input_handling_,
                                  shared_tensors_));
  size_t idx = 4;
  if (mask_tensor != nullptr) {
    ABSL_RETURN_IF_ERROR(BindTensor(*prev_mask_tensor, input_handling_ids_[idx],
                                    *input_handling_, shared_tensors_));
    ABSL_RETURN_IF_ERROR(BindTensor(*mask_tensor, input_handling_ids_[idx + 1],
                                    *input_handling_, shared_tensors_));
    idx += 2;
  }
  if (param_tensor != nullptr) {
    ABSL_RETURN_IF_ERROR(BindTensor(*prev_param_tensor,
                                    input_handling_ids_[idx], *input_handling_,
                                    shared_tensors_));
    ABSL_RETURN_IF_ERROR(BindTensor(*param_tensor, input_handling_ids_[idx + 1],
                                    *input_handling_, shared_tensors_));
  }
  run_inference_func_ = run_inference_func;
  run_inference_arg_ = arg;

  return absl::OkStatus();
}

}  // namespace litert::lm

// C API implementations for TopKWebGpuSampler.
int LiteRtTopKWebGpuSampler_Create(
    const struct LiteRtRuntimeCApiStruct* absl_nullable runtime_c_api,
    LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKWebGpuSampler_ActivationDataType* absl_nullable
        activation_data_type,
    const LiteRtTopKWebGpuSampler_SamplerParameters* absl_nullable
        sampler_params,
    LiteRtTopKWebGpuSampler_Sampler** sampler_out,
    char** absl_nullable error_msg) {
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
  // It's a wrapper of C env, and can be destroyed after sampler creation.
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

  auto sampler = litert::lm::TopKWebGpuSampler::Create(
      cpp_env, batch_size, sequence_size, vocab_size,
      std::move(cpp_activation_data_type), std::move(cpp_sampler_params));
  SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(sampler.status(), error_msg);
  *sampler_out =
      reinterpret_cast<LiteRtTopKWebGpuSampler_Sampler*>(sampler->release());
  return 0;
}

void LiteRtTopKWebGpuSampler_Destroy(LiteRtTopKWebGpuSampler_Sampler* sampler) {
  delete reinterpret_cast<litert::lm::TopKWebGpuSampler*>(sampler);
}

int LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer(
    LiteRtTopKWebGpuSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor,
    const LiteRtTensorBuffer* absl_nullable scores_tensor,
    char** absl_nullable error_msg) {
  if (!sampler) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Sampler not provided."), error_msg);
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKWebGpuSampler*>(sampler);
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

int LiteRtTopKWebGpuSampler_UpdateConfig(
    LiteRtTopKWebGpuSampler_Sampler* sampler,
    const LiteRtTopKWebGpuSampler_SamplerParameters* sampler_params,
    int batch_size, void* rand_gen_shared_ptr, char** error_msg) {
  if (!sampler) {
    SAMPLER_RETURN_AND_SET_ERROR_IF_NOT_OK(
        absl::InvalidArgumentError("Sampler not provided."), error_msg);
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKWebGpuSampler*>(sampler);
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

int LiteRtTopKWebGpuSampler_CanHandleInput(
    LiteRtTopKWebGpuSampler_Sampler* sampler) {
  if (!sampler) {
    return 0;
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKWebGpuSampler*>(sampler);
  return cpp_sampler->CanHandleInput() ? 1 : 0;
}

int LiteRtTopKWebGpuSampler_HandlesInput(
    LiteRtTopKWebGpuSampler_Sampler* sampler) {
  if (!sampler) {
    return 0;
  }
  auto cpp_sampler = reinterpret_cast<litert::lm::TopKWebGpuSampler*>(sampler);
  return cpp_sampler->HandlesInput() ? 1 : 0;
}

int LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors(
    LiteRtTopKWebGpuSampler_Sampler* sampler,
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

  auto cpp_sampler = reinterpret_cast<litert::lm::TopKWebGpuSampler*>(sampler);
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
extern "C" int (*LiteRtTopKWebGpuSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env,
    int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKWebGpuSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKWebGpuSampler_SamplerParameters* sampler_params,
    LiteRtTopKWebGpuSampler_Sampler** sampler_out, char** error_msg);

extern "C" void (*LiteRtTopKWebGpuSampler_Destroy_Static)(
    LiteRtTopKWebGpuSampler_Sampler* sampler);

extern "C" int (*LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKWebGpuSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg);

extern "C" int (*LiteRtTopKWebGpuSampler_UpdateConfig_Static)(
    LiteRtTopKWebGpuSampler_Sampler* sampler,
    const LiteRtTopKWebGpuSampler_SamplerParameters* sampler_params,
    int batch_size, void* rand_gen_shared_ptr, char** error_msg);

extern "C" int (*LiteRtTopKWebGpuSampler_CanHandleInput_Static)(
    LiteRtTopKWebGpuSampler_Sampler* sampler);

extern "C" int (*LiteRtTopKWebGpuSampler_HandlesInput_Static)(
    LiteRtTopKWebGpuSampler_Sampler* sampler);

extern "C" int (
    *LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static)(
    LiteRtTopKWebGpuSampler_Sampler* sampler,
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
    LiteRtTopKWebGpuSampler_Create_Static =
        [](const struct LiteRtRuntimeCApiStruct* runtime_c_api,
           LiteRtEnvironment env, int batch_size, int sequence_size,
           int vocab_size,
           const LiteRtTopKWebGpuSampler_ActivationDataType*
               activation_data_type,
           const LiteRtTopKWebGpuSampler_SamplerParameters* sampler_params,
           LiteRtTopKWebGpuSampler_Sampler** sampler_out, char** error_msg) {
          return LiteRtTopKWebGpuSampler_Create(
              runtime_c_api, env, batch_size, sequence_size, vocab_size,
              activation_data_type, sampler_params, sampler_out, error_msg);
        };
    LiteRtTopKWebGpuSampler_Destroy_Static =
        [](LiteRtTopKWebGpuSampler_Sampler* sampler) {
          LiteRtTopKWebGpuSampler_Destroy(sampler);
        };
    LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static =
        [](LiteRtTopKWebGpuSampler_Sampler* sampler,
           LiteRtTensorBuffer logits_tensor, LiteRtTensorBuffer ids_tensor,
           const LiteRtTensorBuffer* scores_tensor, char** error_msg) {
          return LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer(
              sampler, logits_tensor, ids_tensor, scores_tensor, error_msg);
        };
    LiteRtTopKWebGpuSampler_UpdateConfig_Static =
        [](LiteRtTopKWebGpuSampler_Sampler* sampler,
           const LiteRtTopKWebGpuSampler_SamplerParameters* sampler_params,
           int batch_size, void* rand_gen_shared_ptr, char** error_msg) {
          return LiteRtTopKWebGpuSampler_UpdateConfig(
              sampler, sampler_params, batch_size, rand_gen_shared_ptr,
              error_msg);
        };
    LiteRtTopKWebGpuSampler_CanHandleInput_Static =
        [](LiteRtTopKWebGpuSampler_Sampler* sampler) {
          return LiteRtTopKWebGpuSampler_CanHandleInput(sampler);
        };
    LiteRtTopKWebGpuSampler_HandlesInput_Static =
        [](LiteRtTopKWebGpuSampler_Sampler* sampler) {
          return LiteRtTopKWebGpuSampler_HandlesInput(sampler);
        };
    LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static =
        [](LiteRtTopKWebGpuSampler_Sampler* sampler,
           int (*run_inference_func)(void* arg), void* arg,
           LiteRtTensorBuffer absl_nullable ids_tensor,
           LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
           LiteRtTensorBuffer absl_nullable input_positions_tensor,
           LiteRtTensorBuffer absl_nullable prev_mask_tensor,
           LiteRtTensorBuffer absl_nullable mask_tensor,
           LiteRtTensorBuffer absl_nullable prev_param_tensor,
           LiteRtTensorBuffer absl_nullable param_tensor, char** error_msg) {
          return LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors(
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
