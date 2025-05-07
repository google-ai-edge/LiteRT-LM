#include "runtime/components/top_k_opencl_sampler.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/ml_drift/cl/cl_command_queue.h"
#include "third_party/ml_drift/cl/cl_operation.h"
#include "third_party/ml_drift/cl/environment.h"
#include "third_party/ml_drift/cl/inference_context.h"
#include "third_party/ml_drift/cl/tensor.h"
#include "third_party/ml_drift/common/data_type.h"
#include "third_party/ml_drift/common/gpu_info.h"
#include "third_party/ml_drift/common/gpu_model.h"
#include "third_party/ml_drift/common/gpu_model_builder.h"
#include "third_party/ml_drift/common/model_hints.h"
#include "third_party/ml_drift/common/precision.h"
#include "third_party/ml_drift/common/shape.h"
#include "third_party/ml_drift/common/task/gpu_operation.h"
#include "third_party/ml_drift/common/task/tensor_desc.h"
#include "third_party/ml_drift/common/util.h"
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "util/task/status_macros.h"

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
using ::ml_drift::cl::CLCommandQueueOptions;
using ::ml_drift::cl::ClOperation;
using ::ml_drift::cl::GetFastestStorageType;
using ::ml_drift::cl::InferenceContext;
using ::ml_drift::cl::Tensor;
using ClEnv = ::ml_drift::cl::Environment;

absl::Status InitClOp(ClEnv* env, std::unique_ptr<GPUOperation>&& gpu_op,
                      ClOperation* cl_op) {
  RETURN_IF_ERROR(gpu_op->AssembleCode(env->device().GetInfo()));

  cl_op->Init(std::move(gpu_op));
  {
    ml_drift::cl::CreationContext creation_context;
    creation_context.device = env->GetDevicePtr();
    creation_context.context = &env->context();
    creation_context.queue = env->queue();
    creation_context.cache = env->program_cache();
    RETURN_IF_ERROR(cl_op->Compile(creation_context));
  }
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
    if (dst.GetDataType() == DataType::FLOAT32) {
      op.args_.AddFloat(param_name, 0);
    } else if (dst.GetDataType() == DataType::INT32) {
      op.args_.AddInt(param_name, 0);
    }
  }
  if (dst.GetDataType() == DataType::FLOAT32) {
    op.args_.AddFloat("zero_value", 0);
  } else if (dst.GetDataType() == DataType::INT32) {
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
}  // namespace

// static
absl::StatusOr<std::unique_ptr<TopKOpenClSampler>> TopKOpenClSampler::Create(
    Environment* env, int batch_size, int cache_size, int vocab_size,
    std::optional<ActivationDataType> activation_data_type,
    SamplerParameters sampler_params) {
  // TODO: b/414667552 - Take CL env from LiteRT env.
  auto cl_env = std::make_unique<ClEnv>();
  RETURN_IF_ERROR(ml_drift::cl::CreateEnvironment(cl_env.get()));
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
    create_info.precision = ml_drift::CalculationsPrecision::F16;
  } else {
    create_info.precision = ml_drift::CalculationsPrecision::F32;
  }
  create_info.hints.Add(ml_drift::ModelHints::kFastTuning);
  create_info.hints.Add(ml_drift::ModelHints::kPreferTextureWeights);
  create_info.hints.Add(ml_drift::ModelHints::kAllowSpecialKernels);
  create_info.storage_type = GetFastestStorageType(gpu_info);

  auto handler = absl::WrapUnique(new TopKOpenClSampler(
      std::move(cl_env), std::move(gpu_info), std::move(create_info),
      sampler_params, batch_size, cache_size,
      /*sequence_size=*/1, vocab_size, sampler_params.k()));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status TopKOpenClSampler::InitSampling() {
  GpuModelBuilder::TensorHandle src_logits;
  GpuModelBuilder::TensorHandle constraint_mask_handle;
  GpuModelBuilder::TensorHandle tokens_ids_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle params_f32_handle;
  ASSIGN_OR_RETURN(auto gpu_model,
                   CreateSamplingModel(&src_logits, &constraint_mask_handle,
                                       &params_i32_handle, &params_f32_handle,
                                       &tokens_ids_handle));
  auto create_info_copy = create_info_;

  // Register mutable tensor for logits, such that we can provide logits tensor
  // later at inference. This ensures flexibility and interface unification of
  // the sampler across backends.
  logits_id_ = src_logits.id;
  logits_tensor_desc_ = src_logits.tensor_desc;
  create_info_copy.external_mutable_tensors[logits_id_] = logits_tensor_desc_;

  constraint_mask_ = std::make_unique<Tensor>(Tensor());
  RETURN_IF_ERROR(CreateTensor(env_->context(),
                               constraint_mask_handle.tensor_desc,
                               constraint_mask_.get()));
  create_info_copy.external_immutable_tensors[constraint_mask_handle.id] =
      constraint_mask_.get();

  tokens_ids_ = std::make_unique<Tensor>(Tensor());
  RETURN_IF_ERROR(CreateTensor(env_->context(), GetTokensTensorDescriptor(),
                               tokens_ids_.get()));

  text_params_.params_i32 = std::make_unique<Tensor>(Tensor());
  RETURN_IF_ERROR(CreateTensor(env_->context(), GetParamsTensorDescriptor(),
                               text_params_.params_i32.get()));
  params_f32_ = std::make_unique<Tensor>(Tensor());
  RETURN_IF_ERROR(CreateTensor(env_->context(), params_f32_handle.tensor_desc,
                               params_f32_.get()));

  create_info_copy.external_immutable_tensors[params_i32_handle.id] =
      text_params_.params_i32.get();
  create_info_copy.external_immutable_tensors[params_f32_handle.id] =
      params_f32_.get();
  create_info_copy.external_immutable_tensors[tokens_ids_handle.id] =
      tokens_ids_.get();

  sampling_ = std::make_unique<InferenceContext>(InferenceContext());
  RETURN_IF_ERROR(
      sampling_->InitFromGpuModel(create_info_copy, &gpu_model, env_.get()));

  RETURN_IF_ERROR(InitHelperOps(env_.get()));

  CLCommandQueueOptions queue_options;
  RETURN_IF_ERROR(CreateCLCommandQueue(env_->device(), env_->context(),
                                       &reading_queue_, queue_options));

  return absl::OkStatus();
}

absl::Status TopKOpenClSampler::InitHelperOps(ClEnv* env) {
  if (text_params_.params_i32) {
    text_params_.write_i32_params =
        std::make_unique<ClOperation>(ClOperation());
    RETURN_IF_ERROR(
        InitClOp(env,
                 CreateWriteParamsOp(text_params_.params_i32->GetDescriptor(),
                                     text_params_.params_i32->Channels()),
                 text_params_.write_i32_params.get()));
  }
  if (params_f32_) {
    write_f32_params_ = std::make_unique<ClOperation>(ClOperation());
    RETURN_IF_ERROR(InitClOp(env,
                             CreateWriteParamsOp(params_f32_->GetDescriptor(),
                                                 params_f32_->Channels()),
                             write_f32_params_.get()));
  }
  return absl::OkStatus();
}
}  // namespace litert::lm
