#include "runtime/components/top_k_gpu_sampler.h"

#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "third_party/ml_drift/common/data_type.h"
#include "third_party/ml_drift/common/gpu_model.h"
#include "third_party/ml_drift/common/gpu_model_builder.h"
#include "third_party/ml_drift/common/operations.h"
#include "third_party/ml_drift/common/precision.h"
#include "third_party/ml_drift/common/shape.h"
#include "third_party/ml_drift/common/task/buffer_desc.h"
#include "third_party/ml_drift/common/task/gpu_object_desc.h"
#include "third_party/ml_drift/common/task/gpu_operation.h"
#include "third_party/ml_drift/common/task/tensor_desc.h"
#include "third_party/ml_drift/common/types.h"
#include "third_party/ml_drift/common/util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "util/task/status_macros.h"

namespace litert::lm {

namespace {
using ::litert::lm::proto::SamplerParameters;
using ::ml_drift::BHWC;
using ::ml_drift::BufferDescriptor;
using ::ml_drift::DataType;
using ::ml_drift::GpuModel;
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::GPUOperation;
using ::ml_drift::Layout;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::TensorToGrid;
constexpr float kConstraintMaskPenalty = 0.7f;

GPUOperation CreateParamToTensorOp(const TensorDescriptor& dst,
                                   int param_index) {
  GPUOperation op;
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::FLOAT32;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = ml_drift::MemoryType::GLOBAL;
  op.AddSrcBuffer("src", buffer_desc);
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("param_index", param_index);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.code_ = R"(
MAIN_FUNCTION($0) {
int X = ucl::GetGlobalId<0>();
int Y = ucl::GetGlobalId<1>();
int S = ucl::GetGlobalId<2>();
if (X != 0 || Y != 0 || S != 0) return;
args.dst::type result;
result.x = ucl::Convert<args.dst::scalar_type>(args.src.Read(args.param_index));
args.dst.Write(result, 0, 0, 0);
}
)";
  return op;
}
}  // namespace

absl::Status TopKGpuSampler::Initialize() {
  RETURN_IF_ERROR(InitSampling());
  return absl::OkStatus();
}

absl::StatusOr<GpuModel> TopKGpuSampler::CreateSamplingModel(
    GpuModelBuilder::TensorHandle* src_logits,
    GpuModelBuilder::TensorHandle* constraint_mask_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    GpuModelBuilder::TensorHandle* params_f32_handle,
    GpuModelBuilder::TensorHandle* output_tokens) {
  create_info_.external_immutable_tensors.clear();
  create_info_.external_mutable_tensors.clear();

  config_.batch_size = kv_cache_batch_size_;

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);
  std::vector<GpuModelBuilder::ValueId> input_ids;
  std::vector<GpuModelBuilder::ValueId> output_ids;
  *src_logits =
      model_builder_.AddTensor(config_.batch_size, 1, 1, config_.vocab_size);
  *params_i32_handle = model_builder_.AddTensor(GetParamsTensorDescriptor());
  *params_f32_handle = model_builder_.AddTensor(
      1, 1, 1, GetF32ParamsCount(config_.sequence_size),
      TensorStorageType::BUFFER, DataType::FLOAT32);

  *output_tokens = model_builder_.AddTensor(GetTokensTensorDescriptor());

  auto t = *src_logits;

  if (constraint_mask_handle) {
    *constraint_mask_handle = model_builder_.AddTensor(
        config_.batch_size, 1, 1, config_.vocab_size, TensorStorageType::BUFFER,
        DeduceDataTypeFromPrecision(create_info_.precision));
    // TODO: b/404612147 - Consolidate the penalty value for mask.
    float penalty = t.tensor_desc.GetDataType() == DataType::FLOAT32
                        ? std::numeric_limits<float>::max()
                        : ml_drift::kMaxHalf;
    penalty *= kConstraintMaskPenalty;
    auto mask = model_builder_.Multiplication(*constraint_mask_handle, penalty);
    t = model_builder_.Elementwise(t, mask, ml_drift::OperationType::SUB);
  }

  const int top_k_size = ml_drift::AlignByN(config_.max_top_k, 4);
  // This is reshaped into config_.sequence_size * config_.batch_size
  // because we've two batching dimensions, along the config_.batch_size
  // direction (different prompts) and along the sequence_size
  // direction (consecutive tokens in the same prompt, primarily for speculative
  // decoding). However, we don't have enough dimensions to host both so here we
  // squeeze them both in the same dimension for easier processing.
  t = model_builder_.Reshape(t,
                             BHWC(1, config_.sequence_size * config_.batch_size,
                                  config_.vocab_size / 4, 4));
  auto results = model_builder_.TopK(t, top_k_size);
  GpuModelBuilder::TensorHandle out_max = model_builder_.Reshape(
      results[0],
      BHWC(config_.batch_size * config_.sequence_size, 1, 1, top_k_size));
  GpuModelBuilder::TensorHandle out_ind = model_builder_.Reshape(
      results[1],
      BHWC(config_.batch_size * config_.sequence_size, 1, 1, top_k_size));
  {
    GpuModelBuilder::TensorHandle temperature =
        model_builder_.AddTensor(BHWC(1, 1, 1, 1));
    GPUOperation write_temperature = CreateParamToTensorOp(
        temperature.tensor_desc, config_.batch_size * config_.sequence_size);
    model_builder_.AddGpuOperation(
        {*params_f32_handle}, temperature,
        std::make_unique<GPUOperation>(std::move(write_temperature)),
        "write_temperature");
    out_max = model_builder_.Multiplication(out_max, temperature);
  }

  ml_drift::SoftmaxRuntimeCheckDesc runtime_check = {
      .end_ch_index = LlmRuntimeParams::kTopKIndex};
  out_max = model_builder_.Softmax(out_max, runtime_check, params_i32_handle);
  MakeSampling(out_max, out_ind, *params_i32_handle, *params_f32_handle,
               *output_tokens);

  input_ids = {src_logits->id, params_i32_handle->id, params_f32_handle->id};
  if (constraint_mask_handle) {
    input_ids.push_back(constraint_mask_handle->id);
  }
  output_ids = {output_tokens->id};
  GpuModel gpu_model;
  RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

TensorDescriptor TopKGpuSampler::GetParamsTensorDescriptor() const {
  TensorDescriptor td =
      TensorDescriptor{DataType::INT32, TensorStorageType::BUFFER, Layout::HWC};
  td.SetBHWCShape(BHWC(1, 1, 1, GetI32ParamsCount()));
  return td;
}

TensorDescriptor TopKGpuSampler::GetTokensTensorDescriptor() const {
  TensorDescriptor td =
      TensorDescriptor{DataType::INT32, TensorStorageType::BUFFER, Layout::HWC};
  // tokens will be used as plain buffer, putting everything to last dimension
  td.SetBHWCShape(BHWC(1, 1, 1, kv_cache_batch_size_ * config_.cache_size));
  return td;
}

// TODO: b/380315846 - Refactor this to return GPUOperation and write tests.
void TopKGpuSampler::MakeSampling(
    GpuModelBuilder::TensorHandle& src_logits,
    GpuModelBuilder::TensorHandle& src_indices,
    GpuModelBuilder::TensorHandle& params_i32_handle,
    GpuModelBuilder::TensorHandle& params_f32_handle,
    GpuModelBuilder::TensorHandle& output_tokens) {
  // Some notes on the shader variables:
  // - B variable in shader below is the batch of the shader, which is Width
  // (sequence length) * Batch (batch size).
  // - batch_count is the actual batch count (config_.batch_size) and seq_len is
  // the sequence length (config_.sequence_size).
  // - seq_id is which token are we sampling in the current batch, within [0,
  // seq_len).
  // - actual_batch is which batch are we sampling in the current batch, within
  // [0, batch_count).
  // - We need to decompose B into actual_batch and seq_id because the input
  // (src_logits and src_indices) have their batch dimension set to the product
  // of batch_count and seq_len, while the output (input_token_id) has
  // decomposed batch dimension (into B and W).

  // For the direction and ordering of f32 params, see the decomposition of B
  // below.

  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int B = ucl::GetGlobalId<0>();
  args.src_logits.SetBatchRef(B);
  args.src_indices.SetBatchRef(B);

  int batch_count = args.src_logits.Batch();
  int seq_len = args.sequence_size;
  int actual_batch = B / seq_len;
  int seq_id = B % seq_len;

  float probability = args.params_f32.Read(B);
  float cum_sum = 0.0f;
  int index = -1;
  int top_k = args.params_i32.Read($1);
  for (int s = 0; s < args.src_logits.Slices(); ++s) {
    float4 vals = args.src_logits.Read<float>(0, 0, s);
    int4 inds = args.src_indices.Read<int>(0, 0, s);
    if (probability >= cum_sum && s * 4 + 0 < top_k) { index = inds.x; }
    cum_sum += vals.x;
    if (probability >= cum_sum && s * 4 + 1 < top_k) { index = inds.y; }
    cum_sum += vals.y;
    if (probability >= cum_sum && s * 4 + 2 < top_k) { index = inds.z; }
    cum_sum += vals.z;
    if (probability >= cum_sum && s * 4 + 3 < top_k) { index = inds.w; }
    cum_sum += vals.w;
  }
  int time_step = args.params_i32.Read($0) + seq_id;
  if (time_step <= args.cache_size) {
    args.output_tokens.Write(index, time_step * batch_count + actual_batch);
  }
})",
                                      LlmRuntimeParams::kTokenOffsetIndex,
                                      LlmRuntimeParams::kTopKIndex);
  GPUOperation custom_op;
  custom_op.AddSrcTensor("src_logits", src_logits.tensor_desc);
  custom_op.AddSrcTensor("src_indices", src_indices.tensor_desc);
  BufferDescriptor params_i32_buffer;
  params_i32_buffer.element_type = DataType::INT32;
  params_i32_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params_i32", params_i32_buffer);
  BufferDescriptor params_f32_buffer;
  params_f32_buffer.element_type = DataType::FLOAT32;
  params_f32_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params_f32", params_f32_buffer);
  {
    BufferDescriptor output_tokens_buffer;
    output_tokens_buffer.element_type = DataType::INT32;
    output_tokens_buffer.element_size = 1;
    custom_op.AddDstBuffer("output_tokens", output_tokens_buffer);
    custom_op.args_.AddInt("cache_size", config_.cache_size);
    custom_op.args_.AddInt("sequence_size", config_.sequence_size);
  }
  custom_op.code_ = std::move(code);
  custom_op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  model_builder_.AddGpuOperation(
      {src_logits, src_indices, params_i32_handle, params_f32_handle},
      {output_tokens}, std::make_unique<GPUOperation>(std::move(custom_op)),
      "sampling");
}

TopKGpuSampler::LlmRuntimeParams TopKGpuSampler::CreateLlmRuntimeParams(
    const SamplerParameters& sampler_params, int time_step,
    int output_batch_size) const {
  return {
      .token_index_offset = time_step,
      .active_tokens = time_step + 1,
      .topk = sampler_params.k(),
      .mask_time_step = time_step,
      .mask_batch_size = output_batch_size,
  };
}

std::vector<float> TopKGpuSampler::CreateFloatParams(
    const SamplerParameters& sampler_params,
    std::shared_ptr<std::default_random_engine> rand_gen,
    int params_count) const {
  std::uniform_real_distribution<float> distribution =
      std::uniform_real_distribution<float>(0.0f, sampler_params.p());
  std::vector<float> params(params_count, 0.0f);
  for (int i = 0; i < params.size() - 1; ++i) {
    params[i] = distribution(*rand_gen);
  }
  params[params.size() - 1] = 1.0f / sampler_params.temperature();
  return params;
}

}  // namespace litert::lm
