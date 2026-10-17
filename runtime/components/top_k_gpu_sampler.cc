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

#include "runtime/components/top_k_gpu_sampler.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

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

// Returns `snippet` instantiated for the 4 channels of the slice `s` of a
// top-k tensor, with `$0` replaced by the channel swizzle and `$1` by the
// index of the channel in the top-k.
std::string ForEachChannelOfSlice(absl::string_view snippet) {
  constexpr absl::string_view kSwizzles[] = {"x", "y", "z", "w"};
  std::string code;
  for (int i = 0; i < 4; ++i) {
    absl::StrAppend(&code, absl::Substitute(snippet, kSwizzles[i],
                                            absl::StrCat("s * 4 + ", i)));
  }
  return code;
}

// Returns the kernel code of AddTopKTopPSampling().
std::string GetTopKTopPSamplingCode(int top_k_index, int inv_temperature_index,
                                    int top_p_index) {
  // Unnormalized probabilities of the 4 top-k entries of the slice `s`.
  constexpr absl::string_view kSliceExps =
      "exp((ucl::Convert<float4>(args.logits.Read(W, H, s)) - "
      "ucl::Init<float4>(max_logit)) * inv_temperature)";
  const std::string sum_exps = ForEachChannelOfSlice(R"(
    if ($1 < top_k) {
      sum += e.$0;
    })");
  const std::string grow_nucleus = ForEachChannelOfSlice(R"(
    if (growing && $1 < top_k) {
      nucleus_mass += prob.$0;
      nucleus_size = $1 + 1;
      growing = nucleus_mass < top_p;
    })");
  const std::string pick_token = ForEachChannelOfSlice(R"(
    if (token < 0 && $1 < nucleus_size) {
      cumulative += prob.$0;
      last_token = ids.$0;
      if (threshold <= cumulative) {
        token = ids.$0;
      }
    })");
  // `$$0` is escaped to `$0`, the placeholder of the kernel arguments.
  return absl::Substitute(R"(MAIN_FUNCTION($$0) {
  int BW = ucl::GetGlobalId<0>();
  int H = ucl::GetGlobalId<1>();
  int B = BW % args.dst.Batch();
  int W = BW / args.dst.Batch();
  if (W >= args.dst.Width() || H >= args.dst.Height() ||
      ucl::GetGlobalId<2>() != 0) {
    return;
  }
  args.logits.SetBatchRef(B);
  args.indices.SetBatchRef(B);
  args.dst.SetBatchRef(B);

  int top_k = args.params_i32.Read($0);
  // The Reshape at the end of GpuModelBuilder::TopK() zero-fills the padding
  // channels of the last slice (logit 0, id 0), so never read past Channels().
  // UpdateConfig() rejects k above Channels(); this guards direct callers.
  if (top_k > args.logits.Channels()) {
    top_k = args.logits.Channels();
  }
  float inv_temperature = args.params_f32.Read($1);
  float top_p = args.params_f32.Read($2);
  float random_value = args.params_f32.Read(W * args.dst.Batch() + B);
  // The top-k logits are sorted in descending order.
  float max_logit = ucl::Convert<float4>(args.logits.Read(W, H, 0)).x;

  // Softmax of the top-k logits divided by the temperature.
  float sum = 0.0f;
  for (int s = 0; s < args.logits.Slices(); ++s) {
    float4 e = $3;$4
  }
  float inv_sum = 1.0f / sum;

  // The nucleus is the smallest prefix whose probability mass reaches top_p.
  float nucleus_mass = 0.0f;
  int nucleus_size = 0;
  bool growing = true;
  for (int s = 0; s < args.logits.Slices(); ++s) {
    float4 prob = $3 * inv_sum;$5
  }

  // Inverse transform sampling over the renormalized nucleus.
  float threshold = random_value * nucleus_mass;
  float cumulative = 0.0f;
  int token = -1;
  int last_token = -1;
  for (int s = 0; s < args.logits.Slices(); ++s) {
    float4 prob = $3 * inv_sum;
    int4 ids = args.indices.Read<int>(W, H, s);$6
  }
  // Rounding may leave the threshold above the final cumulative probability.
  if (token < 0) {
    token = last_token;
  }
  args.dst.Write(ucl::Init<int4>(token, 0, 0, 0), W, H, 0);
}
)",
                          top_k_index, inv_temperature_index, top_p_index,
                          kSliceExps, sum_exps, grow_nucleus, pick_token);
}

// Adds an operation that samples one token per (batch, position) from the
// top-k logits and their token indices (both sorted in descending order of
// the logits) with temperature and top-p (nucleus) sampling.
//
// This is the GPU counterpart of TopKTopPSampling() in sampling_cpu_util.cc
// and must stay in sync with it:
// - the probabilities are the softmax of the top-k logits divided by the
//   temperature,
// - the nucleus is the smallest prefix of the top-k whose cumulative
//   probability reaches top-p,
// - the token is picked by inverse transform sampling over the nucleus with
//   the threshold `random_value * nucleus_mass`, where `random_value` is the
//   uniform [0, 1) value drawn on the host for the (batch, position).
// `params_f32` is laid out as described in TopKGpuSampler::FloatParams.
GpuModelBuilder::TensorHandle AddTopKTopPSampling(
    const GpuModelBuilder::TensorHandle& top_k_logits,
    const GpuModelBuilder::TensorHandle& top_k_indices,
    const GpuModelBuilder::TensorHandle& params_i32,
    const GpuModelBuilder::TensorHandle& params_f32, int top_k_index,
    int inv_temperature_index, int top_p_index, GpuModelBuilder& builder) {
  BHWC dst_shape = top_k_logits.tensor_desc.GetBHWCShape();
  dst_shape.c = 1;
  auto dst = builder.AddTensor(dst_shape, DataType::kInt32);

  GPUOperation op;
  op.AddSrcTensor("logits", top_k_logits.tensor_desc);
  op.AddSrcTensor("indices", top_k_indices.tensor_desc);
  BufferDescriptor params_i32_desc;
  params_i32_desc.element_type = DataType::kInt32;
  params_i32_desc.element_size = 1;
  op.AddSrcBuffer("params_i32", params_i32_desc);
  BufferDescriptor params_f32_desc;
  params_f32_desc.element_type = DataType::kFloat32;
  params_f32_desc.element_size = 1;
  op.AddSrcBuffer("params_f32", params_f32_desc);
  op.AddDstTensor("dst", dst.tensor_desc);
  op.code_ =
      GetTopKTopPSamplingCode(top_k_index, inv_temperature_index, top_p_index);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  builder.AddGpuOperation({top_k_logits, top_k_indices, params_i32, params_f32},
                          {dst}, std::make_unique<GPUOperation>(std::move(op)),
                          "top_k_top_p_sampling");
  return dst;
}
}  // namespace

absl::Status TopKGpuSampler::Initialize() {
  ABSL_RETURN_IF_ERROR(InitSampling());
  return absl::OkStatus();
}

absl::StatusOr<GpuModel> TopKGpuSampler::CreateSamplingModel(
    GpuModelBuilder::TensorHandle* src_logits,
    GpuModelBuilder::TensorHandle* constraint_mask_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    GpuModelBuilder::TensorHandle* params_f32_handle,
    GpuModelBuilder::TensorHandle* output_tokens, DataType logits_data_type) {
  create_info_.external_immutable_tensors.clear();
  create_info_.external_mutable_tensors.clear();

  config_.batch_size = kv_cache_batch_size_;

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);
  std::vector<GpuModelBuilder::ValueId> input_ids;
  std::vector<GpuModelBuilder::ValueId> output_ids;
  // For now, logits tensor are not automatically set to Image Buffer and
  // remain to be texture 2D, which results in failing on some specific devices
  // (e.g. Pixel).
  // Thus we explicitly set the storage type to IMAGE_BUFFER here.
  // TODO: b/414667552 - Investigate the cause.
  *src_logits = model_builder_.AddTensor(
      BHWC(config_.batch_size, 1, config_.sequence_size, config_.vocab_size),
      logits_data_type);
  *params_i32_handle = model_builder_.AddTensor(GetParamsTensorDescriptor());
  *params_f32_handle = model_builder_.AddTensor(
      1, 1, 1, GetF32ParamsCount(config_.sequence_size),
      TensorStorageType::kBuffer, DataType::kFloat32);

  *output_tokens = model_builder_.AddTensor(GetTokensTensorDescriptor());

  auto t = *src_logits;

  if (constraint_mask_handle) {
    *constraint_mask_handle = model_builder_.AddTensor(
        config_.batch_size, 1, config_.sequence_size, config_.vocab_size,
        TensorStorageType::kBuffer, logits_data_type);
    // TODO: b/404612147 - Consolidate the penalty value for mask.
    float penalty = t.tensor_desc.GetDataType() == DataType::kFloat32
                        ? std::numeric_limits<float>::max()
                        : ::ml_drift::kMaxHalf;
    penalty *= kConstraintMaskPenalty;
    auto mask = model_builder_.Multiplication(*constraint_mask_handle, penalty);
    t = model_builder_.Elementwise(t, mask, ::ml_drift::OperationType::kSub);
  }

  auto results = model_builder_.TopK(t, config_.max_top_k);
  const int random_values_count = config_.batch_size * config_.sequence_size;
  auto samples = AddTopKTopPSampling(
      /*top_k_logits=*/results[0], /*top_k_indices=*/results[1],
      *params_i32_handle, *params_f32_handle, LlmRuntimeParams::kTopKIndex,
      random_values_count + FloatParams::kInvTemperatureOffset,
      random_values_count + FloatParams::kTopPOffset, model_builder_);
  samples = model_builder_.Transpose(samples, BHWC(2, 1, 0, 3));
  samples = model_builder_.Reshape(
      samples, BHWC(1, 1, 1, kv_cache_batch_size_ * config_.sequence_size));
  if (samples.tensor_desc.GetStorageType() !=
      output_tokens->tensor_desc.GetStorageType()) {
    model_builder_.Copy(samples, *output_tokens);
  } else {
    *output_tokens = samples;
  }

  input_ids = {src_logits->id, params_i32_handle->id, params_f32_handle->id};
  if (constraint_mask_handle) {
    input_ids.push_back(constraint_mask_handle->id);
  }
  output_ids = {output_tokens->id};
  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

absl::Status TopKGpuSampler::CheckTopKWithinMax(
    const SamplerParameters& sampler_params) const {
  if (sampler_params.k() > config_.max_top_k) {
    return absl::InvalidArgumentError(absl::StrCat(
        "k=", sampler_params.k(), " exceeds the k=", config_.max_top_k,
        " the GPU sampler was created with; create a new sampler instead."));
  }
  return absl::OkStatus();
}

TensorDescriptor TopKGpuSampler::GetParamsTensorDescriptor() const {
  TensorDescriptor td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kHWC};
  td.SetBHWCShape(BHWC(1, 1, 1, GetI32ParamsCount()));
  return td;
}

TensorDescriptor TopKGpuSampler::GetTokensTensorDescriptor() const {
  TensorDescriptor td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kHWC};
  // tokens will be used as plain buffer, putting everything to last dimension
  td.SetBHWCShape(BHWC(1, 1, 1, kv_cache_batch_size_ * config_.sequence_size));
  return td;
}

TopKGpuSampler::LlmRuntimeParams TopKGpuSampler::CreateLlmRuntimeParams(
    const SamplerParameters& sampler_params) const {
  return {
      .topk = sampler_params.k(),
  };
}

std::vector<float> TopKGpuSampler::CreateFloatParams(
    const SamplerParameters& sampler_params,
    std::shared_ptr<std::default_random_engine> rand_gen,
    int params_count) const {
  std::vector<float> params(params_count, 0.0f);
  const int random_values_count =
      params_count - FloatParams::kNonRandomParamsCount;
  // Consume `rand_gen` like TopKTopPSampling() in sampling_cpu_util.cc does:
  // one uniform double per sampled token in batch-major order, and nothing
  // for greedy decoding.
  if (std::min(sampler_params.k(), config_.vocab_size) > 1) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const int batch_size = std::min(config_.batch_size, kv_cache_batch_size_);
    const int sequence_size = random_values_count / kv_cache_batch_size_;
    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < sequence_size; ++s) {
        // The sampling model reads the value of (b, s) at s * B + b.
        params[s * kv_cache_batch_size_ + b] =
            static_cast<float>(distribution(*rand_gen));
      }
    }
  }
  params[random_values_count + FloatParams::kInvTemperatureOffset] =
      1.0f / std::max(sampler_params.temperature(),
                      std::numeric_limits<float>::epsilon());
  params[random_values_count + FloatParams::kTopPOffset] = sampler_params.p();
  return params;
}

absl::StatusOr<GpuModel> TopKGpuSampler::CreateInputHandlingModel(
    const TensorDescriptor& last_output_token_desc,
    const TensorDescriptor& input_token_desc,
    const TensorDescriptor& input_position_desc,
    const TensorDescriptor& mask_desc, const TensorDescriptor& param_desc,
    GpuModelBuilder::TensorHandle* last_output_token,
    GpuModelBuilder::TensorHandle* input_token,
    GpuModelBuilder::TensorHandle* prev_input_position,
    GpuModelBuilder::TensorHandle* input_position,
    GpuModelBuilder::TensorHandle* prev_mask,
    GpuModelBuilder::TensorHandle* mask,
    GpuModelBuilder::TensorHandle* prev_param,
    GpuModelBuilder::TensorHandle* param) {
  if (mask) {
    RET_CHECK(prev_mask != nullptr);
  } else {
    RET_CHECK(prev_mask == nullptr);
  }
  if (param) {
    RET_CHECK(prev_param != nullptr);
  } else {
    RET_CHECK(prev_param == nullptr);
  }

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);

  // OP semantic: input_token = last_output_token
  *last_output_token = model_builder_.AddTensor(last_output_token_desc);
  *input_token = model_builder_.AddTensor(input_token_desc);
  model_builder_.Copy(*last_output_token, *input_token);
  std::vector<GpuModelBuilder::ValueId> input_ids{last_output_token->id};
  std::vector<GpuModelBuilder::ValueId> output_ids{input_token->id};

  // OP semantic: input_position = prev_input_position + 1
  *prev_input_position = model_builder_.AddTensor(input_position_desc);
  *input_position = model_builder_.AddTensor(input_position_desc);
  auto updated_input_position = model_builder_.Add(*prev_input_position, 1);
  model_builder_.Copy(updated_input_position, *input_position);
  input_ids.push_back(prev_input_position->id);
  output_ids.push_back(input_position->id);

  if (mask) {
    // OP semantic: mask = prev_mask[0] + prev_mask[0:-1]
    *prev_mask = model_builder_.AddTensor(mask_desc);
    *mask = model_builder_.AddTensor(mask_desc);

    // Transpose mask as concatenating through channel axis generates too long
    // kernel and consumes a lot of compilation time.
    auto transposed = model_builder_.Transpose(*prev_mask, BHWC(0, 1, 3, 2));
    auto mask_0 = model_builder_.GetScalarTensor(transposed, BHWC(0, 0, 0, 0));
    auto mask_size = mask_desc.GetBHWDCShape().c;
    auto mask_rest = model_builder_.SubTensor(transposed, BHWC(0, 0, 0, 0),
                                              BHWC(1, 1, mask_size - 1, 1));
    auto transposed_updated =
        model_builder_.Concat(mask_0, mask_rest, ::ml_drift::Axis::kWidth);
    auto updated_mask =
        model_builder_.Transpose(transposed_updated, BHWC(0, 1, 3, 2));
    model_builder_.Copy(updated_mask, *mask);

    input_ids.push_back(prev_mask->id);
    output_ids.push_back(mask->id);
  }

  if (param) {
    // First 3 elements of params are used in LLM as [start, end, end, 0, ...]
    // for single KV cache update start/end index, and runtime BMM end index.
    // OP semantic: param = prev_param + size(prev_param) elementwise which does
    // +1 if param > 0.
    *prev_param = model_builder_.AddTensor(param_desc);
    *param = model_builder_.AddTensor(param_desc);
    auto sign_of_prev_param = model_builder_.Elementwise(
        *prev_param, ::ml_drift::OperationType::kSign);
    auto updated_param = model_builder_.Add(*prev_param, sign_of_prev_param);
    model_builder_.Copy(updated_param, *param);
    input_ids.push_back(prev_param->id);
    output_ids.push_back(param->id);
  }

  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

}  // namespace litert::lm
