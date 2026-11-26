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

#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder_moe_util.h"  // from @ml_drift
#include "ml_drift/common/kernels/elementwise.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/kernels/padding.h"  // from @ml_drift
#include "ml_drift/common/kernels/reduce.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_object_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/name_check.h"
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/add_with_overflow.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/apply_attention_mask.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/apply_fire_bias.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/flash_decoding_attention.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/mul_with_overflow.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/rotqat_rotate.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/transformer_params.pb.h"
#include "runtime/util/status_macros.h"

namespace ml_drift {
namespace {

const float kStaticKvQuantScale = 6.0f / 128.0f;

bool IsGlobalLayer(const TransformerConfig& config, int num_layer) {
  // By default, global layers are the last layer of the repeating pattern, like
  // LLLLGLLLLG...
  int global_layer_offset = config.num_local_layers_per_global;
  if (config.global_layer_offset.has_value()) {
    global_layer_offset = *config.global_layer_offset;
  }
  return (num_layer % (config.num_local_layers_per_global + 1)) ==
         global_layer_offset;
}

}  // namespace

int GetHeadSize(const TransformerConfig& config, bool is_global) {
  if (is_global && config.global_head_dimension.has_value()) {
    return *config.global_head_dimension;
  }
  return config.head_dimension;
}

int GetNumKvHeads(const TransformerConfig& config, bool is_global) {
  if (is_global && config.global_number_of_kv_heads.has_value()) {
    return *config.global_number_of_kv_heads;
  }
  return config.number_of_kv_heads;
}

namespace {

bool UseRingBuffer(const TransformerConfig& config) {
  return TransformerBuilder::UseSmallLocalCache(config) &&
         config.local_ring_cache_size.has_value();
}

constexpr int kNumActivationSparsityLayers = 10;

GPUOperation CreateFillSegmentPosOp(const TensorDescriptor& dst,
                                    int param_index) {
  GPUOperation op;
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::kInt32;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = MemoryType::kGlobal;
  op.AddSrcBuffer("src", buffer_desc);
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("param_index", param_index);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }
  args.dst::type res_value;
  int offset = args.src.Read(args.param_index);
  res_value.x = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.y = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.z = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.w = ucl::Convert<args.dst::scalar_type>(offset + X);
  args.dst.Write(res_value, X, Y, S);
}
)";
  return op;
}

// If remove_indices is nullptr, then the operation created will clean
// the kv cache AFTER set token offset index.
// If remove_indices is not nullptr, then the operation created will clean the
// kv cache at indices listed in the remove_indices input tensor.
std::unique_ptr<GPUOperation> CleanCacheOp(
    const GpuModelBuilder::TensorHandle& cache_k,
    const GpuModelBuilder::TensorHandle& cache_v,
    const TransformerConfig& config, int kv_cache_batch_size,
    const GpuModelBuilder::TensorHandle* remove_indices = nullptr,
    bool is_global = false) {
  bool create_clean_cache_at_idx_op = remove_indices != nullptr;

  std::string code = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.cache_size || Y >= args.batch_size || S >= args.head_size) {
    return;
  }

  int token_index = X;

  )";

  if (UseRingBuffer(config) && !is_global) {
    // `active_tokens` is the overall number of tokens we've produced into our
    // ring cache, so if our ring size is 8 and active_tokens=20, we'd have:
    // X:            0  1  2  3  4  5  6  7
    // token_index: 16 17 18 19 12 13 14 15
    // ring_offset: 16 16 16 16  8  8  8  8
    // Thus, we round down active_tokens-1 to the nearest ring size multiple,
    // and when our token_index passes active_tokens we wrap around by
    // removing one more ring_size if our ring has been filled at least once.
    code += absl::Substitute(R"(
  // Calculate token_index from ring cache position
  int active_tokens = args.params.Read($0);
  int ring_offset = (active_tokens - 1) / args.cache_size;
  ring_offset *= args.cache_size;
  token_index += ring_offset;
  if (token_index >= active_tokens && active_tokens >= args.cache_size) {
    token_index -= args.cache_size;
  })",
                             LlmRuntimeParams::kActiveTokensIndex);
  }

  if (!create_clean_cache_at_idx_op) {
    code += absl::Substitute(R"(
  // Only clear values after this time step.
  int time_step = args.params.Read($0);
  if (token_index >= args.global_cache_size || token_index < time_step) {
    return;
  })",
                             LlmRuntimeParams::kTokenOffsetIndex);
  } else {
    code += absl::Substitute(R"(
  // Or values listed in the remove_indices
  int remove_indices_size = args.params.Read($0);
  bool found = false;
  for (int i = 0; i < args.remove_indices_size; ++i) {
    int remove_index = args.remove_indices.Read(i);
    if (remove_index == token_index) {
      found = true;
      break;
    }
  }
  if (token_index >= args.global_cache_size || !found) {
    return;
  })",
                             LlmRuntimeParams::kToRemoveIndicesSize);
  }

  code += R"(
  token_index %= args.cache_size;
  {
    // cache_k - kOSpatialIOGroupO4I4 layout, O - cache_size, I - head_size
    // OGroup = o_slices -> actual layout SpatialIOGroupO4I4
    // SpatialIOGroupO4I4 Spatial ISlice OSlice O4 I4
    // index of vec4(I4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + O4Index
    int k_o = token_index;
    int k_o_slice = k_o / 4;
    int k_o4_index = k_o % 4;
    int k_i = S * 4;
    int k_i_slice = k_i / 4;
    int k_sp = Y;
    int i_slices = (args.head_size + 3) / 4;
    int o_slices = (args.cache_size + 3) / 4;
    int k_index = ((k_sp * i_slices + k_i_slice) * o_slices + k_o_slice) * 4 + k_o4_index;
    args.cache_k.WriteLinear(ucl::Init<args.cache_k::type>(0), k_index);
  }
  {
    // cache_v - kOSpatialIOGroupI4O4 layout, O - head_size, I - cache_size
    // OGroup = o_slices -> actual layout SpatialIOGroupI4O4
    // SpatialIOI4O4 Spatial ISlice OSlice I4 O4
    // index of vec4(O4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + I4Index
    int v_o = S * 4;
    int v_o_slice = v_o / 4;
    int v_i = token_index;
    int v_i_slice = v_i / 4;
    int v_i4_index = v_i % 4;
    int v_sp = Y;
    int i_slices = (args.cache_size + 3) / 4;
    int o_slices = (args.head_size + 3) / 4;
    int v_index = ((v_sp * i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    args.cache_v.WriteLinear(ucl::Init<args.cache_v::type>(0), v_index);
  }
})";

  class CleanCacheOp : public GPUOperation {
   public:
    CleanCacheOp() = default;
    int3 GetGridSize() const override {
      return int3(cache_size_, batch_size_, DivideRoundUp(head_size_, 4));
    }

    // Move only
    CleanCacheOp(CleanCacheOp&& operation) = default;
    CleanCacheOp& operator=(CleanCacheOp&& operation) = default;
    CleanCacheOp(const CleanCacheOp&) = delete;
    CleanCacheOp& operator=(const CleanCacheOp&) = delete;

    int batch_size_ = 0;
    int cache_size_ = 0;
    int head_size_ = 0;
  };

  const int head_size = GetHeadSize(config, is_global);
  const int kv_heads_count = GetNumKvHeads(config, is_global);

  CleanCacheOp custom_op;
  custom_op.batch_size_ = kv_cache_batch_size * kv_heads_count;
  custom_op.cache_size_ = TransformerBuilder::GetCacheSize(config, is_global);
  custom_op.head_size_ = head_size;
  custom_op.args_.AddInt("cache_size", custom_op.cache_size_);
  custom_op.args_.AddInt("batch_size", custom_op.batch_size_);
  custom_op.args_.AddInt("global_cache_size", config.cache_size);
  custom_op.args_.AddInt("head_size", head_size);
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);
  if (create_clean_cache_at_idx_op) {
    custom_op.AddSrcBuffer("remove_indices", params_buffer);
  }
  custom_op.AddDstTensor("cache_k", cache_k.tensor_desc);
  custom_op.AddDstTensor("cache_v", cache_v.tensor_desc);
  custom_op.code_ = code;

  return std::make_unique<CleanCacheOp>(std::move(custom_op));
}

float SoftPlus(float x) { return log1p(exp(-abs(x))) + std::max(x, 0.0f); }
float Sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
float GatingFun(float x) {
  float sigmoid_x = Sigmoid(x);
  float sigmoid_square_x = sigmoid_x * sigmoid_x;
  float small_x = sqrt(1 - sigmoid_square_x);
  float large_x = sigmoid_x * sqrt(2.0f * exp(-x) + exp(-2.0 * x));
  return x < 1 ? small_x : large_x;
}

std::vector<GpuModelBuilder::TensorHandle> FilterHandles(
    const std::vector<const GpuModelBuilder::TensorHandle*>& handles) {
  std::vector<GpuModelBuilder::TensorHandle> filtered;
  filtered.reserve(handles.size());
  for (auto* handle : handles) {
    if (handle) {
      filtered.push_back(*handle);
    }
  }
  return filtered;
}

// Replaces all handles in `inputs` with a tensor created from `builder` and
// returns a copy of the tensor handles.
std::vector<GpuModelBuilder::TensorHandle> CreateSubgraphInputs(
    GpuModelBuilder& builder,
    const std::vector<GpuModelBuilder::TensorHandle*>& inputs) {
  std::vector<GpuModelBuilder::TensorHandle> subgraph_inputs;
  subgraph_inputs.reserve(inputs.size());
  for (auto* in : inputs) {
    *in = builder.AddTensor(in->tensor_desc);
    subgraph_inputs.push_back(*in);
  }
  return subgraph_inputs;
}

// NOTE: We don't support PLE in MTP yet, so this remains hard-coded to
// the "transformer" prefix
std::string GetPerLayerEmbeddingNameInternal(int layer) {
  return absl::StrCat("transformer.layer_", layer, ".per_layer_embeddings");
}

std::string GetNgrammerEmbeddingNameInternal(int index) {
  return absl::StrCat("transformer.embedder.text_ngrammer.ngram_embedding_",
                      index);
}

// dst = builder.SelectV2(mask, neg_inf, src);
GPUOperation CreateSelectMaskOp(const TensorDescriptor& src,
                                const TensorDescriptor& mask,
                                const TensorDescriptor& dst) {
  ElementwiseDescriptor op_desc;
  if (src.GetDataType() == DataType::kFloat32) {
    op_desc.args.AddFloat("min_float",
                          std::numeric_limits<float>::max() * -0.7f);
  } else {
    op_desc.args.AddHalf("min_float", static_cast<half>(65504 * -0.7f));
  }
  op_desc.code = R"(
      out_value.x = in2_value.x ? args.min_float : in_value.x;
      out_value.y = in2_value.y ? args.min_float : in_value.y;
      out_value.z = in2_value.z ? args.min_float : in_value.z;
      out_value.w = in2_value.w ? args.min_float : in_value.w;
  )";
  OperationDef definition;
  definition.src_tensors.push_back(src);
  definition.src_tensors.push_back(mask);
  definition.dst_tensors.push_back(dst);
  return CreateGpuOperation(definition, std::move(op_desc), mask.GetBHWCShape(),
                            dst.GetBHWCShape());
}


FlashDecodingAttentionAttributes MakeFlashDecodingAttentionAttributes(
    const TransformerConfig& config, bool is_global) {
  FlashDecodingAttentionAttributes attr;
  attr.num_q_heads = config.number_of_heads;
  attr.num_kv_heads = GetNumKvHeads(config, is_global);
  attr.head_size = GetHeadSize(config, is_global);
  attr.cache_size = TransformerBuilder::GetCacheSize(config, is_global);
  attr.active_tokens_index = LlmRuntimeParams::kActiveTokensIndex;
  return attr;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFlashDecodingAttention(
    GpuModelBuilder& builder, const TransformerConfig& config,
    const GpuModelBuilder::TensorHandle& q,
    const GpuModelBuilder::TensorHandle& k_cache,
    const GpuModelBuilder::TensorHandle& v_cache,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global) {
  GpuModelBuilder::TensorHandle dst = builder.AddTensor(q.tensor_desc);
  ASSIGN_OR_RETURN(
      FlashDecodingAttention op,
      CreateFlashDecodingAttention(
          builder.gpu_info(), q.tensor_desc, k_cache.tensor_desc,
          v_cache.tensor_desc, dst.tensor_desc,
          MakeFlashDecodingAttentionAttributes(config, is_global)));
  builder.AddGpuOperation(
      {q, k_cache, v_cache, params_tensor}, {dst},
      std::make_unique<FlashDecodingAttention>(std::move(op)),
      "flash_decoding_attention");
  return dst;
}

}  // namespace

GpuModelBuilder::TensorHandle MakeSelectMaskOp(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& mask) {
  GpuModelBuilder::TensorHandle dst = builder.AddTensor(src.tensor_desc);
  auto select_mask_op = std::make_unique<GPUOperation>(
      CreateSelectMaskOp(src.tensor_desc, mask.tensor_desc, dst.tensor_desc));
  builder.AddGpuOperation({src, mask}, {dst}, std::move(select_mask_op),
                          "select_mask");
  return dst;
}

// Transforms the given index based on `config` to skip the specified layers.
int SkipMatformIndex(int i, const TransformerConfig& config) {
  if (i < config.layer_skip_start) {
    return i;
  }
  return i + config.num_layers_skipped;
}

std::unique_ptr<GPUOperation> CreateFillTensorOp(const TensorDescriptor& dst) {
  if (dst.GetLayout() == Layout::kLinear) {
    constexpr int kXSize = 32768;
    class FillOp : public GPUOperation {
     public:
      FillOp() = default;
      int3 GetGridSize() const override {
        int grid_size = dst_[0]->Batch() * dst_[0]->Width() *
                        dst_[0]->Height() * dst_[0]->Slices();
        return int3(kXSize, DivideRoundUp(grid_size, kXSize), 1);
      }

      // Move only
      FillOp(FillOp&& operation) = default;
      FillOp& operator=(FillOp&& operation) = default;
      FillOp(const FillOp&) = delete;
      FillOp& operator=(const FillOp&) = delete;
    };
    FillOp op;
    op.AddDstTensor("dst", dst);
    op.args_.AddInt("fill_value", 0);

    op.code_ = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int index = ucl::GetGlobalId<1>() * $0 + ucl::GetGlobalId<0>();
  if (index >= args.dst.Batch() * args.dst.Width() * args.dst.Height() * args.dst.Slices()) {
    return;
  }
  args.dst::type res_value;
  res_value.x = ucl::Convert<args.dst::scalar_type>(args.fill_value);
  res_value.y = ucl::Convert<args.dst::scalar_type>(args.fill_value);
  res_value.z = ucl::Convert<args.dst::scalar_type>(args.fill_value);
  res_value.w = ucl::Convert<args.dst::scalar_type>(args.fill_value);
  args.dst.WriteLinear(res_value, index);
}
)",
                                kXSize);
    return std::make_unique<FillOp>(std::move(op));
  }
  GPUOperation op;
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("fill_value", 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  op.code_ = R"(
MAIN_FUNCTION($0) {
  int linear_xb = ucl::GetGlobalId<0>();
  int X = linear_xb / args.dst.Batch();
  int B = linear_xb % args.dst.Batch();
  args.dst.SetBatchRef(B);
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }
  args.dst::type res_value;
  res_value.x = ucl::Convert<args.dst::scalar_type>(args.fill_value);
  res_value.y = ucl::Convert<args.dst::scalar_type>(args.fill_value);
  res_value.z = ucl::Convert<args.dst::scalar_type>(args.fill_value);
  res_value.w = ucl::Convert<args.dst::scalar_type>(args.fill_value);
  args.dst.Write(res_value, X, Y, S);
}
)";
  return std::make_unique<GPUOperation>(std::move(op));
}

CopySubLinearTensorOp::CopySubLinearTensorOp(const TensorDescriptor& src,
                                             const TensorDescriptor& dst,
                                             int offset)
    : GPUOperation() {
  AddSrcTensor("src", src);
  AddDstTensor("dst", dst);
  args_.AddInt("offset", offset);
  code_ = R"(
  MAIN_FUNCTION($0) {
    int X = ucl::GetGlobalId<0>();
    if (X >= args.dst.Slices()) {
      return;
    }
    args.dst::type result = args.src.Read(X + args.offset);
    args.dst.WriteLinear(result, X);
  })";
}

int3 CopySubLinearTensorOp::GetGridSize() const {
  return int3(dst_[0]->Slices(), 1, 1);
}

void TransformerBuilder::AddDebugTensor(
    const std::string& name, const GpuModelBuilder::TensorHandle& tensor) {
  if (!debug_tensors_.contains(name)) {
    // Transpose twice to prevent linking from altering the values we want to
    // inspect.
    debug_tensors_[name] = model_builder_.Transpose(
        model_builder_.Transpose(tensor, BHWC(0, 3, 2, 1)), BHWC(0, 3, 2, 1));
  }
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeRMSNormalization(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    float scale_shift) {
  return MakeRMSNormalizationWithBuilder(model_builder_, src, name,
                                         scale_shift);
}

absl::StatusOr<GpuModelBuilder::Weights>
TransformerBuilder::GetWeightsFloatTensor(const std::string& w_name,
                                          Layout layout, int in_channels,
                                          int out_channels, DataType data_type,
                                          int weights_batch) {
  const OHWI weights_shape = OHWI(out_channels, weights_batch, 1, in_channels);
  auto weights_desc =
      model_builder_.GetFullyConnectedWeightsDesc(data_type, weights_shape);
  ASSIGN_OR_RETURN(
      auto* weights_tensor,
      tensor_loader_->LoadWeights(w_name, weights_desc, weights_shape,
                                  layout == Layout::kIOHW));
  return CreateExternalWeights(AddExternalTensor(weights_tensor), weights_desc,
                               weights_shape);
}

void TransformerBuilder::RegisterF16RescaleMultiplier(
    const std::string& scales_name,
    const GpuModelBuilder::TensorHandle& scale_th) {
  // The converter may have multiplied quantization scales by a power of two k
  // to avoid F16 subnormals (which some GPUs flush to zero), storing 1/k
  // alongside. MakeConv applies 1/k to the conv output.
  const std::string inv_name =
      absl::StrCat(scales_name, odml::infra::gpu::kF16RescaleInverseSuffix);
  if (!tensor_loader_->HasTensor(inv_name)) {
    return;
  }
  const std::vector<float> inv = tensor_loader_->LoadFloat32(inv_name, 1);
  if (inv.size() == 1 && inv[0] != 1.0f) {
    weights_output_multipliers_[scale_th.id] = inv[0];
  }
}

absl::StatusOr<GpuModelBuilder::Weights> TransformerBuilder::GetWeightsI8Tensor(
    const std::string& w_name, Layout layout, int in_channels, int out_channels,
    int weights_batch) {
  const std::string scales_name = w_name + "_quantized_scale";
  const std::string zero_point_name = w_name + "_quantized_zp";
  const OHWI weights_shape = OHWI(out_channels, weights_batch, 1, in_channels);
  auto weights_desc =
      model_builder_.GetFullyConnectedInt8WeightsDesc(weights_shape);
  const int quantization_i_group_size =
      tensor_loader_->GetQuantizationIGroupSize(w_name, weights_shape);
  const int scale_i_ch = weights_shape.i / quantization_i_group_size;
  const OHWI scale_zp_shape =
      OHWI(weights_shape.o, weights_shape.h, 1, scale_i_ch);
  ASSIGN_OR_RETURN(
      auto* scale_tensor,
      tensor_loader_->LoadScale(scales_name, scale_zp_shape, std::nullopt));
  ASSIGN_OR_RETURN(
      const auto weights_i8_tensor,
      tensor_loader_->LoadInt8Weights(w_name, weights_desc, weights_shape,
                                      layout == Layout::kIOHW));

  auto weights_i8_th = AddExternalTensor(weights_i8_tensor.weights);
  GpuModelBuilder::TensorHandle* weights_sum_i_th_ptr = nullptr;
  GpuModelBuilder::TensorHandle weights_sum_i_th;
  if (weights_i8_tensor.weights_sum_i) {
    weights_sum_i_th = AddExternalTensor(weights_i8_tensor.weights_sum_i);
    weights_sum_i_th_ptr = &weights_sum_i_th;
  }
  auto weights_scale_th = AddExternalTensor(scale_tensor);
  RegisterF16RescaleMultiplier(scales_name, weights_scale_th);
  GpuModelBuilder::TensorHandle* zero_point_th_ptr = nullptr;
  GpuModelBuilder::TensorHandle zero_point_th;
  if (tensor_loader_->HasTensor(zero_point_name)) {
    ASSIGN_OR_RETURN(auto* zero_point_tensor,
                     tensor_loader_->LoadZeroPoint(zero_point_name, scales_name,
                                                   scale_zp_shape));
    zero_point_th = AddExternalTensor(zero_point_tensor);
    zero_point_th_ptr = &zero_point_th;
  }

  return CreateExternalWeights(weights_i8_th, weights_desc, weights_shape,
                               scale_zp_shape, &weights_scale_th,
                               zero_point_th_ptr, weights_sum_i_th_ptr);
}

absl::StatusOr<GpuModelBuilder::Weights> TransformerBuilder::GetWeightsI4Tensor(
    const std::string& w_name, Layout layout, int in_channels, int out_channels,
    std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
        matform_options,
    int weights_batch) {
  const std::string scales_name = w_name + "_quantized_scale";
  const std::string zero_point_name = w_name + "_quantized_zp";
  const OHWI weights_shape = OHWI(out_channels, weights_batch, 1, in_channels);
  auto weights_desc =
      model_builder_.GetFullyConnectedInt4WeightsDesc(weights_shape);
  const int quantization_i_group_size =
      tensor_loader_->GetQuantizationIGroupSize(w_name, weights_shape);
  int scale_i_ch = weights_shape.i / quantization_i_group_size;
  if (matform_options) {
    scale_i_ch = 1;
  }
  const OHWI scale_zp_shape =
      OHWI(weights_shape.o, weights_shape.h, 1, scale_i_ch);
  ASSIGN_OR_RETURN(
      auto* scale_tensor,
      tensor_loader_->LoadScale(scales_name, scale_zp_shape, matform_options));
  ASSIGN_OR_RETURN(const auto weights_i8_tensor,
                   tensor_loader_->LoadInt4Weights(
                       w_name, weights_desc, weights_shape,
                       layout == Layout::kIOHW, matform_options));

  auto weights_i8_th = AddExternalTensor(weights_i8_tensor.weights);
  GpuModelBuilder::TensorHandle* weights_sum_i_th_ptr = nullptr;
  GpuModelBuilder::TensorHandle weights_sum_i_th;
  if (weights_i8_tensor.weights_sum_i) {
    weights_sum_i_th = AddExternalTensor(weights_i8_tensor.weights_sum_i);
    weights_sum_i_th_ptr = &weights_sum_i_th;
  }
  auto weights_scale_th = AddExternalTensor(scale_tensor);
  RegisterF16RescaleMultiplier(scales_name, weights_scale_th);
  GpuModelBuilder::TensorHandle* zero_point_th_ptr = nullptr;
  GpuModelBuilder::TensorHandle zero_point_th;
  if (tensor_loader_->HasTensor(zero_point_name)) {
    ASSIGN_OR_RETURN(auto* zero_point_tensor,
                     tensor_loader_->LoadZeroPoint(zero_point_name, scales_name,
                                                   scale_zp_shape));
    zero_point_th = AddExternalTensor(zero_point_tensor);
    zero_point_th_ptr = &zero_point_th;
  }

  return CreateExternalWeights(weights_i8_th, weights_desc, weights_shape,
                               scale_zp_shape, &weights_scale_th,
                               zero_point_th_ptr, weights_sum_i_th_ptr);
}

absl::StatusOr<GpuModelBuilder::Weights> TransformerBuilder::GetWeightsI2Tensor(
    const std::string& w_name, Layout layout, int in_channels, int out_channels,
    int weights_batch) {
  const std::string scales_name = w_name + "_quantized_scale";
  const std::string zero_point_name = w_name + "_quantized_zp";
  const OHWI weights_shape = OHWI(out_channels, weights_batch, 1, in_channels);
  auto weights_desc =
      model_builder_.GetFullyConnectedInt2WeightsDesc(weights_shape);
  const int quantization_i_group_size =
      tensor_loader_->GetQuantizationIGroupSize(w_name, weights_shape);
  const int scale_i_ch = weights_shape.i / quantization_i_group_size;
  const OHWI scale_zp_shape =
      OHWI(weights_shape.o, weights_shape.h, 1, scale_i_ch);
  ASSIGN_OR_RETURN(
      auto* scale_tensor,
      tensor_loader_->LoadScale(scales_name, scale_zp_shape, std::nullopt));
  ASSIGN_OR_RETURN(
      const auto weights_i2_tensor,
      tensor_loader_->LoadInt2Weights(w_name, weights_desc, weights_shape,
                                      layout == Layout::kIOHW));

  auto weights_i2_th = AddExternalTensor(weights_i2_tensor.weights);
  GpuModelBuilder::TensorHandle* weights_sum_i_th_ptr = nullptr;
  GpuModelBuilder::TensorHandle weights_sum_i_th;
  if (weights_i2_tensor.weights_sum_i) {
    weights_sum_i_th = AddExternalTensor(weights_i2_tensor.weights_sum_i);
    weights_sum_i_th_ptr = &weights_sum_i_th;
  }
  auto weights_scale_th = AddExternalTensor(scale_tensor);
  RegisterF16RescaleMultiplier(scales_name, weights_scale_th);
  GpuModelBuilder::TensorHandle* zero_point_th_ptr = nullptr;
  GpuModelBuilder::TensorHandle zero_point_th;
  if (tensor_loader_->HasTensor(zero_point_name)) {
    ASSIGN_OR_RETURN(auto* zero_point_tensor,
                     tensor_loader_->LoadZeroPoint(zero_point_name, scales_name,
                                                   scale_zp_shape));
    zero_point_th = AddExternalTensor(zero_point_tensor);
    zero_point_th_ptr = &zero_point_th;
  }

  return CreateExternalWeights(weights_i2_th, weights_desc, weights_shape,
                               scale_zp_shape, &weights_scale_th,
                               zero_point_th_ptr, weights_sum_i_th_ptr);
}

absl::StatusOr<GpuModelBuilder::Weights> TransformerBuilder::GetWeightsTensor(
    const std::string& w_name, Layout layout, int in_channels, int out_channels,
    DataType float_type,
    std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
        matform_options,
    int weights_batch) {
  int weights_count = in_channels * out_channels * weights_batch;
  if (matform_options) {
    weights_count *= matform_options->matform_factor;
  }
  DataType data_type =
      tensor_loader_->GetDataTypeForWeights(w_name, weights_count);
  // TODO: b/422747873 - Implement for more data types if needed.
  RET_CHECK(!matform_options || data_type == DataType::kInt4)
      << "Matform only implemented for int4";

  GpuModelBuilder::Weights weights;
  if (data_type == DataType::kInt8) {
    ASSIGN_OR_RETURN(weights, GetWeightsI8Tensor(w_name, layout, in_channels,
                                                 out_channels, weights_batch));
  } else if (data_type == DataType::kInt4) {
    ASSIGN_OR_RETURN(
        weights, GetWeightsI4Tensor(w_name, layout, in_channels, out_channels,
                                    matform_options, weights_batch));
  } else if (data_type == DataType::kInt2) {
    ASSIGN_OR_RETURN(weights, GetWeightsI2Tensor(w_name, layout, in_channels,
                                                 out_channels, weights_batch));
  } else {
    ASSIGN_OR_RETURN(weights, GetWeightsFloatTensor(w_name, layout, in_channels,
                                                    out_channels, float_type,
                                                    weights_batch));
  }
  return weights;
}

absl::StatusOr<GpuModelBuilder::Weights>
TransformerBuilder::GetLoraWeightsTensor(const std::string& w_name,
                                         Layout layout, int in_channels,
                                         int out_channels, DataType data_type,
                                         int lora_rank) {
  const OHWI weights_shape = OHWI(out_channels, 1, 1, in_channels);
  auto weights_desc =
      model_builder_.GetFullyConnectedWeightsDesc(data_type, weights_shape);
  ASSIGN_OR_RETURN(
      auto* weights_tensor,
      tensor_loader_->LoadWeights(w_name, weights_desc, weights_shape,
                                  layout == Layout::kIOHW));

  auto weights_th = AddExternalMutableTensor(weights_tensor->GetDescriptor());
  lora_tensors_[lora_rank][w_name] = LoRATensorInfo{
      .weights_desc = weights_desc,
      .shape = weights_shape,
      .id = weights_th.id,
      .swap_dims = layout == Layout::kIOHW,
  };
  return CreateExternalWeights(weights_th, weights_desc, weights_shape);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeLoRAInjection(const GpuModelBuilder::TensorHandle& src,
                                      const std::string& name,
                                      Layout src_layout, int out_channels,
                                      int lora_rank) {
  if (src_layout != Layout::kIOHW && src_layout != Layout::kOIHW) {
    return absl::InvalidArgumentError("Expected IOHW or OIHW `src_layout`.");
  }
  // B - batch size, S - seq len, D - model dim
  // N - num heads, H - head dim, R - LoRA rank
  // Left LoRA weights have shape DR, right LoRA weights have shape RNH
  // QKV weights:
  // - weights are loaded with IOHW layout
  // - einsum: BxSxD x DxNxH -> BxSxNxH
  // - LoRA transformations should be applied with the left weights first:
  //   BxSxD x DxR -> BxSxR, BxSxR x RxNxH -> BxSxNxH
  // Post attention weights (for Gemini V1)
  // - weights are loaded with OIHW layout
  // - einsum: BxSxNxH x DxNxH -> BxSxD
  // - LoRA transformations should be applied with the right weights first:
  //   BxSxNxH x RxNxH -> BxSxR, BxSxR x DxR -> BxSxD

  // LoRA weights are assumed to be encoded as floats.
  const std::string first_lora_layer =
      src_layout == Layout::kIOHW ? "_prime_left" : "_prime_right";
  const std::string first_name = name + ".w" + first_lora_layer;
  ASSIGN_OR_RETURN(auto first_weights,
                   GetLoraWeightsTensor(
                       first_name, src_layout, src.tensor_desc.GetBHWCShape().c,
                       lora_rank, src.tensor_desc.GetDataType(), lora_rank));
  ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle intermediate_layer,
                   MakeConv(CapInfinity(src), first_weights));

  const std::string second_lora_layer =
      src_layout == Layout::kIOHW ? "_prime_right" : "_prime_left";
  const std::string second_name = name + ".w" + second_lora_layer;
  ASSIGN_OR_RETURN(
      auto second_weights,
      GetLoraWeightsTensor(
          second_name, src_layout,
          intermediate_layer.tensor_desc.GetBHWCShape().c, out_channels,
          intermediate_layer.tensor_desc.GetDataType(), lora_rank));
  ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle pre_scaling_layer,
                   MakeConv(CapInfinity(intermediate_layer), second_weights));

  // Note: We do not scale V3 LoRAs by rank currently, so as to process them
  // more similarly to external.
  // TODO: We should not be hard-coding this scaling factor for GemmaModelV3.
  // We do this now as a workaround for scale_lora from the model being
  // overwritten by the MLDriftOptions settings in a Session.
  if (UseGoogleInternalModelV2() || config_.scale_lora ||
      UseGemmaModelV3NPlus()) {
    return model_builder_.Multiplication(pre_scaling_layer, 1.0f / lora_rank);
  } else {
    return pre_scaling_layer;
  }
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::MakeConv(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::Weights& weights,
    const GpuModelBuilder::TensorHandle* bias_th) {
  // If the converter pre-multiplied the scales to avoid F16 subnormals (see
  // RegisterF16RescaleMultiplier), the conv result has to be scaled back. The
  // bias must be added after that, so it is excluded from the conv itself.
  float output_multiplier = 1.0f;
  if (weights.scale) {
    auto it = weights_output_multipliers_.find(weights.scale->id);
    if (it != weights_output_multipliers_.end()) {
      output_multiplier = it->second;
    }
  }
  const GpuModelBuilder::TensorHandle* post_bias_th = nullptr;
  if (output_multiplier != 1.0f && bias_th) {
    post_bias_th = bias_th;
    bias_th = nullptr;
  }
  std::string id = GetConvId(src, weights.shape, weights.desc);
  if (post_bias_th) {
    id += ":no_bias";
  }
  auto inputs = FilterHandles(
      {&src, &weights.weights, weights.sum_i ? &*weights.sum_i : nullptr,
       weights.scale ? &*weights.scale : nullptr,
       weights.zero_point ? &*weights.zero_point : nullptr, bias_th});
  if (!model_builder_.HasSubgraph(id)) {
    GpuModelBuilder builder = model_builder_.CreateBuilder();
    std::vector<GpuModelBuilder::TensorHandle> subgraph_inputs;
    subgraph_inputs.reserve(inputs.size());
    auto add_tensor = [&](const GpuModelBuilder::TensorHandle* handle)
        -> GpuModelBuilder::TensorHandle* {
      if (!handle) return nullptr;
      return &subgraph_inputs.emplace_back(
          builder.AddTensor(handle->tensor_desc));
    };
    // Make sure all inputs are tracked in `builder`.
    auto* sub_src = add_tensor(&src);
    auto* sub_weights = add_tensor(&weights.weights);
    auto* sub_sum_i = add_tensor(weights.sum_i ? &*weights.sum_i : nullptr);
    auto* sub_scale = add_tensor(weights.scale ? &*weights.scale : nullptr);
    auto* sub_zero_point =
        add_tensor(weights.zero_point ? &*weights.zero_point : nullptr);
    auto* sub_bias = add_tensor(bias_th);
    const GpuModelBuilder::Weights sub_external_weights = CreateExternalWeights(
        *sub_weights, weights.desc, weights.shape, weights.scale_zp_shape,
        sub_scale, sub_zero_point, sub_sum_i);
    GpuModelBuilder::TensorHandle conv;
    if (SizeInBitsOf(weights.desc.type) == 8) {
      conv = builder.FullyConnectedInt8ExternalWeights(
          *sub_src, sub_external_weights, sub_bias);
    } else if (SizeInBitsOf(weights.desc.type) == 4) {
      conv = builder.FullyConnectedInt4ExternalWeights(
          *sub_src, sub_external_weights, sub_bias);
    } else if (SizeInBitsOf(weights.desc.type) == 2) {
      conv = builder.FullyConnectedInt2ExternalWeights(
          *sub_src, sub_external_weights, sub_bias);
    } else {
      ASSIGN_OR_RETURN(conv, builder.FullyConnectedExternalWeights(
                                 *sub_src, sub_external_weights, sub_bias));
    }
    RETURN_IF_ERROR(model_builder_.RegisterSubgraph(std::move(builder), id,
                                                    subgraph_inputs, {conv}));
  }
  ASSIGN_OR_RETURN(auto dsts, model_builder_.Subgraph(id, inputs));
  if (output_multiplier == 1.0f) {
    return dsts[0];
  }
  GpuModelBuilder::TensorHandle result =
      model_builder_.Multiplication(dsts[0], output_multiplier);
  if (post_bias_th) {
    result = model_builder_.Add(result, *post_bias_th);
  }
  return result;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeConvAutoPrecision(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    Layout src_layout, int out_channels) {
  return MakeConvAutoPrecision(src, name, src_layout, out_channels,
                               /*lora_src=*/nullptr,
                               /*matform_options=*/std::nullopt);
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::AddLoRA(
    const GpuModelBuilder::TensorHandle& lora_input,
    const GpuModelBuilder::TensorHandle& base, const std::string& name,
    Layout layout) {
  if (config_.lora_ranks.empty() ||
      (!odml::infra::gpu::IsAttentionTensorName(name) &&
       !odml::infra::gpu::IsFeedForwardTensorName(name))) {
    return base;
  }

  int channels = base.tensor_desc.GetBHWCShape().c;
  for (int lora_rank : config_.lora_ranks) {
    int tag = lora_rank;
    if (odml::infra::gpu::IsAttentionTensorName(name)) {
      tag |= kLoRAAttentionOptionTag;
    } else {
      tag |= kLoRAFeedForwardOptionTag;
    }
    auto context = model_builder_.BeginOptionalNodes(tag, base);
    ASSIGN_OR_RETURN(
        auto lora_injection,
        MakeLoRAInjection(lora_input, name, layout, channels, lora_rank));
    auto lora_result = model_builder_.Add(base, lora_injection);
    RETURN_IF_ERROR(model_builder_.EndOptionalNodes(context, lora_result));
  }
  return base;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeAutoPrecisionOIConvWithIds(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& batched_ids, const std::string& name,
    int out_channels) {
  const int weights_batch = config_.moe_config->num_experts;
  ASSIGN_OR_RETURN(auto weights,
                   GetWeightsTensor(name + ".w", Layout::kOIHW,
                                    src.tensor_desc.GetBHWCShape().c,
                                    out_channels, src.tensor_desc.GetDataType(),
                                    std::nullopt, weights_batch));
  GpuModelBuilder::TensorHandle conv;
  if (src.tensor_desc.GetBHWCShape().h == 1 &&
      src.tensor_desc.GetBHWCShape().w != 1) {
    ASSIGN_OR_RETURN(conv, ml_drift::MakeConvWithPackedGroups(
                               model_builder_, src, batched_ids, weights,
                               config_.moe_config->num_active_experts));
  } else {
    ASSIGN_OR_RETURN(
        conv, MakeConvWithBatchIds(model_builder_, src, batched_ids, weights));
  }
  // Undo any converter rescaling of F16-subnormal scales (see MakeConv).
  if (weights.scale) {
    auto it = weights_output_multipliers_.find(weights.scale->id);
    if (it != weights_output_multipliers_.end()) {
      conv = model_builder_.Multiplication(conv, it->second);
    }
  }
  return conv;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeConvAutoPrecision(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    Layout src_layout, int out_channels,
    const GpuModelBuilder::TensorHandle* lora_src,
    std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
        matform_options) {
  bool bias = tensor_loader_->HasTensor(name + ".bias.b");
  const std::string w_name = name + (bias ? ".linear" : "") + ".w";
  ASSIGN_OR_RETURN(
      auto weights,
      GetWeightsTensor(w_name, src_layout, src.tensor_desc.GetBHWCShape().c,
                       out_channels, src.tensor_desc.GetDataType(),
                       matform_options));
  GpuModelBuilder::TensorHandle* bias_th_ptr = nullptr;
  GpuModelBuilder::TensorHandle bias_th;
  if (bias) {
    const std::string b_name = name + ".bias.b";
    ASSIGN_OR_RETURN(auto* bias_tensor,
                     tensor_loader_->LoadBias(b_name, out_channels));
    bias_th = AddExternalTensor(bias_tensor);
    bias_th_ptr = &bias_th;
  }

  ASSIGN_OR_RETURN(auto conv, MakeConv(src, weights, bias_th_ptr));

  GpuModelBuilder::TensorHandle lora_input = lora_src ? *lora_src : src;
  return AddLoRA(lora_input, conv, name, src_layout);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeLinearAutoPrecisionWithSrq(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels, bool apply_srq,
    std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
        matform_options) {
  const std::string input_scale_name = name + ".input_activation_static_scale";
  const std::string output_scale_name =
      name + ".output_activation_static_scale";

  GpuModelBuilder::TensorHandle input =
      apply_srq ? MakeFakeQuant(src, input_scale_name) : src;
  ASSIGN_OR_RETURN(
      GpuModelBuilder::TensorHandle output,
      MakeLinearAutoPrecision(input, name, out_channels, /*lora_src=*/nullptr,
                              matform_options));
  return apply_srq ? MakeFakeQuant(output, output_scale_name) : output;
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeFakeQuant(
    const GpuModelBuilder::TensorHandle& src, const std::string& scale_name,
    DataType data_type) {
  const std::vector<float> scale = tensor_loader_->LoadFloat32(scale_name, 1);
  QuantizeAndDequantizeAttributes attr;
  attr.scale = scale[0];
  const int bits = SizeInBitsOf(data_type);
  const float max_unscaled_value = static_cast<float>((1 << (bits - 1)) - 1);
  const float min_unscaled_value = -max_unscaled_value - 1.0f;
  // We want to clip the value/scale in between min_unscaled_value and
  // max_unscaled_value. MLD op first clips the value to [attr.min, attr.max]
  // and then scales. So we need to set the min/max to min_unscaled_value *
  // scale or max_unscaled_value * scale.
  attr.min = min_unscaled_value * scale[0];
  attr.max = max_unscaled_value * scale[0];
  return model_builder_.QuantizeAndDequantize(src, attr);
}

float TransformerBuilder::GetQuantScale(const std::string& scale_name,
                                        const DataType& src_type) {
  const std::vector<float> scales = tensor_loader_->LoadFloat32(scale_name, 1);
  float adjusted_scale = scales[0];
  const bool is_fp16 = src_type == DataType::kFloat16;
  if (is_fp16 && adjusted_scale < 0.000062f) {
    // The smallest positive normal number for Half-precision floating-point
    // format is 2^-14 ~ 0.000062f. Therefore, if the scale is lesser than this
    // number, we just reset it accordingly.
    adjusted_scale = 0.000062f;
  }
  return adjusted_scale;
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeQuantInt8(
    const GpuModelBuilder::TensorHandle& src, const std::string& scale_name,
    const DataType& scale_type) {
  const std::vector<float> scale = tensor_loader_->LoadFloat32(scale_name, 1);
  return MakeQuantInt8(src, scale[0], scale_type);
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeQuantInt8(
    const GpuModelBuilder::TensorHandle& src, float scale,
    const DataType& scale_type) {
  QuantizeAndDequantizeAttributes attr;
  attr.scale = scale;
  const int bits = SizeInBitsOf(scale_type);
  const float max_unscaled_value = static_cast<float>((1 << (bits - 1)) - 1);
  const float min_unscaled_value = -max_unscaled_value - 1.0f;
  // We want to clip the value/scale in between min_unscaled_value and
  // max_unscaled_value. MLD op first clips the value to [attr.min, attr.max]
  // and then scales. So we need to set the min/max to min_unscaled_value *
  // scale or max_unscaled_value * scale.
  attr.min = min_unscaled_value * scale;
  attr.max = max_unscaled_value * scale;
  GpuModelBuilder::TensorHandle range_quantized =
      model_builder_.StaticRangeQuantization(src, attr);
  // The int8 range is built into the FullyConnected operation when weights are
  // int8, so to account for proper dequant. later on, we must adjust our range
  // when casting from uint4 values to uint8. Once we have UINT4 cache tensors,
  // this will no longer be necessary.
  if (bits == 4) {
    range_quantized = model_builder_.Add(range_quantized, 128 - 8);
  }

  return model_builder_.Cast(range_quantized, DataType::kUint8);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeLinearAutoPrecision(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels) {
  return MakeLinearAutoPrecision(src, name, out_channels, /*lora_src=*/nullptr,
                                 /*matform_options=*/std::nullopt);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeLinearAutoPrecision(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels, const GpuModelBuilder::TensorHandle* lora_src,
    std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
        matform_options) {
  return MakeConvAutoPrecision(src, name, Layout::kIOHW, out_channels, lora_src,
                               matform_options);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeInputLoRAInjection(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels) {
  GpuModelBuilder::TensorHandle lora_input = src;
  auto layer_name = name + ".input";
  if (!config_.lora_ranks.empty() &&
      tensor_loader_->HasTensor(layer_name + ".w_prime_left") &&
      tensor_loader_->HasTensor(layer_name + ".w_prime_right")) {
    Layout src_layout = Layout::kIOHW;
    for (int lora_rank : config_.lora_ranks) {
      int tag = lora_rank | kLoRAFeedForwardOptionTag;
      auto context = model_builder_.BeginOptionalNodes(tag, lora_input);
      ASSIGN_OR_RETURN(auto lora_injection,
                       MakeLoRAInjection(lora_input, layer_name, src_layout,
                                         config_.model_dimension, lora_rank));
      auto lora_result = model_builder_.Add(lora_input, lora_injection);
      RETURN_IF_ERROR(model_builder_.EndOptionalNodes(context, lora_result));
    }
  }
  return lora_input;
}

void TransformerBuilder::AddValuesToCache(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& k,
    const GpuModelBuilder::TensorHandle& v,
    const GpuModelBuilder::TensorHandle& params_tensor,
    const GpuModelBuilder::TensorHandle& cache_k,
    const GpuModelBuilder::TensorHandle& cache_v, bool is_global) {
  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.src_k.Width() || Y >= args.batch_size || S >= args.src_k.Slices()) {
    return;
  }
  int token_index_offset = args.params.Read($0);
  int active_tokens = args.params.Read($1);
  int token_index = token_index_offset + X;
  if (token_index >= args.global_cache_size || token_index >= active_tokens) {
    return;
  }

  token_index %= args.cache_size;

  int src_y = Y % args.src_k.Height();  // broadcast Height dim used as Batch
  args.src_k::type value_k = args.src_k.Read(X, src_y, S);
  args.src_v::type value_v = args.src_v.Read(X, src_y, S);
  {
    // cache_k - kOSpatialIOGroupO4I4 layout, O - cache_size, I - head_size
    // OGroup = o_slices -> actual layout SpatialIOGroupO4I4
    // SpatialIOGroupO4I4 Spatial ISlice OSlice O4 I4
    // index of vec4(I4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + O4Index
    int k_o = token_index;
    int k_o_slice = k_o / 4;
    int k_o4_index = k_o % 4;
    int k_i = S * 4;
    int k_i_slice = k_i / 4;
    int k_sp = Y;
    int i_slices = (args.head_size + 3) / 4;
    int o_slices = (args.cache_size + 3) / 4;
    int k_index = ((k_sp * i_slices + k_i_slice) * o_slices + k_o_slice) * 4 + k_o4_index;
    args.cache_k.WriteLinear(value_k, k_index);
  }
  {
    // cache_v - kOSpatialIOGroupI4O4 layout, O - head_size, I - cache_size
    // OGroup = o_slices -> actual layout SpatialIOGroupI4O4
    // SpatialIOI4O4 Spatial ISlice OSlice I4 O4
    // index of vec4(O4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + I4Index
    int v_o = S * 4;
    int v_o_slice = v_o / 4;
    int v_i = token_index;
    int v_i_slice = v_i / 4;
    int v_i4_index = v_i % 4;
    int v_sp = Y;
    int i_slices = (args.cache_size + 3) / 4;
    int o_slices = (args.head_size + 3) / 4;
    int v_index = ((v_sp * i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    args.cache_v.WriteLinear(value_v, v_index);
  }
})",
                                      LlmRuntimeParams::kTokenOffsetIndex,
                                      LlmRuntimeParams::kActiveTokensIndex);

  class AddValuesToCacheOp : public GPUOperation {
   public:
    AddValuesToCacheOp() = default;
    int3 GetGridSize() const override {
      return int3(src_[0]->Width(), batch_size_, src_[0]->Slices());
    }

    // Move only
    AddValuesToCacheOp(AddValuesToCacheOp&& operation) = default;
    AddValuesToCacheOp& operator=(AddValuesToCacheOp&& operation) = default;
    AddValuesToCacheOp(const AddValuesToCacheOp&) = delete;
    AddValuesToCacheOp& operator=(const AddValuesToCacheOp&) = delete;

    int batch_size_ = 1;
  };

  const int head_size = GetHeadSize(config_, is_global);
  const int kv_heads_count = GetNumKvHeads(config_, is_global);

  AddValuesToCacheOp custom_op;
  custom_op.batch_size_ = kv_cache_batch_size_ * kv_heads_count;
  custom_op.args_.AddInt("cache_size",
                         TransformerBuilder::GetCacheSize(config_, is_global));
  custom_op.args_.AddInt("batch_size", custom_op.batch_size_);
  custom_op.args_.AddInt("global_cache_size", config_.cache_size);
  custom_op.args_.AddInt("head_size", head_size);
  custom_op.AddSrcTensor("src_k", k.tensor_desc);
  custom_op.AddSrcTensor("src_v", v.tensor_desc);
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);
  custom_op.AddDstTensor("cache_k", cache_k.tensor_desc);
  custom_op.AddDstTensor("cache_v", cache_v.tensor_desc);
  custom_op.code_ = code;

  custom_op.write_size_ = k.tensor_desc.GetMemorySizeInBytes() +
                          v.tensor_desc.GetMemorySizeInBytes();

  builder.AddGpuOperation(
      {k, v, params_tensor}, {cache_k, cache_v},
      std::make_unique<AddValuesToCacheOp>(std::move(custom_op)),
      "cache_update");
}

void TransformerBuilder::ExtractLocalCache(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& k,
    const GpuModelBuilder::TensorHandle& v,
    const GpuModelBuilder::TensorHandle& params_tensor,
    const GpuModelBuilder::TensorHandle& k_local,
    const GpuModelBuilder::TensorHandle& v_local) {
  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int dst_cache_id = ucl::GetGlobalId<0>();
  int kv_id = ucl::GetGlobalId<1>();
  int head_dim_id = ucl::GetGlobalId<2>();
  if (dst_cache_id >= args.local_cache_size || kv_id >= args.kv_heads_count || head_dim_id >= args.head_size / 4) {
    return;
  }

  int ring_offset = args.params.Read($0);
  int src_cache_id = (dst_cache_id + ring_offset) % args.ringbuffer_size;

  {
    // K-cache - kOSpatialIOGroupO4I4 layout, O - local_cache_size, I - head_size
    // OGroup = o_slices -> actual layout SpatialIOGroupO4I4
    // SpatialIOGroupO4I4 Spatial ISlice OSlice O4 I4
    // index of vec4(I4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + O4Index
    int dst_k_o = dst_cache_id;
    int dst_k_o_slice = dst_k_o / 4;
    int dst_k_o4_index = dst_k_o % 4;

    int src_k_o = src_cache_id;
    int src_k_o_slice = src_k_o / 4;
    int src_k_o4_index = src_k_o % 4;

    int k_i = head_dim_id * 4;
    int k_i_slice = k_i / 4;
    int k_sp = kv_id;
    int i_slices = (args.head_size + 3) / 4;
    int src_o_slices = (args.ringbuffer_size + 3) / 4;
    int dst_o_slices = (args.local_cache_size + 3) / 4;
    int src_index = ((k_sp * i_slices + k_i_slice) * src_o_slices + src_k_o_slice) * 4 + src_k_o4_index;
    int dst_index = ((k_sp * i_slices + k_i_slice) * dst_o_slices + dst_k_o_slice) * 4 + dst_k_o4_index;
    args.src_k::type value_k = args.src_k.Read(src_index);
    args.k_local.WriteLinear(value_k, dst_index);
  }
  {
    // V-cache - kOSpatialIOGroupI4O4 layout, O - head_size, I - local_cache_size
    // OGroup = o_slices -> actual layout SpatialIOGroupI4O4
    // SpatialIOI4O4 Spatial ISlice OSlice I4 O4
    // index of vec4(O4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + I4Index
    int v_o = head_dim_id * 4;
    int v_o_slice = v_o / 4;

    int dst_v_i = dst_cache_id;
    int dst_v_i_slice = dst_v_i / 4;
    int dst_v_i4_index = dst_v_i % 4;

    int src_v_i = src_cache_id;
    int src_v_i_slice = src_v_i / 4;
    int src_v_i4_index = src_v_i % 4;

    int v_sp = kv_id;
    int o_slices = (args.head_size + 3) / 4;
    int src_i_slices = (args.ringbuffer_size + 3) / 4;
    int dst_i_slices = (args.local_cache_size + 3) / 4;
    int src_index = ((v_sp * src_i_slices + src_v_i_slice) * o_slices + v_o_slice) * 4 + src_v_i4_index;
    int dst_index = ((v_sp * dst_i_slices + dst_v_i_slice) * o_slices + v_o_slice) * 4 + dst_v_i4_index;
    args.src_v::type value_v = args.src_v.Read(src_index);
    args.v_local.WriteLinear(value_v, dst_index);
  }
})",
                                      LlmRuntimeParams::kRingOffsetIndex);

  class ExtractLocalCacheOp : public GPUOperation {
   public:
    ExtractLocalCacheOp() = default;
    int3 GetGridSize() const override {
      return int3(local_cache_size_, kv_heads_count_,
                  DivideRoundUp(head_size_, 4));
    }

    // Move only
    ExtractLocalCacheOp(ExtractLocalCacheOp&& operation) = default;
    ExtractLocalCacheOp& operator=(ExtractLocalCacheOp&& operation) = default;
    ExtractLocalCacheOp(const ExtractLocalCacheOp&) = delete;
    ExtractLocalCacheOp& operator=(const ExtractLocalCacheOp&) = delete;

    int local_cache_size_ = 1;
    int kv_heads_count_ = 1;
    int head_size_ = 1;
  };

  ExtractLocalCacheOp custom_op;
  int kv_cache_batch_size = kv_cache_batch_size_;
  if (kv_cache_batch_size_ != 1 && config_.batch_size == 1) {
    kv_cache_batch_size = 1;
  }
  custom_op.kv_heads_count_ = kv_cache_batch_size * config_.number_of_kv_heads;
  custom_op.local_cache_size_ = GetLocalCacheSize();
  custom_op.head_size_ = config_.head_dimension;
  custom_op.args_.AddInt("ringbuffer_size", TransformerBuilder::GetCacheSize(
                                                config_, /*is_global=*/false));
  custom_op.args_.AddInt("local_cache_size", custom_op.local_cache_size_);
  custom_op.args_.AddInt("window_size", config_.sliding_window_size.value());
  custom_op.args_.AddInt("global_cache_size", config_.cache_size);
  custom_op.args_.AddInt("head_size", custom_op.head_size_);
  custom_op.args_.AddInt("kv_heads_count", custom_op.kv_heads_count_);
  custom_op.AddSrcTensor("src_k", k.tensor_desc);
  custom_op.AddSrcTensor("src_v", v.tensor_desc);
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);
  custom_op.AddDstTensor("k_local", k_local.tensor_desc);
  custom_op.AddDstTensor("v_local", v_local.tensor_desc);
  custom_op.code_ = code;

  custom_op.read_size_ = k_local.tensor_desc.GetMemorySizeInBytes() +
                         v_local.tensor_desc.GetMemorySizeInBytes();

  builder.AddGpuOperation(
      {k, v, params_tensor}, {k_local, v_local},
      std::make_unique<ExtractLocalCacheOp>(std::move(custom_op)),
      "extract_local_cache");
}

absl::StatusOr<TransformerBuilder::LlmWeightsDesc>
TransformerBuilder::LoadWeights(const std::string& name,
                                const OHWI& weights_shape, bool swap_dims) {
  const std::string linear_substr =
      tensor_loader_->HasTensor(name + ".bias.b") ? ".linear" : "";
  const std::string w_name = name + linear_substr + ".w";

  const DataType data_type = tensor_loader_->GetDataTypeForWeights(
      w_name, weights_shape.DimensionsProduct());

  LlmWeightsDesc result;
  result.shape = weights_shape;

  if (data_type == DataType::kInt8) {
    result.description =
        model_builder_.GetFullyConnectedInt8WeightsDesc(weights_shape);
    ASSIGN_OR_RETURN(const auto weights_i8,
                     tensor_loader_->LoadInt8Weights(w_name, result.description,
                                                     weights_shape, swap_dims));
    result.weights = AddExternalTensor(weights_i8.weights);
  } else if (data_type == DataType::kInt4) {
    result.description =
        model_builder_.GetFullyConnectedInt4WeightsDesc(weights_shape);
    ASSIGN_OR_RETURN(const auto weights_i8_tensor,
                     tensor_loader_->LoadInt4Weights(w_name, result.description,
                                                     weights_shape, swap_dims,
                                                     std::nullopt));
    result.weights = AddExternalTensor(weights_i8_tensor.weights);
  } else if (data_type == DataType::kInt2) {
    result.description =
        model_builder_.GetFullyConnectedInt2WeightsDesc(weights_shape);
    ASSIGN_OR_RETURN(const auto weights_i8_tensor,
                     tensor_loader_->LoadInt2Weights(w_name, result.description,
                                                     weights_shape, swap_dims));
    result.weights = AddExternalTensor(weights_i8_tensor.weights);
  } else {
    result.description =
        model_builder_.GetFullyConnectedWeightsDesc(data_type_, weights_shape);
    ASSIGN_OR_RETURN(auto* weights_tensor,
                     tensor_loader_->LoadWeights(w_name, result.description,
                                                 weights_shape, swap_dims));
    result.weights = AddExternalTensor(weights_tensor);
  }

  if (data_type == DataType::kInt8 || data_type == DataType::kInt4 ||
      data_type == DataType::kInt2) {
    const int quantization_i_group_size =
        tensor_loader_->GetQuantizationIGroupSize(w_name, weights_shape);
    const int scale_i_ch = weights_shape.i / quantization_i_group_size;
    result.scale_zp_shape = OHWI(weights_shape.o, 1, 1, scale_i_ch);

    const std::string scale_name = name + linear_substr + ".w_quantized_scale";
    ASSIGN_OR_RETURN(auto* scale_tensor,
                     tensor_loader_->LoadScale(
                         scale_name, result.scale_zp_shape, std::nullopt));
    const std::string zero_point_name =
        name + linear_substr + ".w_quantized_zp";
    ASSIGN_OR_RETURN(auto* zero_point_tensor,
                     tensor_loader_->LoadZeroPoint(zero_point_name, scale_name,
                                                   result.scale_zp_shape));

    result.scale = AddExternalTensor(scale_tensor);
    result.zero_point = AddExternalTensor(zero_point_tensor);
  }

  return result;
}

// current_tokens is BHWC(batch_size, 1, seq_size, 1)
// tokens_ids_handle is BHWC(1, 1, 1, kv_cache_batch_size * cache_size), plain
// buffer.
// kv_cache_batch_size may be not equal to batch_size, in this case batch_size
// must be 1
void TransformerBuilder::WriteCurrentTokens(
    const GpuModelBuilder::TensorHandle& current_tokens,
    const GpuModelBuilder::TensorHandle& tokens_ids_handle,
    const GpuModelBuilder::TensorHandle& params_tensor) {
  class WriteCurrentTokensOp : public GPUOperation {
   public:
    explicit WriteCurrentTokensOp(int batch_size) : batch_size_(batch_size) {}
    int3 GetGridSize() const override {
      return int3(src_[0]->Width(), batch_size_, 1);
    }

    // Move only
    WriteCurrentTokensOp(WriteCurrentTokensOp&& operation) = default;
    WriteCurrentTokensOp& operator=(WriteCurrentTokensOp&& operation) = default;
    WriteCurrentTokensOp(const WriteCurrentTokensOp&) = delete;
    WriteCurrentTokensOp& operator=(const WriteCurrentTokensOp&) = delete;

    int batch_size_;
  };

  WriteCurrentTokensOp custom_op(kv_cache_batch_size_);
  custom_op.args_.AddInt("tokens_batch_size", kv_cache_batch_size_);
  custom_op.args_.AddInt("cache_size", config_.cache_size);
  custom_op.AddSrcTensor("src", current_tokens.tensor_desc);

  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);

  BufferDescriptor tokens_buffer;
  tokens_buffer.element_type = DataType::kInt32;
  tokens_buffer.element_size = 1;
  custom_op.AddDstBuffer("dst", tokens_buffer);

  custom_op.code_ = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int src_token_index = ucl::GetGlobalId<0>();
  int batch_index = ucl::GetGlobalId<1>();
  int src_batch = batch_index % args.src.Batch();
  args.src.SetBatchRef(src_batch);
  if (src_token_index >= args.src.Width() || batch_index >= args.tokens_batch_size) {
    return;
  }

  int token = args.src.Read(src_token_index, 0, 0).x;
  int dst_token_index = src_token_index + args.params.Read($0);
  if (dst_token_index < args.cache_size) {
    args.dst.Write(token, dst_token_index * args.tokens_batch_size + batch_index);
  }
})",
                                     LlmRuntimeParams::kTokenOffsetIndex);

  model_builder_.AddGpuOperation(
      {current_tokens, params_tensor}, {tokens_ids_handle},
      std::make_unique<WriteCurrentTokensOp>(std::move(custom_op)),
      "write_current_tokens");
}

// previous_tokens is BHWC(batch_size, 1, seq_size, 1)
// tokens_ids_handle is BHWC(1, 1, 1, kv_cache_batch_size * cache_size), plain
// buffer.
// kv_cache_batch_size may be not equal to batch_size, in this case batch_size
// must be 1
void TransformerBuilder::ReadPreviousTokens(
    const GpuModelBuilder::TensorHandle& previous_tokens,
    const GpuModelBuilder::TensorHandle& tokens_ids_handle,
    const GpuModelBuilder::TensorHandle& params_tensor) {
  GPUOperation custom_op;
  custom_op.args_.AddInt("tokens_batch_size", kv_cache_batch_size_);
  custom_op.args_.AddInt("cache_size", config_.cache_size);
  BufferDescriptor src_buffer;
  src_buffer.element_type = DataType::kInt32;
  src_buffer.element_size = 1;
  custom_op.AddSrcBuffer("src", src_buffer);

  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);

  custom_op.AddDstTensor("dst", previous_tokens.tensor_desc);
  custom_op.code_ = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int wb = ucl::GetGlobalId<0>();
  int dst_token_index = wb / args.dst.Batch();
  int batch_index = wb % args.dst.Batch();
  args.dst.SetBatchRef(batch_index);
  if (dst_token_index >= args.dst.Width()) {
    return;
  }

  int prev_token_index = dst_token_index + args.params.Read($0) - 1;
  int token = 0;
  if (prev_token_index >= 0 && prev_token_index < args.cache_size) {
    token = args.src.Read(prev_token_index * args.tokens_batch_size + batch_index);
  }
  int4 result;
  result.x = token;
  result.y = 0;
  result.z = 0;
  result.w = 0;
  args.dst.Write(result, dst_token_index, 0, 0);
})",
                                     LlmRuntimeParams::kTokenOffsetIndex);
  custom_op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  model_builder_.AddGpuOperation(
      {tokens_ids_handle, params_tensor}, {previous_tokens},
      std::make_unique<GPUOperation>(std::move(custom_op)),
      "read_previous_tokens");
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeEmbLookUpOp(
    const GpuModelBuilder::TensorHandle& src,
    const TransformerBuilder::LlmWeightsDesc& embedding) {
  const GpuModelBuilder::TensorHandle* scale =
      embedding.IsQuantized() ? &embedding.scale : nullptr;
  const GpuModelBuilder::TensorHandle* zero_point =
      embedding.IsQuantized() ? &embedding.zero_point : nullptr;
  // We don't support zero-point in our EmbLookUpOp with blockwise int4 yet,
  // since default-empty zero-point tensors throw OOB errors when being read.
  // ("Default-empty tensors" refers to when we don't have any real tensor data
  // to load from the model, but we make *something* as a placeholder anyway.)
  if (zero_point && embedding.description.type == DataType::kUint4 &&
      embedding.scale_zp_shape.i != 1) {
    zero_point = nullptr;
  }
  const GpuModelBuilder::Weights external_weights = CreateExternalWeights(
      embedding.weights, embedding.description, embedding.shape,
      embedding.scale_zp_shape, scale, zero_point);
  return model_builder_.EmbeddingLookup(src, external_weights, data_type_,
                                        Axis::kWidth);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeNGrammerLayer(
    const GpuModelBuilder::TensorHandle& embeddings,
    const std::vector<GpuModelBuilder::TensorHandle>& ngrammer_embs,
    const std::string& name) {
  if (ngrammer_embs.size() != config_.number_of_heads) {
    return absl::InvalidArgumentError(
        "ngrammer_embs size must be equal to number_of_heads");
  }
  auto ngrammer_embs_copy = ngrammer_embs;
  for (int i = 0; i < config_.number_of_heads; ++i) {
    ngrammer_embs_copy[i] = MakeRMSNormalization(
        ngrammer_embs[i], name + ".ngram_layer_norm_" + std::to_string(i));
    ngrammer_embs_copy[i] =
        model_builder_.Transpose(ngrammer_embs_copy[i], BHWC(0, 2, 1, 3));
  }
  auto ngram_emb = model_builder_.Concat(ngrammer_embs_copy, Axis::kWidth);
  const float scale =
      1.0f / std::sqrt(static_cast<float>(config_.number_of_heads));
  ngram_emb = model_builder_.Multiplication(ngram_emb, scale);
  const auto& emb_shape = embeddings.tensor_desc.GetBHWCShape();
  auto t = model_builder_.Reshape(
      embeddings, BHWC(emb_shape.b, emb_shape.w, config_.number_of_heads,
                       config_.model_dimension / config_.number_of_heads));
  t = MakeRMSNormalization(t, name + ".emb_layer_norm_0");
  t = model_builder_.Add(t, ngram_emb);
  t = model_builder_.Multiplication(t, 1.0f / std::sqrt(2.0f));
  return model_builder_.Reshape(t, emb_shape);
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::GetNGrammerEmbeddings(
    const GpuModelBuilder::TensorHandle& current_tokens,
    const GpuModelBuilder::TensorHandle& previous_tokens,
    absl::Span<const int> primes) {
  // Unigram token IDs and vocab_size can exceed 1e5 risking int32 overflow
  // when multiplied.
  GpuModelBuilder::TensorHandle bigrammer_id = MakeAddWithOverflow2Input(
      MakeMulWithOverflow(previous_tokens, config_.vocabulary_size),
      current_tokens);
  std::vector<GpuModelBuilder::TensorHandle> ngrammer_embs(
      config_.number_of_heads);
  for (int i = 0; i < config_.number_of_heads; ++i) {
    auto ngrammer_id =
        MakeAddWithOverflow(MakeMulWithOverflow(bigrammer_id, i + 1), i + 1);
    // mathematically(as well as in python) mod will return always positive
    // number when 'b' is positive. This is not true in c++, for c++ used
    // equation (a/b) * b + a%b = a to find result of mod (a%b = a - (a/b) * b).
    // For negative numbers we will have in c++ negative result of mod. To get
    // positive result we need(for example) to add 'a' to the resut of mod and
    // again apply mod.

    // TODO: b/355494120 - Switch this back to MOD once the WebGPU behavior
    // difference is figured out. WebGPU has strange behavior here when taking
    // the MOD of a negative number, and treats the value as unsigned instead of
    // signed. Instead of directly doing a MOD, we have to manually do an
    // integer division and subtract out.
    auto sub = model_builder_.Elementwise(
        model_builder_.Elementwise(ngrammer_id, OperationType::kDiv, primes[i]),
        OperationType::kMul, primes[i]);
    ngrammer_id =
        model_builder_.Elementwise(ngrammer_id, sub, OperationType::kSub);

    // adding 'b' to the result of mod to exclude negative numbers
    ngrammer_id = model_builder_.Add(ngrammer_id, primes[i]);
    // applying mod again to exclude numbers greater than 'b'
    ngrammer_id =
        model_builder_.Elementwise(ngrammer_id, OperationType::kMod, primes[i]);
    ngrammer_id = model_builder_.Elementwise(ngrammer_id, OperationType::kMod,
                                             *config_.ngram_vocabulary_size);
    const std::string tensor_name = GetNgrammerEmbeddingName(i);
    ASSIGN_OR_RETURN(
        auto ngram_per_head_embedding,
        LoadWeights(tensor_name,
                    OHWI(*config_.ngram_vocabulary_size, 1, 1,
                         config_.model_dimension / config_.number_of_heads),
                    /*swap_dims=*/false));
    ngrammer_embs[i] = MakeEmbLookUpOp(ngrammer_id, ngram_per_head_embedding);
  }
  return ngrammer_embs;
}

// tokens is a tensor of shape [B, 1, W, 1], where B is config_.batch_size, W is
// the config_.sequence_size.
// output is Buffer based tensor of shape [1, 1, 1, C], where C is
// config_.cache_size * config_.batch_size.
void TransformerBuilder::CopyTokensToOutputBuffer(
    const GpuModelBuilder::TensorHandle& tokens,
    const GpuModelBuilder::TensorHandle& params_i32_handle,
    const GpuModelBuilder::TensorHandle& output) {
  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int B = ucl::GetGlobalId<0>();
  int W = ucl::GetGlobalId<1>();
  args.tokens.SetBatchRef(B);

  if (W >= args.tokens.Width() || B >= args.tokens.Batch() || ucl::GetGlobalId<2>() != 0) return;

  int4 token = args.tokens.Read(W, 0, 0);
  int time_step = args.params_i32.Read($0) + W + 1;
  if (time_step < args.cache_size) {
    args.output.Write(token.x, time_step * args.tokens.Batch() + B);
  }
})",
                                      LlmRuntimeParams::kTokenOffsetIndex);

  class CopyTokensToOutputBufferOp : public GPUOperation {
   public:
    int3 GetGridSize() const override {
      return int3(src_[0]->Batch(), src_[0]->Width(), 1);
    }
  };
  CopyTokensToOutputBufferOp custom_op;
  custom_op.AddSrcTensor("tokens", tokens.tensor_desc);
  BufferDescriptor params_i32_buffer;
  params_i32_buffer.element_type = DataType::kInt32;
  params_i32_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params_i32", params_i32_buffer);
  custom_op.args_.AddInt("cache_size", config_.cache_size);
  {
    BufferDescriptor output_buffer;
    output_buffer.element_type = DataType::kInt32;
    output_buffer.element_size = 1;
    custom_op.AddDstBuffer("output", output_buffer);
  }
  custom_op.code_ = std::move(code);

  model_builder_.AddGpuOperation(
      {tokens, params_i32_handle}, {output},
      std::make_unique<CopyTokensToOutputBufferOp>(std::move(custom_op)),
      "copy_tokens_to_output_buffer");
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::PerDimScale(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const std::string& name) {
  const int head_size = src.tensor_desc.GetBHWCShape().c;
  switch (config_.attention_scale_type) {
    case odml::infra::proto::TransformerParameters::
        SCALE_TYPE_INV_SQRT_HEAD_DIM: {
      const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
      return builder.Multiplication(src, scale);
    }
    case odml::infra::proto::TransformerParameters::
        SCALE_TYPE_INV_SQRT_D_MODEL_DIV_NUM_HEADS: {
      const float scale =
          1.0f / std::sqrt(config_.model_dimension / config_.number_of_heads);
      return builder.Multiplication(src, scale);
    }
    case odml::infra::proto::TransformerParameters::SCALE_TYPE_PER_DIM_SCALE: {
      GpuModelBuilder::TensorHandle scale;
      TensorFloat32 scale_tensor;
      const BHWC shape(1, 1, 1, head_size);
      scale_tensor.shape = shape;
      scale_tensor.data = tensor_loader_->LoadFloat32(
          name + ".per_dim_scale.per_dim_scale", shape.c);
      for (int i = 0; i < scale_tensor.data.size(); ++i) {
        scale_tensor.data[i] = SoftPlus(scale_tensor.data[i]) * 1.442695041 /
                               sqrt(static_cast<float>(head_size));
      }
      scale = builder.AddConstantTensor(scale_tensor,
                                        src.tensor_desc.GetDataType());
      return builder.Multiplication(src, scale);
    }
    case odml::infra::proto::TransformerParameters::
        SCALE_TYPE_RESCALE_FACTOR_INV_HEAD_DIM: {
      const float scale = config_.query_rescale_factor / config_.head_dimension;
      return builder.Multiplication(src, scale);
    }
    case odml::infra::proto::TransformerParameters::SCALE_TYPE_NONE: {
      return src;
    }
    default:
      return absl::InvalidArgumentError("Unknown attention scale type.");
  }
}

GpuModelBuilder::TensorHandle TransformerBuilder::GatedResidualConnection(
    const GpuModelBuilder::TensorHandle& x,
    const GpuModelBuilder::TensorHandle& y, const std::string& name) {
  TensorFloat32 scale_tensor;
  const BHWC shape(1, 1, 1, config_.model_dimension);
  scale_tensor.shape = shape;
  scale_tensor.data = tensor_loader_->LoadFloat32(name + ".scale", shape.c);
  TensorFloat32 gating_fun;
  gating_fun.shape = shape;
  gating_fun.data.resize(gating_fun.shape.DimensionsProduct());
  TensorFloat32 sigmoid;
  sigmoid.shape = shape;
  sigmoid.data.resize(sigmoid.shape.DimensionsProduct());
  for (int i = 0; i < gating_fun.data.size(); ++i) {
    gating_fun.data[i] = GatingFun(scale_tensor.data[i]);
    sigmoid.data[i] = Sigmoid(scale_tensor.data[i]);
  }
  auto prefactor =
      model_builder_.AddConstantTensor(gating_fun, x.tensor_desc.GetDataType());
  auto postfactor =
      model_builder_.AddConstantTensor(sigmoid, y.tensor_desc.GetDataType());
  auto prod0 = model_builder_.Multiplication(x, prefactor);
  auto prod1 = model_builder_.Multiplication(y, postfactor);
  return model_builder_.Add(prod0, prod1);
}

// S - sequence_size
// HS - head_size
// CS - cache_size
// HGS - head_group_size (MHA: HGS = 1, MQA: HGS = HC)
// KVHC = KHC = VHC -> HC = KVHC * HGS
// q - hwc(B*KVHC, HGS*S, HS)
// k - hwc(B*KVHC, CS, HS)
// v - hwc(B*KVHC, CS, HS)
//     q_tensor       k_tensor
//  [B, HGS*S, HS]   [B, CS, HS]
//             \    /
//             matmul (with inplace transpose for k)
//         [B, HGS*S, CS]
//               |
//       qk_post_proc_graph(optional)
//               |
//            softmax       v_tensor
//         [B, HGS*S, CS]   [B, CS, HS]
//                 \       /
//                   matmul
//                [B, HGS*S, HS]
//                      |
//                    output
absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeMatMulSoftmaxMatMul(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& q,
    const GpuModelBuilder::TensorHandle& k,
    const GpuModelBuilder::TensorHandle& v,
    const GpuModelBuilder::TensorHandle& params_tensor,
    const std::function<GpuModelBuilder::TensorHandle(
        GpuModelBuilder::TensorHandle)>& qk_post_proc_graph,
    bool is_global) {
  return MakeMatMulSoftmaxMatMul(builder, q, k, nullptr, v, nullptr,
                                 params_tensor, qk_post_proc_graph, is_global);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeMatMulSoftmaxMatMul(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& q,
    const GpuModelBuilder::TensorHandle& k,
    const GpuModelBuilder::TensorHandle* k_scaled_weights,
    const GpuModelBuilder::TensorHandle& v,
    const GpuModelBuilder::TensorHandle* v_scaled_weights,
    const GpuModelBuilder::TensorHandle& params_tensor,
    const std::function<GpuModelBuilder::TensorHandle(
        GpuModelBuilder::TensorHandle)>& qk_post_proc_graph,
    bool is_global) {
  GpuModelBuilder::TensorHandle att;
  auto k_cache = k;
  auto v_cache = v;
  const bool use_local_cache = UseSmallLocalCache(config_) && !is_global;
  int cache_size = use_local_cache ? GetLocalCacheSize() : config_.cache_size;
  int kv_cache_batch_size = kv_cache_batch_size_;
  if (kv_cache_batch_size_ != 1 && config_.batch_size == 1) {
    kv_cache_batch_size = 1;
  }
  const bool use_local_cache_without_copy =
      use_local_cache && SizeInBitsOf(GetKVCacheType()) >= 16;
  const bool use_end_checks = true;
  if (use_local_cache_without_copy) {
    cache_size = GetCacheSize(config_, is_global);
  } else if (use_local_cache) {
    // extract local K/V caches
    auto k_local_desc = GetKCacheTensorDescriptor(config_, kv_cache_batch_size,
                                                  cache_size, GetKVCacheType());
    k_cache = builder.AddTensor(k_local_desc);
    auto v_local_desc = GetVCacheTensorDescriptor(config_, kv_cache_batch_size,
                                                  cache_size, GetKVCacheType());
    v_cache = builder.AddTensor(v_local_desc);
    ExtractLocalCache(builder, k, v, params_tensor, k_cache, v_cache);
  }
  {  // Q*K
    WeightsDescription k_weights_desc = GetKWeightsDescription(
        config_, kv_cache_batch_size, cache_size, GetKVCacheType(), is_global);
    OHWI weights_shape =
        GetKWeightsShape(config_, kv_cache_batch_size, cache_size, is_global);
    ConvRuntimeCheckDesc runtime_check;
    if (use_end_checks) {
      runtime_check.dst_end_ch_index =
          LlmRuntimeParams::kActiveTokensAlignedIndex;
    }
    if (use_local_cache_without_copy) {
      runtime_check.ring_o_offset_index = LlmRuntimeParams::kRingOffsetIndex;
      runtime_check.ring_size = cache_size;
      weights_shape.o = GetLocalCacheSize();
    }

    const OHWI scales_zp_shape =
        k_scaled_weights
            ? OHWI(k_scaled_weights->tensor_desc.GetBHWCShape().c, 1, 1, 1)
            : OHWI(1, 1, 1, 1);
    const GpuModelBuilder::Weights external_weights =
        CreateExternalWeights(k_cache, k_weights_desc, weights_shape,
                              scales_zp_shape, k_scaled_weights);

    if (GetKVCacheType() == DataType::kUint8) {
      att = builder.FullyConnectedInt8ExternalWeights(
          q, external_weights, /*bias=*/nullptr,
          /*src_exp=*/nullptr, runtime_check, &params_tensor);
    } else {
      ASSIGN_OR_RETURN(att,
                       builder.FullyConnectedExternalWeights(
                           q, external_weights,
                           /*bias=*/nullptr,
                           /*src_exp=*/nullptr, runtime_check, &params_tensor));
    }
  }

  att = qk_post_proc_graph(att);
  {  // Att*V
    SoftmaxRuntimeCheckDesc softmax_runtime_params;
    if (use_end_checks) {
      softmax_runtime_params.end_ch_index =
          LlmRuntimeParams::kActiveTokensAlignedIndex;
    }
    GpuModelBuilder::TensorHandle att_exp;
    GpuModelBuilder::TensorHandle* att_exp_ptr = nullptr;
    if (builder.UseFusedSoftmaxWithBatchedMatMul(att)) {
      att_exp_ptr = &att_exp;
      att_exp =
          builder.SoftmaxReduce(att, softmax_runtime_params, &params_tensor);
    } else {
      att = builder.Softmax(att, softmax_runtime_params, &params_tensor);
    }
    const WeightsDescription v_weights_desc = GetVWeightsDescription(
        config_, kv_cache_batch_size, cache_size, GetKVCacheType(), is_global);
    OHWI weights_shape =
        GetVWeightsShape(config_, kv_cache_batch_size, cache_size, is_global);
    ConvRuntimeCheckDesc runtime_check;
    if (use_end_checks) {
      runtime_check.src_end_ch_index =
          LlmRuntimeParams::kActiveTokensAlignedIndex;
    }
    if (use_local_cache_without_copy) {
      runtime_check.ring_i_offset_index = LlmRuntimeParams::kRingOffsetIndex;
      runtime_check.ring_size = cache_size;
      weights_shape.i = GetLocalCacheSize();
    }

    const OHWI scales_zp_shape =
        v_scaled_weights
            ? OHWI(v_scaled_weights->tensor_desc.GetBHWCShape().c, 1, 1, 1)
            : OHWI(1, 1, 1, 1);
    const GpuModelBuilder::Weights external_weights =
        CreateExternalWeights(v_cache, v_weights_desc, weights_shape,
                              scales_zp_shape, v_scaled_weights);

    if (GetKVCacheType() == DataType::kUint8) {
      att = builder.FullyConnectedInt8ExternalWeights(
          att, external_weights,
          /*bias=*/nullptr,
          /*src_exp=*/att_exp_ptr, runtime_check, &params_tensor);
    } else {
      ASSIGN_OR_RETURN(att, builder.FullyConnectedExternalWeights(
                                att, external_weights,
                                /*bias=*/nullptr, att_exp_ptr, runtime_check,
                                &params_tensor));
    }
  }
  return att;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeResidualAdapter(
    const GpuModelBuilder::TensorHandle& feature, const std::string& name) {
  // Layer norm
  ml_drift::Tensor<Linear, DataType::kFloat32> gamma;
  gamma.shape = Linear(feature.tensor_desc.GetBHWCShape().c);
  gamma.data = tensor_loader_->LoadFloat32(
      name + ".layer_norm.scale", feature.tensor_desc.GetBHWCShape().c);
  auto beta = MakeZeroTensor<Linear, DataType::kFloat32>(
      Linear(feature.tensor_desc.GetBHWCShape().c));
  auto t = model_builder_.LayerNormalization(feature, gamma, beta, 1e-6f);

  // Down project gate
  ASSIGN_OR_RETURN(auto x1,
                   MakeLinearAutoPrecision(
                       t, name + ".down_project_gate",
                       config_.residual_adapter->bottleneck_dimension()));
  x1 = model_builder_.MakeGelu(x1);
  // ffn_layer1
  ASSIGN_OR_RETURN(auto x2,
                   MakeLinearAutoPrecision(
                       t, name + ".down_project",
                       config_.residual_adapter->bottleneck_dimension()));
  auto x = model_builder_.Multiplication(x1, x2);

  // ffn_layer2
  ASSIGN_OR_RETURN(
      x, MakeLinearAutoPrecision(x, name + ".up_project",
                                 feature.tensor_desc.GetBHWCShape().c));
  return model_builder_.Add(feature, x);
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::GatherMedia(
    const GpuModelBuilder::TensorHandle& media,
    const GpuModelBuilder::TensorHandle& media_index) {
  // Pad media_handle by 1 on left to account for media indices starting at 1
  // instead of 0.
  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, 1, 0);
  pad_attr.appended = BHWC(0, 0, 0, 0);
  BHWC media_sh = media.tensor_desc.GetBHWCShape();
  media_sh.w += 1;
  GpuModelBuilder::TensorHandle media_padded =
      model_builder_.AddTensor(media_sh, media.tensor_desc.GetDataType());
  OperationDef pad_def;
  pad_def.src_tensors.push_back(media.tensor_desc);
  pad_def.dst_tensors.push_back(media_padded.tensor_desc);
  auto pad_op = std::make_unique<GPUOperation>(
      CreatePadding(gpu_info_, pad_def, pad_attr, media_sh.c));
  model_builder_.AddGpuOperation({media}, media_padded, std::move(pad_op),
                                 "pad");

  // Gather media indices.
  return model_builder_.Gather(media_padded, media_index, Axis::kWidth);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::InterleaveTextAndMedia(
    const GpuModelBuilder::TensorHandle& text,
    const GpuModelBuilder::TensorHandle& media) {
  GpuModelBuilder::TensorHandle t = text;
  GpuModelBuilder::TensorHandle cond_media =
      model_builder_.Elementwise(text, {0}, OperationType::kLess);
  GpuModelBuilder::TensorHandle cond_text =
      model_builder_.Elementwise(text, {0}, OperationType::kGreaterEqual);
  cond_text = model_builder_.Cast(cond_text, DataType::kInt32);
  cond_media = model_builder_.Cast(cond_media, DataType::kInt32);
  t = model_builder_.Elementwise(text, cond_text, OperationType::kMul);
  ASSIGN_OR_RETURN(
      auto embeddings,
      LoadWeights("params.lm.softmax.logits_ffn",
                  OHWI(config_.vocabulary_size, 1, 1, config_.model_dimension),
                  /*swap_dims=*/true));
  t = MakeEmbLookUpOp(t, embeddings);
  t = model_builder_.Multiplication(
      t, sqrt(static_cast<float>(config_.model_dimension)));
  GpuModelBuilder::TensorHandle media_index =
      model_builder_.Elementwise(text, cond_media, OperationType::kMul);
  media_index =
      model_builder_.Elementwise(media_index, {-1}, OperationType::kMul);
  media_index = model_builder_.Reshape(
      media_index, BHWC(1, 1, 1, media_index.tensor_desc.GetBHWCShape().w));

  ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle media_gathered,
                   GatherMedia(media, media_index));
  // Interleave text & media.
  return model_builder_.SelectV2(cond_media, media_gathered, t);
}

GpuModelBuilder::TensorHandle TransformerBuilder::ExtractFirstBatch(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& cache) {
  TensorDescriptor weights_td =
      TensorDescriptor(cache.tensor_desc.GetDataType(),
                       TensorStorageType::kBuffer, Layout::kLinear);
  weights_td.SetBHWCShape(
      BHWC(1, 1, 1, cache.tensor_desc.GetBHWCShape().c / kv_cache_batch_size_));
  auto slice = builder.AddTensor(weights_td);
  builder.AddGpuOperation(
      {cache}, {slice},
      std::make_unique<CopySubLinearTensorOp>(
          CopySubLinearTensorOp(cache.tensor_desc, slice.tensor_desc)),
      "copy_sub_tensor");
  return slice;
}

// B - batch_size
// S - sequence_size
// HS - head_size
// HC - heads_count
// KVHC - kv_heads_count
// src - bhwc(B, 1, S, model_dimension)
//    |
//  src_linear_op
//    |->q - bhwc(B, 1, S, HC*HS)
//    |->k - bhwc(B, 1, S, KVHC*HS)
//    |->v - bhwc(B, 1, S, KVHC*HS)
absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::MakeGetQKV(const GpuModelBuilder::TensorHandle& src,
                               const std::string& name, bool is_global) {
  GpuModelBuilder::TensorHandle q, k, v;
  const bool is_combined_qkv = tensor_loader_->HasTensor(name + ".qkv.w");
  const int head_size = GetHeadSize(config_, is_global);
  const int num_kv_heads = GetNumKvHeads(config_, is_global);
  const int q_ch = head_size * config_.number_of_heads;
  const int k_ch = head_size * num_kv_heads;
  const int v_ch = head_size * num_kv_heads;

  // If SRQ is applied, the qkv tensor should be split into separate tensors.
  if (config_.apply_srq) {
    RET_CHECK(!is_combined_qkv);
  }

  if (is_combined_qkv) {
    ASSIGN_OR_RETURN(auto qkv, MakeLinearAutoPrecision(src, name + ".qkv",
                                                       q_ch + k_ch + v_ch));
    return model_builder_.Split(qkv, Axis::kChannels, {q_ch, k_ch, v_ch});
  } else {
    ASSIGN_OR_RETURN(q, MakeLinearAutoPrecisionWithSrq(src, name + ".q", q_ch,
                                                       config_.apply_srq));
    ASSIGN_OR_RETURN(k, MakeLinearAutoPrecisionWithSrq(src, name + ".k", k_ch,
                                                       config_.apply_srq));
    if (config_.global_k_eq_v && is_global) {
      // v will get re-assigned during QKV transform, but need dimensions to
      // match
      v = k;
    } else {
      ASSIGN_OR_RETURN(v, MakeLinearAutoPrecisionWithSrq(src, name + ".v", v_ch,
                                                         config_.apply_srq));
    }
    return std::vector<GpuModelBuilder::TensorHandle>{q, k, v};
  }
}

// Version of MakeGetQKV which leaves KV alone, for layers using shared KV cache
absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::MakeGetQ(const GpuModelBuilder::TensorHandle& src,
                             const std::string& name, bool is_global) {
  GpuModelBuilder::TensorHandle q, k, v;
  const int head_size = GetHeadSize(config_, is_global);
  const int q_ch = head_size * config_.number_of_heads;

  // Implementation assumes separate qkv
  const bool is_combined_qkv = tensor_loader_->HasTensor(name + ".qkv.w");
  RET_CHECK(!is_combined_qkv);
  ASSIGN_OR_RETURN(q, MakeLinearAutoPrecisionWithSrq(src, name + ".q", q_ch,
                                                     config_.apply_srq));
  return std::vector<GpuModelBuilder::TensorHandle>{q, k, v};
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeRotaryEmbedding(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& segment_pos, float min_timescale,
    float max_timescale, float proportion) {
  RoPEAttributes attr = {
      .min_timescale = min_timescale,
      .max_timescale = max_timescale,
      .proportion = proportion,
  };
  return builder.SplitRoPEConcat(src, segment_pos, attr);
}

// src - bhwc(batch_size, 1, sequence_size, model_dimension)
absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeSelfAttention(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name) {
  // B - batch_size
  // S - sequence_size
  // HC - heads_count
  // HS - head_size
  // HGS - head_group_size (MHA: HGS = 1, MQA: HGS = HC)
  // KHC - HC / HGS
  // VHC - HC / HGS
  // KHC = VHC!
  // KVHC = KHC = VHC -> HC = KVHC * HGS
  auto src_sh = src.tensor_desc.GetBHWCShape();
  if (src_sh.c % config_.number_of_heads != 0) {
    return absl::InvalidArgumentError(
        "Can not build the model. Probably incorrect config.");
  }
  const int heads_count = config_.number_of_heads;
  const int head_size = config_.head_dimension;
  const int batch_size = src_sh.b;
  const int sequence_size = src_sh.w;

  ASSIGN_OR_RETURN(auto qkv_arr, MakeGetQKV(src, name));
  auto q = qkv_arr[0];  // bhwc(B, 1, S, HC*HS)
  auto k = qkv_arr[1];  // bhwc(B, 1, S, KVHC*HS)
  auto v = qkv_arr[2];  // bhwc(B, 1, S, KVHC*HS)

  RETURN_IF_ERROR(QKVRoPE(model_builder_, segment_pos, &q, &k, &v, is_global,
                          name, /*scale_shift=*/1.0f));

  ASSIGN_OR_RETURN(q, PerDimScale(model_builder_, q, name));
  //  q - hwc(B*KVHC, HGS*S, HS)
  //  k - hwc(B*KVHC, S, HS)
  //  v - hwc(B*KVHC, S, HS)

  AddValuesToCache(model_builder_, k, v, params_tensor, key_tensor,
                   value_tensor, is_global);
  if (kv_cache_batch_size_ != 1 && config_.batch_size == 1) {
    // extract first batch slice, this can happen when batched kv cache used
    // with non batched model, that can happen in multi output model
    k = ExtractFirstBatch(model_builder_, key_tensor);
    v = ExtractFirstBatch(model_builder_, value_tensor);
  } else {
    k = key_tensor;
    v = value_tensor;
  }
  auto qk_post_proc_graph = [this, attention_mask](auto src) {
    GpuModelBuilder::TensorHandle val = src;
    if (config_.attn_soft_cap_value > 0.0f) {  // cap_logits
      const float cap_value = config_.attn_soft_cap_value;
      val = model_builder_.Multiplication(val, 1.0f / cap_value);
      val = model_builder_.Elementwise(val, OperationType::kTanh);
      val = model_builder_.Multiplication(val, cap_value);
    }
    return MakeSelectMaskOp(model_builder_, val, attention_mask);
  };
  //  q - hwc(B*KVHC, HGS*S, HS)
  //  k - hwc(B*KVHC, CS, HS)
  //  v - hwc(B*KVHC, CS, HS)
  //  att - hwc(B*KVHC, HGS*S, HS)
  ASSIGN_OR_RETURN(
      auto att, MakeMatMulSoftmaxMatMul(model_builder_, q, k, v, params_tensor,
                                        qk_post_proc_graph,
                                        /*is_global=*/is_global));
  // att transform to bhwc(B, 1, S, KVHC*HGS*HS)
  // att - bhwc(B, KVHC*HGS, S, HS) = bhwc(B, HC, S, HS)
  // att - bhwc(B, S, HC, HS)
  // att - bhwc(B, 1, S, HC*HS)
  att = model_builder_.Reshape(
      att, BHWC{batch_size, heads_count, sequence_size, head_size});
  att = model_builder_.Transpose(att, BHWC(0, 2, 1, 3));
  att = model_builder_.Reshape(
      att, BHWC{batch_size, 1, sequence_size, heads_count * head_size});
  return MakeConvAutoPrecision(att, name + ".post", Layout::kOIHW,
                               config_.model_dimension);
}

GpuModelBuilder::TensorHandle TransformerBuilder::ApplyFireBias(
    const GpuModelBuilder::TensorHandle& position, float log_scale_factor,
    float threshold, const BHWC& dst_shape) {
  auto dst = model_builder_.AddTensor(dst_shape, data_type_);
  model_builder_.AddGpuOperation(
      {position}, {dst},
      std::make_unique<GPUOperation>(
          CreateApplyFireBiasOp(gpu_info_, position.tensor_desc,
                                dst.tensor_desc, log_scale_factor, threshold)),
      "apply_fire_bias");
  return dst;
}

// src - bhwc(batch_size, 1, sequence_size, model_dimension)
absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeSelfAttentionV2(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name) {
  // B - batch_size
  // S - sequence_size
  // HC - heads_count
  // HS - head_size
  // HGS - head_group_size (MHA: HGS = 1, MQA: HGS = HC)
  // KHC - HC / HGS
  // VHC - HC / HGS
  // KHC = VHC!
  // KVHC = KHC = VHC -> HC = KVHC * HGS
  auto src_sh = src.tensor_desc.GetBHWCShape();
  if (src_sh.c % config_.number_of_heads != 0) {
    return absl::InvalidArgumentError(
        "Can not build the model. Probably incorrect config.");
  }

  ASSIGN_OR_RETURN(auto qkv_arr, MakeGetQKV(src, name));
  const auto& q = qkv_arr[0];  // bhwc(B, 1, S, HC*HS)
  const auto& k = qkv_arr[1];  // bhwc(B, 1, S, KVHC*HS)
  const auto& v = qkv_arr[2];  // bhwc(B, 1, S, KVHC*HS)

  GpuModelBuilder::TensorHandle mask = attention_mask;
  GpuModelBuilder::TensorHandle fire_bias;
  if (is_global) {  // global layer, apply fire bias
    const auto fire_log_scale_factor =
        tensor_loader_->LoadFloat32(name + ".fire_pe.log_scale_factor", 1);
    const auto fire_threshold = tensor_loader_->LoadFloat32(
        name + ".fire_pe.normalizer_threshold_factor", 1);
    const float kFireNormalizerThreshold = 8192.0f;
    const float threshold = kFireNormalizerThreshold * abs(fire_threshold[0]);
    fire_bias = ApplyFireBias(segment_pos, fire_log_scale_factor[0], threshold,
                              mask.tensor_desc.GetBHWCShape());
  }

  std::string id = absl::StrCat("MakeSelfAttention:", is_global);
  ASSIGN_OR_RETURN(auto att,
                   MakeSelfAttentionV2Subgraph(
                       id, src_sh, q, k, v, mask, fire_bias, segment_pos,
                       key_tensor, value_tensor, params_tensor, is_global));
  return MakeLinearAutoPrecision(att, name + ".attn_vec_einsum",
                                 config_.model_dimension);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeSelfAttentionV2Subgraph(
    const std::string& subgraph_id, const BHWC& src_sh,
    GpuModelBuilder::TensorHandle q, GpuModelBuilder::TensorHandle k,
    GpuModelBuilder::TensorHandle v, GpuModelBuilder::TensorHandle mask,
    GpuModelBuilder::TensorHandle fire_bias,
    GpuModelBuilder::TensorHandle segment_pos,
    GpuModelBuilder::TensorHandle key_tensor,
    GpuModelBuilder::TensorHandle value_tensor,
    GpuModelBuilder::TensorHandle params_tensor, bool is_global) {
  std::vector<GpuModelBuilder::TensorHandle> main_inputs{
      q, k, v, mask, segment_pos, key_tensor, value_tensor, params_tensor,
  };
  std::vector<GpuModelBuilder::TensorHandle*> subgraph_input_th = {
      &q,
      &k,
      &v,
      &mask,
      &segment_pos,
      &key_tensor,
      &value_tensor,
      &params_tensor};
  if (is_global) {
    main_inputs.push_back(fire_bias);
    subgraph_input_th.push_back(&fire_bias);
  }
  if (!model_builder_.HasSubgraph(subgraph_id)) {
    const int heads_count = config_.number_of_heads;
    const int head_size = config_.head_dimension;
    const int batch_size = src_sh.b;
    const int sequence_size = src_sh.w;

    GpuModelBuilder builder = model_builder_.CreateBuilder();
    auto subgraph_inputs = CreateSubgraphInputs(builder, subgraph_input_th);
    MakeTransformQKV(builder, &q, &k, &v);
    //  q - hwc(B*KVHC, HGS*S, HS)
    //  k - hwc(B*KVHC, S, HS)
    //  v - hwc(B*KVHC, S, HS)

    q = builder.RMSNormalization(q, /*epsilon=*/1e-6f);
    k = builder.RMSNormalization(k, /*epsilon=*/1e-6f);

    if (!is_global) {  // apply RoPE for q and k in local layers
      q = builder.SplitRoPEConcat(q, segment_pos);
      k = builder.SplitRoPEConcat(k, segment_pos);
    }

    const float kQueryRescaleScalar = 32.0f;
    const float query_scale =
        kQueryRescaleScalar / static_cast<float>(head_size);
    q = builder.Multiplication(q, query_scale);
    //  q - hwc(B*KVHC, HGS*S, HS)
    //  k - hwc(B*KVHC, S, HS)
    //  v - hwc(B*KVHC, S, HS)

    AddValuesToCache(builder, k, v, params_tensor, key_tensor, value_tensor,
                     is_global);
    if (kv_cache_batch_size_ != 1 && config_.batch_size == 1) {
      // extract first batch slice, this can happen when batched kv cache used
      // with non batched model, that can happen in multi output model
      k = ExtractFirstBatch(builder, key_tensor);
      v = ExtractFirstBatch(builder, value_tensor);
    } else {
      k = key_tensor;
      v = value_tensor;
    }

    auto qk_post_proc_graph = [&](auto src) {
      GpuModelBuilder::TensorHandle dst = MakeSelectMaskOp(builder, src, mask);
      if (is_global) {
        dst = builder.Add(dst, fire_bias);
      }
      return dst;
    };
    //  q - hwc(B*KVHC, HGS*S, HS)
    //  k - hwc(B*KVHC, CS, HS)
    //  v - hwc(B*KVHC, CS, HS)
    //  att - hwc(B*KVHC, HGS*S, HS)
    ASSIGN_OR_RETURN(auto att,
                     MakeMatMulSoftmaxMatMul(builder, q, k, v, params_tensor,
                                             qk_post_proc_graph, is_global));
    // att transform to bhwc(B, 1, S, KVHC*HGS*HS)
    // att - bhwc(B, KVHC*HGS, S, HS) = bhwc(B, HC, S, HS)
    // att - bhwc(B, S, HC, HS)
    // att - bhwc(B, 1, S, HC*HS)
    att = builder.Reshape(
        att, BHWC{batch_size, heads_count, sequence_size, head_size});
    att = builder.Transpose(att, BHWC(0, 2, 1, 3));
    att = builder.Reshape(
        att, BHWC{batch_size, 1, sequence_size, heads_count * head_size});
    RETURN_IF_ERROR(model_builder_.RegisterSubgraph(
        std::move(builder), subgraph_id, subgraph_inputs, {att}));
  }
  ASSIGN_OR_RETURN(auto dsts,
                   model_builder_.Subgraph(subgraph_id, main_inputs));
  return dsts[0];
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::MakeFFLayer(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  auto t = src;
  // pre_layer_norm
  t = MakeNormalization(t, name + ".pre_layer_norm");

  // Inside the FF layer everything can stay f16
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF16);
  t = CastIfBfloat16Fix(t, DataType::kFloat16);

  // ffn_layer1_gate
  ASSIGN_OR_RETURN(auto gate_value,
                   MakeLinearAutoPrecision(t, name + ".ffn_layer1_gate",
                                           config_.hidden_dimension));
  gate_value = model_builder_.MakeGelu(gate_value);
  // ffn_layer1
  ASSIGN_OR_RETURN(t, MakeLinearAutoPrecision(t, name + ".ffn_layer1",
                                              config_.hidden_dimension));

  // Cap infinity after each op here to avoid fp16 overflow.
  t = CapInfinity(model_builder_.Multiplication(CapInfinity(t), gate_value));

  // ffn_layer2
  ASSIGN_OR_RETURN(t, MakeLinearAutoPrecision(t, name + ".ffn_layer2",
                                              config_.model_dimension));
  t = CapInfinity(t);

  // Final residual overflows if bf16 on f16
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF32);
  t = CastIfBfloat16Fix(t, DataType::kFloat32);

  // post_layer_norm
  if (config_.post_norm_ff) {
    t = MakeNormalization(t, name + ".post_layer_norm");
  }
  t = CapInfinity(model_builder_.Add(t, src));
  return t;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::MakeFFLayerV2(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  GpuModelBuilder::TensorHandle t = src;
  // pre_layer_norm
  t = MakeNormalization(t, name + ".pre_ffw_norm");

  const std::string rotqat_x_scale_name = name + ".mlp.rotqat.x_scale";
  const std::string rotqat_h_scale_name = name + ".mlp.rotqat.hidden_scale";
  const bool use_rotqat = tensor_loader_->HasTensor(rotqat_x_scale_name) &&
                          tensor_loader_->HasTensor(rotqat_h_scale_name);

  const GpuModelBuilder::TensorHandle x_scale_lora_src = t;
  if (use_rotqat) {
    RET_CHECK(config_.model_dimension % 7 == 0);
    ASSIGN_OR_RETURN(t, ApplyRotqatRotate(t, config_.model_dimension / 7,
                                          rotqat_x_scale_name));
  }

  // ffn_layer1_gate
  ASSIGN_OR_RETURN(auto gate_value,
                   MakeLinearAutoPrecision(t, name + ".mlp.ff_gate",
                                           config_.hidden_dimension,
                                           &x_scale_lora_src, std::nullopt));
  gate_value = MakeRelu1p5(gate_value);
  // ffn_layer1
  ASSIGN_OR_RETURN(
      t, MakeLinearAutoPrecision(t, name + ".mlp.ff1", config_.hidden_dimension,
                                 &x_scale_lora_src, std::nullopt));
  // Cap infinity after each op here to avoid fp16 overflow.
  t = CapInfinity(model_builder_.Multiplication(CapInfinity(t), gate_value));

  GpuModelBuilder::TensorHandle h_scale_lora_src = t;
  if (use_rotqat) {
    RET_CHECK(config_.hidden_dimension % 14 == 0);
    ASSIGN_OR_RETURN(t, ApplyRotqatRotate(t, config_.hidden_dimension / 14,
                                          rotqat_h_scale_name));
  }

  // ffn_layer2
  ASSIGN_OR_RETURN(t, MakeLinearAutoPrecision(t, name + ".mlp.linear",
                                              config_.model_dimension,
                                              &h_scale_lora_src, std::nullopt));
  t = CapInfinity(t);

  // no scaling
  t = model_builder_.RMSNormalization(t, /*epsilon=*/1e-6f);

  return GatedResidualConnection(src, t, name + ".ffw_gating");
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeLaurelLayerV3(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& normed_src, const std::string& name) {
  // Laurel for V3, Laurel(x, normed_x) is implemented as follows:
  // where x and normed_x have shape [B, L, D]
  // 1. y = A * normed_x [B, L, R = 64]
  // 2. y = B y. [B, L, d_model=2048]
  // Normalization is done w/
  // self.post_laurel_norm = rms_norm.norm(
  //     num_groups=self.cfg.num_groups_in_norm_layer,
  //     epsilon=1e-6,
  //     use_simple_rmsnorm=self.cfg.use_simple_rmsnorm=True,
  //     use_eps_free_rmsnorm=self.cfg.use_eps_free_rmsnorm=False,
  //     with_scale_shift=self.cfg.with_scale_shift=False,
  // )
  // 3. y = norm(y)
  // use_gated_residual is False, laurel gating is simply addition.
  // 4. y = y + normed_x
  // 5. output = (y + x) / sqrt(2.0)
  GpuModelBuilder::TensorHandle t = normed_src;

  // 1. y = A * normed_x.
  ASSIGN_OR_RETURN(t, MakeLinearAutoPrecision(t, name + ".laurel.linear_left",
                                              config_.laurel_rank));
  // TODO: b/414439862 - Check if we need to cap infinity after each op.
  t = CapInfinity(t);

  // 2. y = By.
  ASSIGN_OR_RETURN(t, MakeLinearAutoPrecision(t, name + ".laurel.linear_right",
                                              config_.model_dimension));
  // TODO: b/414439862 - Check if we need to cap infinity after each op.
  t = CapInfinity(t);

  // 3. Laurel normalization with the specified scale shift.
  t = MakeRMSNormalization(t, name + ".post_laurel_norm",
                           config_.post_laurel_norm_scale_shift);

  // 4. y = y + normed_x
  t = model_builder_.Add(t, normed_src);

  // 5. output = (y + x) / sqrt(2.0)
  t = model_builder_.Add(t, src);
  t = model_builder_.Multiplication(t, 1.0f / sqrt(2.0f));

  return t;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::CreateComputeRouterModalitiesV3(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  GpuModelBuilder::TensorHandle router_norm =
      MakeRMSNormalization(src, name + ".altup.router_norm_layer", 0.0f);
  if (!config_.altup_config.has_value()) {
    return absl::InvalidArgumentError(
        "Altup config is not set. Please set altup_config in the "
        "TransformerConfig.");
  }
  ASSIGN_OR_RETURN(
      auto modality_router,
      MakeLinearAutoPrecision(router_norm, name + ".altup.modality_router",
                              config_.altup_config->altup_num_modalities));
  // The usual algorithm applies this multiplication first, but for F16 that can
  // lead to underflow issues, so we instead swap the ordering and apply it
  // right after the `MakeLinearAutoPrecision`, since this multiplication should
  // be commutative, and that avoids the underflow.
  modality_router = model_builder_.Multiplication(
      modality_router, 1.0f / config_.model_dimension);
  return model_builder_.Elementwise(modality_router, OperationType::kTanh);
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::CreateAltupPredictV3(
    const std::vector<GpuModelBuilder::TensorHandle>& input_handles,
    const std::string& name) {
  if (!config_.altup_config.has_value()) {
    return absl::InvalidArgumentError(
        "Altup config is not set. Please set altup_config in the "
        "TransformerConfig.");
  }
  const int num_altup_inputs = config_.altup_config->altup_num_inputs;
  ASSIGN_OR_RETURN(
      GpuModelBuilder::TensorHandle router_modalities,
      CreateComputeRouterModalitiesV3(
          input_handles[config_.altup_config->altup_active_idx], name));
  // The clipping of prediction coefs is handled inside the converter, so we
  // will skip clipping during inference time.
  ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle all_coefs,
                   MakeLinearAutoPrecision(
                       router_modalities, name + ".altup.prediction.coefs",
                       num_altup_inputs * num_altup_inputs));
  // Stacking the input handles along the height dimension, the new height
  // dimension will be num_altup_inputs.
  // x_stacked: [b, num_altup_inputs, seq_len, model_dim]
  GpuModelBuilder::TensorHandle x_stacked =
      model_builder_.Concat(input_handles, Axis::kHeight);
  // TODO(b/422981791): Try to remove those additional reshape/transpose ops
  // until we find a better way to do FC here. Now move the sequence length
  // dimension to the second dimension. x_stacked_transpose: [b, seq_len,
  // num_altup_inputs, model_dim]
  GpuModelBuilder::TensorHandle x_stacked_transpose =
      model_builder_.Transpose(x_stacked, BHWC(0, 2, 1, 3));
  // Make batch dim as 1, and combine batch size and sequence length into one
  // dimension according to BMM's format requirement.
  // [1, b * seq_len, num_altup_inputs, model_dim]
  // 1, B, N, K
  x_stacked_transpose = model_builder_.Reshape(
      x_stacked_transpose, BHWC(1, config_.batch_size * config_.sequence_size,
                                num_altup_inputs, config_.model_dimension));
  // all_coefs reshaped: [1, b * seq_len, num_altup_inputs, num_altup_inputs]
  // 1, B, M, N
  all_coefs = model_builder_.Reshape(
      all_coefs, BHWC(1, config_.batch_size * config_.sequence_size,
                      num_altup_inputs, num_altup_inputs));
  // out_stacked shape: [1, b * seq_len, num_altup_inputs, model_dim]
  // [1, B, M, N] x [1, B, N, K] = [1, B, M, K]
  ASSIGN_OR_RETURN(
      GpuModelBuilder::TensorHandle out_stacked,
      model_builder_.BatchedMatMul(all_coefs, x_stacked_transpose));
  // out_stacked shape: [b, seq_len, num_altup_inputs, model_dim]
  out_stacked = model_builder_.Reshape(
      out_stacked, BHWC(config_.batch_size, config_.sequence_size,
                        num_altup_inputs, config_.model_dimension));
  // out_stacked shape: [b, num_altup_inputs, seq_len, model_dim]
  out_stacked = model_builder_.Transpose(out_stacked, BHWC(0, 2, 1, 3));

  GpuModelBuilder::TensorHandle final_stacked =
      model_builder_.Add(x_stacked, out_stacked);
  std::vector<GpuModelBuilder::TensorHandle> output_handles;
  if (config_.altup_config->direct_active_x_stream) {
    // Slice via the third dimension.
    BHWC final_stacked_shape = final_stacked.tensor_desc.GetBHWCShape();
    SliceAttributes attr;
    attr.starts = BHWC(0, 0, config_.altup_config->altup_active_idx, 0);
    attr.ends =
        BHWC(final_stacked_shape.b, final_stacked_shape.h,
             config_.altup_config->altup_active_idx + 1, final_stacked_shape.c);
    attr.strides = BHWC(1, 1, 1, 1);
    GpuModelBuilder::TensorHandle activated =
        model_builder_.StridedSlice(final_stacked, attr);
    output_handles.push_back(final_stacked);
    output_handles.push_back(activated);
  } else {
    const std::vector<int> sizes(config_.altup_config->altup_num_inputs, 1);
    output_handles = model_builder_.Split(final_stacked, Axis::kHeight, sizes);
  }

  return output_handles;
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeMagnitude(
    const GpuModelBuilder::TensorHandle& src, Axis axis) {
  GpuModelBuilder::TensorHandle magnitude =
      model_builder_.Reduce(src, Reduce::Type::kMeanSquares, {axis});
  return model_builder_.Elementwise(magnitude, OperationType::kSqrt);
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::PreprocessAltupV3(const GpuModelBuilder::TensorHandle& src,
                                      const std::string& name) {
  const int num_altup_inputs = config_.altup_config->altup_num_inputs;

  const GpuModelBuilder::TensorHandle target_magnitude =
      MakeMagnitude(src, Axis::kChannels);
  std::vector<GpuModelBuilder::TensorHandle> altup_x;
  altup_x.reserve(num_altup_inputs);
  for (int i = 0; i < num_altup_inputs; ++i) {
    altup_x.push_back(src);
  }

  for (int i = 1; i < num_altup_inputs; ++i) {
    // altup_x[i] = self.altup_projection[i - 1](altup_x[i]).astype(x[0].dtype)
    // new_magnitude = altup.get_magnitude(altup_x[i])
    // altup_x[i] *= target_magnitude / jnp.maximum(new_magnitude, 1e-12)
    ASSIGN_OR_RETURN(
        const GpuModelBuilder::TensorHandle altup_projection,
        MakeLinearAutoPrecision(
            altup_x[i], name + ".altup.projection_" + std::to_string(i - 1),
            config_.model_dimension));
    const GpuModelBuilder::TensorHandle new_magnitude =
        MakeMagnitude(altup_projection, Axis::kChannels);
    // For F32 we leave as 1e-12f to match reference implementation, but for F16
    // we choose 1e-3f instead to match epsilon handling of ops like RMS norms.
    // Otherwise an overflow would result.
    const float epsilon =
        (src.tensor_desc.GetDataType() == DataType::kFloat32) ? 1e-12f : 1e-3f;
    const GpuModelBuilder::TensorHandle new_magnitude_bounded =
        model_builder_.Elementwise(new_magnitude, {.param = epsilon},
                                   OperationType::kMaximum);
    const GpuModelBuilder::TensorHandle scale = model_builder_.Elementwise(
        target_magnitude, new_magnitude_bounded, OperationType::kDiv);
    altup_x[i] = model_builder_.Multiplication(altup_projection, scale);
  }

  return altup_x;
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::CreateAltupCorrectV3(
    const std::vector<GpuModelBuilder::TensorHandle>& predictions,
    const GpuModelBuilder::TensorHandle& activated, const std::string& name) {
  if (!config_.altup_config.has_value()) {
    return absl::InvalidArgumentError(
        "Altup config is not set. Please set altup_config in the "
        "TransformerConfig.");
  }
  const int num_altup_inputs = config_.altup_config->altup_num_inputs;
  ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle router_modalities,
                   CreateComputeRouterModalitiesV3(activated, name));

  // The clipping of prediction coefs is handled inside the converter, so we
  // will skip clipping during inference time.
  ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle all_coefs,
                   MakeLinearAutoPrecision(router_modalities,
                                           name + ".altup.correction.coefs",
                                           num_altup_inputs));

  const GpuModelBuilder::TensorHandle& active_x =
      predictions[config_.altup_config->altup_active_idx];
  GpuModelBuilder::TensorHandle innovation =
      model_builder_.Elementwise(activated, active_x, OperationType::kSub);

  all_coefs = model_builder_.Reshape(
      all_coefs,
      BHWC(config_.batch_size, 1, config_.sequence_size, num_altup_inputs));
  all_coefs = model_builder_.Add(all_coefs, 1.0f);
  all_coefs = model_builder_.Transpose(all_coefs, BHWC(0, 3, 2, 1));
  auto x_altups =
      model_builder_.Tile(innovation, Axis::kHeight, num_altup_inputs);
  x_altups = model_builder_.Multiplication(x_altups, all_coefs);
  auto predictions_x1 = model_builder_.Concat(predictions, Axis::kHeight);
  predictions_x1 = model_builder_.Add(predictions_x1, x_altups);
  return model_builder_.Split(predictions_x1, Axis::kHeight,
                              innovation.tensor_desc.GetBHWCShape().h);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::CreatePostProcessEmbeddingsWithAltupV3(
    const std::vector<GpuModelBuilder::TensorHandle>& input_embeds,
    const std::string& name) {
  if (!config_.altup_config.has_value()) {
    return absl::InvalidArgumentError(
        "Altup config is not set. Please set altup_config in the "
        "TransformerConfig.");
  }
  const int num_altup_inputs = config_.altup_config->altup_num_inputs;
  GpuModelBuilder::TensorHandle target_magnitude =
      MakeMagnitude(input_embeds[0], Axis::kChannels);
  std::vector<GpuModelBuilder::TensorHandle> unembed_x;
  unembed_x.reserve(num_altup_inputs);
  unembed_x.push_back(input_embeds[0]);
  for (int i = 1; i < num_altup_inputs; ++i) {
    ASSIGN_OR_RETURN(
        GpuModelBuilder::TensorHandle projection,
        MakeLinearAutoPrecision(
            input_embeds[i],
            name + ".altup.unembed_projection_" + std::to_string(i - 1),
            config_.model_dimension));
    GpuModelBuilder::TensorHandle new_magnitude =
        MakeMagnitude(projection, Axis::kChannels);
    // For F32 we leave as 1e-12f to match reference implementation, but for F16
    // we choose 1e-3f instead to match epsilon handling of ops like RMS norms.
    // Otherwise an overflow would result.
    const float epsilon =
        (new_magnitude.tensor_desc.GetDataType() == DataType::kFloat32) ? 1e-12f
                                                                        : 1e-3f;
    GpuModelBuilder::TensorHandle max = model_builder_.Elementwise(
        new_magnitude, {.param = epsilon}, OperationType::kMaximum);
    GpuModelBuilder::TensorHandle div =
        model_builder_.Elementwise(target_magnitude, max, OperationType::kDiv);
    unembed_x.push_back(
        model_builder_.Elementwise(projection, div, OperationType::kMul));
  }
  GpuModelBuilder::TensorHandle stacked =
      model_builder_.Concat(unembed_x, Axis::kBatch);
  return model_builder_.Reduce(stacked, Reduce::Type::kMean, {Axis::kBatch});
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::ApplyRotqatRotate(const GpuModelBuilder::TensorHandle& src,
                                      int hadamard_dim,
                                      const std::string& scale_name) {
  TensorFloat32 scale_tensor;
  scale_tensor.shape = BHWC(1, 1, 1, src.tensor_desc.GetBHWCShape().c);
  scale_tensor.data.resize(scale_tensor.shape.DimensionsProduct());
  scale_tensor.data =
      tensor_loader_->LoadFloat32(scale_name, scale_tensor.shape.c);
  for (int i = 0; i < scale_tensor.data.size(); ++i) {
    scale_tensor.data[i] += 1.0f;
  }
  GpuModelBuilder::TensorHandle scale = model_builder_.AddConstantTensor(
      scale_tensor, src.tensor_desc.GetDataType());
  GpuModelBuilder::TensorHandle src_scaled =
      model_builder_.Multiplication(src, scale);

  GpuModelBuilder::TensorHandle interm =
      model_builder_.AddTensor(src_scaled.tensor_desc.GetBHWCShape(),
                               src_scaled.tensor_desc.GetDataType());
  ASSIGN_OR_RETURN(RotqatRotateOp op_part1,
                   CreateRotqatRotateOp(
                       gpu_info_, src_scaled.tensor_desc, interm.tensor_desc,
                       /*use_o7_h4=*/true, /*hadamard_dim=*/std::nullopt));
  model_builder_.AddGpuOperation(
      {src_scaled}, {interm},
      std::make_unique<RotqatRotateOp>(std::move(op_part1)),
      "rotqat_rotate_o7_h4");

  GpuModelBuilder::TensorHandle dst =
      model_builder_.AddTensor(src_scaled.tensor_desc.GetBHWCShape(),
                               src_scaled.tensor_desc.GetDataType());
  ASSIGN_OR_RETURN(
      RotqatRotateOp op_part2,
      CreateRotqatRotateOp(gpu_info_, interm.tensor_desc, dst.tensor_desc,
                           /*use_o7_h4=*/false, hadamard_dim));
  model_builder_.AddGpuOperation(
      {interm}, {dst}, std::make_unique<RotqatRotateOp>(std::move(op_part2)),
      "rotqat_rotate_hadamard");
  return dst;
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::MakeProjectedPleInputsV3(
    const GpuModelBuilder::TensorHandle& src,
    const std::string& projection_name, const std::string& norm_name) {
  // Underlying ops do unnecessary processing for the matformed model on the
  // skipped layers. We prefer fewer ops count vs processing only necessary data
  // but with bigger amount of smaller kernels.
  // SRQ used in PLE for V4 but not V3. We check to see if SRQ weights are
  // included for this weight in our current model, and if so, attempt to use
  // them throughout PLE.
  config_.apply_srq_to_ple = tensor_loader_->HasTensor(
      projection_name + ".input_activation_static_scale");

  ASSIGN_OR_RETURN(auto projected_ple_inputs,
                   MakeLinearAutoPrecisionWithSrq(
                       src, projection_name,
                       *config_.per_layer_embedding_dimension *
                           (config_.stack_size + config_.num_layers_skipped),
                       config_.apply_srq_to_ple));
  auto t = model_builder_.Reshape(
      projected_ple_inputs,
      BHWC(projected_ple_inputs.tensor_desc.GetBHWCShape().b,
           projected_ple_inputs.tensor_desc.GetBHWCShape().w,
           (config_.stack_size + config_.num_layers_skipped),
           *config_.per_layer_embedding_dimension));
  t = model_builder_.Transpose(t, BHWC(0, 2, 1, 3));
  // per_layer_projection_norm has no shift in the reference python
  t = MakeRMSNormalization(t, norm_name, /*scale_shift=*/0.0f);
  std::vector<GpuModelBuilder::TensorHandle> projected_ple_inputs_split;
  // Use StridedSlice() instead of Split() here because Split() generates a huge
  // shader for these dimensions that significantly slows down initialization.
  for (int i = 0; i < config_.stack_size; ++i) {
    int slice_i = SkipMatformIndex(i, config_);
    SliceAttributes attr;
    attr.starts = BHWC(0, slice_i, 0, 0);
    attr.ends = t.tensor_desc.GetBHWCShape();
    attr.ends.h = slice_i + 1;
    attr.strides = BHWC(1, 1, 1, 1);
    projected_ple_inputs_split.push_back(model_builder_.StridedSlice(t, attr));
  }
  return projected_ple_inputs_split;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::MakePleLayer(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& per_layer_embedding,
    const std::string& name) {
  ASSIGN_OR_RETURN(auto ple_gate, MakeLinearAutoPrecision(
                                      src, name + ".per_layer_embedding_gate",
                                      *config_.per_layer_embedding_dimension));
  ple_gate = model_builder_.MakeGelu(ple_gate);
  auto ple_output =
      model_builder_.Multiplication(ple_gate, per_layer_embedding);
  ASSIGN_OR_RETURN(ple_output,
                   MakeLinearAutoPrecision(
                       ple_output, name + ".per_layer_embedding_projection",
                       config_.model_dimension));
  return GatedResidualConnection(src, ple_output, name + ".ple_gating");
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakePleLayerV3(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& per_layer_embedding,
    const std::string& name) {
  // Src is output[0] from altup corrected.
  // SRQ used here and below for V4, but not for V3.
  ASSIGN_OR_RETURN(auto ple_gate, MakeLinearAutoPrecisionWithSrq(
                                      src, name + ".per_layer_embedding_gate",
                                      *config_.per_layer_embedding_dimension,
                                      config_.apply_srq_to_ple));
  ple_gate = model_builder_.MakeGelu(ple_gate);
  auto ple_output =
      model_builder_.Multiplication(ple_gate, per_layer_embedding);
  ASSIGN_OR_RETURN(ple_output,
                   MakeLinearAutoPrecisionWithSrq(
                       ple_output, name + ".per_layer_embedding_projection",
                       config_.model_dimension, config_.apply_srq_to_ple));
  auto ple_output_norm =
      MakeRMSNormalization(ple_output, name + ".post_per_layer_input_norm",
                           config_.rms_norm_scale_shift);
  return ple_output_norm;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::MakeLayer(
    const GpuModelBuilder::TensorHandle& inputs,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name) {
  auto inputs_normalized = MakeNormalization(inputs, name + ".pre_layer_norm");
  // self attention

  // Inside layer and self-attention block everything is fine for bf16 on f16.
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF16);
  inputs_normalized = CastIfBfloat16Fix(inputs_normalized, DataType::kFloat16);

  ASSIGN_OR_RETURN(
      auto atten_output,
      MakeSelfAttention(inputs_normalized, attention_mask, segment_pos,
                        key_tensor, value_tensor, params_tensor, is_global,
                        name + ".self_attention"));
  // Post_layer_norm
  if (config_.post_norm_attention) {
    atten_output = MakeNormalization(atten_output, name + ".post_layer_norm");
  }

  // But this residual overflows for bf16 on f16, so we cast back to f32 in that
  // case.
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF32);
  atten_output = CastIfBfloat16Fix(atten_output, DataType::kFloat32);

  atten_output = model_builder_.Add(atten_output, inputs);
  // ff_layer
  ASSIGN_OR_RETURN(auto output, MakeFFLayer(atten_output, name + ".ff_layer"));
  return output;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::MakeLayerV2(
    const GpuModelBuilder::TensorHandle& inputs,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle* per_layer_embedding,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name) {
  auto inputs_normalized =
      MakeNormalization(inputs, name + ".pre_attention_norm");
  ASSIGN_OR_RETURN(
      auto atten_output,
      MakeSelfAttentionV2(inputs_normalized, attention_mask, segment_pos,
                          key_tensor, value_tensor, params_tensor, is_global,
                          name + ".attn"));

  atten_output = model_builder_.RMSNormalization(atten_output, 1e-6f);

  atten_output =
      GatedResidualConnection(inputs, atten_output, name + ".attention_gating");
  ASSIGN_OR_RETURN(auto output, MakeFFLayerV2(atten_output, name));
  if (per_layer_embedding) {
    ASSIGN_OR_RETURN(output, MakePleLayer(output, *per_layer_embedding, name));
  }
  return output;
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeFillSegmentPos(
    const GpuModelBuilder::TensorHandle& params_i32) {
  GpuModelBuilder::TensorHandle segment_pos = model_builder_.AddTensor(
      BHWC(1, 1, config_.sequence_size, 1), DataType::kInt32);
  model_builder_.AddGpuOperation(
      {params_i32}, segment_pos,
      std::make_unique<GPUOperation>(CreateFillSegmentPosOp(
          segment_pos.tensor_desc,
          /*param_index=*/LlmRuntimeParams::kTokenOffsetIndex)),
      "fill_segment_pos");
  return segment_pos;
}

VisionBlockIndicesOp CreateVisionBlockIndicesOp(
    const TensorDescriptor& tokens_desc,
    const TensorDescriptor& block_indices_desc, int batch_size) {
  VisionBlockIndicesOp op;
  op.batch_size_ = batch_size;
  // TODO(b/523093274): This shader currently uses a single thread per batch
  // to compute block indices sequentially due to stateful increment of
  // current_block. Consider implementing a parallel prefix scan if this
  // becomes a performance bottleneck.
  // TODO: This assumes that every image ends with an end-of-image token and
  // that no audio tokens are in the same chunk (they would be treated as
  // vision too). Pass explicit vision segment ids instead.
  op.code_ = R"(
  MAIN_FUNCTION($0) {
    if (ucl::GetGlobalId<0>() > 0 || ucl::GetGlobalId<2>() > 0) return;
    int batch_id = ucl::GetGlobalId<1>();
    if (batch_id >= args.tokens.Batch()) return;
    args.tokens.SetBatchRef(batch_id);
    args.block_indices.SetBatchRef(batch_id);
    int current_block = 0;
    bool in_vision = false;
    // This shader currently uses a single thread per batch to compute block
    // indices sequentially due to stateful increment of current_block.
    //
    // Media tokens do not reach this graph as the -2 placeholder: the artisan
    // executor replaces each of them with its negated 1-based index into the
    // media embedding cache (-1, -2, -3, ...). Each image is therefore a run of
    // consecutive negative ids that ends with the end-of-image embedding. Only
    // the soft tokens before it are bidirectional (as in training), so the
    // last token of each run stays causal.
    int width = args.tokens.Width();
    int next_token_id = 0;
    if (width > 0) {
      next_token_id = args.tokens.Read(0, 0, 0).x;
    }
    for (int i = 0; i < width; ++i) {
      int token_id = next_token_id;
      next_token_id = 0;
      if (i + 1 < width) {
        next_token_id = args.tokens.Read(i + 1, 0, 0).x;
      }
      if (token_id < 0 && next_token_id < 0) {
        if (!in_vision) {
          current_block++;
          in_vision = true;
        }
        args.block_indices::type val_vision;
        val_vision.x = current_block;
        val_vision.y = 0;
        val_vision.z = 0;
        val_vision.w = 0;
        args.block_indices.Write(val_vision, i, 0, 0);
      } else {
        in_vision = false;
        args.block_indices::type val_zero;
        val_zero.x = 0;
        val_zero.y = 0;
        val_zero.z = 0;
        val_zero.w = 0;
        args.block_indices.Write(val_zero, i, 0, 0);
      }
    }
  }
)";
  op.AddSrcTensor("tokens", tokens_desc);
  op.AddDstTensor("block_indices", block_indices_desc);
  return op;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeFillAttention(
    const GpuModelBuilder::TensorHandle& params_i32,
    const GpuModelBuilder::TensorHandle* current_tokens) {
  return MakeFillAttentionMask(params_i32, current_tokens, /*is_local=*/false);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeFillLocalAttention(
    const GpuModelBuilder::TensorHandle& params_i32,
    const GpuModelBuilder::TensorHandle* current_tokens) {
  return MakeFillAttentionMask(params_i32, current_tokens, /*is_local=*/true);
}

absl::StatusOr<int> TransformerBuilder::GetMaskWidth(
    odml::infra::proto::TransformerParameters::AttentionMaskType
        attn_mask_type) {
  switch (attn_mask_type) {
    case odml::infra::proto::TransformerParameters::PREFIX:
      return 1;
    case odml::infra::proto::TransformerParameters::BIDIRECTIONAL:
      return config_.sequence_size;
    case odml::infra::proto::TransformerParameters::CAUSAL:
      return config_.sequence_size;
    default:
      return absl::InvalidArgumentError("Unsupported attention mask type.");
  }
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeFillAttentionMask(
    const GpuModelBuilder::TensorHandle& params_i32,
    const GpuModelBuilder::TensorHandle* current_tokens, bool is_local) {
  ASSIGN_OR_RETURN(const int mask_width,
                   GetMaskWidth(config_.attention_mask_type));
  const int cache_size = is_local ? GetLocalCacheSize() : config_.cache_size;
  const BHWC mask_shape = BHWC(1, 1, mask_width, cache_size);
  if (config_.attention_mask_type ==
      odml::infra::proto::TransformerParameters::BIDIRECTIONAL) {
    auto mask_zeroes = model_builder_.AddTensor(mask_shape, DataType::kBool);
    model_builder_.AddGpuOperation({}, mask_zeroes,
                                   CreateFillTensorOp(mask_zeroes.tensor_desc),
                                   "fill_zeroes");
    return mask_zeroes;
  }

  const TensorDescriptor* block_indices_desc = nullptr;
  std::optional<GpuModelBuilder::TensorHandle> block_indices;
  if (is_local && config_.vision_bidirectional_attention &&
      current_tokens != nullptr) {
    block_indices = model_builder_.AddTensor(current_tokens->tensor_desc);
    model_builder_.AddGpuOperation(
        {*current_tokens}, *block_indices,
        std::make_unique<VisionBlockIndicesOp>(CreateVisionBlockIndicesOp(
            current_tokens->tensor_desc, block_indices->tensor_desc,
            config_.batch_size)),
        "create_vision_block_indices");
    block_indices_desc = &block_indices->tensor_desc;
  }

  auto attention_mask = model_builder_.AddTensor(mask_shape, DataType::kBool);
  std::vector<GpuModelBuilder::TensorHandle> src_tensors = {params_i32};
  if (block_indices.has_value()) {
    src_tensors.push_back(*block_indices);
  }
  model_builder_.AddGpuOperation(
      src_tensors, attention_mask,
      std::make_unique<GPUOperation>(CreateApplyAttentionMaskOp(
          params_i32.tensor_desc, attention_mask.tensor_desc,
          is_local ? config_.sliding_window_size : std::nullopt,
          UseSmallLocalCache(config_) && is_local, block_indices_desc)),
      is_local ? "apply_local_attention_mask" : "apply_attention_mask");
  return attention_mask;
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeAddWithOverflow(
    const GpuModelBuilder::TensorHandle& src, int value) {
  GpuModelBuilder::TensorHandle sum = model_builder_.AddTensor(
      src.tensor_desc.GetBHWCShape(), DataType::kInt32);
  model_builder_.AddGpuOperation(
      {src}, sum,
      std::make_unique<GPUOperation>(
          CreateAddWithOverflow(src.tensor_desc, value)),
      "add_with_overflow");
  return sum;
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeAddWithOverflow2Input(
    const GpuModelBuilder::TensorHandle& src0,
    const GpuModelBuilder::TensorHandle& src1) {
  GpuModelBuilder::TensorHandle sum = model_builder_.AddTensor(
      src0.tensor_desc.GetBHWCShape(), DataType::kInt32);
  model_builder_.AddGpuOperation(
      {src0, src1}, sum,
      std::make_unique<GPUOperation>(
          CreateAddWithOverflow2Input(src0.tensor_desc, src1.tensor_desc)),
      "add_with_overflow");
  return sum;
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeMulWithOverflow(
    const GpuModelBuilder::TensorHandle& src, int value) {
  GpuModelBuilder::TensorHandle product = model_builder_.AddTensor(
      src.tensor_desc.GetBHWCShape(), DataType::kInt32);
  model_builder_.AddGpuOperation(
      {src}, product,
      std::make_unique<GPUOperation>(
          CreateMulWithOverflow(src.tensor_desc, value)),
      "mul_with_overflow");
  return product;
}

std::string TransformerBuilder::GetEmbeddingLookupName() const {
  return "params.lm.softmax.logits_ffn";
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::MakeEmbLookUp(
    const GpuModelBuilder::TensorHandle& src) {
  ASSIGN_OR_RETURN(
      auto embeddings,
      LoadWeights(GetEmbeddingLookupName(),
                  OHWI(config_.vocabulary_size, 1, 1, config_.model_dimension),
                  /*swap_dims=*/true));
  return MakeEmbLookUpOp(src, embeddings);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::GetCentroidEmbeddings(
    const GpuModelBuilder::TensorHandle& activations, const std::string& name) {
  // Name would be "eevee.embedder" in an MTP model.
  if (!config_.centroid_embedder_config.has_value()) {
    return absl::InvalidArgumentError(
        "Centroid embeddings require a centroid_embedder_config");
  }
  const int seq_size = config_.sequence_size;
  const int num_centroids = config_.centroid_embedder_config->num_centroids;
  const int centroid_top_k = config_.centroid_embedder_config->top_k;
  const int d_model = config_.model_dimension;
  const int embeddings_per_centroid = config_.vocabulary_size / num_centroids;
  const int KE = embeddings_per_centroid * centroid_top_k;

  ASSIGN_OR_RETURN(auto ordered_embeddings,
                   LoadWeights(name + ".input_embedding_ordered",
                               OHWI(config_.vocabulary_size, 1, 1, d_model),
                               /*swap_dims=*/false));
  // We load centroids as scales and NOT weights, to directly use in BMM below.
  ml_drift::Tensor<BHWC, DataType::kFloat32> centroids_f32;
  centroids_f32.shape = BHWC(1, 1, d_model, num_centroids);
  centroids_f32.data = tensor_loader_->LoadFloat32(name + ".centroids.scale",
                                                   d_model * num_centroids);
  auto centroids = model_builder_.AddConstantTensor(
      centroids_f32, activations.tensor_desc.GetDataType());

  // S is seq_size, D is d_model, C is num_centroids
  // activations are (1, 1, S, D)
  // centroids are (1, 1, D, C)
  // scores are (1, 1, S, C)
  ASSIGN_OR_RETURN(auto scores,
                   model_builder_.BatchedMatMul(activations, centroids));

  // Get top k: (1, 1, S, K)
  auto topk_results = model_builder_.TopK(scores, centroid_top_k);
  auto top_scores = topk_results[0];
  auto top_indices = topk_results[1];

  // We adjust our indices to allow for direct lookup in ordered_embeddings.
  // We take indices from (1, 1, S, K) to (1, 1, S*K, 1), and then use a
  // broadcasted ADD with an `arange`-style offset vector in order to get our
  // final (1, 1, S*K, E) indices.
  ml_drift::Tensor<BHWC, DataType::kFloat32> arange;
  arange.shape = BHWC(1, 1, 1, embeddings_per_centroid);
  arange.data = std::vector<float>(embeddings_per_centroid);
  for (int i = 0; i < embeddings_per_centroid; ++i) {
    arange.data[i] = static_cast<float>(i);
  }
  auto final_indices =
      model_builder_.Multiplication(top_indices, embeddings_per_centroid);
  final_indices = model_builder_.Reshape(
      final_indices, BHWC(1, 1, seq_size * centroid_top_k, 1));

  // Broadcasted ADD doesn't work in MLDrift for INT32, so we cast to FLOAT32
  // and then back.
  // TODO: b/512533564 - Remove this workaround when possible.
  final_indices = model_builder_.Cast(final_indices, DataType::kFloat32);

  // Unfortunately, the ADD operation doesn't seem to have default broadcast
  // support (at least not through GpuModelBuilder), so we directly create and
  // use the desired Elementwise op here.
  // TODO: b/512533273 - Remove this workaround when possible.
  auto final_indices_out = model_builder_.AddTensor(
      BHWC(1, 1, seq_size * centroid_top_k, embeddings_per_centroid),
      DataType::kFloat32);

  OperationDef op_def;
  ElementwiseAttributes attr;
  attr.param = arange;
  op_def.src_tensors.push_back(final_indices.tensor_desc);
  op_def.dst_tensors.push_back(final_indices_out.tensor_desc);
  GPUOperation operation = CreateElementwiseWithBroadcast(
      gpu_info_, op_def, OperationType::kAdd, attr,
      final_indices.tensor_desc.GetBHWCShape(),
      final_indices_out.tensor_desc.GetBHWCShape());
  model_builder_.AddGpuOperation(
      std::vector<ValueId>{final_indices.id},
      std::vector<ValueId>{final_indices_out.id},
      std::make_unique<GPUOperation>(std::move(operation)), "broadcasted_sum");
  final_indices_out = model_builder_.Cast(final_indices_out, DataType::kInt32);

  // We then reshape (1, 1, S*K, E) as (1, 1, S*K*E, 1) to index into
  // ordered_embeddings and get (1, 1, S*K*E, D)
  final_indices_out =
      model_builder_.Reshape(final_indices_out, BHWC(1, 1, seq_size * KE, 1));
  auto embeddings = MakeEmbLookUpOp(final_indices_out, ordered_embeddings);

  // For final einsum, we want (1, S, D, KE), so we reshape+transpose
  embeddings =
      model_builder_.Reshape(embeddings, BHWC(1, seq_size, KE, d_model));
  embeddings = model_builder_.Transpose(embeddings, BHWC(0, 1, 3, 2));

  // Final einsum:
  // (1, S, 1, D) x (1, S, D, K*E) = (1, S, 1, K*E) --> (1, 1, S, K*E)
  auto activations_reshape =
      model_builder_.Reshape(activations, BHWC(1, seq_size, 1, d_model));
  ASSIGN_OR_RETURN(auto output, model_builder_.BatchedMatMul(
                                    activations_reshape, embeddings));
  return model_builder_.Reshape(output, BHWC(1, 1, seq_size, KE));
}

GpuModelBuilder::TensorHandle TransformerBuilder::DecompressPqPleLayers(
    const GpuModelBuilder::TensorHandle& compressed_table,
    const BHWC& in_shape_one, const BHWC& in_shape_two, const BHWC& out_shape) {
  auto table = model_builder_.AddTensor(
      out_shape, compressed_table.tensor_desc.GetDataType());
  OperationDef op_def;
  ElementwiseAttributes attr;
  // To decompress, we split in half along our axis of size rt_layers * 2, and
  // then use broadcasting and reshaping to perform an outer sum on the two
  // pieces: [1, N] and [N, 1] --> [N, N].
  auto compressed_table_halves =
      model_builder_.Split(compressed_table, Axis::kHeight,
                           compressed_table.tensor_desc.GetBHWCShape().h / 2);
  auto a = model_builder_.Reshape(compressed_table_halves[0], in_shape_one);
  auto b = model_builder_.Reshape(compressed_table_halves[1], in_shape_two);
  op_def.src_tensors.push_back(a.tensor_desc);
  op_def.src_tensors.push_back(b.tensor_desc);
  op_def.dst_tensors.push_back(table.tensor_desc);
  GPUOperation operation = CreateElementwiseTwoInputWithBroadcast(
      gpu_info_, op_def, OperationType::kAdd, in_shape_one, in_shape_two,
      out_shape, attr);
  model_builder_.AddGpuOperation(
      std::vector<ValueId>{a.id, b.id}, std::vector<ValueId>{table.id},
      std::make_unique<GPUOperation>(std::move(operation)), "outer_sum");
  return table;
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::GetPerLayerEmbeddings(
    const GpuModelBuilder::TensorHandle& tokens) {
  if (config_.pq_ple_with_layers) {
    if (config_.load_per_layer_embedding_dynamically) {
      return absl::InvalidArgumentError("Cannot use dynamic PLE with PQ-PLE");
    }
    const int rt_vocab = std::sqrt(config_.vocabulary_size);
    if (rt_vocab * rt_vocab != config_.vocabulary_size) {
      return absl::InvalidArgumentError("Vocab size must be square for PQ-PLE");
    }
    const int rt_layers =
        std::ceil(std::sqrt(static_cast<float>(config_.stack_size)));
    const int ple_dim = *config_.per_layer_embedding_dimension;
    int seq_size = config_.sequence_size;
    int batch_size = config_.batch_size;
    ASSIGN_OR_RETURN(auto embedding_high,
                     LoadWeights("transformer.per_layer_embeddings.high",
                                 OHWI(rt_vocab, 1, 1, ple_dim * rt_layers * 2),
                                 /*swap_dims=*/false));
    ASSIGN_OR_RETURN(auto embedding_low,
                     LoadWeights("transformer.per_layer_embeddings.low",
                                 OHWI(rt_vocab, 1, 1, ple_dim * rt_layers * 2),
                                 /*swap_dims=*/false));

    // Token shape is [1, 1, seq, 1] for prefill and [batch_size, 1, 1, 1] for
    // decode. B=1 must be kept throughout in all shapes for former case!
    // Step 1 (PQ-PLE over vocab): look up high and low "chunks".
    auto tokens_high =
        model_builder_.Elementwise(tokens, OperationType::kDiv, rt_vocab);
    auto tokens_low =
        model_builder_.Elementwise(tokens, OperationType::kMod, rt_vocab);
    auto compressed_high = model_builder_.Reshape(
        MakeEmbLookUpOp(tokens_high, embedding_high),
        BHWC(1, batch_size * seq_size, rt_layers * 2, ple_dim));
    auto compressed_low = model_builder_.Reshape(
        MakeEmbLookUpOp(tokens_low, embedding_low),
        BHWC(1, batch_size * seq_size, rt_layers * 2, ple_dim));
    // We need broadcast dims to be 1 vs. N and N vs. 1, and need B=1, so we
    // bring all other dimensions to C.
    const BHWC in_shape_one(1, rt_layers, 1, batch_size * seq_size * ple_dim);
    const BHWC in_shape_two(1, 1, rt_layers, batch_size * seq_size * ple_dim);
    const BHWC out_shape(1, rt_layers, rt_layers,
                         batch_size * seq_size * ple_dim);
    compressed_high =
        model_builder_.Transpose(compressed_high, BHWC(0, 2, 1, 3));
    compressed_low = model_builder_.Transpose(compressed_low, BHWC(0, 2, 1, 3));

    // Step 2 (PQ-PLE over layers): when PQ-PLE is applied to layers too, we
    // must decompress each piece in order to retrieve all layers for the final
    // addition.
    auto high = DecompressPqPleLayers(compressed_high, in_shape_one,
                                      in_shape_two, out_shape);
    auto low = DecompressPqPleLayers(compressed_low, in_shape_one, in_shape_two,
                                     out_shape);
    // Step 3 (reformatting): after adding, we reshape and transpose so we can
    // nicely split into the final tensor shapes we expect for a full stack of
    // embeddings.
    auto all_embeddings = model_builder_.Reshape(
        model_builder_.Add(high, low),
        BHWC(1, rt_layers * rt_layers, batch_size * seq_size, ple_dim));

    // We want to end up with a vector containing stack_size tensors of shape
    // [batch_size, 1, seq_size, ple_dim]. Below should be correct for the
    // batch_size=1 case, which is all we're supporting right now (b/505555017).
    auto embeddings_vec =
        model_builder_.Split(all_embeddings, Axis::kHeight, 1);
    embeddings_vec.resize(config_.stack_size);
    embeddings_vec.shrink_to_fit();
    return embeddings_vec;
  }

  // Normal PLE
  std::vector<GpuModelBuilder::TensorHandle> embeddings;
  embeddings.reserve(config_.stack_size);
  for (int i = 0; i < config_.stack_size - config_.extra_stack_size; ++i) {
    std::string ple_name =
        GetPerLayerEmbeddingName(SkipMatformIndex(i, config_));
    // EE models may not have PLE on extra layers.
    if (!tensor_loader_->HasTensor(ple_name + ".w")) {
      break;
    }
    ASSIGN_OR_RETURN(auto ple_weight,
                     LoadWeights(ple_name,
                                 OHWI(config_.vocabulary_size, 1, 1,
                                      *config_.per_layer_embedding_dimension),
                                 /*swap_dims=*/false));
    auto ple_embedding = MakeEmbLookUpOp(tokens, ple_weight);
    // BHWC ple_shape = tokens.tensor_desc.GetBHWCShape();
    // ple_shape.c = ple_weight.shape.i;
    // ASSIGN_OR_RETURN(auto ple_embedding,
    //                  DequantizeEmbedding(ple_shape, ple_weight.weights,
    //                                      ple_weight.scale, &tokens));
    embeddings.push_back(ple_embedding);
  }
  return embeddings;
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeNormalization(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  return MakeRMSNormalization(src, name, /*scale_shift=*/1.0f);
}

GpuModelBuilder::TensorHandle TransformerBuilder::MakeRelu1p5(
    const GpuModelBuilder::TensorHandle& src) {
  ReLUAttributes relu_attr;
  relu_attr.alpha = 0.0f;
  relu_attr.activation_min = 0.0f;
  relu_attr.activation_max = 0.0f;
  auto x = model_builder_.ReLU(src, relu_attr);
  ElementwiseAttributes pow_attr;
  pow_attr.param = 1.5f;
  return model_builder_.Elementwise(x, pow_attr, OperationType::kPow);
}

void TransformerBuilder::CreateCache(
    std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
    std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors) {
  int num_cache_layers = config_.stack_size;
  if (UseGemmaModelV3NPlus() &&
      config_.kv_cache_shared_start_layer.has_value()) {
    num_cache_layers = *config_.kv_cache_shared_start_layer;
  }

  for (int i = 0; i < num_cache_layers; ++i) {
    const bool is_global = IsGlobalLayer(config_, i);
    auto key_cache_tensor =
        model_builder_.AddTensor(GetKCacheTensorDescriptor(is_global));
    key_cached_tensors->push_back(key_cache_tensor);
    auto value_cache_tensor =
        model_builder_.AddTensor(GetVCacheTensorDescriptor(is_global));
    value_cached_tensors->push_back(value_cache_tensor);
  }
}

TensorDescriptor TransformerBuilder::GetParamsTensorDescriptor() const {
  TensorDescriptor td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kHWC};
  td.SetBHWCShape(BHWC(1, 1, 1, GetI32ParamsCount()));
  return td;
}

TensorDescriptor TransformerBuilder::GetKCacheTensorDescriptor(
    bool is_global) const {
  int cache_size = TransformerBuilder::GetCacheSize(config_, is_global);
  return GetKCacheTensorDescriptor(config_, kv_cache_batch_size_, cache_size,
                                   GetKVCacheType(), is_global);
}

TensorDescriptor TransformerBuilder::GetKCacheTensorDescriptor(
    const TransformerConfig& config, int kv_cache_batch_size, int cache_size,
    DataType data_type, bool is_global) {
  WeightsDescription weights_desc = GetKWeightsDescription(
      config, kv_cache_batch_size, cache_size, data_type, is_global);
  TensorDescriptor weights_td = TensorDescriptor(
      weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
  weights_td.SetBHWCShape(
      BHWC(1, 1, 1,
           GetTotalElementsCountForLayout(
               weights_desc, GetKWeightsShape(config, kv_cache_batch_size,
                                              cache_size, is_global))));
  return weights_td;
}

TensorDescriptor TransformerBuilder::GetVCacheTensorDescriptor(
    bool is_global) const {
  int cache_size = TransformerBuilder::GetCacheSize(config_, is_global);
  return GetVCacheTensorDescriptor(config_, kv_cache_batch_size_, cache_size,
                                   GetKVCacheType(), is_global);
}

TensorDescriptor TransformerBuilder::GetVCacheTensorDescriptor(
    const TransformerConfig& config, int kv_cache_batch_size, int cache_size,
    DataType data_type, bool is_global) {
  WeightsDescription weights_desc = GetVWeightsDescription(
      config, kv_cache_batch_size, cache_size, data_type, is_global);
  TensorDescriptor weights_td = TensorDescriptor(
      weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
  weights_td.SetBHWCShape(
      BHWC(1, 1, 1,
           GetTotalElementsCountForLayout(
               weights_desc, GetVWeightsShape(config, kv_cache_batch_size,
                                              cache_size, is_global))));
  return weights_td;
}

int TransformerBuilder::GetCacheSize(const TransformerConfig& config,
                                     bool is_global) {
  if (UseRingBuffer(config) && !is_global) {
    return std::min(config.cache_size,
                    AlignByN(*config.local_ring_cache_size, 4));
  }
  return config.cache_size;
}

int TransformerBuilder::GetLocalCacheSize() const {
  if (!UseSmallLocalCache(config_)) {
    // return global cache size
    return config_.cache_size;
  }
  const int local_cache_size =
      config_.sequence_size == 1
          ? config_.sliding_window_size.value()
          : config_.sliding_window_size.value() + config_.sequence_size;
  return std::min(GetCacheSize(config_, /*is_global=*/false),
                  AlignByN(local_cache_size, 4));
}

TensorDescriptor TransformerBuilder::GetTokensTensorDescriptor() const {
  TensorDescriptor td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kHWC};
  // tokens will be used as plain buffer, putting everything to
  // last dimension
  td.SetBHWCShape(BHWC(1, 1, 1, kv_cache_batch_size_ * config_.cache_size));
  return td;
}

TensorDescriptor TransformerBuilder::GetInputTensorDescriptor(
    int batch_size, int sequence_size) const {
  TensorDescriptor td =
      TensorDescriptor{DataType::kInt32, TensorStorageType::kBuffer,
                       batch_size == 1 ? Layout::kHWC : Layout::kBHWC};
  td.SetBHWCShape(BHWC(batch_size, 1, sequence_size, 1));
  return td;
}

TensorDescriptor TransformerBuilder::GetVisionPleTensorDescriptor() const {
  TensorDescriptor td =
      TensorDescriptor{data_type_, TensorStorageType::kTexture2D, Layout::kHWC};
  td.SetBHWCShape(
      BHWC(1, 1, media_cache_size_, *config_.per_layer_embedding_dimension));
  return td;
}

TensorDescriptor TransformerBuilder::GetMediaEmbeddingsTensorDescriptor()
    const {
  TensorDescriptor td =
      TensorDescriptor{data_type_, TensorStorageType::kTexture2D, Layout::kHWC};
  td.SetBHWCShape(BHWC(1, 1, media_cache_size_, config_.model_dimension));
  return td;
}

std::vector<std::string> TransformerBuilder::GetLayerNames() const {
  return GetLayerNames("transformer");
}

std::vector<std::string> TransformerBuilder::GetLayerNames(
    const std::string& prefix) const {
  std::vector<std::string> layer_names;
  layer_names.reserve(config_.stack_size);
  std::string layer_name_prefix =
      (UseGoogleInternalModelV2() || UseGemmaModelV3NPlus())
          ? prefix + ".layer_"
          : "params.lm.transformer.x_layers_";
  int i = 0;
  for (; i < config_.stack_size - config_.extra_stack_size; ++i) {
    layer_names.push_back(layer_name_prefix +
                          std::to_string(SkipMatformIndex(i, config_)));
  }
  layer_name_prefix = "extra." + layer_name_prefix;
  for (; i < config_.stack_size; ++i) {
    layer_names.push_back(layer_name_prefix + std::to_string(i));
  }
  return layer_names;
}

std::string TransformerBuilder::GetPerLayerEmbeddingName(int layer) const {
  return GetPerLayerEmbeddingNameInternal(layer);
}

std::string TransformerBuilder::GetNgrammerEmbeddingName(int index) const {
  return GetNgrammerEmbeddingNameInternal(index);
}

// Helpers for changing our default precision at different locations in the
// stack when applying fixes for bf16-to-f16 overflow issues.
void TransformerBuilder::ChangePrecisionIfBfloat16Fix(
    CalculationsPrecision precision) {
  if (gemma3_bfloat16_fix_) {
    data_type_ = DeduceDataTypeFromPrecision(precision);
    tensor_loader_->SetDataType(data_type_);
  }
}

GpuModelBuilder::TensorHandle TransformerBuilder::CastIfBfloat16Fix(
    const GpuModelBuilder::TensorHandle& src, DataType data_type) {
  if (gemma3_bfloat16_fix_) {
    return model_builder_.Cast(src, data_type);
  } else {
    return src;
  }
}

absl::StatusOr<GpuModel> TransformerBuilder::CreateStackedTransformerModel(
    int sequence_size, int batch_size,
    GpuModelBuilder::TensorHandle* input_handle,
    GpuModelBuilder::TensorHandle* media_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
    std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
    GpuModelBuilder::TensorHandle* output_handle) {
  CleanState();

  config_.sequence_size = sequence_size;
  config_.batch_size = batch_size;

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);

  // Starting tensors should be f32 if bf16-on-f16
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF32);

  *input_handle = model_builder_.AddTensor(
      GetInputTensorDescriptor(batch_size, sequence_size));
  const bool is_media_model = media_handle != nullptr && media_cache_size_ > 0;
  if (is_media_model) {
    *media_handle =
        model_builder_.AddTensor(GetMediaEmbeddingsTensorDescriptor());
  }

  *params_i32_handle = model_builder_.AddTensor(GetParamsTensorDescriptor());

  auto segment_pos = MakeFillSegmentPos(*params_i32_handle);

  // Cache and attention mask can stay f16 for bf16
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF16);

  ASSIGN_OR_RETURN(auto global_attention_mask,
                   MakeFillAttention(*params_i32_handle, input_handle));
  ASSIGN_OR_RETURN(auto local_attention_mask,
                   MakeFillLocalAttention(*params_i32_handle, input_handle));
  CreateCache(key_cached_tensors, value_cached_tensors);

  // But we change other ops here to be f32 for bf16 models
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF32);

  bool can_process_audio_hard_tokens;
  GpuModelBuilder::TensorHandle cond_text;
  GpuModelBuilder::TensorHandle cond_hard_tokens;
  GpuModelBuilder::TensorHandle hard_tokens;
  GpuModelBuilder::TensorHandle gathered_media_embeddings;
  GpuModelBuilder::TensorHandle conded_current_tokens = *input_handle;
  if (is_media_model) {  // Multimodal
    // Get conditional tensors to split on
    cond_text = model_builder_.Elementwise(*input_handle, {0},
                                           OperationType::kGreaterEqual);
    cond_text = model_builder_.Cast(cond_text, DataType::kInt32);
    // Get zero tensor handle of size t
    GpuModelBuilder::TensorHandle zero =
        model_builder_.Elementwise(*input_handle, {0}, OperationType::kMul);
    // Get media indices and gather audio embeddings
    GpuModelBuilder::TensorHandle media_index =
        model_builder_.SelectV2(cond_text, zero, *input_handle);
    media_index =
        model_builder_.Elementwise(media_index, {-1}, OperationType::kMul);
    media_index = model_builder_.Reshape(
        media_index, BHWC(1, 1, 1, media_index.tensor_desc.GetBHWCShape().w));
    ASSIGN_OR_RETURN(gathered_media_embeddings,
                     GatherMedia(*media_handle, media_index));

    // Process audio hard token embeddings
    can_process_audio_hard_tokens = tensor_loader_->HasTensor(
                                        "params.audio.transformer.embedder."
                                        "audio_input_embedding_extra.w") &&
                                    config_.audio_vocab_extra_dim > 0 &&
                                    config_.audio_input_embedding_dim > 0 &&
                                    config_.audio_vocab_offset > 0;

    if (can_process_audio_hard_tokens) {
      // Find audio hard token cond tensor
      cond_hard_tokens = model_builder_.Elementwise(
          *input_handle, {config_.audio_vocab_offset},
          OperationType::kGreaterEqual);
      cond_hard_tokens =
          model_builder_.Cast(cond_hard_tokens, DataType::kInt32);
      // Grab hard tokens from current tokens.
      // Note that we are only using the audio input embedding extra table.
      // As such, in addition to adding the audio vocab offset, we also need to
      // add the length of the audio input embedding table, which is 256000.
      // Note that this calculation assumes that the number of audio hard
      // tokens, represented by the length of audio_input_embedding_table, is
      // equal to config_.vocabulary_size.
      hard_tokens = model_builder_.Elementwise(
          *input_handle, {config_.audio_vocab_offset}, OperationType::kSub);
      hard_tokens =
          model_builder_.SelectV2(cond_hard_tokens, hard_tokens, zero);
      // Adjust cond_text to ignore tokens in the audio hard tokens range.
      cond_text = model_builder_.SelectV2(cond_hard_tokens, zero, cond_text);
    }

    // Get the current tokens after applying cond text
    conded_current_tokens =
        model_builder_.SelectV2(cond_text, *input_handle, zero);
  }

  ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle embeddings,
                   MakeEmbLookUp(conded_current_tokens));
  if (ScaleEmbeddingsBySqrtModelDim()) {
    embeddings = model_builder_.Multiplication(
        embeddings, sqrt(static_cast<float>(config_.model_dimension)));
  }

  if (!config_.skip_absolute_positional_embeddings) {
    // Add absolute positional embedding.
    embeddings = model_builder_.PositionalEmbedding(embeddings, segment_pos);
  }

  // Add media embeddings in
  if (is_media_model) {
    embeddings = model_builder_.SelectV2(cond_text, embeddings,
                                         gathered_media_embeddings);
    // Process audio hard tokens
    if (can_process_audio_hard_tokens) {
      ASSIGN_OR_RETURN(
          auto audio_input_embedding_extra,
          LoadWeights(
              "params.audio.transformer.embedder.audio_input_embedding_extra",
              OHWI(config_.audio_vocab_extra_dim, 1, 1,
                   config_.audio_input_embedding_dim),
              /*swap_dims=*/false));
      auto emb_hard_tokens =
          MakeEmbLookUpOp(hard_tokens, audio_input_embedding_extra);
      emb_hard_tokens = model_builder_.Multiplication(
          emb_hard_tokens, std::sqrt(config_.audio_input_embedding_dim));

      if (config_.audio_dual_norm_soft_and_hard_tokens) {
        emb_hard_tokens = MakeRMSNormalization(
            emb_hard_tokens,
            "params.audio.transformer.embedder.audio_embedding_norm",
            /*scale_shift=*/0.0f);
      }
      ASSIGN_OR_RETURN(
          emb_hard_tokens,
          MakeLinearAutoPrecision(
              emb_hard_tokens,
              "params.audio.transformer.embedder.audio_input_projection",
              config_.model_dimension));
      if (config_.audio_dual_norm_soft_and_hard_tokens) {
        emb_hard_tokens =
            model_builder_.RMSNormalization(emb_hard_tokens, /*epsilon=*/1e-6f);
      }

      // Add in audio hard tokens
      embeddings = model_builder_.SelectV2(cond_hard_tokens, emb_hard_tokens,
                                           embeddings);
    }
  }

  const auto& layer_names = GetLayerNames();
  for (int i = 0; i < config_.stack_size; ++i) {
    const std::string layer_name = layer_names[i];
    GpuModelBuilder::OptionalNodeContext context;
    if (config_.use_optional_transformer_layers) {
      context = model_builder_.BeginOptionalNodes(
          i | kTransformerLayerOptionTag, embeddings);
    }

    const bool is_global = IsGlobalLayer(config_, i);
    const GpuModelBuilder::TensorHandle* attention_mask =
        is_global ? &global_attention_mask : &local_attention_mask;
    ASSIGN_OR_RETURN(
        auto new_t,
        MakeLayer(embeddings, *attention_mask, segment_pos,
                  key_cached_tensors->at(i), value_cached_tensors->at(i),
                  *params_i32_handle, is_global, layer_name));
    if (is_media_model &&
        config_.residual_adapter
            .has_value()) {  // Interleave residual adapter layers.
      if (config_.residual_adapter->where_to_interleave() ==
          odml::infra::proto::TransformerParameters::ALL) {
        ASSIGN_OR_RETURN(
            new_t,
            MakeResidualAdapter(new_t, layer_names[i] + ".residual_adapter"));
      } else if (config_.residual_adapter->where_to_interleave() ==
                 odml::infra::proto::TransformerParameters::EVERY_OTHER_4) {
        if (i % 4 == 0) {
          ASSIGN_OR_RETURN(
              new_t, MakeResidualAdapter(
                         new_t, layer_names[i / 4] + ".residual_adapter"));
        }
      } else {
        return absl::InvalidArgumentError(
            "Invalid residual adapter interleave config");
      }
    }
    if (config_.use_optional_transformer_layers) {
      RETURN_IF_ERROR(
          model_builder_.EndOptionalNodes(std::move(context), new_t));
    } else {
      embeddings = std::move(new_t);
    }
  }

  if (!config_.selectable_final_ln) {
    embeddings = MakeNormalization(embeddings, "params.lm.final_ln");
  } else {
    for (int i = 0; i < *config_.selectable_final_ln; ++i) {
      auto context =
          model_builder_.BeginOptionalNodes(i | kFinalLnOptionTag, embeddings);
      auto final_ln = MakeNormalization(
          embeddings, "params.lm." + std::to_string(i) + ".final_ln");
      RETURN_IF_ERROR(
          model_builder_.EndOptionalNodes(std::move(context), final_ln));
    }
  }

  std::vector<GpuModelBuilder::ValueId> input_ids = {input_handle->id,
                                                     params_i32_handle->id};
  if (is_media_model) {
    input_ids.push_back(media_handle->id);
  }
  *output_handle = embeddings;
  std::vector<GpuModelBuilder::ValueId> output_ids = {output_handle->id,
                                                      input_handle->id};

  for (int i = 0; i < key_cached_tensors->size(); ++i) {
    input_ids.push_back(key_cached_tensors->at(i).id);
    output_ids.push_back(key_cached_tensors->at(i).id);
  }
  for (int i = 0; i < value_cached_tensors->size(); ++i) {
    input_ids.push_back(value_cached_tensors->at(i).id);
    output_ids.push_back(value_cached_tensors->at(i).id);
  }
  GpuModel gpu_model;
  RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

void TransformerBuilder::ExtractLocalSlidingWindowMask(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src_mask,
    const GpuModelBuilder::TensorHandle& dst_mask,
    const GpuModelBuilder::TensorHandle& params_tensor) {
  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int seq_id = ucl::GetGlobalId<0>();
  int dst_cache_slice_id = ucl::GetGlobalId<2>();
  if (seq_id >= args.dst_mask.Width() || dst_cache_slice_id >= args.dst_mask.Slices() || ucl::GetGlobalId<1>() != 0) {
    return;
  }

  int token_index_offset = args.params.Read($0);
  bool mask_values[4];
  for (int i = 0; i < 4; ++i) {
    mask_values[i] = true;
    int dst_cache_id = dst_cache_slice_id * 4 + i;
    int src_cache_id = max(0, token_index_offset - args.window_size + 1) + dst_cache_id;
    if (dst_cache_id < args.dst_mask.Channels() && src_cache_id >= 0 && src_cache_id < args.src_mask.Channels()) {
      args.src_mask.ReadPerChannel(mask_values[i], seq_id, 0, src_cache_id);
    }
  }
  args.dst_mask::type mask_dst_val;
  mask_dst_val.x = mask_values[0];
  mask_dst_val.y = mask_values[1];
  mask_dst_val.z = mask_values[2];
  mask_dst_val.w = mask_values[3];
  args.dst_mask.Write(mask_dst_val, seq_id, 0, dst_cache_slice_id);
})",
                                      LlmRuntimeParams::kTokenOffsetIndex);

  GPUOperation op;
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.AddSrcTensor("src_mask", src_mask.tensor_desc);
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  op.AddSrcBuffer("params", params_buffer);
  op.AddDstTensor("dst_mask", dst_mask.tensor_desc);
  op.args_.AddInt("window_size", config_.sliding_window_size.value());
  op.code_ = code;

  builder.AddGpuOperation({src_mask, params_tensor}, {dst_mask},
                          std::make_unique<GPUOperation>(std::move(op)),
                          "extract_local_mask");
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::GetLocalMask(
    GpuModelBuilder::TensorHandle* params_tensor,
    GpuModelBuilder::TensorHandle* external_local_mask,
    GpuModelBuilder::TensorHandle* current_tokens) {
  ASSIGN_OR_RETURN(const int mask_width,
                   GetMaskWidth(config_.attention_mask_type));
  if (config_.enable_decode_logits) {
    if (UseSmallLocalCache(config_)) {
      // return MakeFillLocalAttention(*params_tensor);
      auto small_local_mask_tensor = model_builder_.AddTensor(
          BHWC(1, 1, mask_width, GetLocalCacheSize()), DataType::kBool);
      ExtractLocalSlidingWindowMask(model_builder_, *external_local_mask,
                                    small_local_mask_tensor, *params_tensor);
      return small_local_mask_tensor;
    } else {
      return *external_local_mask;
    }
  } else {
    return MakeFillLocalAttention(*params_tensor, current_tokens);
  }
}

absl::StatusOr<GpuModel> TransformerBuilder::CreateStackedTransformerModelV2(
    int sequence_size, int batch_size,
    GpuModelBuilder::TensorHandle* current_tokens,
    GpuModelBuilder::TensorHandle* media_handle,
    GpuModelBuilder::TensorHandle* tokens_ids_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
    std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
    GpuModelBuilder::TensorHandle* output_handle,
    std::vector<odml::infra::gpu::EmbeddingWeights>* ple_weights,
    std::vector<odml::infra::gpu::EmbeddingWeights>* ngrammer_weights,
    std::vector<GpuModelBuilder::TensorHandle>* vision_ple_tensors,
    GpuModelBuilder::TensorHandle* segment_pos_tensor,
    GpuModelBuilder::TensorHandle* global_mask_tensor,
    GpuModelBuilder::TensorHandle* local_mask_tensor) {
  RET_CHECK(UseGoogleInternalModelV2());
  CleanState();
  config_.sequence_size = sequence_size;
  config_.batch_size = batch_size;

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);
  *current_tokens = model_builder_.AddTensor(
      GetInputTensorDescriptor(batch_size, sequence_size));
  bool is_media_model = media_handle != nullptr && media_cache_size_ > 0;
  if (is_media_model) {
    *media_handle =
        model_builder_.AddTensor(GetMediaEmbeddingsTensorDescriptor());
  }

  GpuModelBuilder::TensorHandle previous_tokens =
      model_builder_.AddTensor(config_.batch_size, 1, config_.sequence_size, 1,
                               TensorStorageType::kBuffer, DataType::kInt32);

  *params_i32_handle = model_builder_.AddTensor(GetParamsTensorDescriptor());
  *tokens_ids_handle = model_builder_.AddTensor(GetTokensTensorDescriptor());

  WriteCurrentTokens(*current_tokens, *tokens_ids_handle, *params_i32_handle);
  ReadPreviousTokens(previous_tokens, *tokens_ids_handle, *params_i32_handle);

  GpuModelBuilder::TensorHandle conded_current_tokens = *current_tokens;

  // Declare multimodal tensors which will be used after embedding
  GpuModelBuilder::TensorHandle cond_hard_tokens;
  GpuModelBuilder::TensorHandle hard_tokens;
  GpuModelBuilder::TensorHandle cond_text;
  GpuModelBuilder::TensorHandle gathered_media_embeddings;
  std::vector<GpuModelBuilder::TensorHandle> gathered_vision_ple;
  bool can_process_audio_hard_tokens;
  if (is_media_model) {  // Multimodal
    conded_current_tokens = *current_tokens;
    // Get conditional tensors to split on
    cond_text = model_builder_.Elementwise(*current_tokens, {0},
                                           OperationType::kGreaterEqual);
    cond_text = model_builder_.Cast(cond_text, DataType::kInt32);
    // Get zero tensor handle of size current_tokens
    GpuModelBuilder::TensorHandle zero =
        model_builder_.Elementwise(*current_tokens, {0}, OperationType::kMul);
    // Get media indices and gather audio embeddings
    GpuModelBuilder::TensorHandle media_index =
        model_builder_.SelectV2(cond_text, zero, *current_tokens);
    media_index =
        model_builder_.Elementwise(media_index, {-1}, OperationType::kMul);
    media_index = model_builder_.Reshape(
        media_index, BHWC(1, 1, 1, media_index.tensor_desc.GetBHWCShape().w));
    ASSIGN_OR_RETURN(gathered_media_embeddings,
                     GatherMedia(*media_handle, media_index));

    // Process audio hard token embeddings
    can_process_audio_hard_tokens =
        tensor_loader_->HasTensor(
            "transformer.embedder.audio_input_embedding_extra.w") &&
        config_.audio_vocab_extra_dim > 0 &&
        config_.audio_input_embedding_dim > 0 && config_.audio_vocab_offset > 0;

    if (can_process_audio_hard_tokens) {
      // Find audio hard token cond tensor
      cond_hard_tokens = model_builder_.Elementwise(
          *current_tokens, {config_.audio_vocab_offset},
          OperationType::kGreaterEqual);
      cond_hard_tokens =
          model_builder_.Cast(cond_hard_tokens, DataType::kInt32);
      // Grab hard tokens from current tokens.
      // Note that we are only using the audio input embedding extra table.
      // As such, in addition to adding the audio vocab offset, we also need to
      // add the length of the audio input embedding table, which is 256000.
      // Note that this calculation assumes that the number of audio hard
      // tokens, represented by the length of audio_input_embedding_table, is
      // equal to config_.vocabulary_size.
      hard_tokens = model_builder_.Elementwise(
          *current_tokens,
          {config_.audio_vocab_offset + config_.vocabulary_size -
           config_.audio_vocab_extra_dim},
          OperationType::kSub);
      hard_tokens =
          model_builder_.SelectV2(cond_hard_tokens, hard_tokens, zero);
      // Adjust cond_text to ignore tokens in the audio hard tokens range.
      cond_text = model_builder_.SelectV2(cond_hard_tokens, zero, cond_text);
    }

    // Get the current tokens after applying cond text
    conded_current_tokens =
        model_builder_.SelectV2(cond_text, *current_tokens, zero);
    // Process vision ple
    int vision_stack_size = config_.stack_size - config_.extra_stack_size;
    vision_ple_tensors->reserve(vision_stack_size);
    const TensorDescriptor& vision_ple_desc = GetVisionPleTensorDescriptor();
    for (int i = 0; i < vision_stack_size; ++i) {
      vision_ple_tensors->push_back(model_builder_.AddTensor(vision_ple_desc));
    }
    gathered_vision_ple =
        std::vector<GpuModelBuilder::TensorHandle>(vision_ple_tensors->size());
    for (int i = 0; i < vision_ple_tensors->size(); ++i) {
      ASSIGN_OR_RETURN(gathered_vision_ple[i],
                       GatherMedia((*vision_ple_tensors)[i], media_index));
    }
  }

  ASSIGN_OR_RETURN(const int mask_width,
                   GetMaskWidth(config_.attention_mask_type));
  const BHWC mask_shape = BHWC(1, 1, mask_width, config_.cache_size);

  if (config_.enable_decode_logits) {
    RET_CHECK(segment_pos_tensor != nullptr);
    RET_CHECK(global_mask_tensor != nullptr);
    RET_CHECK(local_mask_tensor != nullptr);
    *segment_pos_tensor = model_builder_.AddTensor(
        BHWC(1, 1, config_.sequence_size, 1), DataType::kInt32);
    *global_mask_tensor = model_builder_.AddTensor(mask_shape, DataType::kBool);
    *local_mask_tensor = model_builder_.AddTensor(mask_shape, DataType::kBool);
  }
  auto segment_pos = config_.enable_decode_logits
                         ? *segment_pos_tensor
                         : MakeFillSegmentPos(*params_i32_handle);

  ASSIGN_OR_RETURN(auto global_attention_mask,
                   config_.enable_decode_logits
                       ? *global_mask_tensor
                       : MakeFillAttention(*params_i32_handle, current_tokens));
  ASSIGN_OR_RETURN(
      auto local_attention_mask,
      GetLocalMask(params_i32_handle, local_mask_tensor, current_tokens));

  CreateCache(key_cached_tensors, value_cached_tensors);
  ASSIGN_OR_RETURN(
      auto embeddings_full,
      LoadWeights("transformer.embedder.input_embedding",
                  OHWI(config_.vocabulary_size, 1, 1, config_.model_dimension),
                  /*swap_dims=*/false));
  auto embeddings = MakeEmbLookUpOp(conded_current_tokens, embeddings_full);

  std::vector<GpuModelBuilder::TensorHandle> ngrammer_embs;
  if (config_.load_ngrammer_embedding_dynamically) {
    // Ngrammer embeddings managed externally.
    BHWC ngrammer_shape =
        BHWC(1, 1, config_.sequence_size,
             config_.model_dimension / config_.number_of_heads);
    TensorDescriptor ngrammer_desc =
        GetEmbeddingDesc(ngrammer_shape, *config_.ngram_vocabulary_size,
                         GetNgrammerEmbeddingName(0));

    TensorDescriptor ngrammer_scale_desc(
        DataType::kFloat32, TensorStorageType::kBuffer, Layout::kLinear);
    ngrammer_scale_desc.SetBHWCShape(BHWC(1, 1, 1, config_.sequence_size));

    ngrammer_weights->resize(config_.number_of_heads);
    ngrammer_embs.resize(config_.number_of_heads);
    for (int i = 0; i < config_.number_of_heads; ++i) {
      auto& ngrammer_weight = (*ngrammer_weights)[i];
      ngrammer_weight.weights = model_builder_.AddTensor(ngrammer_desc);
      ngrammer_weight.scales = model_builder_.AddTensor(ngrammer_scale_desc);
      ngrammer_weight.weights_data.resize(ngrammer_desc.GetMemorySizeInBytes());
      ngrammer_weight.scales_data.resize(
          ngrammer_scale_desc.GetMemorySizeInBytes() /
          SizeOf(DataType::kFloat32));
      ASSIGN_OR_RETURN(
          ngrammer_embs[i],
          DequantizeEmbedding(ngrammer_shape, ngrammer_weight.weights,
                              ngrammer_weight.scales));
    }
  } else {
    ASSIGN_OR_RETURN(ngrammer_embs, GetNGrammerEmbeddings(
                                        conded_current_tokens, previous_tokens,
                                        odml::infra::gpu::kNgrammerPrimes));
  }
  ASSIGN_OR_RETURN(embeddings,
                   MakeNGrammerLayer(embeddings, ngrammer_embs,
                                     "transformer.embedder.text_ngrammer"));

  std::vector<GpuModelBuilder::TensorHandle> per_layer_embeddings;
  if (config_.load_per_layer_embedding_dynamically) {
    // Per layer embeddings managed externally.
    BHWC ple_shape = BHWC(1, 1, config_.sequence_size,
                          *config_.per_layer_embedding_dimension);
    TensorDescriptor ple_desc = GetEmbeddingDesc(
        ple_shape, config_.vocabulary_size, GetPerLayerEmbeddingName(0));

    TensorDescriptor ple_scale_desc(
        DataType::kFloat32, TensorStorageType::kBuffer, Layout::kLinear);
    ple_scale_desc.SetBHWCShape(BHWC(1, 1, 1, config_.sequence_size));

    ple_weights->reserve(config_.stack_size);
    per_layer_embeddings.reserve(config_.stack_size);
    for (int i = 0; i < config_.stack_size - config_.extra_stack_size; ++i) {
      // EE models may not have PLE on extra layers.
      if (!tensor_loader_->HasTensor(GetPerLayerEmbeddingName(i) + ".w")) {
        break;
      }
      ple_weights->push_back({});
      auto& ple_weight = (*ple_weights)[i];
      ple_weight.weights = model_builder_.AddTensor(ple_desc);
      ple_weight.scales = model_builder_.AddTensor(ple_scale_desc);
      ple_weight.weights_data.resize(ple_desc.GetMemorySizeInBytes());
      ple_weight.scales_data.resize(ple_scale_desc.GetMemorySizeInBytes() /
                                    SizeOf(DataType::kFloat32));
      ASSIGN_OR_RETURN(auto ple_embedding,
                       DequantizeEmbedding(ple_shape, ple_weight.weights,
                                           ple_weight.scales));
      per_layer_embeddings.push_back(ple_embedding);
    }
  } else {
    ASSIGN_OR_RETURN(per_layer_embeddings,
                     GetPerLayerEmbeddings(conded_current_tokens));
  }
  // Add media embeddings in
  if (is_media_model) {
    embeddings = model_builder_.SelectV2(cond_text, embeddings,
                                         gathered_media_embeddings);
    // Add vision ple
    for (int i = 0; i < per_layer_embeddings.size(); ++i) {
      per_layer_embeddings[i] =
          model_builder_.Add(per_layer_embeddings[i], gathered_vision_ple[i]);
    }
    // Process audio hard tokens
    if (can_process_audio_hard_tokens) {
      ASSIGN_OR_RETURN(
          auto audio_input_embedding_extra,
          LoadWeights("transformer.embedder.audio_input_embedding_extra",
                      OHWI(config_.audio_vocab_extra_dim, 1, 1,
                           config_.audio_input_embedding_dim),
                      /*swap_dims=*/false));
      auto emb_hard_tokens =
          MakeEmbLookUpOp(hard_tokens, audio_input_embedding_extra);
      emb_hard_tokens = model_builder_.Multiplication(
          emb_hard_tokens, std::sqrt(config_.audio_input_embedding_dim));
      ASSIGN_OR_RETURN(
          emb_hard_tokens,
          MakeLinearAutoPrecision(emb_hard_tokens,
                                  "transformer.embedder.audio_input_projection",
                                  config_.model_dimension));

      // Add in audio hard tokens
      embeddings = model_builder_.SelectV2(cond_hard_tokens, emb_hard_tokens,
                                           embeddings);
    }
  }

  auto activations = embeddings;

  const auto& layer_names = GetLayerNames();
  for (int i = 0; i < config_.stack_size; ++i) {
    const std::string layer_name = layer_names[i];
    GpuModelBuilder::OptionalNodeContext context;
    if (config_.use_optional_transformer_layers) {
      context = model_builder_.BeginOptionalNodes(
          i | kTransformerLayerOptionTag, activations);
    }

    const bool is_global = IsGlobalLayer(config_, i);
    const GpuModelBuilder::TensorHandle* active_mask =
        is_global ? &global_attention_mask : &local_attention_mask;
    ASSIGN_OR_RETURN(
        auto new_activations,
        MakeLayerV2(activations, *active_mask, segment_pos,
                    key_cached_tensors->at(i), value_cached_tensors->at(i),
                    per_layer_embeddings.size() > i ? &per_layer_embeddings[i]
                                                    : nullptr,
                    *params_i32_handle, is_global, layer_name));

    if (config_.use_optional_transformer_layers) {
      RETURN_IF_ERROR(
          model_builder_.EndOptionalNodes(std::move(context), new_activations));
    } else {
      activations = std::move(new_activations);
    }
  }

  if (!config_.selectable_final_ln) {
    if (config_.extra_stack_size > 0) {
      activations =
          MakeNormalization(activations, "extra.transformer.final_norm");
    } else {
      activations = MakeNormalization(activations, "transformer.final_norm");
    }
  } else {
    for (int i = 0; i < *config_.selectable_final_ln; ++i) {
      auto context =
          model_builder_.BeginOptionalNodes(i | kFinalLnOptionTag, activations);
      auto final_ln = MakeNormalization(
          activations, "params.lm." + std::to_string(i) + ".final_ln");
      RETURN_IF_ERROR(
          model_builder_.EndOptionalNodes(std::move(context), final_ln));
    }
  }

  std::vector<GpuModelBuilder::ValueId> input_ids = {
      current_tokens->id, tokens_ids_handle->id, params_i32_handle->id};
  if (is_media_model) {
    input_ids.push_back(media_handle->id);
    for (const auto& ple : *vision_ple_tensors) {
      input_ids.push_back(ple.id);
    }
  }
  *output_handle = activations;
  std::vector<GpuModelBuilder::ValueId> output_ids = {
      output_handle->id, current_tokens->id, tokens_ids_handle->id};

  for (int i = 0; i < key_cached_tensors->size(); ++i) {
    input_ids.push_back(key_cached_tensors->at(i).id);
    output_ids.push_back(key_cached_tensors->at(i).id);
  }
  for (int i = 0; i < value_cached_tensors->size(); ++i) {
    input_ids.push_back(value_cached_tensors->at(i).id);
    output_ids.push_back(value_cached_tensors->at(i).id);
  }
  if (ple_weights) {
    // Per layer embeddings managed externally.
    for (const auto& ple : *ple_weights) {
      input_ids.push_back(ple.weights.id);
      input_ids.push_back(ple.scales.id);
    }
  }
  if (ngrammer_weights) {
    // Ngrammer embeddings managed externally.
    for (const auto& ngrammer : *ngrammer_weights) {
      input_ids.push_back(ngrammer.weights.id);
      input_ids.push_back(ngrammer.scales.id);
    }
  }
  if (segment_pos_tensor) {
    input_ids.push_back(segment_pos_tensor->id);
  }
  if (global_mask_tensor) {
    input_ids.push_back(global_mask_tensor->id);
    // Useful for debugging mask.
    // output_ids.push_back(global_mask_tensor->id);
  }
  if (local_mask_tensor) {
    input_ids.push_back(local_mask_tensor->id);
    // Useful for debugging mask.
    // output_ids.push_back(local_mask_tensor->id);
  }

  GpuModel gpu_model;
  RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

absl::StatusOr<GpuModel> TransformerBuilder::CreateSoftmaxModel(
    GpuModelBuilder::TensorHandle* input,
    GpuModelBuilder::TensorHandle* output) {
  CleanState();
  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);
  *output = model_builder_.Softmax(*input);
  GpuModel gpu_model;
  RETURN_IF_ERROR(
      model_builder_.GetGpuModel({input->id}, {output->id}, &gpu_model));
  return std::move(gpu_model);
}

absl::StatusOr<GpuModel> TransformerBuilder::CreatePostProcessingModel(
    GpuModelBuilder::TensorHandle* input_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    GpuModelBuilder::TensorHandle* params_f32_handle,
    GpuModelBuilder::TensorHandle* tokens_ids_handle,
    GpuModelBuilder::TensorHandle* stacked_transformer_input_handle,
    GpuModelBuilder::TensorHandle* pre_softmax_handle,
    GpuModelBuilder::TensorHandle* topk_tokens_handle,
    GpuModelBuilder::TensorHandle* topk_probabilities_handle,
    GpuModelBuilder::TensorHandle* constraint_mask_handle, int sequence_size) {
  CleanState();

  config_.sequence_size = sequence_size;
  config_.batch_size = kv_cache_batch_size_;

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);
  *input_handle = model_builder_.AddTensor(
      BHWC(config_.batch_size, 1, config_.sequence_size,
           config_.model_dimension),
      data_type_);
  auto t = *input_handle;

  if (UseGoogleInternalModelV2() || UseGemmaModelV3NPlus()) {
    if (config_.centroid_embedder_config.has_value()) {
      // Uses efficient centroid embedder for fast decoding (like V4 MTPs)
      ASSIGN_OR_RETURN(t, GetCentroidEmbeddings(t, "eevee.embedder"));
    } else {
      ASSIGN_OR_RETURN(
          t, MakeConvAutoPrecision(t, "transformer.embedder.input_embedding",
                                   Layout::kOIHW, config_.vocabulary_size));
    }
  } else {
    ASSIGN_OR_RETURN(t,
                     MakeLinearAutoPrecision(t, "params.lm.softmax.logits_ffn",
                                             config_.vocabulary_size));
  }

  *params_i32_handle = model_builder_.AddTensor(GetParamsTensorDescriptor());

  *params_f32_handle = model_builder_.AddTensor(
      1, 1, 1, GetF32ParamsCount(config_.sequence_size),
      TensorStorageType::kBuffer, DataType::kFloat32);

  *stacked_transformer_input_handle =
      model_builder_.AddTensor(config_.batch_size, 1, config_.sequence_size, 1,
                               TensorStorageType::kBuffer, DataType::kInt32);

  *tokens_ids_handle = model_builder_.AddTensor(GetTokensTensorDescriptor());

  if (config_.final_proj_soft_cap_value > 0.0f) {
    float cap_value = config_.final_proj_soft_cap_value;
    t = model_builder_.Multiplication(t, 1.0f / cap_value);
    t = model_builder_.Elementwise(t, OperationType::kTanh);
    t = model_builder_.Multiplication(t, cap_value);
  }
  if (constraint_mask_handle) {
    // Add a penalty to the logits of the disallowed tokens before softmax.
    *constraint_mask_handle = model_builder_.AddTensor(
        config_.batch_size, 1, 1, config_.vocabulary_size,
        TensorStorageType::kBuffer, t.tensor_desc.GetDataType());
    float penalty = t.tensor_desc.GetDataType() == DataType::kFloat32
                        ? std::numeric_limits<float>::max()
                        : kMaxHalf;
    penalty *= 0.7f;
    auto mask = model_builder_.Multiplication(*constraint_mask_handle, penalty);
    t = model_builder_.Elementwise(t, mask, OperationType::kSub);
  }
  if (pre_softmax_handle) {
    *pre_softmax_handle = t;
  }
  auto results = model_builder_.TopK(t, config_.max_top_k);
  auto out_max = results[0];
  auto out_ind = results[1];

  auto temperature = model_builder_.GetScalarTensor(
      *params_f32_handle,
      BHWC(0, 0, 0, config_.batch_size * config_.sequence_size));
  out_max = model_builder_.Multiplication(
      out_max,
      model_builder_.Cast(temperature, out_max.tensor_desc.GetDataType()));

  SoftmaxRuntimeCheckDesc runtime_check = {.end_ch_index =
                                               LlmRuntimeParams::kTopKIndex};
  out_max = model_builder_.Softmax(out_max, runtime_check, params_i32_handle);
  auto probabilities = model_builder_.SubTensor(
      *params_f32_handle, BHWC(0, 0, 0, 0),
      BHWC(1, 1, 1, config_.batch_size * config_.sequence_size));
  probabilities = model_builder_.Reshape(
      probabilities, BHWC(1, 1, config_.sequence_size, config_.batch_size));
  probabilities = model_builder_.Transpose(probabilities, BHWC(3, 1, 2, 0));
  auto samples =
      model_builder_.Sampling(out_max, out_ind, probabilities,
                              *params_i32_handle, LlmRuntimeParams::kTopKIndex);
  if (stacked_transformer_input_handle->tensor_desc.GetStorageType() !=
      samples.tensor_desc.GetStorageType()) {
    // storage types not the same, need copy op
    model_builder_.Copy(samples, *stacked_transformer_input_handle);
  } else {
    *stacked_transformer_input_handle = samples;
  }
  CopyTokensToOutputBuffer(*stacked_transformer_input_handle,
                           *params_i32_handle, *tokens_ids_handle);

  std::vector<GpuModelBuilder::ValueId> input_ids = {
      input_handle->id, params_i32_handle->id, params_f32_handle->id};
  if (constraint_mask_handle) {
    input_ids.push_back(constraint_mask_handle->id);
  }
  std::vector<GpuModelBuilder::ValueId> output_ids = {
      tokens_ids_handle->id, stacked_transformer_input_handle->id};

  if (topk_tokens_handle && topk_probabilities_handle) {
    *topk_tokens_handle = out_ind;
    output_ids.push_back(topk_tokens_handle->id);
    *topk_probabilities_handle = out_max;
    output_ids.push_back(topk_probabilities_handle->id);
  } else if (!topk_tokens_handle != !topk_probabilities_handle) {
    return absl::InvalidArgumentError(
        "topk_tokens_handle and topk_probabilities_handle must both be "
        "provided or both be nullptr.");
  }

  if (pre_softmax_handle) {
    output_ids.push_back(pre_softmax_handle->id);
  }

  GpuModel gpu_model;
  RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

absl::StatusOr<GpuModel> TransformerBuilder::CreateCleanCacheModel(
    std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
    std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    GpuModelBuilder::TensorHandle* indices_to_remove_handle) {
  CleanState();
  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);

  // Cache can stay f16 for bf16
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF16);

  *params_i32_handle = model_builder_.AddTensor(GetParamsTensorDescriptor());
  CreateCache(key_cached_tensors, value_cached_tensors);

  std::vector<GpuModelBuilder::TensorHandle> inputs = {*params_i32_handle};
  if (indices_to_remove_handle != nullptr) {
    TensorDescriptor td = TensorDescriptor{
        DataType::kInt32, TensorStorageType::kBuffer, Layout::kHWC};
    td.SetBHWCShape(BHWC(1, 1, config_.max_indices_count_in_clean_cache_at, 1));
    *indices_to_remove_handle = model_builder_.AddTensor(td);
    inputs.push_back(*indices_to_remove_handle);
  }

  for (int i = 0; i < key_cached_tensors->size(); ++i) {
    const std::string op_name = "clean_cache stack#" + std::to_string(i);
    const bool is_global = IsGlobalLayer(config_, i);
    model_builder_.AddGpuOperation(
        inputs, {key_cached_tensors->at(i), value_cached_tensors->at(i)},
        CleanCacheOp(key_cached_tensors->at(i), value_cached_tensors->at(i),
                     config_, kv_cache_batch_size_, indices_to_remove_handle,
                     is_global),
        "clean_cache");
  }

  std::vector<GpuModelBuilder::ValueId> input_ids, output_ids;
  for (int i = 0; i < key_cached_tensors->size(); ++i) {
    input_ids.push_back(key_cached_tensors->at(i).id);
    output_ids.push_back(key_cached_tensors->at(i).id);
  }
  for (int i = 0; i < value_cached_tensors->size(); ++i) {
    input_ids.push_back(value_cached_tensors->at(i).id);
    output_ids.push_back(value_cached_tensors->at(i).id);
  }

  // Set default back to f32 for bf16 models
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF32);

  GpuModel gpu_model;
  RETURN_IF_ERROR(model_builder_.GetGpuModel({}, output_ids, &gpu_model));
  return std::move(gpu_model);
}

TensorDescriptor TransformerBuilder::GetEmbeddingDesc(
    const BHWC& shape, int vocab_size, const std::string& base_name) const {
  DataType data_type = tensor_loader_->GetDataTypeForWeights(
      base_name + ".w", shape.c * vocab_size);
  TensorDescriptor desc(data_type, TensorStorageType::kBuffer, Layout::kLinear);
  int count = shape.DimensionsProduct();
  if (data_type == DataType::kInt4) {
    count /= 2;
  }
  if (data_type == DataType::kInt2) {
    count /= 4;
  }
  desc.SetBHWCShape(BHWC(1, 1, 1, count));
  return desc;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::DequantizeEmbedding(
    const BHWC& dst_shape, const GpuModelBuilder::TensorHandle& weights,
    const GpuModelBuilder::TensorHandle& scales,
    const GpuModelBuilder::TensorHandle* lookup) {
  auto dst = model_builder_.AddTensor(dst_shape, data_type_);
  if (dst.tensor_desc.GetBHWCShape().c % 16) {
    return absl::InternalError("Dest channels must be a multiple of 16.");
  }
  class DequantizeEmbeddingOp : public GPUOperation {
   public:
    DequantizeEmbeddingOp() = default;
    int3 GetGridSize() const override {
      return int3(dst_[0]->Width(), dst_[0]->Slices() / 4, 1);
    }
  };
  DequantizeEmbeddingOp custom_op;
  BufferDescriptor desc;
  desc.element_type = DataType::kUint32;
  const DataType data_type = weights.tensor_desc.GetDataType();
  desc.element_size = SizeInBitsOf(data_type) / 2;
  custom_op.AddSrcBuffer("weights", desc);
  custom_op.AddSrcTensor("scale", scales.tensor_desc);

  custom_op.AddDstTensor("dst", dst.tensor_desc);
  custom_op.code_ = R"(
MAIN_FUNCTION($0) {
  int DST_X = ucl::GetGlobalId<0>();
  int DST_S = ucl::GetGlobalId<1>();
  int DST_Y = ucl::GetGlobalId<2>();
  if (DST_X >= args.dst.Width() || DST_S >= args.dst.Slices() / 4 || DST_Y != 0) return;
  int SRC_X = DST_X;
  Type w0 = ucl::Init<Type>(0.0f);
  Type w1 = ucl::Init<Type>(0.0f);
  Type w2 = ucl::Init<Type>(0.0f);
  Type w3 = ucl::Init<Type>(0.0f);)";

  if (lookup) {
    custom_op.AddSrcTensor("lookup", lookup->tensor_desc);
    absl::StrAppend(&custom_op.code_, R"(
  int index = args.lookup.Read<int>(SRC_X, 0, 0).x;
  index = max(index, 0);
  SRC_X = index;
)");
  }

  absl::StrAppend(&custom_op.code_, R"(
  SType weight_scale;
  args.scale.ReadPerChannel<SType>(weight_scale, SRC_X);
)");

  if (SizeInBitsOf(data_type) == 8) {
    absl::StrAppend(&custom_op.code_, R"(
  uint4 w = args.weights.Read(SRC_X * args.dst.Slices() / 4 + DST_S);
  w = w ^ ucl::Init<uint4>(0x80808080);
  ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);
  SType bias = -weight_scale * 128.0f;)");
  } else if (SizeInBitsOf(data_type) == 4) {
    absl::StrAppend(&custom_op.code_, R"(
  uint2 w = args.weights.Read(SRC_X * args.dst.Slices() / 4 + DST_S);
  w = w ^ ucl::Init<uint2>(0x88888888);
  ucl::U32x2ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);
  SType bias = -weight_scale * 8.0f;)");
  } else {
    absl::StrAppend(&custom_op.code_, R"(
  uint w = args.weights.Read(SRC_X * args.dst.Slices() / 4 + DST_S);
  w = w ^ ucl::Init<uint>(0xAAAAAAAA);
  ucl::U32x1ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);
  SType bias = -weight_scale * 2.0f;)");
  }

  absl::StrAppend(&custom_op.code_, R"(
  w0 = w0 * weight_scale + bias;
  w1 = w1 * weight_scale + bias;
  w2 = w2 * weight_scale + bias;
  w3 = w3 * weight_scale + bias;
  args.dst.Write(w0, DST_X, 0, DST_S * 4 + 0);
  args.dst.Write(w1, DST_X, 0, DST_S * 4 + 1);
  args.dst.Write(w2, DST_X, 0, DST_S * 4 + 2);
  args.dst.Write(w3, DST_X, 0, DST_S * 4 + 3);
})");

  absl::StrReplaceAll(
      {{"SType", ToUclDataType(dst.tensor_desc.GetDataType(), 1)},
       {"Type", ToUclDataType(dst.tensor_desc.GetDataType(), 4)}},
      &custom_op.code_);
  std::vector<GpuModelBuilder::TensorHandle> srcs = {weights, scales};
  if (lookup) {
    srcs.push_back(*lookup);
  }
  model_builder_.AddGpuOperation(
      srcs, {dst},
      std::make_unique<DequantizeEmbeddingOp>(std::move(custom_op)),
      "dequantize_embedding");
  return dst;
}

GpuModelBuilder::TensorHandle TransformerBuilder::AddExternalTensor(
    GpuSpatialTensor* tensor) {
  GpuModelBuilder::TensorHandle handle =
      model_builder_.AddTensor(tensor->GetDescriptor());
  external_immutable_tensors_[handle.id] = tensor;
  return handle;
}

GpuModelBuilder::TensorHandle TransformerBuilder::AddExternalMutableTensor(
    const TensorDescriptor& desc) {
  GpuModelBuilder::TensorHandle handle = model_builder_.AddTensor(desc);
  external_mutable_tensors_[handle.id] = desc;
  return handle;
}

WeightsDescription TransformerBuilder::GetKWeightsDescription(
    const TransformerConfig& config, int kv_cache_batch_size, int cache_size,
    DataType data_type, bool is_global) {
  WeightsDescription weights_desc;
  weights_desc.type = data_type;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
  weights_desc.output_group_size = DivideRoundUp(
      GetKWeightsShape(config, kv_cache_batch_size, cache_size, is_global).o,
      4);
  return weights_desc;
}

OHWI TransformerBuilder::GetKWeightsShape(const TransformerConfig& config,
                                          int kv_cache_batch_size,
                                          int cache_size, bool is_global) {
  return OHWI(cache_size,
              kv_cache_batch_size * GetNumKvHeads(config, is_global), 1,
              GetHeadSize(config, is_global));
}

WeightsDescription TransformerBuilder::GetVWeightsDescription(
    const TransformerConfig& config, int kv_cache_batch_size, int cache_size,
    DataType data_type, bool is_global) {
  WeightsDescription weights_desc;
  weights_desc.type = data_type;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = DivideRoundUp(
      GetVWeightsShape(config, kv_cache_batch_size, cache_size, is_global).o,
      4);
  return weights_desc;
}

OHWI TransformerBuilder::GetVWeightsShape(const TransformerConfig& config,
                                          int kv_cache_batch_size,
                                          int cache_size, bool is_global) {
  return OHWI(GetHeadSize(config, is_global),
              kv_cache_batch_size * GetNumKvHeads(config, is_global), 1,
              cache_size);
}

GpuModelBuilder::TensorHandle TransformerBuilder::CapInfinity(
    const GpuModelBuilder::TensorHandle& src) {
  if (src.tensor_desc.GetDataType() == DataType::kFloat16) {
    auto t = model_builder_.Elementwise(src, {.param = -kMaxHalf},
                                        OperationType::kMaximum);
    return model_builder_.Elementwise(t, {.param = kMaxHalf},
                                      OperationType::kMinimum);
  }
  return src;
}

std::string TransformerBuilder::GetConvId(
    const GpuModelBuilder::TensorHandle& src, const OHWI& weights_shape,
    const WeightsDescription& weights_desc) {
  return absl::StrCat("Conv<", src.ShapeToString(), ":", ToString(data_type_),
                      ":", ToString(weights_shape), ":", weights_desc.type,
                      ">");
}

void TransformerBuilder::CleanState() {
  external_immutable_tensors_.clear();
  external_mutable_tensors_.clear();
  lora_tensors_.clear();
  tensor_reload_fns_.clear();
  weights_output_multipliers_.clear();
  debug_tensors_.clear();
}

absl::StatusOr<std::vector<odml::infra::gpu::EmbeddingRaw>>
GetDynamicPleEmbeddings(const TransformerConfig& config,
                        odml::infra::gpu::LlmTensorLoader* tensor_loader) {
  std::vector<odml::infra::gpu::EmbeddingRaw> ple_embeddings;
  if (!config.load_per_layer_embedding_dynamically) {
    return ple_embeddings;
  }
  for (int i = 0; i < config.stack_size - config.extra_stack_size; ++i) {
    odml::infra::gpu::EmbeddingRaw ple_embedding;
    const std::string w_name =
        GetPerLayerEmbeddingNameInternal(SkipMatformIndex(i, config)) + ".w";
    if (!tensor_loader->HasTensor(w_name)) {
      continue;
    }
    ple_embedding.data_type = tensor_loader->GetDataTypeForWeights(
        w_name, config.vocabulary_size * *config.per_layer_embedding_dimension);
    ASSIGN_OR_RETURN(ple_embedding.weights,
                     tensor_loader->LoadRawTensorData(w_name));
    ASSIGN_OR_RETURN(ple_embedding.scales, tensor_loader->LoadRawTensorData(
                                               w_name + "_quantized_scale"));
    ple_embeddings.push_back(std::move(ple_embedding));
  }
  return ple_embeddings;
}

absl::StatusOr<std::vector<odml::infra::gpu::EmbeddingRaw>>
GetDynamicNgrammerEmbeddings(const TransformerConfig& config,
                             odml::infra::gpu::LlmTensorLoader* tensor_loader) {
  std::vector<odml::infra::gpu::EmbeddingRaw> ngrammer_embeddings;
  if (!config.load_ngrammer_embedding_dynamically) {
    return ngrammer_embeddings;
  }
  for (int i = 0; i < config.number_of_heads; ++i) {
    odml::infra::gpu::EmbeddingRaw ngrammer_embedding;
    const std::string w_name = GetNgrammerEmbeddingNameInternal(i) + ".w";
    ngrammer_embedding.data_type = tensor_loader->GetDataTypeForWeights(
        w_name, *config.ngram_vocabulary_size * config.model_dimension /
                    config.number_of_heads);
    ASSIGN_OR_RETURN(ngrammer_embedding.weights,
                     tensor_loader->LoadRawTensorData(w_name));
    ASSIGN_OR_RETURN(
        ngrammer_embedding.scales,
        tensor_loader->LoadRawTensorData(w_name + "_quantized_scale"));
    ngrammer_embeddings.push_back(std::move(ngrammer_embedding));
  }
  return ngrammer_embeddings;
}

std::unique_ptr<GPUOperation> CreateCopyCacheOp(const TensorDescriptor& cache_k,
                                                const TensorDescriptor& cache_v,
                                                const TransformerConfig& config,
                                                int kv_cache_batch_size,
                                                bool is_global) {
  // This logic is very similar to what is in CleanCacheOp().
  std::string code = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (Y >= args.batch_size || S >= args.head_size) {
    return;
  }
  int token_index = X;
  if (token_index >= args.src_size || token_index >= args.dst_size) {
    return;
  }

  {
    // cache_k - kOSpatialIOGroupO4I4 layout, O - cache_size, I - head_size
    // OGroup = o_slices -> actual layout SpatialIOGroupO4I4
    // SpatialIOGroupO4I4 Spatial ISlice OSlice O4 I4
    // index of vec4(I4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + O4Index
    int k_o = token_index;
    int k_o_slice = k_o / 4;
    int k_o4_index = k_o % 4;
    int k_i = S * 4;
    int k_i_slice = k_i / 4;
    int k_sp = Y;
    int i_slices = (args.head_size + 3) / 4;
    int src_o_slices = (args.src_size + 3) / 4;
    int src_k_index = ((k_sp * i_slices + k_i_slice) * src_o_slices + k_o_slice) * 4 + k_o4_index;
    int dst_o_slices = (args.dst_size + 3) / 4;
    int dst_k_index = ((k_sp * i_slices + k_i_slice) * dst_o_slices + k_o_slice) * 4 + k_o4_index;
    args.src_k::type result = args.src_k.Read(src_k_index);
    args.cache_k.WriteLinear(result, dst_k_index);
  }
  {
    // cache_v - kOSpatialIOGroupI4O4 layout, O - head_size, I - cache_size
    // OGroup = o_slices -> actual layout SpatialIOGroupI4O4
    // SpatialIOI4O4 Spatial ISlice OSlice I4 O4
    // index of vec4(O4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + I4Index
    int v_o = S * 4;
    int v_o_slice = v_o / 4;
    int v_i = token_index;
    int v_i_slice = v_i / 4;
    int v_i4_index = v_i % 4;
    int v_sp = Y;
    int o_slices = (args.head_size + 3) / 4;
    int src_i_slices = (args.src_size + 3) / 4;
    int src_v_index = ((v_sp * src_i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    int dst_i_slices = (args.dst_size + 3) / 4;
    int dst_v_index = ((v_sp * dst_i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    args.src_v::type result = args.src_v.Read(src_v_index);
    args.cache_v.WriteLinear(result, dst_v_index);
  }
})";

  class CopyCacheOp : public GPUOperation {
   public:
    CopyCacheOp() = default;
    int3 GetGridSize() const override {
      return int3(cache_size_, batch_size_, DivideRoundUp(head_size_, 4));
    }

    // Move only
    CopyCacheOp(CopyCacheOp&& operation) = default;
    CopyCacheOp& operator=(CopyCacheOp&& operation) = default;
    CopyCacheOp(const CopyCacheOp&) = delete;
    CopyCacheOp& operator=(const CopyCacheOp&) = delete;

    int batch_size_ = 0;
    int cache_size_ = 0;
    int head_size_ = 0;
  };

  const int head_size = GetHeadSize(config, is_global);
  const int kv_heads_count = GetNumKvHeads(config, is_global);

  CopyCacheOp custom_op;
  custom_op.batch_size_ = kv_cache_batch_size * kv_heads_count;
  custom_op.cache_size_ = TransformerBuilder::GetCacheSize(config, is_global);
  custom_op.head_size_ = head_size;
  custom_op.args_.AddInt("batch_size", custom_op.batch_size_);
  custom_op.args_.AddInt("head_size", head_size);
  custom_op.args_.AddInt("src_size", 0);
  custom_op.args_.AddInt("dst_size", 0);
  custom_op.AddSrcTensor("src_k", cache_k);
  custom_op.AddSrcTensor("src_v", cache_v);
  custom_op.AddDstTensor("cache_k", cache_k);
  custom_op.AddDstTensor("cache_v", cache_v);
  custom_op.code_ = code;

  return std::make_unique<CopyCacheOp>(std::move(custom_op));
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::LoadPerLayerEmbeddings(
    const GpuModelBuilder::TensorHandle& tokens,
    std::vector<odml::infra::gpu::EmbeddingWeights>* ple_weights) {
  if (config_.load_per_layer_embedding_dynamically) {
    std::vector<GpuModelBuilder::TensorHandle> per_layer_embeddings;
    // Per layer embeddings managed externally.
    BHWC ple_shape = BHWC(1, 1, config_.sequence_size,
                          *config_.per_layer_embedding_dimension);
    TensorDescriptor ple_desc = GetEmbeddingDesc(
        ple_shape, config_.vocabulary_size, GetPerLayerEmbeddingName(0));

    TensorDescriptor ple_scale_desc(
        DataType::kFloat32, TensorStorageType::kBuffer, Layout::kLinear);
    ple_scale_desc.SetBHWCShape(BHWC(1, 1, 1, config_.sequence_size));

    ple_weights->reserve(config_.stack_size);
    per_layer_embeddings.reserve(config_.stack_size);
    for (int i = 0; i < config_.stack_size - config_.extra_stack_size; ++i) {
      // EE models may not have PLE on extra layers.
      if (!tensor_loader_->HasTensor(
              GetPerLayerEmbeddingName(SkipMatformIndex(i, config_)) + ".w")) {
        break;
      }
      ple_weights->push_back({});
      auto& ple_weight = (*ple_weights)[i];
      ple_weight.weights = model_builder_.AddTensor(ple_desc);
      ple_weight.scales = model_builder_.AddTensor(ple_scale_desc);
      ple_weight.weights_data.resize(ple_desc.GetMemorySizeInBytes());
      ple_weight.scales_data.resize(ple_scale_desc.GetMemorySizeInBytes() /
                                    SizeOf(DataType::kFloat32));
      ASSIGN_OR_RETURN(auto ple_embedding,
                       DequantizeEmbedding(ple_shape, ple_weight.weights,
                                           ple_weight.scales));
      if (gpu_info_.IsAMD()) {
        // TODO: b/427791108 - Some AMD GPUs failed to compile the
        // dequantize_embedding shader when fused with elementwise ops. This
        // Transpose() is not needed but prevents the ops from being fused.
        ple_embedding =
            model_builder_.Transpose(ple_embedding, BHWC(0, 1, 2, 3));
      }
      per_layer_embeddings.push_back(ple_embedding);
    }
    return per_layer_embeddings;
  } else {
    return GetPerLayerEmbeddings(tokens);
  }
}

absl::StatusOr<GpuModel> TransformerBuilder::CreateStackedTransformerModelV3(
    int sequence_size, int batch_size,
    GpuModelBuilder::TensorHandle* current_tokens,
    GpuModelBuilder::TensorHandle* media_handle,
    GpuModelBuilder::TensorHandle* tokens_ids_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
    std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
    GpuModelBuilder::TensorHandle* output_handle,
    std::vector<odml::infra::gpu::EmbeddingWeights>* ple_weights,
    std::vector<GpuModelBuilder::TensorHandle>* vision_ple_tensors,
    GpuModelBuilder::TensorHandle* segment_pos_tensor,
    GpuModelBuilder::TensorHandle* global_mask_tensor,
    GpuModelBuilder::TensorHandle* local_mask_tensor,
    GpuModelBuilder::TensorHandle* external_embeddings_handle,
    std::vector<GpuModelBuilder::TensorHandle>* external_ple_handles,
    GpuModelBuilder::TensorHandle* input_activations_handle,
    GpuModelBuilder::TensorHandle* output_activations_handle,
    const std::string& base_name) {
  RET_CHECK(UseGemmaModelV3NPlus());

  if (config_.enable_external_embeddings) {
    RET_CHECK(external_embeddings_handle != nullptr)
        << "external_embeddings_handle must be provided when "
        << "enable_external_embeddings is true.";
    RET_CHECK(external_ple_handles != nullptr)
        << "external_ple_handles must be provided when "
        << "enable_external_embeddings is true.";
    RET_CHECK(external_ple_handles->empty())
        << "external_ple_handles must be empty (to be filled) when "
        << "enable_external_embeddings is true.";
    RET_CHECK(ple_weights == nullptr) << "ple_weights must be provided when "
                                      << "enable_external_embeddings is true.";
    RET_CHECK(vision_ple_tensors == nullptr)
        << "vision_ple_tensors must be provided when "
        << "enable_external_embeddings is true.";
    RET_CHECK(current_tokens == nullptr)
        << "current_tokens must be nullptr when "
        << "enable_external_embeddings is true.";
    RET_CHECK(tokens_ids_handle == nullptr)
        << "tokens_ids_handle must be nullptr when "
        << "enable_external_embeddings is true.";
    RET_CHECK(media_handle == nullptr) << "media_handle must be nullptr when "
                                       << "enable_external_embeddings is true.";
  }

  // TODO: b/505555017 - Remove error once multi-output works properly with PLE.
  RET_CHECK(batch_size == 1 ||
            (!config_.per_layer_embedding_dimension.has_value()) ||
            *config_.per_layer_embedding_dimension == 0)
      << "Multi-output is not yet implemented for models with PLE.";

  CleanState();
  config_.sequence_size = sequence_size;
  config_.batch_size = batch_size;

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);

  bool is_media_model = media_handle != nullptr && media_cache_size_ > 0;
  if (is_media_model) {
    *media_handle =
        model_builder_.AddTensor(GetMediaEmbeddingsTensorDescriptor());
  }
  // The runtime check for the params tensor handle.
  *params_i32_handle = model_builder_.AddTensor(GetParamsTensorDescriptor());

  // Pick tokens or embedding input based on the config - whether we are using
  // external embeddings or not.
  GpuModelBuilder::TensorHandle conded_current_tokens;
  GpuModelBuilder::TensorHandle embeddings;
  if (config_.enable_external_embeddings) {
    BHWC shape = BHWC(batch_size, 1, sequence_size, config_.model_dimension);
    *external_embeddings_handle = model_builder_.AddTensor(shape, data_type_);
    embeddings = *external_embeddings_handle;
    embeddings = model_builder_.Multiplication(
        embeddings, std::sqrt(config_.model_dimension));
  } else {
    *tokens_ids_handle = model_builder_.AddTensor(GetTokensTensorDescriptor());
    // If the handle is provided, create the input tensor
    *current_tokens = model_builder_.AddTensor(
        GetInputTensorDescriptor(batch_size, sequence_size));
    WriteCurrentTokens(*current_tokens, *tokens_ids_handle, *params_i32_handle);
    conded_current_tokens = *current_tokens;

    // Declare multimodal tensors which will be used after embedding
    GpuModelBuilder::TensorHandle cond_hard_tokens;
    GpuModelBuilder::TensorHandle hard_tokens;
    GpuModelBuilder::TensorHandle cond_text;
    GpuModelBuilder::TensorHandle gathered_media_embeddings;
    std::vector<GpuModelBuilder::TensorHandle> gathered_vision_ple;
    bool can_process_audio_hard_tokens;
    if (is_media_model) {  // Multimodal
      conded_current_tokens = *current_tokens;
      // Get conditional tensors to split on
      auto cond_text_ge = model_builder_.Elementwise(
          *current_tokens, {0}, OperationType::kGreaterEqual);
      auto cond_text_lt = model_builder_.Elementwise(
          *current_tokens, {config_.vocabulary_size}, OperationType::kLess);
      cond_text = model_builder_.Elementwise(cond_text_ge, cond_text_lt,
                                             OperationType::kLogicalAnd);
      cond_text = model_builder_.Cast(cond_text, DataType::kInt32);
      // Get zero tensor handle of size current_tokens
      GpuModelBuilder::TensorHandle zero =
          model_builder_.Elementwise(*current_tokens, {0}, OperationType::kMul);
      // Get media indices and gather audio embeddings
      GpuModelBuilder::TensorHandle media_index =
          model_builder_.SelectV2(cond_text, zero, *current_tokens);
      media_index =
          model_builder_.Elementwise(media_index, {-1}, OperationType::kMul);
      media_index = model_builder_.Reshape(
          media_index, BHWC(1, 1, 1, media_index.tensor_desc.GetBHWCShape().w));
      ASSIGN_OR_RETURN(gathered_media_embeddings,
                       GatherMedia(*media_handle, media_index));

      // Process audio hard token embeddings
      can_process_audio_hard_tokens = tensor_loader_->HasTensor(
                                          "params.audio.transformer.embedder."
                                          "audio_input_embedding_extra.w") &&
                                      config_.audio_vocab_extra_dim > 0 &&
                                      config_.audio_input_embedding_dim > 0 &&
                                      config_.audio_vocab_offset > 0;

      if (can_process_audio_hard_tokens) {
        // Find audio hard token cond tensor
        cond_hard_tokens = model_builder_.Elementwise(
            *current_tokens, {config_.audio_vocab_offset},
            OperationType::kGreaterEqual);
        cond_hard_tokens =
            model_builder_.Cast(cond_hard_tokens, DataType::kInt32);
        // Grab hard tokens from current tokens.
        // Note that we are only using the audio input embedding extra table.
        // As such, in addition to adding the audio vocab offset, we also need
        // to add the length of the audio input embedding table, which is 256000
        // Note that this calculation assumes that the number of audio hard
        // tokens, represented by the length of audio_input_embedding_table, is
        // equal to config_.vocabulary_size.
        hard_tokens = model_builder_.Elementwise(
            *current_tokens, {config_.audio_vocab_offset}, OperationType::kSub);
        hard_tokens =
            model_builder_.SelectV2(cond_hard_tokens, hard_tokens, zero);
        // Adjust cond_text to ignore tokens in the audio hard tokens range.
        cond_text = model_builder_.SelectV2(cond_hard_tokens, zero, cond_text);
      }

      // Get the current tokens after applying cond text
      conded_current_tokens =
          model_builder_.SelectV2(cond_text, *current_tokens, zero);
    }
    // We don't use base_name here because main model embedder should be reused
    // for MTP stacks as well.
    ASSIGN_OR_RETURN(auto embeddings_full,
                     LoadWeights("transformer.embedder.input_embedding",
                                 OHWI(config_.vocabulary_size, 1, 1,
                                      config_.model_dimension),
                                 /*swap_dims=*/false));
    embeddings = MakeEmbLookUpOp(conded_current_tokens, embeddings_full);
    embeddings = model_builder_.Multiplication(
        embeddings, std::sqrt(config_.model_dimension));

    // Add media embeddings in
    if (is_media_model) {
      embeddings = model_builder_.SelectV2(cond_text, embeddings,
                                           gathered_media_embeddings);
      // Process audio hard tokens
      if (can_process_audio_hard_tokens) {
        ASSIGN_OR_RETURN(
            auto audio_input_embedding_extra,
            LoadWeights(
                "params.audio.transformer.embedder.audio_input_embedding_extra",
                OHWI(config_.audio_vocab_extra_dim, 1, 1,
                     config_.audio_input_embedding_dim),
                /*swap_dims=*/false));
        auto emb_hard_tokens =
            MakeEmbLookUpOp(hard_tokens, audio_input_embedding_extra);
        emb_hard_tokens = model_builder_.Multiplication(
            emb_hard_tokens, std::sqrt(config_.audio_input_embedding_dim));
        emb_hard_tokens = MakeRMSNormalization(
            emb_hard_tokens,
            "params.audio.transformer.embedder.audio_embedding_norm",
            /*scale_shift=*/0.0f);
        ASSIGN_OR_RETURN(
            emb_hard_tokens,
            MakeLinearAutoPrecision(
                emb_hard_tokens,
                "params.audio.transformer.embedder.audio_input_projection",
                config_.model_dimension));
        emb_hard_tokens =
            model_builder_.RMSNormalization(emb_hard_tokens, /*epsilon=*/1e-6f);

        // Add in audio hard tokens
        embeddings = model_builder_.SelectV2(cond_hard_tokens, emb_hard_tokens,
                                             embeddings);
      }
    }
  }

  if (config_.combine_activations_with_input) {
    RET_CHECK(input_activations_handle)
        << "Input activations required for combine_activations_with_input";
    // Concat, then down-project
    // TODO: Double-check that it's really embeddings-->projected_activations
    *input_activations_handle = model_builder_.AddTensor(
        embeddings.tensor_desc.GetBHWCShape(), data_type_);
    auto concat_input = model_builder_.Concat(
        embeddings, *input_activations_handle, Axis::kChannels);
    ASSIGN_OR_RETURN(embeddings, MakeConvAutoPrecision(
                                     concat_input, base_name + ".pre_proj",
                                     Layout::kOIHW, config_.model_dimension));
  }

  ASSIGN_OR_RETURN(const int mask_width,
                   GetMaskWidth(config_.attention_mask_type));
  const BHWC mask_shape = BHWC(1, 1, mask_width, config_.cache_size);
  if (config_.enable_decode_logits) {
    RET_CHECK(segment_pos_tensor != nullptr);
    RET_CHECK(global_mask_tensor != nullptr);
    RET_CHECK(local_mask_tensor != nullptr);
    *segment_pos_tensor = model_builder_.AddTensor(1, 1, config_.sequence_size,
                                                   1, DataType::kInt32);
    *global_mask_tensor = model_builder_.AddTensor(mask_shape, DataType::kBool);
    *local_mask_tensor = model_builder_.AddTensor(mask_shape, DataType::kBool);
  }
  auto segment_pos = config_.enable_decode_logits
                         ? *segment_pos_tensor
                         : MakeFillSegmentPos(*params_i32_handle);
  ASSIGN_OR_RETURN(auto global_attention_mask,
                   config_.enable_decode_logits
                       ? *global_mask_tensor
                       : MakeFillAttention(*params_i32_handle, current_tokens));
  ASSIGN_OR_RETURN(
      auto local_attention_mask,
      GetLocalMask(params_i32_handle, local_mask_tensor, current_tokens));

  CreateCache(key_cached_tensors, value_cached_tensors);

  std::vector<GpuModelBuilder::TensorHandle> per_layer_embeddings;
  if (config_.per_layer_embedding_dimension.has_value()) {
    // Projected per layer inputs would be fusioned into PLE embeddings.
    ASSIGN_OR_RETURN(
        std::vector<GpuModelBuilder::TensorHandle> projected_per_layer_inputs,
        MakeProjectedPleInputsV3(
            embeddings, base_name + ".embedder.per_layer_model_projection",
            base_name + ".embedder.per_layer_projection_norm"));

    if (config_.enable_external_embeddings) {
      external_ple_handles->resize(projected_per_layer_inputs.size());
      per_layer_embeddings.reserve(projected_per_layer_inputs.size());
      BHWC ple_shape = BHWC(batch_size, 1, sequence_size,
                            *config_.per_layer_embedding_dimension);
      for (int i = 0; i < projected_per_layer_inputs.size(); ++i) {
        (*external_ple_handles)[i] =
            model_builder_.AddTensor(ple_shape, data_type_);
        per_layer_embeddings.push_back((*external_ple_handles)[i]);
      }
    } else {
      ASSIGN_OR_RETURN(
          per_layer_embeddings,
          LoadPerLayerEmbeddings(conded_current_tokens, ple_weights));
    }

    RET_CHECK_EQ(per_layer_embeddings.size(),
                 projected_per_layer_inputs.size());
    const float fusion_scale = 1 / std::sqrt(2);
    for (int i = 0; i < per_layer_embeddings.size(); ++i) {
      // Scale embeddings = True for V3 models.
      per_layer_embeddings[i] = model_builder_.Multiplication(
          per_layer_embeddings[i],
          std::sqrt(*config_.per_layer_embedding_dimension));
      per_layer_embeddings[i] = model_builder_.Multiplication(
          model_builder_.Add(per_layer_embeddings[i],
                             projected_per_layer_inputs[i]),
          fusion_scale);
    }
  }

  std::vector<GpuModelBuilder::TensorHandle> activations;
  if (config_.altup_config.has_value()) {
    ASSIGN_OR_RETURN(activations, PreprocessAltupV3(embeddings, base_name));
  } else {
    activations.push_back(embeddings);
  }

  std::vector<GpuModelBuilder::OptionalNodeContext>
      optional_shared_kv_cache_contexts;
  const auto& layer_names = GetLayerNames(base_name);
  for (int i = 0; i < config_.stack_size; ++i) {
    const std::string layer_name = layer_names[i];
    GpuModelBuilder::OptionalNodeContext context;

    if (config_.use_optional_transformer_layers) {
      // TODO: b/413684361 - Support vector of inputs in BeginOptionalNodes.
      // context = model_builder_.BeginOptionalNodes(
      //     i | kTransformerLayerOptionTag, activations);
      return absl::InvalidArgumentError(
          "Optional transformer layers are not supported for V3");
    }

    const bool is_global = IsGlobalLayer(config_, i);
    const GpuModelBuilder::TensorHandle* active_mask =
        is_global ? &global_attention_mask : &local_attention_mask;
    int cache_index = i;
    std::string kv_layer_name = layer_name;
    bool has_shared_kv = config_.kv_cache_shared_start_layer.has_value();
    int kv_cache_shared_start_layer =
        has_shared_kv ? *config_.kv_cache_shared_start_layer : 0;
    if (has_shared_kv && i >= kv_cache_shared_start_layer) {
      if (is_global) {
        cache_index = kv_cache_shared_start_layer - 1;
      } else {
        cache_index = kv_cache_shared_start_layer - 2;
      }
      // TODO: This hacky workaround just changes our indices for MTP models
      // to be 0 and 1 -- we still need to actually share the KV cache!
      if (cache_index < 0) cache_index += 2;
      kv_layer_name = layer_names[cache_index];
    }
    if (has_shared_kv && i == kv_cache_shared_start_layer) {
      // Layers after this are optional when processing input.
      for (const auto& activation : activations) {
        optional_shared_kv_cache_contexts.push_back(
            model_builder_.BeginOptionalNodes(kSharedKvCacheLayersOptionTag,
                                              activation));
      }
    }
    ASSIGN_OR_RETURN(
        std::vector<GpuModelBuilder::TensorHandle> new_activations,
        MakeLayerV3(activations, *active_mask, segment_pos,
                    key_cached_tensors->at(cache_index),
                    value_cached_tensors->at(cache_index),
                    per_layer_embeddings.size() > i ? &per_layer_embeddings[i]
                                                    : nullptr,
                    *params_i32_handle, is_global, layer_name,
                    /*layer_num=*/i, kv_layer_name));

    if (config_.use_optional_transformer_layers) {
      // TODO: b/413684361 - Support vector of inputs in EndOptionalNodes.
      // RETURN_IF_ERROR(
      //     model_builder_.EndOptionalNodes(std::move(context),
      //     new_activations));
      return absl::InvalidArgumentError(
          "Optional transformer layers are not supported for V3");
    } else {
      activations = std::move(new_activations);
    }
  }

  for (int i = 0; i < optional_shared_kv_cache_contexts.size(); ++i) {
    RETURN_IF_ERROR(model_builder_.EndOptionalNodes(
        std::move(optional_shared_kv_cache_contexts[i]), activations[i]));
  }

  GpuModelBuilder::TensorHandle output_embedding;
  if (config_.altup_config.has_value()) {
    ASSIGN_OR_RETURN(output_embedding, CreatePostProcessEmbeddingsWithAltupV3(
                                           activations, base_name));
  } else {
    output_embedding = activations[0];
  }

  // For V3, final norm scale shift is 0.0f. // V4 has shift.
  if (!config_.selectable_final_ln) {
    if (config_.extra_stack_size > 0) {
      output_embedding =
          MakeRMSNormalization(output_embedding, "extra.transformer.final_norm",
                               config_.rms_norm_scale_shift);
    } else {
      output_embedding =
          MakeRMSNormalization(output_embedding, base_name + ".final_norm",
                               config_.rms_norm_scale_shift);
    }
  } else {
    for (int i = 0; i < *config_.selectable_final_ln; ++i) {
      auto context = model_builder_.BeginOptionalNodes(i | kFinalLnOptionTag,
                                                       output_embedding);
      auto final_ln = MakeRMSNormalization(
          output_embedding, "params.lm." + std::to_string(i) + ".final_ln",
          config_.rms_norm_scale_shift);
      RETURN_IF_ERROR(
          model_builder_.EndOptionalNodes(std::move(context), final_ln));
    }
  }

  std::vector<GpuModelBuilder::ValueId> input_ids;
  if (current_tokens != nullptr) {
    input_ids.push_back(current_tokens->id);
    input_ids.push_back(tokens_ids_handle->id);
  }
  input_ids.push_back(params_i32_handle->id);

  if (config_.enable_external_embeddings) {
    RET_CHECK(external_embeddings_handle != nullptr);
    input_ids.push_back(external_embeddings_handle->id);
  }

  if (is_media_model) {
    input_ids.push_back(media_handle->id);
  }

  *output_handle = output_embedding;
  std::vector<GpuModelBuilder::ValueId> output_ids = {output_handle->id};

  // If we ask for final activations, then apply post-project if one exists, and
  // return as additional output.
  if (output_activations_handle) {
    auto final_activations = output_embedding;
    if (config_.post_proj_size.has_value()) {
      ASSIGN_OR_RETURN(
          final_activations,
          MakeConvAutoPrecision(final_activations, base_name + ".post_proj",
                                Layout::kOIHW, *config_.post_proj_size));
    }
    *output_activations_handle = final_activations;
    output_ids.push_back(output_activations_handle->id);
  }

  if (current_tokens != nullptr) {
    output_ids.push_back(current_tokens->id);
    output_ids.push_back(tokens_ids_handle->id);
  }

  for (int i = 0; i < key_cached_tensors->size(); ++i) {
    input_ids.push_back(key_cached_tensors->at(i).id);
    output_ids.push_back(key_cached_tensors->at(i).id);
  }
  for (int i = 0; i < value_cached_tensors->size(); ++i) {
    input_ids.push_back(value_cached_tensors->at(i).id);
    output_ids.push_back(value_cached_tensors->at(i).id);
  }

  if (config_.enable_external_embeddings) {
    for (const auto& ple_handle : *external_ple_handles) {
      input_ids.push_back(ple_handle.id);
    }
  } else {
    if (ple_weights) {
      // Per layer embeddings managed externally.
      for (const auto& ple : *ple_weights) {
        input_ids.push_back(ple.weights.id);
        input_ids.push_back(ple.scales.id);
      }
    }
  }

  if (segment_pos_tensor) {
    input_ids.push_back(segment_pos_tensor->id);
  }
  if (global_mask_tensor) {
    input_ids.push_back(global_mask_tensor->id);
  }
  if (local_mask_tensor) {
    input_ids.push_back(local_mask_tensor->id);
  }

  // Preserve debugging tensors.
  for (const auto& it : debug_tensors_) {
    output_ids.push_back(it.second.id);
  }

  GpuModel gpu_model;
  RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::MakeLayerV3(
    const std::vector<GpuModelBuilder::TensorHandle>& inputs,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle* per_layer_embedding,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name, int layer_num, const std::string& kv_layer_name) {
  GpuModelBuilder::TensorHandle activated_input;
  // V4 does not have altup.
  std::optional<std::vector<GpuModelBuilder::TensorHandle>> altup_predictions;
  if (config_.altup_config.has_value()) {
    ASSIGN_OR_RETURN(altup_predictions, CreateAltupPredictV3(inputs, name));
    activated_input =
        altup_predictions.value()[config_.altup_config->altup_active_idx];
  } else {
    // Somewhat awkward, but the inputs vector just contains a single tensor
    // when altup is not used.
    activated_input = inputs[0];
  }

  // V3 Pre-attention normalization has no scale shift. // V4 has shift
  auto inputs_normalized =
      MakeRMSNormalization(activated_input, name + ".pre_attention_norm",
                           config_.rms_norm_scale_shift);
  const bool shared_kv_cache =
      config_.kv_cache_shared_start_layer.has_value() &&
      layer_num >= *config_.kv_cache_shared_start_layer;
  ASSIGN_OR_RETURN(
      auto atten_output,
      MakeSelfAttentionV3(inputs_normalized, attention_mask, segment_pos,
                          key_tensor, value_tensor, params_tensor, is_global,
                          name + ".attn", shared_kv_cache,
                          kv_layer_name + ".attn"));

  // V3 Post attention norm has learnable scale and no scale shift.
  // V4 has shift.
  atten_output =
      MakeRMSNormalization(atten_output, name + ".post_attention_norm",
                           config_.rms_norm_scale_shift);

  // For v3, use_gated_residual is False, so it falls back to the simple
  // residual connection.
  atten_output = model_builder_.Add(atten_output, activated_input);

  if (config_.laurel_rank > 0) {
    ASSIGN_OR_RETURN(atten_output,
                     MakeLaurelLayerV3(atten_output, inputs_normalized, name));
  }

  GpuModelBuilder::TensorHandle output;
  if (config_.moe_config.has_value()) {
    ASSIGN_OR_RETURN(output, MakeFFLayerMoe(atten_output, name));
  } else {
    bool is_kv_shared_layer = config_.kv_cache_shared_start_layer.has_value() &&
                              layer_num >= *config_.kv_cache_shared_start_layer;
    ASSIGN_OR_RETURN(
        output, MakeFFLayerV3(
                    atten_output, name,
                    /*use_statistical_top_k=*/
                    config_.activation_sparsity_stddev_multiplier.has_value() &&
                        layer_num < kNumActivationSparsityLayers,
                    is_global, is_kv_shared_layer));
  }

  if (config_.altup_config.has_value()) {
    ASSIGN_OR_RETURN(
        std::vector<GpuModelBuilder::TensorHandle> altup_correct_outputs,
        CreateAltupCorrectV3(altup_predictions.value(), output, name));
    RET_CHECK_EQ(altup_correct_outputs.size(),
                 config_.altup_config->altup_num_inputs);
    if (per_layer_embedding) {
      // Load altup corrected scale.
      // TODO: b/413684361 - Load quantized scale.
      ml_drift::Tensor<Linear, DataType::kFloat32> gamma;
      gamma.shape =
          Linear(altup_correct_outputs[0].tensor_desc.GetBHWCShape().c);
      gamma.data = tensor_loader_->LoadFloat32(
          name + ".altup.correct_output.scale",
          altup_correct_outputs[0].tensor_desc.GetBHWCShape().c);

      GpuModelBuilder::TensorHandle gating_input =
          model_builder_.Multiplication(altup_correct_outputs[0], gamma);

      ASSIGN_OR_RETURN(
          auto ple_output,
          MakePleLayerV3(gating_input, *per_layer_embedding, name));

      std::vector<GpuModelBuilder::TensorHandle> altup_outputs(
          config_.altup_config->altup_num_inputs);
      altup_outputs[0] = altup_correct_outputs[0];
      for (int i = 1; i < config_.altup_config->altup_num_inputs; ++i) {
        // For v3, use_gated_residual is False, so it falls back to the simple
        // residual connection.
        altup_outputs[i] =
            model_builder_.Add(ple_output, altup_correct_outputs[i]);
      }

      return altup_outputs;
    }
    return altup_correct_outputs;
  }

  if (per_layer_embedding) {
    // V3 and V4 both use simple residuals for PLE
    ASSIGN_OR_RETURN(auto ple_output,
                     MakePleLayerV3(output, *per_layer_embedding, name));
    output = model_builder_.Add(ple_output, output);
  }

  if (config_.scale_layer) {
    auto skip_scale_name = name + ".skip.scale";
    if (tensor_loader_->HasTensor(skip_scale_name)) {
      // TODO(tmullen): Allow for F16 loading too, in the event these are ever
      // compressed.
      auto scale = tensor_loader_->LoadFloat32(skip_scale_name, 1);
      if (scale.size() != 1) {
        return absl::InvalidArgumentError(skip_scale_name + " had size " +
                                          std::to_string(scale.size()) +
                                          " instead of 1.");
      }
      output = model_builder_.Multiplication(output, scale[0]);
    }
  }
  return std::vector<GpuModelBuilder::TensorHandle>{output};
}

// src - bhwc(batch_size, 1, sequence_size, model_dimension)
// Mirrors SelfAttentionV1 for now.
// TODO(tmullen): Refactor both this and the V1 version to use subgraphs for
// significant load-time improvement.
absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeSelfAttentionV3(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name, bool shared_kv_cache,
    const std::string& kv_layer_name) {
  // B - batch_size
  // S - sequence_size
  // HC - heads_count
  // HS - head_size
  // HGS - head_group_size (MHA: HGS = 1, MQA: HGS = HC)
  // KHC - HC / HGS
  // VHC - HC / HGS
  // KHC = VHC!
  // KVHC = KHC = VHC -> HC = KVHC * HGS
  auto src_sh = src.tensor_desc.GetBHWCShape();
  if (src_sh.c % config_.number_of_heads != 0) {
    return absl::InvalidArgumentError(
        "Can not build the model. Probably incorrect config.");
  }

  // If KV cache is shared, only calculate Q.
  ASSIGN_OR_RETURN(auto qkv_arr, shared_kv_cache
                                     ? MakeGetQ(src, name, is_global)
                                     : MakeGetQKV(src, name, is_global));
  auto q = qkv_arr[0];  // bhwc(B, 1, S, HC*HS)
  auto k = qkv_arr[1];  // bhwc(B, 1, S, KVHC*HS)
  auto v = qkv_arr[2];  // bhwc(B, 1, S, KVHC*HS)

  // The subgraph has to be split into 2 parts because the MakeFakeQuant() ops
  // in the middle are different per layer.
  // For Gemma3N, we need qk norm weights, so we pass in the corresponding
  // prefix in that case.
  const std::string qk_norm_prefix = config_.qk_norm_has_scale ? name : "";
  ASSIGN_OR_RETURN(
      auto res,
      MakeSelfAttentionV3SubgraphPreKvCacheWrite(
          absl::StrCat("MakeSelfAttentionV3PreKvCacheWrite:", is_global, ":",
                       shared_kv_cache),
          q, k, v, segment_pos, is_global, shared_kv_cache, qk_norm_prefix));
  q = res[0];
  // Update if we are not using shared kv cache.
  if (!shared_kv_cache) {
    k = res[1];
    v = res[2];

    if (config_.apply_srq) {
      if (config_.int8_kv_cache) {
        DataType scale_type = DataType::kUint8;
        if (UseGemmaModelV4() && is_global) {
          // In V4, global layers use 4-bit quantization. Even when we have
          // INT8 data types, we'll still want to respect the 4-bit SRQ ranges.
          scale_type = DataType::kUint4;
        }
        k = MakeQuantInt8(k, name + ".k_cache_static_quantized_scale",
                          scale_type);
        v = MakeQuantInt8(v, name + ".v_cache_static_quantized_scale",
                          scale_type);
      } else {
        DataType scale_type = DataType::kUint8;
        if (UseGemmaModelV4() && is_global) {
          // In V4, global layers use 4-bit quantization.
          scale_type = DataType::kUint4;
        }
        k = MakeFakeQuant(k, name + ".k_cache_static_quantized_scale",
                          scale_type);
        v = MakeFakeQuant(v, name + ".v_cache_static_quantized_scale",
                          scale_type);
      }
    } else if (config_.force_int8_kv_cache) {
      DataType scale_type = DataType::kUint8;
      k = MakeQuantInt8(k, kStaticKvQuantScale, scale_type);
      v = MakeQuantInt8(v, kStaticKvQuantScale, scale_type);
    }
    AddValuesToCache(model_builder_, k, v, params_tensor, key_tensor,
                     value_tensor, is_global);
  }

  ASSIGN_OR_RETURN(
      auto att,
      MakeSelfAttentionV3SubgraphPostKvCacheWrite(
          absl::StrCat("MakeSelfAttentionV3PostKvCacheWrite:", is_global),
          src_sh, q, attention_mask, key_tensor, value_tensor, params_tensor,
          kv_layer_name, is_global));

  // NOTE: Layout is assumed IOHW here for Nano models, rather than the OIHW of
  // previous Gemma models.
  ASSIGN_OR_RETURN(att, MakeLinearAutoPrecisionWithSrq(
                            att, name + ".attn_vec_einsum",
                            config_.model_dimension, config_.apply_srq));
  return att;
}

GpuModelBuilder::TensorHandle
TransformerBuilder::MakeRMSNormalizationWithBuilder(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const std::string& name, float scale_shift) {
  ml_drift::Tensor<Linear, DataType::kFloat32> gamma;
  gamma.shape = Linear(src.tensor_desc.GetBHWCShape().c);
  gamma.data = tensor_loader_->LoadFloat32(name + ".scale",
                                           src.tensor_desc.GetBHWCShape().c);
  if (scale_shift != 0.0f) {
    for (int i = 0; i < gamma.data.size(); ++i) {
      gamma.data[i] = scale_shift + gamma.data[i];
    }
  }
  return builder.RMSNormalization(src, /*epsilon*/ 1e-6f, &gamma);
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TransformerBuilder::MakeSelfAttentionV3SubgraphPreKvCacheWrite(
    const std::string& subgraph_id, GpuModelBuilder::TensorHandle q,
    GpuModelBuilder::TensorHandle k, GpuModelBuilder::TensorHandle v,
    GpuModelBuilder::TensorHandle segment_pos, bool is_global,
    bool shared_kv_cache, const std::string& qk_norm_prefix) {
  std::vector<GpuModelBuilder::TensorHandle> main_inputs = {q, segment_pos};
  if (!shared_kv_cache) {
    main_inputs.insert(main_inputs.end(), {k, v});
  }
  // TODO: Figure out how to make subgraphs with a switchable internal op, so
  // the rest of the structure can be reused. In the meantime, we just use
  // unique names for the qk_norm_with_shift case, so we can keep as much of the
  // code the same as possible, even though this is less efficient overall. But
  // this only affects 3N currently.
  const std::string subgraph_name =
      qk_norm_prefix.empty() ? subgraph_id : subgraph_id + ":" + qk_norm_prefix;
  if (!model_builder_.HasSubgraph(subgraph_name)) {
    GpuModelBuilder builder = model_builder_.CreateBuilder();
    std::vector<GpuModelBuilder::TensorHandle*> subgraph_input_refs = {
        &q, &segment_pos};
    if (!shared_kv_cache) {
      subgraph_input_refs.insert(subgraph_input_refs.end(), {&k, &v});
    }
    auto subgraph_inputs = CreateSubgraphInputs(builder, subgraph_input_refs);

    RETURN_IF_ERROR(QKVRoPE(builder, segment_pos, &q,
                            shared_kv_cache ? nullptr : &k,
                            shared_kv_cache ? nullptr : &v, is_global,
                            qk_norm_prefix, config_.rms_norm_scale_shift));

    // Ensure the scale type used does not need the `name` arg.
    RET_CHECK_NE(
        config_.attention_scale_type,
        odml::infra::proto::TransformerParameters::SCALE_TYPE_PER_DIM_SCALE);
    ASSIGN_OR_RETURN(q, PerDimScale(builder, q, ""));

    //  q - hwc(B*KVHC, HGS*S, HS)
    //  k - hwc(B*KVHC, S, HS)
    //  v - hwc(B*KVHC, S, HS)
    // V3 apply Value norm to values with no scaling and no scale shift. Not
    // needed when sharing KV cache.
    if (!shared_kv_cache) {
      v = builder.RMSNormalization(v, /*epsilon=*/1e-6f);
    }

    std::vector<GpuModelBuilder::TensorHandle> subgraph_outputs = {q};
    if (!shared_kv_cache) {
      subgraph_outputs.insert(subgraph_outputs.end(), {k, v});
    }
    RETURN_IF_ERROR(model_builder_.RegisterSubgraph(
        std::move(builder), subgraph_name, subgraph_inputs, subgraph_outputs));
  }
  return model_builder_.Subgraph(subgraph_name, main_inputs);
}

GpuModelBuilder::TensorHandle
TransformerBuilder::CreateKVCacheQuantizedScaleTensor(GpuModelBuilder& builder,
                                                      DataType type,
                                                      int channel_count,
                                                      float scale) {
  TensorDescriptor quantized_scale_desc(type, TensorStorageType::kBuffer,
                                        Layout::kLinear);
  quantized_scale_desc.SetBHWCShape(BHWC(1, 1, 1, channel_count));
  std::vector<float> scale_vec(channel_count, scale);
  quantized_scale_desc.UploadDataRaw(absl::MakeConstSpan(scale_vec));
  return builder.AddConstantTensor(std::move(quantized_scale_desc));
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeSelfAttentionV3SubgraphPostKvCacheWrite(
    const std::string& subgraph_id, const BHWC& src_sh,
    GpuModelBuilder::TensorHandle q, GpuModelBuilder::TensorHandle mask,
    GpuModelBuilder::TensorHandle key_tensor,
    GpuModelBuilder::TensorHandle value_tensor,
    GpuModelBuilder::TensorHandle params_tensor,
    const std::string& kv_layer_name, bool is_global) {
  std::vector<GpuModelBuilder::TensorHandle> main_inputs{
      q, mask, key_tensor, value_tensor, params_tensor};

  const int head_size = GetHeadSize(config_, is_global);

  const DataType data_type = q.tensor_desc.GetDataType();
  GpuModelBuilder::TensorHandle k_quantized_scale_tensor;
  GpuModelBuilder::TensorHandle* k_scale_ptr = nullptr;
  GpuModelBuilder::TensorHandle v_quantized_scale_tensor;
  GpuModelBuilder::TensorHandle* v_scale_ptr = nullptr;
  float k_quant_scale = 0;
  float v_quant_scale = 0;
  if (config_.apply_srq && config_.int8_kv_cache) {
    k_quant_scale = GetQuantScale(
        kv_layer_name + ".k_cache_static_quantized_scale", data_type);
    v_quant_scale = GetQuantScale(
        kv_layer_name + ".v_cache_static_quantized_scale", data_type);
  } else if (config_.force_int8_kv_cache) {
    k_quant_scale = kStaticKvQuantScale;
    v_quant_scale = kStaticKvQuantScale;
  }
  if (k_quant_scale > 0 && v_quant_scale > 0) {
    const bool use_local_cache = UseSmallLocalCache(config_) && !is_global;
    const int cache_size =
        use_local_cache ? GetLocalCacheSize() : config_.cache_size;
    k_quantized_scale_tensor = CreateKVCacheQuantizedScaleTensor(
        model_builder_, data_type, cache_size, k_quant_scale);
    k_scale_ptr = &k_quantized_scale_tensor;
    v_quantized_scale_tensor = CreateKVCacheQuantizedScaleTensor(
        model_builder_, data_type, head_size, v_quant_scale);
    v_scale_ptr = &v_quantized_scale_tensor;
    main_inputs.insert(main_inputs.end(),
                       {k_quantized_scale_tensor, v_quantized_scale_tensor});
  }

  if (!model_builder_.HasSubgraph(subgraph_id)) {
    const int heads_count = config_.number_of_heads;

    const int batch_size = src_sh.b;
    const int sequence_size = src_sh.w;

    GpuModelBuilder builder = model_builder_.CreateBuilder();
    auto subgraph_inputs =
        (k_quant_scale > 0 && v_quant_scale > 0)
            ? CreateSubgraphInputs(builder,
                                   {&q, &mask, &key_tensor, &value_tensor,
                                    &params_tensor, k_scale_ptr, v_scale_ptr})
            : CreateSubgraphInputs(builder, {&q, &mask, &key_tensor,
                                             &value_tensor, &params_tensor});
    GpuModelBuilder::TensorHandle k;
    GpuModelBuilder::TensorHandle v;
    if (kv_cache_batch_size_ != 1 && config_.batch_size == 1) {
      // extract first batch slice, this can happen when batched kv cache used
      // with non batched model, that can happen in multi output model
      k = ExtractFirstBatch(builder, key_tensor);
      v = ExtractFirstBatch(builder, value_tensor);
    } else {
      k = key_tensor;
      v = value_tensor;
    }

    // TODO: Enable FlashDecodingAttention after performance evaluation.
    const bool is_flash_decoding_attention_enabled = false;

    std::string reason_of_not_supported;
    const bool use_flash_decoding_attention =
        is_flash_decoding_attention_enabled && is_global &&
        k_scale_ptr == nullptr &&  // The KV cache is not quantized.
        sequence_size == 1 && batch_size == 1 &&
        config_.attention_mask_type ==
            odml::infra::proto::TransformerParameters::CAUSAL &&
        !config_.enable_decode_logits && config_.attn_soft_cap_value == 0.0f &&
        IsFlashDecodingAttentionSupported(
            builder.gpu_info(),
            MakeFlashDecodingAttentionAttributes(config_, is_global),
            &reason_of_not_supported);

    //  q - hwc(B*KVHC, HGS*S, HS)
    //  k - hwc(B*KVHC, CS, HS)
    //  v - hwc(B*KVHC, CS, HS)
    //  att - hwc(B*KVHC, HGS*S, HS)
    GpuModelBuilder::TensorHandle att;
    if (use_flash_decoding_attention) {
      ASSIGN_OR_RETURN(att,
                       MakeFlashDecodingAttention(builder, config_, q, k, v,
                                                  params_tensor, is_global));
    } else {
      auto qk_post_proc_graph = [this, mask, &builder](auto src) {
        GpuModelBuilder::TensorHandle val = src;
        if (config_.attn_soft_cap_value > 0.0f) {  // cap_logits
          const float cap_value = config_.attn_soft_cap_value;
          val = builder.Multiplication(val, 1.0f / cap_value);
          val = builder.Elementwise(val, OperationType::kTanh);
          val = builder.Multiplication(val, cap_value);
        }
        return MakeSelectMaskOp(builder, val, mask);
      };
      ASSIGN_OR_RETURN(att, MakeMatMulSoftmaxMatMul(
                                builder, q, k, k_scale_ptr, v, v_scale_ptr,
                                params_tensor, qk_post_proc_graph, is_global));
    }
    // att transform to bhwc(B, 1, S, KVHC*HGS*HS)
    // att - bhwc(B, KVHC*HGS, S, HS) = bhwc(B, HC, S, HS)
    // att - bhwc(B, S, HC, HS)
    // att - bhwc(B, 1, S, HC*HS)
    att = builder.Reshape(
        att, BHWC{batch_size, heads_count, sequence_size, head_size});
    att = builder.Transpose(att, BHWC(0, 2, 1, 3));
    att = builder.Reshape(
        att, BHWC{batch_size, 1, sequence_size, heads_count * head_size});
    RETURN_IF_ERROR(model_builder_.RegisterSubgraph(
        std::move(builder), subgraph_id, subgraph_inputs, {att}));
  }
  ASSIGN_OR_RETURN(auto dsts,
                   model_builder_.Subgraph(subgraph_id, main_inputs));
  return dsts[0];
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeFFLayerMoe(const GpuModelBuilder::TensorHandle& src,
                                   const std::string& name) {
  // Shared: src -> pre_ffw2 -> mlp2 -> post_ffw2
  GpuModelBuilder::TensorHandle t = src;
  t = MakeRMSNormalization(t, name + ".pre_ffw2_norm",
                           config_.rms_norm_scale_shift);
  // mlp2: gating, activations, linear
  // ffn_layer1_gate
  ASSIGN_OR_RETURN(auto gate_value,
                   MakeLinearAutoPrecision(t, name + ".mlp2.ff_gate",
                                           config_.hidden_dimension));
  gate_value = model_builder_.MakeGelu(gate_value);
  // ffn_layer1
  ASSIGN_OR_RETURN(t, MakeLinearAutoPrecision(t, name + ".mlp2.ff1",
                                              config_.hidden_dimension));
  t = model_builder_.Multiplication(t, gate_value);
  // ffn_layer2
  ASSIGN_OR_RETURN(t, MakeLinearAutoPrecision(t, name + ".mlp2.linear",
                                              config_.model_dimension));

  t = MakeRMSNormalization(t, name + ".post_ffw2_norm",
                           config_.rms_norm_scale_shift);

  // Experts: src -> pre_ffw -> mlp1(moe) -> post_ffw1
  GpuModelBuilder::TensorHandle x = src;
  x = MakeRMSNormalization(x, name + ".pre_ffw_norm",
                           config_.rms_norm_scale_shift);
  ASSIGN_OR_RETURN(x, MakeFFLayerExperts(x, src, name + ".mlp1"));
  x = MakeRMSNormalization(x, name + ".post_ffw1_norm",
                           config_.rms_norm_scale_shift);

  // Combine both paths
  auto combined = model_builder_.Add(x, t);
  combined = MakeRMSNormalization(combined, name + ".post_ffw_norm",
                                  config_.rms_norm_scale_shift);

  // Final residual
  return model_builder_.Add(src, combined);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBuilder::MakeFFLayerExperts(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& src_no_norm, const std::string& name) {
  // Initial renorm by sqrt(#features = model_dim)
  const int model_dim = config_.model_dimension;
  GpuModelBuilder::TensorHandle t = src_no_norm;
  t = model_builder_.RMSNormalization(t, /*epsilon=*/1e-6f);
  t = model_builder_.Multiplication(t,
                                    1.0f / sqrt(static_cast<float>(model_dim)));

  // Mult by router scales
  ml_drift::Tensor<Linear, DataType::kFloat32> router_scale;
  router_scale.shape = Linear(model_dim);
  router_scale.data =
      tensor_loader_->LoadFloat32(name + ".router.scale", model_dim);
  auto router_input = model_builder_.Multiplication(t, router_scale);

  // Conv with router.logits
  const int num_experts = config_.moe_config->num_experts;
  ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle logits,
                   MakeLinearAutoPrecision(
                       router_input, name + ".router_logits", num_experts));

  logits = model_builder_.Softmax(logits);

  // TODO: Consider approx top K or statistical variant.
  const int num_active_experts = config_.moe_config->num_active_experts;
  auto topk_outputs = model_builder_.TopK(logits, num_active_experts);
  auto top_weights = topk_outputs[0];
  auto top_indices = topk_outputs[1];

  auto moe_hidden = config_.moe_config->hidden_dimension;
  const int sequence_size = config_.sequence_size;
  // We process tokens directly from initial src
  auto expert_src = src;
  auto expert_params = top_indices;
  GpuModelBuilder::TensorHandle experts_packed_remap;
  const bool use_packed_groups =
      sequence_size * num_active_experts > num_experts;
  if (use_packed_groups) {
    auto vals =
        ml_drift::CreateExpertsRemap(model_builder_, top_indices, num_experts);
    experts_packed_remap = vals[2];
    expert_params = vals[1];  // experts count and offsets

    expert_src = ml_drift::ExpertsRemapTo(
        model_builder_, src, experts_packed_remap, num_active_experts);
  } else if (sequence_size != 1) {
    auto t = model_builder_.Tile(src, Axis::kHeight, num_active_experts);
    t = model_builder_.Transpose(t, BHWC(0, 2, 1, 3));
    auto packed_shape = t.tensor_desc.GetBHWCShape();
    packed_shape.h = packed_shape.w * packed_shape.h;
    packed_shape.w = 1;
    expert_src = model_builder_.Reshape(t, packed_shape);
    expert_params = model_builder_.Reshape(
        top_indices, BHWC(1, 1, 1, sequence_size * num_active_experts));
  }

  ASSIGN_OR_RETURN(auto gate_value, MakeAutoPrecisionOIConvWithIds(
                                        expert_src, expert_params,
                                        name + ".ff_gate", moe_hidden));
  // GeLU is elementwise, so fine with batches
  gate_value = model_builder_.MakeGelu(gate_value);
  ASSIGN_OR_RETURN(auto expert_outputs,
                   MakeAutoPrecisionOIConvWithIds(expert_src, expert_params,
                                                  name + ".ff1", moe_hidden));
  expert_outputs = model_builder_.Multiplication(expert_outputs, gate_value);

  // This conv-with-batched-ids op expects inputs and outputs to have
  // H=num_dispatches, and batched_ids to have C=num_dispatches.
  //
  // If sequence_size isn't 1, the num_dispatches is sequence_size *
  // num_active_experts; otherwise it's just num_active_experts there and it's 1
  // for the previous ones.
  ASSIGN_OR_RETURN(expert_outputs,
                   MakeAutoPrecisionOIConvWithIds(expert_outputs, expert_params,
                                                  name + ".linear", model_dim));

  if (use_packed_groups) {
    expert_outputs =
        ml_drift::ExpertsRemapFrom(model_builder_, expert_outputs,
                                   experts_packed_remap, num_active_experts);
    expert_outputs = model_builder_.Transpose(expert_outputs, BHWC(0, 2, 1, 3));
  }
  if (!use_packed_groups) {
    expert_outputs = model_builder_.Reshape(
        expert_outputs, BHWC(1, sequence_size, num_active_experts, model_dim));
  }

  // Now we apply our scale-by-experts
  ml_drift::Tensor<Linear, DataType::kFloat32> scales;
  scales.shape = Linear(num_experts);
  scales.data =
      tensor_loader_->LoadFloat32(name + ".per_expert.scale", num_experts);
  auto scale_tensor =
      model_builder_.AddConstantTensor(scales, DataType::kFloat32);
  ASSIGN_OR_RETURN(
      auto out, ScaleWithBatchIds(model_builder_, expert_outputs, top_indices,
                                  top_weights, scale_tensor));
  return out;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> TransformerBuilder::MakeFFLayerV3(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    bool use_statistical_top_k, bool is_global, bool is_kv_shared_layer) {
  GpuModelBuilder::TensorHandle t = src;

  int hidden_dimension = config_.hidden_dimension;
  if (config_.kv_shared_hidden_dimension.has_value() && is_kv_shared_layer) {
    hidden_dimension = *config_.kv_shared_hidden_dimension;
  }

  // V3 pre_ffw_norm has no scale shift. // V4 has shift.
  t = MakeRMSNormalization(t, name + ".pre_ffw_norm",
                           config_.rms_norm_scale_shift);

  // Apply LoRA injection to FFN layer input.
  // FFN(x + A'Bx)
  ASSIGN_OR_RETURN(
      t, MakeInputLoRAInjection(t, name + ".mlp", config_.model_dimension));

  std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
      matform_options;
  if (config_.matform_factor > 0) {
    matform_options = {
        .axis = Axis::kOutputChannels,
        .matform_factor = config_.matform_factor,
    };
  }
  // ffn_layer1_gate
  ASSIGN_OR_RETURN(
      auto gate_value,
      MakeLinearAutoPrecisionWithSrq(t, name + ".mlp.ff_gate", hidden_dimension,
                                     config_.apply_srq, matform_options));

  if (use_statistical_top_k) {
    gate_value = model_builder_.StatisticalTopK(
        gate_value, *config_.activation_sparsity_stddev_multiplier);
  }
  gate_value = model_builder_.MakeGelu(gate_value);

  // ffn_layer1
  ASSIGN_OR_RETURN(
      t, MakeLinearAutoPrecisionWithSrq(t, name + ".mlp.ff1", hidden_dimension,
                                        config_.apply_srq, matform_options));
  // Cap infinity after each op here to avoid fp16 overflow.
  t = CapInfinity(model_builder_.Multiplication(CapInfinity(t), gate_value));

  if (matform_options) {
    matform_options->axis = Axis::kInputChannels;
  }
  // ffn_layer2
  ASSIGN_OR_RETURN(t, MakeLinearAutoPrecisionWithSrq(
                          t, name + ".mlp.linear", config_.model_dimension,
                          config_.apply_srq, matform_options));
  t = CapInfinity(t);

  // For v3, use_gated_residual is False, so it falls back to the simple
  // residual connection.
  // This norm+add operation overflows for some older checkpoints.

  // post_layer_norm
  if (config_.post_norm_ff) {
    // V3 Post ffw norm has learnable scale and no scale shift. // V4 has shift
    t = MakeRMSNormalization(t, name + ".post_ffw_norm",
                             config_.rms_norm_scale_shift);
  }

  t = CapInfinity(model_builder_.Add(t, src));
  return t;
}

absl::StatusOr<GpuModel> TransformerBuilder::CreateFinalEmbeddingToLogitsModel(
    GpuModelBuilder::TensorHandle* input_handle,
    GpuModelBuilder::TensorHandle* logits_handle, int sequence_size) {
  RET_CHECK(UseGemmaModelV3NPlus());

  CleanState();

  config_.sequence_size = sequence_size;
  config_.batch_size = kv_cache_batch_size_;

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);
  *input_handle = model_builder_.AddTensor(
      BHWC(config_.batch_size, 1, config_.sequence_size,
           config_.model_dimension),
      data_type_);

  auto t = *input_handle;
  ASSIGN_OR_RETURN(
      t, MakeConvAutoPrecision(t, "transformer.embedder.input_embedding",
                               Layout::kOIHW, config_.vocabulary_size));

  if (config_.final_proj_soft_cap_value > 0.0f) {
    float cap_value = config_.final_proj_soft_cap_value;
    t = model_builder_.Multiplication(t, 1.0f / cap_value);
    t = model_builder_.Elementwise(t, OperationType::kTanh);
    t = model_builder_.Multiplication(t, cap_value);
  }

  *logits_handle = t;

  GpuModel gpu_model;
  RETURN_IF_ERROR(model_builder_.GetGpuModel({input_handle->id},
                                             {logits_handle->id}, &gpu_model));
  return std::move(gpu_model);
}

absl::StatusOr<GpuModel> TransformerBuilder::CreateEmbeddingLookupModel(
    int sequence_size, int batch_size,
    GpuModelBuilder::TensorHandle* current_tokens,
    GpuModelBuilder::TensorHandle* output_embeddings,
    std::vector<odml::infra::gpu::EmbeddingWeights>* ple_weights,
    std::vector<GpuModelBuilder::TensorHandle>* output_ple_embeddings) {
  // Only verified for V3
  RET_CHECK(UseGemmaModelV3NPlus());

  CleanState();
  config_.sequence_size = sequence_size;
  config_.batch_size = batch_size;

  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);

  *current_tokens = model_builder_.AddTensor(
      GetInputTensorDescriptor(batch_size, sequence_size));

  // Part 1 - Model Embedding
  ASSIGN_OR_RETURN(
      auto embeddings_full,
      LoadWeights("transformer.embedder.input_embedding",
                  OHWI(config_.vocabulary_size, 1, 1, config_.model_dimension),
                  /*swap_dims=*/false));
  *output_embeddings = MakeEmbLookUpOp(*current_tokens, embeddings_full);

  // Part 2 - Per layer embeddings
  if (config_.per_layer_embedding_dimension.has_value()) {
    ASSIGN_OR_RETURN(*output_ple_embeddings,
                     LoadPerLayerEmbeddings(*current_tokens, ple_weights));
  }

  std::vector<GpuModelBuilder::ValueId> input_ids = {current_tokens->id};
  std::vector<GpuModelBuilder::ValueId> output_ids = {output_embeddings->id};
  for (auto& ple_embedding : *output_ple_embeddings) {
    output_ids.push_back(ple_embedding.id);
  }
  if (ple_weights) {
    // Per layer embeddings managed externally.
    for (const auto& ple : *ple_weights) {
      input_ids.push_back(ple.weights.id);
      input_ids.push_back(ple.scales.id);
    }
  }
  GpuModel gpu_model;
  RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

CreateGpuModelInfo GetCreateGpuModelInfo(
    const CreateGpuModelInfo& create_info,
    const absl::flat_hash_map<ValueId, GpuSpatialTensor*>&
        external_immutable_tensors,
    const absl::flat_hash_map<ValueId, TensorDescriptor>&
        external_mutable_tensors) {
  CreateGpuModelInfo create_info_new = create_info;
  for (const auto& [id, tensor] : external_immutable_tensors) {
    create_info_new.external_immutable_tensors[id] = tensor;
  }
  for (const auto& [id, tensor_desc] : external_mutable_tensors) {
    create_info_new.external_mutable_tensors[id] = tensor_desc;
  }
  return create_info_new;
}

namespace {
int GetRingOffset(const TransformerConfig& config, int token_offset) {
  if (!TransformerBuilder::UseSmallLocalCache(config)) {
    return 0;
  }
  const int ring_cache_size =
      TransformerBuilder::GetCacheSize(config, /*is_global=*/false);
  return std::max(0, token_offset + 1 - *config.sliding_window_size) %
         ring_cache_size;
}
}  // namespace

std::vector<int32_t> GetLlmRuntimeParamsBuffer(
    const LlmRuntimeParams& param_vals, const TransformerConfig& config) {
  std::vector<int32_t> params_i32(LlmRuntimeParams::kTotalParamsCount, 0);
  params_i32[0] = param_vals.token_index_offset;
  if (param_vals.active_tokens.has_value()) {
    params_i32[1] = *param_vals.active_tokens;
    const int active_tokens_aligned = std::min(
        config.cache_size, AlignByN(*param_vals.active_tokens,
                                    ConvRuntimeCheckDesc::kChannelsAlignment));
    params_i32[2] = active_tokens_aligned;
  }
  params_i32[3] = param_vals.topk;
  params_i32[4] = GetRingOffset(config, param_vals.token_index_offset);
  return params_i32;
}
}  // namespace ml_drift
