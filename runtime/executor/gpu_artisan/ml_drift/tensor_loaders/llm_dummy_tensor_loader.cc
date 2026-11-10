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

#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_dummy_tensor_loader.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/name_check.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::gpu {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::BHWDC;
using ::ml_drift::DataType;
using ::ml_drift::GetTotalElementsCountForLayout;
using ::ml_drift::GpuSpatialTensor;
using ::ml_drift::Layout;
using ::ml_drift::Linear;
using ::ml_drift::OHWI;
using ::ml_drift::RearrangeWeights;
using ::ml_drift::SizeOf;
using ::ml_drift::Tensor;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::uint2;
using ::ml_drift::WeightsDescription;

// RearrangeWeights requires 16 extra tail bytes for SIMD over-reads.
constexpr size_t kWeightRearrangeExtraBytes = 16;

}  // namespace

std::vector<float> LlmDummyTensorLoader::LoadFloat32(
    const std::string& tensor_name, int count) {
  const auto& it = tensor_data_.find(tensor_name);
  if (it != tensor_data_.end()) {
    return it->second;
  }
  return GenerateFloat32Data(count);
}

absl::StatusOr<GpuSpatialTensor*> LlmDummyTensorLoader::LoadScale(
    const std::string& tensor_name, const ::ml_drift::OHWI& shape,
    std::optional<MatformOptions> matform_options) {
  Tensor<OHWI, DataType::kFloat32> tensor;
  tensor.shape = shape;
  tensor.data = GenerateFloat32Data(tensor.shape.DimensionsProduct());
  TensorDescriptor td =
      ::ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info_, tensor, data_type_);
  return create_tensor_(td);
}

absl::StatusOr<GpuSpatialTensor*> LlmDummyTensorLoader::LoadZeroPoint(
    const std::string& zero_point_name, const std::string& scale_name,
    const ::ml_drift::OHWI& shape) {
  Tensor<OHWI, DataType::kFloat32> tensor;
  tensor.shape = shape;
  tensor.data = GenerateFloat32Data(tensor.shape.DimensionsProduct());
  TensorDescriptor td =
      ::ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info_, tensor, data_type_);
  return create_tensor_(td);
}

absl::StatusOr<GpuSpatialTensor*> LlmDummyTensorLoader::LoadBias(
    const std::string& tensor_name, int count) {
  return CreateTensorLinearFloat(count);
}

absl::StatusOr<GpuSpatialTensor*> LlmDummyTensorLoader::LoadWeights(
    const std::string& tensor_name, const WeightsDescription& weights_desc,
    const OHWI& shape, bool swap_dims) {
  auto data_weights = LoadFloat32(tensor_name, shape.DimensionsProduct());
  Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      kWeightRearrangeExtraBytes / sizeof(float));
  for (int i = 0; i < weights.shape.i; ++i) {
    for (int h = 0; h < weights.shape.h; ++h) {
      for (int o = 0; o < weights.shape.o; ++o) {
        if (swap_dims) {
          weights.data[weights.shape.LinearIndex({o, h, 0, i})] =
              data_weights[i * shape.o * shape.h + h * shape.o + o];
        } else {
          weights.data[weights.shape.LinearIndex({o, h, 0, i})] =
              data_weights[o * shape.i * shape.h + h * shape.i + i];
        }
      }
    }
  }
  TensorDescriptor weights_td;
  {
    weights_td = TensorDescriptor(weights_desc.type, TensorStorageType::kBuffer,
                                  Layout::kLinear);
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights.shape) *
        SizeOf(weights_desc.type));
    RearrangeWeights(weights, weights_desc, absl::MakeSpan(data));

    weights_td.SetBHWCShape(
        BHWC(1, 1, 1, data.size() / SizeOf(weights_desc.type)));
    weights_td.UploadDataRaw(absl::MakeConstSpan(data));
  }
  return create_tensor_(weights_td);
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
LlmDummyTensorLoader::LoadInt8Weights(const std::string& tensor_name,
                                      const WeightsDescription& weights_desc,
                                      const OHWI& shape, bool swap_dims) {
  TensorDescriptor weights_i8_td;
  if (weights_desc.IsLinearLayout()) {
    weights_i8_td = TensorDescriptor(
        weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
    weights_i8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, shape.DimensionsProduct()));
  } else {
    weights_i8_td = TensorDescriptor(
        DataType::kUint32, TensorStorageType::kTexture2D, Layout::kHW);
    uint2 tex_size = Get2dResourceSize(weights_desc, shape);
    tex_size.x /= 4;  // because we store 4 uint8 as one uint32
    weights_i8_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  }
  weights_i8_td.SetData(GenerateBytesData(shape.DimensionsProduct()));
  QuantizedWeights int8_weights;
  ASSIGN_OR_RETURN(int8_weights.weights, create_tensor_(weights_i8_td));
  ASSIGN_OR_RETURN(int8_weights.weights_sum_i,
                   CreateTensor(DataType::kInt32, shape.o));
  return int8_weights;
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
LlmDummyTensorLoader::LoadInt4Weights(
    const std::string& tensor_name, const WeightsDescription& weights_desc,
    const OHWI& shape, bool swap_dims,
    std::optional<MatformOptions> matform_options) {
  TensorDescriptor weights_i8_td;
  if (weights_desc.IsLinearLayout()) {
    weights_i8_td = TensorDescriptor(
        weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
    weights_i8_td.SetBHWDCShape(
        BHWDC(1, 1, 1, 1, shape.DimensionsProduct() / 2));
  } else {
    weights_i8_td = TensorDescriptor(
        DataType::kUint16, TensorStorageType::kTexture2D, Layout::kHW);
    uint2 tex_size = Get2dResourceSize(weights_desc, shape);
    tex_size.x /= 4;  // because we store 4 uint4 as one uint16
    weights_i8_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  }
  size_t data_size = shape.DimensionsProduct() / 2;
  if (matform_options) {
    data_size /= matform_options->matform_factor;
  }
  weights_i8_td.SetData(GenerateBytesData(data_size));
  QuantizedWeights int4_weights;
  ASSIGN_OR_RETURN(int4_weights.weights, create_tensor_(weights_i8_td));
  ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                   CreateTensor(DataType::kInt32, shape.o));
  return int4_weights;
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
LlmDummyTensorLoader::LoadInt2Weights(const std::string& tensor_name,
                                      const WeightsDescription& weights_desc,
                                      const OHWI& shape, bool swap_dims) {
  TensorDescriptor weights_i8_td;
  if (weights_desc.IsLinearLayout()) {
    weights_i8_td = TensorDescriptor(
        weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
    weights_i8_td.SetBHWDCShape(
        BHWDC(1, 1, 1, 1, shape.DimensionsProduct() / 4));
  } else {
    weights_i8_td = TensorDescriptor(
        DataType::kUint8, TensorStorageType::kTexture2D, Layout::kHW);
    uint2 tex_size = Get2dResourceSize(weights_desc, shape);
    tex_size.x /= 4;  // because we store 4 uint2 as one uint8
    weights_i8_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  }
  weights_i8_td.SetData(GenerateBytesData(shape.DimensionsProduct() / 2));
  QuantizedWeights int2_weights;
  ASSIGN_OR_RETURN(int2_weights.weights, create_tensor_(weights_i8_td));
  ASSIGN_OR_RETURN(int2_weights.weights_sum_i,
                   CreateTensor(DataType::kInt32, shape.o));
  return int2_weights;
}

int LlmDummyTensorLoader::GetTensorElementSizeInBits(
    const std::string& tensor_name, int count) {
  const auto& it = tensor_data_.find(tensor_name);
  if (it != tensor_data_.end()) {
    return 32;
  }
  WeightsType type = weights_type_;
  for (const auto& subname : explicit_tensor_subnames_) {
    if (tensor_name.find(subname.first) != std::string::npos) {
      type = subname.second;
      break;
    }
  }
  switch (type) {
    case WeightsType::kInt8:
      return 8;
    case WeightsType::kInt4:
      return 4;
    case WeightsType::kInt2:
      return 2;
    case WeightsType::kFloat32:
      return 32;
  }
}

int LlmDummyTensorLoader::GetQuantizationIGroupSize(
    const std::string& weights_name,
    const ml_drift::OHWI& weights_shape) const {
  int quantization_i_group_size = quantization_i_group_size_;
  for (const auto& subname : explicit_quantization_i_group_size_subnames_) {
    if (weights_name.find(subname.first) != std::string::npos) {
      quantization_i_group_size = subname.second;
      break;
    }
  }
  quantization_i_group_size = quantization_i_group_size == -1
                                  ? weights_shape.i
                                  : quantization_i_group_size;
  if (weights_shape.i % quantization_i_group_size != 0 ||
      quantization_i_group_size % 4 != 0) {
    ABSL_LOG(ERROR) << "Invalid quantization group size - "
                    << quantization_i_group_size << std::endl;
  }
  return quantization_i_group_size;
}

absl::StatusOr<GpuSpatialTensor*> LlmDummyTensorLoader::CreateTensor(
    DataType data_type, int count) {
  TensorDescriptor desc(
      data_type,
      GetStorageTypeForLinearTensor(gpu_info_, data_type, Linear(count)),
      Layout::kLinear);
  int size = count * SizeOf(data_type);
  desc.SetBHWDCShape(BHWDC(1, 1, 1, 1, count));
  desc.SetData(GenerateBytesData(size));
  return create_tensor_(desc);
}

absl::StatusOr<GpuSpatialTensor*> LlmDummyTensorLoader::CreateTensorLinearFloat(
    int count) {
  ml_drift::Tensor<Linear, DataType::kFloat32> tensor;
  tensor.shape = Linear(count);
  tensor.data = GenerateFloat32Data(count);
  return create_tensor_(CreateConstantLinearTensorDescriptor(
      gpu_info_, data_type_, std::move(tensor)));
}

std::vector<float> LlmDummyTensorLoader::GenerateFloat32Data(int count) {
  std::vector<float> data(count);
  if (use_random_data_) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < count; ++i) {
      data[i] = dist(rng_);
    }
  }
  return data;
}

std::vector<uint8_t> LlmDummyTensorLoader::GenerateBytesData(size_t size) {
  std::vector<uint8_t> data(size);
  if (use_random_data_) {
    const size_t num_u64 = size / sizeof(uint64_t);
    for (size_t i = 0; i < num_u64; ++i) {
      uint64_t val = rng_();
      std::memcpy(data.data() + i * sizeof(uint64_t), &val, sizeof(uint64_t));
    }
    uint64_t tail = rng_();
    for (size_t i = num_u64 * sizeof(uint64_t); i < size; ++i) {
      data[i] = static_cast<uint8_t>(tail & 0xFF);
      tail >>= 8;
    }
  }
  return data;
}

bool NonZeroTensorLoader::HasTensor(const std::string& tensor_name) {
  if (IsLoRATensorName(tensor_name)) {
    return ShouldPopulateLoRA(tensor_name);
  }
  return LlmDummyTensorLoader::HasTensor(tensor_name);
}

std::vector<float> NonZeroTensorLoader::LoadFloat32(
    const std::string& tensor_name, int count) {
  std::vector<float> data(count);
  // LoRA tensors will default to 0.
  if (ShouldPopulateLoRA(tensor_name) || !IsLoRATensorName(tensor_name)) {
    for (int i = 0; i < count; i++) {
      data[i] = start_value_ * i / count;
    }
  }
  return data;
}

bool NonZeroTensorLoader::ShouldPopulateLoRA(const std::string& tensor_name) {
  return lora_mode_ == LoRAMode::kAttentionFeedForward ||
         (lora_mode_ == LoRAMode::kAttention &&
          !IsFeedForwardTensorName(tensor_name));
}

}  // namespace odml::infra::gpu
