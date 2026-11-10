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

#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_file_tensor_loader.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/name_check.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"

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
using ::ml_drift::WeightsLayout;

// RearrangeWeights requires 16 extra tail bytes for SIMD over-reads.
constexpr size_t kWeightRearrangeExtraBytes = 16;

size_t GetFileSize(const std::string& file_path) {
  std::ifstream ifstr(file_path, std::ios::binary);
  ABSL_QCHECK(ifstr) << "Cannot open file! " << file_path << std::endl;
  const auto begin = ifstr.tellg();
  ifstr.seekg(0, std::ios::end);
  const auto end = ifstr.tellg();
  return end - begin;
}

ml_drift::Tensor<OHWI, DataType::kFloat32> ReadScaleOrZeroPointInt(
    const std::string& tensor_name, const ml_drift::OHWI& shape) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = shape;
  scale.data.resize(shape.DimensionsProduct());
  const int scale_size = GetFileSize(tensor_name);
  const int bytes_size = (scale_size - 4) / shape.DimensionsProduct();
  std::ifstream ifstr(tensor_name, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(ERROR) << "Cannot open file! " << tensor_name << std::endl;
    return scale;
  }
  int32_t group_size = 0;
  ifstr.read((char*)&group_size, sizeof(int32_t));
  if (bytes_size == 1) {
    std::vector<int8_t> result(shape.DimensionsProduct());
    ifstr.read((char*)result.data(), sizeof(int8_t) * result.size());
    for (int i = 0; i < shape.DimensionsProduct(); ++i) {
      scale.data[i] = result[i] / 1024.0f;
    }
  } else if (bytes_size == 2) {
    std::vector<int16_t> result(shape.DimensionsProduct());
    ifstr.read((char*)result.data(), sizeof(int16_t) * result.size());
    for (int i = 0; i < shape.DimensionsProduct(); ++i) {
      scale.data[i] = result[i] / 1024.0f;
    }
  }
  return scale;
}

std::vector<float> LoadFloat32File(const std::string& file_name, int count) {
  std::vector<float> result(count);
  std::ifstream ifstr(file_name, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(ERROR) << "Cannot open file! " << file_name << std::endl;
    return {};
  }
  ifstr.read((char*)result.data(), sizeof(float) * count);
  return result;
}

ml_drift::Tensor<OHWI, DataType::kFloat32> LoadScaleOrZeroPoint(
    const std::string& tensor_name, const ml_drift::OHWI& shape) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> scale;
  // Float scale for standard or blockwise quant schemes
  if ((GetFileSize(tensor_name) == shape.o * shape.h * 4) ||
      (GetFileSize(tensor_name) == (shape.o * shape.h * shape.i * 4))) {
    // linear quantization, fp32 values
    scale.shape = shape;
    scale.data = LoadFloat32File(tensor_name, shape.DimensionsProduct());
  } else {
    // grouped quantization, int8/16 values
    scale = ReadScaleOrZeroPointInt(tensor_name, shape);
  }
  return scale;
}

std::vector<uint8_t> ReadFile(const std::string& file_name) {
  std::vector<uint8_t> result(GetFileSize(file_name));
  std::ifstream ifstr(file_name, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(ERROR) << "Cannot open file! " << file_name << std::endl;
    return {};
  }
  ifstr.read((char*)result.data(), result.size());
  return result;
}

}  // namespace

std::vector<float> LlmFileTensorLoader::LoadFloat32(
    const std::string& tensor_name, int count) {
  // Note: All LoRA weights are assumed to be stored as F32.
  return LoadFloat32File(GetTensorPath(tensor_name), count);
}

std::vector<int8_t> LlmFileTensorLoader::LoadInt8(
    const std::string& tensor_name, int count) {
  const std::string path =
      (std::filesystem::path(weights_directory_) / tensor_name).string();
  std::vector<int8_t> result(count);
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(ERROR) << "Cannot open file! " << path << std::endl;
    return {};
  }
  ifstr.read((char*)result.data(), sizeof(int8_t) * count);
  return result;
}

std::vector<int8_t> LlmFileTensorLoader::LoadInt4AsInt8(
    const std::string& tensor_name, int count) {
  const std::string path =
      (std::filesystem::path(weights_directory_) / tensor_name).string();
  if (count % 8 != 0) {
    ABSL_LOG(ERROR) << "Count must be divisible by 8!" << std::endl;
    return {};
  }
  std::vector<int32_t> data_i32(count / 8);
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(ERROR) << "Cannot open file! " << path << std::endl;
    return {};
  }
  ifstr.read((char*)data_i32.data(), sizeof(int32_t) * data_i32.size());

  std::vector<int8_t> data_i8(count);
  for (int i = 0; i < data_i32.size(); ++i) {
    std::vector<int8_t> values(8);
    int value = data_i32[i];
    for (int j = 0; j < 8; ++j) {
      int tmp = (value + 8) & 15;
      values[j] = tmp - 8;
      value = (value >> 4);
    }
    for (int j = 0; j < 8; ++j) {
      data_i8[i * 8 + j] = values[j];
    }
  }
  return data_i8;
}

std::vector<int8_t> LlmFileTensorLoader::LoadInt2AsInt8(
    const std::string& tensor_name, int count) {
  const std::string path =
      (std::filesystem::path(weights_directory_) / tensor_name).string();
  if (count % 16 != 0) {
    ABSL_LOG(ERROR) << "Count must be divisible by 16!" << std::endl;
    return {};
  }
  std::vector<int32_t> data_i32(count / 16);
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(ERROR) << "Cannot open file! " << path << std::endl;
    return {};
  }
  ifstr.read((char*)data_i32.data(), sizeof(int32_t) * data_i32.size());

  std::vector<int8_t> data_i8(count);
  for (int i = 0; i < data_i32.size(); ++i) {
    std::vector<int8_t> values(16);
    int value = data_i32[i];
    for (int j = 0; j < 16; ++j) {
      int tmp = (value + 2) & 3;
      values[j] = tmp - 2;
      value = (value >> 2);
    }
    for (int j = 0; j < 16; ++j) {
      data_i8[i * 16 + j] = values[j];
    }
  }
  return data_i8;
}

std::vector<uint8_t> LlmFileTensorLoader::LoadInt4Packed(
    const std::string& tensor_name, int count) {
  const std::string path =
      (std::filesystem::path(weights_directory_) / tensor_name).string();
  if (count % 2 != 0) {
    ABSL_LOG(FATAL) << "Count must be divisible by 2!" << std::endl;
  }
  std::vector<uint8_t> data_ui8(count / 2);
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(FATAL) << "Cannot open file! " << path << std::endl;
  }
  ifstr.read((char*)&data_ui8[0], sizeof(uint8_t) * data_ui8.size());

  return data_ui8;
}

int LlmFileTensorLoader::GetTensorElementSizeInBits(
    const std::string& tensor_name, int count) {
  return (GetFileSize(GetTensorPath(tensor_name)) * 8) / count;
}

bool LlmFileTensorLoader::HasTensor(const std::string& tensor_name) {
  std::ifstream ifstr(GetTensorPath(tensor_name), std::ios::binary);
  if (ifstr) {
    return true;
  } else {
    return false;
  }
}

int LlmFileTensorLoader::GetQuantizationIGroupSize(
    const std::string& weights_name,
    const ml_drift::OHWI& weights_shape) const {
  auto it = group_size_cache_.find(weights_name);
  if (it != group_size_cache_.end()) {
    return it->second;
  }

  const std::string base_name = weights_name.substr(0, weights_name.rfind('.'));
  const std::string scale_name = base_name + ".w_quantized_scale";
  const std::string scale_path = GetTensorPath(scale_name);
  const int scale_size = GetFileSize(scale_path);
  if (scale_size == weights_shape.o * weights_shape.h * 4) {
    group_size_cache_[weights_name] = weights_shape.i;
    return weights_shape.i;
  } else {
    std::ifstream ifstr(scale_path, std::ios::binary);
    if (!ifstr) {
      ABSL_LOG(ERROR) << "Cannot open file! " << scale_path << std::endl;
      group_size_cache_[weights_name] = weights_shape.i;
      return weights_shape.i;
    }
    int32_t group_size = 0;
    ifstr.read((char*)&group_size, sizeof(int32_t));
    group_size_cache_[weights_name] = group_size;
    return group_size;
  }
}

absl::StatusOr<GpuSpatialTensor*> LlmFileTensorLoader::LoadScale(
    const std::string& tensor_name, const ::ml_drift::OHWI& shape,
    std::optional<MatformOptions> matform_options) {
  if (matform_options) {
    return absl::InternalError("Matform not supported.");
  }
  auto scale = LoadScaleOrZeroPoint(GetTensorPath(tensor_name), shape);
  TensorDescriptor td =
      ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info_, scale, data_type_);
  return create_tensor_(td);
}

absl::StatusOr<GpuSpatialTensor*> LlmFileTensorLoader::LoadZeroPoint(
    const std::string& zero_point_name, const std::string& scale_name,
    const ::ml_drift::OHWI& shape) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> zero_point;
  if (HasTensor(zero_point_name)) {
    zero_point = LoadScaleOrZeroPoint(GetTensorPath(zero_point_name), shape);
    auto scale = LoadScaleOrZeroPoint(GetTensorPath(scale_name), shape);
    for (int i = 0; i < zero_point.data.size(); ++i) {
      if (scale.data[i] == 0.0f) {
        zero_point.data[i] = 0.0f;
      } else {
        zero_point.data[i] /= scale.data[i];
      }
    }
  } else {
    zero_point.shape = shape;
    zero_point.data.resize(shape.DimensionsProduct());
  }
  TensorDescriptor td = ml_drift::ScaleOrZeroPointToTensorDesc(
      gpu_info_, zero_point, data_type_);
  return create_tensor_(td);
}

absl::StatusOr<GpuSpatialTensor*> LlmFileTensorLoader::LoadBias(
    const std::string& tensor_name, int count) {
  ml_drift::Tensor<Linear, DataType::kFloat32> bias;
  bias.shape = Linear(count);
  if (HasTensor(tensor_name)) {
    bias.data = LoadFloat32(tensor_name, count);
  } else {
    bias.data.resize(count);
  }
  return create_tensor_(CreateConstantLinearTensorDescriptor(
      gpu_info_, data_type_, std::move(bias)));
}

absl::StatusOr<GpuSpatialTensor*> LlmFileTensorLoader::LoadWeights(
    const std::string& tensor_name, const WeightsDescription& weights_desc,
    const OHWI& shape, bool swap_dims) {
  auto data_weights = LoadFloat32(tensor_name, shape.DimensionsProduct());

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      kWeightRearrangeExtraBytes / sizeof(float));
  for (int i = 0; i < shape.i; ++i) {
    for (int h = 0; h < shape.h; ++h) {
      for (int o = 0; o < shape.o; ++o) {
        if (swap_dims) {
          weights.data[shape.LinearIndex({o, h, 0, i})] =
              data_weights[i * shape.o * shape.h + h * shape.o + o];
        } else {
          weights.data[shape.LinearIndex({o, h, 0, i})] =
              data_weights[o * shape.i * shape.h + h * shape.i + i];
        }
      }
    }
  }

  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, shape);
  std::vector<uint8_t> data(elements_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(data));
  TensorDescriptor weights_td;
  weights_td = TensorDescriptor(weights_desc.type, TensorStorageType::kBuffer,
                                Layout::kLinear);
  weights_td.SetBHWCShape(BHWC(1, 1, 1, elements_count));
  weights_td.SetData(std::move(data));
  return create_tensor_(weights_td);
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
LlmFileTensorLoader::LoadInt8Weights(const std::string& tensor_name,
                                     const WeightsDescription& weights_desc,
                                     const OHWI& shape, bool swap_dims) {
  const std::vector<int8_t> datai8 =
      LoadInt8(tensor_name, shape.DimensionsProduct());

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = shape;
  if (swap_dims) {
    if (shape.h != 1) {
      return absl::InternalError("swap_dims not supported for 3D int8 weights");
    }
    weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
    for (int o = 0; o < weights_i8.shape.o; ++o) {
      for (int i = 0; i < weights_i8.shape.i; ++i) {
        weights_i8.data[weights_i8.shape.LinearIndex({o, 0, 0, i})] =
            datai8[i * weights_i8.shape.o + o];
      }
    }
  } else {
    weights_i8.data = std::move(datai8);
  }
  TensorDescriptor weights_i8_td =
      ml_drift::GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  QuantizedWeights int8_weights;
  ABSL_ASSIGN_OR_RETURN(int8_weights.weights, create_tensor_(weights_i8_td));
  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  auto weights_sum_i_td =
      CreateConstantLinearTensorDescriptor(gpu_info_, weights_sum_i);
  ABSL_ASSIGN_OR_RETURN(int8_weights.weights_sum_i,
                        create_tensor_(weights_sum_i_td));
  return int8_weights;
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
LlmFileTensorLoader::LoadInt4Weights(
    const std::string& tensor_name, const WeightsDescription& weights_desc,
    const OHWI& shape, bool swap_dims,
    std::optional<MatformOptions> matform_options) {
  if (matform_options) {
    return absl::InternalError("Matform not supported.");
  }

  if (swap_dims && shape.h != 1) {
    return absl::InternalError("swap_dims not supported for 3D int4 weights");
  }

  if (weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      weights_desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    return LoadInt4WeightsFast(tensor_name, weights_desc, shape, swap_dims);
  }

  const std::vector<int8_t> datai8 =
      LoadInt4AsInt8(tensor_name, shape.DimensionsProduct());

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = shape;
  if (swap_dims) {
    weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
    for (int o = 0; o < weights_i8.shape.o; ++o) {
      for (int i = 0; i < weights_i8.shape.i; ++i) {
        weights_i8.data[weights_i8.shape.LinearIndex({o, 0, 0, i})] =
            datai8[i * weights_i8.shape.o + o];
      }
    }
  } else {
    weights_i8.data = std::move(datai8);
  }
  TensorDescriptor weights_i4_td =
      ml_drift::GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  QuantizedWeights int4_weights;
  ABSL_ASSIGN_OR_RETURN(int4_weights.weights, create_tensor_(weights_i4_td));
  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  auto weights_sum_i_td =
      CreateConstantLinearTensorDescriptor(gpu_info_, weights_sum_i);
  ABSL_ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                        create_tensor_(weights_sum_i_td));
  return int4_weights;
}

// Only works for weights_desc.layout == kOSpatialIOGroupI4O4 ||
// k2DYIsSpatialIOAndXIsOGroupI4O4
absl::StatusOr<LlmTensorLoader::QuantizedWeights>
LlmFileTensorLoader::LoadInt4WeightsFast(const std::string& tensor_name,
                                         const WeightsDescription& weights_desc,
                                         const OHWI& shape, bool swap_dims) {
  const std::vector<uint8_t> vector_u8 =
      LoadInt4Packed(tensor_name, shape.DimensionsProduct());

  Tensor<OHWI, DataType::kUint8> tensor_u8;
  tensor_u8.shape = shape;
  tensor_u8.data = std::move(vector_u8);
  std::vector<uint8_t> arranged_vec_u8(tensor_u8.shape.DimensionsProduct() / 2);
  Tensor<Linear, DataType::kInt32> weights_sum_i;
  weights_sum_i.shape = Linear(shape.o);
  weights_sum_i.data.resize(shape.o);
  ABSL_RETURN_IF_ERROR(RearrangeWeightsUInt4Packed(
      tensor_u8, weights_desc, absl::MakeSpan(arranged_vec_u8),
      absl::MakeSpan(weights_sum_i.data), 8u, swap_dims));
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
    weights_i8_td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  }

  weights_i8_td.SetData(std::move(arranged_vec_u8));
  QuantizedWeights int4_weights;
  ABSL_ASSIGN_OR_RETURN(int4_weights.weights, create_tensor_(weights_i8_td));
  auto weights_sum_i_td =
      CreateConstantLinearTensorDescriptor(gpu_info_, weights_sum_i);
  ABSL_ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                        create_tensor_(weights_sum_i_td));
  return int4_weights;
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
LlmFileTensorLoader::LoadInt2Weights(const std::string& tensor_name,
                                     const WeightsDescription& weights_desc,
                                     const OHWI& shape, bool swap_dims) {
  const std::vector<int8_t> datai8 =
      LoadInt2AsInt8(tensor_name, shape.DimensionsProduct());

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = shape;
  if (swap_dims) {
    if (shape.h != 1) {
      return absl::InternalError("swap_dims not supported for 3D int2 weights");
    }

    weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
    for (int o = 0; o < weights_i8.shape.o; ++o) {
      for (int i = 0; i < weights_i8.shape.i; ++i) {
        weights_i8.data[weights_i8.shape.LinearIndex({o, 0, 0, i})] =
            datai8[i * weights_i8.shape.o + o];
      }
    }
  } else {
    weights_i8.data = std::move(datai8);
  }
  std::vector<uint8_t> data(weights_i8.shape.DimensionsProduct() / 4);
  RearrangeWeightsInt8AsUint2(weights_i8, weights_desc, absl::MakeSpan(data), 2,
                              2u);
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
    weights_i8_td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  }
  weights_i8_td.SetData(std::move(data));
  QuantizedWeights int2_weights;
  ABSL_ASSIGN_OR_RETURN(int2_weights.weights, create_tensor_(weights_i8_td));
  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  auto weights_sum_i_td =
      CreateConstantLinearTensorDescriptor(gpu_info_, weights_sum_i);
  ABSL_ASSIGN_OR_RETURN(int2_weights.weights_sum_i,
                        create_tensor_(weights_sum_i_td));
  return int2_weights;
}

absl::StatusOr<std::unique_ptr<llm_utils::DataHolder<uint8_t>>>
LlmFileTensorLoader::LoadRawTensorData(const std::string& name) {
  class VectorDataHolder : public llm_utils::DataHolder<uint8_t> {
   public:
    explicit VectorDataHolder(std::vector<uint8_t> data)
        : data_(std::move(data)) {}
    ~VectorDataHolder() override = default;

    absl::Span<uint8_t> GetData() const override {
      return absl::MakeSpan(data_);
    }

   public:
    mutable std::vector<uint8_t> data_;
  };
  return std::make_unique<VectorDataHolder>(ReadFile(GetTensorPath(name)));
}

std::string LlmFileTensorLoader::GetTensorPath(
    absl::string_view tensor_name) const {
  bool use_lora_dir =
      lora_weights_directory_.has_value() && IsLoRATensorName(tensor_name);
  return (std::filesystem::path(use_lora_dir ? *lora_weights_directory_
                                             : weights_directory_) /
          std::string(tensor_name))
      .string();
}

}  // namespace odml::infra::gpu
