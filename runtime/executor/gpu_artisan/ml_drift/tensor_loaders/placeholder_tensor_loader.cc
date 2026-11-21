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

#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/placeholder_tensor_loader.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::gpu {
namespace {

ml_drift::TensorDescriptor GetLinearTensorDescriptor(
    const ml_drift::GpuInfo& gpu_info, ml_drift::DataType data_type,
    int count) {
  ml_drift::TensorDescriptor desc = ml_drift::TensorDescriptor(
      data_type,
      GetStorageTypeForLinearTensor(gpu_info, data_type,
                                    ml_drift::Linear(count)),
      ml_drift::Layout::kLinear);
  desc.SetBHWDCShape(ml_drift::BHWDC(1, 1, 1, 1, count));
  return desc;
}

absl::StatusOr<ml_drift::TensorDescriptor> GetWeightTensorDescriptor(
    const ::ml_drift::WeightsDescription& weights_desc,
    const ::ml_drift::OHWI& shape) {
  ml_drift::DataType data_type = weights_desc.type;
  int count = shape.DimensionsProduct();
  if (data_type == ml_drift::DataType::kUint4) {
    count /= 2;
  }
  if (data_type == ml_drift::DataType::kUint2) {
    count /= 4;
  }

  if (weights_desc.IsLinearLayout()) {
    ml_drift::TensorDescriptor desc = ml_drift::TensorDescriptor(
        data_type, ml_drift::TensorStorageType::kBuffer,
        ml_drift::Layout::kLinear);
    desc.SetBHWDCShape(ml_drift::BHWDC(1, 1, 1, 1, count));
    return desc;
  }

  if (data_type != ml_drift::DataType::kUint8 &&
      data_type != ml_drift::DataType::kUint4 &&
      data_type != ml_drift::DataType::kUint2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported data type ", ml_drift::ToString(data_type)));
  }

  ml_drift::DataType texture_type = ml_drift::DataType::kUint32;
  if (data_type == ml_drift::DataType::kUint4) {
    texture_type = ml_drift::DataType::kUint16;
  } else if (data_type == ml_drift::DataType::kUint2) {
    texture_type = ml_drift::DataType::kUint8;
  }

  ml_drift::TensorDescriptor desc(texture_type,
                                  ml_drift::TensorStorageType::kTexture2D,
                                  ml_drift::Layout::kHW);
  ml_drift::uint2 tex_size = ml_drift::Get2dResourceSize(weights_desc, shape);
  // Values are packed: 4 x uint2 = uint8, 4 x uint4 = uint16, 4 x uint8 =
  // uint32.
  tex_size.x /= 4;
  desc.SetBHWDCShape(ml_drift::BHWDC(1, tex_size.y, tex_size.x, 1, 4));

  return desc;
}

}  // namespace

std::vector<float> PlaceholderTensorLoader::LoadFloat32(
    const std::string& tensor_name, int count) {
  return loader_.LoadFloat32(tensor_name, count);
}

int PlaceholderTensorLoader::GetTensorElementSizeInBits(
    const std::string& tensor_name, int count) {
  return loader_.GetTensorElementSizeInBits(tensor_name, count);
}

bool PlaceholderTensorLoader::HasTensor(const std::string& tensor_name) {
  return loader_.HasTensor(tensor_name);
}

int PlaceholderTensorLoader::GetQuantizationIGroupSize(
    const std::string& weights_name,
    const ml_drift::OHWI& weights_shape) const {
  return loader_.GetQuantizationIGroupSize(weights_name, weights_shape);
}

ReloadGroupFunctions* PlaceholderTensorLoader::GetGroupReloadFunctions(
    const std::string& name) {
  for (auto& [group_name, group_fns] : reload_group_fns_) {
    if (name.find(group_name) != std::string::npos) {
      return &group_fns;
    }
  }
  return nullptr;
}

absl::StatusOr<::ml_drift::GpuSpatialTensor*>
PlaceholderTensorLoader::LoadScale(
    const std::string& tensor_name, const ::ml_drift::OHWI& shape,
    std::optional<MatformOptions> matform_options) {
  ReloadGroupFunctions* tensor_reload_fns;
  if (!placeholder_mode_ ||
      (tensor_reload_fns = GetGroupReloadFunctions(tensor_name)) == nullptr) {
    return loader_.LoadScale(tensor_name, shape, matform_options);
  }

  ml_drift::TensorDescriptor desc = GetLinearTensorDescriptor(
      loader_.GetGpuInfo(), loader_.GetDataType(), shape.o);

  std::shared_ptr<PlaceholderTensor> tensor =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(std::move(desc));
  (*tensor_reload_fns)[tensor_name] =
      [tensor_name, shape, tensor, matform_options](
          LlmTensorLoader* loader) -> absl::StatusOr<TensorsWithIds> {
    RET_CHECK(tensor);
    ASSIGN_OR_RETURN(ml_drift::GpuSpatialTensor * new_tensor,
                     loader->LoadScale(tensor_name, shape, matform_options));
    return TensorsWithIds{{new_tensor, tensor->GetId()}};
  };
  return tensor.get();
}

absl::StatusOr<::ml_drift::GpuSpatialTensor*>
PlaceholderTensorLoader::LoadZeroPoint(const std::string& zero_point_name,
                                       const std::string& scale_name,
                                       const ::ml_drift::OHWI& shape) {
  ReloadGroupFunctions* tensor_reload_fns;
  if (!placeholder_mode_ || (tensor_reload_fns = GetGroupReloadFunctions(
                                 zero_point_name)) == nullptr) {
    return loader_.LoadZeroPoint(zero_point_name, scale_name, shape);
  }

  ml_drift::TensorDescriptor desc = GetLinearTensorDescriptor(
      loader_.GetGpuInfo(), loader_.GetDataType(), shape.o);

  std::shared_ptr<PlaceholderTensor> tensor =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(std::move(desc));
  (*tensor_reload_fns)[zero_point_name] =
      [zero_point_name, scale_name, shape,
       tensor](LlmTensorLoader* loader) -> absl::StatusOr<TensorsWithIds> {
    RET_CHECK(tensor);
    ASSIGN_OR_RETURN(ml_drift::GpuSpatialTensor * new_tensor,
                     loader->LoadZeroPoint(zero_point_name, scale_name, shape));
    return TensorsWithIds{{new_tensor, tensor->GetId()}};
  };
  return tensor.get();
}

absl::StatusOr<::ml_drift::GpuSpatialTensor*> PlaceholderTensorLoader::LoadBias(
    const std::string& tensor_name, int count) {
  ReloadGroupFunctions* tensor_reload_fns;
  if (!placeholder_mode_ ||
      (tensor_reload_fns = GetGroupReloadFunctions(tensor_name)) == nullptr) {
    return loader_.LoadBias(tensor_name, count);
  }

  ml_drift::TensorDescriptor desc = GetLinearTensorDescriptor(
      loader_.GetGpuInfo(), loader_.GetDataType(), count);

  std::shared_ptr<PlaceholderTensor> tensor =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(std::move(desc));
  (*tensor_reload_fns)[tensor_name] =
      [tensor_name, count,
       tensor](LlmTensorLoader* loader) -> absl::StatusOr<TensorsWithIds> {
    RET_CHECK(tensor);
    ASSIGN_OR_RETURN(ml_drift::GpuSpatialTensor * new_tensor,
                     loader->LoadBias(tensor_name, count));
    return TensorsWithIds{{new_tensor, tensor->GetId()}};
  };
  return tensor.get();
}

absl::StatusOr<::ml_drift::GpuSpatialTensor*>
PlaceholderTensorLoader::LoadWeights(
    const std::string& tensor_name,
    const ::ml_drift::WeightsDescription& weights_desc,
    const ::ml_drift::OHWI& shape, bool swap_dims) {
  ReloadGroupFunctions* tensor_reload_fns;
  if (!placeholder_mode_ ||
      (tensor_reload_fns = GetGroupReloadFunctions(tensor_name)) == nullptr) {
    return loader_.LoadWeights(tensor_name, weights_desc, shape, swap_dims);
  }

  ASSIGN_OR_RETURN(ml_drift::TensorDescriptor desc,
                   GetWeightTensorDescriptor(weights_desc, shape));

  std::shared_ptr<PlaceholderTensor> tensor =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(std::move(desc));
  (*tensor_reload_fns)[tensor_name] =
      [tensor_name, weights_desc, shape, swap_dims,
       tensor](LlmTensorLoader* loader) -> absl::StatusOr<TensorsWithIds> {
    RET_CHECK(tensor);
    ASSIGN_OR_RETURN(
        ml_drift::GpuSpatialTensor * new_tensor,
        loader->LoadWeights(tensor_name, weights_desc, shape, swap_dims));
    return TensorsWithIds{{new_tensor, tensor->GetId()}};
  };

  return tensor.get();
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
PlaceholderTensorLoader::LoadInt8Weights(
    const std::string& tensor_name,
    const ::ml_drift::WeightsDescription& weights_desc,
    const ::ml_drift::OHWI& shape, bool swap_dims) {
  ReloadGroupFunctions* tensor_reload_fns;
  if (!placeholder_mode_ ||
      (tensor_reload_fns = GetGroupReloadFunctions(tensor_name)) == nullptr) {
    return loader_.LoadInt8Weights(tensor_name, weights_desc, shape, swap_dims);
  }

  ASSIGN_OR_RETURN(ml_drift::TensorDescriptor desc_weights,
                   GetWeightTensorDescriptor(weights_desc, shape));
  std::shared_ptr<PlaceholderTensor> tensor =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(
          std::move(desc_weights));

  ml_drift::TensorDescriptor desc_weights_sum_i = GetLinearTensorDescriptor(
      loader_.GetGpuInfo(), ml_drift::DataType::kInt32, shape.o);
  std::shared_ptr<PlaceholderTensor> tensor_sum_i =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(
          std::move(desc_weights_sum_i));

  (*tensor_reload_fns)[tensor_name] =
      [tensor_name, weights_desc, shape, swap_dims, tensor, tensor_sum_i](
          LlmTensorLoader* loader) -> absl::StatusOr<TensorsWithIds> {
    RET_CHECK(tensor);
    RET_CHECK(tensor_sum_i);
    ASSIGN_OR_RETURN(
        LlmTensorLoader::QuantizedWeights new_tensors,
        loader->LoadInt8Weights(tensor_name, weights_desc, shape, swap_dims));
    return TensorsWithIds{{new_tensors.weights, tensor->GetId()},
                          {new_tensors.weights_sum_i, tensor_sum_i->GetId()}};
  };

  return LlmTensorLoader::QuantizedWeights{
      .weights = tensor.get(),
      .weights_sum_i = tensor_sum_i.get(),
  };
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
PlaceholderTensorLoader::LoadInt4Weights(
    const std::string& tensor_name,
    const ::ml_drift::WeightsDescription& weights_desc,
    const ::ml_drift::OHWI& shape, bool swap_dims,
    std::optional<MatformOptions> matform_options) {
  ReloadGroupFunctions* tensor_reload_fns;
  if (!placeholder_mode_ ||
      (tensor_reload_fns = GetGroupReloadFunctions(tensor_name)) == nullptr) {
    return loader_.LoadInt4Weights(tensor_name, weights_desc, shape, swap_dims,
                                   matform_options);
  }

  ASSIGN_OR_RETURN(ml_drift::TensorDescriptor desc_weights,
                   GetWeightTensorDescriptor(weights_desc, shape));
  std::shared_ptr<PlaceholderTensor> tensor =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(
          std::move(desc_weights));

  ml_drift::TensorDescriptor desc_weights_sum_i = GetLinearTensorDescriptor(
      loader_.GetGpuInfo(), ml_drift::DataType::kInt32, shape.o);
  std::shared_ptr<PlaceholderTensor> tensor_sum_i =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(
          std::move(desc_weights_sum_i));

  (*tensor_reload_fns)[tensor_name] =
      [tensor_name, weights_desc, shape, swap_dims, tensor, tensor_sum_i,
       matform_options](
          LlmTensorLoader* loader) -> absl::StatusOr<TensorsWithIds> {
    RET_CHECK(tensor);
    RET_CHECK(tensor_sum_i);
    ASSIGN_OR_RETURN(LlmTensorLoader::QuantizedWeights new_tensors,
                     loader->LoadInt4Weights(tensor_name, weights_desc, shape,
                                             swap_dims, matform_options));
    return TensorsWithIds{{new_tensors.weights, tensor->GetId()},
                          {new_tensors.weights_sum_i, tensor_sum_i->GetId()}};
  };

  return LlmTensorLoader::QuantizedWeights{
      .weights = tensor.get(),
      .weights_sum_i = tensor_sum_i.get(),
  };
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
PlaceholderTensorLoader::LoadInt2Weights(
    const std::string& tensor_name,
    const ::ml_drift::WeightsDescription& weights_desc,
    const ::ml_drift::OHWI& shape, bool swap_dims) {
  ReloadGroupFunctions* tensor_reload_fns;
  if (!placeholder_mode_ ||
      (tensor_reload_fns = GetGroupReloadFunctions(tensor_name)) == nullptr) {
    return loader_.LoadInt2Weights(tensor_name, weights_desc, shape, swap_dims);
  }

  ASSIGN_OR_RETURN(ml_drift::TensorDescriptor desc_weights,
                   GetWeightTensorDescriptor(weights_desc, shape));
  std::shared_ptr<PlaceholderTensor> tensor =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(
          std::move(desc_weights));

  ml_drift::TensorDescriptor desc_weights_sum_i = GetLinearTensorDescriptor(
      loader_.GetGpuInfo(), ml_drift::DataType::kInt32, shape.o);
  std::shared_ptr<PlaceholderTensor> tensor_sum_i =
      std::make_shared<odml::infra::gpu::PlaceholderTensor>(
          std::move(desc_weights_sum_i));

  (*tensor_reload_fns)[tensor_name] =
      [tensor_name, weights_desc, shape, swap_dims, tensor, tensor_sum_i](
          LlmTensorLoader* loader) -> absl::StatusOr<TensorsWithIds> {
    RET_CHECK(tensor);
    RET_CHECK(tensor_sum_i);
    ASSIGN_OR_RETURN(
        LlmTensorLoader::QuantizedWeights new_tensors,
        loader->LoadInt2Weights(tensor_name, weights_desc, shape, swap_dims));
    return TensorsWithIds{{new_tensors.weights, tensor->GetId()},
                          {new_tensors.weights_sum_i, tensor_sum_i->GetId()}};
  };

  return LlmTensorLoader::QuantizedWeights{
      .weights = tensor.get(),
      .weights_sum_i = tensor_sum_i.get(),
  };
}

void PlaceholderTensorLoader::SetCreateTensorFn(CreateTensorFn create_tensor) {
  loader_.SetCreateTensorFn(std::move(create_tensor));
}

void PlaceholderTensorLoader::SetGpuInfo(const ::ml_drift::GpuInfo& gpu_info) {
  loader_.SetGpuInfo(gpu_info);
}

void PlaceholderTensorLoader::SetDataType(::ml_drift::DataType type) {
  loader_.SetDataType(type);
}

std::unique_ptr<llm_utils::DataHolder<uint8_t>>
PlaceholderTensorLoader::GetOwnedData(const uint8_t* location) {
  return loader_.GetOwnedData(location);
}

std::optional<uint64_t> PlaceholderTensorLoader::GetMaxBufferSize() const {
  return loader_.GetMaxBufferSize();
}

const ::ml_drift::GpuInfo& PlaceholderTensorLoader::GetGpuInfo() const {
  return loader_.GetGpuInfo();
}

::ml_drift::DataType PlaceholderTensorLoader::GetDataType() const {
  return loader_.GetDataType();
}

}  // namespace odml::infra::gpu
