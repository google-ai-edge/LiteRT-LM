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

#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/gpu_optimized_tensor_loader.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
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
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/basic_gpu_tensor.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/llm_file_metadata.pb.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::gpu {
namespace {

using ::litert::lm::MemoryMappedFile;
using ::ml_drift::Axis;
using ::ml_drift::DataType;
using ::ml_drift::DivideRoundUp;
using ::ml_drift::GetTotalElementsCountForLayout;
using ::ml_drift::GpuSpatialTensor;
using ::ml_drift::TensorDescriptor;
using ::odml::infra::llm_utils::ModelData;

bool ContainedInSpan(absl::Span<uint8_t> data, const uint8_t* ptr) {
  return ptr >= data.data() && ptr < data.data() + data.size();
}

// A TensorDescriptor which stores a pointer to the backing data instead of
// copying it into a vector.
class UnownedDataTensorDescriptor : public TensorDescriptor {
 public:
  UnownedDataTensorDescriptor(const TensorDescriptor& desc,
                              absl::Span<const uint8_t> data)
      : TensorDescriptor(desc), unowned_data_(data) {}

  absl::Span<const uint8_t> GetData() const override { return unowned_data_; }

 private:
  absl::Span<const uint8_t> unowned_data_;
};

// A DataHolder that owns data in a std::vector.
class VectorDataHolder : public llm_utils::DataHolder<uint8_t> {
 public:
  explicit VectorDataHolder(std::vector<uint8_t> data)
      : data_(std::move(data)) {}
  ~VectorDataHolder() override = default;

  absl::Span<uint8_t> GetData() const override { return absl::MakeSpan(data_); }

 public:
  mutable std::vector<uint8_t> data_;
};

// Copies `data` to a new vector. The new data will be scaled based on
// `matform_factor`, and copied in chunks of `chunk_size`.
std::vector<uint8_t> CopyChunks(absl::Span<const uint8_t> data,
                                int matform_factor, int chunk_size) {
  std::vector<uint8_t> res;
  res.resize(data.size() / matform_factor);
  for (int i = 0; i < res.size(); i += chunk_size) {
    memcpy(res.data() + i, data.data(), chunk_size);
    data.remove_prefix(chunk_size * matform_factor);
  }
  return res;
}

// Calculates the sum_i tensor using similar logic to
// RearrangeWeightsUInt4Packed().
absl::StatusOr<std::vector<int32_t>> CalculateSumPacked(
    absl::Span<const uint8_t> src, const ml_drift::OHWI& shape,
    const ml_drift::WeightsDescription& dst_weight_desc) {
  std::vector<int32_t> sum_i(shape.o);
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  const int dst_slices = DivideRoundUp(shape.o, 4);
  const int src_slices = DivideRoundUp(shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, dst_group_size);
  RET_CHECK(dst_weight_desc.layout ==
                ml_drift::WeightsLayout::kOSpatialIOGroupI4O4 ||
            (dst_weight_desc.layout ==
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4 &&
             dst_weight_desc.GetOutputGroupSize() == dst_slices));
  int idx = 0;
  for (int d = 0; d < dst_groups; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int d_group = 0; d_group < dst_group_size; ++d_group) {
        const int d_ch = 4 * (d * dst_group_size + d_group);
        for (int s_ch = 4 * s; s_ch < 4 * s + 4; ++s_ch) {
          uint8_t val1 = src[idx++];
          uint8_t val2 = src[idx++];
          sum_i[d_ch + 0] += (val1 & 0xf) - 8;
          sum_i[d_ch + 1] += ((val1 >> 4) & 0xf) - 8;
          sum_i[d_ch + 2] += (val2 & 0xf) - 8;
          sum_i[d_ch + 3] += ((val2 >> 4) & 0xf) - 8;
        }
      }
    }
  }
  return sum_i;
}

// Converts a span of float16 data to a vector of float32 data.
void ConvertFp16ToFp32(absl::Span<const uint8_t> data,
                         std::vector<float>& tensor_data) {
  const ml_drift::half* src =
      reinterpret_cast<const ml_drift::half*>(data.data());
  int count = data.size() / sizeof(uint16_t);
  tensor_data.resize(count);
  for (int i = 0; i < count; ++i) {
    tensor_data[i] = static_cast<float>(src[i]);
  }
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<GpuOptimizedTensorLoader>>
GpuOptimizedTensorLoader::CreateLoRA(std::shared_ptr<ModelData> lora_data,
                                     int& lora_rank) {
  std::optional<int> model_rank = lora_data->LoRARank();
  RET_CHECK(model_rank) << "Metadata must have lora_rank.";
  lora_rank = *model_rank;
  return std::make_unique<GpuOptimizedTensorLoader>(nullptr,
                                                    std::move(lora_data));
}

GpuOptimizedTensorLoader::GpuOptimizedTensorLoader(
    std::shared_ptr<llm_utils::ModelData> model_data,
    std::shared_ptr<llm_utils::ModelData> lora_data, bool own_model_data)
    : model_data_(std::move(model_data)),
      lora_data_(std::move(lora_data)),
      own_model_data_(own_model_data) {}

GpuOptimizedTensorLoader::~GpuOptimizedTensorLoader() {
  // TODO: Consider to emove model_data->Clear() from GpuOptimizedTensorLoader's
  // destructor, as it's taking a shared_ptr and not taking the ownership of
  // model_data.
  if (own_model_data_ && model_data_) {
    model_data_->Clear();
  }
  // Notice: The de-construction of `GpuOptimizedTensorLoader` should not
  // release the `lora_data_` because `GpuOptimizedTensorLoader` only owns a
  // shared pointer to the `lora_data_`. If the `lora_data_` is used after
  // `GpuOptimizedTensorLoader`'s de-construction, it may cause an issue.
  if (lora_data_) {
    lora_data_->Clear();
  }
}

std::vector<float> GpuOptimizedTensorLoader::LoadFloat32(
    const std::string& tensor_name, int count) {
  // The data is copied out, so the raw holder is not retained.
  auto holder = ReadTemporaryData(tensor_name);
  if (!holder.ok() || *holder == nullptr) {
    return {};
  }
  absl::Span<const uint8_t> data = (*holder)->GetData();
  if (data.size() != count * sizeof(float)) {
    return {};
  }
  const float* fdata = reinterpret_cast<const float*>(data.data());
  return std::vector<float>(fdata, fdata + count);
}

std::vector<int32_t> GpuOptimizedTensorLoader::LoadInt32(
    const std::string& tensor_name, int count) {
  // The data is copied out, so the raw holder is not retained.
  auto holder = ReadTemporaryData(tensor_name);
  if (!holder.ok() || *holder == nullptr) {
    return {};
  }
  absl::Span<const uint8_t> data = (*holder)->GetData();
  if (data.size() != count * sizeof(int32_t)) {
    return {};
  }
  const int32_t* idata = reinterpret_cast<const int32_t*>(data.data());
  return std::vector<int32_t>(idata, idata + count);
}

int GpuOptimizedTensorLoader::GetTensorElementSizeInBits(
    const std::string& tensor_name, int count) {
  return (GetModelData(tensor_name)->GetTensorSize(tensor_name) * 8) / count;
}

bool GpuOptimizedTensorLoader::HasTensor(const std::string& tensor_name) {
  return GetModelData(tensor_name)->GetTensorSize(tensor_name) > 0;
}

absl::StatusOr<GpuSpatialTensor*> GpuOptimizedTensorLoader::LoadScale(
    const std::string& tensor_name, const ::ml_drift::OHWI& shape,
    std::optional<MatformOptions> matform_options) {
  // Linear scales if 1D or if matform options are being used.
  if ((shape.h == 1) && (shape.i == 1 || matform_options)) {
    return CreateLinearTensor(shape.o, tensor_name, /*optional=*/false,
                              matform_options);
  }

  // 2D scale for blockwise q4
  int count = shape.o * shape.h * shape.i;
  ml_drift::Tensor<ml_drift::OHWI, DataType::kFloat32> ohwi_scale;
  // We try to load float32 first.
  ohwi_scale.data = LoadFloat32(tensor_name, count);
  // If the data is empty, we try to load float16 and convert it to float32.
  if (ohwi_scale.data.empty()) {
    auto holder = ReadTemporaryData(tensor_name);
    if (holder.ok() && *holder != nullptr &&
        (*holder)->GetData().size() == count * sizeof(uint16_t)) {
      ConvertFp16ToFp32((*holder)->GetData(), ohwi_scale.data);
    }
  }
  ohwi_scale.shape = shape;
  if (ohwi_scale.data.size() != count) {
    return absl::InternalError(
        absl::StrCat("Tensor size does not match: ", tensor_name));
  }
  auto desc = ScaleOrZeroPointToTensorDesc(gpu_info_, ohwi_scale, data_type_);
  return create_tensor_(desc);
}

absl::StatusOr<GpuSpatialTensor*> GpuOptimizedTensorLoader::LoadZeroPoint(
    const std::string& zero_point_name, const std::string& scale_name,
    const ::ml_drift::OHWI& shape) {
  return CreateLinearTensor(shape.o * shape.h, zero_point_name,
                            /*optional=*/true);
}

absl::StatusOr<GpuSpatialTensor*> GpuOptimizedTensorLoader::LoadBias(
    const std::string& tensor_name, int count) {
  return CreateLinearTensor(count, tensor_name, /*optional=*/true);
}

absl::StatusOr<GpuSpatialTensor*> GpuOptimizedTensorLoader::LoadWeights(
    const std::string& tensor_name,
    const ml_drift::WeightsDescription& weights_desc,
    const ml_drift::OHWI& shape, bool swap_dims) {
  return CreateWeightsTensor(weights_desc, shape, tensor_name,
                             /*matform_options=*/std::nullopt);
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
GpuOptimizedTensorLoader::LoadInt8Weights(
    const std::string& tensor_name,
    const ml_drift::WeightsDescription& weights_desc,
    const ml_drift::OHWI& shape, bool swap_dims) {
  QuantizedWeights int8_weights;
  ASSIGN_OR_RETURN(int8_weights.weights,
                   CreateWeightsTensor(weights_desc, shape, tensor_name,
                                       /*matform_options=*/std::nullopt));
  const std::string weights_sum_i_name = tensor_name + ".sum_i";
  if (HasTensor(weights_sum_i_name)) {
    ASSIGN_OR_RETURN(int8_weights.weights_sum_i,
                     CreateLinearInt32Tensor(shape.o, weights_sum_i_name));
  } else {
    int8_weights.weights_sum_i = nullptr;
  }
  return int8_weights;
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
GpuOptimizedTensorLoader::LoadInt4Weights(
    const std::string& tensor_name,
    const ml_drift::WeightsDescription& weights_desc,
    const ml_drift::OHWI& shape, bool swap_dims,
    std::optional<MatformOptions> matform_options) {
  QuantizedWeights int4_weights;
  ml_drift::Tensor<ml_drift::Linear, DataType::kInt32> matform_sum_i;
  ASSIGN_OR_RETURN(int4_weights.weights,
                   CreateWeightsTensor(weights_desc, shape, tensor_name,
                                       matform_options, &matform_sum_i.data));
  const std::string weights_sum_i_name = tensor_name + ".sum_i";
  if (HasTensor(weights_sum_i_name)) {
    if (!matform_sum_i.data.empty()) {
      matform_sum_i.shape = ml_drift::Linear(shape.o);
      ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                       create_tensor_(CreateConstantLinearTensorDescriptor(
                           gpu_info_, matform_sum_i)));
    } else {
      ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                       CreateLinearInt32Tensor(shape.o, weights_sum_i_name,
                                               matform_options));
    }
  } else {
    int4_weights.weights_sum_i = nullptr;
  }
  return int4_weights;
}

absl::StatusOr<LlmTensorLoader::QuantizedWeights>
GpuOptimizedTensorLoader::LoadInt2Weights(
    const std::string& tensor_name,
    const ml_drift::WeightsDescription& weights_desc,
    const ml_drift::OHWI& shape, bool swap_dims) {
  QuantizedWeights int2_weights;
  ASSIGN_OR_RETURN(int2_weights.weights,
                   CreateWeightsTensor(weights_desc, shape, tensor_name,
                                       /*matform_options=*/std::nullopt));
  const std::string weights_sum_i_name = tensor_name + ".sum_i";
  if (HasTensor(weights_sum_i_name)) {
    ASSIGN_OR_RETURN(int2_weights.weights_sum_i,
                     CreateLinearInt32Tensor(shape.o, weights_sum_i_name));
  } else {
    int2_weights.weights_sum_i = nullptr;
  }
  return int2_weights;
}

absl::StatusOr<GpuSpatialTensor*> GpuOptimizedTensorLoader::CreateWeightsTensor(
    const ml_drift::WeightsDescription& weights_desc,
    const ml_drift::OHWI& shape, const std::string& name,
    std::optional<MatformOptions> matform_options,
    std::vector<int32_t>* matform_sum_i) {
  DataType data_type = weights_desc.type;
  int count = shape.DimensionsProduct();
  int matform_element_size = 4;
  if (data_type == DataType::kUint4) {
    count /= 2;
    matform_element_size /= 2;
  } else if (data_type == DataType::kUint2) {
    count /= 4;
    matform_element_size /= 4;
  } else if (data_type == DataType::kFloat16) {
    count = GetTotalElementsCountForLayout(weights_desc, shape);
    matform_element_size *= 2;
  }
  ASSIGN_OR_RETURN(auto data, GetData(name));
  int size = count * ml_drift::SizeOf(data_type);
  if (matform_options) {
    switch (matform_options->axis) {
      case Axis::kOutputChannels: {
        // If the tensor is being matformed on the output axis, the data needs
        // to be copied by chunks.
        // TODO: b/422747873 - Format all matformed tensors to avoid copies.
        auto holder = std::make_unique<VectorDataHolder>(
            CopyChunks(data, matform_options->matform_factor,
                       shape.o * matform_element_size));
        data = holder->GetData();
        // Add the data to `tensor_weights_`, which allows for optimized weight
        // uploads.
        tensor_weights_[name] = std::move(holder);
        break;
      }
      case Axis::kInputChannels:
        data = data.first(data.size() / matform_options->matform_factor);
        if (matform_sum_i) {
          // TODO: b/422747873 - This is slow, the matformed sum_i should be
          // part of the optimized model.
          ASSIGN_OR_RETURN(*matform_sum_i,
                           CalculateSumPacked(data, shape, weights_desc));
        }
        break;
      default:
        return absl::InternalError(
            absl::StrCat("Unsupported matform axis: ", matform_options->axis));
    }
  }
  // If LoRA tensors are being loaded without |lora_weights_|, we load as all
  // zero tensors. LoRA weights can be loaded later and copied into these
  // tensors.
  bool allow_empty = IsLoRATensorName(name) && data.empty() &&
                     (!lora_data_ || IsFeedForwardTensorName(name));
  std::vector<float> converted_data;
  if (data_type == DataType::kFloat32 &&
      data.size() == count * sizeof(uint16_t)) {
    // If the data is half precision but the target is float, convert it.
    converted_data.resize(count);
    const ml_drift::half* src =
        reinterpret_cast<const ml_drift::half*>(data.data());
    for (int i = 0; i < count; ++i) {
      converted_data[i] = static_cast<float>(src[i]);
    }
    data = absl::MakeSpan(
        reinterpret_cast<const uint8_t*>(converted_data.data()),
        converted_data.size() * sizeof(float));
    // The half precision source is no longer needed. Free it now rather than
    // when the loader is destroyed.
    tensor_weights_.erase(name);
  }
  if (!allow_empty && data.size() != size) {
    if (size == data.size() * 2 &&
        weights_desc.type == ml_drift::DataType::kFloat32) {
      // If we are forcing f32, the model might still be f16. In this case we
      // need to convert the weights.
      auto new_data =
          std::make_unique<VectorDataHolder>(std::vector<uint8_t>(size));
      const ml_drift::half* src =
          reinterpret_cast<const ml_drift::half*>(data.data());
      float* dst = reinterpret_cast<float*>(new_data->data_.data());
      for (int i = 0; i < data.size() / 2; ++i) {
        dst[i] = static_cast<float>(src[i]);
      }
      data = new_data->GetData();
      tensor_weights_[name] = std::move(new_data);
    } else {
      return absl::InternalError(absl::StrCat("Tensor size does not match: ",
                                              name, ". Expected ", size,
                                              " but got ", data.size()));
    }
  }

  if (weights_desc.IsLinearLayout()) {
    TensorDescriptor tensor_desc(data_type,
                                 ml_drift::TensorStorageType::kBuffer,
                                 ml_drift::Layout::kLinear);
    tensor_desc.SetBHWDCShape(ml_drift::BHWDC(1, 1, 1, 1, count));
    if (allow_empty) {
      if (lora_data_) {
        // If this is a LoRA loader a real tensor is needed because this will be
        // used during inference.
        tensor_desc.SetData(std::vector<uint8_t>(size));
        return create_tensor_(tensor_desc);
      }
      return placeholder_tensors_
          .emplace_back(new BasicGpuSpatialTensor(tensor_desc))
          .get();
    }
    UnownedDataTensorDescriptor unowned_desc(tensor_desc, data);
    return create_tensor_(unowned_desc);
  } else {
    if (data_type != DataType::kUint8 && data_type != DataType::kUint4 &&
        data_type != DataType::kUint2) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported data type ", ml_drift::ToString(data_type),
                       " for tensor ", name));
    }
    DataType texture_type = DataType::kUint32;
    if (data_type == DataType::kUint4) {
      texture_type = DataType::kUint16;
    } else if (data_type == DataType::kUint2) {
      texture_type = DataType::kUint8;
    }
    TensorDescriptor tensor_desc(texture_type,
                                 ml_drift::TensorStorageType::kTexture2D,
                                 ml_drift::Layout::kHW);
    ml_drift::uint2 tex_size = ml_drift::Get2dResourceSize(weights_desc, shape);
    // Values are packed: 4 x uint2 = uint8, 4 x uint4 = uint16, 4 x uint8 =
    // uint32.
    tex_size.x /= 4;
    tensor_desc.SetBHWDCShape(ml_drift::BHWDC(1, tex_size.y, tex_size.x, 1, 4));
    if (allow_empty) {
      if (lora_data_) {
        // If this is a LoRA loader a real tensor is needed because this will be
        // used during inference.
        tensor_desc.SetData(std::vector<uint8_t>(size));
        return create_tensor_(tensor_desc);
      }
      return placeholder_tensors_
          .emplace_back(new BasicGpuSpatialTensor(tensor_desc))
          .get();
    }
    UnownedDataTensorDescriptor unowned_desc(tensor_desc, data);
    return create_tensor_(unowned_desc);
  }
}

absl::StatusOr<GpuSpatialTensor*> GpuOptimizedTensorLoader::CreateLinearTensor(
    int count, const std::string& name, bool optional,
    std::optional<MatformOptions> matform_options) {
  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> tensor;
  if (matform_options && matform_options->axis == Axis::kOutputChannels) {
    tensor.data = LoadFloat32(name, count * matform_options->matform_factor);
    if (!tensor.data.empty()) {
      tensor.data.resize(count);
    }
  } else {
    // We try to load float32 first.
    tensor.data = LoadFloat32(name, count);
    // If the data is empty, we try to load float16 and convert it to float32.
    if (tensor.data.empty()) {
      auto holder = ReadTemporaryData(name);
      if (holder.ok() && *holder != nullptr &&
          (*holder)->GetData().size() == count * sizeof(uint16_t)) {
        ConvertFp16ToFp32((*holder)->GetData(), tensor.data);
      }
    }
  }
  if (!HasTensor(name)) {
    if (!optional) {
      return absl::InternalError(
          absl::StrCat("Missing required tensor: ", name));
    }
    tensor.data.resize(count * ml_drift::SizeOf(data_type_));
  } else if (tensor.data.size() != count) {
    return absl::InternalError(
        absl::StrCat("Tensor size does not match: ", name));
  }
  tensor.shape = ml_drift::Linear(count);
  return create_tensor_(
      CreateConstantLinearTensorDescriptor(gpu_info_, data_type_, tensor));
}

absl::StatusOr<GpuSpatialTensor*>
GpuOptimizedTensorLoader::CreateLinearInt32Tensor(
    int count, const std::string& name,
    std::optional<MatformOptions> matform_options) {
  if (!HasTensor(name)) {
    return absl::InternalError(absl::StrCat("Missing tensor: ", name));
  }
  ml_drift::Tensor<ml_drift::Linear, DataType::kInt32> tensor;
  if (matform_options && matform_options->axis == Axis::kOutputChannels) {
    tensor.data = LoadInt32(name, count * matform_options->matform_factor);
    if (!tensor.data.empty()) {
      tensor.data.resize(count);
    }
  } else {
    tensor.data = LoadInt32(name, count);
  }
  if (tensor.data.size() != count) {
    return absl::InternalError(
        absl::StrCat("Tensor size does not match: ", name));
  }
  tensor.shape = ml_drift::Linear(count);
  return create_tensor_(
      CreateConstantLinearTensorDescriptor(gpu_info_, tensor));
}

GpuOptimizedTensorLoader::Weights GpuOptimizedTensorLoader::GetOwnedData(
    const uint8_t* location) {
  for (auto& kv : tensor_weights_) {
    if (ContainedInSpan(kv.second->GetData(), location)) {
      auto weights = std::move(kv.second);
      tensor_weights_.erase(kv.first);
      return weights;
    }
  }
  return nullptr;
}

absl::StatusOr<std::unique_ptr<llm_utils::DataHolder<uint8_t>>>
GpuOptimizedTensorLoader::LoadRawTensorData(const std::string& name) {
  return GetModelData(name)->ReadTensor(name);
}

std::optional<uint64_t> GpuOptimizedTensorLoader::GetMaxBufferSize() const {
  // Add 2 * the mmap alignment to be safe in case data is not aligned.
  return model_data_->GetMaxTensorSize() +
         2 * MemoryMappedFile::GetOffsetAlignment();
}

absl::StatusOr<GpuOptimizedTensorLoader::Weights>
GpuOptimizedTensorLoader::ReadTemporaryData(const std::string& tensor_name) {
  return GetModelData(tensor_name)->ReadTensor(tensor_name);
}

absl::StatusOr<absl::Span<const uint8_t>> GpuOptimizedTensorLoader::GetData(
    const std::string& tensor_name) {
  ASSIGN_OR_RETURN(auto weights,
                   GetModelData(tensor_name)->ReadTensor(tensor_name));
  if (!weights) {
    return absl::Span<const uint8_t>();
  }
  auto* weights_ptr = weights.get();
  tensor_weights_[tensor_name] = std::move(weights);
  return weights_ptr->GetData();
}

llm_utils::ModelData* GpuOptimizedTensorLoader::GetModelData(
    absl::string_view tensor_name) {
  if (IsLoRATensorName(tensor_name) && lora_data_) {
    return lora_data_.get();
  }
  return model_data_.get();
}

llm_utils::ModelData const* GpuOptimizedTensorLoader::GetModelData(
    absl::string_view tensor_name) const {
  if (IsLoRATensorName(tensor_name) && lora_data_) {
    return lora_data_.get();
  }
  return model_data_.get();
}

int GpuOptimizedTensorLoader::GetQuantizationIGroupSize(
    const std::string& weights_name,
    const ml_drift::OHWI& weights_shape) const {
  const std::string base_name = weights_name.substr(0, weights_name.rfind('.'));
  const std::string scale_name = base_name + ".w_quantized_scale";
  size_t scale_size =
      GetModelData(scale_name)->GetTensorSize(scale_name);

  if (scale_size == (weights_shape.o * weights_shape.h * 4)) {
    return weights_shape.i;
  } else {
    size_t num_blocks = scale_size / (weights_shape.o * weights_shape.h * 4);
    size_t block_size = weights_shape.i / num_blocks;
    return block_size;
  }
}

}  // namespace odml::infra::gpu
