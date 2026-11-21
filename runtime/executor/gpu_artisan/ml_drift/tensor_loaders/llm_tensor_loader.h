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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_TENSOR_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_TENSOR_LOADER_H_

#include <cstdint>
#include <functional>
#include <map>
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
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/name_check.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "support/util/memory_mapped_file.h"

namespace odml::infra::gpu {

// When a float tensor contains values that are subnormal in F16 (and would be
// flushed to zero on some GPUs), the converter multiplies the tensor by a power
// of two k and stores 1/k as a scalar float32 tensor named
// `<tensor_name><kF16RescaleInverseSuffix>`. The engine must multiply by this
// value to undo the rescaling.
inline constexpr char kF16RescaleInverseSuffix[] = ".f16_rescale_inv";

// Interface for loading LLM weights, bias, and quantization parameters as
// tensors suited for GPU processing.
class LlmTensorLoader {
 public:
  struct QuantizedWeights {
    ml_drift::GpuSpatialTensor* weights = nullptr;
    ml_drift::GpuSpatialTensor* weights_sum_i = nullptr;
  };
  // Returns a backend-specific tensor based on a ::ml_drift::TensorDescriptor.
  using CreateTensorFn =
      std::function<absl::StatusOr<::ml_drift::GpuSpatialTensor*>(
          const ::ml_drift::TensorDescriptor&)>;

  struct MatformOptions {
    ml_drift::Axis axis;
    int matform_factor;
  };

  LlmTensorLoader() = default;
  virtual ~LlmTensorLoader() = default;
  virtual std::vector<float> LoadFloat32(const std::string& tensor_name,
                                         int count) = 0;
  virtual int GetTensorElementSizeInBits(const std::string& tensor_name,
                                         int count) = 0;
  ml_drift::DataType GetDataTypeForWeights(const std::string& name, int count);
  virtual bool HasTensor(const std::string& tensor_name) = 0;

  virtual int GetQuantizationIGroupSize(
      const std::string& weights_name,
      const ml_drift::OHWI& weights_shape) const {
    return weights_shape.i;
  }
  virtual absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadScale(
      const std::string& tensor_name, const ::ml_drift::OHWI& shape,
      std::optional<MatformOptions> matform_options) = 0;
  virtual absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadZeroPoint(
      const std::string& zero_point_name, const std::string& scale_name,
      const ::ml_drift::OHWI& shape) = 0;
  virtual absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadBias(
      const std::string& tensor_name, int count) = 0;
  virtual absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadWeights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) {
    return absl::UnimplementedError("LoadWeights is unimplemented!");
  }
  virtual absl::StatusOr<QuantizedWeights> LoadInt8Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) = 0;
  virtual absl::StatusOr<QuantizedWeights> LoadInt4Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims,
      std::optional<MatformOptions> matform_options) = 0;
  virtual absl::StatusOr<QuantizedWeights> LoadInt2Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) = 0;

  virtual void SetCreateTensorFn(CreateTensorFn create_tensor) {
    create_tensor_ = std::move(create_tensor);
  }
  virtual void SetGpuInfo(const ::ml_drift::GpuInfo& gpu_info) {
    gpu_info_ = gpu_info;
  }
  virtual void SetDataType(::ml_drift::DataType type) { data_type_ = type; }
  virtual const ::ml_drift::GpuInfo& GetGpuInfo() const { return gpu_info_; }
  virtual ::ml_drift::DataType GetDataType() const { return data_type_; }

  // Returns a span containing data owned by this loader.
  virtual std::unique_ptr<llm_utils::DataHolder<uint8_t>> GetOwnedData(
      const uint8_t* location) {
    return nullptr;
  }

  virtual absl::StatusOr<std::unique_ptr<llm_utils::DataHolder<uint8_t>>>
  LoadRawTensorData(const std::string& name) {
    return absl::UnimplementedError("Not implemented");
  }

  // Gets the maximum buffer size this loader will need.
  virtual std::optional<uint64_t> GetMaxBufferSize() const {
    return std::nullopt;
  }

  virtual void SetMemoryMappedFileAccess(
      ::litert::support::MemoryMappedFile::Access access) {}

  // Wraps |loader| in a LlmTensorLoader which caches tensors. This ensures the
  // same pointer will be returned from calls with matching tensor names, and
  // the tensor will not be recreated.
  static std::unique_ptr<LlmTensorLoader> MakeCaching(
      std::unique_ptr<LlmTensorLoader> loader);

 protected:
  CreateTensorFn create_tensor_;
  ::ml_drift::GpuInfo gpu_info_;
  ::ml_drift::DataType data_type_;
};

// A wrapper that caches tensors loaded by an `LlmTensorLoader`.
class CachingTensorLoader : public LlmTensorLoader {
 public:
  explicit CachingTensorLoader(std::unique_ptr<LlmTensorLoader> loader)
      : loader_(std::move(loader)) {}
  ~CachingTensorLoader() override = default;

  std::vector<float> LoadFloat32(const std::string& tensor_name,
                                 int count) override {
    std::vector<float>& cached = float_cache_[tensor_name];
    if (cached.empty()) {
      cached = loader_->LoadFloat32(tensor_name, count);
    }
    return cached;
  }

  int GetTensorElementSizeInBits(const std::string& tensor_name,
                                 int count) override {
    return loader_->GetTensorElementSizeInBits(tensor_name, count);
  }

  bool HasTensor(const std::string& tensor_name) override {
    return loader_->HasTensor(tensor_name);
  }

  int GetQuantizationIGroupSize(
      const std::string& weights_name,
      const ml_drift::OHWI& weights_shape) const override {
    auto it = group_size_cache_.find(weights_name);
    if (it != group_size_cache_.end()) {
      return it->second;
    }
    int group_size =
        loader_->GetQuantizationIGroupSize(weights_name, weights_shape);
    group_size_cache_[weights_name] = group_size;
    return group_size;
  }

  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadScale(
      const std::string& tensor_name, const ::ml_drift::OHWI& shape,
      std::optional<MatformOptions> matform_options) override {
    std::string cache_name = GetName(tensor_name, shape);
    if (auto it = cache_.find(cache_name); it != cache_.end()) {
      return it->second;
    }
    return CacheResult(loader_->LoadScale(tensor_name, shape, matform_options),
                       cache_name);
  }

  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadZeroPoint(
      const std::string& zero_point_name, const std::string& scale_name,
      const ::ml_drift::OHWI& shape) override {
    std::string name = zero_point_name;
    // These tensors can be optional and will be filled with empty data. We can
    // cache the empty tensors with the same name to avoid multiple copies.
    if (!loader_->HasTensor(name)) {
      name = "<empty_linear>";
    }
    std::string cache_name = GetName(name, shape);
    if (auto it = cache_.find(cache_name); it != cache_.end()) {
      return it->second;
    }
    return CacheResult(loader_->LoadZeroPoint(name, scale_name, shape),
                       cache_name);
  }

  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadBias(
      const std::string& tensor_name, int count) override {
    std::string name = tensor_name;
    // These tensors can be optional and will be filled with empty data. We can
    // cache the empty tensors with the same name to avoid multiple copies.
    if (!loader_->HasTensor(name)) {
      name = "<empty_linear>";
    }
    std::string cache_name = GetName(name, ::ml_drift::OHWI(count, 1, 1, 1));
    if (auto it = cache_.find(cache_name); it != cache_.end()) {
      return it->second;
    }
    return CacheResult(loader_->LoadBias(name, count), cache_name);
  }

  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadWeights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) override {
    std::string cache_name = GetName(tensor_name, shape);
    if (IsLoRATensorName(tensor_name) && IsFeedForwardTensorName(tensor_name) &&
        !loader_->HasTensor(tensor_name)) {
      // Share cache for empty optional LoRA weights to save memory.
      cache_name = GetName("<empty_lora_weights>", shape);
    }
    if (auto it = cache_.find(cache_name); it != cache_.end()) {
      return it->second;
    }
    return CacheResult(
        loader_->LoadWeights(tensor_name, weights_desc, shape, swap_dims),
        cache_name);
  }

  absl::StatusOr<QuantizedWeights> LoadInt8Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) override {
    std::string weights_name = GetName(tensor_name, shape);
    std::string weights_sum_i_name = weights_name + "_sum_i";
    if (auto it = cache_.find(weights_name); it != cache_.end()) {
      QuantizedWeights weights;
      weights.weights = it->second;
      if (auto it_sum = cache_.find(weights_sum_i_name);
          it_sum != cache_.end()) {
        weights.weights_sum_i = it_sum->second;
      }
      return weights;
    }
    ASSIGN_OR_RETURN(
        auto in_weights,
        loader_->LoadInt8Weights(tensor_name, weights_desc, shape, swap_dims));
    QuantizedWeights out_weights;
    ASSIGN_OR_RETURN(out_weights.weights,
                     CacheResult(in_weights.weights, weights_name));
    if (in_weights.weights_sum_i) {
      ASSIGN_OR_RETURN(
          out_weights.weights_sum_i,
          CacheResult(in_weights.weights_sum_i, weights_sum_i_name));
    }
    return out_weights;
  }

  absl::StatusOr<QuantizedWeights> LoadInt4Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims,
      std::optional<MatformOptions> matform_options) override {
    std::string weights_name = GetName(tensor_name, shape);
    std::string weights_sum_i_name = weights_name + "_sum_i";
    if (auto it = cache_.find(weights_name); it != cache_.end()) {
      QuantizedWeights weights;
      weights.weights = it->second;
      if (auto it_sum = cache_.find(weights_sum_i_name);
          it_sum != cache_.end()) {
        weights.weights_sum_i = it_sum->second;
      }
      return weights;
    }
    ASSIGN_OR_RETURN(auto in_weights,
                     loader_->LoadInt4Weights(tensor_name, weights_desc, shape,
                                              swap_dims, matform_options));
    QuantizedWeights out_weights;
    ASSIGN_OR_RETURN(out_weights.weights,
                     CacheResult(in_weights.weights, weights_name));
    if (in_weights.weights_sum_i) {
      ASSIGN_OR_RETURN(
          out_weights.weights_sum_i,
          CacheResult(in_weights.weights_sum_i, weights_sum_i_name));
    }
    return out_weights;
  }

  absl::StatusOr<QuantizedWeights> LoadInt2Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) override {
    std::string weights_name = GetName(tensor_name, shape);
    std::string weights_sum_i_name = weights_name + "_sum_i";
    if (auto it = cache_.find(weights_name); it != cache_.end()) {
      QuantizedWeights weights;
      weights.weights = it->second;
      if (auto it_sum = cache_.find(weights_sum_i_name);
          it_sum != cache_.end()) {
        weights.weights_sum_i = it_sum->second;
      }
      return weights;
    }
    ASSIGN_OR_RETURN(
        auto in_weights,
        loader_->LoadInt2Weights(tensor_name, weights_desc, shape, swap_dims));
    QuantizedWeights out_weights;
    ASSIGN_OR_RETURN(out_weights.weights,
                     CacheResult(in_weights.weights, weights_name));
    if (in_weights.weights_sum_i) {
      ASSIGN_OR_RETURN(
          out_weights.weights_sum_i,
          CacheResult(in_weights.weights_sum_i, weights_sum_i_name));
    }
    return out_weights;
  }

  void SetCreateTensorFn(CreateTensorFn create_tensor) override {
    loader_->SetCreateTensorFn(std::move(create_tensor));
  }

  void SetGpuInfo(const ::ml_drift::GpuInfo& gpu_info) override {
    loader_->SetGpuInfo(gpu_info);
  }
  void SetDataType(::ml_drift::DataType type) override {
    loader_->SetDataType(type);
  }

  std::unique_ptr<llm_utils::DataHolder<uint8_t>> GetOwnedData(
      const uint8_t* location) override {
    return loader_->GetOwnedData(location);
  }

  absl::StatusOr<std::unique_ptr<llm_utils::DataHolder<uint8_t>>>
  LoadRawTensorData(const std::string& name) override {
    return loader_->LoadRawTensorData(name);
  }

  std::optional<uint64_t> GetMaxBufferSize() const override {
    return loader_->GetMaxBufferSize();
  }

  void SetMemoryMappedFileAccess(
      ::litert::support::MemoryMappedFile::Access access) override {
    loader_->SetMemoryMappedFileAccess(access);
  }

 private:
  static std::string GetName(const std::string& name,
                             const ::ml_drift::OHWI& shape) {
    return absl::StrCat(name, ":", shape.o, ",", shape.h, ",", shape.w, ",",
                        shape.i);
  }

  absl::StatusOr<::ml_drift::GpuSpatialTensor*> CacheResult(
      absl::StatusOr<::ml_drift::GpuSpatialTensor*> tensor,
      const std::string& name) {
    if (tensor.ok()) {
      cache_[name] = *tensor;
    }
    return tensor;
  }

  absl::StatusOr<::ml_drift::GpuSpatialTensor*> CacheResult(
      ::ml_drift::GpuSpatialTensor* tensor, const std::string& name) {
    cache_[name] = tensor;
    return tensor;
  }

  std::unique_ptr<LlmTensorLoader> loader_;
  std::map<std::string, ::ml_drift::GpuSpatialTensor*> cache_;
  std::map<std::string, std::vector<float>> float_cache_;
  std::map<std::string, std::shared_ptr<llm_utils::DataHolder<uint8_t>>>
      shared_data_cache_;
  mutable std::map<std::string, int> group_size_cache_;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_TENSOR_LOADER_H_
