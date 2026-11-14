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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_PLACEHOLDER_TENSOR_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_PLACEHOLDER_TENSOR_LOADER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"

namespace odml::infra::gpu {

// A placeholder tensor that is used to represent the tensor that is not loaded
// yet.
class PlaceholderTensor : public ::ml_drift::GpuSpatialTensor {
 public:
  explicit PlaceholderTensor(ml_drift::TensorDescriptor desc) : desc_(desc) {}
  PlaceholderTensor(const PlaceholderTensor& other) = delete;
  PlaceholderTensor& operator=(const PlaceholderTensor& other) = delete;

  ~PlaceholderTensor() override = default;

  int Width() const override { return desc_.GetBHWDCShape().w; }
  int Height() const override { return desc_.GetBHWDCShape().h; }
  int Depth() const override { return desc_.GetBHWDCShape().d; }
  int Channels() const override { return desc_.GetBHWDCShape().c; }
  int Slices() const override {
    return ml_drift::DivideRoundUp(desc_.GetBHWDCShape().c, 4);
  }
  int Batch() const override { return desc_.GetBHWDCShape().b; }
  ml_drift::TensorDescriptor GetDescriptor() const override { return desc_; }
  ml_drift::ValueId GetId() const { return id_; }
  void SetId(const ml_drift::ValueId& id) { id_ = id; }
  // For testing purpose.
  void SetDescriptor(ml_drift::TensorDescriptor desc) {
    desc_ = std::move(desc);
  }

 private:
  ml_drift::TensorDescriptor desc_;
  ml_drift::ValueId id_;
};

using TensorsWithIds =
    std::vector<std::pair<ml_drift::GpuSpatialTensor*, ml_drift::ValueId>>;
// A function to load the tensor with a |tensor_loader| and the original
// parameters
using LoadTensorFn = std::function<absl::StatusOr<TensorsWithIds>(
    LlmTensorLoader* tensor_loader)>;
using ReloadGroupFunctions = absl::flat_hash_map<std::string, LoadTensorFn>;

// A wrapper that create placeholder tensors loaded by an `LlmTensorLoader`.
// The placeholder mode only be supported by the `GpuOptimizedTensorLoader`.
class PlaceholderTensorLoader : public LlmTensorLoader {
 public:
  // The placeholder tensor loader would not own the `loader`, the caller should
  // make sure the `loader` outlives the placeholder tensor loader.
  explicit PlaceholderTensorLoader(LlmTensorLoader& loader) : loader_(loader) {}
  ~PlaceholderTensorLoader() override = default;

  PlaceholderTensorLoader(LlmTensorLoader&& loader) = delete;

  // Enables the placeholder mode, and set the function to generate placeholder
  // tensor.
  void EnablePlaceholderMode() { placeholder_mode_ = true; }

  // Enables the pass through mode.
  void EnablePassThroughMode() { placeholder_mode_ = false; }

  std::vector<float> LoadFloat32(const std::string& tensor_name,
                                 int count) override;
  int GetTensorElementSizeInBits(const std::string& tensor_name,
                                 int count) override;
  bool HasTensor(const std::string& tensor_name) override;
  int GetQuantizationIGroupSize(
      const std::string& weights_name,
      const ml_drift::OHWI& weights_shape) const override;
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadScale(
      const std::string& tensor_name, const ::ml_drift::OHWI& shape,
      std::optional<MatformOptions> matform_options) override;
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadZeroPoint(
      const std::string& zero_point_name, const std::string& scale_name,
      const ::ml_drift::OHWI& shape) override;
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadBias(
      const std::string& tensor_name, int count) override;
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadWeights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) override;
  absl::StatusOr<QuantizedWeights> LoadInt8Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) override;
  absl::StatusOr<QuantizedWeights> LoadInt4Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims,
      std::optional<MatformOptions> matform_options) override;
  absl::StatusOr<QuantizedWeights> LoadInt2Weights(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims) override;
  void SetCreateTensorFn(CreateTensorFn create_tensor) override;
  void SetGpuInfo(const ::ml_drift::GpuInfo& gpu_info) override;
  void SetDataType(::ml_drift::DataType type) override;
  std::unique_ptr<llm_utils::DataHolder<uint8_t>> GetOwnedData(
      const uint8_t* location) override;
  std::optional<uint64_t> GetMaxBufferSize() const override;
  const ::ml_drift::GpuInfo& GetGpuInfo() const override;
  ::ml_drift::DataType GetDataType() const override;

  absl::flat_hash_map<std::string, ReloadGroupFunctions>& GetTensorReloadFns() {
    return reload_group_fns_;
  }
  void SetReloadGroupNames(const std::vector<std::string>& names) {
    for (const auto& name : names) {
      reload_group_fns_.insert({name, {}});
    }
  }

 private:
  LlmTensorLoader& loader_;

  ReloadGroupFunctions* GetGroupReloadFunctions(const std::string& name);

  // The functions to load the real tensors.
  absl::flat_hash_map<std::string, ReloadGroupFunctions> reload_group_fns_;

  bool placeholder_mode_ = false;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_PLACEHOLDER_TENSOR_LOADER_H_
