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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_FILE_TENSOR_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_FILE_TENSOR_LOADER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"

namespace odml::infra::gpu {

// A `LlmTensorLoader` that loads tensors from files in `weights_directory_`.
// It can also load LoRA weights from `lora_weights_directory_`.
class LlmFileTensorLoader : public LlmTensorLoader {
 public:
  explicit LlmFileTensorLoader(const std::string& weights_directory)
      : weights_directory_(weights_directory) {}

  explicit LlmFileTensorLoader(const std::string& weights_directory,
                               const std::string& lora_weights_directory)
      : weights_directory_(weights_directory),
        lora_weights_directory_(lora_weights_directory) {}

  ~LlmFileTensorLoader() override = default;

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

  absl::StatusOr<std::unique_ptr<llm_utils::DataHolder<uint8_t>>>
  LoadRawTensorData(const std::string& name) override;

 private:
  absl::StatusOr<LlmTensorLoader::QuantizedWeights> LoadInt4WeightsFast(
      const std::string& tensor_name,
      const ::ml_drift::WeightsDescription& weights_desc,
      const ::ml_drift::OHWI& shape, bool swap_dims);

  std::vector<int8_t> LoadInt8(const std::string& tensor_name, int count);
  std::vector<int8_t> LoadInt4AsInt8(const std::string& tensor_name, int count);
  std::vector<int8_t> LoadInt2AsInt8(const std::string& tensor_name, int count);

  std::vector<uint8_t> LoadInt4Packed(const std::string& tensor_name,
                                      int count);

  std::string GetTensorPath(absl::string_view tensor_name) const;

  std::string weights_directory_;
  std::optional<std::string> lora_weights_directory_;
  mutable std::unordered_map<std::string, int> group_size_cache_;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_FILE_TENSOR_LOADER_H_
