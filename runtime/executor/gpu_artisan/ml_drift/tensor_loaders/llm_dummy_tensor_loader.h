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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_DUMMY_TENSOR_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_DUMMY_TENSOR_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"

namespace odml::infra::gpu {

// A `LlmTensorLoader` that loads fake weights rather than loading tensors from
// an actual model.
class LlmDummyTensorLoader : public LlmTensorLoader {
 public:
  // data type to load attention, linear, and embedding weights as
  enum class WeightsType { kInt8, kInt4, kInt2, kFloat32 };

  explicit LlmDummyTensorLoader(WeightsType weights_type,
                                int quantization_i_group_size = -1,
                                bool use_zero_point = false,
                                bool use_random_data = false)
      : weights_type_(weights_type),
        quantization_i_group_size_(quantization_i_group_size),
        use_zero_point_(use_zero_point),
        use_random_data_(use_random_data) {}

  void AddExplicitTensorType(const std::string& tensor_subname,
                             WeightsType weights_type) {
    explicit_tensor_subnames_[tensor_subname] = weights_type;
  }
  void AddExplicitQuantizationIGroupSize(const std::string& tensor_subname,
                                         int i_group_size) {
    explicit_quantization_i_group_size_subnames_[tensor_subname] = i_group_size;
  }

  // For testing purposes only.
  void AddTensorData(const std::string& tensor_name,
                     const std::vector<float>& data) {
    tensor_data_[tensor_name] = data;
  }

  ~LlmDummyTensorLoader() override = default;

  std::vector<float> LoadFloat32(const std::string& tensor_name,
                                 int count) override;

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

  int GetTensorElementSizeInBits(const std::string& tensor_name,
                                 int count) override;

  bool HasTensor(const std::string& tensor_name) override {
    const auto& it = tensor_data_.find(tensor_name);
    if (it != tensor_data_.end()) {
      return true;
    }
    if (use_zero_point_ && absl::StrContains(tensor_name, "w_quantized_zp")) {
      return true;
    }
    return false;
  }
  int GetQuantizationIGroupSize(
      const std::string& weights_name,
      const ml_drift::OHWI& weights_shape) const override;

 protected:
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> CreateTensor(
      ::ml_drift::DataType data_type, int count);

  absl::StatusOr<::ml_drift::GpuSpatialTensor*> CreateTensorLinearFloat(
      int count);

 private:
  std::vector<float> GenerateFloat32Data(int count);
  std::vector<uint8_t> GenerateBytesData(size_t size);

  WeightsType weights_type_;
  int quantization_i_group_size_ = -1;
  std::map<std::string, WeightsType> explicit_tensor_subnames_;
  std::map<std::string, int> explicit_quantization_i_group_size_subnames_;
  std::map<std::string, std::vector<float>> tensor_data_;
  bool use_zero_point_ = false;
  bool use_random_data_ = false;  // Use this on Intel Xe3 for more realistic
                                  // performance estimation.
  std::mt19937_64 rng_{42};
};

enum class LoRAMode {
  kNone,
  kAttention,
  kAttentionFeedForward,
};

class NonZeroTensorLoader : public LlmDummyTensorLoader {
 public:
  explicit NonZeroTensorLoader(LoRAMode lora_mode = LoRAMode::kNone,
                               float start_value = 1.1f)
      : LlmDummyTensorLoader(WeightsType::kFloat32),
        lora_mode_(lora_mode),
        start_value_(start_value) {}
  ~NonZeroTensorLoader() override = default;

  bool HasTensor(const std::string& tensor_name) override;
  std::vector<float> LoadFloat32(const std::string& tensor_name,
                                 int count) override;

 private:
  bool ShouldPopulateLoRA(const std::string& tensor_name);

  LoRAMode lora_mode_;
  float start_value_;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_LLM_DUMMY_TENSOR_LOADER_H_
