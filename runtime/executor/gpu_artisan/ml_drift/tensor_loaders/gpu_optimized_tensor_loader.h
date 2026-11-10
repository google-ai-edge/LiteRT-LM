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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_GPU_OPTIMIZED_TENSOR_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_GPU_OPTIMIZED_TENSOR_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/basic_gpu_tensor.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/llm_file_metadata.pb.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "support/util/memory_mapped_file.h"

namespace odml::infra::gpu {

// A tensor loader which loads weights from:
// 1) A binary file optimized for GPU format. The positions of the weights
//    within the file are specified by a `LlmFileMetadata` proto.
// or
// 2) A TFLite file.
class GpuOptimizedTensorLoader : public LlmTensorLoader {
 public:
// Our clients load buffers with the API below which requires size alignment
// on certain platforms.
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromaddress
#ifdef _WIN32
  // Dawn requires larger alignment on Windows than other platforms.
  static constexpr size_t kAlignment = 65536;
#else
  static constexpr size_t kAlignment = 4096;
#endif

  using Weights = std::unique_ptr<llm_utils::DataHolder<uint8_t>>;
  static absl::StatusOr<std::unique_ptr<GpuOptimizedTensorLoader>> CreateLoRA(
      std::shared_ptr<llm_utils::ModelData> lora_data, int& lora_rank);

  // The returned GpuOptimizedTensorLoader does not own the provided model_data
  // by default. However, if own_model_data is true, the returned
  // GpuOptimizedTensorLoader will call model_data->Clear() in its destructor.
  // own_model_data makes it convenient in case the GpuOptimizedTensorLoader is
  // the last one using the model_data, so model_data.Clear() is not needed to
  // be manually called at the place where the GpuOptimizedTensorLoader is
  // destroyed.
  // TODO: Consider to emove model_data->Clear() from GpuOptimizedTensorLoader's
  // destructor.
  GpuOptimizedTensorLoader(std::shared_ptr<llm_utils::ModelData> model_data,
                           std::shared_ptr<llm_utils::ModelData> lora_data,
                           bool own_model_data = true);
  ~GpuOptimizedTensorLoader() override;

  std::vector<float> LoadFloat32(const std::string& tensor_name,
                                 int count) override;
  std::vector<int32_t> LoadInt32(const std::string& tensor_name, int count);
  int GetTensorElementSizeInBits(const std::string& tensor_name,
                                 int count) override;
  bool HasTensor(const std::string& tensor_name) override;
  absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadScale(
      const std::string& tensor_name, const ::ml_drift::OHWI& shape,
      std::optional<MatformOptions> matform_options) override;
  absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadZeroPoint(
      const std::string& zero_point_name, const std::string& scale_name,
      const ::ml_drift::OHWI& shape) override;
  absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadBias(
      const std::string& tensor_name, int count) override;
  absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadWeights(
      const std::string& tensor_name,
      const ml_drift::WeightsDescription& weights_desc,
      const ml_drift::OHWI& shape, bool swap_dims) override;
  absl::StatusOr<QuantizedWeights> LoadInt8Weights(
      const std::string& tensor_name,
      const ml_drift::WeightsDescription& weights_desc,
      const ml_drift::OHWI& shape, bool swap_dims) override;
  absl::StatusOr<QuantizedWeights> LoadInt4Weights(
      const std::string& tensor_name,
      const ml_drift::WeightsDescription& weights_desc,
      const ml_drift::OHWI& shape, bool swap_dims,
      std::optional<MatformOptions> matform_options) override;
  absl::StatusOr<QuantizedWeights> LoadInt2Weights(
      const std::string& tensor_name,
      const ml_drift::WeightsDescription& weights_desc,
      const ml_drift::OHWI& shape, bool swap_dims) override;
  Weights GetOwnedData(const uint8_t* location) override;
  absl::StatusOr<std::unique_ptr<llm_utils::DataHolder<uint8_t>>>
  LoadRawTensorData(const std::string& name) override;
  std::optional<uint64_t> GetMaxBufferSize() const override;
  int GetQuantizationIGroupSize(
      const std::string& weights_name,
      const ml_drift::OHWI& weights_shape) const override;
  void SetMemoryMappedFileAccess(
      ::litert::support::MemoryMappedFile::Access access) override {
    if (model_data_) {
      model_data_->SetMemoryMappedFileAccess(access);
    }
  }

 private:
  absl::StatusOr<ml_drift::GpuSpatialTensor*> CreateWeightsTensor(
      const ml_drift::WeightsDescription& weights_desc,
      const ml_drift::OHWI& shape, const std::string& name,
      std::optional<MatformOptions> matform_options,
      std::vector<int32_t>* matform_sum_i = nullptr);
  absl::StatusOr<ml_drift::GpuSpatialTensor*> CreateLinearTensor(
      int count, const std::string& name, bool optional,
      std::optional<MatformOptions> matform_options = std::nullopt);
  absl::StatusOr<ml_drift::GpuSpatialTensor*> CreateLinearInt32Tensor(
      int count, const std::string& name,
      std::optional<MatformOptions> matform_options = std::nullopt);
  absl::StatusOr<absl::Span<const uint8_t>> GetData(
      const std::string& tensor_name);
  // Reads a tensor without retaining it in `tensor_weights_`. Use this when the
  // data is copied or converted, rather than uploaded directly.
  absl::StatusOr<Weights> ReadTemporaryData(const std::string& tensor_name);

  llm_utils::ModelData* GetModelData(absl::string_view tensor_name);
  llm_utils::ModelData const* GetModelData(absl::string_view tensor_name) const;

  std::shared_ptr<llm_utils::ModelData> model_data_;
  std::shared_ptr<llm_utils::ModelData> lora_data_;
  std::map<std::string, Weights> tensor_weights_;
  std::vector<std::unique_ptr<BasicGpuSpatialTensor>> placeholder_tensors_;
  const bool own_model_data_;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_GPU_OPTIMIZED_TENSOR_LOADER_H_
