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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/bind_front.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/basic_gpu_tensor.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/placeholder_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/llm_file_metadata.pb.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"

namespace odml::infra::gpu {
namespace {
using ::ml_drift::DataType;
using ::ml_drift::GpuSpatialTensor;
using ::ml_drift::half;
using ::ml_drift::OHWI;
using ::ml_drift::TensorDescriptor;
using ::odml::infra::llm_utils::DataHolder;
using ::odml::infra::llm_utils::ModelData;
using ::odml::infra::proto::LlmFileMetadata;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::IsSupersetOf;

class SpanDataHolder : public DataHolder<uint8_t> {
 public:
  explicit SpanDataHolder(absl::Span<uint8_t> data) : data_(data) {}
  ~SpanDataHolder() override = default;

  absl::Span<uint8_t> GetData() const override { return data_; }

 private:
  absl::Span<uint8_t> data_;
};

class BasicModelData : public ModelData {
 public:
  BasicModelData(const std::vector<uint8_t>& data,
                 const LlmFileMetadata& metadata)
      : data_(data), metadata_(metadata) {}
  ~BasicModelData() override = default;

  std::optional<proto::LlmModelType> GetModelType() override {
    return std::nullopt;
  }

  std::optional<int> LoRARank() override { return std::nullopt; }

  const proto::LlmParameters& GetLlmParameters() override {
    return metadata_.model_params();
  }

  absl::StatusOr<std::string> ReadMetadata(absl::string_view name) override {
    return "";
  }

  uint64_t GetMaxTensorSize() const override { return 0; }

  uint64_t GetTensorSize(absl::string_view name) const override {
    for (const auto& tensor : metadata_.tensors()) {
      if (tensor.name() == name) {
        return tensor.size();
      }
    }
    return 0;
  }

  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    for (const auto& tensor : metadata_.tensors()) {
      if (tensor.name() == name) {
        return std::make_unique<SpanDataHolder>(
            absl::MakeSpan(data_).subspan(tensor.offset(), tensor.size()));
      }
    }
    return nullptr;
  }

  void Clear() override {}

  absl::StatusOr<ModelWithData> ReadModel(absl::string_view name) override {
    return ModelWithData{};
  }

 private:
  std::vector<uint8_t> data_;
  LlmFileMetadata metadata_;
};

// A DataHolder that owns a copy of its data, like the memory-mapped and
// streamed holders used in production, so ASan reports any read of the data
// after the holder is destroyed. Counts the holders that are alive.
class OwningDataHolder : public DataHolder<uint8_t> {
 public:
  OwningDataHolder(absl::Span<const uint8_t> data, int* live_holders)
      : data_(data.begin(), data.end()), live_holders_(live_holders) {
    ++*live_holders_;
  }
  ~OwningDataHolder() override { --*live_holders_; }

  absl::Span<uint8_t> GetData() const override { return absl::MakeSpan(data_); }

 private:
  mutable std::vector<uint8_t> data_;
  int* live_holders_;
};

// A BasicModelData whose reads return OwningDataHolders.
class OwningModelData : public BasicModelData {
 public:
  using BasicModelData::BasicModelData;

  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    ASSIGN_OR_RETURN(std::unique_ptr<DataHolder<uint8_t>> holder,
                     BasicModelData::ReadTensor(name));
    if (holder == nullptr) {
      return nullptr;
    }
    return std::make_unique<OwningDataHolder>(holder->GetData(),
                                              &live_holders_);
  }

  // Returns the number of holders returned by ReadTensor that are alive.
  int live_holders() const { return live_holders_; }

 private:
  int live_holders_ = 0;
};

enum class PlaceholderType {
  kNoPlaceholder,
  kPassThrough,
  kPlaceholder,
};

struct TestParam {
  DataType data_type = DataType::kFloat32;
  PlaceholderType placeholder_type = PlaceholderType::kNoPlaceholder;
};

class ModelBuilder {
 public:
  template <typename T>
  void AddData(const std::string& name, absl::Span<const T> data) {
    uint32_t size = data.size() * sizeof(T);
    auto* tensor = file_metadata_.add_tensors();
    tensor->set_offset(cur_offset_);
    tensor->set_size(size);
    tensor->set_name(name);
    cur_offset_ += size;
    const uint8_t* data_to_insert =
        reinterpret_cast<const uint8_t*>(data.data());
    data_.insert(data_.end(), data_to_insert, data_to_insert + size);
  }

  template <typename T>
  void AddLoRAData(const std::string& name, absl::Span<const T> data) {
    uint32_t size = data.size() * sizeof(T);
    auto* tensor = lora_file_metadata_.add_tensors();
    tensor->set_offset(cur_lora_offset_);
    tensor->set_size(size);
    tensor->set_name(name);
    cur_lora_offset_ += size;
    const uint8_t* data_to_insert =
        reinterpret_cast<const uint8_t*>(data.data());
    lora_data_.insert(lora_data_.end(), data_to_insert, data_to_insert + size);
  }

  std::unique_ptr<LlmTensorLoader> SetPlaceholderMode(
      TestParam param, std::unique_ptr<LlmTensorLoader> loader);

  void AddPlaceholderTensor(GpuSpatialTensor* tensor);

  absl::Status LoadPlaceholderData();

  std::unique_ptr<LlmTensorLoader> ToLoader(TestParam param) {
    std::shared_ptr<ModelData> lora_model_data;
    if (!lora_data_.empty()) {
      lora_model_data =
          std::make_shared<BasicModelData>(lora_data_, lora_file_metadata_);
    }
    std::unique_ptr<LlmTensorLoader> loader =
        std::make_unique<GpuOptimizedTensorLoader>(
            std::make_shared<BasicModelData>(data_, file_metadata_),
            std::move(lora_model_data));

    loader->SetCreateTensorFn(
        absl::bind_front(&ModelBuilder::CreateTensor, this));
    loader->SetDataType(param.data_type);

    return SetPlaceholderMode(param, std::move(loader));
  }

  std::shared_ptr<OwningModelData> ToOwningModelData() const {
    return std::make_shared<OwningModelData>(data_, file_metadata_);
  }

 private:
  absl::StatusOr<GpuSpatialTensor*> CreateTensor(
      const TensorDescriptor& tensor_desc) {
    return tensors_.emplace_back(new BasicGpuSpatialTensor(tensor_desc)).get();
  }

  uint32_t cur_offset_ = 0;
  uint32_t cur_lora_offset_ = 0;
  uint32_t handle_id_ = 0;
  LlmFileMetadata file_metadata_;
  LlmFileMetadata lora_file_metadata_;
  std::vector<uint8_t> data_;
  std::vector<uint8_t> lora_data_;
  std::vector<std::unique_ptr<BasicGpuSpatialTensor>> tensors_;
  // The ownership of the placeholder loaders is owned by the return value of
  // `SetPlaceholderMode`, the caller should not call `LoadPlaceholderData`
  // after destroy the return value.
  absl::flat_hash_map<PlaceholderTensorLoader*,
                      std::unique_ptr<LlmTensorLoader>>
      placeholder_tensors_loader_map_;
  absl::flat_hash_map<uint32_t, PlaceholderTensor*> placeholder_tensors_map_;
};

std::unique_ptr<LlmTensorLoader> ModelBuilder::SetPlaceholderMode(
    TestParam param, std::unique_ptr<LlmTensorLoader> loader) {
  if (param.placeholder_type == PlaceholderType::kNoPlaceholder) {
    return loader;
  }

  auto placeholder_loader = std::make_unique<PlaceholderTensorLoader>(*loader);
  placeholder_tensors_loader_map_[placeholder_loader.get()] = std::move(loader);

  if (param.placeholder_type == PlaceholderType::kPlaceholder) {
    placeholder_loader->EnablePlaceholderMode();
  }

  return placeholder_loader;
}

void ModelBuilder::AddPlaceholderTensor(GpuSpatialTensor* tensor) {
  if (auto placeholder_tensor =
          dynamic_cast<odml::infra::gpu::PlaceholderTensor*>(tensor)) {
    uint32_t id = handle_id_++;
    placeholder_tensor->SetId(id);
    placeholder_tensors_map_[id] = placeholder_tensor;
  }
}

absl::Status ModelBuilder::LoadPlaceholderData() {
  for (auto& [placeholder_loader, real_loader] :
       placeholder_tensors_loader_map_) {
    for (const auto& [group_name, group_reload_fns] :
         placeholder_loader->GetTensorReloadFns()) {
      for (const auto& [name, reload_fn] : group_reload_fns) {
        ASSIGN_OR_RETURN(const auto new_tensors, reload_fn(real_loader.get()));
        for (const auto& [new_tensor, id] : new_tensors) {
          placeholder_tensors_map_[id]->SetDescriptor(
              new_tensor->GetDescriptor());
        }
      }
    }
  }
  return absl::OkStatus();
}

class GpuOptimizedTensorLoaderTest : public testing::TestWithParam<TestParam> {
 public:
  std::vector<float> GetAsFloat(GpuSpatialTensor* tensor) {
    if (GetParam().data_type == DataType::kFloat16) {
      return GetAsType<half>(tensor);
    } else {
      return GetAsType<float>(tensor);
    }
  }

 private:
  template <typename T>
  std::vector<float> GetAsType(GpuSpatialTensor* tensor) {
    // Copy the returned descriptor.
    ml_drift::TensorDescriptor descriptor = tensor->GetDescriptor();
    auto data = descriptor.GetData();
    const T* fdata = reinterpret_cast<const T*>(data.data());
    ABSL_CHECK_EQ(data.size() % sizeof(T), 0);
    return std::vector<float>(fdata, fdata + data.size() / sizeof(T));
  }
};

using GpuOptimizedPlaceholderTensorLoaderTest = GpuOptimizedTensorLoaderTest;

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, HasTensor) {
  ModelBuilder builder;
  builder.AddData<uint8_t>("foo", {1, 2, 3});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  EXPECT_TRUE(loader->HasTensor("foo"));
  EXPECT_FALSE(loader->HasTensor("bar"));
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, GetTensorElementSizeInBits) {
  ModelBuilder builder;
  builder.AddData<uint8_t>("foo", {1, 2, 3, 4, 5, 6, 7, 8});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  EXPECT_EQ(loader->GetTensorElementSizeInBits("foo", 8), 8);
  EXPECT_EQ(loader->GetTensorElementSizeInBits("foo", 4), 16);
  EXPECT_EQ(loader->GetTensorElementSizeInBits("foo", 2), 32);
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadFloat32) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<float>("foo", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  EXPECT_THAT(loader->LoadFloat32("foo", 4),
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));

  // Using the wrong count gives empty data.
  EXPECT_THAT(loader->LoadFloat32("foo", 1), IsEmpty());
}

TEST_P(GpuOptimizedTensorLoaderTest, LoadBias) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<float>("bias", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  absl::StatusOr<ml_drift::GpuSpatialTensor*> tensor =
      loader->LoadBias("bias", 4);
  ASSERT_OK(tensor);
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));

  // Wrong size gives an error.
  EXPECT_FALSE(loader->LoadBias("bias", 2).ok());

  // Non existent tensor gives 0.
  tensor = loader->LoadBias("nonexistent", 4);
  ASSERT_OK(tensor);
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(0.0f, 0.0f, 0.0f, 0.0f));
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadBias) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<float>("bias", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  absl::StatusOr<ml_drift::GpuSpatialTensor*> tensor =
      loader->LoadBias("bias", 4);
  ASSERT_OK(tensor);
  builder.AddPlaceholderTensor(tensor.value());

  ASSERT_OK(builder.LoadPlaceholderData());
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));

  // Non existent tensor gives 0.
  tensor = loader->LoadBias("nonexistent", 4);
  ASSERT_OK(tensor);
  builder.AddPlaceholderTensor(tensor.value());

  ASSERT_OK(builder.LoadPlaceholderData());
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(0.0f, 0.0f, 0.0f, 0.0f));
}

TEST_P(GpuOptimizedTensorLoaderTest, LoadScale) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<float>("scale", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  absl::StatusOr<ml_drift::GpuSpatialTensor*> tensor = loader->LoadScale(
      "scale", OHWI(4, 1, 1, 1), /*matform_options=*/std::nullopt);
  ASSERT_OK(tensor);
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));

  // Wrong size gives an error.
  EXPECT_FALSE(loader
                   ->LoadScale("scale", OHWI(2, 1, 1, 1),
                               /*matform_options=*/std::nullopt)
                   .ok());

  // Non existent tensor gives an error.
  EXPECT_FALSE(loader
                   ->LoadScale("nonexistent", OHWI(4, 1, 1, 1),
                               /*matform_options=*/std::nullopt)
                   .ok());
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadScale) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<float>("scale", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  absl::StatusOr<ml_drift::GpuSpatialTensor*> tensor = loader->LoadScale(
      "scale", OHWI(4, 1, 1, 1), /*matform_options=*/std::nullopt);
  ASSERT_OK(tensor);
  builder.AddPlaceholderTensor(tensor.value());

  ASSERT_OK(builder.LoadPlaceholderData());
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
}

TEST_P(GpuOptimizedTensorLoaderTest, LoadZeroPoint) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<float>("zero_point", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  absl::StatusOr<ml_drift::GpuSpatialTensor*> tensor =
      loader->LoadZeroPoint("zero_point", "", OHWI(4, 1, 1, 1));
  ASSERT_OK(tensor);
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));

  // Wrong size gives an error.
  EXPECT_FALSE(loader->LoadZeroPoint("zero_point", "", OHWI(2, 1, 1, 1)).ok());

  // Non existent tensor gives 0.
  tensor = loader->LoadZeroPoint("nonexistent", "", OHWI(4, 1, 1, 1));
  ASSERT_OK(tensor);
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(0.0f, 0.0f, 0.0f, 0.0f));
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadZeroPoint) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<float>("zero_point", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  absl::StatusOr<ml_drift::GpuSpatialTensor*> tensor =
      loader->LoadZeroPoint("zero_point", "", OHWI(4, 1, 1, 1));
  ASSERT_OK(tensor);
  builder.AddPlaceholderTensor(tensor.value());
  ASSERT_OK(builder.LoadPlaceholderData());
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));

  // Non existent tensor gives 0.
  tensor = loader->LoadZeroPoint("nonexistent", "", OHWI(4, 1, 1, 1));
  ASSERT_OK(tensor);
  builder.AddPlaceholderTensor(tensor.value());
  ASSERT_OK(builder.LoadPlaceholderData());
  EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(0.0f, 0.0f, 0.0f, 0.0f));
}

TEST_P(GpuOptimizedTensorLoaderTest, LoadInt8Weights) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<uint8_t>("weights", {5, 6, 7, 8});
  builder.AddData<int32_t>("weights.sum_i", {11, 15});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kUint8;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(2, 1, 1, 2);
  absl::StatusOr<LlmTensorLoader::QuantizedWeights> int8_buffer_weights =
      loader->LoadInt8Weights("weights", weights_desc, shape, false);
  ASSERT_OK(int8_buffer_weights);
  ml_drift::TensorDescriptor descriptor =
      int8_buffer_weights->weights->GetDescriptor();
  EXPECT_THAT(descriptor.GetData(), ElementsAre(5, 6, 7, 8));
  EXPECT_THAT(descriptor.GetStorageType(),
              ml_drift::TensorStorageType::kBuffer);
  EXPECT_THAT(descriptor.GetDataType(), ml_drift::DataType::kUint8);
  ml_drift::TensorDescriptor descriptor_sum_i =
      int8_buffer_weights->weights_sum_i->GetDescriptor();
  EXPECT_THAT(descriptor_sum_i.GetDataType(), ml_drift::DataType::kInt32);
  ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> tensor_sum_i;
  descriptor_sum_i.DownloadData(&tensor_sum_i);
  EXPECT_THAT(tensor_sum_i.data, ElementsAre(11, 15));

  // Wrong size gives an error.
  EXPECT_FALSE(
      loader->LoadInt8Weights("weights", weights_desc, OHWI(1, 1, 1, 1), false)
          .ok());

  // Non existent tensor gives an error.
  EXPECT_FALSE(
      loader->LoadInt8Weights("nonexistent", weights_desc, shape, false).ok());

  // Texture weights.
  {
    weights_desc.layout =
        ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
    absl::StatusOr<LlmTensorLoader::QuantizedWeights> int8_texture_weights =
        loader->LoadInt8Weights("weights", weights_desc, shape, false);
    ASSERT_OK(int8_texture_weights);
    ml_drift::TensorDescriptor descriptor =
        int8_texture_weights->weights->GetDescriptor();
    EXPECT_THAT(descriptor.GetData(), ElementsAre(5, 6, 7, 8));
    EXPECT_THAT(descriptor.GetStorageType(),
                ml_drift::TensorStorageType::kTexture2D);
    EXPECT_THAT(descriptor.GetDataType(), ml_drift::DataType::kUint32);
  }
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadInt8Weights) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  builder.AddData<uint8_t>("weights", {5, 6, 7, 8});
  builder.AddData<int32_t>("weights.sum_i", {11, 15});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());

  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kUint8;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(2, 1, 1, 2);
  absl::StatusOr<LlmTensorLoader::QuantizedWeights> int8_buffer_weights =
      loader->LoadInt8Weights("weights", weights_desc, shape, false);
  ASSERT_OK(int8_buffer_weights);
  builder.AddPlaceholderTensor(int8_buffer_weights.value().weights);
  builder.AddPlaceholderTensor(int8_buffer_weights.value().weights_sum_i);

  ASSERT_OK(builder.LoadPlaceholderData());

  ml_drift::TensorDescriptor descriptor =
      int8_buffer_weights->weights->GetDescriptor();
  EXPECT_THAT(descriptor.GetData(), ElementsAre(5, 6, 7, 8));
  EXPECT_EQ(descriptor.GetStorageType(), ml_drift::TensorStorageType::kBuffer);
  EXPECT_EQ(descriptor.GetDataType(), ml_drift::DataType::kUint8);
  ml_drift::TensorDescriptor descriptor_sum_i =
      int8_buffer_weights->weights_sum_i->GetDescriptor();
  EXPECT_EQ(descriptor_sum_i.GetDataType(), ml_drift::DataType::kInt32);
  ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> tensor_sum_i;
  descriptor_sum_i.DownloadData(&tensor_sum_i);
  EXPECT_THAT(tensor_sum_i.data, ElementsAre(11, 15));

  // Texture weights.
  {
    weights_desc.layout =
        ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
    absl::StatusOr<LlmTensorLoader::QuantizedWeights> int8_texture_weights =
        loader->LoadInt8Weights("weights", weights_desc, shape, false);
    ASSERT_OK(int8_texture_weights);
    builder.AddPlaceholderTensor(int8_texture_weights.value().weights);
    builder.AddPlaceholderTensor(int8_texture_weights.value().weights_sum_i);
    ASSERT_OK(builder.LoadPlaceholderData());
    ml_drift::TensorDescriptor descriptor =
        int8_texture_weights->weights->GetDescriptor();
    EXPECT_THAT(descriptor.GetData(), ElementsAre(5, 6, 7, 8));
    EXPECT_EQ(descriptor.GetStorageType(),
              ml_drift::TensorStorageType::kTexture2D);
    EXPECT_EQ(descriptor.GetDataType(), ml_drift::DataType::kUint32);
  }
}

TEST_P(GpuOptimizedTensorLoaderTest, LoadInt4Weights) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  uint8_t wo0i0i1 = 5 | 5 << 4u;
  uint8_t wo1i0i1 = 6 | 6 << 4u;
  uint8_t wo2i0i1 = 7 | 7 << 4u;
  uint8_t wo3i0i1 = 8 | 8 << 4u;
  builder.AddData<uint8_t>("weights", {wo0i0i1, wo1i0i1, wo2i0i1, wo3i0i1});
  builder.AddData<int32_t>("weights.sum_i", {10, 12, 14, 16});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kUint4;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(4, 1, 1, 2);
  absl::StatusOr<LlmTensorLoader::QuantizedWeights> int4_buffer_weights =
      loader->LoadInt4Weights("weights", weights_desc, shape, false,
                              /*matform_options=*/std::nullopt);
  ASSERT_OK(int4_buffer_weights);
  ml_drift::TensorDescriptor descriptor =
      int4_buffer_weights->weights->GetDescriptor();
  EXPECT_THAT(descriptor.GetData(),
              ElementsAre(wo0i0i1, wo1i0i1, wo2i0i1, wo3i0i1));
  EXPECT_THAT(descriptor.GetStorageType(),
              ml_drift::TensorStorageType::kBuffer);
  EXPECT_THAT(descriptor.GetDataType(), ml_drift::DataType::kUint4);
  ml_drift::TensorDescriptor descriptor_sum_i =
      int4_buffer_weights->weights_sum_i->GetDescriptor();
  EXPECT_THAT(descriptor_sum_i.GetDataType(), ml_drift::DataType::kInt32);
  ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> tensor_sum_i;
  descriptor_sum_i.DownloadData(&tensor_sum_i);
  EXPECT_THAT(tensor_sum_i.data, ElementsAre(10, 12, 14, 16));

  // Wrong size gives an error.
  EXPECT_FALSE(loader
                   ->LoadInt4Weights("weights", weights_desc, OHWI(1, 1, 1, 1),
                                     false, /*matform_options=*/std::nullopt)
                   .ok());

  // Non existent tensor gives an error.
  EXPECT_FALSE(loader
                   ->LoadInt4Weights("nonexistent", weights_desc, shape, false,
                                     /*matform_options=*/std::nullopt)
                   .ok());

  // Texture weights.
  {
    weights_desc.layout =
        ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
    absl::StatusOr<LlmTensorLoader::QuantizedWeights> int4_texture_weights =
        loader->LoadInt4Weights("weights", weights_desc, shape, false,
                                /*matform_options=*/std::nullopt);
    ASSERT_OK(int4_texture_weights);
    ml_drift::TensorDescriptor descriptor =
        int4_texture_weights->weights->GetDescriptor();
    EXPECT_THAT(descriptor.GetData(),
                ElementsAre(wo0i0i1, wo1i0i1, wo2i0i1, wo3i0i1));
    EXPECT_THAT(descriptor.GetStorageType(),
                ml_drift::TensorStorageType::kTexture2D);
    EXPECT_THAT(descriptor.GetDataType(), ml_drift::DataType::kUint16);
  }
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadInt4Weights) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  uint8_t wo0i0i1 = 5 | 5 << 4u;
  uint8_t wo1i0i1 = 6 | 6 << 4u;
  uint8_t wo2i0i1 = 7 | 7 << 4u;
  uint8_t wo3i0i1 = 8 | 8 << 4u;
  builder.AddData<uint8_t>("weights", {wo0i0i1, wo1i0i1, wo2i0i1, wo3i0i1});
  builder.AddData<int32_t>("weights.sum_i", {10, 12, 14, 16});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());

  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kUint4;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(4, 1, 1, 2);
  absl::StatusOr<LlmTensorLoader::QuantizedWeights> int4_buffer_weights =
      loader->LoadInt4Weights("weights", weights_desc, shape, false,
                              /*matform_options=*/std::nullopt);
  ASSERT_OK(int4_buffer_weights);
  builder.AddPlaceholderTensor(int4_buffer_weights.value().weights);
  builder.AddPlaceholderTensor(int4_buffer_weights.value().weights_sum_i);

  ASSERT_OK(builder.LoadPlaceholderData());

  ml_drift::TensorDescriptor descriptor =
      int4_buffer_weights->weights->GetDescriptor();
  EXPECT_THAT(descriptor.GetData(),
              ElementsAre(wo0i0i1, wo1i0i1, wo2i0i1, wo3i0i1));
  EXPECT_EQ(descriptor.GetStorageType(), ml_drift::TensorStorageType::kBuffer);
  EXPECT_EQ(descriptor.GetDataType(), ml_drift::DataType::kUint4);
  ml_drift::TensorDescriptor descriptor_sum_i =
      int4_buffer_weights->weights_sum_i->GetDescriptor();
  EXPECT_EQ(descriptor_sum_i.GetDataType(), ml_drift::DataType::kInt32);
  ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> tensor_sum_i;
  descriptor_sum_i.DownloadData(&tensor_sum_i);
  EXPECT_THAT(tensor_sum_i.data, ElementsAre(10, 12, 14, 16));

  // Texture weights.
  {
    weights_desc.layout =
        ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
    absl::StatusOr<LlmTensorLoader::QuantizedWeights> int4_texture_weights =
        loader->LoadInt4Weights("weights", weights_desc, shape, false,
                                /*matform_options=*/std::nullopt);
    ASSERT_OK(int4_texture_weights);
    builder.AddPlaceholderTensor(int4_texture_weights.value().weights);
    builder.AddPlaceholderTensor(int4_texture_weights.value().weights_sum_i);
    ASSERT_OK(builder.LoadPlaceholderData());
    ml_drift::TensorDescriptor descriptor =
        int4_texture_weights->weights->GetDescriptor();
    EXPECT_THAT(descriptor.GetData(),
                ElementsAre(wo0i0i1, wo1i0i1, wo2i0i1, wo3i0i1));
    EXPECT_EQ(descriptor.GetStorageType(),
              ml_drift::TensorStorageType::kTexture2D);
    EXPECT_EQ(descriptor.GetDataType(), ml_drift::DataType::kUint16);
  }
}

TEST_P(GpuOptimizedTensorLoaderTest, MatformOptions) {
  ModelBuilder builder;
  std::vector<uint8_t> weights;
  for (uint8_t i = 0; i < 16; i++) {
    uint8_t v = i + 8;
    weights.push_back(v | v << 4u);
  }
  builder.AddData<uint8_t>("weights", weights);
  builder.AddData<int32_t>("weights.sum_i", {10, 12, 14, 16});
  builder.AddData<float>("scale", {1.0f, 2.0f, 3.0f, 4.0f});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kUint4;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  // Matform the input axis.
  {
    LlmTensorLoader::MatformOptions matform_options{
        .axis = ml_drift::Axis::kInputChannels,
        .matform_factor = 2,
    };
    OHWI shape(4, 1, 1, 4);
    absl::StatusOr<LlmTensorLoader::QuantizedWeights> int4_buffer_weights =
        loader->LoadInt4Weights("weights", weights_desc, shape, false,
                                matform_options);
    ASSERT_OK(int4_buffer_weights);
    ml_drift::TensorDescriptor descriptor =
        int4_buffer_weights->weights->GetDescriptor();
    EXPECT_THAT(descriptor.GetData(),
                ElementsAre(weights[0], weights[1], weights[2], weights[3],
                            weights[4], weights[5], weights[6], weights[7]));

    ml_drift::TensorDescriptor descriptor_sum_i =
        int4_buffer_weights->weights_sum_i->GetDescriptor();
    ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> tensor_sum_i;
    descriptor_sum_i.DownloadData(&tensor_sum_i);
    // These are calculated from the weights tensor.
    EXPECT_THAT(tensor_sum_i.data, ElementsAre(12, 12, 16, 16));

    absl::StatusOr<ml_drift::GpuSpatialTensor*> tensor =
        loader->LoadScale("scale", shape, matform_options);
    EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  }
  // Matform the output axis.
  {
    LlmTensorLoader::MatformOptions matform_options{
        .axis = ml_drift::Axis::kOutputChannels,
        .matform_factor = 2,
    };
    OHWI shape(2, 1, 1, 8);
    absl::StatusOr<LlmTensorLoader::QuantizedWeights> int4_buffer_weights =
        loader->LoadInt4Weights("weights", weights_desc, shape, false,
                                matform_options);
    ASSERT_OK(int4_buffer_weights);
    ml_drift::TensorDescriptor descriptor =
        int4_buffer_weights->weights->GetDescriptor();
    EXPECT_THAT(descriptor.GetData(),
                ElementsAre(weights[0], weights[1], weights[2], weights[3],
                            weights[8], weights[9], weights[10], weights[11]));

    ml_drift::TensorDescriptor descriptor_sum_i =
        int4_buffer_weights->weights_sum_i->GetDescriptor();
    ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> tensor_sum_i;
    descriptor_sum_i.DownloadData(&tensor_sum_i);
    EXPECT_THAT(tensor_sum_i.data, ElementsAre(10, 12));

    absl::StatusOr<ml_drift::GpuSpatialTensor*> tensor =
        loader->LoadScale("scale", shape, matform_options);
    EXPECT_THAT(GetAsFloat(*tensor), ElementsAre(1.0f, 2.0f, 0, 0));
  }
}

TEST_P(GpuOptimizedTensorLoaderTest, LoadInt2Weights) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  uint8_t wo0i0i1i2i3 = 1 | (2 << 2u) | (3 << 4u) | (2 << 6u);
  uint8_t wo1i0i1i2i3 = 2 | (3 << 2u) | (2 << 4u) | (1 << 6u);
  uint8_t wo2i0i1i2i3 = 3 | (2 << 2u) | (1 << 4u) | (2 << 6u);
  uint8_t wo3i0i1i2i3 = 2 | (1 << 2u) | (2 << 4u) | (3 << 6u);
  builder.AddData<uint8_t>(
      "weights", {wo0i0i1i2i3, wo1i0i1i2i3, wo2i0i1i2i3, wo3i0i1i2i3});
  builder.AddData<int32_t>("weights.sum_i", {10, 10, 10, 10});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kUint2;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(4, 1, 1, 4);
  absl::StatusOr<LlmTensorLoader::QuantizedWeights> int2_buffer_weights =
      loader->LoadInt2Weights("weights", weights_desc, shape, false);
  ASSERT_OK(int2_buffer_weights);
  ml_drift::TensorDescriptor descriptor =
      int2_buffer_weights->weights->GetDescriptor();
  EXPECT_THAT(descriptor.GetData(),
              ElementsAre(wo0i0i1i2i3, wo1i0i1i2i3, wo2i0i1i2i3, wo3i0i1i2i3));
  EXPECT_THAT(descriptor.GetStorageType(),
              ml_drift::TensorStorageType::kBuffer);
  EXPECT_THAT(descriptor.GetDataType(), ml_drift::DataType::kUint2);
  ml_drift::TensorDescriptor descriptor_sum_i =
      int2_buffer_weights->weights_sum_i->GetDescriptor();
  EXPECT_THAT(descriptor_sum_i.GetDataType(), ml_drift::DataType::kInt32);
  ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> tensor_sum_i;
  descriptor_sum_i.DownloadData(&tensor_sum_i);
  EXPECT_THAT(tensor_sum_i.data, ElementsAre(10, 10, 10, 10));

  // Wrong size gives an error.
  EXPECT_FALSE(
      loader->LoadInt2Weights("weights", weights_desc, OHWI(1, 1, 1, 1), false)
          .ok());

  // Non existent tensor gives an error.
  EXPECT_FALSE(
      loader->LoadInt2Weights("nonexistent", weights_desc, shape, false).ok());

  // Texture weights.
  {
    weights_desc.layout =
        ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
    absl::StatusOr<LlmTensorLoader::QuantizedWeights> int2_texture_weights =
        loader->LoadInt2Weights("weights", weights_desc, shape, false);
    ASSERT_OK(int2_texture_weights);
    ml_drift::TensorDescriptor descriptor =
        int2_texture_weights->weights->GetDescriptor();
    EXPECT_THAT(descriptor.GetData(), ElementsAre(wo0i0i1i2i3, wo1i0i1i2i3,
                                                  wo2i0i1i2i3, wo3i0i1i2i3));
    EXPECT_THAT(descriptor.GetStorageType(),
                ml_drift::TensorStorageType::kTexture2D);
    EXPECT_THAT(descriptor.GetDataType(), ml_drift::DataType::kUint8);
  }
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadInt2Weights) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<uint8_t>("x", {1, 2, 3, 4});
  uint8_t wo0i0i1i2i3 = 1 | (2 << 2u) | (3 << 4u) | (2 << 6u);
  uint8_t wo1i0i1i2i3 = 2 | (3 << 2u) | (2 << 4u) | (1 << 6u);
  uint8_t wo2i0i1i2i3 = 3 | (2 << 2u) | (1 << 4u) | (2 << 6u);
  uint8_t wo3i0i1i2i3 = 2 | (1 << 2u) | (2 << 4u) | (3 << 6u);
  builder.AddData<uint8_t>(
      "weights", {wo0i0i1i2i3, wo1i0i1i2i3, wo2i0i1i2i3, wo3i0i1i2i3});
  builder.AddData<int32_t>("weights.sum_i", {10, 10, 10, 10});
  builder.AddData<uint8_t>("y", {1, 2, 3, 4});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());

  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kUint2;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(4, 1, 1, 4);
  absl::StatusOr<LlmTensorLoader::QuantizedWeights> int2_buffer_weights =
      loader->LoadInt2Weights("weights", weights_desc, shape, false);
  ASSERT_OK(int2_buffer_weights);
  builder.AddPlaceholderTensor(int2_buffer_weights.value().weights);
  builder.AddPlaceholderTensor(int2_buffer_weights.value().weights_sum_i);

  ASSERT_OK(builder.LoadPlaceholderData());

  ml_drift::TensorDescriptor descriptor =
      int2_buffer_weights->weights->GetDescriptor();
  EXPECT_THAT(descriptor.GetData(),
              ElementsAre(wo0i0i1i2i3, wo1i0i1i2i3, wo2i0i1i2i3, wo3i0i1i2i3));
  EXPECT_EQ(descriptor.GetStorageType(), ml_drift::TensorStorageType::kBuffer);
  EXPECT_EQ(descriptor.GetDataType(), ml_drift::DataType::kUint2);
  ml_drift::TensorDescriptor descriptor_sum_i =
      int2_buffer_weights->weights_sum_i->GetDescriptor();
  EXPECT_EQ(descriptor_sum_i.GetDataType(), ml_drift::DataType::kInt32);
  ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> tensor_sum_i;
  descriptor_sum_i.DownloadData(&tensor_sum_i);
  EXPECT_THAT(tensor_sum_i.data, ElementsAre(10, 10, 10, 10));

  // Texture weights.
  {
    weights_desc.layout =
        ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
    absl::StatusOr<LlmTensorLoader::QuantizedWeights> int2_texture_weights =
        loader->LoadInt2Weights("weights", weights_desc, shape, false);
    ASSERT_OK(int2_texture_weights);
    builder.AddPlaceholderTensor(int2_texture_weights.value().weights);
    builder.AddPlaceholderTensor(int2_texture_weights.value().weights_sum_i);
    ASSERT_OK(builder.LoadPlaceholderData());
    ml_drift::TensorDescriptor descriptor =
        int2_texture_weights->weights->GetDescriptor();
    EXPECT_THAT(descriptor.GetData(), ElementsAre(wo0i0i1i2i3, wo1i0i1i2i3,
                                                  wo2i0i1i2i3, wo3i0i1i2i3));
    EXPECT_EQ(descriptor.GetStorageType(),
              ml_drift::TensorStorageType::kTexture2D);
    EXPECT_EQ(descriptor.GetDataType(), ml_drift::DataType::kUint8);
  }
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadWeights) {
  ModelBuilder builder;
  // Add some data before and after.
  builder.AddData<float>("weights", {1.0f, 2.0f, 3.0f, 4.0f});
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kFloat32;
  weights_desc.layout =
      ml_drift::WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4;
  OHWI shape(2, 1, 1, 2);

  // Invalid data type gives an error.
  ASSERT_FALSE(loader->LoadWeights("weights", weights_desc, shape, false).ok());
  // Non existent tensor gives an error.
  EXPECT_FALSE(
      loader->LoadWeights("nonexistent", weights_desc, shape, false).ok());
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadLoRA) {
  ModelBuilder builder;
  builder.AddData<float>("w_prime_left", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<float>("w_prime_right", {5.0f, 6.0f, 7.0f, 8.0f});
  builder.AddLoRAData<float>("w_prime_left", {9.0f, 10.0f, 11.0f, 12.0f});
  builder.AddLoRAData<float>("w_prime_right", {13.0f, 14.0f, 15.0f, 16.0f});

  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kFloat32;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(4, 1, 1, 1);

  absl::StatusOr<ml_drift::GpuSpatialTensor*> left_lora =
      loader->LoadWeights("w_prime_left", weights_desc, shape, false);
  ASSERT_OK(left_lora);
  builder.AddPlaceholderTensor(left_lora.value());
  absl::StatusOr<ml_drift::GpuSpatialTensor*> right_lora =
      loader->LoadWeights("w_prime_right", weights_desc, shape, false);
  ASSERT_OK(right_lora);
  builder.AddPlaceholderTensor(right_lora.value());

  ASSERT_OK(builder.LoadPlaceholderData());

  ml_drift::TensorDescriptor left_desc = (*left_lora)->GetDescriptor();
  ml_drift::TensorDescriptor right_desc = (*right_lora)->GetDescriptor();
  EXPECT_THAT(left_desc.GetStorageType(), ml_drift::TensorStorageType::kBuffer);
  EXPECT_THAT(right_desc.GetStorageType(),
              ml_drift::TensorStorageType::kBuffer);
  EXPECT_THAT(left_desc.GetDataType(), ml_drift::DataType::kFloat32);
  EXPECT_THAT(right_desc.GetDataType(), ml_drift::DataType::kFloat32);
  auto left_data = reinterpret_cast<const float*>(left_desc.GetData().data());
  auto right_data = reinterpret_cast<const float*>(right_desc.GetData().data());
  // Data should be read from the LoRA memory region.
  EXPECT_THAT(std::vector<float>(left_data, left_data + 4),
              ElementsAre(9.0f, 10.0f, 11.0f, 12.0f));
  EXPECT_THAT(std::vector<float>(right_data, right_data + 4),
              ElementsAre(13.0f, 14.0f, 15.0f, 16.0f));
}

TEST_P(GpuOptimizedPlaceholderTensorLoaderTest, LoadEmptyLoRAWeights) {
  ModelBuilder builder;
  std::unique_ptr<LlmTensorLoader> loader = builder.ToLoader(GetParam());
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = ml_drift::DataType::kFloat32;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(4, 1, 1, 1);

  absl::StatusOr<ml_drift::GpuSpatialTensor*> left_lora =
      loader->LoadWeights("w_prime_left", weights_desc, shape, false);
  ASSERT_OK(left_lora);
  builder.AddPlaceholderTensor(left_lora.value());
  absl::StatusOr<ml_drift::GpuSpatialTensor*> right_lora =
      loader->LoadWeights("w_prime_right", weights_desc, shape, false);
  ASSERT_OK(right_lora);
  builder.AddPlaceholderTensor(right_lora.value());

  ASSERT_OK(builder.LoadPlaceholderData());

  ml_drift::TensorDescriptor left_desc = (*left_lora)->GetDescriptor();
  ml_drift::TensorDescriptor right_desc = (*right_lora)->GetDescriptor();
  EXPECT_TRUE(left_desc.GetData().empty());
  EXPECT_TRUE(right_desc.GetData().empty());
}

// Checks when the loader releases the data that it reads. The create function
// stands in for WebGPU's CreateModelTensor: it takes ownership of loader data
// when offered, copies the descriptor data, and then releases the data.
class GpuOptimizedTensorLoaderLifetimeTest : public testing::Test {
 protected:
  std::unique_ptr<GpuOptimizedTensorLoader> CreateLoader(
      const ModelBuilder& builder) {
    model_data_ = builder.ToOwningModelData();
    auto loader = std::make_unique<GpuOptimizedTensorLoader>(
        model_data_, /*lora_data=*/nullptr, /*own_model_data=*/false);
    loader->SetCreateTensorFn(
        [this, raw_loader = loader.get()](const TensorDescriptor& tensor_desc)
            -> absl::StatusOr<GpuSpatialTensor*> {
          live_holders_during_create_.push_back(model_data_->live_holders());
          std::unique_ptr<DataHolder<uint8_t>> owned_data =
              raw_loader->GetOwnedData(tensor_desc.GetData().data());
          took_ownership_.push_back(owned_data != nullptr);
          auto& tensor = tensors_.emplace_back(
              std::make_unique<BasicGpuSpatialTensor>(tensor_desc));
          return tensor.get();
        });
    loader->SetDataType(DataType::kFloat32);
    return loader;
  }

  static std::vector<float> GetFloats(GpuSpatialTensor* tensor) {
    TensorDescriptor descriptor = tensor->GetDescriptor();
    absl::Span<const uint8_t> data = descriptor.GetData();
    const float* floats = reinterpret_cast<const float*>(data.data());
    return std::vector<float>(floats, floats + data.size() / sizeof(float));
  }

  std::shared_ptr<OwningModelData> model_data_;
  std::vector<std::unique_ptr<BasicGpuSpatialTensor>> tensors_;
  std::vector<int> live_holders_during_create_;
  std::vector<bool> took_ownership_;
};

TEST_F(GpuOptimizedTensorLoaderLifetimeTest,
       Fp16WeightsReleasedBeforeConvertedUpload) {
  ModelBuilder builder;
  builder.AddData<half>("weights",
                        {half(1.0f), half(2.0f), half(3.0f), half(4.0f)});
  std::unique_ptr<GpuOptimizedTensorLoader> loader = CreateLoader(builder);
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = DataType::kFloat32;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;

  absl::StatusOr<GpuSpatialTensor*> tensor = loader->LoadWeights(
      "weights", weights_desc, OHWI(4, 1, 1, 1), /*swap_dims=*/false);

  ASSERT_OK(tensor);
  EXPECT_THAT(GetFloats(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  // The fp16 data is released before the converted data is uploaded, and the
  // converted data isn't offered to the backend.
  EXPECT_THAT(live_holders_during_create_, ElementsAre(0));
  EXPECT_THAT(took_ownership_, ElementsAre(false));
  EXPECT_EQ(model_data_->live_holders(), 0);
}

TEST_F(GpuOptimizedTensorLoaderLifetimeTest,
       UnconvertedWeightsOfferedToBackend) {
  ModelBuilder builder;
  builder.AddData<uint8_t>("weights", {5, 6, 7, 8});
  std::unique_ptr<GpuOptimizedTensorLoader> loader = CreateLoader(builder);
  ml_drift::WeightsDescription weights_desc;
  weights_desc.type = DataType::kUint8;
  weights_desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;

  absl::StatusOr<GpuSpatialTensor*> tensor = loader->LoadWeights(
      "weights", weights_desc, OHWI(4, 1, 1, 1), /*swap_dims=*/false);

  ASSERT_OK(tensor);
  TensorDescriptor descriptor = (*tensor)->GetDescriptor();
  EXPECT_THAT(descriptor.GetData(), ElementsAre(5, 6, 7, 8));
  // The loader passes this data to the backend without copying it, so it must
  // stay alive until the backend takes ownership of it.
  EXPECT_THAT(live_holders_during_create_, ElementsAre(1));
  EXPECT_THAT(took_ownership_, ElementsAre(true));
  EXPECT_EQ(model_data_->live_holders(), 0);
}

TEST_F(GpuOptimizedTensorLoaderLifetimeTest, Float32AndInt32ReadsNotRetained) {
  ModelBuilder builder;
  builder.AddData<float>("floats", {1.0f, 2.0f, 3.0f, 4.0f});
  builder.AddData<int32_t>("ints", {7, 8});
  std::unique_ptr<GpuOptimizedTensorLoader> loader = CreateLoader(builder);

  std::vector<float> floats = loader->LoadFloat32("floats", 4);
  std::vector<int32_t> ints = loader->LoadInt32("ints", 2);

  EXPECT_THAT(floats, ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  EXPECT_THAT(ints, ElementsAre(7, 8));
  EXPECT_EQ(model_data_->live_holders(), 0);
}

TEST_F(GpuOptimizedTensorLoaderLifetimeTest, Fp16BiasReleasedAfterConversion) {
  ModelBuilder builder;
  builder.AddData<half>("bias",
                        {half(1.0f), half(2.0f), half(3.0f), half(4.0f)});
  std::unique_ptr<GpuOptimizedTensorLoader> loader = CreateLoader(builder);

  absl::StatusOr<GpuSpatialTensor*> tensor = loader->LoadBias("bias", 4);

  ASSERT_OK(tensor);
  EXPECT_THAT(GetFloats(*tensor), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  EXPECT_EQ(model_data_->live_holders(), 0);
}

TEST_F(GpuOptimizedTensorLoaderLifetimeTest,
       Fp16BlockwiseScaleReleasedAfterConversion) {
  ModelBuilder builder;
  builder.AddData<half>("scale",
                        {half(1.0f), half(2.0f), half(3.0f), half(4.0f)});
  std::unique_ptr<GpuOptimizedTensorLoader> loader = CreateLoader(builder);

  // An input dimension larger than 1 selects a blockwise (2D) scale.
  absl::StatusOr<GpuSpatialTensor*> tensor = loader->LoadScale(
      "scale", OHWI(2, 1, 1, 2), /*matform_options=*/std::nullopt);

  ASSERT_OK(tensor);
  EXPECT_THAT(GetFloats(*tensor), IsSupersetOf({1.0f, 2.0f, 3.0f, 4.0f}));
  EXPECT_EQ(model_data_->live_holders(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    Params, GpuOptimizedTensorLoaderTest,
    testing::Values(
        TestParam{.data_type = DataType::kFloat32,
                  .placeholder_type = PlaceholderType::kNoPlaceholder},
        TestParam{.data_type = DataType::kFloat16,
                  .placeholder_type = PlaceholderType::kNoPlaceholder},
        TestParam{.data_type = DataType::kFloat32,
                  .placeholder_type = PlaceholderType::kPassThrough},
        TestParam{.data_type = DataType::kFloat16,
                  .placeholder_type = PlaceholderType::kPassThrough}));

INSTANTIATE_TEST_SUITE_P(
    Params, GpuOptimizedPlaceholderTensorLoaderTest,
    testing::Values(
        TestParam{.data_type = DataType::kFloat32,
                  .placeholder_type = PlaceholderType::kNoPlaceholder},
        TestParam{.data_type = DataType::kFloat16,
                  .placeholder_type = PlaceholderType::kNoPlaceholder},
        TestParam{.data_type = DataType::kFloat32,
                  .placeholder_type = PlaceholderType::kPassThrough},
        TestParam{.data_type = DataType::kFloat16,
                  .placeholder_type = PlaceholderType::kPassThrough},
        TestParam{.data_type = DataType::kFloat32,
                  .placeholder_type = PlaceholderType::kPlaceholder},
        TestParam{.data_type = DataType::kFloat16,
                  .placeholder_type = PlaceholderType::kPlaceholder}));

}  // namespace
}  // namespace odml::infra::gpu
