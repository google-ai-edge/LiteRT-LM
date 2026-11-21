// Copyright 2026 The ODML Authors.
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

#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/name_check.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/basic_gpu_tensor.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_dummy_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_file_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/test_utils.h"

namespace odml::infra::gpu {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::GpuSpatialTensor;
using ::ml_drift::OHWI;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::WeightsDescription;
using ::ml_drift::WeightsLayout;
using ::testing::ElementsAre;

class VecDataHolder : public llm_utils::DataHolder<uint8_t> {
 public:
  explicit VecDataHolder(std::vector<uint8_t> data) : data_(std::move(data)) {}
  absl::Span<uint8_t> GetData() const override { return absl::MakeSpan(data_); }

 private:
  mutable std::vector<uint8_t> data_;
};

TEST(NameCheckTest, ClassifiesTensorNames) {
  EXPECT_TRUE(IsLoRATensorName("layer.attn.q_prime_left"));
  EXPECT_TRUE(IsLoRATensorName("layer.attn.q_prime_right"));
  EXPECT_FALSE(IsLoRATensorName("layer.attn.q"));

  EXPECT_TRUE(IsFeedForwardTensorName("layer.ffn_layer1.w"));
  EXPECT_TRUE(IsFeedForwardTensorName("layer.mlp.w"));
  EXPECT_FALSE(IsFeedForwardTensorName("layer.attn.w"));

  EXPECT_TRUE(IsAttentionTensorName("layer.self_attention.q"));
  EXPECT_TRUE(IsAttentionTensorName("layer.attn.k"));
  EXPECT_FALSE(IsAttentionTensorName("layer.ffn_layer.w"));
}

TEST(EmbeddingUpdaterTest, UpdatesStandardAndNgrammerEmbeddings) {
  std::vector<uint8_t> raw_weights = {10, 11, 20, 21, 30, 31, 40, 41};
  std::vector<float> raw_scales_floats = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint8_t> raw_scales(raw_scales_floats.size() * sizeof(float));
  std::memcpy(raw_scales.data(), raw_scales_floats.data(), raw_scales.size());

  std::vector<EmbeddingRaw> embeddings(1);
  embeddings[0].data_type = DataType::kInt8;
  embeddings[0].weights = std::make_unique<VecDataHolder>(raw_weights);
  embeddings[0].scales = std::make_unique<VecDataHolder>(raw_scales);

  std::vector<EmbeddingWeights> weights(1);
  weights[0].scales.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 2));
  weights[0].weights.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  weights[0].weights_data.resize(4, 0);
  weights[0].scales_data.resize(2, 0.0f);

  const std::vector<int> tokens = {2, -1};
  UpdateEmbeddings(embeddings, weights, tokens, /*max_emb_size=*/4);
  EXPECT_THAT(weights[0].weights_data, ElementsAre(30, 31, 0, 0));
  EXPECT_THAT(weights[0].scales_data, ElementsAre(3.0f, 0.0f));

  UpdateNgrammerEmbeddings(embeddings, weights, /*previous_token=*/1,
                           /*tokens=*/{1, 0}, /*vocabulary_size=*/2,
                           /*ngram_vocabulary_size=*/4);
  EXPECT_GT(weights[0].scales_data[0], 0.0f);
}

TEST(DummyAndCachingLoaderTest, LoadsAndCachesQuantizedWeights) {
  std::vector<std::unique_ptr<BasicGpuSpatialTensor>> created;
  auto base_loader = std::make_unique<LlmDummyTensorLoader>(
      LlmDummyTensorLoader::WeightsType::kInt4, /*quantization_i_group_size=*/4,
      /*use_zero_point=*/true, /*use_random_data=*/true);
  auto loader = LlmTensorLoader::MakeCaching(std::move(base_loader));
  loader->SetDataType(DataType::kFloat32);
  loader->SetCreateTensorFn([&created](const TensorDescriptor& desc)
                                -> absl::StatusOr<GpuSpatialTensor*> {
    created.push_back(std::make_unique<BasicGpuSpatialTensor>(desc));
    return created.back().get();
  });

  WeightsDescription desc;
  desc.type = DataType::kUint8;
  desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  desc.output_group_size = 1;
  const OHWI shape(4, 1, 1, 4);

  ASSERT_OK_AND_ASSIGN(auto i8_first,
                       loader->LoadInt8Weights("layer.w", desc, shape, false));
  ASSERT_OK_AND_ASSIGN(auto i8_cached,
                       loader->LoadInt8Weights("layer.w", desc, shape, false));
  EXPECT_EQ(i8_first.weights, i8_cached.weights);
  EXPECT_EQ(i8_first.weights_sum_i, i8_cached.weights_sum_i);

  desc.type = DataType::kUint4;
  ASSERT_OK_AND_ASSIGN(
      auto i4_first,
      loader->LoadInt4Weights("layer.w4", desc, shape, false, std::nullopt));
  ASSERT_OK_AND_ASSIGN(
      auto i4_cached,
      loader->LoadInt4Weights("layer.w4", desc, shape, false, std::nullopt));
  EXPECT_EQ(i4_first.weights, i4_cached.weights);

  desc.type = DataType::kUint2;
  ASSERT_OK_AND_ASSIGN(auto i2_first,
                       loader->LoadInt2Weights("layer.w2", desc, shape, false));
  ASSERT_OK_AND_ASSIGN(auto i2_cached,
                       loader->LoadInt2Weights("layer.w2", desc, shape, false));
  EXPECT_EQ(i2_first.weights, i2_cached.weights);

  desc.type = DataType::kFloat32;
  ASSERT_OK_AND_ASSIGN(auto f32_first,
                       loader->LoadWeights("layer.wf", desc, shape, false));
  ASSERT_OK_AND_ASSIGN(auto f32_cached,
                       loader->LoadWeights("layer.wf", desc, shape, false));
  EXPECT_EQ(f32_first, f32_cached);
}

TEST(LlmFileTensorLoaderTest, LoadsWeightsAndLoRAFromDirectory) {
  const std::filesystem::path base_dir =
      std::filesystem::path(::testing::TempDir()) / "file_loader_test";
  const std::filesystem::path weights_dir = base_dir / "weights";
  const std::filesystem::path lora_dir = base_dir / "lora";
  std::filesystem::create_directories(weights_dir);
  std::filesystem::create_directories(lora_dir);

  auto write_bytes = [](const std::filesystem::path& p,
                        absl::Span<const uint8_t> bytes) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  };

  const std::vector<float> f32_lora = {1.5f, 2.5f, 3.5f, 4.5f};
  write_bytes(lora_dir / "q_prime_left",
              absl::Span<const uint8_t>(
                  reinterpret_cast<const uint8_t*>(f32_lora.data()),
                  f32_lora.size() * sizeof(float)));

  const std::vector<uint8_t> i8_data(16, 2);
  write_bytes(weights_dir / "w_i8", i8_data);

  const std::vector<uint8_t> i4_packed(8, 0x99);
  write_bytes(weights_dir / "w_i4", i4_packed);

  const std::vector<uint8_t> i2_packed(4, 0xAA);
  write_bytes(weights_dir / "w_i2", i2_packed);

  std::vector<std::unique_ptr<BasicGpuSpatialTensor>> created;
  LlmFileTensorLoader loader(weights_dir.string(), lora_dir.string());
  loader.SetDataType(DataType::kFloat32);
  loader.SetCreateTensorFn([&created](const TensorDescriptor& desc)
                               -> absl::StatusOr<GpuSpatialTensor*> {
    created.push_back(std::make_unique<BasicGpuSpatialTensor>(desc));
    return created.back().get();
  });

  EXPECT_TRUE(loader.HasTensor("q_prime_left"));
  EXPECT_FALSE(loader.HasTensor("missing"));
  EXPECT_THAT(loader.LoadFloat32("q_prime_left", 4),
              ElementsAre(1.5f, 2.5f, 3.5f, 4.5f));

  WeightsDescription desc;
  desc.type = DataType::kUint8;
  desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  desc.output_group_size = 1;
  const OHWI shape(4, 1, 1, 8);
  const std::vector<uint8_t> i8_data8(32, 2);
  write_bytes(weights_dir / "w_i8_8", i8_data8);
  ASSERT_OK(loader.LoadInt8Weights("w_i8_8", desc, shape, false));

  const std::vector<uint8_t> i4_packed8(16, 0x99);
  write_bytes(weights_dir / "w_i4_8", i4_packed8);
  desc.type = DataType::kUint4;
  ASSERT_OK(loader.LoadInt4Weights("w_i4_8", desc, shape, false, std::nullopt));

  desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
  ASSERT_OK(loader.LoadInt4Weights("w_i4_8", desc, shape, false, std::nullopt));

  const std::vector<uint8_t> i2_packed8(8, 0xAA);
  write_bytes(weights_dir / "w_i2_8", i2_packed8);
  desc.type = DataType::kUint2;
  desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  ASSERT_OK(loader.LoadInt2Weights("w_i2_8", desc, shape, false));

  const std::vector<float> f32_weights(32, 0.5f);
  write_bytes(weights_dir / "w_f32",
              absl::Span<const uint8_t>(
                  reinterpret_cast<const uint8_t*>(f32_weights.data()),
                  f32_weights.size() * sizeof(float)));
  desc.type = DataType::kFloat32;
  ASSERT_OK(loader.LoadWeights("w_f32", desc, shape, false));
}

}  // namespace
}  // namespace odml::infra::gpu
