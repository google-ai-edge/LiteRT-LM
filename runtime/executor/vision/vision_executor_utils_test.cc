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

#include "runtime/executor/vision/vision_executor_utils.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

constexpr absl::string_view kTestVisionModelWithAdapterPath =
    "litert_lm/runtime/testdata/dummy_vision_with_adapter.litertlm";
constexpr absl::string_view kTestVisionModelWithoutAdapterPath =
    "litert_lm/runtime/testdata/dummy_vision_without_adapter.litertlm";

class FakeModelResources : public ModelResources {
 public:
  using ModelResources::HasTFLiteModel;

  FakeModelResources(const litert::Model* encoder, const litert::Model* adapter,
                     absl::Status encoder_status = absl::OkStatus(),
                     absl::Status adapter_status = absl::OkStatus())
      : encoder_(encoder),
        adapter_(adapter),
        encoder_status_(encoder_status),
        adapter_status_(adapter_status) {}

  bool HasTFLiteModel(proto::LlmMetadata::TfLiteModelType model_type) override {
    if (model_type == proto::LlmMetadata::TF_LITE_VISION_ENCODER) {
      return !encoder_status_.ok() || encoder_ != nullptr;
    }
    if (model_type == proto::LlmMetadata::TF_LITE_VISION_ADAPTER) {
      return !adapter_status_.ok() || adapter_ != nullptr;
    }
    return false;
  }

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    if (model_type == proto::LlmMetadata::TF_LITE_VISION_ENCODER) {
      if (!encoder_status_.ok()) {
        return encoder_status_;
      }
      return encoder_;
    }
    if (model_type == proto::LlmMetadata::TF_LITE_VISION_ADAPTER) {
      if (!adapter_status_.ok()) {
        return adapter_status_;
      }
      if (adapter_ == nullptr) {
        return absl::NotFoundError("Adapter not found");
      }
      return adapter_;
    }
    return absl::NotFoundError("Model not found");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }

 private:
  const litert::Model* encoder_;
  const litert::Model* adapter_;
  absl::Status encoder_status_;
  absl::Status adapter_status_;
};

TEST(VisionExecutorUtilsTest, GetPropertiesWithAdapter) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(*resources));

  // For dummy_vision_with_adapter:
  // Adapter output: [1, 5, 6] -> num_tokens_per_image = 5
  EXPECT_EQ(properties.num_tokens_per_image, 5);
}

TEST(VisionExecutorUtilsTest, GetPropertiesWithoutAdapter) {
  const std::string model_path =
      (std::filesystem::path(::testing::SrcDir()) /
       std::string(kTestVisionModelWithoutAdapterPath))
          .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(*resources));

  // For dummy_vision_without_adapter:
  // Encoder output: [1, 5, 5] -> num_tokens_per_image = 5
  EXPECT_EQ(properties.num_tokens_per_image, 5);
}

TEST(VisionExecutorUtilsTest, GetPropertiesWithoutAdapterMocked) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));

  ASSERT_OK_AND_ASSIGN(auto resources,
                       BuildLiteRtCompiledModelResources(model_assets));
  ASSERT_NE(resources, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto encoder_model,
      resources->GetTFLiteModel(proto::LlmMetadata::TF_LITE_VISION_ENCODER));

  // Hide the adapter
  FakeModelResources fake_resources(encoder_model, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(fake_resources));

  // Should fallback to encoder output: [1, 5, 5] -> num_tokens_per_image = 5
  EXPECT_EQ(properties.num_tokens_per_image, 5);
}

TEST(VisionExecutorUtilsTest, EncoderError) {
  FakeModelResources fake_resources(
      nullptr, nullptr, absl::InternalError("Encoder error"), absl::OkStatus());
  auto properties_or =
      GetVisionExecutorPropertiesFromModelResources(fake_resources);
  EXPECT_EQ(properties_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(properties_or.status().message(),
              ::testing::HasSubstr("Encoder error"));
}

TEST(VisionExecutorUtilsTest, AdapterError) {
  FakeModelResources fake_resources(nullptr, nullptr, absl::OkStatus(),
                                    absl::InternalError("Adapter error"));
  auto properties_or =
      GetVisionExecutorPropertiesFromModelResources(fake_resources);
  EXPECT_EQ(properties_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(properties_or.status().message(),
              ::testing::HasSubstr("Adapter error"));
}

std::vector<uint8_t> BuildMultiSigVisionEncoderBuffer(
    const std::vector<std::string>& signature_keys,
    const std::vector<int32_t>& patch_counts,
    const std::vector<int32_t>& token_counts) {
  flatbuffers::FlatBufferBuilder builder;
  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ABS);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  for (size_t i = 0; i < signature_keys.size(); ++i) {
    std::vector<int32_t> images_dims = {1, patch_counts[i], 768};
    std::vector<int32_t> pos_dims = {1, patch_counts[i], 2};
    std::vector<int32_t> features_dims = {1, token_counts[i], 768};

    auto pos_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(pos_dims), tflite::TensorType_INT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("positions_xy_", i)));
    auto images_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(images_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("images_", i)));
    auto features_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(features_dims),
        tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("features_", i)));
    // Deliberately place `positions_xy` at input index 0 and `images` at index
    // 1 to verify lookup by tensor name (`images`) as well.
    auto tensors_vec =
        builder.CreateVector({pos_tensor, images_tensor, features_tensor});

    std::vector<int32_t> op_inputs = {1};
    std::vector<int32_t> op_outputs = {2};
    auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                     builder.CreateVector(op_inputs),
                                     builder.CreateVector(op_outputs));
    auto ops_vec = builder.CreateVector({op});

    std::vector<int32_t> sg_inputs = {0, 1};
    std::vector<int32_t> sg_outputs = {2};
    auto subgraph = tflite::CreateSubGraph(
        builder, tensors_vec, builder.CreateVector(sg_inputs),
        builder.CreateVector(sg_outputs), ops_vec,
        builder.CreateString(absl::StrCat("subgraph_", i)));
    subgraphs.push_back(subgraph);

    auto pos_map = tflite::CreateTensorMap(
        builder, builder.CreateString("positions_xy"), 0);
    auto images_map =
        tflite::CreateTensorMap(builder, builder.CreateString("images"), 1);
    auto features_map =
        tflite::CreateTensorMap(builder, builder.CreateString("features"), 2);
    auto inputs_map_vec = builder.CreateVector({pos_map, images_map});
    auto outputs_map_vec = builder.CreateVector({features_map});

    auto sig_def = tflite::CreateSignatureDef(
        builder, inputs_map_vec, outputs_map_vec,
        builder.CreateString(signature_keys[i]), /*subgraph_index=*/i);
    sig_defs.push_back(sig_def);
  }

  auto buffer = tflite::CreateBuffer(builder);
  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector(subgraphs),
      builder.CreateString("multi_sig_vision_encoder"),
      builder.CreateVector({buffer}),
      /*metadata_buffer=*/0, /*metadata=*/0, builder.CreateVector(sig_defs));
  tflite::FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

std::vector<uint8_t> BuildMultiSigVisionAdapterBuffer(
    const std::vector<std::string>& signature_keys,
    const std::vector<int32_t>& token_counts) {
  flatbuffers::FlatBufferBuilder builder;
  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ABS);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  for (size_t i = 0; i < signature_keys.size(); ++i) {
    std::vector<int32_t> in_dims = {1, token_counts[i], 768};
    std::vector<int32_t> out_dims = {1, token_counts[i], 2560};

    auto in_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(in_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("soft_tokens_", i)));
    auto out_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(out_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("mm_embedding_", i)));
    auto tensors_vec = builder.CreateVector({in_tensor, out_tensor});

    std::vector<int32_t> op_inputs = {0};
    std::vector<int32_t> op_outputs = {1};
    auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                     builder.CreateVector(op_inputs),
                                     builder.CreateVector(op_outputs));
    auto ops_vec = builder.CreateVector({op});

    std::vector<int32_t> sg_inputs = {0};
    std::vector<int32_t> sg_outputs = {1};
    auto subgraph = tflite::CreateSubGraph(
        builder, tensors_vec, builder.CreateVector(sg_inputs),
        builder.CreateVector(sg_outputs), ops_vec,
        builder.CreateString(absl::StrCat("subgraph_", i)));
    subgraphs.push_back(subgraph);

    auto in_map = tflite::CreateTensorMap(
        builder, builder.CreateString("soft_tokens"), 0);
    auto out_map = tflite::CreateTensorMap(
        builder, builder.CreateString("mm_embedding"), 1);
    auto sig_def = tflite::CreateSignatureDef(
        builder, builder.CreateVector({in_map}),
        builder.CreateVector({out_map}),
        builder.CreateString(signature_keys[i]), /*subgraph_index=*/i);
    sig_defs.push_back(sig_def);
  }

  auto buffer = tflite::CreateBuffer(builder);
  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector(subgraphs),
      builder.CreateString("multi_sig_vision_adapter"),
      builder.CreateVector({buffer}),
      /*metadata_buffer=*/0, /*metadata=*/0, builder.CreateVector(sig_defs));
  tflite::FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

TEST(VisionExecutorUtilsTest, MultiSignatureMismatchedOrderWithAdapter) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, ::litert::Environment::Create(
                    std::vector<::litert::Environment::Option>()));
  // Reproduces Converter V2 DenseMap ordering: encoder has [vision_140,
  // vision_70] while adapter has [vision_adapter_70, vision_adapter_140].
  auto enc_buf = BuildMultiSigVisionEncoderBuffer(
      {"vision_140", "vision_70"}, {1260, 630}, {140, 70});
  auto adp_buf = BuildMultiSigVisionAdapterBuffer(
      {"vision_adapter_70", "vision_adapter_140"}, {70, 140});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto enc_model,
      ::litert::Model::CreateFromBuffer(
          env, ::litert::BufferRef<uint8_t>(enc_buf.data(), enc_buf.size())));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto adp_model,
      ::litert::Model::CreateFromBuffer(
          env, ::litert::BufferRef<uint8_t>(adp_buf.data(), adp_buf.size())));

  FakeModelResources fake_resources(&enc_model, &adp_model);
  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(fake_resources));
  EXPECT_EQ(properties.num_tokens_per_image, 140);
  EXPECT_EQ(properties.patch_num_shrink_factor, 9);
}

TEST(VisionExecutorUtilsTest, MultiSignatureAscendingOrderWithAdapter) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, ::litert::Environment::Create(
                    std::vector<::litert::Environment::Option>()));
  auto enc_buf = BuildMultiSigVisionEncoderBuffer(
      {"vision_70", "vision_140"}, {630, 1260}, {70, 140});
  auto adp_buf = BuildMultiSigVisionAdapterBuffer(
      {"vision_adapter_70", "vision_adapter_140"}, {70, 140});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto enc_model,
      ::litert::Model::CreateFromBuffer(
          env, ::litert::BufferRef<uint8_t>(enc_buf.data(), enc_buf.size())));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto adp_model,
      ::litert::Model::CreateFromBuffer(
          env, ::litert::BufferRef<uint8_t>(adp_buf.data(), adp_buf.size())));

  FakeModelResources fake_resources(&enc_model, &adp_model);
  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(fake_resources));
  EXPECT_EQ(properties.num_tokens_per_image, 140);
  EXPECT_EQ(properties.patch_num_shrink_factor, 9);
}

TEST(VisionExecutorUtilsTest, MultiSignatureWithoutAdapter) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, ::litert::Environment::Create(
                    std::vector<::litert::Environment::Option>()));
  auto enc_buf = BuildMultiSigVisionEncoderBuffer(
      {"vision_70", "vision_140"}, {630, 1260}, {70, 140});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto enc_model,
      ::litert::Model::CreateFromBuffer(
          env, ::litert::BufferRef<uint8_t>(enc_buf.data(), enc_buf.size())));

  FakeModelResources fake_resources(&enc_model, nullptr);
  ASSERT_OK_AND_ASSIGN(
      auto properties,
      GetVisionExecutorPropertiesFromModelResources(fake_resources));
  EXPECT_EQ(properties.num_tokens_per_image, 140);
  EXPECT_EQ(properties.patch_num_shrink_factor, 9);
}

TEST(VisionExecutorUtilsTest,
     MultiSignatureNoMatchingSignatureNameReturnsError) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, ::litert::Environment::Create(
                    std::vector<::litert::Environment::Option>()));
  auto enc_buf = BuildMultiSigVisionEncoderBuffer(
      {"vision_140", "vision_280"}, {1260, 2520}, {140, 280});
  auto adp_buf = BuildMultiSigVisionAdapterBuffer(
      {"vision_adapter_70", "vision_adapter_560"}, {70, 560});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto enc_model,
      ::litert::Model::CreateFromBuffer(
          env, ::litert::BufferRef<uint8_t>(enc_buf.data(), enc_buf.size())));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto adp_model,
      ::litert::Model::CreateFromBuffer(
          env, ::litert::BufferRef<uint8_t>(adp_buf.data(), adp_buf.size())));

  FakeModelResources fake_resources(&enc_model, &adp_model);
  auto properties_or =
      GetVisionExecutorPropertiesFromModelResources(fake_resources);
  EXPECT_EQ(properties_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ValidateTemporalFrameInputTest, SingleFrameSignatureNeedsNoSecondFrame) {
  const std::vector<absl::string_view> signature = {"images", "positions_xy"};
  EXPECT_TRUE(ValidateTemporalFrameInput(signature, {"images"}).ok());
  EXPECT_TRUE(
      ValidateTemporalFrameInput(signature, {"images", "positions_xy"}).ok());
}

TEST(ValidateTemporalFrameInputTest, TwoFrameSignatureRequiresSecondFrame) {
  const std::vector<absl::string_view> signature = {"images", "images_t1",
                                                    "positions_xy"};
  EXPECT_TRUE(ValidateTemporalFrameInput(
                  signature, {"images", "images_t1", "positions_xy"})
                  .ok());
  EXPECT_EQ(
      ValidateTemporalFrameInput(signature, {"images", "positions_xy"}).code(),
      absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace litert::lm
