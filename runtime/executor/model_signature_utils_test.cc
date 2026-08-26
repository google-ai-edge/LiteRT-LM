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

#include "runtime/executor/model_signature_utils.h"

#include <cstddef>
#include <cstdint>
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
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::_;  // NOLINT: Required by ASSERT_OK_AND_ASSIGN().
using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::testing::status::StatusIs;

std::vector<uint8_t> BuildDummyMultiSignatureEncoderModelBuffer(
    const std::vector<std::string>& signature_keys,
    const std::vector<int32_t>& seq_lengths, int embedding_dim) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  for (size_t i = 0; i < signature_keys.size(); ++i) {
    int32_t seq_len = seq_lengths[i];
    std::vector<int32_t> embeddings_dims = {1, seq_len, embedding_dim};
    std::vector<int32_t> mask_dims = {1, seq_len, 1};
    std::vector<int32_t> output_dims = {1, seq_len, embedding_dim};

    auto embeddings_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(embeddings_dims),
        tflite::TensorType_FLOAT32, /*buffer=*/0,
        builder.CreateString(absl::StrCat("embeddings_", i)));
    auto mask_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(mask_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("input_mask_", i)));
    auto output_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(output_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("output_tensor_", i)));
    auto tensors_vec =
        builder.CreateVector({embeddings_tensor, mask_tensor, output_tensor});

    std::vector<int32_t> op_inputs = {0, 1};
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

    auto embeddings_map =
        tflite::CreateTensorMap(builder, builder.CreateString("embeddings"), 0);
    auto mask_map =
        tflite::CreateTensorMap(builder, builder.CreateString("input_mask"), 1);
    auto output_map =
        tflite::CreateTensorMap(builder, builder.CreateString("output"), 2);
    auto inputs_map_vec = builder.CreateVector({embeddings_map, mask_map});
    auto outputs_map_vec = builder.CreateVector({output_map});

    auto sig_def = tflite::CreateSignatureDef(
        builder, inputs_map_vec, outputs_map_vec,
        builder.CreateString(signature_keys[i]), /*subgraph_index=*/i);
    sig_defs.push_back(sig_def);
  }

  auto subgraphs_vec = builder.CreateVector(subgraphs);
  auto sig_defs_vec = builder.CreateVector(sig_defs);

  auto buffer = tflite::CreateBuffer(builder);
  auto buffers_vec = builder.CreateVector({buffer});

  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, subgraphs_vec,
      builder.CreateString("dummy_encoder_model"), buffers_vec,
      /*metadata_buffer=*/0, /*metadata=*/0, sig_defs_vec);
  tflite::FinishModelBuffer(builder, model);

  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

std::vector<uint8_t> BuildDummyVisionModelBuffer(
    const std::vector<std::string>& signature_keys,
    const std::vector<int32_t>& num_tokens_vec, int feature_dim) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ABS);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  for (size_t i = 0; i < signature_keys.size(); ++i) {
    int32_t num_tokens = num_tokens_vec[i];
    std::vector<int32_t> image_dims = {1, num_tokens * 4, 3};
    std::vector<int32_t> output_dims = {1, num_tokens, feature_dim};

    auto image_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(image_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("images_", i)));
    auto features_tensor = tflite::CreateTensor(
        builder, builder.CreateVector(output_dims), tflite::TensorType_FLOAT32,
        /*buffer=*/0, builder.CreateString(absl::StrCat("features_", i)));
    auto tensors_vec = builder.CreateVector({image_tensor, features_tensor});

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

    auto image_map =
        tflite::CreateTensorMap(builder, builder.CreateString("images"), 0);
    auto features_map =
        tflite::CreateTensorMap(builder, builder.CreateString("features"), 1);
    auto inputs_map_vec = builder.CreateVector({image_map});
    auto outputs_map_vec = builder.CreateVector({features_map});

    auto sig_def = tflite::CreateSignatureDef(
        builder, inputs_map_vec, outputs_map_vec,
        builder.CreateString(signature_keys[i]), /*subgraph_index=*/i);
    sig_defs.push_back(sig_def);
  }

  auto subgraphs_vec = builder.CreateVector(subgraphs);
  auto sig_defs_vec = builder.CreateVector(sig_defs);

  auto buffer = tflite::CreateBuffer(builder);
  auto buffers_vec = builder.CreateVector({buffer});

  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, subgraphs_vec,
      builder.CreateString("dummy_vision_model"), buffers_vec,
      /*metadata_buffer=*/0, /*metadata=*/0, sig_defs_vec);
  tflite::FinishModelBuffer(builder, model);

  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

class FakeModelResources : public ModelResources {
 public:
  FakeModelResources(const litert::Model* text_model,
                     const litert::Model* vision_model)
      : text_model_(text_model), vision_model_(vision_model) {}

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    if (model_type == ModelType::kTfLiteTextEncoder) {
      if (text_model_ == nullptr) {
        return absl::NotFoundError("Text encoder model not found.");
      }
      return text_model_;
    }
    if (model_type == ModelType::kTfLiteVisionEncoder ||
        model_type == ModelType::kTfLiteVisionAdapter) {
      if (vision_model_ == nullptr) {
        return absl::NotFoundError("Vision model not found.");
      }
      return vision_model_;
    }
    return absl::NotFoundError("Model not found.");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("");
  }

 private:
  const litert::Model* text_model_;
  const litert::Model* vision_model_;
};

TEST(ModelSignatureUtilsTest, GetAvailableSignaturesForTextEncoder) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, ::litert::Environment::Create(
                    std::vector<::litert::Environment::Option>()));
  auto model_buffer = BuildDummyMultiSignatureEncoderModelBuffer(
      {"encoder_256", "encoder_512", "encoder_1024"}, {256, 512, 1024},
      /*embedding_dim=*/64);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, ::litert::Model::CreateFromBuffer(
                      env, ::litert::BufferRef<uint8_t>(model_buffer.data(),
                                                        model_buffer.size())));

  ASSERT_OK_AND_ASSIGN(
      auto signatures,
      GetAvailableSignatures(model, ModelType::kTfLiteTextEncoder));

  ASSERT_THAT(signatures, SizeIs(3));

  EXPECT_EQ(signatures[0].signature_name, "encoder_256");
  EXPECT_EQ(signatures[0].length, 256);
  EXPECT_THAT(signatures[0].input_shape, ElementsAre(1, 256, 64));
  EXPECT_THAT(signatures[0].output_shape, ElementsAre(1, 256, 64));

  EXPECT_EQ(signatures[1].signature_name, "encoder_512");
  EXPECT_EQ(signatures[1].length, 512);
  EXPECT_THAT(signatures[1].input_shape, ElementsAre(1, 512, 64));
  EXPECT_THAT(signatures[1].output_shape, ElementsAre(1, 512, 64));

  EXPECT_EQ(signatures[2].signature_name, "encoder_1024");
  EXPECT_EQ(signatures[2].length, 1024);
  EXPECT_THAT(signatures[2].input_shape, ElementsAre(1, 1024, 64));
  EXPECT_THAT(signatures[2].output_shape, ElementsAre(1, 1024, 64));
}

TEST(ModelSignatureUtilsTest, GetAvailableSignaturesForVisionModel) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, ::litert::Environment::Create(
                    std::vector<::litert::Environment::Option>()));
  auto model_buffer = BuildDummyVisionModelBuffer(
      {"vision_70", "vision_280"}, {70, 280}, /*feature_dim=*/128);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, ::litert::Model::CreateFromBuffer(
                      env, ::litert::BufferRef<uint8_t>(model_buffer.data(),
                                                        model_buffer.size())));

  ASSERT_OK_AND_ASSIGN(
      auto signatures,
      GetAvailableSignatures(model, ModelType::kTfLiteVisionEncoder));

  ASSERT_THAT(signatures, SizeIs(2));

  EXPECT_EQ(signatures[0].signature_name, "vision_70");
  EXPECT_EQ(signatures[0].length, 70);
  EXPECT_THAT(signatures[0].input_shape, ElementsAre(1, 280, 3));
  EXPECT_THAT(signatures[0].output_shape, ElementsAre(1, 70, 128));

  EXPECT_EQ(signatures[1].signature_name, "vision_280");
  EXPECT_EQ(signatures[1].length, 280);
  EXPECT_THAT(signatures[1].input_shape, ElementsAre(1, 1120, 3));
  EXPECT_THAT(signatures[1].output_shape, ElementsAre(1, 280, 128));
}

TEST(ModelSignatureUtilsTest, GetAvailableSignaturesFromModelResources) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, ::litert::Environment::Create(
                    std::vector<::litert::Environment::Option>()));
  auto text_buffer = BuildDummyMultiSignatureEncoderModelBuffer(
      {"encoder_128"}, {128}, /*embedding_dim=*/32);
  auto vision_buffer =
      BuildDummyVisionModelBuffer({"vision_280"}, {280}, /*feature_dim=*/64);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto text_model, ::litert::Model::CreateFromBuffer(
                           env, ::litert::BufferRef<uint8_t>(
                                    text_buffer.data(), text_buffer.size())));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_model,
      ::litert::Model::CreateFromBuffer(
          env, ::litert::BufferRef<uint8_t>(vision_buffer.data(),
                                            vision_buffer.size())));

  FakeModelResources resources(&text_model, &vision_model);

  ASSERT_OK_AND_ASSIGN(
      auto text_signatures,
      GetAvailableSignatures(resources, ModelType::kTfLiteTextEncoder));
  ASSERT_THAT(text_signatures, SizeIs(1));
  EXPECT_EQ(text_signatures[0].signature_name, "encoder_128");
  EXPECT_EQ(text_signatures[0].length, 128);

  ASSERT_OK_AND_ASSIGN(
      auto vision_signatures,
      GetAvailableSignatures(resources, ModelType::kTfLiteVisionEncoder));
  ASSERT_THAT(vision_signatures, SizeIs(1));
  EXPECT_EQ(vision_signatures[0].signature_name, "vision_280");
  EXPECT_EQ(vision_signatures[0].length, 280);
}

TEST(ModelSignatureUtilsTest, UnsupportedModelTypeReturnsError) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, ::litert::Environment::Create(
                    std::vector<::litert::Environment::Option>()));
  auto text_buffer = BuildDummyMultiSignatureEncoderModelBuffer(
      {"encoder_128"}, {128}, /*embedding_dim=*/32);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto text_model, ::litert::Model::CreateFromBuffer(
                           env, ::litert::BufferRef<uint8_t>(
                                    text_buffer.data(), text_buffer.size())));

  EXPECT_THAT(
      GetAvailableSignatures(text_model, ModelType::kTfLitePrefillDecode),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelSignatureUtilsTest, SelectTextEncoderSignatures_Success) {
  std::vector<SignatureInfo> signatures = {
      {.signature_name = "encoder_1024", .length = 1024},
      {.signature_name = "encoder_256", .length = 256},
      {.signature_name = "encoder_512", .length = 512},
  };

  ASSERT_OK_AND_ASSIGN(auto result, SelectTextEncoderSignatures(
                                        signatures, /*max_input_length=*/500));
  EXPECT_THAT(result.signature_names,
              ElementsAre("encoder_256", "encoder_512"));
  EXPECT_THAT(result.signature_lengths, ElementsAre(256, 512));
  EXPECT_EQ(result.max_signature_length, 512);

  // Exact match
  ASSERT_OK_AND_ASSIGN(
      auto result_exact,
      SelectTextEncoderSignatures(signatures, /*max_input_length=*/256));
  EXPECT_THAT(result_exact.signature_names, ElementsAre("encoder_256"));
  EXPECT_THAT(result_exact.signature_lengths, ElementsAre(256));
  EXPECT_EQ(result_exact.max_signature_length, 256);

  // Exceeds all signatures -> loads all
  ASSERT_OK_AND_ASSIGN(
      auto result_all,
      SelectTextEncoderSignatures(signatures, /*max_input_length=*/2000));
  EXPECT_THAT(result_all.signature_names,
              ElementsAre("encoder_256", "encoder_512", "encoder_1024"));
  EXPECT_THAT(result_all.signature_lengths, ElementsAre(256, 512, 1024));
  EXPECT_EQ(result_all.max_signature_length, 1024);
}

TEST(ModelSignatureUtilsTest,
     SelectTextEncoderSignatures_InvalidLengthReturnsError) {
  std::vector<SignatureInfo> signatures = {
      {.signature_name = "encoder_256", .length = 256},
  };
  EXPECT_THAT(SelectTextEncoderSignatures(signatures, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SelectTextEncoderSignatures(signatures, -10),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelSignatureUtilsTest,
     SelectTextEncoderSignatures_EmptySignaturesReturnsError) {
  std::vector<SignatureInfo> signatures;
  EXPECT_THAT(SelectTextEncoderSignatures(signatures, 100),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelSignatureUtilsTest, SelectVisionEncoderSignatures_Success) {
  std::vector<SignatureInfo> signatures = {
      {.signature_name = "vision_280",
       .length = 280,
       .input_shape = {1, 1120, 3}},
      {.signature_name = "vision_70", .length = 70, .input_shape = {1, 280, 3}},
      {.signature_name = "vision_140",
       .length = 140,
       .input_shape = {1, 560, 3}},
  };

  ASSERT_OK_AND_ASSIGN(auto result_70,
                       SelectVisionEncoderSignatures(signatures, 70));
  EXPECT_THAT(result_70.signature_names, ElementsAre("vision_70"));
  EXPECT_THAT(result_70.signature_lengths, ElementsAre(70));
  EXPECT_EQ(result_70.max_signature_length, 70);

  ASSERT_OK_AND_ASSIGN(auto result_100,
                       SelectVisionEncoderSignatures(signatures, 100));
  EXPECT_THAT(result_100.signature_names,
              ElementsAre("vision_70", "vision_140"));
  EXPECT_THAT(result_100.signature_lengths, ElementsAre(70, 140));
  EXPECT_EQ(result_100.max_signature_length, 140);

  // Exceeds all capacities -> loads all
  ASSERT_OK_AND_ASSIGN(auto result_all,
                       SelectVisionEncoderSignatures(signatures, 500));
  EXPECT_THAT(result_all.signature_names,
              ElementsAre("vision_70", "vision_140", "vision_280"));
  EXPECT_THAT(result_all.signature_lengths, ElementsAre(70, 140, 280));
  EXPECT_EQ(result_all.max_signature_length, 280);
}

TEST(ModelSignatureUtilsTest,
     SelectVisionEncoderSignatures_InvalidLengthReturnsError) {
  std::vector<SignatureInfo> signatures = {
      {.signature_name = "vision_70", .length = 70},
  };
  EXPECT_THAT(SelectVisionEncoderSignatures(signatures, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SelectVisionEncoderSignatures(signatures, -5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelSignatureUtilsTest,
     SelectVisionEncoderSignatures_EmptySignaturesReturnsError) {
  std::vector<SignatureInfo> signatures;
  EXPECT_THAT(SelectVisionEncoderSignatures(signatures, 100),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelSignatureUtilsTest, SelectVisionAdapterSignatures_Success) {
  std::vector<SignatureInfo> signatures = {
      {.signature_name = "adapter_280", .length = 280},
      {.signature_name = "adapter_70", .length = 70},
      {.signature_name = "adapter_140", .length = 140},
  };

  ASSERT_OK_AND_ASSIGN(auto result_70,
                       SelectVisionAdapterSignatures(
                           signatures, /*vision_tokens_per_image=*/70));
  ASSERT_TRUE(result_70.has_value());
  EXPECT_THAT(result_70->signature_names, ElementsAre("adapter_70"));
  EXPECT_THAT(result_70->signature_lengths, ElementsAre(70));
  EXPECT_EQ(result_70->max_signature_length, 70);

  ASSERT_OK_AND_ASSIGN(auto result_100,
                       SelectVisionAdapterSignatures(
                           signatures, /*vision_tokens_per_image=*/100));
  ASSERT_TRUE(result_100.has_value());
  EXPECT_THAT(result_100->signature_names,
              ElementsAre("adapter_70", "adapter_140"));
  EXPECT_THAT(result_100->signature_lengths, ElementsAre(70, 140));
  EXPECT_EQ(result_100->max_signature_length, 140);

  // Exceeds all capacities -> loads all
  ASSERT_OK_AND_ASSIGN(auto result_all,
                       SelectVisionAdapterSignatures(
                           signatures, /*vision_tokens_per_image=*/500));
  ASSERT_TRUE(result_all.has_value());
  EXPECT_THAT(result_all->signature_names,
              ElementsAre("adapter_70", "adapter_140", "adapter_280"));
  EXPECT_THAT(result_all->signature_lengths, ElementsAre(70, 140, 280));
  EXPECT_EQ(result_all->max_signature_length, 280);
}

TEST(ModelSignatureUtilsTest,
     SelectVisionAdapterSignatures_EmptyReturnsNullopt) {
  std::vector<SignatureInfo> signatures;
  ASSERT_OK_AND_ASSIGN(auto result,
                       SelectVisionAdapterSignatures(
                           signatures, /*vision_tokens_per_image=*/70));
  EXPECT_FALSE(result.has_value());
}

TEST(ModelSignatureUtilsTest,
     SelectVisionAdapterSignatures_InvalidLengthReturnsError) {
  std::vector<SignatureInfo> signatures = {
      {.signature_name = "adapter_70", .length = 70},
  };
  EXPECT_THAT(
      SelectVisionAdapterSignatures(signatures, /*vision_tokens_per_image=*/0),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      SelectVisionAdapterSignatures(signatures, /*vision_tokens_per_image=*/-1),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelSignatureUtilsTest, GetSignatureInputLength_ReturnsCorrectDimension) {
  SignatureInfo sig3d{.input_shape = {1, 280, 64}};
  EXPECT_EQ(GetSignatureInputLength(sig3d), 280);

  SignatureInfo sig2d{.input_shape = {128, 64}};
  EXPECT_EQ(GetSignatureInputLength(sig2d), 64);

  SignatureInfo sig1d{.input_shape = {512}};
  EXPECT_EQ(GetSignatureInputLength(sig1d), 512);

  SignatureInfo empty_sig;
  EXPECT_EQ(GetSignatureInputLength(empty_sig), 0);
}

}  // namespace
}  // namespace litert::lm
