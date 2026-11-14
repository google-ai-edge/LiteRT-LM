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

#include "runtime/core/engine_vision_utils.h"

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
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

std::vector<uint8_t> BuildDummyVisionModelBuffer(
    const std::vector<std::string>& signature_keys,
    const std::vector<int32_t>& num_tokens_vec, int feature_dim = 64) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ABS);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  for (size_t i = 0; i < signature_keys.size(); ++i) {
    const int32_t num_tokens = num_tokens_vec[i];
    std::vector<int32_t> image_dims = {1, num_tokens, 3};
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

    auto subgraph = tflite::CreateSubGraph(
        builder, tensors_vec, builder.CreateVector(op_inputs),
        builder.CreateVector(op_outputs), ops_vec,
        builder.CreateString(absl::StrCat("subgraph_", i)));
    subgraphs.push_back(subgraph);

    auto image_map =
        tflite::CreateTensorMap(builder, builder.CreateString("images"), 0);
    auto features_map =
        tflite::CreateTensorMap(builder, builder.CreateString("features"), 1);
    auto sig_def = tflite::CreateSignatureDef(
        builder, builder.CreateVector({image_map}),
        builder.CreateVector({features_map}),
        builder.CreateString(signature_keys[i]), /*subgraph_index=*/i);
    sig_defs.push_back(sig_def);
  }

  auto buffer = tflite::CreateBuffer(builder);
  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector(subgraphs),
      builder.CreateString("dummy_vision_model"),
      builder.CreateVector({buffer}), /*metadata_buffer=*/0, /*metadata=*/0,
      builder.CreateVector(sig_defs));
  tflite::FinishModelBuffer(builder, model);

  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

class FakeVisionModelResources : public ModelResources {
 public:
  using ModelResources::GetTFLiteModel;
  using ModelResources::HasTFLiteModel;

  explicit FakeVisionModelResources(
      const litert::Model* encoder_model,
      const litert::Model* adapter_model = nullptr)
      : encoder_model_(encoder_model), adapter_model_(adapter_model) {}

  bool HasTFLiteModel(proto::LlmMetadata::TfLiteModelType model_type) override {
    return HasTFLiteModel(TfLiteModelTypeToWireString(model_type));
  }

  bool HasTFLiteModel(absl::string_view model_type_str) override {
    if (model_type_str == "tf_lite_vision_encoder") {
      return encoder_model_ != nullptr;
    }
    if (model_type_str == "tf_lite_vision_adapter") {
      return adapter_model_ != nullptr;
    }
    return false;
  }

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return GetTFLiteModel(TfLiteModelTypeToWireString(model_type));
  }

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      absl::string_view model_type_str) override {
    if (model_type_str == "tf_lite_vision_encoder" &&
        encoder_model_ != nullptr) {
      return encoder_model_;
    }
    if (model_type_str == "tf_lite_vision_adapter" &&
        adapter_model_ != nullptr) {
      return adapter_model_;
    }
    return absl::NotFoundError("Model not found.");
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
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("");
  }

 private:
  const litert::Model* encoder_model_;
  const litert::Model* adapter_model_;
};

absl::StatusOr<EngineSettings> CreateVisionEngineSettings() {
  ABSL_ASSIGN_OR_RETURN(ModelAssets assets,
                        ModelAssets::Create("test_model_path"));
  return EngineSettings::CreateDefault(std::move(assets), Backend::CPU,
                                       /*vision_backend=*/Backend::CPU);
}

TEST(EngineVisionUtilsTest, SelectsSignaturesFromGemma4Metadata) {
  auto encoder_buf = BuildDummyVisionModelBuffer(
      {"vision_70", "vision_280", "vision_560"}, {70, 280, 560});
  ASSERT_OK_AND_ASSIGN(
      litert::Model encoder_model,
      CreateLiteRtModelFromBuffer(
          litert::BufferRef<uint8_t>(encoder_buf.data(), encoder_buf.size())));
  FakeVisionModelResources resources(&encoder_model);

  ASSERT_OK_AND_ASSIGN(EngineSettings settings, CreateVisionEngineSettings());
  proto::LlmMetadata metadata;
  // Default Gemma4 pooling_kernel_size is 3 -> 2520 / 9 = 280 tokens.
  metadata.mutable_llm_model_type()->mutable_gemma4()->set_max_num_patches(
      2520);

  ASSERT_OK(MaybeSelectVisionSignatures(settings, resources, &metadata));
  EXPECT_EQ(settings.GetMaxVisionTokensPerImage(), 280);
  EXPECT_THAT(
      settings.GetVisionExecutorSettings()->GetEncoderSelectedSignatures(),
      ElementsAre("vision_70", "vision_280"));
}

TEST(EngineVisionUtilsTest, SelectsSignaturesFromLfm2Metadata) {
  auto encoder_buf = BuildDummyVisionModelBuffer(
      {"vision_64", "vision_256", "vision_512"}, {64, 256, 512});
  ASSERT_OK_AND_ASSIGN(
      litert::Model encoder_model,
      CreateLiteRtModelFromBuffer(
          litert::BufferRef<uint8_t>(encoder_buf.data(), encoder_buf.size())));
  FakeVisionModelResources resources(&encoder_model);

  ASSERT_OK_AND_ASSIGN(EngineSettings settings, CreateVisionEngineSettings());
  proto::LlmMetadata metadata;
  // Default LFM2 pooling_kernel_size is 2 -> 1024 / 4 = 256 tokens.
  metadata.mutable_llm_model_type()->mutable_lfm2()->set_max_num_patches(1024);

  ASSERT_OK(MaybeSelectVisionSignatures(settings, resources, &metadata));
  EXPECT_EQ(settings.GetMaxVisionTokensPerImage(), 256);
  EXPECT_THAT(
      settings.GetVisionExecutorSettings()->GetEncoderSelectedSignatures(),
      ElementsAre("vision_64", "vision_256"));
}

TEST(EngineVisionUtilsTest, RejectsNonPositiveMaxVisionTokensPerImage) {
  FakeVisionModelResources empty_resources(/*encoder_model=*/nullptr);
  ASSERT_OK_AND_ASSIGN(EngineSettings settings, CreateVisionEngineSettings());
  settings.SetMaxVisionTokensPerImage(0);
  EXPECT_THAT(MaybeSelectVisionSignatures(settings, empty_resources, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EngineVisionUtilsTest, SkipsSelectionWhenVisionEncoderIsAbsent) {
  FakeVisionModelResources empty_resources(/*encoder_model=*/nullptr);
  ASSERT_OK_AND_ASSIGN(EngineSettings settings, CreateVisionEngineSettings());
  proto::LlmMetadata metadata;
  metadata.mutable_llm_model_type()->mutable_gemma4()->set_max_num_patches(
      2520);

  ASSERT_OK(MaybeSelectVisionSignatures(settings, empty_resources, &metadata));
  EXPECT_THAT(
      settings.GetVisionExecutorSettings()->GetEncoderSelectedSignatures(),
      IsEmpty());
}

}  // namespace
}  // namespace litert::lm
