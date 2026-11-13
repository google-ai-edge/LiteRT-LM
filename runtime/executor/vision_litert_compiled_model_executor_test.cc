// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/vision_litert_compiled_model_executor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::litert::BufferRef;
using ::litert::ElementType;
using ::litert::Environment;
using ::litert::Layout;
using ::litert::Model;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::lm::Backend;
using ::litert::lm::LitertLmLoader;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelResourcesLitertLm;
using ::litert::lm::VisionExecutorSettings;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

constexpr absl::string_view kTestLmModelPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";
constexpr absl::string_view kTestVisionModelWithAdapterPath =
    "litert_lm/runtime/testdata/dummy_vision_with_adapter.litertlm";
constexpr absl::string_view kTestVisionModelWithoutAdapterPath =
    "litert_lm/runtime/testdata/dummy_vision_without_adapter.litertlm";

TEST(VisionLiteRtCompiledModelExecutorTest, CreateExecutorTest) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestLmModelPath))
                                      .string();

  ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(model_path));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(scoped_file)));
  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesLitertLm::Create(std::move(loader)));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));

  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::GPU,
                                            /*adapter_backend=*/Backend::GPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  auto vision_executor =
      VisionLiteRtCompiledModelExecutor::Create(settings, env);
  EXPECT_THAT(vision_executor,
              StatusIs(absl::StatusCode::kNotFound,
                       "tf_lite_vision_encoder not found in the model."));
}

template <typename T>
absl::StatusOr<TensorBuffer> CreateTensorBuffer(
    absl::Span<T> data, RankedTensorType ranked_tensor_type) {
  auto output_expected = TensorBuffer::CreateManagedHostMemory(
      ranked_tensor_type, data.size() * sizeof(T));
  if (!output_expected) {
    return absl::InternalError("Failed to create TensorBuffer");
  }
  auto output = std::move(*output_expected);
  LITERT_RETURN_IF_ERROR(output.template Write<T>(absl::MakeSpan(data)));
  return output;
}

template <typename T>
absl::StatusOr<std::vector<T>> ReadTensorBuffer(
    litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(const size_t size,
                          tensor_type.Layout().NumElements());
  std::vector<T> data(size);
  LITERT_RETURN_IF_ERROR(tensor_buffer.Read<T>(absl::MakeSpan(data)));
  return data;
}

TEST(VisionLiteRtCompiledModelExecutorTest,
     CreateWithSharedResourcesKeepsResourcesAliveUntilDestroyed) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();
  ASSERT_OK_AND_ASSIGN(ScopedFile scoped_file, ScopedFile::Open(model_path));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LitertLmLoader> loader,
                       LitertLmLoader::Create(std::move(scoped_file)));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModelResources> litert_lm_resources,
                       ModelResourcesLitertLm::Create(std::move(loader)));
  std::shared_ptr<ModelResources> resources = std::move(litert_lm_resources);
  std::weak_ptr<ModelResources> weak_resources = resources;

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, Environment::Create(std::vector<Environment::Option>()));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VisionLiteRtCompiledModelExecutor> vision_executor,
      VisionLiteRtCompiledModelExecutor::Create(settings, env, resources));

  // The executor keeps the resources alive after the caller releases them,
  // e.g. after another executor sharing them is destroyed.
  resources.reset();
  EXPECT_FALSE(weak_resources.expired());

  std::vector<float> input_data(10 * 8, 1.0f);
  RankedTensorType input_tensor_type(ElementType::Float32,
                                     Layout(litert::Dimensions{1, 10, 8}));
  ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      CreateTensorBuffer<float>(absl::MakeSpan(input_data), input_tensor_type));
  EXPECT_OK(vision_executor->Encode(input_buffer));

  // The resources are released with the last executor using them.
  vision_executor.reset();
  EXPECT_TRUE(weak_resources.expired());
}

TEST(VisionLiteRtCompiledModelExecutorTest,
     CreateWithNullSharedResourcesFails) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, Environment::Create(std::vector<Environment::Option>()));

  EXPECT_THAT(VisionLiteRtCompiledModelExecutor::Create(
                  settings, env, std::shared_ptr<ModelResources>()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(VisionLiteRtCompiledModelExecutorTest, GetSupportedVariants) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VisionLiteRtCompiledModelExecutor> vision_executor,
      VisionLiteRtCompiledModelExecutor::Create(settings, env));

  // The dummy vision encoder has a single signature with an input of
  // [1, 10, 8] (10 patches) and an output of [1, 5, 5] (5 soft tokens).
  ASSERT_OK_AND_ASSIGN(std::vector<VisionEncoderVariant> variants,
                       vision_executor->GetSupportedVariants());
  ASSERT_EQ(variants.size(), 1);
  EXPECT_FALSE(variants[0].signature_name.empty());
  EXPECT_EQ(variants[0].max_num_patches, 10);
  EXPECT_EQ(variants[0].num_soft_tokens, 5);
}

// Builds a vision encoder model with one signature per entry of
// `signature_keys`. Signature i maps an input named `image_input_names[i]` of
// [1, num_tokens[i] * 4, 3] (patches), or `image_dims` if set, to a `features`
// output of [1, num_tokens[i], 8] (soft tokens).
std::vector<uint8_t> BuildDummyVisionEncoderBuffer(
    const std::vector<std::string>& signature_keys,
    const std::vector<int32_t>& num_tokens,
    const std::vector<std::string>& image_input_names,
    std::optional<std::vector<int32_t>> image_dims_override = std::nullopt) {
  flatbuffers::FlatBufferBuilder builder;
  auto opcodes = builder.CreateVector(
      {tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ABS)});
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> signature_defs;
  for (size_t i = 0; i < signature_keys.size(); ++i) {
    const std::vector<int32_t> image_dims = image_dims_override.value_or(
        std::vector<int32_t>{1, num_tokens[i] * 4, 3});
    const std::vector<int32_t> features_dims = {1, num_tokens[i], 8};
    auto tensors = builder.CreateVector({
        tflite::CreateTensor(builder, builder.CreateVector(image_dims),
                             tflite::TensorType_FLOAT32, /*buffer=*/0,
                             builder.CreateString(absl::StrCat("images_", i))),
        tflite::CreateTensor(
            builder, builder.CreateVector(features_dims),
            tflite::TensorType_FLOAT32, /*buffer=*/0,
            builder.CreateString(absl::StrCat("features_", i))),
    });
    const std::vector<int32_t> inputs = {0};
    const std::vector<int32_t> outputs = {1};
    auto ops = builder.CreateVector({tflite::CreateOperator(
        builder, /*opcode_index=*/0, builder.CreateVector(inputs),
        builder.CreateVector(outputs))});
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, tensors, builder.CreateVector(inputs),
        builder.CreateVector(outputs), ops,
        builder.CreateString(absl::StrCat("subgraph_", i))));
    auto input_map = builder.CreateVector({tflite::CreateTensorMap(
        builder, builder.CreateString(image_input_names[i]), 0)});
    auto output_map = builder.CreateVector({tflite::CreateTensorMap(
        builder, builder.CreateString("features"), 1)});
    signature_defs.push_back(tflite::CreateSignatureDef(
        builder, input_map, output_map, builder.CreateString(signature_keys[i]),
        /*subgraph_index=*/i));
  }
  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes, builder.CreateVector(subgraphs),
      builder.CreateString("dummy_vision_encoder"),
      builder.CreateVector({tflite::CreateBuffer(builder)}),
      /*metadata_buffer=*/0, /*metadata=*/0,
      builder.CreateVector(signature_defs));
  tflite::FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

class GetVisionEncoderVariantsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    LITERT_ASSERT_OK_AND_ASSIGN(
        env_, Environment::Create(std::vector<Environment::Option>()));
  }

  // Creates a model from `buffer`, which must outlive the model.
  absl::StatusOr<Model> CreateModel(const std::vector<uint8_t>& buffer) {
    LITERT_ASSIGN_OR_RETURN(
        Model model,
        Model::CreateFromBuffer(
            *env_, BufferRef<uint8_t>(buffer.data(), buffer.size())));
    return model;
  }

  std::optional<Environment> env_;
};

TEST_F(GetVisionEncoderVariantsTest, MultiSignatureSortedBySoftTokens) {
  // Signatures without the "vision_" prefix are never selected by Encode().
  const std::vector<uint8_t> buffer = BuildDummyVisionEncoderBuffer(
      {"vision_256", "vision_64", "serving_default", "vision_128"},
      {256, 64, 32, 128}, {"images", "images", "images", "images"});
  ASSERT_OK_AND_ASSIGN(Model model, CreateModel(buffer));

  EXPECT_THAT(GetVisionEncoderVariants(model, /*selected_signatures=*/{}),
              IsOkAndHolds(ElementsAre(
                  VisionEncoderVariant{.signature_name = "vision_64",
                                       .max_num_patches = 256,
                                       .num_soft_tokens = 64},
                  VisionEncoderVariant{.signature_name = "vision_128",
                                       .max_num_patches = 512,
                                       .num_soft_tokens = 128},
                  VisionEncoderVariant{.signature_name = "vision_256",
                                       .max_num_patches = 1024,
                                       .num_soft_tokens = 256})));
}

TEST_F(GetVisionEncoderVariantsTest, MultiSignatureFilteredBySelection) {
  const std::vector<uint8_t> buffer = BuildDummyVisionEncoderBuffer(
      {"vision_256", "vision_64", "vision_128"}, {256, 64, 128},
      {"images", "images", "images"});
  ASSERT_OK_AND_ASSIGN(Model model, CreateModel(buffer));

  EXPECT_THAT(GetVisionEncoderVariants(model, {"vision_256", "vision_64"}),
              IsOkAndHolds(ElementsAre(
                  VisionEncoderVariant{.signature_name = "vision_64",
                                       .max_num_patches = 256,
                                       .num_soft_tokens = 64},
                  VisionEncoderVariant{.signature_name = "vision_256",
                                       .max_num_patches = 1024,
                                       .num_soft_tokens = 256})));
  // A selected signature without the "vision_" prefix is not a candidate.
  EXPECT_THAT(GetVisionEncoderVariants(model, {"serving_default"}),
              IsOkAndHolds(IsEmpty()));
}

TEST_F(GetVisionEncoderVariantsTest, SingleSignatureIsAlwaysUsed) {
  // Like Encode(), a single-signature model uses its only signature regardless
  // of its name and of the selected signatures, and an unnamed image input.
  const std::vector<uint8_t> buffer = BuildDummyVisionEncoderBuffer(
      {"serving_default"}, {16}, {"pixel_values"});
  ASSERT_OK_AND_ASSIGN(Model model, CreateModel(buffer));

  const std::vector<VisionEncoderVariant> expected = {
      {.signature_name = "serving_default",
       .max_num_patches = 64,
       .num_soft_tokens = 16}};
  EXPECT_THAT(GetVisionEncoderVariants(model, /*selected_signatures=*/{}),
              IsOkAndHolds(expected));
  EXPECT_THAT(GetVisionEncoderVariants(model, {"vision_64"}),
              IsOkAndHolds(expected));
}

TEST_F(GetVisionEncoderVariantsTest, NonPatchifiedImageInputFails) {
  // The number of patches can only be read from a [batch, num_patches,
  // patch_dim] input with a positive num_patches, not e.g. from a [batch,
  // height, width, channels] image or a rank 2 tensor. The error reports the
  // actual shape.
  for (const auto& [image_dims, shape] :
       std::vector<std::pair<std::vector<int32_t>, std::string>>{
           {{1, 224, 224, 3}, "[1, 224, 224, 3]"},
           {{64, 3}, "[64, 3]"},
           {{1, 0, 3}, "[1, 0, 3]"},
       }) {
    SCOPED_TRACE(shape);
    const std::vector<uint8_t> buffer = BuildDummyVisionEncoderBuffer(
        {"serving_default"}, {16}, {"images"}, image_dims);
    ASSERT_OK_AND_ASSIGN(Model model, CreateModel(buffer));
    EXPECT_THAT(GetVisionEncoderVariants(model, /*selected_signatures=*/{}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         AllOf(HasSubstr("[batch, num_patches, patch_dim]"),
                               HasSubstr(absl::StrCat("got ", shape)))));
  }
}

TEST_F(GetVisionEncoderVariantsTest, MultiSignatureWithoutImagesInputFails) {
  // Encode() feeds a multi-signature model by input name, so the patch input
  // must be named `images`.
  const std::vector<uint8_t> buffer = BuildDummyVisionEncoderBuffer(
      {"vision_64", "vision_128"}, {64, 128}, {"images", "pixel_values"});
  ASSERT_OK_AND_ASSIGN(Model model, CreateModel(buffer));

  EXPECT_THAT(GetVisionEncoderVariants(model, /*selected_signatures=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(VisionLiteRtCompiledModelExecutorTest, EncodeWithCancellation) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestVisionModelWithAdapterPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      auto vision_executor,
      VisionLiteRtCompiledModelExecutor::Create(settings, env));

  std::vector<float> input_data(10 * 8, 1.0f);
  RankedTensorType input_tensor_type(ElementType::Float32,
                                     Layout(litert::Dimensions{1, 10, 8}));
  ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      CreateTensorBuffer<float>(absl::MakeSpan(input_data), input_tensor_type));
  ASSERT_OK_AND_ASSIGN(auto expected_output,
                       vision_executor->Encode(input_buffer));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * expected_embeddings,
                       expected_output.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto expected_vector,
                       ReadTensorBuffer<float>(*expected_embeddings));

  // Not cancelled: same output as without a cancellation callback.
  ASSERT_OK_AND_ASSIGN(
      auto output, vision_executor->Encode(input_buffer, [] { return false; }));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * embeddings,
                       output.GetMutableEmbeddingsPtr());
  EXPECT_THAT(ReadTensorBuffer<float>(*embeddings),
              IsOkAndHolds(expected_vector));

  // Cancelled before any input is copied: `should_cancel` is called once.
  int num_checks = 0;
  EXPECT_THAT(vision_executor->Encode(input_buffer,
                                      [&] {
                                        ++num_checks;
                                        return true;
                                      }),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 1);

  // Cancelled at the check right before the vision encoder runs.
  num_checks = 0;
  EXPECT_THAT(
      vision_executor->Encode(input_buffer, [&] { return ++num_checks == 2; }),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 2);

  // Cancelled by a one-shot `should_cancel` during the runs (per-op checks):
  // the cancellation is latched, and `should_cancel` is not called again.
  num_checks = 0;
  EXPECT_THAT(
      vision_executor->Encode(input_buffer, [&] { return ++num_checks == 3; }),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 3);

  // The executor is usable after a cancelled call, and produces the same
  // output.
  ASSERT_OK_AND_ASSIGN(output, vision_executor->Encode(input_buffer));
  ASSERT_OK_AND_ASSIGN(embeddings, output.GetMutableEmbeddingsPtr());
  EXPECT_THAT(ReadTensorBuffer<float>(*embeddings),
              IsOkAndHolds(expected_vector));
}

// Tests that the cancellation state of a cancellable call (its
// CancellationLatch and `should_cancel`) is scoped to the call: it is neither
// used by nor leaked into the following calls, of both the vision encoder and
// the vision adapter.
class VisionCancellationScopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string model_path =
        (std::filesystem::path(::testing::SrcDir()) /
         std::string(kTestVisionModelWithAdapterPath))
            .string();
    ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                         ModelAssets::Create(model_path));
    ASSERT_OK_AND_ASSIGN(VisionExecutorSettings settings,
                         VisionExecutorSettings::CreateDefault(
                             model_assets,
                             /*encoder_backend=*/Backend::CPU,
                             /*adapter_backend=*/Backend::CPU));
    LITERT_ASSERT_OK_AND_ASSIGN(
        env_, Environment::Create(std::vector<Environment::Option>()));
    ASSERT_OK_AND_ASSIGN(
        executor_, VisionLiteRtCompiledModelExecutor::Create(settings, *env_));
    RankedTensorType input_tensor_type(ElementType::Float32,
                                       Layout(litert::Dimensions{1, 10, 8}));
    ASSERT_OK_AND_ASSIGN(input_buffer_,
                         CreateTensorBuffer<float>(absl::MakeSpan(input_data_),
                                                   input_tensor_type));
    ASSERT_OK_AND_ASSIGN(expected_output_, EncodeToVector(std::nullopt));
  }

  // Encodes the input, with `should_cancel` if set, and returns the
  // embeddings.
  absl::StatusOr<std::vector<float>> EncodeToVector(
      std::optional<absl::FunctionRef<bool()>> should_cancel) {
    ABSL_ASSIGN_OR_RETURN(
        ExecutorVisionData output,
        should_cancel.has_value()
            ? executor_->Encode(*input_buffer_, *should_cancel)
            : executor_->Encode(*input_buffer_));
    ABSL_ASSIGN_OR_RETURN(TensorBuffer * embeddings,
                          output.GetMutableEmbeddingsPtr());
    return ReadTensorBuffer<float>(*embeddings);
  }

  // Checks that a plain call still works and produces the expected output.
  void ExpectPlainCallWorks() {
    EXPECT_THAT(EncodeToVector(std::nullopt), IsOkAndHolds(expected_output_));
  }

  std::optional<Environment> env_;
  std::unique_ptr<VisionLiteRtCompiledModelExecutor> executor_;
  std::vector<float> input_data_ = std::vector<float>(10 * 8, 1.0f);
  std::optional<TensorBuffer> input_buffer_;
  std::vector<float> expected_output_;
};

TEST_F(VisionCancellationScopeTest, LaterCallsDontUseTheCallback) {
  bool cancel = false;
  int num_checks = 0;
  auto should_cancel = [&] {
    ++num_checks;
    return cancel;
  };
  EXPECT_THAT(EncodeToVector(should_cancel), IsOkAndHolds(expected_output_));
  ASSERT_GT(num_checks, 0);
  const int num_checks_after_call = num_checks;

  // The callback, still alive, would now cancel. Neither the pre-run checks
  // nor the per-op checks of the following plain calls call it.
  cancel = true;
  for (int i = 0; i < 3; ++i) {
    ExpectPlainCallWorks();
  }
  EXPECT_EQ(num_checks, num_checks_after_call);
}

TEST_F(VisionCancellationScopeTest, CallbackCanBeDestroyedRightAfterTheCall) {
  // Both after a completed and after a cancelled call. If the latch or the
  // callback were still referenced (e.g. by the per-op checks registered with
  // LiteRT), the following calls would use freed memory (caught by ASan).
  for (const bool cancel_during_run : {false, true}) {
    SCOPED_TRACE(cancel_during_run);
    int num_checks = 0;
    auto should_cancel = std::make_unique<std::function<bool()>>(
        [&] { return ++num_checks > 2 && cancel_during_run; });
    absl::StatusOr<std::vector<float>> result = EncodeToVector(*should_cancel);
    if (cancel_during_run) {
      EXPECT_THAT(result, StatusIs(absl::StatusCode::kCancelled));
    } else {
      EXPECT_THAT(result, IsOkAndHolds(expected_output_));
    }
    should_cancel.reset();

    ExpectPlainCallWorks();
    EXPECT_THAT(EncodeToVector([] { return false; }),
                IsOkAndHolds(expected_output_));
  }
}

TEST_F(VisionCancellationScopeTest, EachCallHasItsOwnLatch) {
  // A call cancelled during the runs by a one-shot callback: its latch is
  // cancelled.
  int num_checks = 0;
  EXPECT_THAT(EncodeToVector([&] { return ++num_checks == 3; }),
              StatusIs(absl::StatusCode::kCancelled));

  // The next call starts with a fresh latch, which calls its own callback
  // (at entry and before the runs) instead of reporting the previous
  // cancellation.
  num_checks = 0;
  EXPECT_THAT(EncodeToVector([&] {
                ++num_checks;
                return false;
              }),
              IsOkAndHolds(expected_output_));
  EXPECT_GE(num_checks, 2);

  // And a call cancelled at entry is cancelled by its own callback only.
  num_checks = 0;
  EXPECT_THAT(EncodeToVector([&] {
                ++num_checks;
                return true;
              }),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 1);
  ExpectPlainCallWorks();
}

TEST(VisionLiteRtCompiledModelExecutorTest, EncodeInputMapsWithCancellation) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestVisionModelWithAdapterPath))
                                      .string();
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      auto vision_executor,
      VisionLiteRtCompiledModelExecutor::Create(settings, env));

  std::vector<float> input_data(10 * 8, 1.0f);
  RankedTensorType input_tensor_type(ElementType::Float32,
                                     Layout(litert::Dimensions{1, 10, 8}));
  ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      CreateTensorBuffer<float>(absl::MakeSpan(input_data), input_tensor_type));
  absl::flat_hash_map<std::string, TensorBuffer> input_maps;
  input_maps.insert_or_assign("images", std::move(input_buffer));

  ASSERT_OK_AND_ASSIGN(auto expected_output,
                       vision_executor->Encode(input_maps));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * expected_embeddings,
                       expected_output.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto expected_vector,
                       ReadTensorBuffer<float>(*expected_embeddings));

  // Not cancelled: same output as without a cancellation callback.
  ASSERT_OK_AND_ASSIGN(
      auto output, vision_executor->Encode(input_maps, [] { return false; }));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * embeddings,
                       output.GetMutableEmbeddingsPtr());
  EXPECT_THAT(ReadTensorBuffer<float>(*embeddings),
              IsOkAndHolds(expected_vector));

  // Cancelled before any input is copied.
  int num_checks = 0;
  EXPECT_THAT(vision_executor->Encode(input_maps,
                                      [&] {
                                        ++num_checks;
                                        return true;
                                      }),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 1);

  // Cancelled at the check right before the vision encoder runs.
  num_checks = 0;
  EXPECT_THAT(
      vision_executor->Encode(input_maps, [&] { return ++num_checks == 2; }),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 2);

  // Cancelled by a one-shot `should_cancel` during the runs.
  num_checks = 0;
  EXPECT_THAT(
      vision_executor->Encode(input_maps, [&] { return ++num_checks == 3; }),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(num_checks, 3);

  // The executor is usable after a cancelled call.
  ASSERT_OK_AND_ASSIGN(output, vision_executor->Encode(input_maps));
  ASSERT_OK_AND_ASSIGN(embeddings, output.GetMutableEmbeddingsPtr());
  EXPECT_THAT(ReadTensorBuffer<float>(*embeddings),
              IsOkAndHolds(expected_vector));
}

TEST(VisionLiteRtCompiledModelExecutorTest,
     CreateAndRunExecutorWithAdapterTest) {
  const std::string& model_path = (std::filesystem::path(::testing::SrcDir()) /
                                   std::string(kTestVisionModelWithAdapterPath))
                                      .string();

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));

  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  ASSERT_OK_AND_ASSIGN(
      auto vision_executor,
      VisionLiteRtCompiledModelExecutor::Create(settings, env));

  // Encoder input: [1, 10, 8]
  std::vector<float> input_data(10 * 8, 1.0f);
  Layout input_layout(litert::Dimensions{1, 10, 8});
  RankedTensorType input_tensor_type(ElementType::Float32,
                                     std::move(input_layout));
  ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      CreateTensorBuffer<float>(absl::MakeSpan(input_data), input_tensor_type));

  // Run Encode
  ASSERT_OK_AND_ASSIGN(auto output_data, vision_executor->Encode(input_buffer));

  // Verify output shape.
  // Encoder output is [1, 5, 5] (shrink factor 2, dense 5)
  // Adapter output is [1, 5, 6] (dense 6)
  // So final output should be [1, 5, 6] -> 30 elements.
  ASSERT_OK_AND_ASSIGN(TensorBuffer * output_embeddings,
                       output_data.GetMutableEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_type,
                              output_embeddings->TensorType());
  EXPECT_EQ(output_type.Layout().Dimensions().size(), 3);
  EXPECT_EQ(output_type.Layout().Dimensions()[0], 1);
  EXPECT_EQ(output_type.Layout().Dimensions()[1], 5);
  EXPECT_EQ(output_type.Layout().Dimensions()[2], 6);

  ASSERT_OK_AND_ASSIGN(auto output_vector,
                       ReadTensorBuffer<float>(*output_embeddings));
  EXPECT_EQ(output_vector.size(), 30);

  // Verify buffer reuse across repeated single-buffer Encode() calls.
  std::vector<float> second_input_data(10 * 8, 2.0f);
  ASSERT_OK_AND_ASSIGN(
      auto second_input_buffer,
      CreateTensorBuffer<float>(absl::MakeSpan(second_input_data),
                                input_tensor_type));
  ASSERT_OK_AND_ASSIGN(auto second_output_data,
                       vision_executor->Encode(second_input_buffer));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * second_output_embeddings,
                       second_output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto second_output_vector,
                       ReadTensorBuffer<float>(*second_output_embeddings));
  EXPECT_NE(second_output_vector, output_vector);

  ASSERT_OK_AND_ASSIGN(auto repeat_output_data,
                       vision_executor->Encode(input_buffer));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * repeat_output_embeddings,
                       repeat_output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto repeat_output_vector,
                       ReadTensorBuffer<float>(*repeat_output_embeddings));
  EXPECT_EQ(repeat_output_vector, output_vector);

  // Verify buffer reuse across repeated map-based Encode() calls.
  absl::flat_hash_map<std::string, TensorBuffer> input_maps;
  input_maps.insert_or_assign("images", std::move(input_buffer));
  ASSERT_OK_AND_ASSIGN(auto map_output_data,
                       vision_executor->Encode(input_maps));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * map_output_embeddings,
                       map_output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto map_output_vector,
                       ReadTensorBuffer<float>(*map_output_embeddings));
  EXPECT_EQ(map_output_vector, output_vector);

  absl::flat_hash_map<std::string, TensorBuffer> second_input_maps;
  second_input_maps.insert_or_assign("images", std::move(second_input_buffer));
  ASSERT_OK_AND_ASSIGN(auto second_map_output_data,
                       vision_executor->Encode(second_input_maps));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * second_map_output_embeddings,
                       second_map_output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto second_map_output_vector,
                       ReadTensorBuffer<float>(*second_map_output_embeddings));
  EXPECT_EQ(second_map_output_vector, second_output_vector);
}

TEST(VisionLiteRtCompiledModelExecutorTest,
     CreateAndRunExecutorWithoutAdapterTest) {
  const std::string& model_path =
      (std::filesystem::path(::testing::SrcDir()) /
       std::string(kTestVisionModelWithoutAdapterPath))
          .string();

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));

  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  ASSERT_OK_AND_ASSIGN(
      auto vision_executor,
      VisionLiteRtCompiledModelExecutor::Create(settings, env));

  // Encoder input: [1, 10, 8]
  std::vector<float> input_data(10 * 8, 1.0f);
  Layout input_layout(litert::Dimensions{1, 10, 8});
  RankedTensorType input_tensor_type(ElementType::Float32,
                                     std::move(input_layout));
  ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      CreateTensorBuffer<float>(absl::MakeSpan(input_data), input_tensor_type));

  // Run Encode
  ASSERT_OK_AND_ASSIGN(auto output_data, vision_executor->Encode(input_buffer));

  // Verify output shape.
  // Encoder output is [1, 5, 5] (shrink factor 2, dense 5)
  // Adapter is missing, so final output should be [1, 5, 5] -> 25 elements.
  ASSERT_OK_AND_ASSIGN(TensorBuffer * output_embeddings,
                       output_data.GetMutableEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_type,
                              output_embeddings->TensorType());
  EXPECT_EQ(output_type.Layout().Dimensions().size(), 3);
  EXPECT_EQ(output_type.Layout().Dimensions()[0], 1);
  EXPECT_EQ(output_type.Layout().Dimensions()[1], 5);
  EXPECT_EQ(output_type.Layout().Dimensions()[2], 5);

  ASSERT_OK_AND_ASSIGN(auto output_vector,
                       ReadTensorBuffer<float>(*output_embeddings));
  EXPECT_EQ(output_vector.size(), 25);

  // Verify buffer reuse across repeated single-buffer Encode() calls.
  std::vector<float> second_input_data(10 * 8, 2.0f);
  ASSERT_OK_AND_ASSIGN(
      auto second_input_buffer,
      CreateTensorBuffer<float>(absl::MakeSpan(second_input_data),
                                input_tensor_type));
  ASSERT_OK_AND_ASSIGN(auto second_output_data,
                       vision_executor->Encode(second_input_buffer));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * second_output_embeddings,
                       second_output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto second_output_vector,
                       ReadTensorBuffer<float>(*second_output_embeddings));
  EXPECT_NE(second_output_vector, output_vector);

  ASSERT_OK_AND_ASSIGN(auto repeat_output_data,
                       vision_executor->Encode(input_buffer));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * repeat_output_embeddings,
                       repeat_output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto repeat_output_vector,
                       ReadTensorBuffer<float>(*repeat_output_embeddings));
  EXPECT_EQ(repeat_output_vector, output_vector);

  // Verify buffer reuse across repeated map-based Encode() calls.
  absl::flat_hash_map<std::string, TensorBuffer> input_maps;
  input_maps.insert_or_assign("images", std::move(input_buffer));
  ASSERT_OK_AND_ASSIGN(auto map_output_data,
                       vision_executor->Encode(input_maps));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * map_output_embeddings,
                       map_output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto map_output_vector,
                       ReadTensorBuffer<float>(*map_output_embeddings));
  EXPECT_EQ(map_output_vector, output_vector);

  absl::flat_hash_map<std::string, TensorBuffer> second_input_maps;
  second_input_maps.insert_or_assign("images", std::move(second_input_buffer));
  ASSERT_OK_AND_ASSIGN(auto second_map_output_data,
                       vision_executor->Encode(second_input_maps));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * second_map_output_embeddings,
                       second_map_output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto second_map_output_vector,
                       ReadTensorBuffer<float>(*second_map_output_embeddings));
  EXPECT_EQ(second_map_output_vector, second_output_vector);
}

TEST(VisionLiteRtCompiledModelExecutorTest,
     CreateExecutorWithSelectedSignaturesTest) {
  const std::string& model_path =
      (std::filesystem::path(::testing::SrcDir()) /
       std::string(kTestVisionModelWithoutAdapterPath))
          .string();

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));

  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::CPU));
  settings.SetEncoderSelectedSignatures({"serving_default"});
  settings.SetAdapterSelectedSignatures({"serving_default"});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));

  ASSERT_OK_AND_ASSIGN(
      auto vision_executor,
      VisionLiteRtCompiledModelExecutor::Create(settings, env));

  std::vector<float> input_data(10 * 8, 1.0f);
  Layout input_layout(litert::Dimensions{1, 10, 8});
  RankedTensorType input_tensor_type(ElementType::Float32,
                                     std::move(input_layout));
  ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      CreateTensorBuffer<float>(absl::MakeSpan(input_data), input_tensor_type));

  ASSERT_OK_AND_ASSIGN(auto output_data, vision_executor->Encode(input_buffer));
  ASSERT_OK_AND_ASSIGN(TensorBuffer * output_embeddings,
                       output_data.GetMutableEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto output_vector,
                       ReadTensorBuffer<float>(*output_embeddings));
  EXPECT_EQ(output_vector.size(), 25);
}

TEST(VisionLiteRtCompiledModelExecutorTest,
     CreateExecutorGpuOptionConfigurationPaths) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, Environment::Create(std::vector<Environment::Option>()));

  // Exercise GPU option setup for the vision encoder. Headless Linux hosts may
  // lack an OpenCL ICD when compiling the GPU delegate, so either successful
  // creation or a delegate unavailability error is valid, whereas
  // kInvalidArgument indicates invalid compilation options (e.g., missing
  // hardware accelerator flags or unsupported backend dispatch).
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings encoder_gpu_settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::GPU,
                                            /*adapter_backend=*/Backend::CPU));
  encoder_gpu_settings.SetCacheDir(":nocache");
  absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
      encoder_gpu_executor =
          VisionLiteRtCompiledModelExecutor::Create(encoder_gpu_settings, env);
  if (!encoder_gpu_executor.ok()) {
    EXPECT_THAT(encoder_gpu_executor,
                Not(StatusIs(absl::StatusCode::kInvalidArgument)));
  }

  // Exercise GPU option setup (including IR model toggle) for the vision
  // adapter alongside a CPU encoder.
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings adapter_gpu_settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::GPU));
  adapter_gpu_settings.SetCacheDir(":nocache");
  adapter_gpu_settings.SetUseIrModel(true);
  absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
      adapter_gpu_executor =
          VisionLiteRtCompiledModelExecutor::Create(adapter_gpu_settings, env);
  if (adapter_gpu_executor.ok()) {
    std::vector<float> input_data(10 * 8, 1.0f);
    RankedTensorType input_tensor_type(ElementType::Float32,
                                       Layout(litert::Dimensions{1, 10, 8}));
    ASSERT_OK_AND_ASSIGN(TensorBuffer input_buffer,
                         CreateTensorBuffer<float>(absl::MakeSpan(input_data),
                                                   input_tensor_type));
    EXPECT_OK((*adapter_gpu_executor)->Encode(input_buffer));
  } else {
    EXPECT_THAT(adapter_gpu_executor,
                Not(StatusIs(absl::StatusCode::kInvalidArgument)));
  }
}

#if !defined(LITERT_DISABLE_NPU)
TEST(VisionLiteRtCompiledModelExecutorTest,
     CreateExecutorNpuOptionConfigurationDoesNotQueryCpuWeightCache) {
  const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                  std::string(kTestVisionModelWithAdapterPath))
                                     .string();
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::NPU,
                                            /*adapter_backend=*/Backend::NPU));
  int cache_provider_calls = 0;
  settings.SetCacheFileProvider(
      [&](const ExecutorSettingsBase::CacheFileRequest&)
          -> absl::StatusOr<std::shared_ptr<ScopedFile>> {
        ++cache_provider_calls;
        return absl::UnimplementedError("Unexpected CPU cache lookup on NPU.");
      });
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, Environment::Create(std::vector<Environment::Option>()));

  absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
      vision_executor =
          VisionLiteRtCompiledModelExecutor::Create(settings, env);
  EXPECT_EQ(cache_provider_calls, 0);
  if (vision_executor.ok()) {
    std::vector<float> input_data(10 * 8, 1.0f);
    RankedTensorType input_tensor_type(ElementType::Float32,
                                       Layout(litert::Dimensions{1, 10, 8}));
    ASSERT_OK_AND_ASSIGN(TensorBuffer input_buffer,
                         CreateTensorBuffer<float>(absl::MakeSpan(input_data),
                                                   input_tensor_type));
    ASSERT_OK_AND_ASSIGN(ExecutorVisionData output_data,
                         (*vision_executor)->Encode(input_buffer));
    ASSERT_OK_AND_ASSIGN(TensorBuffer * output_embeddings,
                         output_data.GetMutableEmbeddingsPtr());
    ASSERT_OK_AND_ASSIGN(std::vector<float> output_vector,
                         ReadTensorBuffer<float>(*output_embeddings));
    EXPECT_EQ(output_vector.size(), 30);
  } else {
    EXPECT_THAT(vision_executor,
                Not(StatusIs(absl::StatusCode::kInvalidArgument)));
  }
}
#endif  // !defined(LITERT_DISABLE_NPU)

TEST(VisionLiteRtCompiledModelExecutorTest,
     EncodeInputMapsSubCapacityCapsOutputRowsAndZerosAdapterTail) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, Environment::Create(std::vector<Environment::Option>()));
  RankedTensorType full_tensor_type(ElementType::Float32,
                                    Layout(litert::Dimensions{1, 10, 8}));
  RankedTensorType sub_tensor_type(ElementType::Float32,
                                   Layout(litert::Dimensions{1, 4, 8}));
  std::vector<float> full_input_data(10 * 8, 2.0f);
  std::vector<float> sub_input_data(4 * 8, 1.0f);

  for (const bool with_adapter : {true, false}) {
    SCOPED_TRACE(with_adapter);
    const absl::string_view relative_path =
        with_adapter ? kTestVisionModelWithAdapterPath
                     : kTestVisionModelWithoutAdapterPath;
    const int expected_output_dim = with_adapter ? 6 : 5;
    const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                    std::string(relative_path))
                                       .string();
    ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                         ModelAssets::Create(model_path));
    ASSERT_OK_AND_ASSIGN(VisionExecutorSettings settings,
                         VisionExecutorSettings::CreateDefault(
                             model_assets,
                             /*encoder_backend=*/Backend::CPU,
                             /*adapter_backend=*/Backend::CPU));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VisionLiteRtCompiledModelExecutor> reused_executor,
        VisionLiteRtCompiledModelExecutor::Create(settings, env));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VisionLiteRtCompiledModelExecutor> fresh_executor,
        VisionLiteRtCompiledModelExecutor::Create(settings, env));

    // Populate all 5 token rows of the cached encoder/adapter buffers first.
    ASSERT_OK_AND_ASSIGN(
        TensorBuffer full_buffer,
        CreateTensorBuffer<float>(absl::MakeSpan(full_input_data),
                                  full_tensor_type));
    absl::flat_hash_map<std::string, TensorBuffer> full_maps;
    full_maps.insert_or_assign("images", std::move(full_buffer));
    ASSERT_OK(reused_executor->Encode(full_maps));

    // Now encode a 4-patch input (shrink factor 2 -> 2 visual tokens < 5
    // capacity) and verify both row capping and zero tail-padding against a
    // freshly initialized executor.
    ASSERT_OK(reused_executor->StartProfiling());
    ASSERT_OK_AND_ASSIGN(TensorBuffer sub_buffer,
                         CreateTensorBuffer<float>(
                             absl::MakeSpan(sub_input_data), sub_tensor_type));
    absl::flat_hash_map<std::string, TensorBuffer> sub_maps;
    sub_maps.insert_or_assign("images", std::move(sub_buffer));
    ASSERT_OK_AND_ASSIGN(ExecutorVisionData reused_output,
                         reused_executor->Encode(sub_maps));
    ASSERT_OK_AND_ASSIGN(ExecutorStats stats, reused_executor->StopProfiling());
    EXPECT_EQ(stats.GetMetricAsInt64("Vision num patches"), 2);

    ASSERT_OK_AND_ASSIGN(TensorBuffer * reused_embeddings,
                         reused_output.GetMutableEmbeddingsPtr());
    LITERT_ASSERT_OK_AND_ASSIGN(RankedTensorType reused_type,
                                reused_embeddings->TensorType());
    EXPECT_THAT(reused_type.Layout().Dimensions(),
                ElementsAre(1, 2, expected_output_dim));
    ASSERT_OK_AND_ASSIGN(std::vector<float> reused_vector,
                         ReadTensorBuffer<float>(*reused_embeddings));
    EXPECT_EQ(reused_vector.size(),
              static_cast<size_t>(2 * expected_output_dim));

    ASSERT_OK_AND_ASSIGN(TensorBuffer fresh_sub_buffer,
                         CreateTensorBuffer<float>(
                             absl::MakeSpan(sub_input_data), sub_tensor_type));
    absl::flat_hash_map<std::string, TensorBuffer> fresh_sub_maps;
    fresh_sub_maps.insert_or_assign("images", std::move(fresh_sub_buffer));
    ASSERT_OK_AND_ASSIGN(ExecutorVisionData fresh_output,
                         fresh_executor->Encode(fresh_sub_maps));
    ASSERT_OK_AND_ASSIGN(TensorBuffer * fresh_embeddings,
                         fresh_output.GetMutableEmbeddingsPtr());
    ASSERT_OK_AND_ASSIGN(std::vector<float> fresh_vector,
                         ReadTensorBuffer<float>(*fresh_embeddings));
    EXPECT_EQ(reused_vector, fresh_vector);
  }
}

TEST(VisionLiteRtCompiledModelExecutorTest,
     ProfilingCollectsVisionStatsWithAndWithoutAdapter) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, Environment::Create(std::vector<Environment::Option>()));
  RankedTensorType input_tensor_type(ElementType::Float32,
                                     Layout(litert::Dimensions{1, 10, 8}));
  std::vector<float> input_data(10 * 8, 1.0f);

  for (const bool with_adapter : {true, false}) {
    SCOPED_TRACE(with_adapter);
    const absl::string_view relative_path =
        with_adapter ? kTestVisionModelWithAdapterPath
                     : kTestVisionModelWithoutAdapterPath;
    const std::string model_path = (std::filesystem::path(::testing::SrcDir()) /
                                    std::string(relative_path))
                                       .string();
    ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                         ModelAssets::Create(model_path));
    ASSERT_OK_AND_ASSIGN(VisionExecutorSettings settings,
                         VisionExecutorSettings::CreateDefault(
                             model_assets,
                             /*encoder_backend=*/Backend::CPU,
                             /*adapter_backend=*/Backend::CPU));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VisionLiteRtCompiledModelExecutor> vision_executor,
        VisionLiteRtCompiledModelExecutor::Create(settings, env));

    EXPECT_THAT(vision_executor->StopProfiling(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
    ASSERT_OK(vision_executor->StartProfiling());

    ASSERT_OK_AND_ASSIGN(TensorBuffer single_buffer,
                         CreateTensorBuffer<float>(absl::MakeSpan(input_data),
                                                   input_tensor_type));
    ASSERT_OK(vision_executor->Encode(single_buffer));

    ASSERT_OK_AND_ASSIGN(TensorBuffer map_buffer,
                         CreateTensorBuffer<float>(absl::MakeSpan(input_data),
                                                   input_tensor_type));
    absl::flat_hash_map<std::string, TensorBuffer> input_maps;
    input_maps.insert_or_assign("images", std::move(map_buffer));
    ASSERT_OK(vision_executor->Encode(input_maps));

    ASSERT_OK_AND_ASSIGN(ExecutorStats stats, vision_executor->StopProfiling());
    EXPECT_EQ(stats.module_name, "Vision");
    EXPECT_EQ(stats.GetMetricAsInt64("Vision num images"), 2);
    EXPECT_EQ(stats.GetMetricAsInt64("Vision num patches"), 5);
    EXPECT_TRUE(stats.GetLatency("Vision encoder inference").has_value());
    EXPECT_EQ(stats.GetLatency("Vision adapter inference").has_value(),
              with_adapter);
  }
}

}  // namespace
}  // namespace litert::lm
