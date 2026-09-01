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
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::ElementType;
using ::litert::Layout;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::lm::Backend;
using ::litert::lm::LitertLmLoader;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelResourcesLitertLm;
using ::litert::lm::VisionExecutorSettings;
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
                       "TF_LITE_VISION_ENCODER not found in the model."));
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

}  // namespace
}  // namespace litert::lm
