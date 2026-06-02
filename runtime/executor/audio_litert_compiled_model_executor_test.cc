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

#include "runtime/executor/audio_litert_compiled_model_executor.h"

#include <array>
#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "runtime/util/test_utils.h"     //NOLINT

namespace litert::lm {
namespace {

constexpr absl::string_view kTestAudioModelPath =
    "litert_lm/runtime/testdata/dummy_audio_only.litertlm";
constexpr absl::string_view kTestAudioNoMaskModelPath =
    "litert_lm/runtime/testdata/dummy_audio_no_mask.litertlm";
constexpr int kNoMaskEmbeddingDimensions = 5;
constexpr int kSpectrogramFrequencySlots = 8;
constexpr int kSpectrogramSequenceLength = 10;
constexpr int kEmbeddingSequenceLength = 5;
constexpr int kEmbeddingDimensions = 6;

using ::testing::ElementsAre;

template <typename T>
absl::StatusOr<std::vector<T>> GetDataAsVector(
    litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  std::vector<T> data(elements);
  LITERT_RETURN_IF_ERROR(tensor_buffer.Read<T>(absl::MakeSpan(data)));
  return data;
}

template <typename T>
absl::StatusOr<TensorBuffer> CreateTensorBuffer(
    absl::Span<T> data, RankedTensorType ranked_tensor_type) {
  LITERT_ASSIGN_OR_RETURN(auto output,
                          TensorBuffer::CreateManagedHostMemory(
                              ranked_tensor_type, data.size() * sizeof(T)));
  LITERT_RETURN_IF_ERROR(output.template Write<T>(absl::MakeSpan(data)));
  return output;
}

absl::StatusOr<std::unique_ptr<AudioLiteRtCompiledModelExecutor>>
CreateAudioExecutor(Environment& env, const std::string& model_path,
                    int max_sequence_length, Backend backend) {
  ASSIGN_OR_RETURN(auto model_file, litert::lm::ScopedFile::Open(model_path));
  auto model_file_ptr =
      std::make_shared<litert::lm::ScopedFile>(std::move(model_file));
  ASSIGN_OR_RETURN(auto model_assets,
                   litert::lm::ModelAssets::Create(model_file_ptr));
  // Create the audio executor settings.
  ASSIGN_OR_RETURN(auto audio_executor_settings,
                   litert::lm::AudioExecutorSettings::CreateDefault(
                       model_assets, max_sequence_length, backend));
  // Create the audio executor.
  return litert::lm::AudioLiteRtCompiledModelExecutor::Create(
      audio_executor_settings, env);
}
// TODO: b/441514829 - Enable the tests on Windows once the bug is fixed.
#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
    !defined(__NT__) && !defined(_WIN64)
class AudioLiteRtCompiledModelExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto env = Environment::Create(std::vector<Environment::Option>());
    env_ = std::make_unique<Environment>(std::move(*env));
  }
  std::unique_ptr<Environment> env_;
};

TEST_F(AudioLiteRtCompiledModelExecutorTest, CreateExecutorTest) {
  EXPECT_OK(CreateAudioExecutor(*env_,
                                (std::filesystem::path(::testing::SrcDir()) /
                                 std::string(kTestAudioModelPath))
                                    .string(),
                                /*max_sequence_length=*/0, Backend::CPU));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       EncodeTest_WithoutMaskFitSequenceLength) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  constexpr std::array<float,
                       kSpectrogramSequenceLength * kSpectrogramFrequencySlots>
      mel_spectrogram_data = {
          0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.,
          0., 1., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
          0., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0.,
          1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
          0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1.};

  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kSpectrogramSequenceLength,
                                              kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto executor_audio_data,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_ptr,
                       executor_audio_data.GetMutableEmbeddingsPtr());
  auto audio_embeddings_type = audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(audio_embeddings_type.HasValue());
  auto dims = audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims,
              ElementsAre(1, kEmbeddingSequenceLength, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_data,
                       GetDataAsVector<float>(*audio_embeddings_ptr));
  EXPECT_EQ(audio_embeddings_data.size(),
            kEmbeddingSequenceLength * kEmbeddingDimensions);
  EXPECT_THAT(
      audio_embeddings_data,
      ElementsAre(0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 3., 3., 0., 1., 2.,
                  4., 4., 4., 1., 2., 3., 5., 5., 5., 0., 1., 2., 4., 4., 4.));
  EXPECT_EQ(executor_audio_data.GetValidTokens(), kEmbeddingSequenceLength);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       EncodeTest_WithMaskFitSequenceLength) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  constexpr std::array<float,
                       kSpectrogramSequenceLength * kSpectrogramFrequencySlots>
      mel_spectrogram_data = {
          1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 1.,
          1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1.,
          1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0.,
          1., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 1., 0.,
          1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
      };
  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kSpectrogramSequenceLength,
                                              kSpectrogramFrequencySlots})))));

  constexpr std::array<bool, kSpectrogramSequenceLength>
      mel_spectrogram_mask_data = {true, true,  true,  true,  true,
                                   true, false, false, false, false};

  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_mask_tensor_buffer,
      CreateTensorBuffer<const bool>(
          mel_spectrogram_mask_data,
          RankedTensorType(
              GetElementType<bool>(),
              Layout(Dimensions({1, kSpectrogramSequenceLength})))));

  ASSERT_OK_AND_ASSIGN(
      auto executor_audio_data,
      audio_executor->Encode(mel_spectrogram_tensor_buffer,
                             mel_spectrogram_mask_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_ptr,
                       executor_audio_data.GetMutableEmbeddingsPtr());
  auto audio_embeddings_type = audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(audio_embeddings_type.HasValue());
  auto dims = audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, 3, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_data,
                       GetDataAsVector<float>(*audio_embeddings_ptr));
  EXPECT_EQ(audio_embeddings_data.size(), 3 * kEmbeddingDimensions);
  EXPECT_THAT(audio_embeddings_data,
              ElementsAre(1., 2., 4., 6., 6., 6., 1., 3., 6., 9., 9., 9., 1.,
                          3., 5., 8., 8., 8.));
  EXPECT_EQ(executor_audio_data.GetValidTokens(), 3);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       EncodeTest_WithoutMaskLongerThanSequenceLength) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  constexpr std::array<float, 13 * kSpectrogramFrequencySlots>
      mel_spectrogram_data = {
          1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1.,
          1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0.,
          1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1.,
          1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.,
          0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
          0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1.,
          0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 1.};
  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 13, kSpectrogramFrequencySlots})))));

  ASSERT_OK_AND_ASSIGN(auto executor_audio_data,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_ptr,
                       executor_audio_data.GetMutableEmbeddingsPtr());
  auto audio_embeddings_type = audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(audio_embeddings_type.HasValue());
  auto dims = audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, 7, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_data,
                       GetDataAsVector<float>(*audio_embeddings_ptr));
  EXPECT_EQ(audio_embeddings_data.size(), 7 * kEmbeddingDimensions);
  EXPECT_THAT(
      audio_embeddings_data,
      ElementsAre(1., 2., 4., 6., 6., 6., 1., 3., 6., 9., 9., 9., 1., 3., 5.,
                  8., 8., 8., 1., 2., 4., 7., 7., 7., 1., 3., 6., 9., 9., 9.,
                  0., 1., 2., 3., 3., 3., 0., 1., 2., 3., 3., 3.));
  EXPECT_EQ(executor_audio_data.GetValidTokens(), 7);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       EncodeTest_WithMaskLongerThanSequenceLength) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  constexpr std::array<float, 13 * kSpectrogramFrequencySlots>
      mel_spectrogram_data = {
          1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1.,
          1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0.,
          1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1.,
          1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.,
          0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
          0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1.,
          0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 1.};
  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 13, kSpectrogramFrequencySlots})))));

  constexpr std::array<bool, 13> mel_spectrogram_mask_data = {
      true, true, true, true, true,  true, true,
      true, true, true, true, false, false};
  ASSERT_OK_AND_ASSIGN(auto mel_spectrogram_mask_tensor_buffer,
                       CreateTensorBuffer<const bool>(
                           mel_spectrogram_mask_data,
                           RankedTensorType(GetElementType<bool>(),
                                            Layout(Dimensions({1, 13})))));

  ASSERT_OK_AND_ASSIGN(
      auto executor_audio_data,
      audio_executor->Encode(mel_spectrogram_tensor_buffer,
                             mel_spectrogram_mask_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_ptr,
                       executor_audio_data.GetMutableEmbeddingsPtr());
  auto audio_embeddings_type = audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(audio_embeddings_type.HasValue());
  auto dims = audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, 6, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_data,
                       GetDataAsVector<float>(*audio_embeddings_ptr));
  EXPECT_EQ(audio_embeddings_data.size(), 6 * kEmbeddingDimensions);
  EXPECT_THAT(audio_embeddings_data,
              ElementsAre(1., 2., 4., 6., 6., 6., 1., 3., 6., 9., 9., 9., 1.,
                          3., 5., 8., 8., 8., 1., 2., 4., 7., 7., 7., 1., 3.,
                          6., 9., 9., 9., 0., 1., 2., 3., 3., 3.));
  EXPECT_EQ(executor_audio_data.GetValidTokens(), 6);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       EncodeTest_NoMaskModel_WithoutMask) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioNoMaskModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  constexpr std::array<float,
                       kSpectrogramSequenceLength * kSpectrogramFrequencySlots>
      mel_spectrogram_data = {
          0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.,
          0., 1., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
          0., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0.,
          1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
          0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1.};

  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kSpectrogramSequenceLength,
                                              kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto executor_audio_data,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_ptr,
                       executor_audio_data.GetMutableEmbeddingsPtr());
  auto audio_embeddings_type = audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(audio_embeddings_type.HasValue());
  auto dims = audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, kEmbeddingSequenceLength,
                                kNoMaskEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_data,
                       GetDataAsVector<float>(*audio_embeddings_ptr));
  EXPECT_EQ(audio_embeddings_data.size(),
            kEmbeddingSequenceLength * kNoMaskEmbeddingDimensions);
  EXPECT_EQ(executor_audio_data.GetValidTokens(), kEmbeddingSequenceLength);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest, EncodeTest_NoMaskModel_WithMask) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioNoMaskModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  constexpr std::array<float,
                       kSpectrogramSequenceLength * kSpectrogramFrequencySlots>
      mel_spectrogram_data = {
          1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 1.,
          1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1.,
          1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0.,
          1., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 1., 0.,
          1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
      };
  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kSpectrogramSequenceLength,
                                              kSpectrogramFrequencySlots})))));

  constexpr std::array<bool, kSpectrogramSequenceLength>
      mel_spectrogram_mask_data = {true, true,  true,  true,  true,
                                   true, false, false, false, false};

  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_mask_tensor_buffer,
      CreateTensorBuffer<const bool>(
          mel_spectrogram_mask_data,
          RankedTensorType(
              GetElementType<bool>(),
              Layout(Dimensions({1, kSpectrogramSequenceLength})))));

  ASSERT_OK_AND_ASSIGN(
      auto executor_audio_data,
      audio_executor->Encode(mel_spectrogram_tensor_buffer,
                             mel_spectrogram_mask_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_ptr,
                       executor_audio_data.GetMutableEmbeddingsPtr());
  auto audio_embeddings_type = audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(audio_embeddings_type.HasValue());
  auto dims = audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, 3, kNoMaskEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_data,
                       GetDataAsVector<float>(*audio_embeddings_ptr));
  EXPECT_EQ(audio_embeddings_data.size(), 3 * kNoMaskEmbeddingDimensions);
  EXPECT_EQ(executor_audio_data.GetValidTokens(), 3);
}
#endif  // !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
        // !defined(__NT__) && !defined(_WIN64)

}  // namespace

}  // namespace litert::lm
