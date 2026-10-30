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
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "runtime/util/test_utils.h"     //NOLINT

namespace litert::lm {
namespace {

constexpr absl::string_view kTestAudioModelPath =
    "litert_lm/runtime/testdata/dummy_audio_only.litertlm";
constexpr absl::string_view kTestAudioStreamingModelPath =
    "litert_lm/runtime/testdata/dummy_audio_streaming.litertlm";
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
                    int max_sequence_length, Backend backend,
                    bool audio_buffering_enabled = false) {
  ABSL_ASSIGN_OR_RETURN(auto model_file,
                        litert::lm::ScopedFile::Open(model_path));
  auto model_file_ptr =
      std::make_shared<litert::lm::ScopedFile>(std::move(model_file));
  ABSL_ASSIGN_OR_RETURN(auto model_assets,
                        litert::lm::ModelAssets::Create(model_file_ptr));
  // Create the audio executor settings.
  ABSL_ASSIGN_OR_RETURN(
      auto audio_executor_settings,
      litert::lm::AudioExecutorSettings::CreateDefault(
          model_assets, max_sequence_length, backend, backend));
  audio_executor_settings.SetAudioBufferingEnabled(audio_buffering_enabled);
  // Create the audio executor.
  return litert::lm::AudioLiteRtCompiledModelExecutor::Create(
      audio_executor_settings, env);
}

// The dummy streaming model (chunk_size = 10, overlap = 3, stride = 7,
// shrinking factor = 2) emits output token k with mask segment_mask[2 * k].
// Valid tokens are projected to `kStreamingValidFrame`, and masked tokens to
// zeros.
constexpr std::array<float, kEmbeddingDimensions> kStreamingValidFrame = {
    1.f, 3.f, 6.f, 10.f, 15.f, 15.f};

// Returns the expected projected embeddings of the dummy streaming model: the
// first `num_masked_tokens` tokens are zeros (they start with a masked
// padding frame) and the remaining tokens are `kStreamingValidFrame`.
std::vector<float> ExpectedStreamingEmbeddings(int num_masked_tokens,
                                               int num_tokens) {
  std::vector<float> expected;
  expected.reserve(num_tokens * kEmbeddingDimensions);
  expected.insert(expected.end(), num_masked_tokens * kEmbeddingDimensions,
                  0.0f);
  for (int i = num_masked_tokens; i < num_tokens; ++i) {
    expected.insert(expected.end(), kStreamingValidFrame.begin(),
                    kStreamingValidFrame.end());
  }
  return expected;
}

// Encodes `num_frames` all-ones spectrogram frames and returns the valid
// projected audio embeddings.
absl::StatusOr<std::vector<float>> EncodeOnesAndGetEmbeddings(
    AudioLiteRtCompiledModelExecutor& audio_executor, int num_frames) {
  std::vector<float> mel_spectrogram_data(
      num_frames * kSpectrogramFrequencySlots, 1.0f);
  ABSL_ASSIGN_OR_RETURN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions(
                               {1, num_frames, kSpectrogramFrequencySlots})))));
  ABSL_ASSIGN_OR_RETURN(auto audio_data,
                        audio_executor.Encode(mel_spectrogram_tensor_buffer));
  ABSL_ASSIGN_OR_RETURN(auto projected_audio_embeddings_ptr,
                        audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  ABSL_ASSIGN_OR_RETURN(
      auto embeddings, GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  embeddings.resize(audio_data.GetValidTokens() * kEmbeddingDimensions);
  return embeddings;
}

}  // namespace

class AudioLiteRtCompiledModelExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto env = Environment::Create(std::vector<Environment::Option>());
    env_ = std::make_unique<Environment>(std::move(*env));
  }

  static bool ShouldShareProducerBuffer(TensorBufferType producer_type,
                                        size_t producer_packed_size,
                                        TensorBufferType consumer_type,
                                        size_t consumer_packed_size) {
    return AudioLiteRtCompiledModelExecutor::ShouldShareProducerBuffer(
        producer_type, producer_packed_size, consumer_type,
        consumer_packed_size);
  }

  static absl::StatusOr<bool> ShouldShareProducerBuffer(
      const TensorBuffer& producer, const TensorBuffer& consumer) {
    return AudioLiteRtCompiledModelExecutor::ShouldShareProducerBuffer(
        producer, consumer);
  }

  static TensorBuffer* GetEncoderOutputMaskBuffer(
      AudioLiteRtCompiledModelExecutor& executor) {
    return executor.audio_encoder_->GetMutableOutputMaskBuffer();
  }

  static TensorBuffer* GetAdapterMaskBuffer(
      AudioLiteRtCompiledModelExecutor& executor) {
    return executor.audio_adapter_->GetMutableMaskBuffer();
  }

  static TensorBuffer& GetEncoderOutputFeaturesBuffer(
      AudioLiteRtCompiledModelExecutor& executor) {
    return executor.audio_encoder_->GetMutableOutputFeaturesBuffer();
  }

  static TensorBuffer& GetAdapterFeaturesBuffer(
      AudioLiteRtCompiledModelExecutor& executor) {
    return executor.audio_adapter_->GetMutableFeaturesBuffer();
  }

  std::unique_ptr<Environment> env_;
};

namespace {

TEST_F(AudioLiteRtCompiledModelExecutorTest, CreateExecutorTest) {
  EXPECT_OK(CreateAudioExecutor(*env_,
                                (std::filesystem::path(::testing::SrcDir()) /
                                 std::string(kTestAudioModelPath))
                                    .string(),
                                /*max_sequence_length=*/0, Backend::CPU));
}

#if !defined(LITERT_DISABLE_NPU)
TEST_F(AudioLiteRtCompiledModelExecutorTest, CreateNpuExecutorTest) {
  EXPECT_OK(CreateAudioExecutor(*env_,
                                (std::filesystem::path(::testing::SrcDir()) /
                                 std::string(kTestAudioModelPath))
                                    .string(),
                                /*max_sequence_length=*/0, Backend::NPU));
  EXPECT_OK(CreateAudioExecutor(*env_,
                                (std::filesystem::path(::testing::SrcDir()) /
                                 std::string(kTestAudioStreamingModelPath))
                                    .string(),
                                /*max_sequence_length=*/0, Backend::NPU));
}
#endif  // !defined(LITERT_DISABLE_NPU)

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
  ASSERT_OK_AND_ASSIGN(
      auto projected_audio_embeddings_ptr,
      executor_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto projected_audio_embeddings_type =
      projected_audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(projected_audio_embeddings_type.HasValue());
  auto dims = projected_audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims,
              ElementsAre(1, kEmbeddingSequenceLength, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto projected_audio_embeddings_data,
                       GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  EXPECT_EQ(projected_audio_embeddings_data.size(),
            kEmbeddingSequenceLength * kEmbeddingDimensions);
  EXPECT_THAT(
      projected_audio_embeddings_data,
      ElementsAre(0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 3., 3., 0., 1., 2.,
                  4., 4., 4., 1., 2., 3., 5., 5., 5., 0., 1., 2., 4., 4., 4.));
  EXPECT_EQ(executor_audio_data.GetValidTokens(), kEmbeddingSequenceLength);

  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_ptr,
                       executor_audio_data.GetAudioEmbeddingsPtr());
  auto audio_embeddings_type = audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(audio_embeddings_type.HasValue());
  auto audio_dims = audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(audio_dims, ElementsAre(1, kEmbeddingSequenceLength, 4));

  ASSERT_OK_AND_ASSIGN(
      auto audio_embeddings_data,
      GetDataAsVector<float>(
          *const_cast<litert::TensorBuffer*>(audio_embeddings_ptr)));
  EXPECT_THAT(audio_embeddings_data,
              ElementsAre(0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 2., 1.,
                          1., 1., 2., 0., 1., 1., 2.));
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
  ASSERT_OK_AND_ASSIGN(
      auto projected_audio_embeddings_ptr,
      executor_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto projected_audio_embeddings_type =
      projected_audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(projected_audio_embeddings_type.HasValue());
  auto dims = projected_audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, 3, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto projected_audio_embeddings_data,
                       GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  EXPECT_EQ(projected_audio_embeddings_data.size(), 3 * kEmbeddingDimensions);
  EXPECT_THAT(projected_audio_embeddings_data,
              ElementsAre(1., 2., 4., 6., 6., 6., 1., 3., 6., 9., 9., 9., 1.,
                          3., 5., 8., 8., 8.));
  EXPECT_EQ(executor_audio_data.GetValidTokens(), 3);

  ASSERT_OK_AND_ASSIGN(auto audio_embeddings_ptr,
                       executor_audio_data.GetAudioEmbeddingsPtr());
  auto audio_embeddings_type = audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(audio_embeddings_type.HasValue());
  auto audio_dims = audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(audio_dims, ElementsAre(1, 3, 4));

  ASSERT_OK_AND_ASSIGN(
      auto audio_embeddings_data,
      GetDataAsVector<float>(
          *const_cast<litert::TensorBuffer*>(audio_embeddings_ptr)));
  EXPECT_THAT(audio_embeddings_data,
              ElementsAre(1., 1., 2., 2., 1., 2., 3., 3., 1., 2., 2., 3.));
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
  ASSERT_OK_AND_ASSIGN(
      auto projected_audio_embeddings_ptr,
      executor_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto projected_audio_embeddings_type =
      projected_audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(projected_audio_embeddings_type.HasValue());
  auto dims = projected_audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, 7, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto projected_audio_embeddings_data,
                       GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  EXPECT_EQ(projected_audio_embeddings_data.size(), 7 * kEmbeddingDimensions);
  EXPECT_THAT(
      projected_audio_embeddings_data,
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
  ASSERT_OK_AND_ASSIGN(
      auto projected_audio_embeddings_ptr,
      executor_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto projected_audio_embeddings_type =
      projected_audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(projected_audio_embeddings_type.HasValue());
  auto dims = projected_audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, 6, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto projected_audio_embeddings_data,
                       GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  EXPECT_EQ(projected_audio_embeddings_data.size(), 6 * kEmbeddingDimensions);
  EXPECT_THAT(projected_audio_embeddings_data,
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
  ASSERT_OK_AND_ASSIGN(
      auto projected_audio_embeddings_ptr,
      executor_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto projected_audio_embeddings_type =
      projected_audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(projected_audio_embeddings_type.HasValue());
  auto dims = projected_audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, kEmbeddingSequenceLength,
                                kNoMaskEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto projected_audio_embeddings_data,
                       GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  EXPECT_EQ(projected_audio_embeddings_data.size(),
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
  ASSERT_OK_AND_ASSIGN(
      auto projected_audio_embeddings_ptr,
      executor_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto projected_audio_embeddings_type =
      projected_audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(projected_audio_embeddings_type.HasValue());
  auto dims = projected_audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims, ElementsAre(1, 3, kNoMaskEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto projected_audio_embeddings_data,
                       GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  EXPECT_EQ(projected_audio_embeddings_data.size(),
            3 * kNoMaskEmbeddingDimensions);
  EXPECT_EQ(executor_audio_data.GetValidTokens(), 3);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest, EncodeTest_Streaming_LargeInput) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  // Input size 24, left padded with 3 masked zero frames for the first chunk
  // of the stream -> 27 frames.
  // chunk_size = 10, overlap = 3, stride = 7. Buffering is disabled, so the
  // trailing frames are flushed.
  // 4 chunks: [0, 10), [7, 17), [14, 24), [21, 27).
  // Total output tokens: 3 * (10 / 2) + 6 / 2 = 18.
  constexpr int kInputSequenceLength = 24;
  constexpr int kExpectedOutputTokens = 18;
  // Tokens 0 and 1 of the first chunk start with padding frames 0 and 2, so
  // they are masked.
  constexpr int kExpectedMaskedTokens = 2;

  std::vector<float> mel_spectrogram_data(
      kInputSequenceLength * kSpectrogramFrequencySlots, 1.0f);

  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kInputSequenceLength,
                                              kSpectrogramFrequencySlots})))));

  ASSERT_OK_AND_ASSIGN(auto executor_audio_data,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer));

  ASSERT_OK_AND_ASSIGN(
      auto projected_audio_embeddings_ptr,
      executor_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto projected_audio_embeddings_type =
      projected_audio_embeddings_ptr->TensorType();
  ASSERT_TRUE(projected_audio_embeddings_type.HasValue());
  auto dims = projected_audio_embeddings_type->Layout().Dimensions();
  EXPECT_THAT(dims,
              ElementsAre(1, kExpectedOutputTokens, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto projected_audio_embeddings_data,
                       GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  EXPECT_EQ(projected_audio_embeddings_data.size(),
            kExpectedOutputTokens * kEmbeddingDimensions);

  EXPECT_EQ(projected_audio_embeddings_data,
            ExpectedStreamingEmbeddings(kExpectedMaskedTokens,
                                        kExpectedOutputTokens));
  EXPECT_EQ(executor_audio_data.GetValidTokens(), kExpectedOutputTokens);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest, FlushTest_NoBuffering) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(auto executor_audio_data, audio_executor->Flush());
  EXPECT_EQ(executor_audio_data.GetValidTokens(), 0);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       FlushTest_WithBuffering_PartialWindow) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU,
                          /*audio_buffering_enabled=*/true));

  std::vector<float> mel_spectrogram_data(5 * kSpectrogramFrequencySlots, 1.0f);

  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 5, kSpectrogramFrequencySlots})))));

  ASSERT_OK_AND_ASSIGN(auto executor_audio_data_encode,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer));
  // 5 frames < window size 10 -> buffered unpadded, 0 tokens.
  EXPECT_EQ(executor_audio_data_encode.GetValidTokens(), 0);

  // Flush pads the 5 buffered frames with 3 masked zero frames -> 8 frames,
  // a single chunk [0, 8) -> 8 / 2 = 4 tokens.
  ASSERT_OK_AND_ASSIGN(auto executor_audio_data_flush, audio_executor->Flush());
  EXPECT_EQ(executor_audio_data_flush.GetValidTokens(), 4);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       FlushTest_WithBuffering_MultiChunk) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU,
                          /*audio_buffering_enabled=*/true));

  std::vector<float> mel_spectrogram_data1(5 * kSpectrogramFrequencySlots,
                                           1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer1,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data1,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 5, kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto audio_data1,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer1));
  // 5 frames < window size 10 -> buffered unpadded, 0 tokens.
  EXPECT_EQ(audio_data1.GetValidTokens(), 0);

  std::vector<float> mel_spectrogram_data2(8 * kSpectrogramFrequencySlots,
                                           1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto mel_spectrogram_tensor_buffer2,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data2,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 8, kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto audio_data2,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer2));
  // 5 buffered + 8 new frames, left padded with 3 masked zero frames -> 16
  // frames. Only the full chunk [0, 10) is encoded -> 5 tokens; frames
  // [7, 16) are buffered.
  EXPECT_EQ(audio_data2.GetValidTokens(), 5);

  // Flush encodes the 9 buffered frames without padding again: a single chunk
  // [0, 9) -> ceil(9 / 2) = 5 tokens.
  ASSERT_OK_AND_ASSIGN(auto audio_data_flush, audio_executor->Flush());
  EXPECT_EQ(audio_data_flush.GetValidTokens(), 5);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       ContextManagement_StaticEncoder_NotSupported) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  EXPECT_EQ(audio_executor->CreateNewContext().status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(audio_executor->CloneContext().status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       ContextManagement_StreamingEncoder_CreateCloneRestore) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(auto new_context, audio_executor->CreateNewContext());
  EXPECT_NE(new_context, nullptr);

  std::vector<float> chunk1_data(10 * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto chunk1_buffer,
      CreateTensorBuffer<const float>(
          chunk1_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 10, kSpectrogramFrequencySlots})))));
  ASSERT_OK(audio_executor->Encode(chunk1_buffer).status());

  ASSERT_OK_AND_ASSIGN(auto cloned_context, audio_executor->CloneContext());
  EXPECT_NE(cloned_context, nullptr);

  std::vector<float> chunk2_data(10 * kSpectrogramFrequencySlots, 2.0f);
  ASSERT_OK_AND_ASSIGN(
      auto chunk2_buffer,
      CreateTensorBuffer<const float>(
          chunk2_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 10, kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto audio_data_result1,
                       audio_executor->Encode(chunk2_buffer));
  ASSERT_OK_AND_ASSIGN(
      auto projected_embeddings1_ptr,
      audio_data_result1.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto projected_embeddings1,
                       GetDataAsVector<float>(*projected_embeddings1_ptr));

  ASSERT_OK(audio_executor->RestoreContext(std::move(cloned_context)));

  ASSERT_OK_AND_ASSIGN(auto audio_data_result2,
                       audio_executor->Encode(chunk2_buffer));
  ASSERT_OK_AND_ASSIGN(
      auto projected_embeddings2_ptr,
      audio_data_result2.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto projected_embeddings2,
                       GetDataAsVector<float>(*projected_embeddings2_ptr));

  EXPECT_EQ(audio_data_result1.GetValidTokens(),
            audio_data_result2.GetValidTokens());
  EXPECT_EQ(projected_embeddings1, projected_embeddings2);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       ContextManagement_StreamingEncoder_BufferedSpectrogram) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU,
                          /*audio_buffering_enabled=*/true));

  // Encode partial chunk (5 frames < window size 10) -> buffered, 0 tokens
  // returned.
  std::vector<float> chunk_data(5 * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto chunk_buffer,
      CreateTensorBuffer<const float>(
          chunk_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 5, kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto encode_data, audio_executor->Encode(chunk_buffer));
  EXPECT_EQ(encode_data.GetValidTokens(), 0);

  // Clone context after buffering 5 frames.
  ASSERT_OK_AND_ASSIGN(auto cloned_context, audio_executor->CloneContext());
  ASSERT_NE(cloned_context, nullptr);

  auto* streaming_context =
      static_cast<AudioStreamingContext*>(cloned_context.get());
  EXPECT_EQ(streaming_context->buffered_spectrogram().size(),
            5 * kSpectrogramFrequencySlots);

  // Test AudioStreamingContext::Clone() copy constructor logic.
  ASSERT_OK_AND_ASSIGN(auto cloned_cloned_context, cloned_context->Clone());
  auto* streaming_context2 =
      static_cast<AudioStreamingContext*>(cloned_cloned_context.get());
  EXPECT_EQ(streaming_context2->buffered_spectrogram(),
            streaming_context->buffered_spectrogram());

  // Flush the executor once -> pads the 5 buffered frames with 3 masked zero
  // frames -> a single chunk [0, 8) -> 4 tokens.
  ASSERT_OK_AND_ASSIGN(auto flush_data1, audio_executor->Flush());
  EXPECT_EQ(flush_data1.GetValidTokens(), 4);

  // Restore context -> restores the 5 buffered frames back into executor, and
  // marks the next chunk as the first chunk of the stream again.
  ASSERT_OK(audio_executor->RestoreContext(std::move(cloned_context)));

  // Flush again after restore -> processes restored 5 buffered frames with
  // the same padding.
  ASSERT_OK_AND_ASSIGN(auto flush_data2, audio_executor->Flush());
  EXPECT_EQ(flush_data1.GetValidTokens(), flush_data2.GetValidTokens());
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       ContextManagement_StreamingEncoder_CreateNewContextResetsStateIsolated) {
  const std::filesystem::path src_model_path =
      std::filesystem::path(::testing::SrcDir()) /
      std::string(kTestAudioStreamingModelPath);
  std::ifstream src_file(src_model_path, std::ios::binary);
  ASSERT_TRUE(src_file.is_open());
  std::string model_bytes((std::istreambuf_iterator<char>(src_file)),
                          std::istreambuf_iterator<char>());
  // Rename the FlatBuffer signature input "prev_key_0" (length 10) to
  // "prev_mask" (length 9 + 1 trailing null byte) so we can exercise both
  // kPrevMaskName (initialized to 1s) and other state buffers (initialized to
  // 0s) in CreateNewContext().
  const std::string from_name("\x0a\x00\x00\x00prev_key_0", 14);
  const std::string to_name("\x09\x00\x00\x00prev_mask\x00", 14);
  const size_t pos = model_bytes.find(from_name);
  ASSERT_NE(pos, std::string::npos);
  model_bytes.replace(pos, from_name.size(), to_name);

  const std::filesystem::path temp_model_path =
      std::filesystem::path(::testing::TempDir()) /
      "dummy_audio_streaming_prev_mask.litertlm";
  {
    std::ofstream dst_file(temp_model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(dst_file.is_open());
    dst_file.write(model_bytes.data(), model_bytes.size());
  }

  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_, temp_model_path.string(),
                          /*max_sequence_length=*/0, Backend::CPU,
                          /*audio_buffering_enabled=*/true));

  // Buffer 5 frames in the active session.
  std::vector<float> partial_chunk_data(5 * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto partial_chunk_buffer,
      CreateTensorBuffer<const float>(
          partial_chunk_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 5, kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto encode_partial,
                       audio_executor->Encode(partial_chunk_buffer));
  EXPECT_EQ(encode_partial.GetValidTokens(), 0);

  // Mutate prev_mask (to all 0s) and feature_state_0 (to all 1.0f) in the
  // active session's state before creating a new context for another session.
  ASSERT_OK_AND_ASSIGN(auto mutated_active_context,
                       audio_executor->CloneContext());
  auto* mutated_streaming_context =
      static_cast<AudioStreamingContext*>(mutated_active_context.get());
  ASSERT_TRUE(mutated_streaming_context->state_buffers().contains("prev_mask"));
  ASSERT_OK_AND_ASSIGN(
      auto initial_prev_mask_data,
      GetDataAsVector<uint8_t>(
          mutated_streaming_context->state_buffers().at("prev_mask")));
  EXPECT_THAT(initial_prev_mask_data, ::testing::Each(uint8_t{1}));
  std::vector<uint8_t> all_zeros(initial_prev_mask_data.size(), 0);
  ASSERT_TRUE(mutated_streaming_context->state_buffers()
                  .at("prev_mask")
                  .Write<uint8_t>(absl::MakeSpan(all_zeros))
                  .HasValue());
  ASSERT_TRUE(
      mutated_streaming_context->state_buffers().contains("feature_state_0"));
  ASSERT_OK_AND_ASSIGN(
      auto initial_feature_state_data,
      GetDataAsVector<float>(
          mutated_streaming_context->state_buffers().at("feature_state_0")));
  std::vector<float> all_ones_float(initial_feature_state_data.size(), 1.0f);
  ASSERT_TRUE(mutated_streaming_context->state_buffers()
                  .at("feature_state_0")
                  .Write<float>(absl::MakeSpan(all_ones_float))
                  .HasValue());
  ASSERT_OK(audio_executor->RestoreContext(std::move(mutated_active_context)));

  // Create a brand-new context for another session while the active session
  // has mutated state buffers and 5 buffered frames in the encoder.
  ASSERT_OK_AND_ASSIGN(auto new_context, audio_executor->CreateNewContext());
  ASSERT_NE(new_context, nullptr);
  auto* new_streaming_context =
      static_cast<AudioStreamingContext*>(new_context.get());
  EXPECT_TRUE(new_streaming_context->buffered_spectrogram().empty());
  ASSERT_TRUE(new_streaming_context->state_buffers().contains("prev_mask"));
  ASSERT_OK_AND_ASSIGN(
      auto new_prev_mask_data,
      GetDataAsVector<uint8_t>(
          new_streaming_context->state_buffers().at("prev_mask")));
  EXPECT_THAT(new_prev_mask_data, ::testing::Each(uint8_t{1}));

  ASSERT_TRUE(
      new_streaming_context->state_buffers().contains("feature_state_0"));
  ASSERT_OK_AND_ASSIGN(
      auto new_feature_state_data,
      GetDataAsVector<float>(
          new_streaming_context->state_buffers().at("feature_state_0")));
  EXPECT_THAT(new_feature_state_data, ::testing::Each(0.0f));

  // Verify the active session's buffered spectrogram and state buffers in the
  // encoder were NOT cleared or overwritten by CreateNewContext().
  ASSERT_OK_AND_ASSIGN(auto active_cloned_context,
                       audio_executor->CloneContext());
  auto* active_streaming_context =
      static_cast<AudioStreamingContext*>(active_cloned_context.get());
  EXPECT_EQ(active_streaming_context->buffered_spectrogram().size(),
            5 * kSpectrogramFrequencySlots);
  ASSERT_OK_AND_ASSIGN(
      auto active_prev_mask_data,
      GetDataAsVector<uint8_t>(
          active_streaming_context->state_buffers().at("prev_mask")));
  EXPECT_THAT(active_prev_mask_data, ::testing::Each(uint8_t{0}));
  ASSERT_OK_AND_ASSIGN(
      auto active_feature_state_data,
      GetDataAsVector<float>(
          active_streaming_context->state_buffers().at("feature_state_0")));
  EXPECT_THAT(active_feature_state_data, ::testing::Each(1.0f));

  // Restoring the new context should clear the encoder's buffered spectrogram.
  ASSERT_OK(audio_executor->RestoreContext(std::move(new_context)));
  ASSERT_OK_AND_ASSIGN(auto flush_after_new_restore, audio_executor->Flush());
  EXPECT_EQ(flush_after_new_restore.GetValidTokens(), 0);
}

// Encoding 10 frames as the first chunk of a stream (no buffering, so the
// trailing frames are flushed): padded to 13 frames -> chunks [0, 10) and
// [7, 13) -> 5 + 3 = 8 tokens, the first 2 masked by the padding.
constexpr int kFirstEncodeTokens = 8;
constexpr int kFirstEncodeMaskedTokens = 2;
// Encoding 10 frames later in the stream: no padding -> a single chunk
// [0, 10) -> 5 tokens, none masked.
constexpr int kLaterEncodeTokens = 5;

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StreamingEncoder_PadsOnlyFirstChunkOfStream) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(auto first_embeddings,
                       EncodeOnesAndGetEmbeddings(*audio_executor, 10));
  EXPECT_EQ(first_embeddings,
            ExpectedStreamingEmbeddings(kFirstEncodeMaskedTokens,
                                        kFirstEncodeTokens));

  // A second Encode call on the same stream is not padded again.
  ASSERT_OK_AND_ASSIGN(auto second_embeddings,
                       EncodeOnesAndGetEmbeddings(*audio_executor, 10));
  EXPECT_EQ(second_embeddings,
            ExpectedStreamingEmbeddings(
                /*num_masked_tokens=*/0, kLaterEncodeTokens));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StreamingEncoder_ResetAndNewContextPadFirstChunkAgain) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));
  const std::vector<float> first_chunk_embeddings =
      ExpectedStreamingEmbeddings(kFirstEncodeMaskedTokens, kFirstEncodeTokens);

  ASSERT_OK_AND_ASSIGN(auto embeddings,
                       EncodeOnesAndGetEmbeddings(*audio_executor, 10));
  EXPECT_EQ(embeddings, first_chunk_embeddings);

  // Reset() starts a new stream, so the next chunk is padded again.
  ASSERT_OK(audio_executor->Reset());
  ASSERT_OK_AND_ASSIGN(embeddings,
                       EncodeOnesAndGetEmbeddings(*audio_executor, 10));
  EXPECT_EQ(embeddings, first_chunk_embeddings);

  // A new context starts a new stream, so the next chunk is padded again.
  ASSERT_OK_AND_ASSIGN(auto new_context, audio_executor->CreateNewContext());
  EXPECT_TRUE(
      static_cast<AudioStreamingContext*>(new_context.get())->IsFirstChunk());
  ASSERT_OK(audio_executor->RestoreContext(std::move(new_context)));
  ASSERT_OK_AND_ASSIGN(embeddings,
                       EncodeOnesAndGetEmbeddings(*audio_executor, 10));
  EXPECT_EQ(embeddings, first_chunk_embeddings);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StreamingEncoder_CloneAndRestoreContextPreserveFirstChunkState) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  // Saved before any encode: the next chunk is the first chunk.
  ASSERT_OK_AND_ASSIGN(auto context_before_encode,
                       audio_executor->CloneContext());
  EXPECT_TRUE(static_cast<AudioStreamingContext*>(context_before_encode.get())
                  ->IsFirstChunk());

  ASSERT_OK(EncodeOnesAndGetEmbeddings(*audio_executor, 10).status());

  // Saved after the first encode: the next chunk is not the first chunk.
  ASSERT_OK_AND_ASSIGN(auto context_after_encode,
                       audio_executor->CloneContext());
  EXPECT_FALSE(static_cast<AudioStreamingContext*>(context_after_encode.get())
                   ->IsFirstChunk());

  // AudioStreamingContext::Clone() carries the first-chunk state.
  ASSERT_OK_AND_ASSIGN(auto cloned_before_encode,
                       context_before_encode->Clone());
  EXPECT_TRUE(static_cast<AudioStreamingContext*>(cloned_before_encode.get())
                  ->IsFirstChunk());
  ASSERT_OK_AND_ASSIGN(auto cloned_after_encode, context_after_encode->Clone());
  EXPECT_FALSE(static_cast<AudioStreamingContext*>(cloned_after_encode.get())
                   ->IsFirstChunk());

  // Restoring the context saved after the first encode does not pad.
  ASSERT_OK(audio_executor->RestoreContext(std::move(context_after_encode)));
  ASSERT_OK_AND_ASSIGN(auto embeddings,
                       EncodeOnesAndGetEmbeddings(*audio_executor, 10));
  EXPECT_EQ(embeddings, ExpectedStreamingEmbeddings(/*num_masked_tokens=*/0,
                                                    kLaterEncodeTokens));

  // Restoring the context saved before any encode pads the next chunk again.
  ASSERT_OK(audio_executor->RestoreContext(std::move(context_before_encode)));
  ASSERT_OK_AND_ASSIGN(embeddings,
                       EncodeOnesAndGetEmbeddings(*audio_executor, 10));
  EXPECT_EQ(embeddings, ExpectedStreamingEmbeddings(kFirstEncodeMaskedTokens,
                                                    kFirstEncodeTokens));
}

TEST_F(
    AudioLiteRtCompiledModelExecutorTest,
    StreamingEncoder_CloneAndRestoreContextPreservesBufferTypes) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  ASSERT_OK_AND_ASSIGN(auto initial_context,
                       audio_executor->CreateNewContext());
  auto* initial_streaming_context =
      static_cast<AudioStreamingContext*>(initial_context.get());
  ASSERT_FALSE(initial_streaming_context->state_buffers().empty());

  std::vector<std::pair<std::string, std::pair<TensorBufferType, size_t>>>
      expected_buffer_specs;
  for (const auto& [name, buffer] :
       initial_streaming_context->state_buffers()) {
    auto buffer_type = buffer.BufferType();
    auto buffer_size = buffer.Size();
    ASSERT_TRUE(buffer_type.HasValue()) << "for state buffer " << name;
    ASSERT_TRUE(buffer_size.HasValue()) << "for state buffer " << name;
    expected_buffer_specs.push_back(
        {std::string(name), {*buffer_type, *buffer_size}});
  }

  auto make_chunk = [](float fill_value) -> absl::StatusOr<TensorBuffer> {
    std::vector<float> chunk_data(10 * kSpectrogramFrequencySlots, fill_value);
    return CreateTensorBuffer<const float>(
        chunk_data,
        RankedTensorType(
            GetElementType<float>(),
            Layout(Dimensions({1, 10, kSpectrogramFrequencySlots}))));
  };

  // Session A: encode chunk 1, snapshot context, then encode chunk 2.
  ASSERT_OK_AND_ASSIGN(auto session_a_chunk1, make_chunk(1.0f));
  ASSERT_OK(audio_executor->Encode(session_a_chunk1).status());
  ASSERT_OK_AND_ASSIGN(auto context_a, audio_executor->CloneContext());
  auto* streaming_context_a =
      static_cast<AudioStreamingContext*>(context_a.get());
  for (const auto& [name, spec] : expected_buffer_specs) {
    ASSERT_TRUE(streaming_context_a->state_buffers().contains(name))
        << "for state buffer " << name;
    const TensorBuffer& buffer = streaming_context_a->state_buffers().at(name);
    auto buffer_type = buffer.BufferType();
    auto buffer_size = buffer.Size();
    ASSERT_TRUE(buffer_type.HasValue()) << "for state buffer " << name;
    ASSERT_TRUE(buffer_size.HasValue()) << "for state buffer " << name;
    EXPECT_EQ(*buffer_type, spec.first) << "for state buffer " << name;
    EXPECT_EQ(*buffer_size, spec.second) << "for state buffer " << name;
  }

  ASSERT_OK_AND_ASSIGN(auto session_a_chunk2, make_chunk(2.0f));
  ASSERT_OK_AND_ASSIGN(auto expected_a_audio,
                       audio_executor->Encode(session_a_chunk2));
  ASSERT_OK_AND_ASSIGN(
      auto expected_a_embeddings_ptr,
      expected_a_audio.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto expected_a_embeddings,
                       GetDataAsVector<float>(*expected_a_embeddings_ptr));

  // Session B: restore fresh context, encode different chunk 1, snapshot, and
  // encode chunk 2.
  ASSERT_OK(audio_executor->RestoreContext(std::move(initial_context)));
  ASSERT_OK_AND_ASSIGN(auto session_b_chunk1, make_chunk(3.0f));
  ASSERT_OK(audio_executor->Encode(session_b_chunk1).status());
  ASSERT_OK_AND_ASSIGN(auto context_b, audio_executor->CloneContext());
  auto* streaming_context_b =
      static_cast<AudioStreamingContext*>(context_b.get());
  for (const auto& [name, spec] : expected_buffer_specs) {
    ASSERT_TRUE(streaming_context_b->state_buffers().contains(name))
        << "for state buffer " << name;
    const TensorBuffer& buffer_b =
        streaming_context_b->state_buffers().at(name);
    const TensorBuffer& buffer_a =
        streaming_context_a->state_buffers().at(name);
    auto buffer_type = buffer_b.BufferType();
    auto buffer_size = buffer_b.Size();
    ASSERT_TRUE(buffer_type.HasValue()) << "for state buffer " << name;
    ASSERT_TRUE(buffer_size.HasValue()) << "for state buffer " << name;
    EXPECT_EQ(*buffer_type, spec.first) << "for state buffer " << name;
    EXPECT_EQ(*buffer_size, spec.second) << "for state buffer " << name;
    EXPECT_NE(buffer_b.Get(), buffer_a.Get()) << "for state buffer " << name;
  }

  ASSERT_OK_AND_ASSIGN(auto session_b_chunk2, make_chunk(0.5f));
  ASSERT_OK_AND_ASSIGN(auto expected_b_audio,
                       audio_executor->Encode(session_b_chunk2));
  ASSERT_OK_AND_ASSIGN(
      auto expected_b_embeddings_ptr,
      expected_b_audio.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto expected_b_embeddings,
                       GetDataAsVector<float>(*expected_b_embeddings_ptr));
  EXPECT_NE(expected_a_embeddings, expected_b_embeddings);

  // Switch back to Session A and verify chunk 2 matches Session A's baseline.
  ASSERT_OK(audio_executor->RestoreContext(std::move(context_a)));
  ASSERT_OK_AND_ASSIGN(auto restored_a_audio,
                       audio_executor->Encode(session_a_chunk2));
  ASSERT_OK_AND_ASSIGN(
      auto restored_a_embeddings_ptr,
      restored_a_audio.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto restored_a_embeddings,
                       GetDataAsVector<float>(*restored_a_embeddings_ptr));
  EXPECT_EQ(restored_a_audio.GetValidTokens(),
            expected_a_audio.GetValidTokens());
  EXPECT_EQ(restored_a_embeddings, expected_a_embeddings);

  // Switch back to Session B and verify chunk 2 matches Session B's baseline.
  ASSERT_OK(audio_executor->RestoreContext(std::move(context_b)));
  ASSERT_OK_AND_ASSIGN(auto restored_b_audio,
                       audio_executor->Encode(session_b_chunk2));
  ASSERT_OK_AND_ASSIGN(
      auto restored_b_embeddings_ptr,
      restored_b_audio.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto restored_b_embeddings,
                       GetDataAsVector<float>(*restored_b_embeddings_ptr));
  EXPECT_EQ(restored_b_audio.GetValidTokens(),
            expected_b_audio.GetValidTokens());
  EXPECT_EQ(restored_b_embeddings, expected_b_embeddings);
}

// Name of the audio LoRA input that test model inputs are renamed to.
constexpr absl::string_view kTestAudioLoRAInputName = "key_w_prime_left0";

// Returns the 4-byte little-endian length prefix of a FlatBuffer string.
std::string FlatBufferStringLength(size_t length) {
  std::string bytes(4, '\0');
  for (int i = 0; i < 4; ++i) {
    bytes[i] = static_cast<char>((length >> (8 * i)) & 0xff);
  }
  return bytes;
}

// Writes a copy of the test model `src_model_path` to `dst_path` whose
// encoder input `input_name` is renamed to kTestAudioLoRAInputName, so that the
// input is provided by a LoRA. The new name doesn't fit in the signature
// entry's name string, so the signature entry is pointed at the (longer)
// tensor name string "serving_default_<input_name>:0", and both are renamed by
// overwriting the tensor name string.
absl::Status WriteModelWithLoRAInput(absl::string_view src_model_path,
                                     absl::string_view input_name,
                                     const std::string& dst_path) {
  std::ifstream src_file(
      std::filesystem::path(::testing::SrcDir()) / std::string(src_model_path),
      std::ios::binary);
  if (!src_file.is_open()) {
    return absl::NotFoundError("Failed to open the test model.");
  }
  std::string model_bytes((std::istreambuf_iterator<char>(src_file)),
                          std::istreambuf_iterator<char>());
  // A signature entry whose name string directly follows its offset field.
  const std::string from_sig_entry = absl::StrCat(
      FlatBufferStringLength(4), FlatBufferStringLength(input_name.size()),
      input_name);
  const std::string tensor_name =
      absl::StrCat("serving_default_", input_name, ":0");
  const std::string from_tensor_name =
      absl::StrCat(FlatBufferStringLength(tensor_name.size()), tensor_name);
  const size_t sig_pos = model_bytes.find(from_sig_entry);
  const size_t tensor_pos = model_bytes.find(from_tensor_name);
  if (sig_pos == std::string::npos || tensor_pos == std::string::npos ||
      tensor_pos <= sig_pos ||
      tensor_name.size() < kTestAudioLoRAInputName.size()) {
    return absl::NotFoundError("Unexpected test model layout.");
  }
  // The signature entry starts with the offset of its name string, relative to
  // the offset field itself.
  model_bytes.replace(sig_pos, 4, FlatBufferStringLength(tensor_pos - sig_pos));
  std::string to_tensor_name =
      absl::StrCat(FlatBufferStringLength(kTestAudioLoRAInputName.size()),
                   kTestAudioLoRAInputName);
  to_tensor_name.resize(from_tensor_name.size(), '\0');
  model_bytes.replace(tensor_pos, from_tensor_name.size(), to_tensor_name);

  std::ofstream dst_file(dst_path, std::ios::binary | std::ios::trunc);
  if (!dst_file.is_open()) {
    return absl::InternalError("Failed to write the test model.");
  }
  dst_file.write(model_bytes.data(), model_bytes.size());
  return absl::OkStatus();
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StreamingEncoder_LoRAInputNotPartOfStreamingState) {
  const std::string model_path =
      (std::filesystem::path(::testing::TempDir()) /
       "dummy_audio_streaming_lora_input.litertlm")
          .string();
  ASSERT_OK(WriteModelWithLoRAInput(kTestAudioStreamingModelPath,
                                    "segment_mask", model_path));
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_, model_path, /*max_sequence_length=*/0,
                          Backend::CPU, /*audio_buffering_enabled=*/true));

  // The LoRA input is not part of the streaming state, so contexts don't hold a
  // copy of it that RestoreContext() would write back over the LoRA buffer.
  ASSERT_OK_AND_ASSIGN(auto new_context, audio_executor->CreateNewContext());
  auto* new_streaming_context =
      static_cast<AudioStreamingContext*>(new_context.get());
  EXPECT_FALSE(
      new_streaming_context->state_buffers().contains(kTestAudioLoRAInputName));
  EXPECT_TRUE(
      new_streaming_context->state_buffers().contains("feature_state_0"));

  ASSERT_OK_AND_ASSIGN(auto cloned_context, audio_executor->CloneContext());
  auto* cloned_streaming_context =
      static_cast<AudioStreamingContext*>(cloned_context.get());
  EXPECT_FALSE(cloned_streaming_context->state_buffers().contains(
      kTestAudioLoRAInputName));
  EXPECT_TRUE(
      cloned_streaming_context->state_buffers().contains("feature_state_0"));

  EXPECT_OK(audio_executor->RestoreContext(std::move(new_context)));
  EXPECT_OK(audio_executor->Reset());
  EXPECT_OK(audio_executor->UseLoRA(std::nullopt));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       ShouldShareProducerBufferDecisionLogic) {
  constexpr std::array<uint8_t, kEmbeddingSequenceLength> kMaskData = {1, 1, 1,
                                                                       0, 0};
  const RankedTensorType mask_tensor_type(
      GetElementType<bool>(),
      Layout(Dimensions({1, kEmbeddingSequenceLength})));
  ASSERT_OK_AND_ASSIGN(auto host_producer, CreateTensorBuffer<const uint8_t>(
                                               kMaskData, mask_tensor_type));
  ASSERT_OK_AND_ASSIGN(auto host_consumer, CreateTensorBuffer<const uint8_t>(
                                               kMaskData, mask_tensor_type));
  ASSERT_OK_AND_ASSIGN(bool share_host_to_host,
                       ShouldShareProducerBuffer(host_producer, host_consumer));
  EXPECT_TRUE(share_host_to_host);

  EXPECT_TRUE(ShouldShareProducerBuffer(TensorBufferType::kHostMemory, 5,
                                        TensorBufferType::kHostMemory, 5));
  EXPECT_TRUE(
      ShouldShareProducerBuffer(TensorBufferType::kWebGpuBufferPacked, 5,
                                TensorBufferType::kWebGpuBufferPacked, 5));
  EXPECT_TRUE(ShouldShareProducerBuffer(TensorBufferType::kHostMemory, 5,
                                        TensorBufferType::kWebGpuBufferPacked,
                                        5));
  EXPECT_FALSE(ShouldShareProducerBuffer(TensorBufferType::kWebGpuBufferPacked,
                                         5, TensorBufferType::kHostMemory, 5));
  EXPECT_FALSE(ShouldShareProducerBuffer(TensorBufferType::kOpenClBufferPacked,
                                         5, TensorBufferType::kHostMemory, 5));
  EXPECT_TRUE(ShouldShareProducerBuffer(TensorBufferType::kWebGpuBufferPacked,
                                        5, TensorBufferType::kHostMemory, 8));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       Encode_PopulatesUnsharedAdapterHostMaskBuffer) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  TensorBuffer* encoder_mask_buffer =
      GetEncoderOutputMaskBuffer(*audio_executor);
  TensorBuffer* adapter_mask_buffer = GetAdapterMaskBuffer(*audio_executor);
  ASSERT_NE(encoder_mask_buffer, nullptr);
  ASSERT_NE(adapter_mask_buffer, nullptr);
  TensorBuffer& encoder_features_buffer =
      GetEncoderOutputFeaturesBuffer(*audio_executor);
  TensorBuffer& adapter_features_buffer =
      GetAdapterFeaturesBuffer(*audio_executor);

  // Replace the adapter's mask and features buffers with separate host buffers
  // so they do not alias the encoder's output buffers (matching the state when
  // ShouldShareProducerBuffer returns false for a host adapter behind a device
  // encoder).
  constexpr std::array<uint8_t, kEmbeddingSequenceLength> kInitialAdapterMask =
      {0, 0, 0, 0, 0};
  auto mask_type = adapter_mask_buffer->TensorType();
  ASSERT_TRUE(mask_type.HasValue());
  ASSERT_OK_AND_ASSIGN(
      *adapter_mask_buffer,
      CreateTensorBuffer<const uint8_t>(kInitialAdapterMask, *mask_type));
  ASSERT_NE(adapter_mask_buffer->Get(), encoder_mask_buffer->Get());

  constexpr std::array<float, kEmbeddingSequenceLength * 4>
      kInitialAdapterFeatures = {};
  auto features_type = adapter_features_buffer.TensorType();
  ASSERT_TRUE(features_type.HasValue());
  ASSERT_OK_AND_ASSIGN(
      adapter_features_buffer,
      CreateTensorBuffer<const float>(kInitialAdapterFeatures, *features_type));
  ASSERT_NE(adapter_features_buffer.Get(), encoder_features_buffer.Get());

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

  constexpr std::array<bool, kSpectrogramSequenceLength> partial_mask_data = {
      true, true, true, true, true, true, false, false, false, false};
  ASSERT_OK_AND_ASSIGN(
      auto partial_mask_tensor_buffer,
      CreateTensorBuffer<const bool>(
          partial_mask_data,
          RankedTensorType(
              GetElementType<bool>(),
              Layout(Dimensions({1, kSpectrogramSequenceLength})))));

  ASSERT_OK_AND_ASSIGN(auto executor_audio_data,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer,
                                              partial_mask_tensor_buffer));
  EXPECT_EQ(executor_audio_data.GetValidTokens(), 3);
  EXPECT_NE(adapter_mask_buffer->Get(), encoder_mask_buffer->Get());
  EXPECT_NE(adapter_features_buffer.Get(), encoder_features_buffer.Get());

  ASSERT_OK_AND_ASSIGN(auto encoder_mask_after_partial,
                       GetDataAsVector<uint8_t>(*encoder_mask_buffer));
  ASSERT_OK_AND_ASSIGN(auto adapter_mask_after_partial,
                       GetDataAsVector<uint8_t>(*adapter_mask_buffer));
  EXPECT_EQ(adapter_mask_after_partial, encoder_mask_after_partial);
  EXPECT_THAT(adapter_mask_after_partial, ElementsAre(1, 1, 1, 0, 0));

  ASSERT_OK_AND_ASSIGN(auto encoder_features_after_partial,
                       GetDataAsVector<float>(encoder_features_buffer));
  ASSERT_OK_AND_ASSIGN(auto adapter_features_after_partial,
                       GetDataAsVector<float>(adapter_features_buffer));
  EXPECT_EQ(adapter_features_after_partial, encoder_features_after_partial);

  ASSERT_OK_AND_ASSIGN(
      auto projected_audio_embeddings_ptr,
      executor_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto projected_audio_embeddings_data,
                       GetDataAsVector<float>(*projected_audio_embeddings_ptr));
  EXPECT_THAT(projected_audio_embeddings_data,
              ElementsAre(1., 2., 4., 6., 6., 6., 1., 3., 6., 9., 9., 9., 1.,
                          3., 5., 8., 8., 8.));

  // A subsequent Encode with a full mask updates the unshared adapter mask
  // buffer again.
  ASSERT_OK_AND_ASSIGN(auto full_audio_data,
                       audio_executor->Encode(mel_spectrogram_tensor_buffer));
  EXPECT_EQ(full_audio_data.GetValidTokens(), kEmbeddingSequenceLength);
  ASSERT_OK_AND_ASSIGN(auto adapter_mask_after_full,
                       GetDataAsVector<uint8_t>(*adapter_mask_buffer));
  EXPECT_THAT(adapter_mask_after_full, ElementsAre(1, 1, 1, 1, 1));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       TrailingZeroPadding_OverwritesStaleInputBuffers_StaticModel) {
  // First executor encodes a full 10-frame window with all-ones, populating
  // all input spectrogram and mask buffers with non-zero data.
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  std::vector<float> full_spectrogram(
      kSpectrogramSequenceLength * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto full_tensor_buffer,
      CreateTensorBuffer<const float>(
          full_spectrogram,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kSpectrogramSequenceLength,
                                              kSpectrogramFrequencySlots})))));
  ASSERT_OK(audio_executor->Encode(full_tensor_buffer));

  // Next, encode a partial window (6 frames of ones, 4 masked out).
  constexpr std::array<bool, kSpectrogramSequenceLength> partial_mask_data = {
      true, true, true, true, true, true, false, false, false, false};
  ASSERT_OK_AND_ASSIGN(
      auto partial_mask_tensor,
      CreateTensorBuffer<const bool>(
          partial_mask_data,
          RankedTensorType(
              GetElementType<bool>(),
              Layout(Dimensions({1, kSpectrogramSequenceLength})))));
  ASSERT_OK_AND_ASSIGN(
      auto audio_data_reused,
      audio_executor->Encode(full_tensor_buffer, partial_mask_tensor));

  // Reference executor: freshly created, encodes only the partial window.
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor_clean,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));
  ASSERT_OK_AND_ASSIGN(
      auto audio_data_clean,
      audio_executor_clean->Encode(full_tensor_buffer, partial_mask_tensor));

  ASSERT_OK_AND_ASSIGN(
      auto reused_proj_ptr,
      audio_data_reused.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto reused_proj_data,
                       GetDataAsVector<float>(*reused_proj_ptr));
  ASSERT_OK_AND_ASSIGN(
      auto clean_proj_ptr,
      audio_data_clean.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto clean_proj_data,
                       GetDataAsVector<float>(*clean_proj_ptr));
  EXPECT_EQ(reused_proj_data, clean_proj_data);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       TrailingZeroPadding_OverwritesStaleInputBuffers_StreamingModel) {
  // First executor encodes a full chunk (10 frames), then a partial chunk (5
  // frames) flushed.
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU,
                          /*audio_buffering_enabled=*/true));

  std::vector<float> full_spectrogram(10 * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto full_tensor_buffer,
      CreateTensorBuffer<const float>(
          full_spectrogram,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 10, kSpectrogramFrequencySlots})))));
  ASSERT_OK(audio_executor->Encode(full_tensor_buffer));

  // Reset executor to clear streaming state but keep buffers allocated/dirty.
  ASSERT_OK(audio_executor->Reset());

  std::vector<float> partial_spectrogram(5 * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto partial_tensor_buffer,
      CreateTensorBuffer<const float>(
          partial_spectrogram,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 5, kSpectrogramFrequencySlots})))));
  ASSERT_OK(audio_executor->Encode(partial_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(auto flush_data_reused, audio_executor->Flush());

  // Reference executor: freshly created.
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor_clean,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU,
                          /*audio_buffering_enabled=*/true));
  ASSERT_OK(audio_executor_clean->Encode(partial_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(auto flush_data_clean, audio_executor_clean->Flush());

  ASSERT_OK_AND_ASSIGN(
      auto reused_proj_ptr,
      flush_data_reused.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto reused_proj_data,
                       GetDataAsVector<float>(*reused_proj_ptr));
  ASSERT_OK_AND_ASSIGN(
      auto clean_proj_ptr,
      flush_data_clean.GetMutableProjectedAudioEmbeddingsPtr());
  ASSERT_OK_AND_ASSIGN(auto clean_proj_data,
                       GetDataAsVector<float>(*clean_proj_ptr));
  EXPECT_EQ(reused_proj_data, clean_proj_data);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       OutputTensorShape_FullVsPartialWindow_StaticModelWithAdapter) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  std::vector<float> mel_spectrogram_data(
      kSpectrogramSequenceLength * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto mel_tensor,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kSpectrogramSequenceLength,
                                              kSpectrogramFrequencySlots})))));

  // 1. Full unmasked window: buffer_tokens == max_tokens == 5.
  ASSERT_OK_AND_ASSIGN(auto full_audio_data,
                       audio_executor->Encode(mel_tensor));
  EXPECT_EQ(full_audio_data.GetValidTokens(), 5);

  ASSERT_OK_AND_ASSIGN(auto full_proj_ptr,
                       full_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto full_proj_type = full_proj_ptr->TensorType();
  ASSERT_TRUE(full_proj_type.HasValue());
  EXPECT_THAT(full_proj_type->Layout().Dimensions(),
              ElementsAre(1, 5, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto full_audio_ptr,
                       full_audio_data.GetAudioEmbeddingsPtr());
  auto full_audio_type = full_audio_ptr->TensorType();
  ASSERT_TRUE(full_audio_type.HasValue());
  EXPECT_THAT(full_audio_type->Layout().Dimensions(), ElementsAre(1, 5, 4));

  // 2. Partial/masked window: buffer_tokens == 3 != max_tokens == 5.
  constexpr std::array<bool, kSpectrogramSequenceLength> partial_mask_data = {
      true, true, true, true, true, true, false, false, false, false};
  ASSERT_OK_AND_ASSIGN(
      auto partial_mask_tensor,
      CreateTensorBuffer<const bool>(
          partial_mask_data,
          RankedTensorType(
              GetElementType<bool>(),
              Layout(Dimensions({1, kSpectrogramSequenceLength})))));

  ASSERT_OK_AND_ASSIGN(auto partial_audio_data,
                       audio_executor->Encode(mel_tensor, partial_mask_tensor));
  EXPECT_EQ(partial_audio_data.GetValidTokens(), 3);

  ASSERT_OK_AND_ASSIGN(
      auto partial_proj_ptr,
      partial_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto partial_proj_type = partial_proj_ptr->TensorType();
  ASSERT_TRUE(partial_proj_type.HasValue());
  EXPECT_THAT(partial_proj_type->Layout().Dimensions(),
              ElementsAre(1, 3, kEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto partial_audio_ptr,
                       partial_audio_data.GetAudioEmbeddingsPtr());
  auto partial_audio_type = partial_audio_ptr->TensorType();
  ASSERT_TRUE(partial_audio_type.HasValue());
  EXPECT_THAT(partial_audio_type->Layout().Dimensions(), ElementsAre(1, 3, 4));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       OutputTensorShape_FullVsPartialWindow_StaticModelWithoutAdapter) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioNoMaskModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  std::vector<float> mel_spectrogram_data(
      kSpectrogramSequenceLength * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto mel_tensor,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kSpectrogramSequenceLength,
                                              kSpectrogramFrequencySlots})))));

  // 1. Full unmasked window: buffer_tokens == max_tokens == 5.
  ASSERT_OK_AND_ASSIGN(auto full_audio_data,
                       audio_executor->Encode(mel_tensor));
  EXPECT_EQ(full_audio_data.GetValidTokens(), 5);

  ASSERT_OK_AND_ASSIGN(auto full_proj_ptr,
                       full_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto full_proj_type = full_proj_ptr->TensorType();
  ASSERT_TRUE(full_proj_type.HasValue());
  EXPECT_THAT(full_proj_type->Layout().Dimensions(),
              ElementsAre(1, 5, kNoMaskEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto full_audio_ptr,
                       full_audio_data.GetAudioEmbeddingsPtr());
  auto full_audio_type = full_audio_ptr->TensorType();
  ASSERT_TRUE(full_audio_type.HasValue());
  EXPECT_THAT(full_audio_type->Layout().Dimensions(),
              ElementsAre(1, 5, kNoMaskEmbeddingDimensions));

  // 2. Partial/masked window: buffer_tokens == 3 != max_tokens == 5.
  constexpr std::array<bool, kSpectrogramSequenceLength> partial_mask_data = {
      true, true, true, true, true, true, false, false, false, false};
  ASSERT_OK_AND_ASSIGN(
      auto partial_mask_tensor,
      CreateTensorBuffer<const bool>(
          partial_mask_data,
          RankedTensorType(
              GetElementType<bool>(),
              Layout(Dimensions({1, kSpectrogramSequenceLength})))));

  ASSERT_OK_AND_ASSIGN(auto partial_audio_data,
                       audio_executor->Encode(mel_tensor, partial_mask_tensor));
  EXPECT_EQ(partial_audio_data.GetValidTokens(), 3);

  ASSERT_OK_AND_ASSIGN(
      auto partial_proj_ptr,
      partial_audio_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto partial_proj_type = partial_proj_ptr->TensorType();
  ASSERT_TRUE(partial_proj_type.HasValue());
  EXPECT_THAT(partial_proj_type->Layout().Dimensions(),
              ElementsAre(1, 3, kNoMaskEmbeddingDimensions));

  ASSERT_OK_AND_ASSIGN(auto partial_audio_ptr,
                       partial_audio_data.GetAudioEmbeddingsPtr());
  auto partial_audio_type = partial_audio_ptr->TensorType();
  ASSERT_TRUE(partial_audio_type.HasValue());
  EXPECT_THAT(partial_audio_type->Layout().Dimensions(),
              ElementsAre(1, 3, kNoMaskEmbeddingDimensions));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       OutputTensorShape_FullVsPartialWindow_StreamingModel) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU,
                          /*audio_buffering_enabled=*/true));

  // 1. Partial window flush: 5 frames buffered -> flush pads to 8 frames -> 4
  // tokens.
  std::vector<float> mel_spectrogram_data(5 * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto partial_tensor_buffer,
      CreateTensorBuffer<const float>(
          mel_spectrogram_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 5, kSpectrogramFrequencySlots})))));

  ASSERT_OK_AND_ASSIGN(auto encode_data,
                       audio_executor->Encode(partial_tensor_buffer));
  EXPECT_EQ(encode_data.GetValidTokens(), 0);

  ASSERT_OK_AND_ASSIGN(auto flush_data, audio_executor->Flush());
  EXPECT_EQ(flush_data.GetValidTokens(), 4);

  ASSERT_OK_AND_ASSIGN(auto flush_proj_ptr,
                       flush_data.GetMutableProjectedAudioEmbeddingsPtr());
  auto flush_proj_type = flush_proj_ptr->TensorType();
  ASSERT_TRUE(flush_proj_type.HasValue());
  EXPECT_THAT(flush_proj_type->Layout().Dimensions(),
              ElementsAre(1, 4, kEmbeddingDimensions));
}

}  // namespace

}  // namespace litert::lm
