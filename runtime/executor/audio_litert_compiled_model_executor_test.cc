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
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "runtime/util/test_utils.h"     //NOLINT

namespace litert::lm {

class AudioLiteRtCompiledModelExecutorTestPeer {
 public:
  static absl::flat_hash_map<absl::string_view, TensorBuffer>&
  GetMutableInputBuffersMap(AudioLiteRtCompiledModelExecutor& executor) {
    return executor.audio_encoder_->GetMutableInputBuffersMap();
  }
  static bool IsLoRAEnabled(const AudioLiteRtCompiledModelExecutor& executor) {
    return executor.audio_encoder_->IsLoRAEnabled();
  }
};

namespace {

constexpr absl::string_view kTestAudioModelPath =
    "litert_lm/runtime/testdata/dummy_audio_only.litertlm";
constexpr absl::string_view kTestAudioStreamingModelPath =
    "litert_lm/runtime/testdata/dummy_audio_streaming.litertlm";
constexpr absl::string_view kTestAudioNoMaskModelPath =
    "litert_lm/runtime/testdata/dummy_audio_no_mask.litertlm";
constexpr absl::string_view kTestLoRAModelPath =
    "litert_lm/runtime/testdata/test_lora_rank32_f16_all_ones.tflite";
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

absl::StatusOr<ModelAssets> CreatePatchedLoRAModelAssets(
    absl::Span<const uint8_t> payload_bytes, absl::string_view temp_filename) {
  const std::filesystem::path src_lora_path =
      std::filesystem::path(::testing::SrcDir()) /
      std::string(kTestLoRAModelPath);
  std::ifstream src_file(src_lora_path, std::ios::binary);
  if (!src_file.is_open()) {
    return absl::InternalError("Failed to open test LoRA model file.");
  }
  constexpr size_t kFlatBufferHeaderBytes = 20480;
  constexpr size_t kAlignedDataOffset = 65536;
  std::string lora_bytes(kFlatBufferHeaderBytes, '\0');
  src_file.read(lora_bytes.data(), kFlatBufferHeaderBytes);
  lora_bytes.resize(kAlignedDataOffset, '\0');

  // Rename "post_w_prime_right_34" (at 1944) to "key_w_prime_left0" and point
  // its Buffer entry (at 1984) to offset 65536 with size payload_bytes.size().
  const std::string from_tensor_name("\x15\0\0\0post_w_prime_right_34\0\0\0",
                                     28);
  const std::string to_tensor_name("\x11\0\0\0key_w_prime_left0\0\0\0\0\0\0\0",
                                   28);
  const size_t name_pos = lora_bytes.find(from_tensor_name);
  if (name_pos == std::string::npos) {
    return absl::InternalError("Failed to find tensor name in test LoRA file.");
  }
  lora_bytes.replace(name_pos, from_tensor_name.size(), to_tensor_name);

  const uint64_t data_offset = kAlignedDataOffset;
  const uint64_t data_size = payload_bytes.size();
  std::string buffer_entry(16, '\0');
  for (int i = 0; i < 8; ++i) {
    buffer_entry[i] = static_cast<char>((data_offset >> (8 * i)) & 0xFF);
    buffer_entry[8 + i] = static_cast<char>((data_size >> (8 * i)) & 0xFF);
  }
  lora_bytes.replace(name_pos + 40, buffer_entry.size(), buffer_entry);
  lora_bytes.append(reinterpret_cast<const char*>(payload_bytes.data()),
                    payload_bytes.size());
  lora_bytes.resize(2 * kAlignedDataOffset, '\0');

  const std::filesystem::path temp_lora_path =
      std::filesystem::path(::testing::TempDir()) / std::string(temp_filename);
  {
    std::ofstream dst_file(temp_lora_path, std::ios::binary | std::ios::trunc);
    if (!dst_file.is_open()) {
      return absl::InternalError("Failed to write patched LoRA file.");
    }
    dst_file.write(lora_bytes.data(), lora_bytes.size());
  }
  return ModelAssets::Create(temp_lora_path.string());
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

  // Input size 24.
  // chunk_size = 10, overlap = 3, stride = 7.
  // 3 chunks: [0, 10), [7, 17), [14, 24).
  // Total output tokens: 3 * (10 / 2) = 15.
  constexpr int kInputSequenceLength = 24;
  constexpr int kExpectedOutputTokens = 15;

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

  // Each output frame should be [1, 3, 6, 10, 15, 15] as derived.
  std::vector<float> expected_frame = {1.f, 3.f, 6.f, 10.f, 15.f, 15.f};
  std::vector<float> expected_embeddings_data;
  expected_embeddings_data.reserve(kExpectedOutputTokens *
                                   kEmbeddingDimensions);
  for (int i = 0; i < kExpectedOutputTokens; ++i) {
    expected_embeddings_data.insert(expected_embeddings_data.end(),
                                    expected_frame.begin(),
                                    expected_frame.end());
  }

  EXPECT_EQ(projected_audio_embeddings_data, expected_embeddings_data);
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
  EXPECT_EQ(executor_audio_data_encode.GetValidTokens(), 0);

  ASSERT_OK_AND_ASSIGN(auto executor_audio_data_flush, audio_executor->Flush());
  EXPECT_EQ(executor_audio_data_flush.GetValidTokens(), 3);
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
  EXPECT_EQ(audio_data2.GetValidTokens(), 5);

  ASSERT_OK_AND_ASSIGN(auto audio_data_flush, audio_executor->Flush());
  EXPECT_EQ(audio_data_flush.GetValidTokens(), 3);
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

  // Flush the executor once -> processes 5 buffered frames.
  ASSERT_OK_AND_ASSIGN(auto flush_data1, audio_executor->Flush());
  EXPECT_EQ(flush_data1.GetValidTokens(), 3);

  // Restore context -> restores the 5 buffered frames back into executor.
  ASSERT_OK(audio_executor->RestoreContext(std::move(cloned_context)));

  // Flush again after restore -> processes restored 5 buffered frames.
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

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StaticEncoder_LoRABuffersPreservedAcrossMultiWindowAndReset) {
  const std::filesystem::path src_model_path =
      std::filesystem::path(::testing::SrcDir()) /
      std::string(kTestAudioModelPath);
  std::ifstream src_file(src_model_path, std::ios::binary);
  ASSERT_TRUE(src_file.is_open());
  std::string model_bytes((std::istreambuf_iterator<char>(src_file)),
                          std::istreambuf_iterator<char>());

  // Swap the two entries in SubGraph.inputs (at 18004) so "src_inputs"
  // (tensor 1, shape [1, 10, 8]) is input 0 (used by audio_executor_utils when
  // "mask" is absent), and redirect "mask" (uoffset 4 at 16600) to
  // "CONVERSION_METADATA" (at 16640, relative uoffset 40 = 0x28) renamed to
  // "key_w_prime_left0".
  const std::string from_subgraph_inputs("\x02\0\0\0\x00\0\0\0\x01\0\0\0", 12);
  const std::string to_subgraph_inputs("\x02\0\0\0\x01\0\0\0\x00\0\0\0", 12);
  const size_t vec_pos = model_bytes.find(from_subgraph_inputs);
  ASSERT_NE(vec_pos, std::string::npos);
  model_bytes.replace(vec_pos, from_subgraph_inputs.size(), to_subgraph_inputs);

  const std::string from_mask_entry("\x04\0\0\0\x04\0\0\0mask", 12);
  const std::string to_mask_entry("\x28\0\0\0\x04\0\0\0mask", 12);
  const size_t mask_pos = model_bytes.find(from_mask_entry);
  ASSERT_NE(mask_pos, std::string::npos);
  const std::string from_metadata("\x13\0\0\0CONVERSION_METADATA", 23);
  const std::string to_lora_name("\x11\0\0\0key_w_prime_left0\0\0", 23);
  const size_t metadata_pos = model_bytes.find(from_metadata, mask_pos);
  ASSERT_EQ(metadata_pos - mask_pos, 40);
  model_bytes.replace(mask_pos, from_mask_entry.size(), to_mask_entry);
  model_bytes.replace(metadata_pos, from_metadata.size(), to_lora_name);

  const std::filesystem::path temp_model_path =
      std::filesystem::path(::testing::TempDir()) /
      "dummy_audio_only_lora.litertlm";
  {
    std::ofstream dst_file(temp_model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(dst_file.is_open());
    dst_file.write(model_bytes.data(), model_bytes.size());
  }

  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_, temp_model_path.string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  auto& input_buffers_map =
      AudioLiteRtCompiledModelExecutorTestPeer::GetMutableInputBuffersMap(
          *audio_executor);
  ASSERT_TRUE(input_buffers_map.contains("key_w_prime_left0"));

  // 1. Verify LoRA input buffer is zero-initialized on Create when no LoRA is
  // loaded.
  ASSERT_OK_AND_ASSIGN(
      auto initial_lora_data,
      GetDataAsVector<uint8_t>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(initial_lora_data, ::testing::Each(uint8_t{0}));

  // 2. LoadLoRA(0) and UseLoRA(0) via the real LoraManager pipeline, and verify
  // multi-window Encode() and Reset() do not wipe the bound LoRA buffer or the
  // LoRA weights stored inside LoraManager.
  std::vector<uint8_t> non_zero_lora(initial_lora_data.size(), 1);
  ASSERT_OK_AND_ASSIGN(
      auto lora_assets,
      CreatePatchedLoRAModelAssets(non_zero_lora, "static_audio_lora.tflite"));
  ASSERT_OK(audio_executor->LoadLoRA(0, lora_assets));
  ASSERT_OK(audio_executor->UseLoRA(0));
  EXPECT_TRUE(
      AudioLiteRtCompiledModelExecutorTestPeer::IsLoRAEnabled(*audio_executor));

  // Encode 20 frames (2 full windows of 10 frames). In this dummy graph, the
  // second input ("key_w_prime_left0") drives mask_out, so both windows
  // produce 5 valid tokens (10 total) only if "key_w_prime_left0" remains all
  // ones across both window 0 and window 1.
  std::vector<float> multi_window_spec(
      2 * kSpectrogramSequenceLength * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto spec_buffer,
      CreateTensorBuffer<const float>(
          multi_window_spec,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, 2 * kSpectrogramSequenceLength,
                                              kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto encode_result, audio_executor->Encode(spec_buffer));
  EXPECT_EQ(encode_result.GetValidTokens(), 2 * kEmbeddingSequenceLength);

  ASSERT_OK_AND_ASSIGN(
      auto after_encode_lora_data,
      GetDataAsVector<uint8_t>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(after_encode_lora_data, ::testing::Each(uint8_t{1}));

  // Reset() must also preserve bound LoRA buffers while clearing non-LoRA input
  // buffers.
  ASSERT_OK(audio_executor->Reset());
  ASSERT_OK_AND_ASSIGN(
      auto after_reset_lora_data,
      GetDataAsVector<uint8_t>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(after_reset_lora_data, ::testing::Each(uint8_t{1}));

  // 3. UseLoRA(std::nullopt) must replace the bound LoRA buffer with a fresh
  // zero-initialized buffer WITHOUT zeroing the shared LoRA buffer inside
  // LoraManager.
  ASSERT_OK(audio_executor->UseLoRA(std::nullopt));
  EXPECT_FALSE(
      AudioLiteRtCompiledModelExecutorTestPeer::IsLoRAEnabled(*audio_executor));
  ASSERT_OK_AND_ASSIGN(
      auto after_unload_lora_data,
      GetDataAsVector<uint8_t>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(after_unload_lora_data, ::testing::Each(uint8_t{0}));

  // Encoding with LoRA disabled now sees the zeroed "key_w_prime_left0" buffer
  // (producing 0 valid tokens), and re-enabling UseLoRA(0) without reloading
  // restores the non-zero LoRA weights from LoraManager.
  ASSERT_OK_AND_ASSIGN(auto base_encode_result,
                       audio_executor->Encode(spec_buffer));
  EXPECT_EQ(base_encode_result.GetValidTokens(), 0);

  ASSERT_OK(audio_executor->UseLoRA(0));
  EXPECT_TRUE(
      AudioLiteRtCompiledModelExecutorTestPeer::IsLoRAEnabled(*audio_executor));
  ASSERT_OK_AND_ASSIGN(auto reenabled_encode_result,
                       audio_executor->Encode(spec_buffer));
  EXPECT_EQ(reenabled_encode_result.GetValidTokens(),
            2 * kEmbeddingSequenceLength);
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StaticEncoder_MultiWindowPartialChunkMasksTrailingFramesAndKeepsLoRA) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));

  auto& input_buffers_map =
      AudioLiteRtCompiledModelExecutorTestPeer::GetMutableInputBuffersMap(
          *audio_executor);
  std::vector<float> lora_weights(16, 2.5f);
  ASSERT_OK_AND_ASSIGN(
      auto shared_lora_buffer,
      CreateTensorBuffer<const float>(
          lora_weights, RankedTensorType(GetElementType<float>(),
                                         Layout(Dimensions({1, 16})))));
  auto bound_lora_buffer = shared_lora_buffer.Duplicate();
  ASSERT_TRUE(bound_lora_buffer.HasValue());
  input_buffers_map["post_w_prime_right0"] = std::move(*bound_lora_buffer);

  // 15 frames = 1 full window (10 frames -> 5 tokens) + 1 partial window
  // (5 frames -> 3 tokens), total 8 valid tokens.
  constexpr int kTotalFrames = 15;
  std::vector<float> spec_data(kTotalFrames * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto spec_buffer,
      CreateTensorBuffer<const float>(
          spec_data,
          RankedTensorType(GetElementType<float>(),
                           Layout(Dimensions({1, kTotalFrames,
                                              kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto result, audio_executor->Encode(spec_buffer));
  EXPECT_EQ(result.GetValidTokens(), 8);

  ASSERT_OK_AND_ASSIGN(
      auto lora_after,
      GetDataAsVector<float>(input_buffers_map.at("post_w_prime_right0")));
  EXPECT_THAT(lora_after, ::testing::Each(2.5f));
  ASSERT_OK_AND_ASSIGN(auto shared_lora_after,
                       GetDataAsVector<float>(shared_lora_buffer));
  EXPECT_THAT(shared_lora_after, ::testing::Each(2.5f));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StreamingEncoder_LoRABuffersPreservedAcrossContextsMultiWindowAndReset) {
  const std::filesystem::path src_model_path =
      std::filesystem::path(::testing::SrcDir()) /
      std::string(kTestAudioStreamingModelPath);
  std::ifstream src_file(src_model_path, std::ios::binary);
  ASSERT_TRUE(src_file.is_open());
  std::string model_bytes((std::istreambuf_iterator<char>(src_file)),
                          std::istreambuf_iterator<char>());

  // Swap tensor 0 ("feature_state_0") and tensor 3 ("src_inputs" [1, 10, 8]) in
  // SubGraph.inputs (at 18180), rename "segment_mask" to "prev_mask" (so
  // IsStreamingEncoder detects "prev_" and Initialize sets the mask buffer to
  // all 1s), and redirect "prev_key_0" (uoffset 4 at 16676) to
  // "CONVERSION_METADATA" (at 16748, relative uoffset 72 = 0x48) renamed to
  // "key_w_prime_left0".
  const std::string from_subgraph_inputs(
      "\x04\0\0\0\x00\0\0\0\x01\0\0\0\x02\0\0\0\x03\0\0\0", 20);
  const std::string to_subgraph_inputs(
      "\x04\0\0\0\x03\0\0\0\x01\0\0\0\x02\0\0\0\x00\0\0\0", 20);
  const size_t vec_pos = model_bytes.find(from_subgraph_inputs);
  ASSERT_NE(vec_pos, std::string::npos);
  model_bytes.replace(vec_pos, from_subgraph_inputs.size(), to_subgraph_inputs);

  const std::string from_seg_mask("\x0c\0\0\0segment_mask", 16);
  const std::string to_prev_mask("\x09\0\0\0prev_mask\0\0\0", 16);
  const size_t seg_mask_pos = model_bytes.find(from_seg_mask);
  ASSERT_NE(seg_mask_pos, std::string::npos);
  model_bytes.replace(seg_mask_pos, from_seg_mask.size(), to_prev_mask);

  const std::string from_prev_key("\x04\0\0\0\x0a\0\0\0prev_key_0", 18);
  const std::string to_prev_key("\x48\0\0\0\x0a\0\0\0prev_key_0", 18);
  const size_t prev_key_pos = model_bytes.find(from_prev_key);
  ASSERT_NE(prev_key_pos, std::string::npos);
  const std::string from_metadata("\x13\0\0\0CONVERSION_METADATA", 23);
  const std::string to_lora_name("\x11\0\0\0key_w_prime_left0\0\0", 23);
  const size_t metadata_pos = model_bytes.find(from_metadata, prev_key_pos);
  ASSERT_EQ(metadata_pos - prev_key_pos, 72);
  model_bytes.replace(prev_key_pos, from_prev_key.size(), to_prev_key);
  model_bytes.replace(metadata_pos, from_metadata.size(), to_lora_name);

  const std::filesystem::path temp_model_path =
      std::filesystem::path(::testing::TempDir()) /
      "dummy_audio_streaming_lora.litertlm";
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

  auto& input_buffers_map =
      AudioLiteRtCompiledModelExecutorTestPeer::GetMutableInputBuffersMap(
          *audio_executor);
  ASSERT_TRUE(input_buffers_map.contains("key_w_prime_left0"));

  // 1. Verify zero-initialization on Create when no LoRA is loaded.
  ASSERT_OK_AND_ASSIGN(
      auto initial_lora_data,
      GetDataAsVector<float>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(initial_lora_data, ::testing::Each(0.0f));

  // 2. LoadLoRA(0) and UseLoRA(0) via the real LoraManager pipeline.
  std::vector<float> non_zero_lora(initial_lora_data.size(), 3.5f);
  absl::Span<const uint8_t> lora_bytes(
      reinterpret_cast<const uint8_t*>(non_zero_lora.data()),
      non_zero_lora.size() * sizeof(float));
  ASSERT_OK_AND_ASSIGN(
      auto lora_assets,
      CreatePatchedLoRAModelAssets(lora_bytes, "streaming_audio_lora.tflite"));
  ASSERT_OK(audio_executor->LoadLoRA(0, lora_assets));
  ASSERT_OK(audio_executor->UseLoRA(0));

  // 3. CreateNewContext() must skip LoRA buffers and leave the bound LoRA
  // buffer untouched.
  ASSERT_OK_AND_ASSIGN(auto new_context, audio_executor->CreateNewContext());
  auto* new_streaming_context =
      static_cast<AudioStreamingContext*>(new_context.get());
  EXPECT_FALSE(
      new_streaming_context->state_buffers().contains("key_w_prime_left0"));
  EXPECT_TRUE(
      new_streaming_context->state_buffers().contains("feature_state_0"));

  // 4. CloneContext() must skip LoRA buffers.
  ASSERT_OK_AND_ASSIGN(auto cloned_context, audio_executor->CloneContext());
  auto* cloned_streaming_context =
      static_cast<AudioStreamingContext*>(cloned_context.get());
  EXPECT_FALSE(
      cloned_streaming_context->state_buffers().contains("key_w_prime_left0"));

  // 5. RestoreContext() must not overwrite or zero the bound LoRA buffer.
  ASSERT_OK(audio_executor->RestoreContext(std::move(new_context)));
  ASSERT_OK_AND_ASSIGN(
      auto after_restore_lora_data,
      GetDataAsVector<float>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(after_restore_lora_data, ::testing::Each(3.5f));

  // 6. Multi-window streaming Encode() (19 frames -> 2 windows at pos=0 and
  // pos=7, plus 5 buffered frames) + Flush() + Reset() must preserve LoRA.
  std::vector<float> stream_spec(19 * kSpectrogramFrequencySlots, 1.0f);
  ASSERT_OK_AND_ASSIGN(
      auto stream_spec_buffer,
      CreateTensorBuffer<const float>(
          stream_spec,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 19, kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(auto stream_encode_data,
                       audio_executor->Encode(stream_spec_buffer));
  EXPECT_EQ(stream_encode_data.GetValidTokens(), 2 * kEmbeddingSequenceLength);

  ASSERT_OK_AND_ASSIGN(auto flush_data, audio_executor->Flush());
  EXPECT_EQ(flush_data.GetValidTokens(), kEmbeddingSequenceLength);

  ASSERT_OK(audio_executor->Reset());
  ASSERT_OK_AND_ASSIGN(
      auto after_reset_lora_data,
      GetDataAsVector<float>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(after_reset_lora_data, ::testing::Each(3.5f));

  // 7. UseLoRA(std::nullopt) replaces the encoder's LoRA buffer with a
  // zero-initialized buffer without zeroing LoraManager's stored LoRA weights,
  // so a subsequent UseLoRA(0) restores the 3.5f weights.
  ASSERT_OK(audio_executor->UseLoRA(std::nullopt));
  ASSERT_OK_AND_ASSIGN(
      auto after_unload_lora_data,
      GetDataAsVector<float>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(after_unload_lora_data, ::testing::Each(0.0f));

  ASSERT_OK(audio_executor->UseLoRA(0));
  ASSERT_OK_AND_ASSIGN(
      auto after_reload_lora_data,
      GetDataAsVector<float>(input_buffers_map.at("key_w_prime_left0")));
  EXPECT_THAT(after_reload_lora_data, ::testing::Each(3.5f));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StreamingEncoder_LoRAParamTensorZeroInitializedAndPreserved) {
  const std::filesystem::path src_model_path =
      std::filesystem::path(::testing::SrcDir()) /
      std::string(kTestAudioStreamingModelPath);
  std::ifstream src_file(src_model_path, std::ios::binary);
  ASSERT_TRUE(src_file.is_open());
  std::string model_bytes((std::istreambuf_iterator<char>(src_file)),
                          std::istreambuf_iterator<char>());

  const std::string from_subgraph_inputs(
      "\x04\0\0\0\x00\0\0\0\x01\0\0\0\x02\0\0\0\x03\0\0\0", 20);
  const std::string to_subgraph_inputs(
      "\x04\0\0\0\x03\0\0\0\x01\0\0\0\x02\0\0\0\x00\0\0\0", 20);
  const size_t vec_pos = model_bytes.find(from_subgraph_inputs);
  ASSERT_NE(vec_pos, std::string::npos);
  model_bytes.replace(vec_pos, from_subgraph_inputs.size(), to_subgraph_inputs);

  const std::string from_seg_mask("\x0c\0\0\0segment_mask", 16);
  const std::string to_prev_mask("\x09\0\0\0prev_mask\0\0\0", 16);
  const size_t seg_mask_pos = model_bytes.find(from_seg_mask);
  ASSERT_NE(seg_mask_pos, std::string::npos);
  model_bytes.replace(seg_mask_pos, from_seg_mask.size(), to_prev_mask);

  const std::string from_prev_key("\x04\0\0\0\x0a\0\0\0prev_key_0", 18);
  const std::string to_prev_key("\x48\0\0\0\x0a\0\0\0prev_key_0", 18);
  const size_t prev_key_pos = model_bytes.find(from_prev_key);
  ASSERT_NE(prev_key_pos, std::string::npos);
  const std::string from_metadata("\x13\0\0\0CONVERSION_METADATA", 23);
  const std::string to_lora_param("\x11\0\0\0lora_param_tensor\0\0", 23);
  const size_t metadata_pos = model_bytes.find(from_metadata, prev_key_pos);
  ASSERT_EQ(metadata_pos - prev_key_pos, 72);
  model_bytes.replace(prev_key_pos, from_prev_key.size(), to_prev_key);
  model_bytes.replace(metadata_pos, from_metadata.size(), to_lora_param);

  const std::filesystem::path temp_model_path =
      std::filesystem::path(::testing::TempDir()) /
      "dummy_audio_streaming_lora_param.litertlm";
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

  auto& input_buffers_map =
      AudioLiteRtCompiledModelExecutorTestPeer::GetMutableInputBuffersMap(
          *audio_executor);
  ASSERT_TRUE(input_buffers_map.contains("lora_param_tensor"));
  ASSERT_OK_AND_ASSIGN(
      auto initial_param_data,
      GetDataAsVector<float>(input_buffers_map.at("lora_param_tensor")));
  EXPECT_THAT(initial_param_data, ::testing::Each(0.0f));

  std::vector<float> non_zero_param(initial_param_data.size(), 16.0f);
  auto param_tensor_type =
      input_buffers_map.at("lora_param_tensor").TensorType();
  ASSERT_TRUE(param_tensor_type.HasValue());
  ASSERT_OK_AND_ASSIGN(
      auto shared_param_buffer,
      CreateTensorBuffer<const float>(non_zero_param, *param_tensor_type));
  auto bound_param_buffer = shared_param_buffer.Duplicate();
  ASSERT_TRUE(bound_param_buffer.HasValue());
  input_buffers_map["lora_param_tensor"] = std::move(*bound_param_buffer);

  ASSERT_OK_AND_ASSIGN(auto new_context, audio_executor->CreateNewContext());
  auto* new_streaming_context =
      static_cast<AudioStreamingContext*>(new_context.get());
  EXPECT_FALSE(
      new_streaming_context->state_buffers().contains("lora_param_tensor"));
  ASSERT_OK(audio_executor->RestoreContext(std::move(new_context)));
  ASSERT_OK(audio_executor->Reset());

  ASSERT_OK_AND_ASSIGN(
      auto after_reset_param_data,
      GetDataAsVector<float>(input_buffers_map.at("lora_param_tensor")));
  EXPECT_THAT(after_reset_param_data, ::testing::Each(16.0f));
}

TEST_F(AudioLiteRtCompiledModelExecutorTest,
       StreamingEncoder_BufferedSpectrogramSlicesToValidFramesOnly) {
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(*env_,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioStreamingModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU,
                          /*audio_buffering_enabled=*/true));

  // Pass a 10-frame tensor where only the first 4 frames are marked valid in
  // the mask (total_frames = 4 < window_size = 10). Only 4 frames should be
  // buffered.
  std::vector<float> spec_data(10 * kSpectrogramFrequencySlots, 1.0f);
  std::vector<uint8_t> mask_data = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  ASSERT_OK_AND_ASSIGN(
      auto spec_buffer,
      CreateTensorBuffer<const float>(
          spec_data,
          RankedTensorType(
              GetElementType<float>(),
              Layout(Dimensions({1, 10, kSpectrogramFrequencySlots})))));
  ASSERT_OK_AND_ASSIGN(
      auto mask_buffer,
      CreateTensorBuffer<const uint8_t>(
          mask_data, RankedTensorType(GetElementType<uint8_t>(),
                                      Layout(Dimensions({1, 10})))));
  ASSERT_OK_AND_ASSIGN(auto encode_data,
                       audio_executor->Encode(spec_buffer, mask_buffer));
  EXPECT_EQ(encode_data.GetValidTokens(), 0);

  ASSERT_OK_AND_ASSIGN(auto cloned_context, audio_executor->CloneContext());
  auto* streaming_context =
      static_cast<AudioStreamingContext*>(cloned_context.get());
  EXPECT_EQ(streaming_context->buffered_spectrogram().size(),
            4 * kSpectrogramFrequencySlots);
}
#endif  // !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
        // !defined(__NT__) && !defined(_WIN64)

}  // namespace

}  // namespace litert::lm
