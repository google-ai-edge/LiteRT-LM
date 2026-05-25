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

#include "runtime/components/preprocessor/audio_preprocessor_miniaudio.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

constexpr absl::string_view kFrontendModelPath =
    "litert_lm/runtime/components/testdata/frontend.tflite";
constexpr absl::string_view kSlV1FrontendModelPath =
    "litert_lm/runtime/components/testdata/frontend_sl_v1.tflite";
constexpr absl::string_view kDecodedAudioPath =
    "litert_lm/runtime/components/testdata/decoded_audio_samples.bin";
constexpr absl::string_view kAudioPath =
    "litert_lm/runtime/components/testdata/audio_sample.wav";

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
absl::StatusOr<std::vector<T>> GetDataAsVector(
    const litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  std::vector<T> data(elements);
  LITERT_RETURN_IF_ERROR(const_cast<litert::TensorBuffer&>(tensor_buffer)
                             .Read<T>(absl::MakeSpan(data)));
  return data;
}

absl::StatusOr<std::string> GetContents(const std::string& path) {
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return absl::InternalError(absl::StrCat("Could not open file: ", path));
  }

  std::string content;
  content.assign((std::istreambuf_iterator<char>(input_stream)),
                 (std::istreambuf_iterator<char>()));
  return std::move(content);
}

absl::StatusOr<std::vector<float>> GetDecodedAudioData() {
  ASSIGN_OR_RETURN(
      auto decoded_audio_data,
      GetContents(absl::StrCat(::testing::SrcDir(), "/", kDecodedAudioPath)));
  std::vector<float> decoded_audio_vector(
      reinterpret_cast<const float*>(decoded_audio_data.data()),
      reinterpret_cast<const float*>(decoded_audio_data.data() +
                                     decoded_audio_data.size()));
  return decoded_audio_vector;
}

absl::StatusOr<std::string> GetRawAudioData() {
  return GetContents(absl::StrCat(::testing::SrcDir(), "/", kAudioPath));
}

class FrontendModelWrapper {
 public:
  static constexpr int kUsmInputTensorLength = 523426;
  static constexpr int kSlV1InputTensorLength = 131200;
  static absl::StatusOr<std::unique_ptr<FrontendModelWrapper>> Create(
      absl::string_view model_path, int input_tensor_length) {
    LITERT_ASSIGN_OR_RETURN(auto env, litert::Environment::Create({}));

    LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);

    LITERT_ASSIGN_OR_RETURN(
        auto compiled_model,
        litert::CompiledModel::Create(
            env, absl::StrCat(::testing::SrcDir(), "/", model_path), options));

    auto wrapper =
        std::unique_ptr<FrontendModelWrapper>(new FrontendModelWrapper(
            input_tensor_length, std::move(env), std::move(compiled_model)));
    LITERT_RETURN_IF_ERROR(wrapper->InitializeBuffers());
    return wrapper;
  }

  absl::Status Run(const std::vector<float>& audio_data,
                   std::vector<float>* output_spectrogram,
                   std::vector<uint8_t>* output_mask) {
    if (input_buffers_.empty()) {
      return absl::FailedPreconditionError("Model not initialized.");
    }

    // Data in memory needs to be continuous, but the bool type of std library
    // vector is not guaranteed to be continuous for memory. So here we use a
    // bool* to create a continuous memory buffer. This prevent the UBSan check
    // error. See go/ubsan.
    input_buffers_[0].Clear();
    input_buffers_[1].Clear();
    bool* mask_data_ptr = new bool[input_tensor_length_];
    for (int i = 0; i < input_tensor_length_; ++i) {
      if (i < audio_data.size()) {
        mask_data_ptr[i] = true;
      } else {
        mask_data_ptr[i] = false;
      }
    }
    LITERT_RETURN_IF_ERROR(input_buffers_[0].Write(
        absl::MakeConstSpan(mask_data_ptr, input_tensor_length_)));
    delete[] mask_data_ptr;
    LITERT_RETURN_IF_ERROR(input_buffers_[1].Write(absl::MakeSpan(audio_data)));

    compiled_model_.Run(input_buffers_, output_buffers_);
    LITERT_ASSIGN_OR_RETURN(*output_mask,
                            GetDataAsVector<uint8_t>(output_buffers_[0]));
    LITERT_ASSIGN_OR_RETURN(*output_spectrogram,
                            GetDataAsVector<float>(output_buffers_[1]));
    return absl::OkStatus();
  }

 private:
  FrontendModelWrapper(int input_tensor_length, Environment env,
                       CompiledModel compiled_model)
      : input_tensor_length_(input_tensor_length),
        env_(std::move(env)),
        compiled_model_(std::move(compiled_model)) {}

  absl::Status InitializeBuffers() {
    LITERT_ASSIGN_OR_RETURN(auto signatures, compiled_model_.GetSignatures());
    if (signatures.size() != 1) {
      return absl::InvalidArgumentError(
          "Model must have exactly one signature.");
    }

    LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_.CreateInputBuffers(
                                                /*signature_index=*/0));

    LITERT_ASSIGN_OR_RETURN(output_buffers_,
                            compiled_model_.CreateOutputBuffers(
                                /*signature_index=*/0));
    if (output_buffers_.empty()) {
      return absl::InvalidArgumentError("Model must have at least one output.");
    }

    return absl::OkStatus();
  }

  int input_tensor_length_;
  Environment env_;
  litert::CompiledModel compiled_model_;
  std::vector<litert::TensorBuffer> input_buffers_;
  std::vector<litert::TensorBuffer> output_buffers_;
};

// TODO: b/441514829 - Enable the tests on Windows once the bug is fixed.
#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
    !defined(__NT__) && !defined(_WIN64)
TEST(AudioPreprocessorMiniAudioTest, DecodeAudio) {
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, /*num_channels=*/1, /*sample_rate_hz=*/16000,
      pcm_frames));
  ASSERT_OK_AND_ASSIGN(auto decoded_audio_data, GetDecodedAudioData());
  EXPECT_EQ(pcm_frames.size(), decoded_audio_data.size());
  for (int i = 0; i < pcm_frames.size(); ++i) {
    EXPECT_NEAR(pcm_frames[i], decoded_audio_data[i], 1e-6);
  }
}

TEST(AudioPreprocessorMiniAudioTest, UsmPreprocessing) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(
      auto frontend_model,
      FrontendModelWrapper::Create(
          kFrontendModelPath, FrontendModelWrapper::kUsmInputTensorLength));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  ASSERT_OK(frontend_model->Run(pcm_frames, &frontend_mel_spectrogram,
                                &frontend_mask));
  int true_count = 0;
  for (int i = 0; i < frontend_mask.size(); ++i) {
    if (frontend_mask[i] == 1) {
      true_count++;
    }
  }
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));

  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }
}

TEST(AudioPreprocessorMiniAudioTest, SlV1Preprocessing) {
  AudioPreprocessorConfig config = AudioPreprocessorConfig::Create(
      /* sample_rate_hz= */ 16000,
      /* num_channels= */ 1,
      /* frame_length= */ 320,
      /* hop_length= */ 160,
      /* fft_length = */ 512,
      /* input_scale = */ 1.0,
      /* pre_emphasis_factor = */ 0.0,
      /* num_mel_bins= */ 128,
      /* mel_low_hz= */ 0.0,
      /* mel_high_hz= */ 8000.0,
      /* mel_floor= */ 1e-3,
      /* normalize_mel= */ false,
      /* add_floor_to_mel_before_log= */ true,
      /* semicausal_padding= */ true,
      /* non_zero_hanning= */ false,
      /* periodic_hanning= */ true,
      /* fft_padding_type= */ AudioPreprocessorConfig::FftPaddingType::kCenter);
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(auto frontend_model,
                       FrontendModelWrapper::Create(
                           kSlV1FrontendModelPath,
                           FrontendModelWrapper::kSlV1InputTensorLength));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  std::vector<float> padded_pcm_frames = pcm_frames;
  // Semicausal padding. The front end model expects the input to be padded
  padded_pcm_frames.insert(padded_pcm_frames.begin(), config.GetHopLength(),
                           0.0f);
  ASSERT_OK(frontend_model->Run(padded_pcm_frames, &frontend_mel_spectrogram,
                                &frontend_mask));
  int true_count = 0;
  for (int i = 0; i < frontend_mask.size(); ++i) {
    if (frontend_mask[i] == 1) {
      true_count++;
    }
  }
  // We drop the last frame for frontend model because in audio preprocessor,
  // the last uncomplete frame is buffered and will not be output.
  true_count -= 1;
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));

  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                2e-3);
  }
}

TEST(AudioPreprocessorMiniAudioTest, UsmPreprocessingWithPcmFrames) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(
      auto frontend_model,
      FrontendModelWrapper::Create(
          kFrontendModelPath, FrontendModelWrapper::kUsmInputTensorLength));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  ASSERT_OK(frontend_model->Run(pcm_frames, &frontend_mel_spectrogram,
                                &frontend_mask));
  int true_count = 0;
  for (int i = 0; i < frontend_mask.size(); ++i) {
    if (frontend_mask[i] == 1) {
      true_count++;
    }
  }
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(pcm_frames)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));

  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }
}

TEST(AudioPreprocessorMiniAudioTest, UsmPreprocessingTwice) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(
      auto frontend_model,
      FrontendModelWrapper::Create(
          kFrontendModelPath, FrontendModelWrapper::kUsmInputTensorLength));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  ASSERT_OK(frontend_model->Run(pcm_frames, &frontend_mel_spectrogram,
                                &frontend_mask));
  int true_count = 0;
  for (int i = 0; i < frontend_mask.size(); ++i) {
    if (frontend_mask[i] == 1) {
      true_count++;
    }
  }
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));

  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }

  // Preprocess the same audio data again without resetting the preprocessor.
  auto result = preprocessor->Preprocess(InputAudio(raw_audio_data));
  EXPECT_THAT(result, ::testing::status::StatusIs(absl::StatusCode::kOk));

  // Preprocess the same audio data again after resetting the preprocessor.
  preprocessor->Reset();
  ASSERT_OK_AND_ASSIGN(preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  ASSERT_OK_AND_ASSIGN(preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));
  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }
}

TEST(AudioPreprocessorMiniAudioTest, UsmPreprocessingCopy) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(
      auto frontend_model,
      FrontendModelWrapper::Create(
          kFrontendModelPath, FrontendModelWrapper::kUsmInputTensorLength));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  ASSERT_OK(frontend_model->Run(pcm_frames, &frontend_mel_spectrogram,
                                &frontend_mask));
  int true_count = 0;
  for (int i = 0; i < frontend_mask.size(); ++i) {
    if (frontend_mask[i] == 1) {
      true_count++;
    }
  }
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());
  std::vector<float> pcm_frames_front_half(
      pcm_frames.begin(), pcm_frames.begin() + pcm_frames.size() / 2);
  std::vector<float> pcm_frames_back_half(
      pcm_frames.begin() + pcm_frames.size() / 2, pcm_frames.end());

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_audio_front_half,
      preprocessor->Preprocess(InputAudio(pcm_frames_front_half)));

  // Copy the preprocessor and reset the original preprocessor. The copy
  // should not be affected by the reset.
  AudioPreprocessorMiniAudio preprocessor_copy = *preprocessor;
  preprocessor->Reset();

  // Preprocess the back half of the audio data with the new preprocessor.
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_audio_back_half,
      preprocessor_copy.Preprocess(InputAudio(pcm_frames_back_half)));

  // Get the preprocessed mel spectrogram from the front and back half and
  // concatenate them.
  ASSERT_OK_AND_ASSIGN(
      auto tensor_front_half,
      preprocessed_audio_front_half.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      auto tensor_back_half,
      preprocessed_audio_back_half.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(auto data_front_half,
                       GetDataAsVector<float>(*tensor_front_half));
  ASSERT_OK_AND_ASSIGN(auto data_back_half,
                       GetDataAsVector<float>(*tensor_back_half));
  std::vector<float> preprocessed_mel_spectrogram(data_front_half.begin(),
                                                  data_front_half.end());
  preprocessed_mel_spectrogram.insert(preprocessed_mel_spectrogram.end(),
                                      data_back_half.begin(),
                                      data_back_half.end());

  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }
}

TEST(AudioPreprocessorMiniAudioTest, SkipMelSpectrogramExtraction) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  config.SetSkipMelSpectrogramExtraction(true);
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());

  auto tensor_type_or = preprocessed_tensor->TensorType();
  ASSERT_TRUE(tensor_type_or.HasValue());
  auto tensor_type = *tensor_type_or;
  EXPECT_EQ(tensor_type.ElementType(), GetElementType<float>());

  auto dims = tensor_type.Layout().Dimensions();
  ASSERT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0], 1);

  const int expected_num_frames =
      1 + (pcm_frames.size() - config.GetFrameLength()) / config.GetHopLength();
  EXPECT_EQ(dims[1], expected_num_frames);
  EXPECT_EQ(dims[2], config.GetFrameLength());

  // Verify target content: the written pcm_frames matches the decoded audio.
  ASSERT_OK_AND_ASSIGN(auto data, GetDataAsVector<float>(*preprocessed_tensor));
  EXPECT_EQ(data.size(), expected_num_frames * config.GetFrameLength());

  const int frame_length = config.GetFrameLength();
  const int hop_length = config.GetHopLength();
  for (int f = 0; f < expected_num_frames; ++f) {
    for (int offset = 0; offset < frame_length; ++offset) {
      int pcm_idx = f * hop_length + offset;
      int data_idx = f * frame_length + offset;
      if (pcm_idx < pcm_frames.size()) {
        EXPECT_NEAR(data[data_idx], pcm_frames[pcm_idx], 1e-6);
      } else {
        EXPECT_NEAR(data[data_idx], 0.0f, 1e-6);
      }
    }
  }
}

TEST(AudioPreprocessorMiniAudioTest, SkipMelSpectrogramExtractionStreaming) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  config.SetSkipMelSpectrogramExtraction(true);
  config.SetBufferLastFrame(true);
  config.SetFrameLength(640);
  config.SetHopLength(640);  // No overlap.

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));

  // Generate toy PCM data: 1000 sequential floats.
  std::vector<float> pcm_data(1000);
  for (int i = 0; i < pcm_data.size(); ++i) {
    pcm_data[i] = static_cast<float>(i);
  }

  // Chunk 1: First 600 floats.
  std::vector<float> chunk1(pcm_data.begin(), pcm_data.begin() + 600);
  auto preprocessed_audio1_or = preprocessor->Preprocess(InputAudio(chunk1));
  EXPECT_FALSE(preprocessed_audio1_or.ok());
  EXPECT_EQ(preprocessed_audio1_or.status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Chunk 2: Next 400 floats (making 1000 total).
  std::vector<float> chunk2(pcm_data.begin() + 600, pcm_data.end());
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio2,
                       preprocessor->Preprocess(InputAudio(chunk2)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_tensor2,
                       preprocessed_audio2.GetPreprocessedAudioTensor());

  // Verify Chunk 2 yields 1 frame (since 600 + 400 = 1000 >= 640).
  auto tensor_type_or2 = preprocessed_tensor2->TensorType();
  ASSERT_TRUE(tensor_type_or2.HasValue());
  {
    auto dims = tensor_type_or2->Layout().Dimensions();
    ASSERT_EQ(dims.size(), 3);
    EXPECT_EQ(dims[0], 1);
    EXPECT_EQ(dims[1], 1);
    EXPECT_EQ(dims[2], 640);
  }

  // Verify the single frame starts with chunk1, then chunk2.
  ASSERT_OK_AND_ASSIGN(auto data,
                       GetDataAsVector<float>(*preprocessed_tensor2));
  EXPECT_EQ(data.size(), 640);
  for (int i = 0; i < 640; ++i) {
    EXPECT_NEAR(data[i], pcm_data[i], 1e-6);
  }
}

TEST(AudioPreprocessorMiniAudioTest,
     SkipMelSpectrogramExtractionWithTrailingZeroPadding) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  config.SetSkipMelSpectrogramExtraction(true);
  config.SetBufferLastFrame(false);
  config.SetFrameLength(640);
  config.SetHopLength(640);  // No overlap.

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));

  // Generate toy PCM data: 1000 sequential floats.
  std::vector<float> pcm_data(1000);
  for (int i = 0; i < pcm_data.size(); ++i) {
    pcm_data[i] = static_cast<float>(i);
  }

  // Chunk 1: First 600 floats.
  std::vector<float> chunk1(pcm_data.begin(), pcm_data.begin() + 600);
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio1,
                       preprocessor->Preprocess(InputAudio(chunk1)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_tensor1,
                       preprocessed_audio1.GetPreprocessedAudioTensor());

  // Verify Chunk 1 yields exactly 1 frame (padded from 600 to 640).
  auto tensor_type_or1 = preprocessed_tensor1->TensorType();
  ASSERT_TRUE(tensor_type_or1.HasValue());
  {
    auto dims = tensor_type_or1->Layout().Dimensions();
    ASSERT_EQ(dims.size(), 3);
    EXPECT_EQ(dims[0], 1);
    EXPECT_EQ(dims[1], 1);
    EXPECT_EQ(dims[2], 640);
  }

  ASSERT_OK_AND_ASSIGN(auto data1,
                       GetDataAsVector<float>(*preprocessed_tensor1));
  EXPECT_EQ(data1.size(), 640);
  for (int i = 0; i < 600; ++i) {
    EXPECT_NEAR(data1[i], chunk1[i], 1e-6);
  }
  for (int i = 600; i < 640; ++i) {
    EXPECT_NEAR(data1[i], 0.0f, 1e-6);
  }

  // Chunk 2: Next 400 floats.
  std::vector<float> chunk2(pcm_data.begin() + 600, pcm_data.end());
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio2,
                       preprocessor->Preprocess(InputAudio(chunk2)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_tensor2,
                       preprocessed_audio2.GetPreprocessedAudioTensor());

  // Verify Chunk 2 yields exactly 1 frame (padded from 400 to 640).
  auto tensor_type_or2 = preprocessed_tensor2->TensorType();
  ASSERT_TRUE(tensor_type_or2.HasValue());
  {
    auto dims = tensor_type_or2->Layout().Dimensions();
    ASSERT_EQ(dims.size(), 3);
    EXPECT_EQ(dims[0], 1);
    EXPECT_EQ(dims[1], 1);
    EXPECT_EQ(dims[2], 640);
  }

  ASSERT_OK_AND_ASSIGN(auto data2,
                       GetDataAsVector<float>(*preprocessed_tensor2));
  EXPECT_EQ(data2.size(), 640);
  for (int i = 0; i < 400; ++i) {
    EXPECT_NEAR(data2[i], chunk2[i], 1e-6);
  }
  for (int i = 400; i < 640; ++i) {
    EXPECT_NEAR(data2[i], 0.0f, 1e-6);
  }
}

#endif  // !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) &&
        // !defined(__NT__) && !defined(_WIN64)

}  // namespace
}  // namespace litert::lm
