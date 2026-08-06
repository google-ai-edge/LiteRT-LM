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

#include "support/preprocessor/audio_preprocessor_pffft.h"

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
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "support/preprocessor/audio_preprocessor.h"
#include "support/util/io_types.h"

namespace litert::support {
namespace {

constexpr absl::string_view kFrontendModelPath =
    "litert_lm/support/"
    "preprocessor/testdata/frontend.tflite";
constexpr absl::string_view kSlV1FrontendModelPath =
    "litert_lm/support/"
    "preprocessor/testdata/frontend_sl_v1.tflite";
constexpr absl::string_view kDecodedAudioPath =
    "litert_lm/support/"
    "preprocessor/testdata/decoded_audio_samples.bin";

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
  LITERT_ASSIGN_OR_RETURN(
      auto decoded_audio_data,
      GetContents(absl::StrCat(::testing::SrcDir(), "/", kDecodedAudioPath)));
  std::vector<float> decoded_audio_vector(
      reinterpret_cast<const float*>(decoded_audio_data.data()),
      reinterpret_cast<const float*>(decoded_audio_data.data() +
                                     decoded_audio_data.size()));
  return decoded_audio_vector;
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

#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
    !defined(__NT__) && !defined(_WIN64)

TEST(AudioPreprocessorPffftTest, VerifyPcmFramesToSpectrogram) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));

  // Create a synthesized signal with energy at DC (0 Hz), mid frequency, and
  // Nyquist (fs / 2) frequency.
  std::vector<float> pcm_frames(config.GetFftLength());
  for (int i = 0; i < pcm_frames.size(); ++i) {
    pcm_frames[i] =
        (1.0f + ((i % 4 == 0) ? 0.5f : ((i % 4 == 2) ? -0.5f : 0.0f)) +
         0.25f * ((i % 2 == 0) ? 1.0f : -1.0f)) /
        163840.0f;
  }

  std::vector<float> spectrograms;
  ASSERT_OK(preprocessor->PcmFramesToSpectrogramForTesting(pcm_frames,
                                                           spectrograms));

  const int fft_bins = config.GetFftBins();
  ASSERT_GE(spectrograms.size(), fft_bins);
  ASSERT_EQ(spectrograms.size() % fft_bins, 0);

  // Assert exact values for DC (bin 0) and Nyquist (bin fft_bins - 1) to ensure
  // mutations (such as offsetting or zeroing bins in ComputeSpectrogram) fail
  // the test.
  EXPECT_NEAR(spectrograms[0], 2.359288f, 1e-4);
  EXPECT_NEAR(spectrograms[fft_bins / 4], 0.0f, 1e-4);
  EXPECT_NEAR(spectrograms[fft_bins - 1], 635.846558f, 1e-4);
}

TEST(AudioPreprocessorPffftTest, UsmPreprocessingWithPcmFrames) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto pcm_frames, GetDecodedAudioData());

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

  // Create PFFFT preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));
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

TEST(AudioPreprocessorPffftTest, SlV1Preprocessing) {
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
  ASSERT_OK_AND_ASSIGN(auto pcm_frames, GetDecodedAudioData());

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(auto frontend_model,
                       FrontendModelWrapper::Create(
                           kSlV1FrontendModelPath,
                           FrontendModelWrapper::kSlV1InputTensorLength));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  std::vector<float> padded_pcm_frames = pcm_frames;
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
  true_count -= 1;
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());

  // Create PFFFT preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));
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
                2e-3);
  }
}

TEST(AudioPreprocessorPffftTest, InvalidFrameLength) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  config.SetFrameLength(0);
  EXPECT_THAT(AudioPreprocessorPffft::Create(config),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AudioPreprocessorPffftTest, PreprocessWithTensorBufferInput) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));

  litert::RankedTensorType tensor_type(
      litert::GetElementType<float>(),
      litert::Layout(litert::Dimensions({1, 10, 128})));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tensor_buffer,
      litert::TensorBuffer::CreateManagedHostMemory(
          tensor_type, 1 * 10 * 128 * sizeof(float)));
  InputAudio test_input_audio(std::move(input_tensor_buffer));

  ASSERT_OK_AND_ASSIGN(auto result,
                       preprocessor->Preprocess(test_input_audio));
  EXPECT_TRUE(result.IsTensorBuffer());
}

TEST(AudioPreprocessorPffftTest, PreprocessWithRawBytesFails) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));

  std::string dummy_audio_data = "\x01\x02\x03\x04";
  InputAudio test_input_audio(dummy_audio_data);
  auto result = preprocessor->Preprocess(test_input_audio);
  EXPECT_THAT(result,
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AudioPreprocessorPffftTest, PreprocessWithSkipMelSpectrogramExtraction) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  config.SetSkipMelSpectrogramExtraction(true);
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));
  ASSERT_OK_AND_ASSIGN(auto pcm_frames, GetDecodedAudioData());

  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(pcm_frames)));
  EXPECT_TRUE(preprocessed_audio.IsTensorBuffer());
  ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(auto output_data,
                       GetDataAsVector<float>(*tensor_buffer));
  EXPECT_FALSE(output_data.empty());
}

TEST(AudioPreprocessorPffftTest, SkipMelSpectrogramNotEnoughSamples) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  config.SetSemicausalPadding(false);
  config.SetSkipMelSpectrogramExtraction(true);
  config.SetBufferLastFrame(true);
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));

  std::vector<float> short_pcm(10, 0.0f);
  auto result = preprocessor->Preprocess(InputAudio(short_pcm));
  EXPECT_THAT(result,
              testing::status::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(AudioPreprocessorPffftTest, CopyAndResetOperations) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));
  ASSERT_OK_AND_ASSIGN(auto pcm_frames, GetDecodedAudioData());

  ASSERT_OK(preprocessor->Preprocess(InputAudio(pcm_frames)));
  preprocessor->Reset();

  AudioPreprocessorPffft copied_preprocessor(*preprocessor);
  AudioPreprocessorPffft assigned_preprocessor = copied_preprocessor;
  assigned_preprocessor = *preprocessor;

  ASSERT_OK(assigned_preprocessor.Preprocess(InputAudio(pcm_frames)));
}

#endif  // !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) &&
        // !defined(__NT__) && !defined(_WIN64)

}  // namespace
}  // namespace litert::support
