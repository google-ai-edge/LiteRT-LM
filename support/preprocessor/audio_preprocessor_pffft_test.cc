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

#include <cstddef>
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
constexpr absl::string_view kAudioPath =
    "litert_lm/support/"
    "preprocessor/testdata/audio_sample.wav";

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

absl::StatusOr<std::string> GetRawAudioData() {
  return GetContents(absl::StrCat(::testing::SrcDir(), "/", kAudioPath));
}

std::string CreateWav(uint16_t audio_format, uint16_t num_channels,
                      uint32_t sample_rate, uint16_t bits_per_sample,
                      absl::string_view pcm_payload) {
  std::string wav;
  uint32_t fmt_size = 16;
  uint32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
  uint16_t block_align = num_channels * (bits_per_sample / 8);
  uint32_t data_size = pcm_payload.size();
  uint32_t riff_size = 4 + (8 + fmt_size) + (8 + data_size);

  wav.append("RIFF", 4);
  wav.append(reinterpret_cast<const char*>(&riff_size), 4);
  wav.append("WAVE", 4);

  wav.append("fmt ", 4);
  wav.append(reinterpret_cast<const char*>(&fmt_size), 4);
  wav.append(reinterpret_cast<const char*>(&audio_format), 2);
  wav.append(reinterpret_cast<const char*>(&num_channels), 2);
  wav.append(reinterpret_cast<const char*>(&sample_rate), 4);
  wav.append(reinterpret_cast<const char*>(&byte_rate), 4);
  wav.append(reinterpret_cast<const char*>(&block_align), 2);
  wav.append(reinterpret_cast<const char*>(&bits_per_sample), 2);

  wav.append("data", 4);
  wav.append(reinterpret_cast<const char*>(&data_size), 4);
  wav.append(pcm_payload.data(), pcm_payload.size());

  return wav;
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
  ASSERT_OK(
      preprocessor->PcmFramesToSpectrogramForTesting(pcm_frames, spectrograms));

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
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_tensor_buffer,
                              litert::TensorBuffer::CreateManagedHostMemory(
                                  tensor_type, 1 * 10 * 128 * sizeof(float)));
  InputAudio test_input_audio(std::move(input_tensor_buffer));

  ASSERT_OK_AND_ASSIGN(auto result, preprocessor->Preprocess(test_input_audio));
  EXPECT_TRUE(result.IsTensorBuffer());
}

TEST(AudioPreprocessorPffftTest, DecodeAudio) {
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorPffft::DecodeAudio(
      raw_audio_data, /*num_channels=*/1, /*sample_rate_hz=*/16000,
      pcm_frames));
  ASSERT_OK_AND_ASSIGN(auto decoded_audio_data, GetDecodedAudioData());
  EXPECT_EQ(pcm_frames.size(), decoded_audio_data.size());
  for (size_t i = 0; i < pcm_frames.size(); ++i) {
    EXPECT_NEAR(pcm_frames[i], decoded_audio_data[i], 1e-6);
  }
}

TEST(AudioPreprocessorPffftTest, PreprocessWithWavBytes) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  ASSERT_OK_AND_ASSIGN(auto pcm_frames, GetDecodedAudioData());

  ASSERT_OK_AND_ASSIGN(auto wav_result,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  preprocessor->Reset();
  ASSERT_OK_AND_ASSIGN(auto pcm_result,
                       preprocessor->Preprocess(InputAudio(pcm_frames)));

  ASSERT_OK_AND_ASSIGN(auto wav_tensor,
                       wav_result.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(auto pcm_tensor,
                       pcm_result.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(auto wav_vec, GetDataAsVector<float>(*wav_tensor));
  ASSERT_OK_AND_ASSIGN(auto pcm_vec, GetDataAsVector<float>(*pcm_tensor));

  ASSERT_EQ(wav_vec.size(), pcm_vec.size());
  for (size_t i = 0; i < wav_vec.size(); ++i) {
    EXPECT_NEAR(wav_vec[i], pcm_vec[i], 1e-5);
  }
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

TEST(AudioPreprocessorPffftTest, DecodeAudioStereo16BitPcm) {
  std::vector<int16_t> stereo_samples = {
      16384,  16384,   // left, right -> (0.5 + 0.5) / 2 = 0.5
      -16384, 16384,   // left, right -> (-0.5 + 0.5) / 2 = 0.0
      0,      0,       // 0.0
      32767,  -32768,  // (32767 - 32768) / (2 * 32768) ~= -0.00001525
  };
  std::string pcm_payload(reinterpret_cast<const char*>(stereo_samples.data()),
                          stereo_samples.size() * sizeof(int16_t));
  std::string wav =
      CreateWav(/*audio_format=*/1, /*num_channels=*/2,
                /*sample_rate=*/16000, /*bits_per_sample=*/16, pcm_payload);

  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorPffft::DecodeAudio(
      wav, /*num_channels=*/1, /*sample_rate_hz=*/16000, pcm_frames));
  ASSERT_EQ(pcm_frames.size(), 4);
  EXPECT_NEAR(pcm_frames[0], 0.5f, 1e-4);
  EXPECT_NEAR(pcm_frames[1], 0.0f, 1e-4);
  EXPECT_NEAR(pcm_frames[2], 0.0f, 1e-4);
  EXPECT_NEAR(pcm_frames[3], -1.0f / 65536.0f, 1e-4);
}

TEST(AudioPreprocessorPffftTest, DecodeAudioMono32BitFloat) {
  std::vector<float> mono_samples = {-0.5f, 0.0f, 0.25f, 0.75f};
  std::string pcm_payload(reinterpret_cast<const char*>(mono_samples.data()),
                          mono_samples.size() * sizeof(float));
  std::string wav =
      CreateWav(/*audio_format=*/3, /*num_channels=*/1,
                /*sample_rate=*/16000, /*bits_per_sample=*/32, pcm_payload);

  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorPffft::DecodeAudio(
      wav, /*num_channels=*/1, /*sample_rate_hz=*/16000, pcm_frames));
  ASSERT_EQ(pcm_frames.size(), mono_samples.size());
  for (size_t i = 0; i < pcm_frames.size(); ++i) {
    EXPECT_FLOAT_EQ(pcm_frames[i], mono_samples[i]);
  }
}

TEST(AudioPreprocessorPffftTest, DecodeAudioStereo32BitFloat) {
  std::vector<float> stereo_samples = {
      -0.5f, 0.5f,  // downmix -> 0.0
      0.2f,  0.4f,  // downmix -> 0.3
      1.0f,  0.0f,  // downmix -> 0.5
  };
  std::string pcm_payload(reinterpret_cast<const char*>(stereo_samples.data()),
                          stereo_samples.size() * sizeof(float));
  std::string wav =
      CreateWav(/*audio_format=*/3, /*num_channels=*/2,
                /*sample_rate=*/16000, /*bits_per_sample=*/32, pcm_payload);

  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorPffft::DecodeAudio(
      wav, /*num_channels=*/1, /*sample_rate_hz=*/16000, pcm_frames));
  ASSERT_EQ(pcm_frames.size(), 3);
  EXPECT_NEAR(pcm_frames[0], 0.0f, 1e-6);
  EXPECT_NEAR(pcm_frames[1], 0.3f, 1e-6);
  EXPECT_NEAR(pcm_frames[2], 0.5f, 1e-6);
}

TEST(AudioPreprocessorPffftTest, PreprocessWithStereoAndFloatWavBytes) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorPffft::Create(config));

  // 1024 stereo 16-bit frames (2048 int16 samples).
  std::vector<int16_t> stereo_16(2048, 16384);
  std::string stereo_16_payload(reinterpret_cast<const char*>(stereo_16.data()),
                                stereo_16.size() * sizeof(int16_t));
  std::string stereo_16_wav = CreateWav(
      /*audio_format=*/1, /*num_channels=*/2, /*sample_rate=*/16000,
      /*bits_per_sample=*/16, stereo_16_payload);

  ASSERT_OK_AND_ASSIGN(auto result1,
                       preprocessor->Preprocess(InputAudio(stereo_16_wav)));
  EXPECT_TRUE(result1.IsTensorBuffer());
  preprocessor->Reset();

  // 1024 mono 32-bit float frames.
  std::vector<float> mono_float(1024, 0.5f);
  std::string mono_float_payload(
      reinterpret_cast<const char*>(mono_float.data()),
      mono_float.size() * sizeof(float));
  std::string mono_float_wav = CreateWav(
      /*audio_format=*/3, /*num_channels=*/1, /*sample_rate=*/16000,
      /*bits_per_sample=*/32, mono_float_payload);

  ASSERT_OK_AND_ASSIGN(auto result2,
                       preprocessor->Preprocess(InputAudio(mono_float_wav)));
  EXPECT_TRUE(result2.IsTensorBuffer());
  preprocessor->Reset();

  // 1024 stereo 32-bit float frames (2048 float samples).
  std::vector<float> stereo_float(2048, 0.5f);
  std::string stereo_float_payload(
      reinterpret_cast<const char*>(stereo_float.data()),
      stereo_float.size() * sizeof(float));
  std::string stereo_float_wav = CreateWav(
      /*audio_format=*/3, /*num_channels=*/2, /*sample_rate=*/16000,
      /*bits_per_sample=*/32, stereo_float_payload);

  ASSERT_OK_AND_ASSIGN(auto result3,
                       preprocessor->Preprocess(InputAudio(stereo_float_wav)));
  EXPECT_TRUE(result3.IsTensorBuffer());
}

TEST(AudioPreprocessorPffftTest, DecodeAudioValidationErrors) {
  std::vector<float> pcm_frames;

  // Invalid target num_channels != 1.
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio("dummy", /*num_channels=*/2,
                                                  16000, pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Only mono audio is supported")));

  // Data too short (< 12 bytes).
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(
                  "RIFF1234", /*num_channels=*/1, 16000, pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("too small to be a valid WAV file")));

  // Invalid RIFF header.
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(
                  "NOT_RIFF_123456", /*num_channels=*/1, 16000, pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("not a valid RIFF/WAVE file")));

  // Invalid WAVE header.
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(
                  "RIFF1234NOTW", /*num_channels=*/1, 16000, pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("not a valid RIFF/WAVE file")));

  // Corrupted non-data chunk size.
  std::string corrupted_chunk;
  corrupted_chunk.append("RIFF", 4);
  uint32_t riff_size = 32;
  corrupted_chunk.append(reinterpret_cast<const char*>(&riff_size), 4);
  corrupted_chunk.append("WAVE", 4);
  corrupted_chunk.append("JUNK", 4);
  uint32_t junk_size = 65535;
  corrupted_chunk.append(reinterpret_cast<const char*>(&junk_size), 4);
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(
                  corrupted_chunk, /*num_channels=*/1, 16000, pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Corrupted WAV chunk size")));

  std::vector<int16_t> dummy_pcm = {0, 0};
  std::string dummy_payload(reinterpret_cast<const char*>(dummy_pcm.data()),
                            dummy_pcm.size() * sizeof(int16_t));

  // fmt chunk size < 16.
  std::string small_fmt_wav;
  small_fmt_wav.append("RIFF", 4);
  uint32_t small_riff = 32;
  small_fmt_wav.append(reinterpret_cast<const char*>(&small_riff), 4);
  small_fmt_wav.append("WAVE", 4);
  small_fmt_wav.append("fmt ", 4);
  uint32_t small_fmt_size = 8;
  small_fmt_wav.append(reinterpret_cast<const char*>(&small_fmt_size), 4);
  const uint8_t small_fmt_data[] = {1, 0, 1, 0, 0x80, 0x3e, 0, 0};
  small_fmt_wav.append(reinterpret_cast<const char*>(small_fmt_data),
                       sizeof(small_fmt_data));
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(
                  small_fmt_wav, /*num_channels=*/1, 16000, pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Invalid WAV fmt chunk size")));

  // Unsupported audio format (e.g. format 7 = mu-law).
  std::string unsupported_format_wav = CreateWav(
      /*audio_format=*/7, /*num_channels=*/1, 16000, 16, dummy_payload);
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(unsupported_format_wav,
                                                  /*num_channels=*/1, 16000,
                                                  pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Unsupported WAV audio format")));

  // Unsupported sample rate.
  std::string unsupported_sample_rate_wav = CreateWav(
      /*audio_format=*/1, /*num_channels=*/1, /*sample_rate=*/44100, 16,
      dummy_payload);
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(unsupported_sample_rate_wav,
                                                  /*num_channels=*/1, 16000,
                                                  pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Unsupported WAV sample rate")));

  // Unsupported channel count (e.g. 4 channels).
  std::string unsupported_channels_wav = CreateWav(
      /*audio_format=*/1, /*num_channels=*/4, 16000, 16, dummy_payload);
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(unsupported_channels_wav,
                                                  /*num_channels=*/1, 16000,
                                                  pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Unsupported WAV channel count")));

  // Unsupported bit depth for integer PCM (e.g. 8-bit).
  std::string unsupported_pcm_bits_wav = CreateWav(
      /*audio_format=*/1, /*num_channels=*/1, 16000, /*bits_per_sample=*/8,
      dummy_payload);
  EXPECT_THAT(
      AudioPreprocessorPffft::DecodeAudio(unsupported_pcm_bits_wav,
                                          /*num_channels=*/1, 16000,
                                          pcm_frames),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Unsupported WAV bit depth for integer PCM")));

  // Unsupported bit depth for IEEE float (e.g. 64-bit).
  std::string unsupported_float_bits_wav = CreateWav(
      /*audio_format=*/3, /*num_channels=*/1, 16000, /*bits_per_sample=*/64,
      dummy_payload);
  EXPECT_THAT(
      AudioPreprocessorPffft::DecodeAudio(unsupported_float_bits_wav,
                                          /*num_channels=*/1, 16000,
                                          pcm_frames),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Unsupported WAV bit depth for IEEE float")));

  // Missing data chunk (only fmt chunk present).
  std::string missing_data_wav;
  missing_data_wav.append("RIFF", 4);
  uint32_t missing_riff = 28;
  missing_data_wav.append(reinterpret_cast<const char*>(&missing_riff), 4);
  missing_data_wav.append("WAVE", 4);
  missing_data_wav.append("fmt ", 4);
  uint32_t missing_fmt_size = 16;
  missing_data_wav.append(reinterpret_cast<const char*>(&missing_fmt_size), 4);
  uint16_t audio_fmt = 1, channels = 1, align = 2, bits = 16;
  uint32_t rate = 16000, byte_rate = 32000;
  missing_data_wav.append(reinterpret_cast<const char*>(&audio_fmt), 2);
  missing_data_wav.append(reinterpret_cast<const char*>(&channels), 2);
  missing_data_wav.append(reinterpret_cast<const char*>(&rate), 4);
  missing_data_wav.append(reinterpret_cast<const char*>(&byte_rate), 4);
  missing_data_wav.append(reinterpret_cast<const char*>(&align), 2);
  missing_data_wav.append(reinterpret_cast<const char*>(&bits), 2);
  EXPECT_THAT(AudioPreprocessorPffft::DecodeAudio(
                  missing_data_wav, /*num_channels=*/1, 16000, pcm_frames),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Missing fmt or data chunk")));
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
