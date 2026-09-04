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
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_utils.h"
#include "support/preprocessor/mel_filterbank.h"
#include "support/util/io_types.h"
#include "third_party/pffft/src/pffft.h"

namespace litert::support {

namespace {

// Use Pffft to compute `spectrograms` from `windowed_signals`.
absl::Status ComputeSpectrogram(
    const std::vector<std::vector<float>>& windowed_signals, int fft_length,
    int fft_bins, std::vector<float>& spectrograms) {
  if (fft_length / 2 + 1 != fft_bins) {
    return absl::InvalidArgumentError(absl::StrCat(
        "fft_bins (", fft_bins, ") must equal fft_length (", fft_length,
        ") / 2 + 1, but do not."));
  }

  PFFFT_Setup* setup = pffft_new_setup(fft_length, PFFFT_REAL);
  if (setup == nullptr) {
    return absl::InternalError("Failed to create PFFFT setup.");
  }

  std::vector<float> data(fft_length);
  std::vector<float> work(fft_length);

  for (const auto& current_window : windowed_signals) {
    pffft_transform_ordered(setup, current_window.data(), data.data(),
                            work.data(), PFFFT_FORWARD);

    // The DC and half-frequency bins are stashed together as the real and
    // imaginary values in the first output bin.
    float first_bin = data[0];
    spectrograms.push_back(first_bin * first_bin);
    for (size_t i = 2; i < data.size(); i += 2) {
      float real = data[i];
      float imag = data[i + 1];
      spectrograms.push_back(real * real + imag * imag);
    }
    float last_bin = data[1];
    spectrograms.push_back(last_bin * last_bin);
  }

  pffft_destroy_setup(setup);
  return absl::OkStatus();
}

inline uint32_t ReadLittleEndianUint32(const uint8_t* ptr) {
  return static_cast<uint32_t>(ptr[0]) |
         (static_cast<uint32_t>(ptr[1]) << 8) |
         (static_cast<uint32_t>(ptr[2]) << 16) |
         (static_cast<uint32_t>(ptr[3]) << 24);
}

inline float ReadLittleEndianFloat(const uint8_t* ptr) {
  return absl::bit_cast<float>(ReadLittleEndianUint32(ptr));
}

}  // namespace

AudioPreprocessorPffft::~AudioPreprocessorPffft() = default;

absl::StatusOr<absl_nonnull std::unique_ptr<AudioPreprocessorPffft>>
AudioPreprocessorPffft::Create(const AudioPreprocessorConfig& config) {
  if (config.GetFrameLength() <= 0) {
    return absl::InvalidArgumentError("Frame length must be positive.");
  }
  auto mel_filterbank = std::make_unique<MelFilterbank>();
  LITERT_RETURN_IF_ERROR(mel_filterbank->Initialize(
      config.GetFftBins(), config.GetSampleRateHz(), config.GetNumMelBins(),
      config.GetMelLowHz(), config.GetMelHighHz()));
  return absl::WrapUnique(
      new AudioPreprocessorPffft(config, std::move(mel_filterbank)));
}

absl::Status AudioPreprocessorPffft::PcmFramesToSpectrogram(
    absl::Span<const float> pcm_frames, std::vector<float>& spectrograms) {
  LITERT_ASSIGN_OR_RETURN(auto windowed_signals,
                        GetWindowedSignalsForFft(config_, pcm_frames,
                                                 input_queue_,
                                                 samples_to_next_step_));
  return ComputeSpectrogram(windowed_signals, config_.GetFftLength(),
                            config_.GetFftBins(), spectrograms);
}

absl::Status AudioPreprocessorPffft::DecodeAudio(
    absl::string_view audio_bytes, int num_channels, int sample_rate_hz,
    std::vector<float>& pcm_frames) {
  if (num_channels != 1) {
    return absl::InvalidArgumentError("Only mono audio is supported.");
  }
  if (audio_bytes.size() < 12) {
    return absl::InvalidArgumentError(
        "Audio data is too small to be a valid WAV file.");
  }

  // RIFF header: "RIFF" <4-byte file size> "WAVE"
  if (audio_bytes.substr(0, 4) != "RIFF" ||
      audio_bytes.substr(8, 4) != "WAVE") {
    return absl::InvalidArgumentError(
        "Audio data is not a valid RIFF/WAVE file.");
  }

  size_t offset = 12;
  bool fmt_found = false;
  bool data_found = false;
  uint16_t audio_format = 0;
  uint16_t wav_channels = 0;
  uint32_t wav_sample_rate = 0;
  uint16_t bits_per_sample = 0;
  size_t data_offset = 0;
  size_t data_bytes = 0;

  while (offset + 8 <= audio_bytes.size()) {
    absl::string_view chunk_id = audio_bytes.substr(offset, 4);
    const uint8_t* size_ptr =
        reinterpret_cast<const uint8_t*>(audio_bytes.data() + offset + 4);
    uint32_t chunk_size = ReadLittleEndianUint32(size_ptr);
    offset += 8;

    if (chunk_size > audio_bytes.size() - offset) {
      if (chunk_id == "data") {
        chunk_size = static_cast<uint32_t>(audio_bytes.size() - offset);
      } else {
        return absl::InvalidArgumentError("Corrupted WAV chunk size.");
      }
    }

    if (chunk_id == "fmt ") {
      if (chunk_size < 16) {
        return absl::InvalidArgumentError("Invalid WAV fmt chunk size.");
      }
      const uint8_t* fmt_data =
          reinterpret_cast<const uint8_t*>(audio_bytes.data() + offset);
      audio_format = static_cast<uint16_t>(fmt_data[0] | (fmt_data[1] << 8));
      wav_channels = static_cast<uint16_t>(fmt_data[2] | (fmt_data[3] << 8));
      wav_sample_rate = ReadLittleEndianUint32(fmt_data + 4);
      bits_per_sample =
          static_cast<uint16_t>(fmt_data[14] | (fmt_data[15] << 8));

      if (audio_format != 1 && audio_format != 3) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported WAV audio format: ", audio_format,
                         ". Only PCM (1) and IEEE float (3) are supported."));
      }
      if (wav_sample_rate != static_cast<uint32_t>(sample_rate_hz)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported WAV sample rate: ", wav_sample_rate,
                         ", expected: ", sample_rate_hz));
      }
      if (wav_channels != 1 && wav_channels != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported WAV channel count: ", wav_channels,
                         ". Only mono (1) and stereo (2) are supported."));
      }
      if (audio_format == 1 && bits_per_sample != 16) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported WAV bit depth for integer PCM: ",
                         bits_per_sample, ". Only 16-bit is supported."));
      }
      if (audio_format == 3 && bits_per_sample != 32) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported WAV bit depth for IEEE float: ",
                         bits_per_sample, ". Only 32-bit float is supported."));
      }
      fmt_found = true;
    } else if (chunk_id == "data") {
      data_offset = offset;
      data_bytes = chunk_size;
      data_found = true;
    }

    size_t padded_chunk_size = chunk_size + (chunk_size % 2);
    offset += padded_chunk_size;
  }

  if (!fmt_found || !data_found) {
    return absl::InvalidArgumentError("Missing fmt or data chunk in WAV file.");
  }

  const size_t bytes_per_sample = bits_per_sample / 8;
  const size_t bytes_per_frame = wav_channels * bytes_per_sample;
  if (bytes_per_frame == 0) {
    return absl::InvalidArgumentError("Invalid WAV frame size.");
  }
  const size_t num_frames = data_bytes / bytes_per_frame;
  pcm_frames.resize(num_frames);

  const uint8_t* ptr =
      reinterpret_cast<const uint8_t*>(audio_bytes.data() + data_offset);

  if (audio_format == 1 && bits_per_sample == 16) {
    if (wav_channels == 1) {
      for (size_t i = 0; i < num_frames; ++i) {
        int16_t sample = static_cast<int16_t>(ptr[0] | (ptr[1] << 8));
        pcm_frames[i] = static_cast<float>(sample) / 32768.0f;
        ptr += 2;
      }
    } else {  // stereo downmix
      for (size_t i = 0; i < num_frames; ++i) {
        int16_t left = static_cast<int16_t>(ptr[0] | (ptr[1] << 8));
        int16_t right = static_cast<int16_t>(ptr[2] | (ptr[3] << 8));
        pcm_frames[i] = (static_cast<float>(left) + static_cast<float>(right)) /
                        (2.0f * 32768.0f);
        ptr += 4;
      }
    }
  } else if (audio_format == 3 && bits_per_sample == 32) {
    if (wav_channels == 1) {
      for (size_t i = 0; i < num_frames; ++i) {
        pcm_frames[i] = ReadLittleEndianFloat(ptr);
        ptr += sizeof(float);
      }
    } else {  // stereo downmix
      for (size_t i = 0; i < num_frames; ++i) {
        float left = ReadLittleEndianFloat(ptr);
        float right = ReadLittleEndianFloat(ptr + sizeof(float));
        pcm_frames[i] = (left + right) * 0.5f;
        ptr += 2 * sizeof(float);
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<InputAudio> AudioPreprocessorPffft::Preprocess(
    const InputAudio& input_audio) {
  if (input_audio.IsTensorBuffer()) {
    LITERT_ASSIGN_OR_RETURN(auto processed_audio_tensor,
                            input_audio.GetPreprocessedAudioTensor());
    LITERT_ASSIGN_OR_RETURN(auto processed_audio_tensor_with_reference,
                            processed_audio_tensor->Duplicate());
    InputAudio processed_audio(
        std::move(processed_audio_tensor_with_reference));
    return processed_audio;
  }
  std::vector<float> decoded_pcm_frames;
  absl::Span<const float> pcm_frames;
  if (input_audio.IsPcmFrames()) {
    LITERT_ASSIGN_OR_RETURN(pcm_frames, input_audio.GetPcmFrames());
  } else {
    LITERT_ASSIGN_OR_RETURN(auto raw_audio_bytes,
                            input_audio.GetRawAudioBytes());
    LITERT_RETURN_IF_ERROR(
        DecodeAudio(raw_audio_bytes, config_.GetNumChannels(),
                    config_.GetSampleRateHz(), decoded_pcm_frames));
    pcm_frames = decoded_pcm_frames;
  }

  if (!config_.SkipMelSpectrogramExtraction()) {
    std::vector<float> spectrograms;
    LITERT_RETURN_IF_ERROR(PcmFramesToSpectrogram(pcm_frames, spectrograms));

    std::vector<float> log_mel_spectrograms;
    LITERT_RETURN_IF_ERROR(ToLogMelSpectrogram(config_, *mel_filterbank_,
                                             spectrograms,
                                             log_mel_spectrograms));

    const int num_frames =
        log_mel_spectrograms.size() / config_.GetNumMelBins();
    RankedTensorType mel_tensor_type(
        GetElementType<float>(),
        Layout(Dimensions({1, num_frames, config_.GetNumMelBins()})));
    LITERT_ASSIGN_OR_RETURN(
        auto mel_spectrograms_tensor,
        TensorBuffer::CreateManagedHostMemory(
            mel_tensor_type, log_mel_spectrograms.size() * sizeof(float)));
    LITERT_RETURN_IF_ERROR(mel_spectrograms_tensor.Write<float>(
        absl::MakeSpan(log_mel_spectrograms)));
    return InputAudio(std::move(mel_spectrograms_tensor));
  } else {
    std::vector<float> pcm_vector(pcm_frames.begin(), pcm_frames.end());
    LITERT_ASSIGN_OR_RETURN(auto windowed_signals,
                          GetFramedSegments(config_, pcm_vector, input_queue_,
                                            samples_to_next_step_));

    const int num_frames = windowed_signals.size();
    if (num_frames == 0) {
      return absl::FailedPreconditionError(
          "Not enough samples to form any frame.");
    }
    RankedTensorType mel_tensor_type(
        GetElementType<float>(),
        Layout(Dimensions({1, num_frames, config_.GetFrameLength()})));
    LITERT_ASSIGN_OR_RETURN(
        auto mel_spectrograms_tensor,
        TensorBuffer::CreateManagedHostMemory(
            mel_tensor_type,
            num_frames * config_.GetFrameLength() * sizeof(float)));

    std::vector<float> flat_frames;
    flat_frames.reserve(num_frames * config_.GetFrameLength());
    for (const auto& frame : windowed_signals) {
      flat_frames.insert(flat_frames.end(), frame.begin(), frame.end());
    }
    LITERT_RETURN_IF_ERROR(
        mel_spectrograms_tensor.Write<float>(absl::MakeSpan(flat_frames)));
    return InputAudio(std::move(mel_spectrograms_tensor));
  }
}

}  // namespace litert::support
