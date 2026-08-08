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
#include <memory>
#include <utility>
#include <vector>

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
  absl::Span<const float> pcm_frames;
  if (input_audio.IsPcmFrames()) {
    LITERT_ASSIGN_OR_RETURN(pcm_frames, input_audio.GetPcmFrames());
  } else {
    // Note: Unlike AudioPreprocessorMiniAudio, raw audio file decoding (e.g.,
    // WAV, MP3) is currently unsupported here to avoid non-allowlisted or large
    // media dependencies. For the intended use case (i.e. Chrome), audio file
    // decoding is handled upstream by Chromium's media pipeline, and
    // pre-decoded PCM frames are passed directly to this preprocessor.
    return absl::InvalidArgumentError(
        "AudioPreprocessorPffft does not support decoding raw audio bytes; "
        "input must be PCM frames.");
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
