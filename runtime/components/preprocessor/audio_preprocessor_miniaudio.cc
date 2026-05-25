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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/components/preprocessor/mel_filterbank.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "miniaudio.h"  // from @miniaudio
#include "kiss_fftr.h"  // from @kissfft

namespace litert::lm {

namespace {
// Pads or truncates the input vector to the given fft_length.
// Args:
//   - input: The input vector to be padded or truncated.
//   - fft_length: The fft length to be padded or truncated to.
//   - padding_type: The padding mode to be used for padding.
//   - output: The output vector to be padded or truncated to.
// Returns:
//   A status object indicating whether the padding or truncation was
//   successful.
absl::Status PadOrTruncateForFft(
    const std::vector<float>& input, int fft_length,
    AudioPreprocessorConfig::FftPaddingType padding_type,
    std::vector<float>& output) {
  int input_dim = input.size();
  if (input_dim == fft_length) {
    output = input;
    return absl::OkStatus();
  }

  output.assign(fft_length, 0.0f);
  if (input_dim < fft_length) {
    int pad_amount = fft_length - input_dim;
    int pad_left = 0;
    if (padding_type == AudioPreprocessorConfig::FftPaddingType::kCenter) {
      pad_left = pad_amount / 2;
    } else if (padding_type ==
               AudioPreprocessorConfig::FftPaddingType::kRight) {
      pad_left = 0;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported padding: ", padding_type));
    }
    absl::c_copy(input, output.begin() + pad_left);
  } else {
    int trim_left = 0;
    if (padding_type == AudioPreprocessorConfig::FftPaddingType::kCenter) {
      trim_left = (input_dim - fft_length) / 2;
    } else if (padding_type ==
               AudioPreprocessorConfig::FftPaddingType::kRight) {
      trim_left = 0;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported padding: ", padding_type));
    }
    std::copy(input.begin() + trim_left, input.begin() + trim_left + fft_length,
              output.begin());
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status AudioPreprocessorMiniAudio::DecodeAudio(
    absl::string_view audio_bytes, int num_channels, int sample_rate_hz,
    std::vector<float>& pcm_frames) {
  if (num_channels != 1) {
    return absl::InvalidArgumentError("Only mono audio is supported.");
  }
  ma_decoder_config decoder_config =
      ma_decoder_config_init(ma_format_f32, num_channels, sample_rate_hz);
  ma_decoder decoder;
  ma_result decode_result = ma_decoder_init_memory(
      audio_bytes.data(), audio_bytes.size(), &decoder_config, &decoder);
  if (decode_result != ma_result::MA_SUCCESS) {
    ma_decoder_uninit(&decoder);
    return absl::InternalError(absl::StrCat(
        "Failed to initialize miniaudio decoder, error code: ", decode_result));
  }

  ma_uint64 frame_count;
  ma_uint64 frames_read;
  ma_result get_count_result =
      ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
  if (get_count_result != MA_SUCCESS) {
    ma_decoder_uninit(&decoder);
    return absl::InternalError(absl::StrCat(
        "Failed to get frame count, error code: ", get_count_result));
  }

  pcm_frames.resize(frame_count);
  ma_result read_frame_result = ma_decoder_read_pcm_frames(
      &decoder, pcm_frames.data(), frame_count, &frames_read);
  if (read_frame_result != MA_SUCCESS) {
    ma_decoder_uninit(&decoder);
    return absl::InternalError(absl::StrCat(
        "Failed to read pcm frames, error code: ", read_frame_result));
  }
  if (frames_read != frame_count) {
    ABSL_LOG(WARNING) << "Read " << frames_read << " PCM frames instead of "
                      << frame_count << " frames as requested.";
  }
  ma_decoder_uninit(&decoder);

  return absl::OkStatus();
}

std::vector<float> GetHanningWindow(int window_length,
                                    bool use_periodic_hanning,
                                    bool non_zero_hanning) {
  int even = 1 - window_length % 2;
  int n = window_length + static_cast<int>(use_periodic_hanning) * even - 1;
  float arg = M_PI * 2.0 / n;
  std::vector<float> hanning_window(window_length, 0);
  const float shift = non_zero_hanning ? 0.5 : 0.0;
  for (int i = 0; i < window_length; ++i) {
    hanning_window[i] = 0.5 - (0.5 * cos(arg * (i + shift)));
  }
  return hanning_window;
}

bool AudioPreprocessorMiniAudio::GetNextWindowOfSamples(
    const std::vector<float>& pcm_frames, int& input_start) {
  auto input_it = pcm_frames.begin() + input_start;
  int input_remaining = pcm_frames.end() - input_it;
  if (samples_to_next_step_ > input_remaining) {
    // Copy in as many samples are left and return false, no full window.
    input_queue_.insert(input_queue_.end(), input_it, pcm_frames.end());
    input_start += input_remaining;  // Increases it to input.size().
    samples_to_next_step_ -= input_remaining;
    return false;  // Not enough for a full window.
  } else {
    // Copy just enough into queue to make a new window.
    if (samples_to_next_step_ < config_.GetFrameLength()) {
      input_queue_.erase(
          input_queue_.begin(),
          input_queue_.begin() + input_queue_.size() -
              (config_.GetFrameLength() - samples_to_next_step_));
      input_queue_.insert(input_queue_.end(), input_it,
                          input_it + samples_to_next_step_);
    } else {
      input_queue_.assign(
          input_it + samples_to_next_step_ - config_.GetFrameLength(),
          input_it + samples_to_next_step_);
    }
    input_start += samples_to_next_step_;
    samples_to_next_step_ = config_.GetHopLength();  // Be ready for next step.
    return true;  // Yes, input_queue_ now contains exactly a window-full.
  }
}

absl::StatusOr<std::vector<std::vector<float>>>
AudioPreprocessorMiniAudio::GetFramedSegments(
    const std::vector<float>& pcm_frames) {
  std::vector<std::vector<float>> windowed_signals;
  int input_start = 0;
  while (GetNextWindowOfSamples(pcm_frames, input_start)) {
    if (input_queue_.size() != config_.GetFrameLength()) {
      return absl::InternalError(
          absl::StrCat("Input queue size is not equal to frame length: ",
                       input_queue_.size(), " vs ", config_.GetFrameLength()));
    }
    windowed_signals.push_back(input_queue_);
  }

  if (!config_.BufferLastFrame() && !input_queue_.empty() &&
      input_queue_.size() < config_.GetFrameLength()) {
    input_queue_.resize(config_.GetFrameLength(), 0.0f);
    windowed_signals.push_back(input_queue_);
    input_queue_.clear();
    if (config_.GetSemicausalPadding()) {
      samples_to_next_step_ = config_.GetFrameLength() - config_.GetHopLength();
      input_queue_.resize(config_.GetHopLength(), 0.0f);
    } else {
      samples_to_next_step_ = config_.GetFrameLength();
    }
  }
  return windowed_signals;
}

absl::Status AudioPreprocessorMiniAudio::PcmFramesToSpectrogram(
    absl::Span<const float> pcm_frames, std::vector<float>& spectrograms) {
  const float input_scale = config_.GetInputScale();
  const float pre_emphasis_factor = config_.GetPreEmphasisFactor();
  std::vector<float> scaled_pcm_frames(pcm_frames.size(), 0);
  absl::c_transform(pcm_frames, scaled_pcm_frames.begin(),
                    [&input_scale](float x) { return x * input_scale; });

  ASSIGN_OR_RETURN(auto raw_windowed_signals,
                   GetFramedSegments(scaled_pcm_frames));

  std::vector<std::vector<float>> windowed_signals;
  windowed_signals.reserve(raw_windowed_signals.size());
  for (int i = 0; i < raw_windowed_signals.size(); ++i) {
    const auto& input_frame = raw_windowed_signals[i];
    windowed_signals.push_back(std::vector<float>(config_.GetFrameLength(), 0));
    std::vector<float>& current_frame = windowed_signals.back();
    current_frame[0] = input_frame[0] * (1 - pre_emphasis_factor);
    for (int j = 1; j < config_.GetFrameLength(); ++j) {
      current_frame[j] =
          input_frame[j] - pre_emphasis_factor * input_frame[j - 1];
    }
  }
  const std::vector<float> hanning_window =
      GetHanningWindow(config_.GetFrameLength(), config_.GetPeriodicHanning(),
                       config_.GetNonZeroHanning());
  for (int i = 0; i < windowed_signals.size(); ++i) {
    std::vector<float>& current_frame = windowed_signals[i];
    for (int j = 0; j < current_frame.size(); ++j) {
      current_frame[j] *= hanning_window[j];
    }
    std::vector<float> output_frame;
    auto status =
        PadOrTruncateForFft(current_frame, config_.GetFftLength(),
                            config_.GetFftPaddingType(), output_frame);
    if (!status.ok()) {
      return status;
    }
    current_frame = std::move(output_frame);
  }

  kiss_fftr_cfg fft_alloc = kiss_fftr_alloc(config_.GetFftLength(),
                                            /*inverse_fft=*/0,
                                            /*mem=*/nullptr,
                                            /*lenmem=*/nullptr);
  kiss_fft_cpx* temp_out =
      (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * (config_.GetFftBins()));
  for (int i = 0; i < windowed_signals.size(); ++i) {
    std::vector<float>& current_frame = windowed_signals[i];
    kiss_fftr(fft_alloc, current_frame.data(), temp_out);
    for (int j = 0; j < config_.GetFftBins(); ++j) {
      spectrograms.push_back(temp_out[j].r * temp_out[j].r +
                             temp_out[j].i * temp_out[j].i);
    }
  }
  free(temp_out);
  kiss_fftr_free(fft_alloc);

  return absl::OkStatus();
}

absl::Status AudioPreprocessorMiniAudio::ToLogMelSpectrogram(
    const std::vector<float>& spectrograms,
    std::vector<float>& log_mel_spectrograms) {
  std::vector<double> spectrograms_double(spectrograms.size());
  for (int i = 0; i < spectrograms.size(); ++i) {
    spectrograms_double[i] = spectrograms[i];
  }
  int fft_bins = config_.GetFftBins();
  const int frames = spectrograms.size() / fft_bins;
  log_mel_spectrograms.reserve(frames * config_.GetNumMelBins());
  std::vector<double> tmp_log_mel(config_.GetNumMelBins(), 0);
  for (int i = 0; i < frames; ++i) {
    RETURN_IF_ERROR(mel_filterbank_->ToMelSpectrum(
        absl::MakeSpan(spectrograms_double.data() + i * fft_bins, fft_bins),
        &tmp_log_mel));
    for (int j = 0; j < tmp_log_mel.size(); ++j) {
      float log_mel;
      if (config_.GetAddFloorToMelBeforeLog()) {
        log_mel = std::log(static_cast<float>(tmp_log_mel[j]) +
                           config_.GetMelFloor());
      } else {
        log_mel = std::max(std::log(static_cast<float>(tmp_log_mel[j])),
                           config_.GetMelFloor());
      }
      if (config_.GetNormalizeMel()) {
        log_mel = (log_mel - AudioPreprocessorConfig::kUsmMelMean[j]) /
                  AudioPreprocessorConfig::kUsmMelStdDev[j];
      }
      log_mel_spectrograms.push_back(log_mel);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AudioPreprocessorMiniAudio>>
AudioPreprocessorMiniAudio::Create(const AudioPreprocessorConfig& config) {
  auto mel_filterbank = std::make_unique<MelFilterbank>();
  RETURN_IF_ERROR(mel_filterbank->Initialize(
      config.GetFftBins(), config.GetSampleRateHz(), config.GetNumMelBins(),
      config.GetMelLowHz(), config.GetMelHighHz()));
  return absl::WrapUnique(
      new AudioPreprocessorMiniAudio(config, std::move(mel_filterbank)));
}

// The preprocessing steps are:
// 1. Decode the audio bytes to PCM frames.
// 2. Convert PCM frames to spectrograms. (STFT)
// 3. Convert spectrograms to log mel spectrograms. (Mel filterbank)
// 4. Create a tensor buffer for the log mel spectrograms.
absl::StatusOr<InputAudio> AudioPreprocessorMiniAudio::Preprocess(
    const InputAudio& input_audio) {
  if (input_audio.IsTensorBuffer()) {
    ASSIGN_OR_RETURN(auto processed_audio_tensor,
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
    ASSIGN_OR_RETURN(pcm_frames, input_audio.GetPcmFrames());
  } else {
    ASSIGN_OR_RETURN(auto raw_audio_bytes, input_audio.GetRawAudioBytes());
    RETURN_IF_ERROR(DecodeAudio(raw_audio_bytes, config_.GetNumChannels(),
                                config_.GetSampleRateHz(), decoded_pcm_frames));
    pcm_frames = decoded_pcm_frames;
  }
  if (!config_.SkipMelSpectrogramExtraction()) {
    std::vector<float> spectrograms;
    RETURN_IF_ERROR(PcmFramesToSpectrogram(pcm_frames, spectrograms));

    std::vector<float> log_mel_spectrograms;
    RETURN_IF_ERROR(ToLogMelSpectrogram(spectrograms, log_mel_spectrograms));

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
    ASSIGN_OR_RETURN(auto windowed_signals, GetFramedSegments(pcm_vector));

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

}  // namespace litert::lm
