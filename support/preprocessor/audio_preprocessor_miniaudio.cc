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

#include "support/preprocessor/audio_preprocessor_miniaudio.h"

#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

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
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_utils.h"
#include "support/preprocessor/mel_filterbank.h"
#include "support/util/io_types.h"
#include "miniaudio.h"  // from @miniaudio
#include "kiss_fftr.h"  // from @kissfft

namespace litert::support {

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

absl::Status AudioPreprocessorMiniAudio::PcmFramesToSpectrogram(
    absl::Span<const float> pcm_frames, std::vector<float>& spectrograms) {
  LITERT_ASSIGN_OR_RETURN(
      auto windowed_signals,
      GetWindowedSignalsForFft(config_, pcm_frames, input_queue_,
                               samples_to_next_step_));

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

absl::StatusOr<std::unique_ptr<AudioPreprocessorMiniAudio>>
AudioPreprocessorMiniAudio::Create(const AudioPreprocessorConfig& config) {
  if (config.GetFrameLength() <= 0) {
    return absl::InvalidArgumentError("Frame length must be positive.");
  }
  auto mel_filterbank = std::make_unique<MelFilterbank>();
  LITERT_RETURN_IF_ERROR(mel_filterbank->Initialize(
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
    LITERT_RETURN_IF_ERROR(ToLogMelSpectrogram(
        config_, *mel_filterbank_, spectrograms, log_mel_spectrograms));

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
