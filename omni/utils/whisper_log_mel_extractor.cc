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

#include "omni/utils/whisper_log_mel_extractor.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/utils/audio_utils.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"
#include "support/util/io_types.h"

namespace litert::omni {

absl::StatusOr<std::unique_ptr<WhisperLogMelExtractor>>
WhisperLogMelExtractor::Create(const Config& config) {
  if (config.sample_rate_hz <= 0 || config.n_fft <= 0 ||
      config.hop_length <= 0 || config.n_mels <= 0 || config.n_frames <= 0 ||
      config.mel_floor <= 0.0f) {
    return absl::InvalidArgumentError(
        "WhisperLogMelExtractor::Config parameters must be positive.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> preprocessor,
      litert::support::AudioPreprocessorMiniAudio::Create(
          CreateWhisperAudioPreprocessorConfig(
              config.sample_rate_hz, config.n_fft, config.hop_length,
              config.n_mels, config.mel_floor)));
  return absl::WrapUnique(
      new WhisperLogMelExtractor(config, std::move(preprocessor)));
}

absl::StatusOr<std::unique_ptr<WhisperLogMelExtractor>>
WhisperLogMelExtractor::Create() {
  return Create(Config{});
}

WhisperLogMelExtractor::WhisperLogMelExtractor(
    const Config& config,
    std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nonnull
    preprocessor)
    : config_(config), preprocessor_(std::move(preprocessor)) {}

WhisperLogMelExtractor::~WhisperLogMelExtractor() = default;

void WhisperLogMelExtractor::Reset() { preprocessor_->Reset(); }

const WhisperLogMelExtractor::Config& WhisperLogMelExtractor::GetConfig()
    const {
  return config_;
}

absl::StatusOr<std::vector<float>> WhisperLogMelExtractor::Extract(
    absl::Span<const float> pcm_samples) {
  preprocessor_->Reset();
  const size_t target_samples =
      static_cast<size_t>(config_.n_frames) * config_.hop_length;
  const size_t padded_samples = std::max(
      target_samples, static_cast<size_t>(config_.n_fft + config_.hop_length));
  const size_t copy_len = std::min(pcm_samples.size(), target_samples);
  const int valid_frames = static_cast<int>(copy_len / config_.hop_length);

  std::vector<float> raw_speech(padded_samples, 0.0f);
  if (copy_len > 0) {
    std::copy_n(pcm_samples.data(), copy_len, raw_speech.data());
  }
  PutSamplesInCenter(raw_speech, config_.n_fft, config_.hop_length);

  litert::support::InputAudio input_audio(std::move(raw_speech));
  ABSL_ASSIGN_OR_RETURN(litert::support::InputAudio processed_audio,
                        preprocessor_->Preprocess(input_audio));
  ABSL_ASSIGN_OR_RETURN(const ::litert::TensorBuffer* tensor_buffer,
                        processed_audio.GetPreprocessedAudioTensor());
  LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType tensor_type,
                          tensor_buffer->TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t elements, tensor_type.Layout().NumElements());
  const int num_frames = static_cast<int>(elements) / config_.n_mels;
  if (num_frames <= 0) {
    return absl::InternalError(absl::StrCat(
        "Audio preprocessor output has insufficient elements (", elements,
        ") for n_mels (", config_.n_mels, ")."));
  }
  const int stat_frames = std::clamp(valid_frames, 1, num_frames);

  std::vector<std::vector<float>> mel_spec(config_.n_mels,
                                           std::vector<float>(num_frames));
  {
    LITERT_ASSIGN_OR_RETURN(
        (std::pair<::litert::TensorBufferScopedLock, const float*>
             lock_and_addr),
        ::litert::TensorBufferScopedLock::Create<const float>(
            *const_cast<::litert::TensorBuffer*>(tensor_buffer),
            ::litert::TensorBuffer::LockMode::kRead));
    const float* log_mel_flat = lock_and_addr.second;
    for (int f = 0; f < num_frames; ++f) {
      for (int m = 0; m < config_.n_mels; ++m) {
        mel_spec[m][f] = log_mel_flat[f * config_.n_mels + m];
      }
    }
  }

  NormalizeWhisperMelSpectrogram(mel_spec, stat_frames, num_frames);
  return FlattenMelSpectrogram(mel_spec, config_.n_frames, config_.n_mels,
                               config_.transpose);
}

}  // namespace litert::omni
