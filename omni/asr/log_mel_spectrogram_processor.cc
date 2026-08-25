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

#include "omni/asr/log_mel_spectrogram_processor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/audio_preprocessor.h"
#include "omni/base/stage.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"
#include "support/util/io_types.h"

namespace litert::omni::asr {
namespace {

constexpr float kEpsilon = 1e-5f;
constexpr float kLogZeroGuardValue = 5.960464477539063e-8f;  // 2**-24
constexpr float kLn10 = 2.302585092994046f;

int GetSmallestPowerOfTwoGreaterOrEqualTo(int n) {
  if (n > 0 && (n & (n - 1)) == 0) {
    return n;
  }
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

void ApplyPreemphasisInPlace(std::vector<float>& audio, float coeff) {
  if (audio.size() <= 1) return;
  for (size_t i = audio.size() - 1; i > 0; --i) {
    audio[i] = audio[i] - coeff * audio[i - 1];
  }
}

std::vector<float> ComputeMean(const std::vector<std::vector<float>>& mel_spec,
                               int n_frames) {
  std::vector<float> result(mel_spec.size(), 0.0f);
  if (n_frames <= 0) return result;
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    float sum = 0.0f;
    int count = std::min(static_cast<int>(mel_spec[m].size()), n_frames);
    if (count <= 0) continue;
    for (int f = 0; f < count; ++f) {
      sum += mel_spec[m][f];
    }
    result[m] = sum / count;
  }
  return result;
}

std::vector<float> ComputeStd(const std::vector<std::vector<float>>& mel_spec,
                              const std::vector<float>& mean, int n_frames) {
  std::vector<float> result(mel_spec.size(), 0.0f);
  if (n_frames <= 1) return result;
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    float sum_sq = 0.0f;
    int count = std::min(static_cast<int>(mel_spec[m].size()), n_frames);
    if (count <= 1) continue;
    for (int f = 0; f < count; ++f) {
      float diff = mel_spec[m][f] - mean[m];
      sum_sq += diff * diff;
    }
    // Sample variance (N - 1)
    result[m] = std::sqrt(sum_sq / (count - 1));
  }
  return result;
}

}  // namespace

absl::StatusOr<std::unique_ptr<LogMelSpectrogramProcessor>>
LogMelSpectrogramProcessor::Create(
    int sample_rate_hz, const LogMelSpectrogramConfig& config,
    Stage<std::vector<float>>* absl_nonnull audio_source) {
  if (sample_rate_hz <= 0) {
    return absl::InvalidArgumentError("Sample rate must be positive.");
  }
  if (config.n_fft <= 0) {
    return absl::InvalidArgumentError("n_fft must be positive.");
  }
  if (config.hop_length <= 0) {
    return absl::InvalidArgumentError("hop_length must be positive.");
  }
  if (config.n_mels <= 0) {
    return absl::InvalidArgumentError("n_mels must be positive.");
  }
  int padded_n_fft = GetSmallestPowerOfTwoGreaterOrEqualTo(config.n_fft);
  litert::support::AudioPreprocessorConfig preprocessor_config =
      litert::support::AudioPreprocessorConfig::Create(
          sample_rate_hz, /*num_channels=*/1, /*frame_length=*/config.n_fft,
          /*hop_length=*/config.hop_length, /*fft_length=*/padded_n_fft,
          /*input_scale=*/1.0f,
          /*pre_emphasis_factor=*/0.0f, /*num_mel_bins=*/config.n_mels,
          /*mel_low_hz=*/0.0f, /*mel_high_hz=*/sample_rate_hz / 2.0f,
          /*mel_floor=*/kLogZeroGuardValue, /*normalize_mel=*/false,
          /*add_floor_to_mel_before_log=*/true, /*semicausal_padding=*/false,
          /*non_zero_hanning=*/true, /*periodic_hanning=*/true,
          litert::support::AudioPreprocessorConfig::FftPaddingType::kRight);

  ABSL_ASSIGN_OR_RETURN(
      auto preprocessor,
      litert::support::AudioPreprocessorMiniAudio::Create(preprocessor_config));
  return absl::WrapUnique(new LogMelSpectrogramProcessor(
      sample_rate_hz, config, audio_source, std::move(preprocessor)));
}

LogMelSpectrogramProcessor::LogMelSpectrogramProcessor(
    int sample_rate_hz, const LogMelSpectrogramConfig& config,
    Stage<std::vector<float>>* absl_nonnull audio_source,
    std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> preprocessor)
    : AudioPreprocessor(audio_source),
      sample_rate_hz_(sample_rate_hz),
      config_(config),
      preprocessor_(std::move(preprocessor)) {}

void LogMelSpectrogramProcessor::ResetInternal() {
  if (preprocessor_) {
    preprocessor_->Reset();
  }
}

absl::Status LogMelSpectrogramProcessor::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto pcm_samples, audio_source_.GetOutput());
  ABSL_ASSIGN_OR_RETURN(auto features, Process(std::move(pcm_samples)));
  PushOutput(std::move(features));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> LogMelSpectrogramProcessor::Process(
    std::vector<float> raw_speech) {
  if (raw_speech.empty()) {
    return std::vector<float>();
  }

  if (config_.preemphasis > 0.0f) {
    ApplyPreemphasisInPlace(raw_speech, config_.preemphasis);
  }

  int valid_frames = raw_speech.size() / config_.hop_length;
  if (!preprocessor_) {
    return absl::InternalError(
        "AudioPreprocessorMiniAudio is not initialized.");
  }

  litert::support::InputAudio input_audio(std::move(raw_speech));
  ABSL_ASSIGN_OR_RETURN(auto processed_audio,
                        preprocessor_->Preprocess(input_audio));
  ABSL_ASSIGN_OR_RETURN(const auto* tensor_buffer,
                        processed_audio.GetPreprocessedAudioTensor());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer->TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  std::vector<float> log_mel_flat(elements);
  LITERT_RETURN_IF_ERROR(const_cast<litert::TensorBuffer*>(tensor_buffer)
                             ->Read<float>(absl::MakeSpan(log_mel_flat)));

  int num_frames = log_mel_flat.size() / config_.n_mels;
  std::vector<std::vector<float>> mel_spec(config_.n_mels,
                                           std::vector<float>(num_frames));
  for (int f = 0; f < num_frames; ++f) {
    for (int m = 0; m < config_.n_mels; ++m) {
      mel_spec[m][f] = log_mel_flat[f * config_.n_mels + m];
    }
  }

  if (config_.norm_type == NormType::kWhisper) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t m = 0; m < mel_spec.size(); ++m) {
      for (int f = 0; f < valid_frames && f < mel_spec[m].size(); ++f) {
        mel_spec[m][f] = mel_spec[m][f] / kLn10;
        if (mel_spec[m][f] > max_val) {
          max_val = mel_spec[m][f];
        }
      }
    }

    float clip_val = max_val - 8.0f;
    for (size_t m = 0; m < mel_spec.size(); ++m) {
      for (int f = 0; f < valid_frames && f < mel_spec[m].size(); ++f) {
        mel_spec[m][f] = (std::max(mel_spec[m][f], clip_val) + 4.0f) / 4.0f;
      }
    }
  } else {
    auto means = ComputeMean(mel_spec, valid_frames);
    auto stds = ComputeStd(mel_spec, means, valid_frames);
    for (size_t m = 0; m < mel_spec.size(); ++m) {
      float mean = means[m];
      float std = stds[m] + kEpsilon;
      for (int f = 0; f < valid_frames && f < mel_spec[m].size(); ++f) {
        mel_spec[m][f] = (mel_spec[m][f] - mean) / std;
      }
    }
  }

  int n_frames = (config_.n_frames > 0)
                     ? config_.n_frames
                     : (mel_spec.empty() ? 0 : mel_spec[0].size());
  std::vector<float> result(n_frames * config_.n_mels, 0.0f);
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    for (size_t f = 0; f < mel_spec[m].size(); ++f) {
      if (f >= n_frames) break;
      float value = (f < valid_frames) ? mel_spec[m][f] : 0.0f;
      if (config_.transpose) {
        result[f * config_.n_mels + m] = value;
      } else {
        result[m * n_frames + f] = value;
      }
    }
  }

  return result;
}

}  // namespace litert::omni::asr
