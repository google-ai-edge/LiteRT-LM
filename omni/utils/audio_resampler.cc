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

#include "omni/utils/audio_resampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "miniaudio.h"  // from @miniaudio

namespace litert::omni {

AudioResampler::~AudioResampler() {
  if (is_initialized_) {
    ma_resampler_uninit(&resampler_, /*pAllocationCallbacks=*/nullptr);
  }
}

void AudioResampler::Reset() {
  if (is_initialized_) {
    ma_resampler_reset(&resampler_);
  }
}

absl::Status AudioResampler::EnsureInitialized(int in_rate, int out_rate) {
  if (in_rate <= 0 || out_rate <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Sample rates must be positive: input=", in_rate,
                     ", target=", out_rate));
  }
  if (is_initialized_ && input_sample_rate_hz_ == in_rate &&
      target_sample_rate_hz_ == out_rate) {
    return absl::OkStatus();
  }

  if (is_initialized_) {
    ma_resampler_uninit(&resampler_, /*pAllocationCallbacks=*/nullptr);
    is_initialized_ = false;
  }

  ma_resampler_config config =
      ma_resampler_config_init(ma_format_f32, /*channels=*/1, in_rate, out_rate,
                               ma_resample_algorithm_linear);
  ma_result res =
      ma_resampler_init(&config, /*pAllocationCallbacks=*/nullptr, &resampler_);
  if (res != MA_SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "Failed to initialize miniaudio resampler (error ", res, ")"));
  }

  input_sample_rate_hz_ = in_rate;
  target_sample_rate_hz_ = out_rate;
  is_initialized_ = true;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> AudioResampler::Resample(
    absl::Span<const float> input_pcm, int input_sample_rate_hz,
    int target_sample_rate_hz) {
  if (input_sample_rate_hz <= 0 || target_sample_rate_hz <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Sample rates must be positive: input=", input_sample_rate_hz,
        ", target=", target_sample_rate_hz));
  }
  if (input_pcm.empty()) {
    return std::vector<float>();
  }
  if (input_sample_rate_hz == target_sample_rate_hz) {
    return std::vector<float>(input_pcm.begin(), input_pcm.end());
  }

  ABSL_RETURN_IF_ERROR(  // NOLINT - prefer ABSL_ as consistency
      EnsureInitialized(input_sample_rate_hz, target_sample_rate_hz));

  ma_uint64 expected_out = 0;
  ma_result len_res = ma_resampler_get_expected_output_frame_count(
      &resampler_, input_pcm.size(), &expected_out);
  if (len_res != MA_SUCCESS || expected_out == 0) {
    expected_out = static_cast<ma_uint64>(std::max<int64_t>(
        1, std::llround(static_cast<double>(input_pcm.size()) *
                        target_sample_rate_hz / input_sample_rate_hz)));
  }

  std::vector<float> output_pcm(expected_out + 32);
  ma_uint64 frames_in_processed = 0;
  ma_uint64 frames_out_processed = 0;

  while (frames_in_processed < input_pcm.size()) {
    ma_uint64 in_count = input_pcm.size() - frames_in_processed;
    ma_uint64 out_capacity = output_pcm.size() - frames_out_processed;
    if (out_capacity == 0) {
      output_pcm.resize(output_pcm.size() + 1024);
      out_capacity = output_pcm.size() - frames_out_processed;
    }
    ma_result res = ma_resampler_process_pcm_frames(
        &resampler_, input_pcm.data() + frames_in_processed, &in_count,
        output_pcm.data() + frames_out_processed, &out_capacity);
    if (res != MA_SUCCESS) {
      return absl::InternalError(absl::StrCat(
          "Failed to process PCM frames in resampler (error ", res, ")"));
    }
    frames_in_processed += in_count;
    frames_out_processed += out_capacity;
    if (in_count == 0 && out_capacity == 0) {
      break;
    }
  }

  output_pcm.resize(frames_out_processed);
  return output_pcm;
}

}  // namespace litert::omni
