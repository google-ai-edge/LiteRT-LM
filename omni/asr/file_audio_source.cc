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

#include "omni/asr/file_audio_source.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "miniaudio.h"  // from @miniaudio

namespace litert::omni::asr {

void FileAudioSource::MaDecoderDeleter::operator()(ma_decoder* decoder) const {
  if (decoder != nullptr) {
    ma_decoder_uninit(decoder);
    delete decoder;
  }
}

absl::StatusOr<std::unique_ptr<FileAudioSource>> FileAudioSource::Create(
    absl::string_view file_path, absl::Duration interval,
    absl::Duration overlap, int sample_rate_hz, int num_channels) {
  if (sample_rate_hz <= 0 || num_channels <= 0) {
    return absl::InvalidArgumentError(
        "sample_rate_hz and num_channels must be positive.");
  }
  if (interval <= overlap) {
    return absl::InvalidArgumentError(
        "interval must be strictly longer than overlap.");
  }
  int samples_per_interval =
      static_cast<int>(sample_rate_hz * absl::ToDoubleSeconds(interval));
  int overlap_samples =
      static_cast<int>(sample_rate_hz * absl::ToDoubleSeconds(overlap));
  if (samples_per_interval <= overlap_samples) {
    return absl::InvalidArgumentError(
        "samples_per_interval must be strictly greater than overlap_samples.");
  }
  int step = samples_per_interval - overlap_samples;
  if (step <= 0) {
    return absl::InvalidArgumentError("step must be positive.");
  }

  ma_decoder_config config =
      ma_decoder_config_init(ma_format_f32, num_channels, sample_rate_hz);
  auto decoder = std::unique_ptr<ma_decoder, MaDecoderDeleter>(
      new ma_decoder(), MaDecoderDeleter());
  ma_result init_res = ma_decoder_init_file(std::string(file_path).c_str(),
                                            &config, decoder.get());
  if (init_res != MA_SUCCESS) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open or decode audio file: ", file_path));
  }

  ma_uint64 length_frames = 0;
  ma_result len_res =
      ma_decoder_get_length_in_pcm_frames(decoder.get(), &length_frames);
  if (len_res != MA_SUCCESS) {
    return absl::InternalError("Failed to get audio file length in frames.");
  }

  return std::unique_ptr<FileAudioSource>(new FileAudioSource(
      std::move(decoder), sample_rate_hz, num_channels, samples_per_interval,
      overlap_samples, step, static_cast<int64_t>(length_frames)));
}

FileAudioSource::FileAudioSource(
    std::unique_ptr<ma_decoder, MaDecoderDeleter> decoder, int sample_rate_hz,
    int num_channels, int samples_per_interval, int overlap_samples, int step,
    int64_t total_frames)
    : decoder_(std::move(decoder)),
      sample_rate_hz_(sample_rate_hz),
      num_channels_(num_channels),
      samples_per_interval_(samples_per_interval),
      overlap_samples_(overlap_samples),
      step_(step),
      total_frames_(total_frames),
      current_frame_(0),
      ring_buffer_(samples_per_interval_, 0.0f),
      ring_pos_(0),
      is_closed_(false) {}

FileAudioSource::~FileAudioSource() = default;

void FileAudioSource::ResetInternal() {
  if (decoder_ != nullptr) {
    ma_decoder_seek_to_pcm_frame(decoder_.get(), 0);
  }
  current_frame_ = 0;
  ring_pos_ = 0;
  is_closed_ = false;
  std::fill(ring_buffer_.begin(), ring_buffer_.end(), 0.0f);
}

bool FileAudioSource::NeedScheduleInternal() const {
  return !is_closed_ && current_frame_ < total_frames_;
}

absl::Status FileAudioSource::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (is_closed_ || current_frame_ >= total_frames_ || decoder_ == nullptr) {
    is_closed_ = true;
    return absl::OutOfRangeError("End of audio stream reached.");
  }

  auto* decoder = decoder_.get();
  int frames_to_read = (current_frame_ == 0) ? samples_per_interval_ : step_;
  int read_len = std::min(frames_to_read, samples_per_interval_ - ring_pos_);
  ABSL_ASSIGN_OR_RETURN(
      ma_uint64 frames_read,
      ReadFrames(decoder, ring_buffer_.data() + ring_pos_, read_len));

  if (frames_to_read > read_len) {
    read_len = frames_to_read - read_len;
    auto second_read_len =
        ReadFrames(decoder, ring_buffer_.data() + ring_pos_, read_len);
    if (second_read_len.ok()) {
      frames_read += *second_read_len;
    } else if (second_read_len.status().code() !=
               absl::StatusCode::kOutOfRange) {
      return second_read_len.status();
    }
  }

  if (current_frame_ >= total_frames_ ||
      frames_read < static_cast<ma_uint64>(frames_to_read)) {
    is_closed_ = true;
  }

  std::vector<float> chunk(samples_per_interval_);
  int first_part_len = samples_per_interval_ - ring_pos_;
  std::copy_n(ring_buffer_.begin() + ring_pos_, first_part_len, chunk.begin());
  std::copy_n(ring_buffer_.begin(), ring_pos_, chunk.begin() + first_part_len);

  PushOutput(std::move(chunk));
  return absl::OkStatus();
}

absl::StatusOr<ma_uint64> FileAudioSource::ReadFrames(ma_decoder* decoder,
                                                      float* buffer,
                                                      int num_frames_to_read) {
  ma_uint64 read_len = 0;
  ma_result res = ma_decoder_read_pcm_frames(
      decoder, ring_buffer_.data() + ring_pos_, num_frames_to_read, &read_len);
  if (res != MA_SUCCESS && res != MA_AT_END) {
    is_closed_ = true;
    return absl::InternalError("Failed to read PCM frames from file.");
  } else if (read_len == 0) {
    is_closed_ = true;
    return absl::OutOfRangeError("End of audio stream reached.");
  }

  // Pad with zeros if the read is shorter than the requested number of frames.
  if (read_len < static_cast<ma_uint64>(num_frames_to_read)) {
    std::fill_n(ring_buffer_.begin() + ring_pos_ + read_len,
                num_frames_to_read - read_len, 0.0f);
  }

  ring_pos_ = (ring_pos_ + num_frames_to_read) % samples_per_interval_;
  current_frame_ += num_frames_to_read;
  return read_len;
}

}  // namespace litert::omni::asr
