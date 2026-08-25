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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_FILE_AUDIO_SOURCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_FILE_AUDIO_SOURCE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/asr/audio_source.h"
#include "miniaudio.h"  // from @miniaudio

namespace litert::omni::asr {

// AudioSource implementation that reads PCM frames from an audio file on disk
// on-demand without loading the entire file into memory, yielding
// sliding-window chunks.
class FileAudioSource : public AudioSource {
 public:
  ~FileAudioSource() override;

  static absl::StatusOr<std::unique_ptr<FileAudioSource>> Create(
      absl::string_view file_path, absl::Duration interval,
      absl::Duration overlap, int sample_rate_hz = 16000, int num_channels = 1);

  int GetSampleRateHz() const override { return sample_rate_hz_; }
  int GetNumChannels() const override { return num_channels_; }

 protected:
  void ResetInternal() override;
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  struct MaDecoderDeleter {
    void operator()(struct ma_decoder* decoder) const;
  };

  FileAudioSource(std::unique_ptr<struct ma_decoder, MaDecoderDeleter> decoder,
                  int sample_rate_hz, int num_channels,
                  int samples_per_interval, int overlap_samples, int step,
                  int64_t total_frames);

  // Reads up to num_frames_to_read from the decoder into the buffer.
  // Returns the number of frames actually read, or an error status.
  // Pads the buffer with zeros if the read is shorter than the requested
  // number of frames.
  absl::StatusOr<ma_uint64> ReadFrames(ma_decoder* decoder, float* buffer,
                                       int num_frames_to_read);

  const std::unique_ptr<struct ma_decoder, MaDecoderDeleter> decoder_;
  const int sample_rate_hz_;
  const int num_channels_;
  const int samples_per_interval_;
  const int overlap_samples_;
  const int step_;
  const int64_t total_frames_;

  int64_t current_frame_;
  std::vector<float> ring_buffer_;
  int ring_pos_;
  bool is_closed_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_FILE_AUDIO_SOURCE_H_
