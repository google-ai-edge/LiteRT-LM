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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_MEL_FILTERBANK_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_MEL_FILTERBANK_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::support {

// MelFilterbank is a class that converts a squared-magnitude spectrogram slice
// to a triangular-mel-weighted linear-magnitude filterbank, and vice versa.
class MelFilterbank {
 public:
  MelFilterbank();

  // Initializes the MelFilterbank.
  // Args:
  // - fft_length: Number of unique FFT bins fftsize/2+1.
  // - sample_rate: Sample rate of the input audio.
  // - mel_channel_count: Number of mel channels in the filterbank.
  // - lower_frequency_limit: Lower frequency limit of the mel filterbank.
  // - upper_frequency_limit: Upper frequency limit of the mel filterbank.
  absl::Status Initialize(int fft_length, double sample_rate,
                          int mel_channel_count, double lower_frequency_limit,
                          double upper_frequency_limit);

  // Takes a squared-magnitude spectrogram slice as input, computes a
  // triangular-mel-weighted linear-magnitude filterbank, and places the result
  // in mel.
  // Args:
  // - squared_magnitude_fft: Squared-magnitude spectrogram slice flattened as a
  //  1D array. The spectrogram slice should at least contain
  //  `upper_frequency_limit / (sample_rate_ / (2.0 * (fft_length_ - 1)))`
  //  values to compute the mel spectrum.
  // - mel: Output Mel spectrum. The output mel spectrum will be a 1D array
  //  `mel_channel_count` values.
  absl::Status ToMelSpectrum(absl::Span<const double> squared_magnitude_fft,
                             std::vector<double>* mel) const;

  // Takes a triangular-mel-weighted linear-magnitude filterbank and estimates
  // the squared-magnitude spectrogram slice that corresponds to it. This is
  // merely an estimate, so ToMelSpectrum() followed by ToSquaredMagnitudeFft()
  // will yield a good approximation to the original mel filterbank, but the
  // sequence of operations will not be a perfect roundtrip.
  // Args:
  // - mel: Mel spectrum. The input mel spectrum must contain
  //  `mel_channel_count` values.
  // - squared_magnitude_fft: Output squared-magnitude spectrogram slice. The
  //  output squared-magnitude spectrogram slice will be a 1D array
  //  `fft_length` values.
  absl::Status ToSquaredMagnitudeFft(
      absl::Span<const double> mel,
      std::vector<double>* squared_magnitude_fft) const;

 private:
  double FreqToMel(double freq) const;
  bool initialized_;
  int num_mel_channels_;
  double sample_rate_;
  int fft_length_;

  // Each FFT bin b contributes to two triangular mel channels, with
  // proportion weights_[b] going into mel channel band_mapper_[b], and
  // proportion (1 - weights_[b]) going into channel band_mapper_[b] + 1.
  // Thus, weights_ contains the weighting applied to each FFT bin for the
  // upper-half of the triangular band.
  std::vector<double> weights_;  // Right-side weight for this fft  bin.

  // FFT bin i contributes to the upper side of mel channel band_mapper_[i]
  std::vector<int> band_mapper_;

  // Holds the sum of all weights for a specific Mel channel. This includes the
  // weights on both the left and right sides of the triangle.
  std::vector<double> channel_weights_sum_;

  int start_index_;  // Lowest FFT bin used to calculate mel spectrum.
  int end_index_;    // Highest FFT bin used to calculate mel spectrum.

  MelFilterbank(const MelFilterbank&) = delete;
  MelFilterbank& operator=(const MelFilterbank&) = delete;
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_MEL_FILTERBANK_H_
