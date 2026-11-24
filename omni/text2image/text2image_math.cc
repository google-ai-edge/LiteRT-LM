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

#include "omni/text2image/text2image_math.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::text2image {
namespace {

// Maps a normalized activation `v` in `[-1, 1]` to an 8-bit channel value in
// `[0, 255]`, clamping out-of-range values and mapping non-finite (`NaN`)
// values to `0`.
uint8_t FloatToUint8Channel(float v) {
  const float clamped = std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f);
  return static_cast<uint8_t>(
      std::lround(std::isfinite(clamped) ? clamped * 255.0f : 0.0f));
}

}  // namespace

std::vector<float> SampleGaussianLatents(size_t count, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = dist(rng);
  }
  return out;
}

absl::StatusOr<std::vector<uint8_t>> ConvertNchwToRgb888(
    absl::Span<const float> nchw, int height, int width) {
  if (height <= 0 || width <= 0) {
    return absl::InvalidArgumentError("height and width must be positive.");
  }
  const size_t w = static_cast<size_t>(width);
  const size_t h = static_cast<size_t>(height);
  // Guard `width * height`, `3 * width * height`, and the input float byte size
  // (`3 * width * height * sizeof(float)`) against `size_t` overflow on both
  // 32-bit and 64-bit platforms.
  constexpr size_t kChannels = 3;
  constexpr size_t kBytesPerPixel = kChannels * sizeof(float);
  if (w > std::numeric_limits<size_t>::max() / kBytesPerPixel ||
      h > std::numeric_limits<size_t>::max() / (kBytesPerPixel * w)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Image dimensions (", height, "x", width, ") overflow size_t."));
  }
  const size_t plane_size = h * w;
  const size_t expected_size = plane_size * kChannels;
  if (nchw.size() != expected_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "nchw size (", nchw.size(), ") does not match 3 * height * width (",
        expected_size, ")."));
  }

  std::vector<uint8_t> rgb(expected_size);
  const float* r_plane = nchw.data();
  const float* g_plane = nchw.data() + plane_size;
  const float* b_plane = nchw.data() + 2 * plane_size;

  for (size_t i = 0; i < plane_size; ++i) {
    rgb[i * 3 + 0] = FloatToUint8Channel(r_plane[i]);
    rgb[i * 3 + 1] = FloatToUint8Channel(g_plane[i]);
    rgb[i * 3 + 2] = FloatToUint8Channel(b_plane[i]);
  }
  return rgb;
}

absl::StatusOr<std::pair<std::vector<float>, std::vector<float>>>
BuildMultiAxisRotaryPosEmbed(absl::Span<const float> joint_ids, int num_tokens,
                             absl::Span<const int> axes_dim, float theta) {
  if (num_tokens <= 0) {
    return absl::InvalidArgumentError("num_tokens must be positive.");
  }
  if (axes_dim.empty()) {
    return absl::InvalidArgumentError("axes_dim must not be empty.");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    return absl::InvalidArgumentError("theta must be finite and positive.");
  }
  int total_dim = 0;
  for (int dim : axes_dim) {
    if (dim <= 0 || dim % 2 != 0) {
      return absl::InvalidArgumentError(
          "Each entry in axes_dim must be a positive even integer.");
    }
    if (dim > std::numeric_limits<int>::max() - total_dim) {
      return absl::InvalidArgumentError("Sum of axes_dim overflows int.");
    }
    total_dim += dim;
  }

  const size_t num_tokens_size = static_cast<size_t>(num_tokens);
  const size_t total_dim_size = static_cast<size_t>(total_dim);
  constexpr size_t kMaxOutputElements =
      std::min(
          std::numeric_limits<size_t>::max(),
          static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())) /
      sizeof(float);
  if (num_tokens_size > kMaxOutputElements / total_dim_size) {
    return absl::InvalidArgumentError(
        "num_tokens * sum(axes_dim) overflows output buffer size.");
  }

  const size_t num_axes = axes_dim.size();
  if (num_tokens_size > std::numeric_limits<size_t>::max() / num_axes) {
    return absl::InvalidArgumentError(
        "num_tokens * axes_dim.size() overflows size_t.");
  }
  const size_t expected_joint_ids = num_tokens_size * num_axes;
  if (joint_ids.size() != expected_joint_ids) {
    return absl::InvalidArgumentError(
        absl::StrCat("joint_ids size (", joint_ids.size(),
                     ") must equal num_tokens * axes_dim.size() (",
                     expected_joint_ids, ")."));
  }

  // Precompute `inv_freq` once per axis in a single contiguous buffer of length
  // `total_dim / 2` to avoid redundant `std::pow` calls in the token loop while
  // keeping bit-identical `double` precision.
  std::vector<double> inv_freqs(total_dim_size / 2);
  {
    const double theta_d = static_cast<double>(theta);
    size_t freq_offset = 0;
    for (const int dim : axes_dim) {
      const int half_dim = dim / 2;
      for (int j = 0; j < half_dim; ++j) {
        inv_freqs[freq_offset + static_cast<size_t>(j)] =
            1.0 / std::pow(theta_d, static_cast<double>(2 * j) / dim);
      }
      freq_offset += static_cast<size_t>(half_dim);
    }
  }

  const size_t total_elements = num_tokens_size * total_dim_size;
  std::vector<float> cos_out(total_elements);
  std::vector<float> sin_out(total_elements);

  for (size_t n = 0; n < num_tokens_size; ++n) {
    float* cos_row = cos_out.data() + n * total_dim_size;
    float* sin_row = sin_out.data() + n * total_dim_size;
    size_t col_offset = 0;
    size_t freq_offset = 0;
    for (size_t a = 0; a < num_axes; ++a) {
      const int dim = axes_dim[a];
      const int half_dim = dim / 2;
      const double pos = static_cast<double>(joint_ids[n * num_axes + a]);
      for (int j = 0; j < half_dim; ++j) {
        const double angle =
            pos * inv_freqs[freq_offset + static_cast<size_t>(j)];
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));
        const size_t pair_offset = col_offset + static_cast<size_t>(2) * j;
        cos_row[pair_offset] = c;
        cos_row[pair_offset + 1] = c;
        sin_row[pair_offset] = s;
        sin_row[pair_offset + 1] = s;
      }
      col_offset += static_cast<size_t>(dim);
      freq_offset += static_cast<size_t>(half_dim);
    }
  }
  return std::make_pair(std::move(cos_out), std::move(sin_out));
}

}  // namespace litert::omni::text2image
