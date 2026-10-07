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

#include "omni/text2image/flux2/flux2_math.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::text2image {
namespace {

constexpr int kPackedChannels = 128;
constexpr int kVaeLatentChannels = 32;
constexpr int kPatchSize = 2;

// Maps a normalized activation `v` in `[-1, 1]` to an 8-bit channel value in
// `[0, 255]`, clamping out-of-range values and mapping non-finite (`NaN`)
// values to `0`.
uint8_t FloatToUint8Channel(float v) {
  const float clamped = std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f);
  return static_cast<uint8_t>(
      std::lround(std::isfinite(clamped) ? clamped * 255.0f : 0.0f));
}

}  // namespace

// Adapted from `compute_empirical_mu` in HuggingFace Diffusers
// (`diffusers/pipelines/flux2/pipeline_flux2_klein.py`) and exponential time
// shifting (`_time_shift_exponential`) in `FlowMatchEulerDiscreteScheduler`
// (`diffusers/schedulers/scheduling_flow_match_euler_discrete.py`):
//   1. Computes the empirical time-shift parameter `mu` as a function of the
//      image patch sequence length (`image_tokens`) and `steps`, linearly
//      interpolating between empirical linear fits at 10 steps (`m10`) and 200
//      steps (`m200`) for `image_tokens <= 4300` (and using `m200` above 4300).
//   2. Applies `sigma(t) = exp(mu) / (exp(mu) + (1 / t - 1)^1.0)` to the
//      linearly spaced schedule `t in linspace(1.0, 1.0 / steps, steps)`,
//      followed by a terminal sigma of 0.0.
absl::StatusOr<std::vector<float>> ComputeFlowMatchSigmas(int steps,
                                                          int image_tokens) {
  if (steps <= 0) {
    return absl::InvalidArgumentError("steps must be positive.");
  }
  if (image_tokens <= 0) {
    return absl::InvalidArgumentError("image_tokens must be positive.");
  }

  const double m200 = 0.00016927 * image_tokens + 0.45666666;
  const double m10 = 8.73809524e-05 * image_tokens + 1.89833333;
  const double a = (m200 - m10) / 190.0;
  const double b = m200 - 200.0 * a;
  const double mu = (image_tokens > 4300) ? m200 : (a * steps + b);
  const double exp_mu = std::exp(mu);
  const int denom_steps = std::max(steps - 1, 1);

  std::vector<float> sigmas(steps + 1, 0.0f);
  for (int i = 0; i < steps; ++i) {
    const double lin =
        1.0 - static_cast<double>(i) * (1.0 - 1.0 / steps) / denom_steps;
    const double shifted = exp_mu / (exp_mu + (1.0 / lin - 1.0));
    sigmas[i] = static_cast<float>(shifted);
  }
  sigmas[steps] = 0.0f;
  return sigmas;
}

std::vector<float> BuildImagePositionIds(int grid_h, int grid_w) {
  std::vector<float> ids(static_cast<size_t>(grid_h) * grid_w * 4, 0.0f);
  size_t offset = 0;
  for (int h = 0; h < grid_h; ++h) {
    for (int w = 0; w < grid_w; ++w) {
      ids[offset + 0] = 0.0f;
      ids[offset + 1] = static_cast<float>(h);
      ids[offset + 2] = static_cast<float>(w);
      ids[offset + 3] = 0.0f;
      offset += 4;
    }
  }
  return ids;
}

std::vector<float> BuildTextPositionIds(int seq_len) {
  std::vector<float> ids(static_cast<size_t>(seq_len) * 4, 0.0f);
  for (int i = 0; i < seq_len; ++i) {
    ids[static_cast<size_t>(i) * 4 + 3] = static_cast<float>(i);
  }
  return ids;
}

std::vector<float> SampleGaussianLatents(size_t count, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = dist(rng);
  }
  return out;
}

absl::StatusOr<std::vector<float>> UnpatchifyLatents(
    absl::Span<const float> packed_latents, int grid_h, int grid_w,
    absl::Span<const float> bn_scale, absl::Span<const float> bn_shift) {
  if (grid_h <= 0 || grid_w <= 0) {
    return absl::InvalidArgumentError("grid_h and grid_w must be positive.");
  }
  if (bn_scale.size() != kPackedChannels ||
      bn_shift.size() != kPackedChannels) {
    return absl::InvalidArgumentError(
        "bn_scale and bn_shift must have 128 elements.");
  }
  const size_t expected_size =
      static_cast<size_t>(grid_h) * grid_w * kPackedChannels;
  if (packed_latents.size() != expected_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("packed_latents size (", packed_latents.size(),
                     ") does not match expected size (", expected_size, ")."));
  }

  const int out_h = grid_h * kPatchSize;
  const int out_w = grid_w * kPatchSize;
  std::vector<float> nchw(
      static_cast<size_t>(kVaeLatentChannels) * out_h * out_w, 0.0f);

  // Equivalent to:
  //   z = packed_latents * bn_scale + bn_shift
  //   z.reshape(grid_h, grid_w, 32, 2, 2).transpose(2, 0, 3, 1, 4)
  //    .reshape(1, 32, 2 * grid_h, 2 * grid_w)
  for (int h = 0; h < grid_h; ++h) {
    for (int w = 0; w < grid_w; ++w) {
      const size_t token_idx = static_cast<size_t>(h) * grid_w + w;
      const float* token_ptr =
          packed_latents.data() + token_idx * kPackedChannels;
      for (int c = 0; c < kVaeLatentChannels; ++c) {
        for (int ph = 0; ph < kPatchSize; ++ph) {
          for (int pw = 0; pw < kPatchSize; ++pw) {
            const int p = (c * kPatchSize + ph) * kPatchSize + pw;
            const float z = token_ptr[p] * bn_scale[p] + bn_shift[p];
            const int y = h * kPatchSize + ph;
            const int x = w * kPatchSize + pw;
            const size_t out_idx =
                (static_cast<size_t>(c) * out_h + y) * out_w + x;
            nchw[out_idx] = z;
          }
        }
      }
    }
  }
  return nchw;
}

absl::StatusOr<std::vector<uint8_t>> ConvertNchwToRgb888(
    absl::Span<const float> nchw, int height, int width) {
  if (height <= 0 || width <= 0) {
    return absl::InvalidArgumentError("height and width must be positive.");
  }
  const size_t plane_size = static_cast<size_t>(height) * width;
  if (nchw.size() != plane_size * 3) {
    return absl::InvalidArgumentError(absl::StrCat(
        "nchw size (", nchw.size(), ") does not match 3 * height * width (",
        plane_size * 3, ")."));
  }

  std::vector<uint8_t> rgb(plane_size * 3);
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

}  // namespace litert::omni::text2image
