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

// Bit-exact port of Pillow's BICUBIC resampling for 8-bit images
// (src/libImaging/Resample.c). Two separable passes with per-output-pixel
// coefficient tables; the bicubic kernel is the a = -0.5 cubic convolution and
// the filter support is scaled by the down-scaling ratio to antialias.
//
// The details that matter for bit-exactness, and that a "resample in double"
// implementation gets wrong by +/-1 per byte:
//   - Coefficients are computed in double but quantized to int32 fixed point
//     with 22 fractional bits, and the accumulation happens in int32.
//   - The horizontal pass writes a uint8 intermediate, so its rounding is
//     visible to the vertical pass.
//   - A pass is skipped entirely when the size along that axis is unchanged.

#include "runtime/components/preprocessor/pil_resize.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace litert::lm {
namespace {

constexpr double kBicubicSupport = 2.0;

// Pillow's PRECISION_BITS: 32 bits minus 8 for the value minus 2 for headroom.
constexpr int kPrecisionBits = 32 - 8 - 2;
constexpr int32_t kFixedOne = 1 << kPrecisionBits;

constexpr int kChannels = 3;

// Pillow's bicubic filter (a = -0.5).
double BicubicFilter(double x) {
  constexpr double a = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  }
  if (x < 2.0) {
    return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  }
  return 0.0;
}

// Pillow's clip8: arithmetic shift down out of fixed point, then saturate.
uint8_t Clip8(int32_t value) {
  const int32_t shifted = value >> kPrecisionBits;
  if (shifted < 0) return 0;
  if (shifted > 255) return 255;
  return static_cast<uint8_t>(shifted);
}

// Pillow's normalize_coeffs_8bpc: round away from zero into fixed point.
int32_t ToFixed(double value) {
  return static_cast<int32_t>(value < 0 ? -0.5 + value * kFixedOne
                                        : 0.5 + value * kFixedOne);
}

// Per-output-pixel resampling coefficients for one dimension, matching
// Pillow's precompute_coeffs followed by normalize_coeffs_8bpc.
struct Coeffs {
  int ksize = 0;
  // Two entries per output pixel: the first contributing input index and the
  // number of contributing inputs.
  std::vector<int> bounds;
  // out_size * ksize fixed-point coefficients.
  std::vector<int32_t> kk;
};

Coeffs PrecomputeCoeffs(int in_size, int out_size) {
  const double scale = static_cast<double>(in_size) / out_size;
  const double filterscale = std::max(scale, 1.0);
  const double support = kBicubicSupport * filterscale;
  const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

  Coeffs coeffs;
  coeffs.ksize = ksize;
  coeffs.bounds.resize(static_cast<size_t>(out_size) * 2);
  coeffs.kk.assign(static_cast<size_t>(out_size) * ksize, 0);

  std::vector<double> window(ksize);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double inverse_filterscale = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;

    double weight_sum = 0.0;
    for (int x = 0; x < xmax; ++x) {
      const double weight =
          BicubicFilter((x + xmin - center + 0.5) * inverse_filterscale);
      window[x] = weight;
      weight_sum += weight;
    }
    int32_t* kk = &coeffs.kk[static_cast<size_t>(xx) * ksize];
    for (int x = 0; x < xmax; ++x) {
      if (weight_sum != 0.0) window[x] /= weight_sum;
      kk[x] = ToFixed(window[x]);
    }
    coeffs.bounds[xx * 2 + 0] = xmin;
    coeffs.bounds[xx * 2 + 1] = xmax;
  }
  return coeffs;
}

// Resamples along x: [src_h, src_w, 3] -> [src_h, dst_w, 3].
std::vector<uint8_t> ResampleHorizontal(const uint8_t* src, int src_w,
                                        int src_h, int dst_w) {
  const Coeffs coeffs = PrecomputeCoeffs(src_w, dst_w);
  std::vector<uint8_t> out(static_cast<size_t>(src_h) * dst_w * kChannels);
  for (int yy = 0; yy < src_h; ++yy) {
    const uint8_t* src_row = src + static_cast<size_t>(yy) * src_w * kChannels;
    uint8_t* out_row = out.data() + static_cast<size_t>(yy) * dst_w * kChannels;
    for (int xx = 0; xx < dst_w; ++xx) {
      const int xmin = coeffs.bounds[xx * 2 + 0];
      const int xsize = coeffs.bounds[xx * 2 + 1];
      const int32_t* kk = &coeffs.kk[static_cast<size_t>(xx) * coeffs.ksize];
      for (int ch = 0; ch < kChannels; ++ch) {
        int32_t sum = 1 << (kPrecisionBits - 1);
        for (int x = 0; x < xsize; ++x) {
          sum += src_row[(xmin + x) * kChannels + ch] * kk[x];
        }
        out_row[xx * kChannels + ch] = Clip8(sum);
      }
    }
  }
  return out;
}

// Resamples along y: [src_h, width, 3] -> [dst_h, width, 3].
std::vector<uint8_t> ResampleVertical(const std::vector<uint8_t>& src,
                                      int width, int src_h, int dst_h) {
  const Coeffs coeffs = PrecomputeCoeffs(src_h, dst_h);
  std::vector<uint8_t> out(static_cast<size_t>(dst_h) * width * kChannels);
  for (int yy = 0; yy < dst_h; ++yy) {
    const int ymin = coeffs.bounds[yy * 2 + 0];
    const int ysize = coeffs.bounds[yy * 2 + 1];
    const int32_t* kk = &coeffs.kk[static_cast<size_t>(yy) * coeffs.ksize];
    uint8_t* out_row = out.data() + static_cast<size_t>(yy) * width * kChannels;
    for (int xx = 0; xx < width; ++xx) {
      for (int ch = 0; ch < kChannels; ++ch) {
        int32_t sum = 1 << (kPrecisionBits - 1);
        for (int y = 0; y < ysize; ++y) {
          sum += src[(static_cast<size_t>(ymin + y) * width + xx) * kChannels +
                     ch] *
                 kk[y];
        }
        out_row[xx * kChannels + ch] = Clip8(sum);
      }
    }
  }
  return out;
}

}  // namespace

std::vector<uint8_t> PilResizeBicubicRgb(const uint8_t* src, int src_w,
                                         int src_h, int dst_w, int dst_h) {
  // Pillow skips a pass when the corresponding axis is unchanged, which also
  // skips its rounding, so an unconditional two-pass resample would not be
  // bit-exact for same-width or same-height resizes.
  std::vector<uint8_t> intermediate;
  if (src_w != dst_w) {
    intermediate = ResampleHorizontal(src, src_w, src_h, dst_w);
  } else {
    intermediate.assign(src,
                        src + static_cast<size_t>(src_w) * src_h * kChannels);
  }
  if (src_h == dst_h) {
    return intermediate;
  }
  return ResampleVertical(intermediate, dst_w, src_h, dst_h);
}

}  // namespace litert::lm
