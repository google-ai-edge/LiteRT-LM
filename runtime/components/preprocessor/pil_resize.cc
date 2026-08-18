// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Faithful port of Pillow's BICUBIC resampling (src/libImaging/Resample.c).
// Two separable passes with per-output-pixel coefficient tables. Bicubic
// kernel is the a = -0.5 cubic convolution; the filter support is scaled by
// the down-scaling ratio to antialias. Coefficients are kept in double and
// the accumulation is done in double, then rounded to uint8 — matching
// Image.resize(BICUBIC) to within rounding.

#include "runtime/components/preprocessor/pil_resize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace litert::lm {
namespace {

constexpr double kBicubicSupport = 2.0;

// Pillow's bicubic filter (a = -0.5).
inline double BicubicFilter(double x) {
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

inline uint8_t Clip8Double(double v) {
  // Pillow rounds to nearest then clamps.
  long r = std::lround(v);
  if (r < 0) return 0;
  if (r > 255) return 255;
  return static_cast<uint8_t>(r);
}

// Per-output-pixel resampling coefficients for one dimension (Pillow's
// precompute_coeffs), kept in double.
struct Coeffs {
  int ksize;
  std::vector<int> bounds;    // 2 * out: (min, size) per output pixel.
  std::vector<double> kk;     // out * ksize coefficients.
};

Coeffs PrecomputeCoeffs(int in_size, int out_size) {
  const double scale = static_cast<double>(in_size) / out_size;
  const double filterscale = std::max(scale, 1.0);
  const double support = kBicubicSupport * filterscale;
  const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

  Coeffs c;
  c.ksize = ksize;
  c.bounds.resize(static_cast<size_t>(out_size) * 2);
  c.kk.assign(static_cast<size_t>(out_size) * ksize, 0.0);

  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &c.kk[static_cast<size_t>(xx) * ksize];
    double ww = 0.0;
    for (int x = 0; x < xmax; ++x) {
      const double w = BicubicFilter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x) {
      if (ww != 0.0) k[x] /= ww;
    }
    c.bounds[xx * 2 + 0] = xmin;
    c.bounds[xx * 2 + 1] = xmax;
  }
  return c;
}

}  // namespace

std::vector<uint8_t> PilResizeBicubicRgb(const uint8_t* src, int src_w,
                                         int src_h, int dst_w, int dst_h) {
  constexpr int C = 3;
  const Coeffs hc = PrecomputeCoeffs(src_w, dst_w);
  const Coeffs vc = PrecomputeCoeffs(src_h, dst_h);

  // Horizontal pass -> float intermediate [src_h x dst_w x C].
  std::vector<float> horiz(static_cast<size_t>(src_h) * dst_w * C);
  for (int yy = 0; yy < src_h; ++yy) {
    const uint8_t* srow = src + static_cast<size_t>(yy) * src_w * C;
    float* orow = horiz.data() + static_cast<size_t>(yy) * dst_w * C;
    for (int xx = 0; xx < dst_w; ++xx) {
      const int xmin = hc.bounds[xx * 2 + 0];
      const int xsize = hc.bounds[xx * 2 + 1];
      const double* k = &hc.kk[static_cast<size_t>(xx) * hc.ksize];
      for (int ch = 0; ch < C; ++ch) {
        double ss = 0.0;
        for (int x = 0; x < xsize; ++x) {
          ss += srow[(xmin + x) * C + ch] * k[x];
        }
        orow[xx * C + ch] = static_cast<float>(ss);
      }
    }
  }

  // Vertical pass -> uint8 output [dst_h x dst_w x C].
  std::vector<uint8_t> out(static_cast<size_t>(dst_h) * dst_w * C);
  for (int yy = 0; yy < dst_h; ++yy) {
    const int ymin = vc.bounds[yy * 2 + 0];
    const int ysize = vc.bounds[yy * 2 + 1];
    const double* k = &vc.kk[static_cast<size_t>(yy) * vc.ksize];
    uint8_t* orow = out.data() + static_cast<size_t>(yy) * dst_w * C;
    for (int xx = 0; xx < dst_w; ++xx) {
      for (int ch = 0; ch < C; ++ch) {
        double ss = 0.0;
        for (int y = 0; y < ysize; ++y) {
          ss += horiz[(static_cast<size_t>(ymin + y) * dst_w + xx) * C + ch] *
                k[y];
        }
        orow[xx * C + ch] = Clip8Double(ss);
      }
    }
  }
  return out;
}

}  // namespace litert::lm
