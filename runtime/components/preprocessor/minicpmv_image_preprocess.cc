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

#include "runtime/components/preprocessor/minicpmv_image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"       // from @com_google_absl
#include "absl/status/statusor.h"     // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "runtime/components/preprocessor/pil_resize.h"

// stb_image is a single-header library whose implementation
// (STB_IMAGE_IMPLEMENTATION) must be compiled in exactly one translation unit
// per binary. That TU is the stock support/preprocessor/stb_image_preprocessor,
// which is always linked into litert_lm_main (other data processors depend on
// it). We therefore include only the declarations here and take an explicit
// build dep on :stb_image_preprocessor so the stbi_* symbols are guaranteed to
// be present, rather than relying on an implicit link-time coincidence.
#include "stb_image.h"  // from @stb

namespace litert::lm {

// ============================ Multi-slice ============================
namespace {

int EnsureDivide(int length, int patch) {
  int v = static_cast<int>(std::lround(static_cast<double>(length) / patch)) * patch;
  return v > patch ? v : patch;
}

// Returns (best_w, best_h) both divisible by patch.
std::pair<int, int> FindBestResize(int w, int h, int scale, int patch,
                                   bool allow_upscale) {
  if (static_cast<long long>(w) * h > static_cast<long long>(scale) * scale ||
      allow_upscale) {
    double r = static_cast<double>(w) / h;
    h = static_cast<int>(scale / std::sqrt(r));
    w = static_cast<int>(h * r);
  }
  return {EnsureDivide(w, patch), EnsureDivide(h, patch)};
}

std::pair<int, int> GetRefineSize(int w, int h, int gx, int gy, int scale,
                                  int patch) {
  int rw = EnsureDivide(w, gx);
  int rh = EnsureDivide(h, gy);
  auto bg = FindBestResize(rw / gx, rh / gy, scale, patch, /*allow_upscale=*/true);
  return {bg.first * gx, bg.second * gy};
}

// Returns {gx, gy} or {0,0} for "no slicing".
std::pair<int, int> GetSlicedGrid(int w, int h, int scale, int max_slice) {
  double log_ratio = std::log(static_cast<double>(w) / h);
  double ratio = static_cast<double>(w) * h / (static_cast<double>(scale) * scale);
  int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice);
  if (multiple <= 1) return {0, 0};
  std::vector<int> candidate_nums;
  for (int i : {multiple - 1, multiple, multiple + 1}) {
    if (i == 1 || i > max_slice) continue;
    candidate_nums.push_back(i);
  }
  std::vector<std::pair<int, int>> grids;
  for (int n : candidate_nums) {
    for (int m = 1; m <= n; ++m) {
      if (n % m == 0) grids.push_back({m, n / m});
    }
  }
  std::pair<int, int> best = {1, 1};
  double min_err = 1e18;
  for (auto& g : grids) {
    double e = std::abs(log_ratio - std::log(static_cast<double>(g.first) / g.second));
    if (e < min_err) { best = g; min_err = e; }
  }
  return best;
}

// Normalize an RGB HWC uint8 image to CHW float, then reshape_by_patch to
// [3, patch, num_patches*patch]. Mirrors torch.nn.functional.unfold +
// reshape/permute in image_processing_minicpmv.reshape_by_patch.
std::vector<float> NormalizeAndReshapeByPatch(const std::vector<uint8_t>& rgb,
                                              int W, int H, int patch,
                                              const float mean[3],
                                              const float std_[3]) {
  const int tgt_w = W / patch, tgt_h = H / patch;
  const int num_patches = tgt_h * tgt_w;
  // Output strip [3, patch, num_patches*patch].
  std::vector<float> strip(static_cast<size_t>(3) * patch * num_patches * patch);
  // reshape_by_patch layout: for channel c, row py in [0,patch), the strip
  // column index = patch_index * patch + px, where patch_index iterates over
  // patches in row-major (patch_row * tgt_w + patch_col), and value = normalized
  // pixel at (patch_row*patch+py, patch_col*patch+px).
  const int strip_w = num_patches * patch;
  for (int pr = 0; pr < tgt_h; ++pr) {
    for (int pc = 0; pc < tgt_w; ++pc) {
      const int patch_index = pr * tgt_w + pc;
      for (int py = 0; py < patch; ++py) {
        for (int px = 0; px < patch; ++px) {
          const int img_y = pr * patch + py;
          const int img_x = pc * patch + px;
          const uint8_t* pxl = &rgb[(static_cast<size_t>(img_y) * W + img_x) * 3];
          const int col = patch_index * patch + px;
          for (int ch = 0; ch < 3; ++ch) {
            const float v =
                (static_cast<float>(pxl[ch]) / 255.0f - mean[ch]) / std_[ch];
            // strip[ch][py][col]
            strip[(static_cast<size_t>(ch) * patch + py) * strip_w + col] = v;
          }
        }
      }
    }
  }
  return strip;
}

// position_ids for a tgt_h x tgt_w slice, bucketized to the pps x pps grid.
// MUST use float32 arithmetic (matches torch.arange float32); double diverges
// at bucket boundaries.
std::vector<int64_t> ComputePositionIds(int tgt_h, int tgt_w, int pps) {
  // boundaries = arange(1/pps, 1.0, 1/pps)  (pps-1 values)
  std::vector<float> boundaries;
  for (int i = 1; i < pps; ++i) boundaries.push_back(static_cast<float>(i) / pps);
  auto bucketize_right = [&](float x) {
    // torch.bucketize(x, boundaries, right=True): number of boundaries <= x.
    int lo = 0, hi = static_cast<int>(boundaries.size());
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (boundaries[mid] <= x) lo = mid + 1; else hi = mid;
    }
    return lo;
  };
  // fractional coords: arange(0, 1-1e-6, 1/n) in float32.
  auto frac = [](int n) {
    // torch.arange(0, 1-1e-6, 1/n) in float32: value[i] = i * (1/n), not
    // accumulated (accumulation drifts and diverges at bucket boundaries).
    std::vector<float> v;
    const float step = 1.0f / n;
    for (int i = 0;; ++i) {
      float x = static_cast<float>(i) * step;
      if (x >= 1.0f - 1e-6f) break;
      v.push_back(x);
    }
    return v;
  };
  std::vector<float> fh = frac(tgt_h), fw = frac(tgt_w);
  std::vector<int> bh(fh.size()), bw(fw.size());
  for (size_t i = 0; i < fh.size(); ++i) bh[i] = bucketize_right(fh[i]);
  for (size_t i = 0; i < fw.size(); ++i) bw[i] = bucketize_right(fw[i]);
  std::vector<int64_t> pos;
  pos.reserve(static_cast<size_t>(tgt_h) * tgt_w);
  for (int i = 0; i < tgt_h; ++i)
    for (int j = 0; j < tgt_w; ++j)
      pos.push_back(static_cast<int64_t>(bh[i]) * pps + bw[j]);
  return pos;
}

}  // namespace

absl::StatusOr<MinicpmvSliced> PreprocessImageSliced(
    const std::string& image_bytes, const MinicpmvSliceConfig& cfg) {
  int W = 0, H = 0, C = 0;
  unsigned char* pixels = stbi_load_from_memory(
      reinterpret_cast<const unsigned char*>(image_bytes.data()),
      static_cast<int>(image_bytes.size()), &W, &H, &C, 3);
  if (pixels == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("stb_image failed to decode image: %s",
                        stbi_failure_reason() ? stbi_failure_reason() : "?"));
  }
  std::vector<uint8_t> src(pixels, pixels + static_cast<size_t>(W) * H * 3);
  stbi_image_free(pixels);

  const int scale = cfg.scale_resolution, patch = cfg.patch_size;
  auto grid = GetSlicedGrid(W, H, scale, cfg.max_slice_nums);

  MinicpmvSliced result;
  result.grid_x = grid.first;
  result.grid_y = grid.second;

  std::vector<std::vector<uint8_t>> slice_rgb;   // each HWC uint8
  std::vector<std::pair<int, int>> slice_wh;     // (W,H)

  if (grid.first == 0) {
    // No slicing: single upscaled thumbnail.
    auto bs = FindBestResize(W, H, scale, patch, /*allow_upscale=*/true);
    slice_rgb.push_back(PilResizeBicubicRgb(src.data(), W, H, bs.first, bs.second));
    slice_wh.push_back(bs);
  } else {
    // Thumbnail (source image, down-sampled, divisible by patch).
    auto br = FindBestResize(W, H, scale, patch, /*allow_upscale=*/false);
    slice_rgb.push_back(PilResizeBicubicRgb(src.data(), W, H, br.first, br.second));
    slice_wh.push_back(br);
    // Refine image, then split into grid_x*grid_y sub-images (row-major).
    auto rf = GetRefineSize(W, H, grid.first, grid.second, scale, patch);
    std::vector<uint8_t> refine =
        PilResizeBicubicRgb(src.data(), W, H, rf.first, rf.second);
    const int RW = rf.first, RH = rf.second;
    const int gw = RW / grid.first, gh = RH / grid.second;
    for (int i = 0; i < RH; i += gh) {
      for (int j = 0; j < RW; j += gw) {
        // crop [j, i, j+gw, i+gh) from refine (no resize; already sized).
        std::vector<uint8_t> crop(static_cast<size_t>(gw) * gh * 3);
        for (int y = 0; y < gh; ++y)
          for (int x = 0; x < gw; ++x)
            for (int c = 0; c < 3; ++c)
              crop[(static_cast<size_t>(y) * gw + x) * 3 + c] =
                  refine[((static_cast<size_t>(i + y) * RW + (j + x)) * 3) + c];
        slice_rgb.push_back(std::move(crop));
        slice_wh.push_back({gw, gh});
      }
    }
  }

  // Fused vision bakes the resampler 2D sin-cos pos table; runtime only needs
  // per-patch coords (built in the data processor) and ViT bucketized ids.
  const int pps = cfg.num_patches_per_side;
  for (size_t s = 0; s < slice_rgb.size(); ++s) {
    const int sw = slice_wh[s].first, sh = slice_wh[s].second;
    MinicpmvSlice sl;
    sl.tgt_w = sw / patch;
    sl.tgt_h = sh / patch;
    sl.num_patches = sl.tgt_h * sl.tgt_w;
    sl.strip = NormalizeAndReshapeByPatch(slice_rgb[s], sw, sh, patch,
                                          cfg.norm_mean, cfg.norm_std);
    sl.position_ids = ComputePositionIds(sl.tgt_h, sl.tgt_w, pps);
    result.slices.push_back(std::move(sl));
  }
  return result;
}

}  // namespace litert::lm
