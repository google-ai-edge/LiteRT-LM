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

#include "runtime/components/preprocessor/minicpmv_image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "runtime/components/preprocessor/pil_resize.h"
#include "stb_image.h"  // from @stb

namespace litert::lm {
namespace {

int EnsureDivide(int length, int patch) {
  const int value =
      static_cast<int>(std::lrint(static_cast<double>(length) / patch)) * patch;
  return value > patch ? value : patch;
}

// Returns (best_w, best_h), both divisible by patch and clamped so that
// (best_w / patch) * (best_h / patch) <= kMiniCpmVMaxPatchLen.
std::pair<int, int> FindBestResize(int w, int h, int scale, int patch,
                                   bool allow_upscale) {
  if (static_cast<int64_t>(w) * h > static_cast<int64_t>(scale) * scale ||
      allow_upscale) {
    const double ratio = static_cast<double>(w) / h;
    h = static_cast<int>(scale / std::sqrt(ratio));
    w = static_cast<int>(h * ratio);
  }
  int best_w = EnsureDivide(w, patch);
  int best_h = EnsureDivide(h, patch);
  while ((best_w / patch) * (best_h / patch) > kMiniCpmVMaxPatchLen) {
    if (best_w >= best_h && best_w > patch) {
      const int max_w_patches = kMiniCpmVMaxPatchLen / (best_h / patch);
      best_w = std::max(patch, max_w_patches * patch);
    } else if (best_h > patch) {
      const int max_h_patches = kMiniCpmVMaxPatchLen / (best_w / patch);
      best_h = std::max(patch, max_h_patches * patch);
    } else {
      break;
    }
  }
  return {best_w, best_h};
}

std::pair<int, int> GetRefineSize(int w, int h, int grid_x, int grid_y,
                                  int scale, int patch) {
  const int refine_w = EnsureDivide(w, grid_x);
  const int refine_h = EnsureDivide(h, grid_y);
  const auto best_grid = FindBestResize(refine_w / grid_x, refine_h / grid_y,
                                        scale, patch, /*allow_upscale=*/true);
  return {best_grid.first * grid_x, best_grid.second * grid_y};
}

// Returns {grid_x, grid_y}, or {0, 0} for "no slicing".
std::pair<int, int> GetSlicedGrid(int w, int h, int scale, int max_slice) {
  const double log_ratio = std::log(static_cast<double>(w) / h);
  const double ratio =
      static_cast<double>(w) * h / (static_cast<double>(scale) * scale);
  const int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice);
  if (multiple <= 1) return {0, 0};

  std::vector<int> candidate_nums;
  for (const int candidate : {multiple - 1, multiple, multiple + 1}) {
    if (candidate == 1 || candidate > max_slice) continue;
    candidate_nums.push_back(candidate);
  }
  std::vector<std::pair<int, int>> grids;
  for (const int n : candidate_nums) {
    for (int m = 1; m <= n; ++m) {
      if (n % m == 0) grids.push_back({m, n / m});
    }
  }

  std::pair<int, int> best = {1, 1};
  double min_error = std::numeric_limits<double>::max();
  for (const auto& grid : grids) {
    const double error = std::abs(
        log_ratio - std::log(static_cast<double>(grid.first) / grid.second));
    if (error < min_error) {
      best = grid;
      min_error = error;
    }
  }
  return best;
}

// Normalizes an RGB HWC uint8 image to CHW float, then applies
// reshape_by_patch to produce [3, patch, num_patches * patch]. Mirrors
// torch.nn.functional.unfold + reshape/permute in
// image_processing_minicpmv.reshape_by_patch.
std::vector<float> NormalizeAndReshapeByPatch(const std::vector<uint8_t>& rgb,
                                              int width, int height, int patch,
                                              const float mean[3],
                                              const float std_dev[3]) {
  const int tgt_w = width / patch;
  const int tgt_h = height / patch;
  const int num_patches = tgt_h * tgt_w;
  // Output strip [3, patch, num_patches * patch].
  std::vector<float> strip(static_cast<size_t>(3) * patch * num_patches *
                           patch);
  // reshape_by_patch layout: for channel c and row py in [0, patch), the strip
  // column index is patch_index * patch + px, where patch_index iterates over
  // patches in row-major order (patch_row * tgt_w + patch_col), and the value
  // is the normalized pixel at
  // (patch_row * patch + py, patch_col * patch + px).
  const int strip_w = num_patches * patch;
  for (int patch_row = 0; patch_row < tgt_h; ++patch_row) {
    for (int patch_col = 0; patch_col < tgt_w; ++patch_col) {
      const int patch_index = patch_row * tgt_w + patch_col;
      for (int py = 0; py < patch; ++py) {
        for (int px = 0; px < patch; ++px) {
          const int img_y = patch_row * patch + py;
          const int img_x = patch_col * patch + px;
          const uint8_t* pixel =
              &rgb[(static_cast<size_t>(img_y) * width + img_x) * 3];
          const int col = patch_index * patch + px;
          for (int ch = 0; ch < 3; ++ch) {
            const float value =
                (static_cast<float>(pixel[ch]) / 255.0f - mean[ch]) /
                std_dev[ch];
            // strip[ch][py][col]
            strip[(static_cast<size_t>(ch) * patch + py) * strip_w + col] =
                value;
          }
        }
      }
    }
  }
  return strip;
}

// torch.bucketize(x, boundaries, right=True): the number of boundaries <= x.
int BucketizeRight(const std::vector<float>& boundaries, float x) {
  int lo = 0;
  int hi = static_cast<int>(boundaries.size());
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (boundaries[mid] <= x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// torch.arange(0, 1 - 1e-6, 1/n) in float32: value[i] = i * (1/n), computed by
// multiplication rather than accumulation (accumulation drifts and diverges at
// bucket boundaries).
std::vector<float> FractionalCoords(int n) {
  if (n <= 0) {
    return {};
  }
  std::vector<float> values;
  values.reserve(n);
  const float step = 1.0f / n;
  for (int i = 0; i < n; ++i) {
    values.push_back(static_cast<float>(i) * step);
  }
  return values;
}

// position_ids for a tgt_h x tgt_w slice, bucketized onto the
// patches_per_side x patches_per_side grid.
//
// This MUST use float32 arithmetic to match torch.arange(dtype=float32);
// computing in double diverges at bucket boundaries.
std::vector<int64_t> ComputePositionIds(int tgt_h, int tgt_w,
                                        int patches_per_side) {
  if (tgt_h <= 0 || tgt_w <= 0 || patches_per_side <= 0) {
    return {};
  }
  // boundaries = arange(1/pps, 1.0, 1/pps), which is pps - 1 values.
  std::vector<float> boundaries;
  boundaries.reserve(patches_per_side - 1);
  for (int i = 1; i < patches_per_side; ++i) {
    boundaries.push_back(static_cast<float>(i) / patches_per_side);
  }

  const std::vector<float> frac_h = FractionalCoords(tgt_h);
  const std::vector<float> frac_w = FractionalCoords(tgt_w);
  std::vector<int> bucket_h(frac_h.size());
  std::vector<int> bucket_w(frac_w.size());
  for (size_t i = 0; i < frac_h.size(); ++i) {
    bucket_h[i] = BucketizeRight(boundaries, frac_h[i]);
  }
  for (size_t i = 0; i < frac_w.size(); ++i) {
    bucket_w[i] = BucketizeRight(boundaries, frac_w[i]);
  }

  std::vector<int64_t> positions;
  positions.reserve(static_cast<size_t>(tgt_h) * tgt_w);
  for (int i = 0; i < tgt_h; ++i) {
    for (int j = 0; j < tgt_w; ++j) {
      positions.push_back(static_cast<int64_t>(bucket_h[i]) * patches_per_side +
                          bucket_w[j]);
    }
  }
  return positions;
}

}  // namespace

absl::StatusOr<MiniCpmVSliced> PreprocessImageSliced(
    const std::string& image_bytes, const MiniCpmVSliceConfig& config) {
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* pixels = stbi_load_from_memory(
      reinterpret_cast<const unsigned char*>(image_bytes.data()),
      static_cast<int>(image_bytes.size()), &width, &height, &channels,
      /*desired_channels=*/3);
  if (pixels == nullptr) {
    const char* reason = stbi_failure_reason();
    return absl::InvalidArgumentError(absl::StrFormat(
        "stb_image failed to decode image: %s", reason ? reason : "unknown"));
  }
  const std::vector<uint8_t> src(
      pixels, pixels + static_cast<size_t>(width) * height * 3);
  stbi_image_free(pixels);

  const int scale = config.scale_resolution;
  const int patch = config.patch_size;
  const std::pair<int, int> grid =
      GetSlicedGrid(width, height, scale, config.max_slice_nums);

  MiniCpmVSliced result;
  result.grid_x = grid.first;
  result.grid_y = grid.second;

  std::vector<std::vector<uint8_t>> slice_rgb;  // Each entry is HWC uint8.
  std::vector<std::pair<int, int>> slice_wh;  // Each entry is (width, height).

  if (grid.first == 0) {
    // No slicing: a single upscaled thumbnail.
    const auto best_size = FindBestResize(width, height, scale, patch,
                                          /*allow_upscale=*/true);
    slice_rgb.push_back(PilResizeBicubicRgb(src.data(), width, height,
                                            best_size.first, best_size.second));
    slice_wh.push_back(best_size);
  } else {
    // Thumbnail: the source image, down-sampled, divisible by patch.
    const auto thumbnail_size = FindBestResize(width, height, scale, patch,
                                               /*allow_upscale=*/false);
    slice_rgb.push_back(PilResizeBicubicRgb(src.data(), width, height,
                                            thumbnail_size.first,
                                            thumbnail_size.second));
    slice_wh.push_back(thumbnail_size);

    // Refine image, then split into grid_x * grid_y sub-images (row-major).
    const auto refine_size =
        GetRefineSize(width, height, grid.first, grid.second, scale, patch);
    const std::vector<uint8_t> refine = PilResizeBicubicRgb(
        src.data(), width, height, refine_size.first, refine_size.second);
    const int refine_w = refine_size.first;
    const int refine_h = refine_size.second;
    const int cell_w = refine_w / grid.first;
    const int cell_h = refine_h / grid.second;
    for (int top = 0; top < refine_h; top += cell_h) {
      for (int left = 0; left < refine_w; left += cell_w) {
        // Crop [left, top, left + cell_w, top + cell_h) from the refine image.
        // No resize is needed: it is already sized.
        std::vector<uint8_t> crop(static_cast<size_t>(cell_w) * cell_h * 3);
        for (int y = 0; y < cell_h; ++y) {
          for (int x = 0; x < cell_w; ++x) {
            for (int ch = 0; ch < 3; ++ch) {
              crop[(static_cast<size_t>(y) * cell_w + x) * 3 + ch] = refine
                  [(static_cast<size_t>(top + y) * refine_w + (left + x)) * 3 +
                   ch];
            }
          }
        }
        slice_rgb.push_back(std::move(crop));
        slice_wh.push_back({cell_w, cell_h});
      }
    }
  }

  // The fused vision graph bakes in the resampler's 2D sin-cos position table,
  // so at runtime we only need the per-patch coordinates (built in the data
  // processor) and the bucketized ViT ids.
  const int patches_per_side = config.num_patches_per_side;
  for (size_t s = 0; s < slice_rgb.size(); ++s) {
    const int slice_w = slice_wh[s].first;
    const int slice_h = slice_wh[s].second;
    MiniCpmVSlice slice;
    slice.tgt_w = slice_w / patch;
    slice.tgt_h = slice_h / patch;
    slice.num_patches = slice.tgt_h * slice.tgt_w;
    slice.strip =
        NormalizeAndReshapeByPatch(slice_rgb[s], slice_w, slice_h, patch,
                                   config.norm_mean, config.norm_std);
    slice.position_ids =
        ComputePositionIds(slice.tgt_h, slice.tgt_w, patches_per_side);
    result.slices.push_back(std::move(slice));
  }
  return result;
}

}  // namespace litert::lm
