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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_IMAGE_PREPROCESS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_IMAGE_PREPROCESS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {

// MiniCPM-V-4 geometry, fixed by the fused vision tflite in the bundle.
// Shared by the image preprocessor and the data processor so that the packing
// layouts match.

// ViT patch edge, in pixels.
inline constexpr int kMinicpmvPatchSize = 14;
// Fused vision signature length. The official slicing (scale = 448) yields at
// most ~1036 patches per slice; 1088 = 17 * 64, so ceil(L / shrink) with
// shrink = 17 is always exactly kMinicpmvTokensPerSlice soft tokens per slice.
inline constexpr int kMinicpmvMaxPatchLen = 1088;
// 980 / 14: the ViT absolute-position embedding grid side (bucketize target).
inline constexpr int kMinicpmvNumPatchesPerSide = 70;
// Soft tokens produced per slice by the fused resampler, which is also the
// number of placeholder tokens the LLM expects per slice.
inline constexpr int kMinicpmvTokensPerSlice = 64;

// Official MiniCPM-V multi-slice config: a thumbnail plus optional grid
// sub-images. Each slice is resized so its edges are divisible by patch_size,
// normalized, and packed as a reshape_by_patch strip for the navit ViT.
struct MinicpmvSliceConfig {
  int scale_resolution = 448;
  int patch_size = kMinicpmvPatchSize;
  int max_slice_nums = 9;
  int num_patches_per_side = kMinicpmvNumPatchesPerSide;
  float norm_mean[3] = {0.5f, 0.5f, 0.5f};
  float norm_std[3] = {0.5f, 0.5f, 0.5f};
};

// One preprocessed slice. Contains valid patches only; the data processor pads
// up to kMinicpmvMaxPatchLen.
struct MinicpmvSlice {
  // [3, patch_size, num_patches * patch_size], row-major (channel, py, col).
  std::vector<float> strip;
  // Patch rows (height / patch_size).
  int tgt_h = 0;
  // Patch columns (width / patch_size).
  int tgt_w = 0;
  // tgt_h * tgt_w, before padding.
  int num_patches = 0;
  // Bucketized linear ids on the 70x70 ViT position grid. After the data
  // processor pads the slice, -1 marks a pad entry.
  std::vector<int64_t> position_ids;
};

struct MinicpmvSliced {
  // [thumbnail, sub-images...] in row-major grid order.
  std::vector<MinicpmvSlice> slices;
  // Slice grid; grid_x is 0 when the image is small enough to need only the
  // thumbnail.
  int grid_x = 0;
  int grid_y = 0;
};

// Slices raw image bytes into a thumbnail plus sub-images. Bit-faithful to the
// Hugging Face MiniCPMVImageProcessor (PIL bicubic, integer grid math).
absl::StatusOr<MinicpmvSliced> PreprocessImageSliced(
    const std::string& image_bytes, const MinicpmvSliceConfig& config = {});

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_IMAGE_PREPROCESS_H_
