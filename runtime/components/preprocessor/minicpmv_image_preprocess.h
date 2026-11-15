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
#include "runtime/components/preprocessor/image_preprocessor.h"

namespace litert::lm {

// MiniCPM-V-4 geometry, fixed by the fused vision tflite in the bundle.
// Shared by the image preprocessor and the data processor so that the packing
// layouts match.

// ViT patch edge, in pixels.
inline constexpr int kMiniCpmVPatchSize = 14;
// Fused vision signature length. FindBestResize clamps slice dimensions so
// that each slice yields at most kMiniCpmVMaxPatchLen patches (1088 = 17 * 64,
// so ceil(L / shrink) with shrink = 17 is always exactly
// kMiniCpmVTokensPerSlice soft tokens per slice).
inline constexpr int kMiniCpmVMaxPatchLen = 1088;
// 980 / 14: the ViT absolute-position embedding grid side (bucketize target).
inline constexpr int kMiniCpmVNumPatchesPerSide = 70;
// Soft tokens produced per slice by the fused resampler, which is also the
// number of placeholder tokens the LLM expects per slice.
inline constexpr int kMiniCpmVTokensPerSlice = 64;

// Official MiniCPM-V multi-slice config: a thumbnail plus optional grid
// sub-images. Each slice is resized so its edges are divisible by patch_size,
// normalized, and packed as a reshape_by_patch strip for the navit ViT.
struct MiniCpmVSliceConfig {
  int scale_resolution = 448;
  int patch_size = kMiniCpmVPatchSize;
  int max_slice_nums = 9;
  int num_patches_per_side = kMiniCpmVNumPatchesPerSide;
  float norm_mean[3] = {0.5f, 0.5f, 0.5f};
  float norm_std[3] = {0.5f, 0.5f, 0.5f};
};

// One preprocessed slice. Contains valid patches only; the data processor pads
// up to kMiniCpmVMaxPatchLen.
struct MiniCpmVSlice {
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

struct MiniCpmVSliced {
  // [thumbnail, sub-images...] in row-major grid order.
  std::vector<MiniCpmVSlice> slices;
  // Slice grid; grid_x is 0 when the image is small enough to need only the
  // thumbnail.
  int grid_x = 0;
  int grid_y = 0;
};

// Slices raw image bytes into a thumbnail plus sub-images. Bit-faithful to the
// Hugging Face MiniCPMVImageProcessor (PIL bicubic, integer grid math).
//
// `image_decoder` only decodes `image_bytes` into raw RGB pixels; it keeps this
// function free of any image codec dependency so the codec stays swappable per
// platform. Resampling is deliberately not delegated to it: it has to
// reproduce PIL exactly, which no general purpose backend does. Creating an
// ImagePreprocessor is not free, so callers should hold one and pass it in on
// every call rather than creating one per image.
absl::StatusOr<MiniCpmVSliced> PreprocessImageSliced(
    const std::string& image_bytes, const ImagePreprocessor& image_decoder,
    const MiniCpmVSliceConfig& config = {});

// Widens a slice strip from its valid width to `padded_num_patches` patches of
// edge `patch_size`, padding with zeros, and pads `slice.position_ids` with
// -1. The strip is [3, patch_size, num_patches * patch_size], so widening
// re-packs each row rather than flat-resizing.
void PadSliceToSignatureLength(MiniCpmVSlice& slice, int padded_num_patches,
                               int patch_size = kMiniCpmVPatchSize);

// One slice packed into the 3-input fused vision encoder layout:
//   `images`:        [num_patches, 3 * patch_size * patch_size] float
//   `positions_xy`:  [num_patches, 2] int32 (contiguous (w, h); -1 pad)
//   `vit_positions`: [num_patches, 2] int32 (bucketized ViT (w, h); -1 pad)
struct MiniCpmVPackedSlice {
  std::vector<float> images;
  std::vector<int32_t> positions_xy;
  std::vector<int32_t> vit_positions;
};

// Rearranges `slice.strip` from [3, patch_size, num_patches * patch_size] into
// patch-flattened `[num_patches, 3 * patch_size * patch_size]` and unpacks
// `slice.position_ids` into 2D resampler (`positions_xy`) and ViT
// (`vit_positions`) coordinates (with `(-1, -1)` for padded entries).
MiniCpmVPackedSlice PackSlicePatches(
    const MiniCpmVSlice& slice, int patch_size = kMiniCpmVPatchSize,
    int patches_per_side = kMiniCpmVNumPatchesPerSide);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_IMAGE_PREPROCESS_H_
