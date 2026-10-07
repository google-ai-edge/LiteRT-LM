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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MATH_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MATH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::text2image {

// Computes the empirical FLUX.2-klein FlowMatch-Euler sigma schedule of length
// `steps + 1` (ending with `0.0f`) as a function of `steps` and `image_tokens`
// (`grid_h * grid_w`).
absl::StatusOr<std::vector<float>> ComputeFlowMatchSigmas(int steps,
                                                          int image_tokens);

// Builds 4D image token position IDs of shape `[grid_h * grid_w, 4]` where row
// `(h, w)` is `[0.0f, h, w, 0.0f]`.
std::vector<float> BuildImagePositionIds(int grid_h, int grid_w);

// Builds 4D text token position IDs of shape `[seq_len, 4]` where row `i` is
// `[0.0f, 0.0f, 0.0f, i]`.
std::vector<float> BuildTextPositionIds(int seq_len);

// Samples `count` standard normal N(0, 1) floats using `std::mt19937_64(seed)`.
std::vector<float> SampleGaussianLatents(size_t count, uint64_t seed);

// Applies per-packed-channel BatchNorm affine transform
// (`z = lat * bn_scale + bn_shift`) and 2x2 unpatchifies packed latents of
// shape `[1, grid_h * grid_w, 128]` into NCHW VAE latents of shape
// `[1, 32, 2 * grid_h, 2 * grid_w]`.
absl::StatusOr<std::vector<float>> UnpatchifyLatents(
    absl::Span<const float> packed_latents, int grid_h, int grid_w,
    absl::Span<const float> bn_scale, absl::Span<const float> bn_shift);

// Converts planar NCHW float image output of shape `[1, 3, height, width]` in
// `[-1, 1]` to interleaved RGB888 bytes of shape `[height, width, 3]` via
// `round(clamp(y * 0.5 + 0.5, 0.0, 1.0) * 255.0)`.
absl::StatusOr<std::vector<uint8_t>> ConvertNchwToRgb888(
    absl::Span<const float> nchw, int height, int width);

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MATH_H_
