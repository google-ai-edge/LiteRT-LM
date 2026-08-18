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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_PIL_RESIZE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_PIL_RESIZE_H_

#include <cstdint>
#include <vector>

namespace litert::lm {

// A faithful, dependency-free (stdlib-only) port of PIL/Pillow's BICUBIC
// resize (Image.resize with resample=BICUBIC). Matches Pillow's separable
// two-pass resampling: a cubic convolution kernel with a = -0.5, with the
// filter support scaled by the down-scaling ratio (antialiasing), horizontal
// pass then vertical pass, in float, coefficients normalized per output pixel.
//
// Input:  src_rgb = HWC uint8 [src_h * src_w * 3].
// Output: HWC uint8 [dst_h * dst_w * 3], clamped/rounded to match Pillow.
//
// This replaces the earlier stb_image_resize2-backed "PillowResize" which
// incorrectly mapped BICUBIC to a B-spline filter and resampled in uint8.
std::vector<uint8_t> PilResizeBicubicRgb(const uint8_t* src_rgb, int src_w,
                                         int src_h, int dst_w, int dst_h);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_PIL_RESIZE_H_
