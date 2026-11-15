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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_PIL_RESIZE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_PIL_RESIZE_H_

#include <cstdint>
#include <vector>

namespace litert::lm {

// A faithful, dependency-free (stdlib-only) port of PIL/Pillow's BICUBIC
// resize (Image.resize with resample=BICUBIC). Matches Pillow's separable
// two-pass resampling: a cubic convolution kernel with a = -0.5, with the
// filter support scaled by the down-scaling ratio (antialiasing), horizontal
// pass then vertical pass, with coefficients normalized per output pixel.
//
// Input:  src_rgb = HWC uint8 [src_h * src_w * 3].
// Output: HWC uint8 [dst_h * dst_w * 3], clamped/rounded to match Pillow.
//
// Why not reuse the stb-based resize in the image preprocessor: Pillow's
// BICUBIC and STBIR_FILTER_CATMULLROM are the same kernel family (a = -0.5),
// but stb resamples with an sRGB colorspace conversion, whereas Pillow uses
// its own int32 fixed-point pipeline over a support window scaled by the
// down-scaling ratio. Models whose reference implementation is the Hugging
// Face image processor (e.g. MiniCPM-V) are sensitive to that difference: it
// does not raise an error, it silently degrades output quality.
//
// TODO: if more models need Hugging Face parity, consider folding this into
// the general ImagePreprocessor behind a resample-filter option.
std::vector<uint8_t> PilResizeBicubicRgb(const uint8_t* src_rgb, int src_w,
                                         int src_h, int dst_w, int dst_h);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_PIL_RESIZE_H_
