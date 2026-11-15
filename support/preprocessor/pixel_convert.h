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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_PIXEL_CONVERT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_PIXEL_CONVERT_H_

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert::support {

// Converts interleaved (HWC) uint8 pixels to float with a per-channel affine
// transform, i.e. for every pixel `i` and channel `c`:
//
//   dst[i * channels + c] = src[i * channels + c] * scale[c] + offset[c]
//
// This folds the usual `rescale_factor` and `(x - mean) / std` normalization
// into one multiply-add per element. The 3-channel (RGB) and 1-channel paths
// are vectorized with Highway and dispatched at runtime to the best SIMD
// target available on the CPU (NEON / SVE on Arm, SSE4 / AVX2 / AVX-512 on
// x86, WASM SIMD, RVV, or a portable scalar fallback). Other channel counts
// use a scalar loop.
//
// args:
//   src: `num_pixels * channels` interleaved uint8 samples.
//   dst: destination for `num_pixels * channels` floats. Must not alias `src`.
//   num_pixels: number of pixels (not samples) to convert.
//   scale: per-channel multiplier; `scale.size()` must equal `channels`.
//   offset: per-channel additive term; `offset.size()` must equal `channels`.
//
// returns:
//   Nothing. Callers are responsible for validating the span sizes; the
//   function does not bounds check.
void ConvertUint8ToFloat(const uint8_t* src, float* dst, size_t num_pixels,
                         int channels, absl::Span<const float> scale,
                         absl::Span<const float> offset);

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_PIXEL_CONVERT_H_
