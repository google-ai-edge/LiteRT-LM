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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_STB_RESIZE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_STB_RESIZE_H_

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl

namespace litert::support {

// Resizes interleaved 8-bit sRGB pixels with stb_image_resize2, using a
// Catmull-Rom filter and clamped edges.
//
// stb_image_resize2 is compiled into this library with internal linkage, so it
// cannot clash with other copies of stb_image_resize (v1 or v2) linked into
// the same binary.
//
// args:
//   input: `input_width * input_height * channels` bytes, tightly packed HWC.
//   input_width: Width of the input image in pixels.
//   input_height: Height of the input image in pixels.
//   output: Destination for `output_width * output_height * channels` bytes.
//   output_width: Width of the output image in pixels.
//   output_height: Height of the output image in pixels.
//   channels: Number of interleaved channels, 1 to 4.
//
// returns:
//   OkStatus on success, InvalidArgumentError for an unsupported channel
//   count, or InternalError if stb_image_resize2 fails.
absl::Status ResizeSrgbUint8(const uint8_t* input, int input_width,
                             int input_height, uint8_t* output,
                             int output_width, int output_height, int channels);

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_STB_RESIZE_H_
