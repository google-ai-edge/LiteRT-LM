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

#include "support/preprocessor/stb_resize.h"

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

// Compile a private copy of stb_image_resize2: STB_IMAGE_RESIZE_STATIC gives
// every stbir_* function internal linkage, so this translation unit neither
// exports nor clashes with the stbir_* symbols of other stb_image_resize (v1
// or v2) implementations linked into the same binary.
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb_image_resize2.h"  // from @stb  // NOLINT
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace litert::support {

absl::Status ResizeSrgbUint8(const uint8_t* input, int input_width,
                             int input_height, uint8_t* output,
                             int output_width, int output_height,
                             int channels) {
  if (channels < 1 || channels > 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported channel count for resizing: ", channels));
  }
  if (stbir_resize(
          input, input_width, input_height, /*input_stride_in_bytes=*/0, output,
          output_width, output_height,
          /*output_stride_in_bytes=*/0,
          static_cast<stbir_pixel_layout>(channels), STBIR_TYPE_UINT8_SRGB,
          STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM) == nullptr) {
    return absl::InternalError("Failed to resize image.");
  }
  return absl::OkStatus();
}

}  // namespace litert::support
