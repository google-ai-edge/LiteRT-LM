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

#include "runtime/components/preprocessor/image_preprocessor_utils.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/preprocessor/image_preprocessor.h"
namespace litert::lm {

absl::StatusOr<std::pair<int, int>> GetAspectRatioPreservingSize(
    int width, int height,
    const ImagePreprocessParameter::PatchifyConfig& patchify_config) {
  if (patchify_config.patch_width != patchify_config.patch_height) {
    return absl::InvalidArgumentError(
        "Patch width must be equal to patch height.");
  }
  float total_px = width * height;
  float target_px =
      patchify_config.max_num_patches *
      (patchify_config.patch_width * patchify_config.patch_height);
  float factor = std::sqrt(target_px / total_px);
  float ideal_height = factor * height;
  float ideal_width = factor * width;
  int side_mult =
      patchify_config.pooling_kernel_size * patchify_config.patch_width;

  int target_height =
      static_cast<int>(std::floor(ideal_height / side_mult)) * side_mult;
  int target_width =
      static_cast<int>(std::floor(ideal_width / side_mult)) * side_mult;

  if (target_height == 0 && target_width == 0) {
    return absl::InvalidArgumentError("Attempting to resize to a 0 x 0 image.");
  }

  int max_side_length = (patchify_config.max_num_patches /
                         (patchify_config.pooling_kernel_size *
                          patchify_config.pooling_kernel_size)) *
                        side_mult;
  if (target_height == 0) {
    target_height = side_mult;
    target_width = std::min(
        static_cast<int>(std::floor(static_cast<float>(width) / height)) *
            side_mult,
        max_side_length);
  } else if (target_width == 0) {
    target_width = side_mult;
    target_height = std::min(
        static_cast<int>(std::floor(static_cast<float>(height) / width)) *
            side_mult,
        max_side_length);
  }

  if (target_height * target_width > target_px) {
    return absl::InvalidArgumentError("Resizing exceeds max patches.");
  }

  return std::make_pair(target_height, target_width);
}

}  // namespace litert::lm
