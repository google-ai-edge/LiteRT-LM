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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_IMAGE_PREPROCESSOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_IMAGE_PREPROCESSOR_UTILS_H_

#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/preprocessor/image_preprocessor.h"

namespace litert::lm {

// Returns the dimensions of the image after resizing it to preserve the
// aspect ratio.
//
// The resizing is done such that the number of patches is less than or equal
// to the maximum number of patches specified in the patchify config, and the
// resulted width and height are integer multiples of the (pooling kernel size x
// patch size). The patch width must be equal to the patch height.
//
// args:
//   width: The original width of the image.
//   height: The original height of the image.
//   patchify_config: The patchify config to use for resizing.
//
// returns:
//   The dimensions of the image after resizing it to preserve the aspect ratio.
absl::StatusOr<std::pair<int, int>> GetAspectRatioPreservingSize(
    int width, int height,
    const ImagePreprocessParameter::PatchifyConfig& patchify_config);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_IMAGE_PREPROCESSOR_UTILS_H_
