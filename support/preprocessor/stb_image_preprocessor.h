// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_STB_IMAGE_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_STB_IMAGE_PREPROCESSOR_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/preprocessor/image_preprocessor.h"
#include "support/util/io_types.h"

namespace litert::support {

// Preprocessor for image using stb image library.
// Main purpose is to process raw image bytes into a resized image TensorBuffer.
class StbImagePreprocessor : public ImagePreprocessor {
 public:
  // Preprocesses the raw image bytes into a resized image TensorBuffer.
  //
  // If PatchifyConfig is set, the image will be patchified, with the image
  // resized to the largest possible size that fits the max_num_patches limit
  // while maintaining the original aspect ratio.
  absl::StatusOr<InputImage> Preprocess(
      const InputImage& input_image,
      const ImagePreprocessParameter& parameter) override;
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_STB_IMAGE_PREPROCESSOR_H_
