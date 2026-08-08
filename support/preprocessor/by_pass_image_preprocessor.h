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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_BY_PASS_IMAGE_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_BY_PASS_IMAGE_PREPROCESSOR_H_

#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "support/preprocessor/image_preprocessor.h"
#include "support/util/io_types.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep

namespace litert::support {

// Image preprocessor implementation that bypasses the preprocessing and returns
// the input image directly if it's already a TensorBuffer.
class ByPassImagePreprocessor : public ImagePreprocessor {
 public:
  absl::StatusOr<InputImage> Preprocess(
      const InputImage& input_image,
      const ImagePreprocessParameter& parameter) override {
    if (input_image.IsTensorBuffer()) {
      ASSIGN_OR_RETURN(auto processed_image_tensor,
                       input_image.GetPreprocessedImageTensor());
      LITERT_ASSIGN_OR_RETURN(auto processed_image_tensor_with_reference,
                              processed_image_tensor->Duplicate());
      InputImage processed_image(
          std::move(processed_image_tensor_with_reference));
      return processed_image;
    }
    return absl::InvalidArgumentError("Input image is not preprocessed.");
  };
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_BY_PASS_IMAGE_PREPROCESSOR_H_
