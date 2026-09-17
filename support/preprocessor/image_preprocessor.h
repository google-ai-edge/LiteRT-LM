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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_IMAGE_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_IMAGE_PREPROCESSOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "support/util/io_types.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep

namespace litert::support {

class ImagePreprocessParameter {
 public:
  // The config for patchify.
  struct PatchifyConfig {
    // The width of the patch.
    int patch_width;
    // The height of the patch.
    int patch_height;
    // The maximum number of patches.
    int max_num_patches;
    // The pooling kernel size.
    int pooling_kernel_size;
    // use position tensor.
    bool emit_positions = true;
    // Whether to perform 2-stage sub-patch merging in the image preprocessor.
    bool merge_patches = false;
  };

  // The config for normalization
  struct NormalizationConfig {
    std::vector<float> mean;
    std::vector<float> std;
    float rescale_factor = 1.0;
  };

  // Gets the target dimensions for preprocessing.
  const Dimensions& GetTargetDimensions() const { return dimensions_; }

  // Sets the target dimensions for preprocessing.
  void SetTargetDimensions(const Dimensions& dimensions) {
    dimensions_ = dimensions;
  }

  // Gets the patchify config for preprocessing.
  const std::optional<PatchifyConfig>& GetPatchifyConfig() const {
    return patchify_config_;
  }

  // Sets the patchify config for preprocessing.
  void SetPatchifyConfig(const PatchifyConfig& patchify_config) {
    patchify_config_ = patchify_config;
  }

  // Gets the Normalization config for preprocessing.
  const std::optional<NormalizationConfig>& GetNormalizationConfig() const {
    return normalization_config_;
  }

  // Sets the Normalization config for preprocessing
  void SetNormalizationConfig(const NormalizationConfig& normalization_config) {
    normalization_config_ = normalization_config;
  }

 private:
  Dimensions dimensions_;
  std::optional<PatchifyConfig> patchify_config_;
  std::optional<NormalizationConfig> normalization_config_;
};

// A decoded image in 8-bit RGB, HWC (row-major, 3 interleaved channels)
// layout. This is the lowest common denominator every image backend can
// produce, and is what model-specific preprocessors that implement their own
// resampling operate on.
struct DecodedImage {
  // `width` * `height` * 3 bytes.
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
};

// Preprocessor for image.
// Main purpose is to process raw image bytes into a resized image TensorBuffer.
class ImagePreprocessor {
 public:
  // Creates an ImagePreprocessor.
  //
  // If `enable_skia_image_preprocessor` is enabled, then an instance of
  // SkiaImagePreprocessor is returned. Otherwise, the StbImagePreprocessor is
  // returned.
  static std::unique_ptr<ImagePreprocessor> Create();

  virtual ~ImagePreprocessor() = default;

  // Preprocesses the raw image bytes into a resized image TensorBuffer.
  // Input is a string_view of the raw image bytes.
  // Output is a TensorBuffer of the resized RGB image with target dimensions.
  virtual absl::StatusOr<InputImage> Preprocess(
      const InputImage& input_image,
      const ImagePreprocessParameter& parameter) {
    if (input_image.IsTensorBuffer()) {
      ABSL_ASSIGN_OR_RETURN(auto processed_image_tensor,
                            input_image.GetPreprocessedImageTensor());
      LITERT_ASSIGN_OR_RETURN(auto processed_image_tensor_with_reference,
                              processed_image_tensor->Duplicate());
      InputImage processed_image(
          std::move(processed_image_tensor_with_reference));
      return processed_image;
    }
    return absl::UnimplementedError("Image preprocessor is not implemented.");
  };

  // Decodes encoded image bytes (PNG, JPEG, BMP, ...) into 8-bit RGB pixels,
  // without resizing or normalizing them. Images with an alpha channel have it
  // dropped.
  //
  // This exists for model-specific preprocessors that cannot use Preprocess()
  // because they must reproduce a reference Python implementation's resampling
  // bit-for-bit, or because they fan a single image out into several tensors.
  // Going through this method keeps them free of any direct image codec
  // dependency, so the codec stays swappable per platform.
  virtual absl::StatusOr<DecodedImage> Decode(
      absl::string_view image_bytes) const {
    return absl::UnimplementedError("Image decoding is not implemented.");
  }
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_IMAGE_PREPROCESSOR_H_
