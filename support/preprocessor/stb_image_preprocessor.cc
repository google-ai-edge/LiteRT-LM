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

#include "support/preprocessor/stb_image_preprocessor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/preprocessor/image_preprocessor.h"
#include "support/preprocessor/image_preprocessor_utils.h"
#include "support/util/io_types.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"  // from @stb
#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif
#include "stb_image_resize2.h"  // from @stb

namespace litert::support {

std::unique_ptr<ImagePreprocessor> ImagePreprocessor::Create() {
  return std::make_unique<StbImagePreprocessor>();
}

namespace {

// The desired number of channels for the image encoder. If the input image has
// a different number of channels, it will be converted to this number of
// channels.
constexpr int kDesiredChannels = 3;

absl::Status MaybeResizeImageWithSameAspectRatio(
    std::vector<unsigned char>& image_data,
    std::vector<unsigned char>& resized_image_data,
    ImagePreprocessParameter& parameter) {
  const Dimensions& target_dimensions = parameter.GetTargetDimensions();
  const int height = target_dimensions[1];
  const int width = target_dimensions[2];
  const int patch_width = parameter.GetPatchifyConfig()->patch_width;
  const int patch_height = parameter.GetPatchifyConfig()->patch_height;
  const int max_num_patches = parameter.GetPatchifyConfig()->max_num_patches;

  ASSIGN_OR_RETURN(auto size,
                   GetAspectRatioPreservingSize(
                       width, height, parameter.GetPatchifyConfig().value()));
  int new_height = size.first;
  int new_width = size.second;

  if (new_height == height && new_width == width) {
    resized_image_data = std::move(image_data);
    return absl::OkStatus();
  }
  ABSL_LOG(INFO) << "Resize image from " << width << "x" << height << " to "
                 << new_width << "x" << new_height << " which will result in "
                 << static_cast<int>(new_width / patch_width) *
                        (new_height / patch_height)
                 << " patches to fit the max_num_patches: " << max_num_patches
                 << " limit.";
  parameter.SetTargetDimensions(
      {target_dimensions[0], new_height, new_width, target_dimensions[3]});

  resized_image_data.resize(static_cast<size_t>(target_dimensions[0]) *
                            new_height * new_width * target_dimensions[3]);

  int alpha_channel = -1;
  if (target_dimensions[3] == 4) {
    alpha_channel = 3;
  } else if (target_dimensions[3] == 2) {
    alpha_channel = 1;
  }

  const int batch_size = target_dimensions[0];
  const int channels = target_dimensions[3];

  for (int i = 0; i < batch_size; ++i) {
    unsigned char* input_data =
        image_data.data() + i * height * width * channels;
    unsigned char* output_data =
        resized_image_data.data() + i * new_height * new_width * channels;

    if (stbir_resize(input_data, width, height, 0, output_data, new_width,
                     new_height, 0,
                     static_cast<stbir_pixel_layout>(channels),
                     STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_CATMULLROM) == 0) {
      return absl::InternalError("Failed to resize image.");
    }
  }

  return absl::OkStatus();
}


}  // namespace



absl::StatusOr<InputImage> StbImagePreprocessor::Preprocess(
    const InputImage& input_image, const ImagePreprocessParameter& parameter) {
  if (input_image.IsTensorBuffer()) {
    ASSIGN_OR_RETURN(auto processed_image_tensor,
                     input_image.GetPreprocessedImageTensor());
    LITERT_ASSIGN_OR_RETURN(auto processed_image_tensor_with_reference,
                            processed_image_tensor->Duplicate());
    InputImage processed_image(
        std::move(processed_image_tensor_with_reference));
    return processed_image;
  }

  ASSIGN_OR_RETURN(absl::string_view input_image_bytes,
                   input_image.GetRawImageBytes());

  const Dimensions& target_dimensions = parameter.GetTargetDimensions();

  int original_width, original_height, original_channels;
  if (!parameter.GetPatchifyConfig().has_value() &&
      target_dimensions.size() != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Target dimensions must be (batch, height, width, "
                     "channels). Got dimensions size: ",
                     target_dimensions.size()));
  }
  const int desired_channels = parameter.GetPatchifyConfig().has_value()
                                   ? kDesiredChannels
                                   : target_dimensions.at(3);
  unsigned char* decoded_image = stbi_load_from_memory(
      reinterpret_cast<const stbi_uc*>(input_image_bytes.data()),
      input_image_bytes.size(), &original_width, &original_height,
      &original_channels, desired_channels);
  if (decoded_image == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to decode image. Reason: ", stbi_failure_reason()));
  }
  // Use a unique_ptr to ensure the decoded image is freed.
  std::unique_ptr<unsigned char[], void (*)(void*)> decoded_image_ptr(
      decoded_image, stbi_image_free);

  std::vector<unsigned char> resized_image;
  ImagePreprocessParameter updated_parameter = parameter;
  if (parameter.GetTargetDimensions().empty()) {
    // No fixed target dimensions: resize while preserving the aspect ratio so
    // the image fits the patchify constraints (used by patchifying encoders).
    const size_t num_elements = static_cast<size_t>(original_width) *
                                original_height * kDesiredChannels;
    updated_parameter.SetTargetDimensions(
        {1, original_height, original_width, kDesiredChannels});
    std::vector<unsigned char> image_data(decoded_image,
                                          decoded_image + num_elements);
    RETURN_IF_ERROR(MaybeResizeImageWithSameAspectRatio(
        image_data, resized_image, updated_parameter));
  } else {
    // Fixed target dimensions: resize directly to the requested size.
    const int target_height = target_dimensions.at(1);
    const int target_width = target_dimensions.at(2);
    const int target_channels = target_dimensions.at(3);

    resized_image.resize(static_cast<size_t>(target_width) * target_height *
                         target_channels);

    int alpha_channel = -1;
    if (target_channels == 4) {
      alpha_channel = 3;
    } else if (target_channels == 2) {
      alpha_channel = 1;
    }

    if (stbir_resize(decoded_image, original_width, original_height, 0,
                     resized_image.data(), target_width, target_height, 0,
                     static_cast<stbir_pixel_layout>(target_channels),
                     STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_CATMULLROM) == 0) {
      return absl::InternalError("Failed to resize image.");
    }
  }

  const Dimensions& final_dimensions = updated_parameter.GetTargetDimensions();
  const int batch_size = final_dimensions.at(0);
  const int target_height = final_dimensions.at(1);
  const int target_width = final_dimensions.at(2);
  const int target_channels = final_dimensions.at(3);

  // Rescale the pixel values into floats. Defaults to the [0, 255] -> [0, 1]
  // mapping unless an explicit rescale factor is provided.
  const float rescale_factor =
      updated_parameter.GetNormalizationConfig().has_value()
          ? updated_parameter.GetNormalizationConfig()->rescale_factor
          : (1.0f / 255.0f);
  std::vector<float> float_image(resized_image.size());
  for (size_t i = 0; i < resized_image.size(); ++i) {
    float_image[i] = static_cast<float>(resized_image[i]) * rescale_factor;
  }

  // Optionally apply per-channel mean/std normalization.
  if (updated_parameter.GetNormalizationConfig().has_value()) {
    const auto& normalization_config =
        *updated_parameter.GetNormalizationConfig();
    if (normalization_config.mean.size() !=
            static_cast<size_t>(desired_channels) ||
        normalization_config.std.size() !=
            static_cast<size_t>(desired_channels)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Normalization mean/std size does not match the number of channels (",
          desired_channels, ")."));
    }
    for (size_t index = 0; index < float_image.size(); ++index) {
      const int ch = index % desired_channels;
      float_image[index] =
          (float_image[index] - normalization_config.mean[ch]) /
          normalization_config.std[ch];
    }
  }

  if (updated_parameter.GetPatchifyConfig().has_value()) {
    return PatchifyImage(std::move(float_image), updated_parameter);
  }

  const size_t num_elements = static_cast<size_t>(batch_size) * target_height *
                              target_width * target_channels;
  const size_t buffer_size = num_elements * sizeof(float);
  LITERT_ASSIGN_OR_RETURN(
      auto processed_tensor_buffer,
      ::litert::TensorBuffer::CreateManagedHostMemory(
          MakeRankedTensorType<float>(
              {batch_size, target_height, target_width, target_channels}),
          buffer_size));
  LITERT_ASSIGN_OR_RETURN(
      auto processed_tensor_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          processed_tensor_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  float* float_buffer_ptr =
      reinterpret_cast<float*>(processed_tensor_lock_and_addr.second);
  for (size_t i = 0; i < float_image.size(); ++i) {
    float_buffer_ptr[i] = float_image[i];
  }
  InputImage processed_image(std::move(processed_tensor_buffer));

  return processed_image;
}

}  // namespace litert::support
