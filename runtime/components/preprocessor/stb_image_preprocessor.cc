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

#include "runtime/components/preprocessor/stb_image_preprocessor.h"

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
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/image_preprocessor_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"  // from @stb
#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif
#include "stb_image_resize2.h"  // from @stb

namespace litert::lm {

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

absl::StatusOr<InputImage> StbImagePreprocessor::PatchifyImage(
    std::vector<float> image_data, const ImagePreprocessParameter& parameter) {
  const auto& patchify_config = parameter.GetPatchifyConfig();
  if (!patchify_config.has_value()) {
    return absl::InternalError("Patchify config is not set.");
  }
  const int patch_width = patchify_config->patch_width;
  if (patch_width <= 0) {
    return absl::InvalidArgumentError("Patch width must be positive.");
  }

  const int patch_height = patchify_config->patch_height;
  if (patch_height <= 0) {
    return absl::InvalidArgumentError("Patch height must be positive.");
  }

  const Dimensions& target_dimensions = parameter.GetTargetDimensions();
  if (target_dimensions.size() != 4) {
    return absl::InvalidArgumentError("Target dimensions must be 4.");
  }
  const int batch_size = target_dimensions[0];
  const int height = target_dimensions[1];
  const int width = target_dimensions[2];
  const int channels = target_dimensions[3];

  if (image_data.size() != batch_size * height * width * channels) {
    return absl::InvalidArgumentError(
        "Image data size does not match target dimensions.");
  }

  if (height % patch_height != 0 || width % patch_width != 0) {
    return absl::InvalidArgumentError(
        "Image dimensions must be divisible by patch size.");
  }

  const int num_patches_h = height / patch_height;
  const int num_patches_w = width / patch_width;
  const int num_patches = num_patches_h * num_patches_w;

  if (patchify_config->max_num_patches > 0 &&
      num_patches > patchify_config->max_num_patches) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of patches (", num_patches, ") exceeds max_num_patches (",
        patchify_config->max_num_patches, ")."));
  }

  const int patch_dim = patch_width * patch_height * channels;

  LITERT_ASSIGN_OR_RETURN(
      auto patches_buffer,
      ::litert::TensorBuffer::CreateManagedHostMemory(
          MakeRankedTensorType<float>({batch_size, num_patches, patch_dim}),
          batch_size * num_patches * patch_dim * sizeof(float)));

  LITERT_ASSIGN_OR_RETURN(
      auto positions_buffer,
      ::litert::TensorBuffer::CreateManagedHostMemory(
          MakeRankedTensorType<int32_t>({batch_size, num_patches, 2}),
          batch_size * num_patches * 2 * sizeof(int32_t)));

  LITERT_ASSIGN_OR_RETURN(
      auto patches_lock,
      ::litert::TensorBufferScopedLock::Create(
          patches_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  float* patches_ptr = reinterpret_cast<float*>(patches_lock.second);

  LITERT_ASSIGN_OR_RETURN(
      auto positions_lock,
      ::litert::TensorBufferScopedLock::Create(
          positions_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  int32_t* positions_ptr = reinterpret_cast<int32_t*>(positions_lock.second);

  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < num_patches_h; ++h) {
      for (int w = 0; w < num_patches_w; ++w) {
        int patch_idx = h * num_patches_w + w;
        int global_patch_idx = b * num_patches + patch_idx;

        positions_ptr[global_patch_idx * 2] = w;
        positions_ptr[global_patch_idx * 2 + 1] = h;

        for (int ph = 0; ph < patch_height; ++ph) {
          for (int pw = 0; pw < patch_width; ++pw) {
            for (int c = 0; c < channels; ++c) {
              int src_h = h * patch_height + ph;
              int src_w = w * patch_width + pw;
              int src_idx =
                  ((b * height + src_h) * width + src_w) * channels + c;
              int dest_idx = global_patch_idx * patch_dim +
                             ((ph * patch_width + pw) * channels + c);
              patches_ptr[dest_idx] = static_cast<float>(image_data[src_idx]);
            }
          }
        }
      }
    }
  }

  absl::flat_hash_map<std::string, TensorBuffer> tensor_map;
  tensor_map["images"] = std::move(patches_buffer);
  tensor_map["positions_xy"] = std::move(positions_buffer);

  return InputImage(std::move(tensor_map));
}

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

  if (parameter.GetPatchifyConfig().has_value()) {
    // Patchify the image if patchify config is set.
    const size_t num_elements = static_cast<size_t>(original_width) *
                                original_height * kDesiredChannels;
    // Resize the image if needed.
    ImagePreprocessParameter updated_parameter = parameter;
    updated_parameter.SetTargetDimensions(
        {1, original_height, original_width, kDesiredChannels});
    std::vector<unsigned char> image_data(decoded_image,
                                          decoded_image + num_elements);
    std::vector<unsigned char> resized_image_data;
    RETURN_IF_ERROR(MaybeResizeImageWithSameAspectRatio(
        image_data, resized_image_data, updated_parameter));
    // Convert the image to float.
    std::vector<float> float_image(resized_image_data.size());
    for (size_t i = 0; i < resized_image_data.size(); ++i) {
      float_image[i] = static_cast<float>(resized_image_data[i]) / 255.0f;
    }
    return PatchifyImage(std::move(float_image), updated_parameter);
  } else {
    const int batch_size = target_dimensions.at(0);
    const int target_height = target_dimensions.at(1);
    const int target_width = target_dimensions.at(2);
    const int target_channels = target_dimensions.at(3);
    std::vector<uint8_t> resized_image(static_cast<size_t>(target_width) *
                                       target_height * target_channels);

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
    const int num_elements =
        batch_size * target_height * target_width * target_channels;
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
    // Normalize pixel values from [0, 255] to [0.0f, 1.0f].
    for (size_t i = 0; i < resized_image.size(); ++i) {
      float_buffer_ptr[i] = static_cast<float>(resized_image[i]) / 255.0f;
    }

    InputImage processed_image(std::move(processed_tensor_buffer));

    return processed_image;
  }
}

}  // namespace litert::lm
