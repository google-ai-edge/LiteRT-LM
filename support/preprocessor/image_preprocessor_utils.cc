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

#include "support/preprocessor/image_preprocessor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/preprocessor/image_preprocessor.h"
#include "support/util/io_types.h"

namespace litert::support {

absl::StatusOr<std::pair<int, int>> GetAspectRatioPreservingSize(
    int width, int height,
    const ImagePreprocessParameter::PatchifyConfig& patchify_config) {
  if (patchify_config.patch_width != patchify_config.patch_height) {
    return absl::InvalidArgumentError(
        "Patch width must be equal to patch height.");
  }
  if (width == 0 || height == 0) {
    return absl::InvalidArgumentError("Input image has zero width or height.");
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

absl::StatusOr<InputImage> PatchifyImage(
    absl::Span<const float> image_data,
    const ImagePreprocessParameter& parameter) {
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

  const bool merge_patches = patchify_config->merge_patches &&
                             patchify_config->pooling_kernel_size > 1;
  const int effective_kernel_size =
      merge_patches ? patchify_config->pooling_kernel_size : 1;

  const int model_patch_height = patch_height * effective_kernel_size;
  const int model_patch_width = patch_width * effective_kernel_size;
  if (height % model_patch_height != 0 || width % model_patch_width != 0) {
    return absl::InvalidArgumentError(
        "Image dimensions must be divisible by patch size.");
  }

  const int num_model_patches_h = height / model_patch_height;
  const int num_model_patches_w = width / model_patch_width;
  const int num_model_patches = num_model_patches_h * num_model_patches_w;
  const int num_sub_patches = (height / patch_height) * (width / patch_width);

  if (patchify_config->max_num_patches > 0 &&
      num_sub_patches > patchify_config->max_num_patches) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of patches (", num_sub_patches, ") exceeds max_num_patches (",
        patchify_config->max_num_patches, ")."));
  }

  const int teacher_patch_dim = patch_width * patch_height * channels;
  const int model_patch_dim =
      teacher_patch_dim * effective_kernel_size * effective_kernel_size;

  // Transformer (ViT) encoders consume a per-patch "positions_xy" tensor, while
  // single input encoders (e.g. LFM2 VL) only consume the "images" tensor.
  const bool emit_positions = patchify_config->emit_positions;

  LITERT_ASSIGN_OR_RETURN(
      auto patches_buffer,
      ::litert::TensorBuffer::CreateManagedHostMemory(
          MakeRankedTensorType<float>(
              {batch_size, num_model_patches, model_patch_dim}),
          batch_size * num_model_patches * model_patch_dim * sizeof(float)));

  LITERT_ASSIGN_OR_RETURN(
      auto patches_lock,
      ::litert::TensorBufferScopedLock::Create(
          patches_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  float* patches_ptr = reinterpret_cast<float*>(patches_lock.second);

  std::optional<::litert::TensorBuffer> positions_buffer;
  std::optional<::litert::TensorBufferScopedLock> positions_lock;
  int32_t* positions_ptr = nullptr;
  if (emit_positions) {
    LITERT_ASSIGN_OR_RETURN(
        positions_buffer,
        ::litert::TensorBuffer::CreateManagedHostMemory(
            MakeRankedTensorType<int32_t>({batch_size, num_model_patches, 2}),
            batch_size * num_model_patches * 2 * sizeof(int32_t)));
    LITERT_ASSIGN_OR_RETURN(
        auto positions_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            *positions_buffer, ::litert::TensorBuffer::LockMode::kWrite));
    positions_lock = std::move(positions_lock_and_addr.first);
    positions_ptr = reinterpret_cast<int32_t*>(positions_lock_and_addr.second);
  }

  for (int b = 0; b < batch_size; ++b) {
    for (int mh = 0; mh < num_model_patches_h; ++mh) {
      for (int mw = 0; mw < num_model_patches_w; ++mw) {
        int model_patch_idx = mh * num_model_patches_w + mw;
        int global_patch_idx = b * num_model_patches + model_patch_idx;

        if (positions_ptr != nullptr) {
          positions_ptr[global_patch_idx * 2 + 0] = mw;
          positions_ptr[global_patch_idx * 2 + 1] = mh;
        }

        for (int ph = 0; ph < model_patch_height; ++ph) {
          for (int pw = 0; pw < model_patch_width; ++pw) {
            for (int c = 0; c < channels; ++c) {
              int src_h = mh * model_patch_height + ph;
              int src_w = mw * model_patch_width + pw;
              int src_idx =
                  ((b * height + src_h) * width + src_w) * channels + c;
              int dest_idx = global_patch_idx * model_patch_dim +
                             (ph * model_patch_width + pw) * channels + c;
              patches_ptr[dest_idx] = image_data[src_idx];
            }
          }
        }
      }
    }
  }

  absl::flat_hash_map<std::string, TensorBuffer> tensor_map;
  tensor_map["images"] = std::move(patches_buffer);
  if (emit_positions) {
    // Release the write lock before handing the buffer off.
    positions_lock.reset();
    tensor_map["positions_xy"] = std::move(*positions_buffer);
  }
  return InputImage(std::move(tensor_map));
}

}  // namespace litert::support
