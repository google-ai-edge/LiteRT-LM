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

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/preprocessor/image_preprocessor.h"
#include "support/util/test_utils.h"  // NOLINT

namespace litert::support {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::status::StatusIs;

TEST(GetAspectRatioPreservingSizeTest, SquareImageNoResize) {
  ImagePreprocessParameter::PatchifyConfig config;
  config.patch_width = 10;
  config.patch_height = 10;
  config.max_num_patches = 10000;
  config.pooling_kernel_size = 1;

  absl::StatusOr<std::pair<int, int>> result =
      GetAspectRatioPreservingSize(1000, 1000, config);
  ASSERT_TRUE(result.ok());
  EXPECT_THAT(result.value(), Pair(1000, 1000));
}

TEST(GetAspectRatioPreservingSizeTest, RectangularImageDownscale) {
  ImagePreprocessParameter::PatchifyConfig config;
  config.patch_width = 14;
  config.patch_height = 14;
  config.max_num_patches = 256;
  config.pooling_kernel_size = 2;

  absl::StatusOr<std::pair<int, int>> result =
      GetAspectRatioPreservingSize(1600, 900, config);
  ASSERT_TRUE(result.ok());
  // target_height, target_width
  EXPECT_THAT(result.value(), Pair(168, 280));
}

TEST(GetAspectRatioPreservingSizeTest, InvalidPatchSize) {
  ImagePreprocessParameter::PatchifyConfig config;
  config.patch_width = 14;
  config.patch_height = 16;
  config.max_num_patches = 256;
  config.pooling_kernel_size = 2;

  absl::StatusOr<std::pair<int, int>> result =
      GetAspectRatioPreservingSize(1600, 900, config);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              HasSubstr("Patch width must be equal to patch height"));
}

TEST(GetAspectRatioPreservingSizeTest, ZeroTargetHeightAdjusted) {
  ImagePreprocessParameter::PatchifyConfig config;
  config.patch_width = 14;
  config.patch_height = 14;
  config.max_num_patches = 256;
  config.pooling_kernel_size = 2;

  // width = 1000, height = 10 -> ideal width and height will result in height <
  // 28
  absl::StatusOr<std::pair<int, int>> result =
      GetAspectRatioPreservingSize(1000, 10, config);
  ASSERT_TRUE(result.ok());
  EXPECT_THAT(result.value(), Pair(28, 1792));
}

TEST(GetAspectRatioPreservingSizeTest, ZeroTargetWidthAdjusted) {
  ImagePreprocessParameter::PatchifyConfig config;
  config.patch_width = 14;
  config.patch_height = 14;
  config.max_num_patches = 256;
  config.pooling_kernel_size = 2;

  // width = 10, height = 1000 -> ideal width and height will result in width <
  // 28
  absl::StatusOr<std::pair<int, int>> result =
      GetAspectRatioPreservingSize(10, 1000, config);
  ASSERT_TRUE(result.ok());
  EXPECT_THAT(result.value(), Pair(1792, 28));
}

TEST(GetAspectRatioPreservingSizeTest, ZeroByZeroImageError) {
  ImagePreprocessParameter::PatchifyConfig config;
  config.patch_width = 14;
  config.patch_height = 14;
  config.max_num_patches = 1;
  config.pooling_kernel_size = 2;

  // With max_num_patches=1, factor=14. ideal width/height = 14.
  // floor(14 / 28) = 0.
  absl::StatusOr<std::pair<int, int>> result =
      GetAspectRatioPreservingSize(1, 1, config);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              HasSubstr("Attempting to resize to a 0 x 0 image."));
}

TEST(GetAspectRatioPreservingSizeTest, ZeroHeightOrWidthError) {
  ImagePreprocessParameter::PatchifyConfig config;
  config.patch_width = 16;
  config.patch_height = 16;
  config.max_num_patches = 1;
  config.pooling_kernel_size = 1;

  absl::StatusOr<std::pair<int, int>> result =
      GetAspectRatioPreservingSize(1, 0, config);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              HasSubstr("Input image has zero width or height."));

  result = GetAspectRatioPreservingSize(0, 1, config);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              HasSubstr("Input image has zero width or height."));
}

TEST(ImagePreprocessorUtilsTest, PatchifyImageSuccess) {
  // Input image: 2x2 patches of 2x2 pixels, 3 channels (RGB)
  // Total height = 4, width = 4, channels = 3.
  // Total size = 4 * 4 * 3 = 48 elements.
  std::vector<float> image_data(48);
  for (int i = 0; i < 48; ++i) {
    image_data[i] = static_cast<float>(i);
  }

  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 4, 4, 3});
  parameter.SetPatchifyConfig({.patch_width = 2,
                               .patch_height = 2,
                               .max_num_patches = 4,
                               .pooling_kernel_size = 1});

  ASSERT_OK_AND_ASSIGN(auto preprocessed_image,
                       PatchifyImage(image_data, parameter));

  ASSERT_TRUE(preprocessed_image.IsTensorBufferMap());
  ASSERT_OK_AND_ASSIGN(auto tensor_map,
                       preprocessed_image.GetPreprocessedImageTensorMap());
  ASSERT_NE(tensor_map, nullptr);
  EXPECT_TRUE(tensor_map->contains("images"));
  EXPECT_TRUE(tensor_map->contains("positions_xy"));

  const auto& images_tensor = tensor_map->at("images");
  auto images_tensor_type = images_tensor.TensorType();
  ASSERT_TRUE(images_tensor_type.HasValue());
  EXPECT_THAT(images_tensor_type.Value().Layout().Dimensions(),
              ElementsAre(1, 4, 12));

  const auto& positions_tensor = tensor_map->at("positions_xy");
  auto positions_tensor_type = positions_tensor.TensorType();
  ASSERT_TRUE(positions_tensor_type.HasValue());
  EXPECT_THAT(positions_tensor_type.Value().Layout().Dimensions(),
              ElementsAre(1, 4, 2));

  // Verify positions.
  auto positions_lock = ::litert::TensorBufferScopedLock::Create(
      positions_tensor, ::litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(positions_lock.HasValue());
  const int32_t* positions_ptr =
      reinterpret_cast<const int32_t*>(positions_lock->second);
  EXPECT_EQ(positions_ptr[0], 0);
  EXPECT_EQ(positions_ptr[1], 0);
  EXPECT_EQ(positions_ptr[2], 1);
  EXPECT_EQ(positions_ptr[3], 0);
  EXPECT_EQ(positions_ptr[4], 0);
  EXPECT_EQ(positions_ptr[5], 1);
  EXPECT_EQ(positions_ptr[6], 1);
  EXPECT_EQ(positions_ptr[7], 1);

  // Verify image values.
  auto images_lock = ::litert::TensorBufferScopedLock::Create(
      images_tensor, ::litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(images_lock.HasValue());
  const float* data = reinterpret_cast<const float*>(images_lock->second);

  // Patch 0 should contain: 0,1,2, 3,4,5, 12,13,14, 15,16,17.
  EXPECT_EQ(data[0], 0.0f);
  EXPECT_EQ(data[1], 1.0f);
  EXPECT_EQ(data[2], 2.0f);
  EXPECT_EQ(data[3], 3.0f);
  EXPECT_EQ(data[4], 4.0f);
  EXPECT_EQ(data[5], 5.0f);
  EXPECT_EQ(data[6], 12.0f);
  EXPECT_EQ(data[7], 13.0f);
  EXPECT_EQ(data[8], 14.0f);
  EXPECT_EQ(data[9], 15.0f);
  EXPECT_EQ(data[10], 16.0f);
  EXPECT_EQ(data[11], 17.0f);

  // Patch 1:
  // P02 (3 elements) -> data[12, 13, 14] -> 6, 7, 8
  // P03 (3 elements) -> data[15, 16, 17] -> 9, 10, 11
  // P12 (3 elements) -> data[18, 19, 20] -> 18, 19, 20
  // P13 (3 elements) -> data[21, 22, 23] -> 21, 22, 23
  EXPECT_EQ(data[12], 6.0f);
  EXPECT_EQ(data[13], 7.0f);
  EXPECT_EQ(data[14], 8.0f);
  EXPECT_EQ(data[15], 9.0f);
  EXPECT_EQ(data[16], 10.0f);
  EXPECT_EQ(data[17], 11.0f);
  EXPECT_EQ(data[18], 18.0f);
  EXPECT_EQ(data[19], 19.0f);
  EXPECT_EQ(data[20], 20.0f);
  EXPECT_EQ(data[21], 21.0f);
  EXPECT_EQ(data[22], 22.0f);
  EXPECT_EQ(data[23], 23.0f);
}

TEST(ImagePreprocessorUtilsTest, PatchifyImageMissingConfig) {
  std::vector<float> image_data(48);
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 4, 4, 3});
  // No patchify config set.

  EXPECT_THAT(
      PatchifyImage(image_data, parameter),
      StatusIs(absl::StatusCode::kInternal, "Patchify config is not set."));
}

TEST(ImagePreprocessorUtilsTest, PatchifyImageInvalidPatchSize) {
  std::vector<float> image_data(48);
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 4, 4, 3});
  parameter.SetPatchifyConfig({.patch_width = 0,  // Invalid
                               .patch_height = 2,
                               .max_num_patches = 4,
                               .pooling_kernel_size = 1});

  EXPECT_THAT(PatchifyImage(image_data, parameter),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Patch width must be positive."));

  parameter.SetPatchifyConfig({.patch_width = 2,
                               .patch_height = -1,  // Invalid
                               .max_num_patches = 4,
                               .pooling_kernel_size = 1});

  EXPECT_THAT(PatchifyImage(image_data, parameter),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Patch height must be positive."));
}

TEST(ImagePreprocessorUtilsTest, PatchifyImageMismatchedDataSize) {
  std::vector<float> image_data(47);  // Mismatched, expected 48
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 4, 4, 3});
  parameter.SetPatchifyConfig({.patch_width = 2,
                               .patch_height = 2,
                               .max_num_patches = 4,
                               .pooling_kernel_size = 1});

  EXPECT_THAT(PatchifyImage(image_data, parameter),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Image data size does not match target dimensions."));
}

TEST(ImagePreprocessorUtilsTest, PatchifyImageNotDivisible) {
  std::vector<float> image_data(48);
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 4, 4, 3});
  parameter.SetPatchifyConfig({.patch_width = 3,  // 4 is not divisible by 3
                               .patch_height = 2,
                               .max_num_patches = 4,
                               .pooling_kernel_size = 1});

  EXPECT_THAT(PatchifyImage(image_data, parameter),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Image dimensions must be divisible by patch size."));
}

TEST(ImagePreprocessorUtilsTest, PatchifyImageExceedsMaxPatches) {
  std::vector<float> image_data(48);
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 4, 4, 3});
  parameter.SetPatchifyConfig(
      {.patch_width = 2,
       .patch_height = 2,
       .max_num_patches = 3,  // Max is 3, but we have 4 patches
       .pooling_kernel_size = 1});

  EXPECT_THAT(PatchifyImage(image_data, parameter),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds max_num_patches")));
}

TEST(ImagePreprocessorUtilsTest, PatchifyImageZeroWidthOrHeight) {
  std::vector<float> image_data(48);
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 0, 4, 3});
  parameter.SetPatchifyConfig({.patch_width = 1,
                               .patch_height = 1,
                               .max_num_patches = 1,
                               .pooling_kernel_size = 1});

  EXPECT_THAT(PatchifyImage(image_data, parameter),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Image data size does not match target dimensions."));

  parameter.SetTargetDimensions({1, 4, 0, 3});
  EXPECT_THAT(PatchifyImage(image_data, parameter),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Image data size does not match target dimensions."));
}

}  // namespace
}  // namespace litert::support
