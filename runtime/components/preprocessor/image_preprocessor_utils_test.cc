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

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/preprocessor/image_preprocessor.h"

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::Pair;

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

}  // namespace
}  // namespace litert::lm
